#include "ultima/kernels/dequant_q5_0.hpp"
#include "ultima/kernels/matvec.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace ultima::kernels {

namespace {

float fp16_to_float(std::uint16_t h) noexcept {
    const std::uint32_t sign = (h & 0x8000u) << 16;
    std::uint32_t exp  = (h & 0x7C00u) >> 10;
    std::uint32_t mant = (h & 0x03FFu);
    std::uint32_t out_bits;
    if (exp == 0) {
        if (mant == 0) {
            out_bits = sign;
        } else {
            unsigned shift = 0;
            while ((mant & 0x0400u) == 0) { mant <<= 1; ++shift; }
            mant &= 0x03FFu;
            exp   = 127u - 14u - shift;
            out_bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        out_bits = sign | 0x7F800000u | (mant << 13);
    } else {
        exp = exp - 15u + 127u;
        out_bits = sign | (exp << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out_bits, sizeof(f));
    return f;
}

inline float read_fp16_le(const std::uint8_t* p) noexcept {
    const std::uint16_t h = static_cast<std::uint16_t>(p[0])
                          | (static_cast<std::uint16_t>(p[1]) << 8);
    return fp16_to_float(h);
}

inline float hsum256(__m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

// Expand qs[16] into 32 nibbles: lane i in [0,16) is the low nibble of qs[i],
// lane i in [16,32) is the high nibble of qs[i-16]. That is exactly the
// element order dequant_q5_0_block produces.
inline __m256i nibbles_from_qs(const std::uint8_t* qs) noexcept {
    const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
    const __m256i both = _mm256_insertf128_si256(
        _mm256_castsi128_si256(packed), _mm_srli_epi16(packed, 4), 1);
    return _mm256_and_si256(_mm256_set1_epi8(0x0F), both);
}

// Expand the 32-bit qh plane into a byte mask: lane i is 0xFF when bit i of
// qh is set, 0x00 otherwise.
inline __m256i mask_from_qh(std::uint32_t qh) noexcept {
    const __m256i shuf = _mm256_set_epi64x(
        0x0303030303030303, 0x0202020202020202,
        0x0101010101010101, 0x0000000000000000);
    __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(static_cast<int>(qh)), shuf);
    // Per lane, OR in every bit except the one being tested, then a full-ones
    // compare isolates that bit.
    const __m256i bit_mask = _mm256_set1_epi64x(
        static_cast<long long>(0x7FBFDFEFF7FBFDFEULL));
    bytes = _mm256_or_si256(bytes, bit_mask);
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi8(static_cast<char>(0xFF)));
}

} // namespace

void matvec_q5_0_f32_scalar(const std::uint8_t* w, const float* x, float* y,
                            std::size_t M, std::size_t K) noexcept {
    if (K == 0 || (K % 32u) != 0) {
        for (std::size_t m = 0; m < M; ++m) y[m] = 0.0f;
        return;
    }
    const std::size_t blocks_per_row = K / 32u;
    const std::size_t bytes_per_row  = blocks_per_row * 22u;

    float scratch[32];
    for (std::size_t m = 0; m < M; ++m) {
        const std::uint8_t* row = w + m * bytes_per_row;
        float acc = 0.0f;
        for (std::size_t b = 0; b < blocks_per_row; ++b) {
            dequant_q5_0_block(row + b * 22u, scratch);
            const float* xslice = x + b * 32u;
            for (std::size_t i = 0; i < 32; ++i) acc += scratch[i] * xslice[i];
        }
        y[m] = acc;
    }
}

// Fused unpack: build the signed int8 quants directly, with no f32 scratch.
//
// The offset-binary "-16" is folded into the bit pattern rather than
// subtracted: where the fifth bit is clear, OR in 0xF0 so the byte read as
// signed int8 is (nibble - 16); where it is set, the nibble alone already
// equals (16 + nibble) - 16.
void matvec_q5_0_f32_avx2(const std::uint8_t* w, const float* x, float* y,
                          std::size_t M, std::size_t K) noexcept {
    if (K == 0 || (K % 32u) != 0) {
        for (std::size_t m = 0; m < M; ++m) y[m] = 0.0f;
        return;
    }
    const std::size_t blocks_per_row = K / 32u;
    const std::size_t bytes_per_row  = blocks_per_row * 22u;
    const __m256i hi_fill = _mm256_set1_epi8(static_cast<char>(0xF0));

    for (std::size_t m = 0; m < M; ++m) {
        const std::uint8_t* row = w + m * bytes_per_row;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(),
               a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();

        for (std::size_t b = 0; b < blocks_per_row; ++b) {
            const std::uint8_t* blk = row + b * 22u;
            const float d = read_fp16_le(blk);
            std::uint32_t qh;
            std::memcpy(&qh, blk + 2, sizeof(qh));

            __m256i q = nibbles_from_qs(blk + 6);
            const __m256i set5 = mask_from_qh(qh);
            q = _mm256_or_si256(q, _mm256_andnot_si256(set5, hi_fill));

            const __m128i lo = _mm256_castsi256_si128(q);
            const __m128i hi = _mm256_extracti128_si256(q, 1);
            const __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo));
            const __m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)));
            const __m256 f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi));
            const __m256 f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)));

            const __m256 vd = _mm256_set1_ps(d);
            const float* xs = x + b * 32u;
            a0 = _mm256_fmadd_ps(_mm256_mul_ps(f0, vd), _mm256_loadu_ps(xs +  0), a0);
            a1 = _mm256_fmadd_ps(_mm256_mul_ps(f1, vd), _mm256_loadu_ps(xs +  8), a1);
            a2 = _mm256_fmadd_ps(_mm256_mul_ps(f2, vd), _mm256_loadu_ps(xs + 16), a2);
            a3 = _mm256_fmadd_ps(_mm256_mul_ps(f3, vd), _mm256_loadu_ps(xs + 24), a3);
        }
        y[m] = hsum256(_mm256_add_ps(_mm256_add_ps(a0, a1),
                                     _mm256_add_ps(a2, a3)));
    }
}

void matvec_q5_0_f32(const std::uint8_t* w, const float* x, float* y,
                     std::size_t M, std::size_t K) noexcept {
    matvec_q5_0_f32_avx2(w, x, y, M, K);
}

} // namespace ultima::kernels
