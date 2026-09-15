#include "ultima/kernels/dequant_q4k.hpp"
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

inline void unpack_scale_min_6bit(const std::uint8_t* sm, unsigned s,
                                  unsigned& out_scale, unsigned& out_min) noexcept {
    if (s < 4) {
        out_scale = sm[s]     & 0x3Fu;
        out_min   = sm[s + 4] & 0x3Fu;
    } else {
        const unsigned idx = s - 4u;
        out_scale = (sm[8u + idx] & 0x0Fu) | ((sm[    idx] >> 6) << 4);
        out_min   = (sm[8u + idx] >>   4)  | ((sm[4u + idx] >> 6) << 4);
    }
}

inline float hsum256(__m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

// Dot 32 unsigned nibbles (one __m256i, values 0..15) against 32 floats,
// folding the sub-block dequant scale/min into each value:
//   acc_g += (nib * dvec - mvec) * x[g*8 .. g*8+8]   for g in 0..3
inline void dot32_nibbles(__m256i nib, const float* x,
                          __m256 dvec, __m256 mvec,
                          __m256& a0, __m256& a1, __m256& a2, __m256& a3) noexcept {
    const __m128i b0 = _mm256_castsi256_si128(nib);        // bytes  0..15
    const __m128i b1 = _mm256_extracti128_si256(nib, 1);   // bytes 16..31
    const __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b0));
    const __m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(b0, 8)));
    const __m256 f2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b1));
    const __m256 f3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(b1, 8)));
    a0 = _mm256_fmadd_ps(_mm256_fmsub_ps(f0, dvec, mvec), _mm256_loadu_ps(x +  0), a0);
    a1 = _mm256_fmadd_ps(_mm256_fmsub_ps(f1, dvec, mvec), _mm256_loadu_ps(x +  8), a1);
    a2 = _mm256_fmadd_ps(_mm256_fmsub_ps(f2, dvec, mvec), _mm256_loadu_ps(x + 16), a2);
    a3 = _mm256_fmadd_ps(_mm256_fmsub_ps(f3, dvec, mvec), _mm256_loadu_ps(x + 24), a3);
}

} // namespace

// ----- Scalar oracle --------------------------------------------------------
void matvec_q4k_f32_scalar(const std::uint8_t* w, const float* x, float* y,
                           std::size_t M, std::size_t K) noexcept {
    if (K == 0 || (K % 256u) != 0) {
        for (std::size_t m = 0; m < M; ++m) y[m] = 0.0f;
        return;
    }
    const std::size_t blocks_per_row = K / 256u;
    const std::size_t bytes_per_row  = blocks_per_row * 144u;

    float scratch[256];
    for (std::size_t m = 0; m < M; ++m) {
        const std::uint8_t* row = w + m * bytes_per_row;
        float acc = 0.0f;
        for (std::size_t b = 0; b < blocks_per_row; ++b) {
            dequant_q4k_block(row + b * 144u, scratch);
            const float* xslice = x + b * 256u;
            for (std::size_t i = 0; i < 256; ++i) acc += scratch[i] * xslice[i];
        }
        y[m] = acc;
    }
}

// ----- AVX2 + FMA -----------------------------------------------------------
// Per sub-block pair (64 values sharing 32 qs bytes): one SIMD load, mask for
// the low/high nibbles, then pure SIMD convert + FMA. Scale/min are folded
// per value (v = nib*d - m) so the whole row accumulates in vector registers
// with a single horizontal sum at the end.
void matvec_q4k_f32_avx2(const std::uint8_t* w, const float* x, float* y,
                         std::size_t M, std::size_t K) noexcept {
    if (K == 0 || (K % 256u) != 0) {
        for (std::size_t m = 0; m < M; ++m) y[m] = 0.0f;
        return;
    }
    const std::size_t blocks_per_row = K / 256u;
    const std::size_t bytes_per_row  = blocks_per_row * 144u;

    const __m256i mask0f = _mm256_set1_epi8(0x0F);

    for (std::size_t m = 0; m < M; ++m) {
        const std::uint8_t* row = w + m * bytes_per_row;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(),
               a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps(),
               a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps(),
               a6 = _mm256_setzero_ps(), a7 = _mm256_setzero_ps();

        for (std::size_t b = 0; b < blocks_per_row; ++b) {
            const std::uint8_t* blk = row + b * 144u;
            const float d    = read_fp16_le(blk + 0);
            const float dmin = read_fp16_le(blk + 2);
            const std::uint8_t* sm = blk + 4;
            const std::uint8_t* qs = blk + 16;
            const float* xblk = x + b * 256u;

            for (unsigned j = 0; j < 4; ++j) {
                unsigned scale_q0 = 0, min_q0 = 0, scale_q1 = 0, min_q1 = 0;
                unpack_scale_min_6bit(sm, 2 * j,     scale_q0, min_q0);
                unpack_scale_min_6bit(sm, 2 * j + 1, scale_q1, min_q1);

                const __m256 d1 = _mm256_set1_ps(d    * static_cast<float>(scale_q0));
                const __m256 m1 = _mm256_set1_ps(dmin * static_cast<float>(min_q0));
                const __m256 d2 = _mm256_set1_ps(d    * static_cast<float>(scale_q1));
                const __m256 m2 = _mm256_set1_ps(dmin * static_cast<float>(min_q1));

                const __m256i v  = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(qs + j * 32u));
                const __m256i lo = _mm256_and_si256(v, mask0f);
                const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), mask0f);

                const float* xsub = xblk + j * 64u;
                dot32_nibbles(lo, xsub,      d1, m1, a0, a1, a2, a3);
                dot32_nibbles(hi, xsub + 32, d2, m2, a4, a5, a6, a7);
            }
        }
        const __m256 s01 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        const __m256 s23 = _mm256_add_ps(_mm256_add_ps(a4, a5), _mm256_add_ps(a6, a7));
        y[m] = hsum256(_mm256_add_ps(s01, s23));
    }
}

void matvec_q4k_f32(const std::uint8_t* w, const float* x, float* y,
                    std::size_t M, std::size_t K) noexcept {
    matvec_q4k_f32_avx2(w, x, y, M, K);
}

} // namespace ultima::kernels
