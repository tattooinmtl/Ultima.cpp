#include "ultima/kernels/dequant_q8_0.hpp"
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
            exp   = 127u - 14u - shift + 1u;
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

} // namespace

// ----- Scalar oracle --------------------------------------------------------
void matvec_q8_0_f32_scalar(const std::uint8_t* w, const float* x, float* y,
                            std::size_t M, std::size_t K) noexcept {
    if (K == 0 || (K % 32u) != 0) {
        for (std::size_t m = 0; m < M; ++m) y[m] = 0.0f;
        return;
    }
    const std::size_t blocks_per_row = K / 32u;
    const std::size_t bytes_per_row  = blocks_per_row * 34u;

    float scratch[32];
    for (std::size_t m = 0; m < M; ++m) {
        const std::uint8_t* row = w + m * bytes_per_row;
        float acc = 0.0f;
        for (std::size_t b = 0; b < blocks_per_row; ++b) {
            dequant_q8_0_block(row + b * 34u, scratch);
            const float* xslice = x + b * 32u;
            for (std::size_t i = 0; i < 32; ++i) acc += scratch[i] * xslice[i];
        }
        y[m] = acc;
    }
}

// ----- AVX2 + FMA -----------------------------------------------------------
// Fused: no F32 weight scratch. Per 32-elt Q8_0 block we do 4 chunks of 8
// int8 → int32 → f32 → FMA against the x slice; then scale the block's
// running acc by d and fold into the row acc.
void matvec_q8_0_f32_avx2(const std::uint8_t* w, const float* x, float* y,
                          std::size_t M, std::size_t K) noexcept {
    if (K == 0 || (K % 32u) != 0) {
        for (std::size_t m = 0; m < M; ++m) y[m] = 0.0f;
        return;
    }
    const std::size_t blocks_per_row = K / 32u;
    const std::size_t bytes_per_row  = blocks_per_row * 34u;

    for (std::size_t m = 0; m < M; ++m) {
        const std::uint8_t* row = w + m * bytes_per_row;
        __m256 row_acc = _mm256_setzero_ps();

        for (std::size_t b = 0; b < blocks_per_row; ++b) {
            const std::uint8_t* blk = row + b * 34u;
            const float d = read_fp16_le(blk);
            const std::int8_t* qs = reinterpret_cast<const std::int8_t*>(blk + 2);
            const float* xslice = x + b * 32u;

            __m256 blk_acc = _mm256_setzero_ps();
            for (int c = 0; c < 4; ++c) {
                __m128i qs8 = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(qs + c * 8));
                __m256i qs32 = _mm256_cvtepi8_epi32(qs8);
                __m256  qsf  = _mm256_cvtepi32_ps(qs32);
                __m256  xf   = _mm256_loadu_ps(xslice + c * 8);
                blk_acc = _mm256_fmadd_ps(qsf, xf, blk_acc);
            }
            __m256 dv = _mm256_set1_ps(d);
            row_acc = _mm256_fmadd_ps(dv, blk_acc, row_acc);
        }
        y[m] = hsum256(row_acc);
    }
}

void matvec_q8_0_f32(const std::uint8_t* w, const float* x, float* y,
                     std::size_t M, std::size_t K) noexcept {
    matvec_q8_0_f32_avx2(w, x, y, M, K);
}

} // namespace ultima::kernels
