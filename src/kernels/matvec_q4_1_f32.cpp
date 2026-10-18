#include "ultima/kernels/dequant_q4_1.hpp"
#include "ultima/kernels/matvec.hpp"

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace ultima::kernels {

namespace {

inline float hsum256(__m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

} // namespace

void matvec_q4_1_f32_scalar(const std::uint8_t* w, const float* x, float* y,
                            std::size_t M, std::size_t K) noexcept {
    if (K == 0 || (K % 32u) != 0) {
        for (std::size_t m = 0; m < M; ++m) y[m] = 0.0f;
        return;
    }
    const std::size_t blocks_per_row = K / 32u;
    const std::size_t bytes_per_row  = blocks_per_row * 20u;

    float scratch[32];
    for (std::size_t m = 0; m < M; ++m) {
        const std::uint8_t* row = w + m * bytes_per_row;
        float acc = 0.0f;
        for (std::size_t b = 0; b < blocks_per_row; ++b) {
            dequant_q4_1_block(row + b * 20u, scratch);
            const float* xslice = x + b * 32u;
            for (std::size_t i = 0; i < 32; ++i) acc += scratch[i] * xslice[i];
        }
        y[m] = acc;
    }
}

// Dequantize a block into a stack scratch, then FMA it against the matching
// slice of x. One block is exactly four AVX2 lanes, so the dot unrolls flat.
void matvec_q4_1_f32_avx2(const std::uint8_t* w, const float* x, float* y,
                          std::size_t M, std::size_t K) noexcept {
    if (K == 0 || (K % 32u) != 0) {
        for (std::size_t m = 0; m < M; ++m) y[m] = 0.0f;
        return;
    }
    const std::size_t blocks_per_row = K / 32u;
    const std::size_t bytes_per_row  = blocks_per_row * 20u;

    alignas(32) float scratch[32];
    for (std::size_t m = 0; m < M; ++m) {
        const std::uint8_t* row = w + m * bytes_per_row;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(),
               a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();

        for (std::size_t b = 0; b < blocks_per_row; ++b) {
            dequant_q4_1_block(row + b * 20u, scratch);
            const float* xs = x + b * 32u;
            a0 = _mm256_fmadd_ps(_mm256_load_ps (scratch +  0),
                                 _mm256_loadu_ps(xs +  0), a0);
            a1 = _mm256_fmadd_ps(_mm256_load_ps (scratch +  8),
                                 _mm256_loadu_ps(xs +  8), a1);
            a2 = _mm256_fmadd_ps(_mm256_load_ps (scratch + 16),
                                 _mm256_loadu_ps(xs + 16), a2);
            a3 = _mm256_fmadd_ps(_mm256_load_ps (scratch + 24),
                                 _mm256_loadu_ps(xs + 24), a3);
        }
        y[m] = hsum256(_mm256_add_ps(_mm256_add_ps(a0, a1),
                                     _mm256_add_ps(a2, a3)));
    }
}

void matvec_q4_1_f32(const std::uint8_t* w, const float* x, float* y,
                     std::size_t M, std::size_t K) noexcept {
    matvec_q4_1_f32_avx2(w, x, y, M, K);
}

} // namespace ultima::kernels
