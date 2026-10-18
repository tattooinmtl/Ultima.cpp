#include "ultima/kernels/dequant_q5k.hpp"
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

void matvec_q5k_f32_scalar(const std::uint8_t* w, const float* x, float* y,
                           std::size_t M, std::size_t K) noexcept {
    if (K == 0 || (K % 256u) != 0) {
        for (std::size_t m = 0; m < M; ++m) y[m] = 0.0f;
        return;
    }
    const std::size_t blocks_per_row = K / 256u;
    const std::size_t bytes_per_row  = blocks_per_row * 176u;

    float scratch[256];
    for (std::size_t m = 0; m < M; ++m) {
        const std::uint8_t* row = w + m * bytes_per_row;
        float acc = 0.0f;
        for (std::size_t b = 0; b < blocks_per_row; ++b) {
            dequant_q5k_block(row + b * 176u, scratch);
            const float* xslice = x + b * 256u;
            for (std::size_t i = 0; i < 256; ++i) acc += scratch[i] * xslice[i];
        }
        y[m] = acc;
    }
}

// Dequantize a super-block into a stack scratch, then FMA it against the
// matching slice of x. The unpack itself stays scalar — Q5_K's split high-bit
// plane makes a fully fused SIMD unpack fiddly, and the dot dominates.
void matvec_q5k_f32_avx2(const std::uint8_t* w, const float* x, float* y,
                         std::size_t M, std::size_t K) noexcept {
    if (K == 0 || (K % 256u) != 0) {
        for (std::size_t m = 0; m < M; ++m) y[m] = 0.0f;
        return;
    }
    const std::size_t blocks_per_row = K / 256u;
    const std::size_t bytes_per_row  = blocks_per_row * 176u;

    alignas(32) float scratch[256];
    for (std::size_t m = 0; m < M; ++m) {
        const std::uint8_t* row = w + m * bytes_per_row;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(),
               a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();

        for (std::size_t b = 0; b < blocks_per_row; ++b) {
            dequant_q5k_block(row + b * 176u, scratch);
            const float* xs = x + b * 256u;
            for (unsigned i = 0; i < 256; i += 32) {
                a0 = _mm256_fmadd_ps(_mm256_load_ps (scratch + i +  0),
                                     _mm256_loadu_ps(xs + i +  0), a0);
                a1 = _mm256_fmadd_ps(_mm256_load_ps (scratch + i +  8),
                                     _mm256_loadu_ps(xs + i +  8), a1);
                a2 = _mm256_fmadd_ps(_mm256_load_ps (scratch + i + 16),
                                     _mm256_loadu_ps(xs + i + 16), a2);
                a3 = _mm256_fmadd_ps(_mm256_load_ps (scratch + i + 24),
                                     _mm256_loadu_ps(xs + i + 24), a3);
            }
        }
        y[m] = hsum256(_mm256_add_ps(_mm256_add_ps(a0, a1),
                                     _mm256_add_ps(a2, a3)));
    }
}

void matvec_q5k_f32(const std::uint8_t* w, const float* x, float* y,
                    std::size_t M, std::size_t K) noexcept {
    matvec_q5k_f32_avx2(w, x, y, M, K);
}

} // namespace ultima::kernels
