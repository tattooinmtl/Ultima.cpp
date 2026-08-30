#include "ultima/kernels/matvec.hpp"

#include <immintrin.h>

namespace ultima::kernels {

// ----- Scalar reference (correctness oracle) --------------------------------
void matvec_f32_f32_scalar(const float* w, const float* x, float* y,
                           std::size_t M, std::size_t K) noexcept {
    for (std::size_t m = 0; m < M; ++m) {
        float acc = 0.0f;
        const float* row = w + m * K;
        for (std::size_t k = 0; k < K; ++k) {
            acc += row[k] * x[k];
        }
        y[m] = acc;
    }
}

// ----- AVX2 + FMA path -------------------------------------------------------
// Processes one output row at a time, 8 lanes per FMA. Tail (K % 8) done
// scalar to keep the vector loop clean.
void matvec_f32_f32_avx2(const float* w, const float* x, float* y,
                         std::size_t M, std::size_t K) noexcept {
    const std::size_t K8 = (K / 8) * 8;

    for (std::size_t m = 0; m < M; ++m) {
        const float* row = w + m * K;
        __m256 acc = _mm256_setzero_ps();

        for (std::size_t k = 0; k < K8; k += 8) {
            __m256 wv = _mm256_loadu_ps(row + k);
            __m256 xv = _mm256_loadu_ps(x + k);
            acc = _mm256_fmadd_ps(wv, xv, acc);
        }

        // Horizontal sum of the 8 lanes.
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 s  = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        float sum = _mm_cvtss_f32(s);

        // Scalar tail
        for (std::size_t k = K8; k < K; ++k) {
            sum += row[k] * x[k];
        }
        y[m] = sum;
    }
}

// ----- Public dispatcher -----------------------------------------------------
void matvec_f32_f32(const float* w, const float* x, float* y,
                    std::size_t M, std::size_t K) noexcept {
    matvec_f32_f32_avx2(w, x, y, M, K);
}

} // namespace ultima::kernels
