#include "ultima/kernels/norms.hpp"

#include <cmath>
#include <cstddef>
#include <immintrin.h>

namespace ultima::kernels {

// ---- Scalar oracle ---------------------------------------------------------
void rmsnorm_f32_scalar(const float* x, const float* scale, float* y,
                        std::size_t n, float eps) noexcept {
    float ss = 0.0f;
    for (std::size_t i = 0; i < n; ++i) ss += x[i] * x[i];
    const float mean_sq = ss / static_cast<float>(n);
    const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);
    for (std::size_t i = 0; i < n; ++i) y[i] = x[i] * scale[i] * inv_rms;
}

// ---- AVX2 + FMA ------------------------------------------------------------
// Two passes: sum-of-squares reduction, then broadcast normalize.
void rmsnorm_f32_avx2(const float* x, const float* scale, float* y,
                      std::size_t n, float eps) noexcept {
    const std::size_t n8 = (n / 8) * 8;

    // ---- pass 1: sum of squares ----
    __m256 acc = _mm256_setzero_ps();
    for (std::size_t i = 0; i < n8; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    // Horizontal sum of the 8 lanes.
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float ss = _mm_cvtss_f32(s);

    for (std::size_t i = n8; i < n; ++i) ss += x[i] * x[i];

    const float mean_sq = ss / static_cast<float>(n);
    const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);

    // ---- pass 2: x * scale * inv_rms ----
    __m256 vinv = _mm256_set1_ps(inv_rms);
    for (std::size_t i = 0; i < n8; i += 8) {
        __m256 vx  = _mm256_loadu_ps(x + i);
        __m256 vsc = _mm256_loadu_ps(scale + i);
        __m256 t   = _mm256_mul_ps(vx, vsc);
        _mm256_storeu_ps(y + i, _mm256_mul_ps(t, vinv));
    }
    for (std::size_t i = n8; i < n; ++i) y[i] = x[i] * scale[i] * inv_rms;
}

void rmsnorm_f32(const float* x, const float* scale, float* y,
                 std::size_t n, float eps) noexcept {
    rmsnorm_f32_avx2(x, scale, y, n, eps);
}

} // namespace ultima::kernels
