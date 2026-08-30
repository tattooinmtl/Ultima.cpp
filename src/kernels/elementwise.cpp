#include "ultima/kernels/elementwise.hpp"

#include <cmath>
#include <cstddef>
#include <immintrin.h>

namespace ultima::kernels {

// ----- add ------------------------------------------------------------------
void add_f32_scalar(const float* a, const float* b, float* y, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) y[i] = a[i] + b[i];
}

void add_f32_avx2(const float* a, const float* b, float* y, std::size_t n) noexcept {
    const std::size_t n8 = (n / 8) * 8;
    for (std::size_t i = 0; i < n8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(y + i, _mm256_add_ps(va, vb));
    }
    for (std::size_t i = n8; i < n; ++i) y[i] = a[i] + b[i];
}

void add_f32(const float* a, const float* b, float* y, std::size_t n) noexcept {
    add_f32_avx2(a, b, y, n);
}

// ----- mul ------------------------------------------------------------------
void mul_f32_scalar(const float* a, const float* b, float* y, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) y[i] = a[i] * b[i];
}

void mul_f32_avx2(const float* a, const float* b, float* y, std::size_t n) noexcept {
    const std::size_t n8 = (n / 8) * 8;
    for (std::size_t i = 0; i < n8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(y + i, _mm256_mul_ps(va, vb));
    }
    for (std::size_t i = n8; i < n; ++i) y[i] = a[i] * b[i];
}

void mul_f32(const float* a, const float* b, float* y, std::size_t n) noexcept {
    mul_f32_avx2(a, b, y, n);
}

// ----- silu -----------------------------------------------------------------
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// std::exp is the bottleneck. AVX2 polynomial exp = v0.2 perf item.
void silu_f32_scalar(const float* x, float* y, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        const float xi = x[i];
        y[i] = xi / (1.0f + std::exp(-xi));
    }
}

void silu_f32(const float* x, float* y, std::size_t n) noexcept {
    silu_f32_scalar(x, y, n);
}

// ----- swiglu ----------------------------------------------------------------
void swiglu_f32_scalar(const float* gate, const float* up, float* y, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        const float g = gate[i];
        const float s = g / (1.0f + std::exp(-g));
        y[i] = s * up[i];
    }
}

void swiglu_f32(const float* gate, const float* up, float* y, std::size_t n) noexcept {
    swiglu_f32_scalar(gate, up, y, n);
}

} // namespace ultima::kernels
