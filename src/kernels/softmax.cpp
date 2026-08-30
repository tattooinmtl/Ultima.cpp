#include "ultima/kernels/softmax.hpp"

#include <cmath>
#include <cstddef>
#include <limits>

namespace ultima::kernels {

void softmax_f32_scalar(const float* x, float* y, std::size_t n) noexcept {
    if (n == 0) return;

    // Pass 1: max (numerical stability shift)
    float m = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        if (x[i] > m) m = x[i];
    }

    // Pass 2: exp(x - m), accumulate sum
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        const float e = std::exp(x[i] - m);
        y[i] = e;
        sum += e;
    }

    // Pass 3: normalize
    const float inv = 1.0f / sum;
    for (std::size_t i = 0; i < n; ++i) y[i] *= inv;
}

void softmax_f32(const float* x, float* y, std::size_t n) noexcept {
    // v0.1 uses the scalar path — std::exp is the bottleneck and vector expf
    // requires a polynomial approximation (deferred to v0.2 perf work).
    softmax_f32_scalar(x, y, n);
}

} // namespace ultima::kernels
