#pragma once

#include <cstddef>

namespace ultima::kernels {

// Numerically stable softmax:
//   m  = max(x)
//   e  = exp(x - m)
//   y  = e / sum(e)
//
// In-place safe when y == x.
// v0.1 uses scalar std::exp inside the loop; AVX2 polynomial exp is a
// v0.2 perf item.
void softmax_f32       (const float* x, float* y, std::size_t n) noexcept;
void softmax_f32_scalar(const float* x, float* y, std::size_t n) noexcept;

} // namespace ultima::kernels
