#pragma once

#include <cstddef>

namespace ultima::kernels {

// Root-mean-square normalization used by Qwen and most modern transformers.
//   ms       = (1/n) * sum(x[i]^2)
//   inv_rms  = 1 / sqrt(ms + eps)
//   y[i]     = x[i] * scale[i] * inv_rms
//
// `scale` has length n. In-place safe when y == x.
void rmsnorm_f32       (const float* x, const float* scale, float* y,
                        std::size_t n, float eps) noexcept;
void rmsnorm_f32_scalar(const float* x, const float* scale, float* y,
                        std::size_t n, float eps) noexcept;
void rmsnorm_f32_avx2  (const float* x, const float* scale, float* y,
                        std::size_t n, float eps) noexcept;

} // namespace ultima::kernels
