#pragma once

#include <cstddef>
#include <cstdint>

namespace ultima::kernels {

// y[m] = sum_k W[m,k] * x[k]     row-major W of shape [M, K], x length K
//
// The main public entry uses the best available implementation for the
// runtime CPU (AVX2 in v0.1). The `*_scalar` variant exists for correctness
// oracles and unit tests.
void matvec_f32_f32       (const float* w, const float* x, float* y,
                           std::size_t M, std::size_t K) noexcept;

void matvec_f32_f32_scalar(const float* w, const float* x, float* y,
                           std::size_t M, std::size_t K) noexcept;

void matvec_f32_f32_avx2  (const float* w, const float* x, float* y,
                           std::size_t M, std::size_t K) noexcept;

} // namespace ultima::kernels
