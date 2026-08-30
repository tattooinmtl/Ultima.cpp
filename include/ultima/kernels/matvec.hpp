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

// y[m] = sum_k dequant(W_q4k)[m, k] * x[k]
//
// W is Q4_K weights, row-major logically [M, K]. K must be a multiple of 256
// (super-block size). Dequant is fused inside the inner loop — no F32 scratch
// buffer for the whole weight matrix, only per-block during matvec.
//
// v0.1 uses a scalar fused path for correctness. AVX2 acceleration of the
// fused loop is a follow-up optimization but the public API is stable.
void matvec_q4k_f32       (const std::uint8_t* w, const float* x, float* y,
                           std::size_t M, std::size_t K) noexcept;

void matvec_q4k_f32_scalar(const std::uint8_t* w, const float* x, float* y,
                           std::size_t M, std::size_t K) noexcept;

} // namespace ultima::kernels
