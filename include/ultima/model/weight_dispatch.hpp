#pragma once

#include "ultima/core/error.hpp"
#include "ultima/model/dtype.hpp"

#include <cstddef>
#include <cstdint>

namespace ultima::model {

// Multiply a row-major weight matrix W by an f32 vector x:
//   y[m] = sum_k W[m, k] * x[k]   for m in [0, M), k in [0, K)
// dtype selects which fused-dequant kernel runs. Errors on dtypes v0.1
// doesn't support (BF16, Q4_0, Q5_0, Q2/3/5_K, etc.).
core::Result<void>
matvec_dispatch(DataType dtype, const void* w, const float* x, float* y,
                std::size_t M, std::size_t K);

// Same, threaded across rows via the provided ThreadPool. Falls back to
// sequential when M is below a small threshold.
} // namespace ultima::model

namespace ultima::runtime { class ThreadPool; }

namespace ultima::model {

core::Result<void>
matvec_dispatch_threaded(runtime::ThreadPool& pool,
                         DataType dtype, const void* w, const float* x, float* y,
                         std::size_t M, std::size_t K);

// Batched form: multiply W once against `n_tok` activations.
//
//   Y[t][m] = sum_k W[m, k] * X[t][k]
//
// X is [n_tok][K] and Y is [n_tok][M], both row-major and token-major.
//
// This exists because decode is memory-bound on the weights: a matvec
// streams all of W to produce one token, so N separate matvecs move N*M*K
// bytes. Here each weight row is touched once and reused across every
// token, so the traffic is M*K regardless of N. On the reference box that
// is worth about 5x on prompt processing.
//
// Falls through to matvec_dispatch_threaded at n_tok == 1, where
// materializing rows would be pure overhead.
core::Result<void>
matmul_dispatch_threaded(runtime::ThreadPool& pool,
                         DataType dtype, const void* w,
                         const float* X, float* Y,
                         std::size_t M, std::size_t K, std::size_t n_tok);

// Convert a fp16 half (uint16 raw bits, LE) to a float.
float fp16_to_float(std::uint16_t h) noexcept;

// Dequantize a 1D tensor (norm weights, biases) into a caller-provided
// f32 buffer of length `n`. Handles F32, F16, Q4_K, Q6_K, Q8_0.
core::Result<void>
dequant_1d(DataType dtype, const void* src, float* dst, std::size_t n);

// Read one row of a 2D weight tensor into a caller-provided f32 buffer
// of length K (cols). Supports F32, F16, Q4_K, Q6_K, Q8_0. Rows must be
// K-aligned per the dtype's block size (256 for Q4_K/Q6_K, 32 for Q8_0).
core::Result<void>
read_row(DataType dtype, const void* base, std::size_t row, std::size_t K,
         float* dst);

} // namespace ultima::model
