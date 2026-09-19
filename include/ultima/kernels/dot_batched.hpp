#pragma once

#include <cstddef>

namespace ultima::kernels {

// Dot one dequantized weight row against several activation vectors.
//
//   out[t * out_stride] = sum_k row[k] * X[t * K + k]
//
// This is the inner loop of the batched projection path. The row has
// already been materialized as f32, so the point here is register reuse:
// the row is loaded once per group of tokens rather than once per token,
// which is what makes batching beat a fused matvec per token despite the
// materialization.
void dot_rows_f32(const float* row, const float* X, std::size_t K,
                  std::size_t n_tok, float* out, std::size_t out_stride) noexcept;

void dot_rows_f32_scalar(const float* row, const float* X, std::size_t K,
                         std::size_t n_tok, float* out,
                         std::size_t out_stride) noexcept;

} // namespace ultima::kernels
