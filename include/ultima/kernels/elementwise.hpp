#pragma once

#include <cstddef>

namespace ultima::kernels {

// y[i] = a[i] + b[i]. In-place safe when y == a or y == b.
void add_f32       (const float* a, const float* b, float* y, std::size_t n) noexcept;
void add_f32_scalar(const float* a, const float* b, float* y, std::size_t n) noexcept;
void add_f32_avx2  (const float* a, const float* b, float* y, std::size_t n) noexcept;

// y[i] = a[i] * b[i]. In-place safe.
void mul_f32       (const float* a, const float* b, float* y, std::size_t n) noexcept;
void mul_f32_scalar(const float* a, const float* b, float* y, std::size_t n) noexcept;
void mul_f32_avx2  (const float* a, const float* b, float* y, std::size_t n) noexcept;

// y[i] = silu(x[i]) = x[i] / (1 + exp(-x[i])).
// In-place safe when y == x.
// v0.1 uses scalar std::exp — an AVX2 polynomial variant is a v0.2 perf item.
void silu_f32       (const float* x, float* y, std::size_t n) noexcept;
void silu_f32_scalar(const float* x, float* y, std::size_t n) noexcept;

// SwiGLU: y[i] = silu(gate[i]) * up[i]. This is the fused-per-element operator
// the Qwen2/3 MLP wants after its gate/up projections.
// In-place safe when y == gate or y == up.
void swiglu_f32       (const float* gate, const float* up, float* y, std::size_t n) noexcept;
void swiglu_f32_scalar(const float* gate, const float* up, float* y, std::size_t n) noexcept;

// GEGLU: y[i] = gelu(gate[i]) * up[i]. Same shape as SwiGLU with GELU in place
// of SiLU — Spark-X2.5 uses this.
//
// Uses the tanh approximation, evaluated directly:
//   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Not the exact erf form — that is a visibly different function here. ggml
// evaluates the same approximation through an fp16-indexed lookup table, so
// it is slightly *less* precise than this; expect agreement on all but
// near-tied logits rather than bit-exact equality.
// In-place safe when y == gate or y == up.
void geglu_f32       (const float* gate, const float* up, float* y, std::size_t n) noexcept;
void geglu_f32_scalar(const float* gate, const float* up, float* y, std::size_t n) noexcept;

} // namespace ultima::kernels
