#pragma once

#include <cstddef>

// Gated delta-net linear attention — the mixer qwen35 (Qwen3-Next / Qwen3.5)
// runs on three of every four blocks.
//
// Unlike attention, this carries a fixed-size state instead of a growing KV
// cache, which is why the architecture stays cheap at long context. The cost
// is that the recurrence is sequential: token t's state depends on t-1, so
// prefill cannot be batched across positions the way a QK^T can.
//
// Layout note. The per-head state is S[i][j] with i indexing the key
// dimension and j the value dimension, stored row-major so j is contiguous.
// Both are d_state wide.

namespace ultima::kernels {

// y[i] = x[i] / sqrt(sum(x^2) + eps), over one head's worth of values.
//
// This is true L2 normalization, not RMS. The reference spells it as
// rms_norm(x, eps/n) * 1/sqrt(n), which reduces to exactly this; written
// directly it is both clearer and cheaper.
// In-place safe when y == x.
void l2_norm_f32(const float* x, float* y, std::size_t n, float eps) noexcept;

// y[i] = log1p(exp(x[i])), evaluated so large x does not overflow.
// In-place safe.
void softplus_f32(const float* x, float* y, std::size_t n) noexcept;

// One step of a causal depthwise convolution, for a single token.
//
// `state` holds the previous (width-1) inputs per channel, laid out
// [channels][width-1] with the oldest sample first. `x` is this token's
// input, one value per channel. `w` is the kernel as GGUF stores it:
// [channels][width], so taps for a channel are contiguous.
//
//   y[c] = sum_k w[c][k] * history[c][k]
//
// where history is state[c] followed by x[c]. The state is then rolled
// forward so the caller does not have to.
//
// `state` is mutated. y may alias x.
void causal_conv1d_step_f32(float* state, const float* x, const float* w,
                            float* y, std::size_t channels,
                            std::size_t width) noexcept;

// One step of the gated delta rule, for a single head and a single token.
//
//   S[i][j] *= decay                        (decay = exp(g), scalar per head)
//   sk[j]    = sum_i S[i][j] * k[i]
//   d[j]     = beta * (v[j] - sk[j])
//   S[i][j] += k[i] * d[j]
//   out[j]   = sum_i S[i][j] * q[i]
//
// `q` must already carry the 1/sqrt(d_state) scale. The output is read from
// the state *after* the update, not before. `scratch` needs d_state floats.
//
// `state` is mutated in place.
void delta_net_step_f32(float* state, const float* q, const float* k,
                        const float* v, float decay, float beta,
                        float* out, float* scratch,
                        std::size_t d_state) noexcept;

} // namespace ultima::kernels
