#pragma once

#include <cstddef>
#include <cstdint>

namespace ultima::kernels {

// RoPE (rotary position embedding).
//
// Input `x` is shape [n_heads, head_dim] laid out contiguously. Applies the
// position rotation in-place. Only the first `rope_dim` dims of each head are
// rotated; any trailing dims (head_dim > rope_dim) pass through unchanged.
//
// Two conventions are in use across GGUF architectures and they are NOT
// interchangeable — a model roped the wrong way still produces fluent text,
// just with degraded long-range attention, so the choice is per-architecture
// data (see ArchSpec::rope_style), never a default:
//
//   NeoX ("half-split"): pairs are (j, j + rope_dim/2).
//                        Qwen2/3, Qwen3-MoE, Phi-3, GPT-NeoX.
//   Norm ("interleaved"): pairs are (2j, 2j+1).
//                        Llama, Granite.
//
// freq_base: 10000 for classic GPT-NeoX, 1e6 for Qwen2.5, 1e7 for Qwen3.
// position:  absolute token position (0-indexed).
//
// `freq_factors`, when non-null, is a rope_dim/2 array of per-dimension
// divisors applied to theta (LongRoPE / NTK-by-parts). `attn_factor` scales
// the resulting cos and sin; both default to the identity in rope_f32.

void rope_f32       (float* x,
                     std::size_t n_heads,
                     std::size_t head_dim,
                     std::size_t rope_dim,
                     std::size_t position,
                     float       freq_base) noexcept;

void rope_f32_scalar(float* x,
                     std::size_t n_heads,
                     std::size_t head_dim,
                     std::size_t rope_dim,
                     std::size_t position,
                     float       freq_base) noexcept;

// Full form. `neox` selects the pairing convention; `freq_factors` may be
// null; `attn_factor` of 1.0f means no magnitude scaling.
void rope_f32_ext   (float* x,
                     std::size_t  n_heads,
                     std::size_t  head_dim,
                     std::size_t  rope_dim,
                     std::size_t  position,
                     float        freq_base,
                     bool         neox,
                     const float* freq_factors,
                     float        attn_factor) noexcept;

} // namespace ultima::kernels
