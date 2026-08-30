#pragma once

#include <cstddef>
#include <cstdint>

namespace ultima::kernels {

// RoPE (rotary position embedding), Qwen2 / GPT-NeoX convention.
//
// Input `x` is shape [n_heads, head_dim] laid out contiguously. Applies
// the position rotation in-place. The rotation uses the "half-split"
// convention: element pairs are (i, i + rope_dim/2) rather than
// (2i, 2i+1). Only the first `rope_dim` dims of each head are rotated;
// any trailing dims (head_dim > rope_dim) are passed through unchanged.
//
// freq_base: 10000 for classic GPT-NeoX, 1_000_000 for Qwen2 / Qwen2.5.
// position:  absolute token position (0-indexed).
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

} // namespace ultima::kernels
