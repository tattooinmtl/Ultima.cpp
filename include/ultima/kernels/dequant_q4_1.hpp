#pragma once

#include <cstddef>
#include <cstdint>

namespace ultima::kernels {

// ---- Q4_1 block layout (32 elements, 20 bytes) -----------------------------
//
// The simplest affine 4-bit format: a per-block scale and a per-block minimum,
// both fp16, with 32 unsigned nibbles in [0, 15].
//
// Block byte layout (20 bytes total):
//   offset 0..1  : fp16 d   (scale)
//   offset 2..3  : fp16 m   (minimum)
//   offset 4..19 : uint8[16] (32 nibbles, low-first)
//
// Value packing (matches ggml's dequantize_row_q4_1):
//   For j in 0..15:
//     value[j]      = (qs[j] & 0x0F) * d + m
//     value[j + 16] = (qs[j] >>   4) * d + m
//
// Unlike Q4_0 there is no offset-binary bias; the minimum is stored outright.

// Dequantize one Q4_1 block (20 bytes -> 32 floats).
// `out` must have room for 32 elements.
void dequant_q4_1_block(const std::uint8_t* block_bytes, float* out) noexcept;

// Dequantize a range of Q4_1 blocks into `out`.
// `out` must have room for block_count * 32 floats.
void dequant_q4_1(const std::uint8_t* blocks, std::size_t block_count,
                  float* out) noexcept;

} // namespace ultima::kernels
