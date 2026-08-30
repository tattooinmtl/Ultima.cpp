#pragma once

#include <cstddef>
#include <cstdint>

namespace ultima::kernels {

// ---- Q4_K super-block layout (256 elements, 144 bytes) ---------------------
//
// The K-quant Q4_K format packs 256 4-bit values into 8 sub-blocks of 32,
// each with its own 6-bit scale and 6-bit min, plus two fp16 super-block
// scales that multiply the quantized scales and mins respectively.
//
// Super-block byte layout (144 bytes total):
//   offset  0..1   : fp16 d        (super-scale for the quantized scales)
//   offset  2..3   : fp16 dmin     (super-scale for the quantized mins)
//   offset  4..15  : uint8[12]     (packed 6-bit scales + 6-bit mins, 8 of each)
//   offset 16..143 : uint8[128]    (packed 4-bit values: 256 nibbles, low-first)
//
// Sub-block scales/mins packing across scales_and_mins[12]:
//   bytes  0..3  low 6 bits : scales[0..3]
//   bytes  4..7  low 6 bits : mins[0..3]
//   bytes  8..11 low 4 bits : scales[4..7]  (low nibble)
//   bytes  8..11 high 4 bits: mins[4..7]    (low nibble)
//   bytes  0..3  high 2 bits: scales[4..7]  (high 2 bits, shifted << 4)
//   bytes  4..7  high 2 bits: mins[4..7]    (high 2 bits, shifted << 4)
//
// Value packing in qs[128]:
//   For each sub-block s in 0..7:
//     qs[s*16 + i] low  nibble = q4[s*32 + 2*i    ]  for i in 0..15
//     qs[s*16 + i] high nibble = q4[s*32 + 2*i + 1]  for i in 0..15
//
// Dequant formula for element e in sub-block s of super-block b:
//   scale_s = unpack6(scales_and_mins, s)
//   min_s   = unpack6(scales_and_mins, s + 8)
//   real_scale = d * scale_s
//   real_min   = dmin * min_s
//   value = real_scale * q4[b*256 + s*32 + e] - real_min

// Dequantize one Q4_K super-block (144 bytes -> 256 floats).
// `out` must have room for 256 elements.
void dequant_q4k_block(const std::uint8_t* block_bytes, float* out) noexcept;

// Dequantize a range of Q4_K blocks (block_count super-blocks) into `out`.
// `out` must have room for block_count * 256 floats.
void dequant_q4k(const std::uint8_t* blocks, std::size_t block_count,
                 float* out) noexcept;

} // namespace ultima::kernels
