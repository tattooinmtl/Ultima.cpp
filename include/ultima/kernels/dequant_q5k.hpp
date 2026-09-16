#pragma once

#include <cstddef>
#include <cstdint>

namespace ultima::kernels {

// ---- Q5_K super-block layout (256 elements, 176 bytes) ---------------------
//
// Q5_K is Q4_K plus a fifth bit per value, held in a separate 32-byte plane.
// Scales and mins are packed exactly as in Q4_K (8 sub-blocks of 32, each with
// a 6-bit scale and a 6-bit min, both scaled by an fp16 super-block factor).
//
// Super-block byte layout (176 bytes total):
//   offset   0..1   : fp16 d        (super-scale for the quantized scales)
//   offset   2..3   : fp16 dmin     (super-scale for the quantized mins)
//   offset   4..15  : uint8[12]     (packed 6-bit scales + 6-bit mins, 8 each)
//   offset  16..47  : uint8[32]     (qh — bit 5 of each of the 256 values)
//   offset  48..175 : uint8[128]    (qs — low 4 bits of each value)
//
// Value packing (matches ggml's dequantize_row_q5_K):
//   Sub-block pair (2j, 2j+1) shares 32 consecutive bytes qs[j*32 .. j*32+32).
//   The high bit for element l of that pair lives in qh[l], selected by the
//   masks u1 = 1 << (2j) and u2 = 1 << (2j + 1):
//     q5[2j*32     + l] = (qs[j*32 + l] & 0x0F) | (qh[l] & u1 ? 16 : 0)
//     q5[(2j+1)*32 + l] = (qs[j*32 + l] >>   4) | (qh[l] & u2 ? 16 : 0)
//
// Dequant formula for element e in sub-block s:
//   value = (d * scale_s) * q5[s*32 + e] - (dmin * min_s)
//
// Note the sign convention differs from Q5_0: Q5_K values are unsigned in
// [0, 31] with an explicit subtracted min, not offset-binary.

// Dequantize one Q5_K super-block (176 bytes -> 256 floats).
// `out` must have room for 256 elements.
void dequant_q5k_block(const std::uint8_t* block_bytes, float* out) noexcept;

// Dequantize a range of Q5_K blocks (block_count super-blocks) into `out`.
// `out` must have room for block_count * 256 floats.
void dequant_q5k(const std::uint8_t* blocks, std::size_t block_count,
                 float* out) noexcept;

} // namespace ultima::kernels
