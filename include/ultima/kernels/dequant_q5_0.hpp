#pragma once

#include <cstddef>
#include <cstdint>

namespace ultima::kernels {

// ---- Q5_0 block layout (32 elements, 22 bytes) -----------------------------
//
//   offset 0..1  : fp16 d       (per-block scale)
//   offset 2..5  : uint8[4] qh  (bit i = 5th bit of element i)
//   offset 6..21 : uint8[16] qs (4-bit nibbles: low = pos j, high = pos j+16)
//
// Per element i in [0, 32):
//   j     = i & 15
//   nib   = (i < 16) ? (qs[j] & 0x0F) : ((qs[j] >> 4) & 0x0F)
//   bit5  = (qh_u32 >> i) & 1
//   q     = (bit5 << 4) | nib          (unsigned in [0, 31])
//   value = d * (q - 16)                (signed range [-16, 15] * d)
//
// Symmetric around zero (no min offset), so this is a "5-bit signed"
// per-block quantization with a single fp16 scale.
//
// See ggml's block_q5_0 in ggml-common.h for the reference layout.

void dequant_q5_0_block(const std::uint8_t* block_bytes, float* out) noexcept;

void dequant_q5_0(const std::uint8_t* blocks, std::size_t block_count,
                  float* out) noexcept;

} // namespace ultima::kernels
