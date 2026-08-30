#pragma once

#include <cstddef>
#include <cstdint>

namespace ultima::kernels {

// ---- Q6_K super-block layout (256 elements, 210 bytes) ---------------------
//
//   offset  0..127 : ql[128]      (low 4 bits of each 6-bit value, 2 per byte)
//   offset 128..191: qh[64]       (high 2 bits of each 6-bit value, 4 per byte)
//   offset 192..207: scales[16]   (signed int8 scale per sub-block; 16 subs of 16 elts)
//   offset 208..209: fp16 d       (super-block scale)
//
// Per-value dequant for element i in [0, 256):
//   sub       = i / 16
//   scale     = scales[sub]              (int8, signed)
//   low4      = (ql[i / 2] >> ((i & 1) * 4)) & 0x0F
//   high2     = (qh[i / 4] >> ((i & 3) * 2)) & 0x03
//   q6        = low4 | (high2 << 4)     (unsigned in [0, 63])
//   q6_signed = q6 - 32                  (centered to [-32, 31])
//   value     = d * scale * q6_signed
//
// NOTE: This implementation uses the straightforward sequential packing. Real
// GGUFs may use interleaved sub-orderings for cache friendliness — verify
// against actual model weights at M4 wiring; adjust indexing if needed
// without changing the public API.

void dequant_q6k_block(const std::uint8_t* block_bytes, float* out) noexcept;

void dequant_q6k(const std::uint8_t* blocks, std::size_t block_count,
                 float* out) noexcept;

} // namespace ultima::kernels
