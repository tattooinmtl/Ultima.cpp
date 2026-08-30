#pragma once

#include <cstddef>
#include <cstdint>

namespace ultima::kernels {

// ---- Q6_K super-block layout (256 elements, 210 bytes) ---------------------
//
//   offset  0..127 : ql[128]      (lower 4 bits of each 6-bit value)
//   offset 128..191: qh[64]       (upper 2 bits of each 6-bit value)
//   offset 192..207: scales[16]   (signed int8 scale, one per 16-elt sub-block)
//   offset 208..209: fp16 d       (super-block scale)
//
// Packing matches ggml's `block_q6_K` (ggml/src/ggml-quants.c). The 256
// elements are laid out as two 128-elt halves. Within each half:
//
//   for l in [0, 32):
//     is = l / 16
//     q1 = ((ql[l   ] & 0x0F) | (((qh[l] >> 0) & 0x03) << 4)) - 32   -> y[l    ], scale sc[is+0]
//     q2 = ((ql[l+32] & 0x0F) | (((qh[l] >> 2) & 0x03) << 4)) - 32   -> y[l+32], scale sc[is+2]
//     q3 = ((ql[l   ]  >> 4) | (((qh[l] >> 4) & 0x03) << 4)) - 32   -> y[l+64], scale sc[is+4]
//     q4 = ((ql[l+32]  >> 4) | (((qh[l] >> 6) & 0x03) << 4)) - 32   -> y[l+96], scale sc[is+6]
//
// The second half advances ql by 64, qh by 32, sc by 8, y by 128.
//
// Note: this is NOT the naive `ql[i/2] | (qh[i/4]<<4)` layout; the ggml
// interleaving stripes across the super-block so int8 dot-product kernels
// can consume it with clean lane alignment. All scale indices still resolve
// to `sc[i / 16]` when mapped back to output positions.
//
// value = d * scale * q6_signed  (q6_signed in [-32, 31])

void dequant_q6k_block(const std::uint8_t* block_bytes, float* out) noexcept;

void dequant_q6k(const std::uint8_t* blocks, std::size_t block_count,
                 float* out) noexcept;

} // namespace ultima::kernels
