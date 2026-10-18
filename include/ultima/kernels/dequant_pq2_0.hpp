#pragma once

#include <cstddef>
#include <cstdint>

namespace ultima::kernels {

// ---- PQ2_0 block layout (128 elements, 34 bytes) ---------------------------
//
//   offset 0..1  : fp16 d      (per-group scale)
//   offset 2..33 : uint8[32] qs (2-bit codes, 4 per byte, low bits first)
//
// Per element i in [0, 128):
//   byte  = qs[i / 4]
//   code  = (byte >> (2 * (i % 4))) & 0x03
//   value = (code - 1) * d
//
// so the code maps 0,1,2,3 -> -1, 0, +1, +2. The checkpoints are ternary;
// the fourth code exists because two bits are available, and the reference
// dequantizer emits +2 for it rather than treating it as reserved.
//
// 2.125 bits per weight. This is a Prism-private format: its type id (142)
// is not in the public GGUF spec, and stock llama.cpp rejects a file using
// it at tensor-info parse rather than misreading it.
//
// See block_pq2_0 in the PrismML-Eng llama.cpp fork's ggml-common.h.
//
// NOTE: weights in these files also carry a folded Hadamard rotation, which
// has to be undone by transforming the *activation* before the matvec. A
// tensor dequantized here is therefore the rotated weight, not the original
// one; see the Hadamard pass for the other half.

void dequant_pq2_0_block(const std::uint8_t* block_bytes, float* out) noexcept;

void dequant_pq2_0(const std::uint8_t* blocks, std::size_t block_count,
                   float* out) noexcept;

// ---- Q2_0 (upstream type 42, 64 elements, 18 bytes) ------------------------
//
// Byte-for-byte the same layout and the same (code - 1) * d mapping, with a
// group of 64 rather than 128. Sharing the implementation is not a
// convenience here — it is the same format, and the only local file using
// it is the dspark drafter's token embedding.
void dequant_q2_0_block(const std::uint8_t* block_bytes, float* out) noexcept;

void dequant_q2_0(const std::uint8_t* blocks, std::size_t block_count,
                  float* out) noexcept;

} // namespace ultima::kernels
