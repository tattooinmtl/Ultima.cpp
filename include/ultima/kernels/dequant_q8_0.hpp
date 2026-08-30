#pragma once

#include <cstddef>
#include <cstdint>

namespace ultima::kernels {

// ---- Q8_0 block layout (32 elements, 34 bytes) -----------------------------
//
//   offset 0..1  : fp16 d       (per-block scale)
//   offset 2..33 : int8[32] qs  (signed quantized values in [-127, 127])
//
// Dequant: value[i] = d * qs[i]

void dequant_q8_0_block(const std::uint8_t* block_bytes, float* out) noexcept;

void dequant_q8_0(const std::uint8_t* blocks, std::size_t block_count,
                  float* out) noexcept;

} // namespace ultima::kernels
