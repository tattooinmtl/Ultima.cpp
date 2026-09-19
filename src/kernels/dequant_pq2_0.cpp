#include "ultima/kernels/dequant_pq2_0.hpp"

#include <cstring>

namespace ultima::kernels {

namespace {

float fp16_to_float(std::uint16_t h) noexcept {
    const std::uint32_t sign = (h & 0x8000u) << 16;
    std::uint32_t exp  = (h & 0x7C00u) >> 10;
    std::uint32_t mant = (h & 0x03FFu);
    std::uint32_t out_bits;
    if (exp == 0) {
        if (mant == 0) {
            out_bits = sign;
        } else {
            unsigned shift = 0;
            while ((mant & 0x0400u) == 0) { mant <<= 1; ++shift; }
            mant &= 0x03FFu;
            exp   = 127u - 14u - shift;
            out_bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        out_bits = sign | 0x7F800000u | (mant << 13);
    } else {
        exp = exp - 15u + 127u;
        out_bits = sign | (exp << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out_bits, sizeof(f));
    return f;
}

inline float read_fp16_le(const std::uint8_t* p) noexcept {
    const std::uint16_t h = static_cast<std::uint16_t>(p[0])
                          | (static_cast<std::uint16_t>(p[1]) << 8);
    return fp16_to_float(h);
}

constexpr std::size_t k_block  = 128;
constexpr std::size_t k_bytes  = 34;

} // namespace

void dequant_pq2_0_block(const std::uint8_t* block_bytes, float* out) noexcept {
    const float d = read_fp16_le(block_bytes);
    const std::uint8_t* qs = block_bytes + 2;

    // Four codes per byte, low pair first. Unrolled so the shifts are
    // constants — the whole block is one scale and 32 byte loads.
    for (std::size_t b = 0; b < k_block / 4; ++b) {
        const std::uint8_t byte = qs[b];
        out[4 * b + 0] = (static_cast<int>( byte        & 0x03) - 1) * d;
        out[4 * b + 1] = (static_cast<int>((byte >> 2)  & 0x03) - 1) * d;
        out[4 * b + 2] = (static_cast<int>((byte >> 4)  & 0x03) - 1) * d;
        out[4 * b + 3] = (static_cast<int>((byte >> 6)  & 0x03) - 1) * d;
    }
}

void dequant_pq2_0(const std::uint8_t* blocks, std::size_t block_count,
                   float* out) noexcept {
    for (std::size_t i = 0; i < block_count; ++i) {
        dequant_pq2_0_block(blocks + i * k_bytes, out + i * k_block);
    }
}

} // namespace ultima::kernels
