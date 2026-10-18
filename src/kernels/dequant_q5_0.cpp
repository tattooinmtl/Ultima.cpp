#include "ultima/kernels/dequant_q5_0.hpp"

#include <cstdint>
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

} // namespace

void dequant_q5_0_block(const std::uint8_t* block, float* out) noexcept {
    const float d = read_fp16_le(block);
    // qh is a little-endian uint32 packed across bytes 2..5.
    const std::uint32_t qh =
          static_cast<std::uint32_t>(block[2])
        | (static_cast<std::uint32_t>(block[3]) << 8)
        | (static_cast<std::uint32_t>(block[4]) << 16)
        | (static_cast<std::uint32_t>(block[5]) << 24);
    const std::uint8_t* qs = block + 6;

    // Lower half: elements 0..15 use low nibble of qs[j] plus bit j of qh.
    // Upper half: elements 16..31 use high nibble of qs[j-16] plus bit j of qh.
    for (unsigned i = 0; i < 32; ++i) {
        const unsigned j    = i & 15u;
        const unsigned nib  = (i < 16) ? (qs[j] & 0x0Fu)
                                       : ((qs[j] >> 4) & 0x0Fu);
        const unsigned bit5 = (qh >> i) & 1u;
        const int      q    = static_cast<int>((bit5 << 4) | nib);
        out[i] = d * static_cast<float>(q - 16);
    }
}

void dequant_q5_0(const std::uint8_t* blocks, std::size_t block_count,
                  float* out) noexcept {
    for (std::size_t b = 0; b < block_count; ++b) {
        dequant_q5_0_block(blocks + b * 22u, out + b * 32u);
    }
}

} // namespace ultima::kernels
