#include "ultima/kernels/dequant_q8_0.hpp"

#include <cstdint>
#include <cstring>

namespace ultima::kernels {

namespace {

// (Duplicated locally to keep this TU standalone; a shared fp16 helper is a
// v0.2 refactor candidate.)
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

float read_fp16_le(const std::uint8_t* bytes) noexcept {
    const std::uint16_t h = static_cast<std::uint16_t>(bytes[0])
                          | (static_cast<std::uint16_t>(bytes[1]) << 8);
    return fp16_to_float(h);
}

} // namespace

void dequant_q8_0_block(const std::uint8_t* block, float* out) noexcept {
    const float d = read_fp16_le(block);
    const std::int8_t* qs = reinterpret_cast<const std::int8_t*>(block + 2);
    for (unsigned i = 0; i < 32; ++i) {
        out[i] = d * static_cast<float>(qs[i]);
    }
}

void dequant_q8_0(const std::uint8_t* blocks, std::size_t block_count,
                  float* out) noexcept {
    for (std::size_t b = 0; b < block_count; ++b) {
        dequant_q8_0_block(blocks + b * 34u, out + b * 32u);
    }
}

} // namespace ultima::kernels
