#include "ultima/kernels/dequant_q6k.hpp"

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

float read_fp16_le(const std::uint8_t* bytes) noexcept {
    const std::uint16_t h = static_cast<std::uint16_t>(bytes[0])
                          | (static_cast<std::uint16_t>(bytes[1]) << 8);
    return fp16_to_float(h);
}

} // namespace

// Matches ggml's dequantize_row_q6_K packing.
void dequant_q6k_block(const std::uint8_t* block, float* out) noexcept {
    const std::uint8_t* ql_base = block + 0;
    const std::uint8_t* qh_base = block + 128;
    const std::int8_t*  sc_base = reinterpret_cast<const std::int8_t*>(block + 192);
    const float d = read_fp16_le(block + 208);

    for (unsigned half = 0; half < 2; ++half) {
        const std::uint8_t* ql = ql_base + half * 64u;
        const std::uint8_t* qh = qh_base + half * 32u;
        const std::int8_t*  sc = sc_base + half * 8u;
        float* y = out + half * 128u;

        for (unsigned l = 0; l < 32; ++l) {
            const unsigned is = l >> 4;   // 0 for l<16, 1 for l>=16
            const int q1 = static_cast<int>(
                (ql[l     ] & 0x0Fu) | (((qh[l] >> 0) & 0x03u) << 4)) - 32;
            const int q2 = static_cast<int>(
                (ql[l + 32] & 0x0Fu) | (((qh[l] >> 2) & 0x03u) << 4)) - 32;
            const int q3 = static_cast<int>(
                (ql[l     ] >> 4)    | (((qh[l] >> 4) & 0x03u) << 4)) - 32;
            const int q4 = static_cast<int>(
                (ql[l + 32] >> 4)    | (((qh[l] >> 6) & 0x03u) << 4)) - 32;

            y[l     ] = d * static_cast<float>(sc[is + 0]) * static_cast<float>(q1);
            y[l + 32] = d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2);
            y[l + 64] = d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3);
            y[l + 96] = d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4);
        }
    }
}

void dequant_q6k(const std::uint8_t* blocks, std::size_t block_count,
                 float* out) noexcept {
    for (std::size_t b = 0; b < block_count; ++b) {
        dequant_q6k_block(blocks + b * 210u, out + b * 256u);
    }
}

} // namespace ultima::kernels
