#include "ultima/kernels/dequant_q4k.hpp"

#include <cstdint>
#include <cstring>

namespace ultima::kernels {

namespace {

// Convert IEEE 754 half-precision (fp16) to float, without hardware intrinsics.
// Handles subnormals and infinity/NaN. Compact and correct — perf-critical
// callers can inline or replace with hardware conversion later.
float fp16_to_float(std::uint16_t h) noexcept {
    const std::uint32_t sign = (h & 0x8000u) << 16;
    std::uint32_t exp  = (h & 0x7C00u) >> 10;
    std::uint32_t mant = (h & 0x03FFu);

    std::uint32_t out_bits;
    if (exp == 0) {
        if (mant == 0) {
            out_bits = sign;                 // +/- zero
        } else {
            // Subnormal — normalize.
            unsigned shift = 0;
            while ((mant & 0x0400u) == 0) { mant <<= 1; ++shift; }
            mant &= 0x03FFu;
            exp   = 127u - 14u - shift;
            out_bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        out_bits = sign | 0x7F800000u | (mant << 13);   // Inf/NaN
    } else {
        exp = exp - 15u + 127u;
        out_bits = sign | (exp << 23) | (mant << 13);
    }

    float f;
    std::memcpy(&f, &out_bits, sizeof(f));
    return f;
}

// Read a little-endian fp16 at bytes[0..1].
float read_fp16_le(const std::uint8_t* bytes) noexcept {
    std::uint16_t h = static_cast<std::uint16_t>(bytes[0])
                    | (static_cast<std::uint16_t>(bytes[1]) << 8);
    return fp16_to_float(h);
}

// Extract the 6-bit scale[s] and min[s] from the 12-byte scales_and_mins array.
// See dequant_q4k.hpp for the packing description.
void unpack_scale_min_6bit(const std::uint8_t* sm, unsigned s,
                           unsigned& out_scale, unsigned& out_min) noexcept {
    if (s < 4) {
        out_scale = sm[s]     & 0x3Fu;
        out_min   = sm[s + 4] & 0x3Fu;
    } else {
        // s in [4, 8): low 4 bits from bytes 8..11, high 2 bits from bytes 0..7.
        const unsigned idx = s - 4u;
        out_scale = (sm[8u + idx] & 0x0Fu) | ((sm[    idx] >> 6) << 4);
        out_min   = (sm[8u + idx] >>   4)  | ((sm[4u + idx] >> 6) << 4);
    }
}

} // namespace

void dequant_q4k_block(const std::uint8_t* block, float* out) noexcept {
    const float d    = read_fp16_le(block + 0);
    const float dmin = read_fp16_le(block + 2);

    const std::uint8_t* sm = block + 4;    // 12 bytes of scales+mins
    const std::uint8_t* qs = block + 16;   // 128 bytes of 4-bit values

    for (unsigned s = 0; s < 8; ++s) {
        unsigned scale_q = 0, min_q = 0;
        unpack_scale_min_6bit(sm, s, scale_q, min_q);

        const float scale = d    * static_cast<float>(scale_q);
        const float min_v = dmin * static_cast<float>(min_q);

        const std::uint8_t* sub_qs = qs + s * 16;   // 16 bytes = 32 nibbles
        float* sub_out = out + s * 32;
        for (unsigned i = 0; i < 16; ++i) {
            const unsigned lo = sub_qs[i] & 0x0Fu;
            const unsigned hi = (sub_qs[i] >> 4) & 0x0Fu;
            sub_out[2 * i    ] = scale * static_cast<float>(lo) - min_v;
            sub_out[2 * i + 1] = scale * static_cast<float>(hi) - min_v;
        }
    }
}

void dequant_q4k(const std::uint8_t* blocks, std::size_t block_count,
                 float* out) noexcept {
    for (std::size_t b = 0; b < block_count; ++b) {
        dequant_q4k_block(blocks + b * 144u, out + b * 256u);
    }
}

} // namespace ultima::kernels
