#include "ultima/kernels/dequant_q5k.hpp"

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

// Identical packing to Q4_K — see dequant_q4k.hpp for the bit layout.
void unpack_scale_min_6bit(const std::uint8_t* sm, unsigned s,
                           unsigned& out_scale, unsigned& out_min) noexcept {
    if (s < 4) {
        out_scale = sm[s]     & 0x3Fu;
        out_min   = sm[s + 4] & 0x3Fu;
    } else {
        const unsigned idx = s - 4u;
        out_scale = (sm[8u + idx] & 0x0Fu) | ((sm[    idx] >> 6) << 4);
        out_min   = (sm[8u + idx] >>   4)  | ((sm[4u + idx] >> 6) << 4);
    }
}

} // namespace

void dequant_q5k_block(const std::uint8_t* block, float* out) noexcept {
    const float d    = read_fp16_le(block + 0);
    const float dmin = read_fp16_le(block + 2);

    const std::uint8_t* sm = block + 4;    // 12 bytes of scales+mins
    const std::uint8_t* qh = block + 16;   // 32 bytes of high bits
    const std::uint8_t* ql = block + 48;   // 128 bytes of 4-bit values

    // Each iteration emits 64 outputs from one 32-byte qs run plus the
    // matching pair of high-bit masks.
    for (unsigned j = 0; j < 4; ++j) {
        unsigned scale_q0 = 0, min_q0 = 0, scale_q1 = 0, min_q1 = 0;
        unpack_scale_min_6bit(sm, 2 * j,     scale_q0, min_q0);
        unpack_scale_min_6bit(sm, 2 * j + 1, scale_q1, min_q1);

        const float d1 = d    * static_cast<float>(scale_q0);
        const float m1 = dmin * static_cast<float>(min_q0);
        const float d2 = d    * static_cast<float>(scale_q1);
        const float m2 = dmin * static_cast<float>(min_q1);

        const std::uint8_t u1 = static_cast<std::uint8_t>(1u << (2 * j));
        const std::uint8_t u2 = static_cast<std::uint8_t>(2u << (2 * j));

        const std::uint8_t* q = ql + j * 32u;
        float* out1 = out + j * 64;
        float* out2 = out + j * 64 + 32;
        for (unsigned l = 0; l < 32; ++l) {
            const unsigned lo = (q[l] & 0x0Fu) | ((qh[l] & u1) ? 16u : 0u);
            out1[l] = d1 * static_cast<float>(lo) - m1;
        }
        for (unsigned l = 0; l < 32; ++l) {
            const unsigned hi = (q[l] >> 4) | ((qh[l] & u2) ? 16u : 0u);
            out2[l] = d2 * static_cast<float>(hi) - m2;
        }
    }
}

void dequant_q5k(const std::uint8_t* blocks, std::size_t block_count,
                 float* out) noexcept {
    for (std::size_t b = 0; b < block_count; ++b) {
        dequant_q5k_block(blocks + b * 176u, out + b * 256u);
    }
}

} // namespace ultima::kernels
