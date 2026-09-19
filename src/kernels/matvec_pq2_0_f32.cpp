#include "ultima/kernels/matvec.hpp"

#include "ultima/kernels/dequant_pq2_0.hpp"

#include <cstring>

namespace ultima::kernels {

namespace {

constexpr std::size_t k_block = 128;
constexpr std::size_t k_bytes = 34;

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

// No hand-vectorized variant yet, unlike the other quant matvecs.
//
// PQ2_0 packs four codes per byte, so the weights a SIMD lane would want
// sit at stride 4 in the unpacked order. Reaching them needs either a
// gather per eight values or a four-way interleave of the unpacked
// vectors, and at this block size both plausibly cost more than the
// straight-line loop below, which is a handful of shifts and an FMA per
// weight with one scale multiply per 128. Measure before assuming a
// vectorized version would be faster.
//
// The compiler is given the easy version of the problem: the accumulation
// is over integer codes, with the fp16 scale applied once per block.
void matvec_pq2_0_f32_scalar(const std::uint8_t* w, const float* x, float* y,
                             std::size_t M, std::size_t K) noexcept {
    if (K == 0 || (K % k_block) != 0) {
        for (std::size_t m = 0; m < M; ++m) y[m] = 0.0f;
        return;
    }
    const std::size_t blocks_per_row = K / k_block;
    const std::size_t bytes_per_row  = blocks_per_row * k_bytes;

    for (std::size_t m = 0; m < M; ++m) {
        const std::uint8_t* row = w + m * bytes_per_row;
        float acc = 0.0f;
        for (std::size_t b = 0; b < blocks_per_row; ++b) {
            const std::uint8_t* blk = row + b * k_bytes;
            const float         d   = read_fp16_le(blk);
            const std::uint8_t* qs  = blk + 2;
            const float*        xs  = x + b * k_block;

            float sub = 0.0f;
            for (std::size_t i = 0; i < k_block / 4; ++i) {
                const unsigned byte = qs[i];
                sub += (static_cast<int>( byte        & 0x03u) - 1) * xs[4 * i + 0];
                sub += (static_cast<int>((byte >> 2)  & 0x03u) - 1) * xs[4 * i + 1];
                sub += (static_cast<int>((byte >> 4)  & 0x03u) - 1) * xs[4 * i + 2];
                sub += (static_cast<int>((byte >> 6)  & 0x03u) - 1) * xs[4 * i + 3];
            }
            acc += d * sub;
        }
        y[m] = acc;
    }
}

void matvec_pq2_0_f32(const std::uint8_t* w, const float* x, float* y,
                      std::size_t M, std::size_t K) noexcept {
    matvec_pq2_0_f32_scalar(w, x, y, M, K);
}

} // namespace ultima::kernels
