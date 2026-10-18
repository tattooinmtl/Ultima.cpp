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
constexpr std::size_t k_block_q2_0 = 64;
constexpr std::size_t k_bytes_q2_0 = 18;

// Shared by both group sizes: the packing and the code mapping are
// identical, only the run length differs.
inline void dequant_2bit_group(const std::uint8_t* block_bytes, float* out,
                               std::size_t n) noexcept {
    float d = read_fp16_le(block_bytes);

    // A non-finite scale is a defect in the file, not a representable
    // weight: the codes are ternary, so the only meaning a NaN or infinite
    // scale can carry is "this block holds nothing". Treating it as zero
    // confines the damage to its own 64 or 128 weights instead of letting
    // one bad group turn an entire forward pass into NaN.
    //
    // This is not hypothetical. Ternary-Bonsai-27B-dspark's token embedding
    // carries NaN scales in roughly 0.08% of its blocks, concentrated in
    // the unused tail of the vocabulary — including the row for its own
    // mask token, which is exactly the row a draft block reads.
    if (!(d == d) || d > 3.0e38f || d < -3.0e38f) d = 0.0f;
    const std::uint8_t* qs = block_bytes + 2;

    // Four codes per byte, low pair first.
    for (std::size_t b = 0; b < n / 4; ++b) {
        const std::uint8_t byte = qs[b];
        out[4 * b + 0] = (static_cast<int>( byte        & 0x03) - 1) * d;
        out[4 * b + 1] = (static_cast<int>((byte >> 2)  & 0x03) - 1) * d;
        out[4 * b + 2] = (static_cast<int>((byte >> 4)  & 0x03) - 1) * d;
        out[4 * b + 3] = (static_cast<int>((byte >> 6)  & 0x03) - 1) * d;
    }
}

} // namespace

void dequant_pq2_0_block(const std::uint8_t* block_bytes, float* out) noexcept {
    dequant_2bit_group(block_bytes, out, k_block);
}

void dequant_pq2_0(const std::uint8_t* blocks, std::size_t block_count,
                   float* out) noexcept {
    for (std::size_t i = 0; i < block_count; ++i) {
        dequant_pq2_0_block(blocks + i * k_bytes, out + i * k_block);
    }
}

void dequant_q2_0_block(const std::uint8_t* block_bytes, float* out) noexcept {
    dequant_2bit_group(block_bytes, out, k_block_q2_0);
}

void dequant_q2_0(const std::uint8_t* blocks, std::size_t block_count,
                  float* out) noexcept {
    for (std::size_t i = 0; i < block_count; ++i) {
        dequant_q2_0_block(blocks + i * k_bytes_q2_0, out + i * k_block_q2_0);
    }
}

} // namespace ultima::kernels
