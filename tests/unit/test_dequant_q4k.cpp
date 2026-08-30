#include <doctest/doctest.h>

#include "ultima/kernels/dequant_q4k.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

// Pack fp32 into fp16 bits (round-to-nearest-even is not strictly needed for
// our fixture values which we choose to be exactly representable in fp16).
std::uint16_t float_to_fp16(float f) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    std::int32_t exp = static_cast<std::int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
    std::uint32_t mant = bits & 0x7FFFFFu;

    if (exp <= 0) {
        // Subnormal or zero (fixture avoids this).
        return static_cast<std::uint16_t>(sign);
    }
    if (exp >= 0x1F) {
        // Inf / NaN (fixture avoids this).
        return static_cast<std::uint16_t>(sign | 0x7C00u);
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | (mant >> 13));
}

void write_fp16_le(std::uint8_t* dst, float f) {
    const std::uint16_t h = float_to_fp16(f);
    dst[0] = static_cast<std::uint8_t>(h        & 0xFFu);
    dst[1] = static_cast<std::uint8_t>((h >> 8) & 0xFFu);
}

// Build a Q4_K super-block from explicit parameters. All 8 scales set to
// `scale_q` (0..63), all 8 mins set to `min_q` (0..63). qs pattern: within
// each 32-element sub-block, values are 0,1,2,...,15,0,1,2,...,15.
std::array<std::uint8_t, 144> build_block(float d, float dmin,
                                          unsigned scale_q, unsigned min_q) {
    std::array<std::uint8_t, 144> b{};

    write_fp16_le(&b[0], d);
    write_fp16_le(&b[2], dmin);

    // scales[0..3] and mins[0..3] go in the low 6 bits of bytes 0..3 (scales)
    // and 4..7 (mins). scales[4..7] use bytes 8..11 low nibble + bytes 0..3
    // high 2 bits; mins[4..7] use bytes 8..11 high nibble + bytes 4..7 high 2 bits.
    for (unsigned s = 0; s < 8; ++s) {
        const unsigned sv = scale_q & 0x3Fu;
        const unsigned mv = min_q   & 0x3Fu;
        if (s < 4) {
            b[4 + s]     = static_cast<std::uint8_t>((b[4 + s]     & 0xC0u) | sv);   // preserve high 2
            b[4 + s + 4] = static_cast<std::uint8_t>((b[4 + s + 4] & 0xC0u) | mv);
            // Wait: our public offset naming has scales_and_mins starting at
            // block+4. So sm[i] == b[4+i]. Redo with that:
        }
    }
    // Rewrite the above with the correct offsets: sm is at b+4, so sm[s] = b[4+s].
    // Reset the scales area first.
    for (unsigned i = 0; i < 12; ++i) b[4u + i] = 0;

    for (unsigned s = 0; s < 8; ++s) {
        const unsigned sv = scale_q & 0x3Fu;
        const unsigned mv = min_q   & 0x3Fu;
        if (s < 4) {
            b[4u + s      ] = static_cast<std::uint8_t>(sv);          // scales[0..3]
            b[4u + s + 4  ] = static_cast<std::uint8_t>(mv);          // mins[0..3]
        } else {
            const unsigned idx = s - 4u;
            // Low 4 bits of scale in sm[8+idx] low nibble.
            // Low 4 bits of min   in sm[8+idx] high nibble.
            b[4u + 8u + idx] = static_cast<std::uint8_t>((sv & 0x0Fu) | ((mv & 0x0Fu) << 4));
            // High 2 bits of scale into top of sm[idx].
            b[4u + idx      ] = static_cast<std::uint8_t>(b[4u + idx      ] | ((sv >> 4) << 6));
            // High 2 bits of min   into top of sm[4+idx].
            b[4u + 4u + idx ] = static_cast<std::uint8_t>(b[4u + 4u + idx ] | ((mv >> 4) << 6));
        }
    }

    // qs[128] pattern: byte i (i in 0..127) = (2i % 16) low + (2i+1 % 16) high
    // Which within a 16-byte sub-block yields values 0,1,2,...,15,0,1,...,15.
    for (unsigned i = 0; i < 128; ++i) {
        const unsigned local_i = i % 16;
        const unsigned lo = (2 * local_i    ) & 0x0Fu;
        const unsigned hi = (2 * local_i + 1) & 0x0Fu;
        b[16u + i] = static_cast<std::uint8_t>(lo | (hi << 4));
    }

    return b;
}

} // namespace

TEST_CASE("dequant_q4k: identity-ish (scale=1, min=0, d=1, dmin=0) yields the raw 4-bit values") {
    auto block = build_block(/*d=*/1.0f, /*dmin=*/0.0f,
                             /*scale_q=*/1u, /*min_q=*/0u);
    float out[256]{};
    ultima::kernels::dequant_q4k_block(block.data(), out);

    // Each sub-block of 32 elements should be 0,1,2,...,15,0,1,2,...,15.
    for (unsigned s = 0; s < 8; ++s) {
        for (unsigned e = 0; e < 32; ++e) {
            const float expected = static_cast<float>(e % 16u);
            CHECK(out[s * 32 + e] == doctest::Approx(expected).epsilon(1e-6));
        }
    }
}

TEST_CASE("dequant_q4k: uniform scale factor multiplies output linearly") {
    auto block = build_block(/*d=*/2.0f, /*dmin=*/0.0f,
                             /*scale_q=*/3u, /*min_q=*/0u);
    float out[256]{};
    ultima::kernels::dequant_q4k_block(block.data(), out);
    // real_scale = d * scale_q = 6. Expected values: 0, 6, 12, ..., 90.
    for (unsigned s = 0; s < 8; ++s) {
        for (unsigned e = 0; e < 32; ++e) {
            const float expected = 6.0f * static_cast<float>(e % 16u);
            CHECK(out[s * 32 + e] == doctest::Approx(expected).epsilon(1e-4));
        }
    }
}

TEST_CASE("dequant_q4k: min term subtracts uniformly") {
    // d = 1, scale_q = 1, dmin = 0.5, min_q = 4  =>  real_min = 2.0
    // Output values = 1 * 1 * raw - 2.0 = raw - 2.
    auto block = build_block(/*d=*/1.0f, /*dmin=*/0.5f,
                             /*scale_q=*/1u, /*min_q=*/4u);
    float out[256]{};
    ultima::kernels::dequant_q4k_block(block.data(), out);
    for (unsigned s = 0; s < 8; ++s) {
        for (unsigned e = 0; e < 32; ++e) {
            const float raw = static_cast<float>(e % 16u);
            const float expected = raw - 2.0f;
            CHECK(out[s * 32 + e] == doctest::Approx(expected).epsilon(1e-4));
        }
    }
}

TEST_CASE("dequant_q4k: multiple blocks dequant independently") {
    auto b1 = build_block(1.0f, 0.0f, 1u, 0u);
    auto b2 = build_block(2.0f, 0.0f, 3u, 0u);
    std::array<std::uint8_t, 288> two_blocks{};
    std::memcpy(two_blocks.data(),         b1.data(), 144);
    std::memcpy(two_blocks.data() + 144u,  b2.data(), 144);

    float out[512]{};
    ultima::kernels::dequant_q4k(two_blocks.data(), 2, out);

    // Block 0: raw 4-bit values pattern
    CHECK(out[0]   == doctest::Approx(0.0f));
    CHECK(out[15]  == doctest::Approx(15.0f));
    // Block 1: 6 * raw
    CHECK(out[256] == doctest::Approx(0.0f));
    CHECK(out[271] == doctest::Approx(90.0f));
}
