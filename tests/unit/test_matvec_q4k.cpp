#include <doctest/doctest.h>

#include "ultima/kernels/dequant_q4k.hpp"
#include "ultima/kernels/matvec.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

// Reuse the fixture builder shape from test_dequant_q4k, kept local here to
// avoid coupling test TUs.
std::uint16_t float_to_fp16(float f) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    std::int32_t exp = static_cast<std::int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
    std::uint32_t mant = bits & 0x7FFFFFu;
    if (exp <= 0) return static_cast<std::uint16_t>(sign);
    if (exp >= 0x1F) return static_cast<std::uint16_t>(sign | 0x7C00u);
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | (mant >> 13));
}
void write_fp16_le(std::uint8_t* dst, float f) {
    const std::uint16_t h = float_to_fp16(f);
    dst[0] = static_cast<std::uint8_t>(h        & 0xFFu);
    dst[1] = static_cast<std::uint8_t>((h >> 8) & 0xFFu);
}
std::array<std::uint8_t, 144> build_block(float d, float dmin,
                                          unsigned scale_q, unsigned min_q) {
    std::array<std::uint8_t, 144> b{};
    write_fp16_le(&b[0], d);
    write_fp16_le(&b[2], dmin);
    for (unsigned i = 0; i < 12; ++i) b[4u + i] = 0;
    for (unsigned s = 0; s < 8; ++s) {
        const unsigned sv = scale_q & 0x3Fu;
        const unsigned mv = min_q   & 0x3Fu;
        if (s < 4) {
            b[4u + s    ] = static_cast<std::uint8_t>(sv);
            b[4u + s + 4] = static_cast<std::uint8_t>(mv);
        } else {
            const unsigned idx = s - 4u;
            b[4u + 8u + idx] = static_cast<std::uint8_t>((sv & 0x0Fu) | ((mv & 0x0Fu) << 4));
            b[4u + idx     ] = static_cast<std::uint8_t>(b[4u + idx     ] | ((sv >> 4) << 6));
            b[4u + 4u + idx] = static_cast<std::uint8_t>(b[4u + 4u + idx] | ((mv >> 4) << 6));
        }
    }
    for (unsigned i = 0; i < 128; ++i) {
        const unsigned local_i = i % 16;
        const unsigned lo = (2 * local_i    ) & 0x0Fu;
        const unsigned hi = (2 * local_i + 1) & 0x0Fu;
        b[16u + i] = static_cast<std::uint8_t>(lo | (hi << 4));
    }
    return b;
}

} // namespace

TEST_CASE("matvec_q4k_f32: dot product matches manual reference") {
    // Build one 256-wide row of Q4_K. d=1, scale_q=1, dmin=0 so dequant
    // gives the raw 4-bit values. Each sub-block holds 32 values arranged
    // as the pattern (0..15) repeated twice — sum per sub-block = 240.
    // 8 sub-blocks * 240 = 1920 for the full 256-wide row dotted with ones.
    auto row = build_block(1.0f, 0.0f, 1u, 0u);
    std::vector<float> x(256, 1.0f);
    std::vector<float> y(1, 0.0f);
    ultima::kernels::matvec_q4k_f32(row.data(), x.data(), y.data(),
                                    /*M=*/1, /*K=*/256);
    CHECK(y[0] == doctest::Approx(1920.0f).epsilon(1e-4));
}

TEST_CASE("matvec_q4k_f32: two rows, two blocks per row") {
    // Row 0: identity (raw values). Row 1: 2x scale factor.
    auto r0b0 = build_block(1.0f, 0.0f, 1u, 0u);
    auto r0b1 = build_block(1.0f, 0.0f, 1u, 0u);
    auto r1b0 = build_block(2.0f, 0.0f, 1u, 0u);   // real_scale = 2 * 1 = 2
    auto r1b1 = build_block(2.0f, 0.0f, 1u, 0u);

    // Layout: row-major. M=2, K=512, bytes_per_row = 288.
    std::vector<std::uint8_t> W(2 * 288, 0);
    std::memcpy(W.data() +   0,       r0b0.data(), 144);
    std::memcpy(W.data() + 144,       r0b1.data(), 144);
    std::memcpy(W.data() + 288,       r1b0.data(), 144);
    std::memcpy(W.data() + 288 + 144, r1b1.data(), 144);

    std::vector<float> x(512, 1.0f);
    std::vector<float> y(2, 0.0f);
    ultima::kernels::matvec_q4k_f32(W.data(), x.data(), y.data(),
                                    /*M=*/2, /*K=*/512);

    // Row 0 dot ones: 2 blocks * 1920 (per block) = 3840.
    // Row 1 dot ones: 2x scale -> 2 blocks * 3840 = 7680.
    CHECK(y[0] == doctest::Approx(3840.0f).epsilon(1e-4));
    CHECK(y[1] == doctest::Approx(7680.0f).epsilon(1e-4));
}

TEST_CASE("matvec_q4k_f32: matches direct dequant + f32 matvec") {
    // Cross-check: y = matvec_q4k(w, x)  should equal
    //              y' = matvec_f32_f32(dequant(w), x)
    auto row = build_block(1.5f, 0.25f, 5u, 3u);

    std::vector<float> x(256);
    for (std::size_t i = 0; i < 256; ++i) {
        x[i] = 0.01f * static_cast<float>(i) - 1.28f;   // some varied pattern
    }

    // Method A: fused
    std::vector<float> y_fused(1, 0.0f);
    ultima::kernels::matvec_q4k_f32(row.data(), x.data(), y_fused.data(), 1, 256);

    // Method B: full dequant then f32 matvec (using scalar oracle)
    std::array<float, 256> deq{};
    ultima::kernels::dequant_q4k_block(row.data(), deq.data());
    std::vector<float> y_ref(1, 0.0f);
    ultima::kernels::matvec_f32_f32_scalar(deq.data(), x.data(), y_ref.data(), 1, 256);

    CHECK(y_fused[0] == doctest::Approx(y_ref[0]).epsilon(1e-4));
}
