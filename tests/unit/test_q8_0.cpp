#include <doctest/doctest.h>

#include "ultima/kernels/dequant_q8_0.hpp"
#include "ultima/kernels/matvec.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

// fp16 write (matches encoder used in the Q4_K tests).
std::uint16_t float_to_fp16(float f) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    std::int32_t exp = static_cast<std::int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
    std::uint32_t mant = bits & 0x7FFFFFu;
    if (exp <= 0)   return static_cast<std::uint16_t>(sign);
    if (exp >= 0x1F) return static_cast<std::uint16_t>(sign | 0x7C00u);
    return static_cast<std::uint16_t>(sign
        | (static_cast<std::uint32_t>(exp) << 10) | (mant >> 13));
}
void write_fp16_le(std::uint8_t* dst, float f) {
    const std::uint16_t h = float_to_fp16(f);
    dst[0] = static_cast<std::uint8_t>(h        & 0xFFu);
    dst[1] = static_cast<std::uint8_t>((h >> 8) & 0xFFu);
}

// Build a Q8_0 block from a scale and 32 explicit int8 values.
std::array<std::uint8_t, 34> build_block(float d, const std::array<std::int8_t, 32>& qs) {
    std::array<std::uint8_t, 34> b{};
    write_fp16_le(&b[0], d);
    std::memcpy(&b[2], qs.data(), 32);
    return b;
}

} // namespace

TEST_CASE("dequant_q8_0: d=1 gives raw int8 values as floats") {
    std::array<std::int8_t, 32> qs{};
    for (int i = 0; i < 32; ++i) qs[i] = static_cast<std::int8_t>(i - 16);   // -16..15
    auto block = build_block(1.0f, qs);

    float out[32]{};
    ultima::kernels::dequant_q8_0_block(block.data(), out);
    for (int i = 0; i < 32; ++i) {
        CHECK(out[i] == doctest::Approx(static_cast<float>(i - 16)));
    }
}

TEST_CASE("dequant_q8_0: scale multiplies linearly") {
    std::array<std::int8_t, 32> qs{};
    for (int i = 0; i < 32; ++i) qs[i] = static_cast<std::int8_t>(i);
    auto block = build_block(0.5f, qs);

    float out[32]{};
    ultima::kernels::dequant_q8_0_block(block.data(), out);
    for (int i = 0; i < 32; ++i) {
        CHECK(out[i] == doctest::Approx(0.5f * static_cast<float>(i)).epsilon(1e-5));
    }
}

TEST_CASE("dequant_q8_0: multi-block") {
    std::array<std::int8_t, 32> qs1{}, qs2{};
    for (int i = 0; i < 32; ++i) { qs1[i] = 1; qs2[i] = 2; }
    auto b1 = build_block(1.0f, qs1);
    auto b2 = build_block(3.0f, qs2);
    std::array<std::uint8_t, 68> two{};
    std::memcpy(two.data(),      b1.data(), 34);
    std::memcpy(two.data() + 34, b2.data(), 34);

    float out[64]{};
    ultima::kernels::dequant_q8_0(two.data(), 2, out);
    CHECK(out[0]  == doctest::Approx(1.0f));   // 1 * 1
    CHECK(out[31] == doctest::Approx(1.0f));
    CHECK(out[32] == doctest::Approx(6.0f));   // 3 * 2
    CHECK(out[63] == doctest::Approx(6.0f));
}

TEST_CASE("matvec_q8_0_f32: dot product against ones") {
    // d=1, qs = 1,1,...,1. sum = 32. K=32 -> y = 32.
    std::array<std::int8_t, 32> qs{};
    for (auto& v : qs) v = 1;
    auto row = build_block(1.0f, qs);

    std::vector<float> x(32, 1.0f);
    std::vector<float> y(1, 0.0f);
    ultima::kernels::matvec_q8_0_f32(row.data(), x.data(), y.data(), 1, 32);
    CHECK(y[0] == doctest::Approx(32.0f));
}

TEST_CASE("dequant_q8_0: subnormal fp16 scale round-trips at correct magnitude") {
    // Regression: earlier the subnormal exponent formula was off by one and
    // returned 2x the correct magnitude. Build a block whose fp16 d is a
    // subnormal and confirm dequant matches the exact bit-pattern value.
    //
    // fp16 subnormal with raw bits (sign=0, exp=0, mant=1) = 2^-24.
    // With qs[i] = 1, dequant should yield exactly 2^-24.
    std::array<std::uint8_t, 34> block{};
    block[0] = 0x01u;   // low byte: mant=1, exp=0, sign=0
    block[1] = 0x00u;   // high byte
    std::array<std::int8_t, 32> qs{};
    for (auto& v : qs) v = 1;
    std::memcpy(&block[2], qs.data(), 32);

    float out[32]{};
    ultima::kernels::dequant_q8_0_block(block.data(), out);
    const float expected = std::ldexp(1.0f, -24);   // 2^-24
    for (int i = 0; i < 32; ++i) {
        CHECK(out[i] == doctest::Approx(expected));
    }

    // Mid-range subnormal: raw bits 0x00FF => mant=255, exp=0. Value = 255*2^-24.
    // With the pre-fix formula this returned 510*2^-24 (2x). Confirms the
    // fix works across the whole subnormal range, not just the smallest value.
    block[0] = 0xFFu;
    block[1] = 0x00u;
    ultima::kernels::dequant_q8_0_block(block.data(), out);
    const float expected_255 = std::ldexp(255.0f, -24);
    for (int i = 0; i < 32; ++i) {
        CHECK(out[i] == doctest::Approx(expected_255).epsilon(1e-6));
    }
}

TEST_CASE("matvec_q8_0_f32: two rows, two blocks per row") {
    std::array<std::int8_t, 32> ones{};
    for (auto& v : ones) v = 1;
    auto b_ones = build_block(1.0f, ones);
    auto b_twos = build_block(2.0f, ones);   // real value 2

    std::vector<std::uint8_t> W(2 * 68, 0);
    // Row 0: two blocks of ones
    std::memcpy(W.data() + 0,  b_ones.data(), 34);
    std::memcpy(W.data() + 34, b_ones.data(), 34);
    // Row 1: two blocks of twos
    std::memcpy(W.data() + 68,      b_twos.data(), 34);
    std::memcpy(W.data() + 68 + 34, b_twos.data(), 34);

    std::vector<float> x(64, 1.0f);
    std::vector<float> y(2, 0.0f);
    ultima::kernels::matvec_q8_0_f32(W.data(), x.data(), y.data(), 2, 64);
    CHECK(y[0] == doctest::Approx(64.0f));    // 2 blocks * 32 * 1
    CHECK(y[1] == doctest::Approx(128.0f));   // 2 blocks * 32 * 2
}
