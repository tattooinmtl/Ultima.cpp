#include <doctest/doctest.h>

#include "ultima/kernels/dequant_q6k.hpp"
#include "ultima/kernels/matvec.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

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

// Build a Q6_K super-block where every element has the same signed 6-bit
// value (q6_signed) and all 16 sub-block scales are the same int8.
std::array<std::uint8_t, 210> build_uniform_block(float d, std::int8_t scale,
                                                  int q6_signed) {
    std::array<std::uint8_t, 210> b{};

    // Every value has the same q6 encoding.
    const unsigned q6_u = static_cast<unsigned>(q6_signed + 32) & 0x3Fu;
    const unsigned low4 = q6_u & 0x0Fu;
    const unsigned high2 = (q6_u >> 4) & 0x03u;

    // ql: each byte has two identical low nibbles.
    const std::uint8_t ql_byte = static_cast<std::uint8_t>(low4 | (low4 << 4));
    for (unsigned i = 0; i < 128; ++i) b[i] = ql_byte;

    // qh: each byte has four identical 2-bit fields.
    const std::uint8_t qh_byte = static_cast<std::uint8_t>(
        (high2 << 0) | (high2 << 2) | (high2 << 4) | (high2 << 6));
    for (unsigned i = 0; i < 64; ++i) b[128 + i] = qh_byte;

    // scales: all identical int8.
    for (unsigned i = 0; i < 16; ++i) b[192 + i] = static_cast<std::uint8_t>(scale);

    // fp16 d at bytes 208..209.
    write_fp16_le(&b[208], d);
    return b;
}

} // namespace

TEST_CASE("dequant_q6k: uniform block dequants to expected value") {
    // q6_signed = 5, scale = 3, d = 2  =>  value = 2*3*5 = 30 everywhere.
    auto block = build_uniform_block(2.0f, 3, 5);
    float out[256]{};
    ultima::kernels::dequant_q6k_block(block.data(), out);
    for (unsigned i = 0; i < 256; ++i) {
        CHECK(out[i] == doctest::Approx(30.0f).epsilon(1e-4));
    }
}

TEST_CASE("dequant_q6k: negative signed value works") {
    // q6_signed = -32, scale = 1, d = 1  =>  value = -32.
    auto block = build_uniform_block(1.0f, 1, -32);
    float out[256]{};
    ultima::kernels::dequant_q6k_block(block.data(), out);
    for (unsigned i = 0; i < 256; ++i) {
        CHECK(out[i] == doctest::Approx(-32.0f));
    }
}

TEST_CASE("dequant_q6k: zero centered (q6_signed = 0) yields zeros") {
    auto block = build_uniform_block(1.0f, 7, 0);
    float out[256]{};
    ultima::kernels::dequant_q6k_block(block.data(), out);
    for (unsigned i = 0; i < 256; ++i) {
        CHECK(out[i] == doctest::Approx(0.0f));
    }
}

TEST_CASE("matvec_q6k_f32: uniform-block dot ones") {
    // Value = 30 everywhere. Dot with 256 ones -> 30 * 256 = 7680.
    auto row = build_uniform_block(2.0f, 3, 5);
    std::vector<float> x(256, 1.0f);
    std::vector<float> y(1, 0.0f);
    ultima::kernels::matvec_q6k_f32(row.data(), x.data(), y.data(), 1, 256);
    CHECK(y[0] == doctest::Approx(7680.0f).epsilon(1e-3));
}
