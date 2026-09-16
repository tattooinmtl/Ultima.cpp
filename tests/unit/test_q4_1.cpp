#include <doctest/doctest.h>

#include "ultima/kernels/dequant_q4_1.hpp"
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
    if (exp <= 0)    return static_cast<std::uint16_t>(sign);
    if (exp >= 0x1F) return static_cast<std::uint16_t>(sign | 0x7C00u);
    return static_cast<std::uint16_t>(sign
        | (static_cast<std::uint32_t>(exp) << 10) | (mant >> 13));
}
void write_fp16_le(std::uint8_t* dst, float f) {
    const std::uint16_t h = float_to_fp16(f);
    dst[0] = static_cast<std::uint8_t>(h        & 0xFFu);
    dst[1] = static_cast<std::uint8_t>((h >> 8) & 0xFFu);
}

// q[i] in [0, 16) for i in [0, 32).
std::array<std::uint8_t, 20> encode_q4_1(float d, float m,
                                         const std::array<unsigned, 32>& q) {
    std::array<std::uint8_t, 20> b{};
    write_fp16_le(&b[0], d);
    write_fp16_le(&b[2], m);
    for (unsigned j = 0; j < 16; ++j) {
        b[4 + j] = static_cast<std::uint8_t>((q[j] & 0x0Fu) | ((q[j + 16] & 0x0Fu) << 4));
    }
    return b;
}

} // namespace

TEST_CASE("dequant_q4_1: uniform block") {
    std::array<unsigned, 32> q{};
    q.fill(5);
    auto block = encode_q4_1(2.0f, -1.0f, q);

    float out[32]{};
    ultima::kernels::dequant_q4_1_block(block.data(), out);
    for (unsigned i = 0; i < 32; ++i) {
        CHECK(out[i] == doctest::Approx(9.0f));   // 5*2 + (-1)
    }
}

TEST_CASE("dequant_q4_1: low/high nibble ordering") {
    // Distinct values everywhere: a swapped nibble order would permute the
    // first and second half of the block.
    std::array<unsigned, 32> q{};
    for (unsigned i = 0; i < 32; ++i) q[i] = i & 0x0Fu;
    auto block = encode_q4_1(1.0f, 0.0f, q);

    float out[32]{};
    ultima::kernels::dequant_q4_1_block(block.data(), out);
    for (unsigned i = 0; i < 32; ++i) {
        CHECK(out[i] == doctest::Approx(static_cast<float>(i & 0x0Fu)));
    }
}

TEST_CASE("dequant_q4_1: min offset applies to every element") {
    std::array<unsigned, 32> q{};
    q.fill(0);
    auto block = encode_q4_1(3.0f, 0.5f, q);

    float out[32]{};
    ultima::kernels::dequant_q4_1_block(block.data(), out);
    for (unsigned i = 0; i < 32; ++i) CHECK(out[i] == doctest::Approx(0.5f));
}

TEST_CASE("matvec_q4_1_f32: agrees with dequant-then-dot across rows") {
    constexpr std::size_t K = 64;    // two blocks per row
    constexpr std::size_t M = 3;

    std::vector<std::uint8_t> w(M * 2 * 20);
    std::vector<float> ref_rows(M * K);

    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t blk = 0; blk < 2; ++blk) {
            std::array<unsigned, 32> q{};
            for (unsigned i = 0; i < 32; ++i) {
                q[i] = (i + static_cast<unsigned>(m) * 3u + static_cast<unsigned>(blk)) & 0x0Fu;
            }
            const float d = 0.5f + 0.25f * static_cast<float>(m);
            const float mn = -0.5f * static_cast<float>(blk);
            auto enc = encode_q4_1(d, mn, q);
            std::memcpy(&w[(m * 2 + blk) * 20], enc.data(), 20);
            ultima::kernels::dequant_q4_1_block(enc.data(), &ref_rows[m * K + blk * 32]);
        }
    }

    std::vector<float> x(K);
    for (std::size_t i = 0; i < K; ++i) x[i] = 0.1f * static_cast<float>(i % 7) - 0.3f;

    std::vector<float> y(M, 0.0f), y_scalar(M, 0.0f);
    ultima::kernels::matvec_q4_1_f32(w.data(), x.data(), y.data(), M, K);
    ultima::kernels::matvec_q4_1_f32_scalar(w.data(), x.data(), y_scalar.data(), M, K);

    for (std::size_t m = 0; m < M; ++m) {
        float ref = 0.0f;
        for (std::size_t i = 0; i < K; ++i) ref += ref_rows[m * K + i] * x[i];
        CHECK(y[m]        == doctest::Approx(ref).epsilon(1e-4));
        CHECK(y_scalar[m] == doctest::Approx(ref).epsilon(1e-4));
    }
}
