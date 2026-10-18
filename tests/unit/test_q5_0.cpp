#include <doctest/doctest.h>

#include "ultima/kernels/dequant_q5_0.hpp"
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

// Reference Q5_0 encoder, written from the format description rather than
// from the decoder.
//
//   byte  0..1  : fp16 d
//   byte  2..5  : uint32 qh, little-endian; bit i is the fifth bit of value i
//   byte  6..21 : qs[16]; value i uses the low nibble of qs[i] for i < 16
//                 and the high nibble of qs[i-16] for i >= 16
//
// Values are offset-binary: the decoded result is d * (q - 16), q in [0, 32).
std::array<std::uint8_t, 22> encode_q5_0(float d, const std::array<unsigned, 32>& q) {
    std::array<std::uint8_t, 22> b{};
    write_fp16_le(&b[0], d);

    std::uint32_t qh = 0;
    for (unsigned i = 0; i < 32; ++i) {
        const unsigned v    = q[i] & 0x1Fu;
        const unsigned low4 = v & 0x0Fu;
        const unsigned j    = i & 15u;
        if (i < 16) b[6 + j] = static_cast<std::uint8_t>(b[6 + j] | low4);
        else        b[6 + j] = static_cast<std::uint8_t>(b[6 + j] | (low4 << 4));
        if (v & 0x10u) qh |= (1u << i);
    }
    std::memcpy(&b[2], &qh, sizeof(qh));
    return b;
}

std::array<unsigned, 32> pattern(unsigned seed) {
    std::array<unsigned, 32> q{};
    for (unsigned i = 0; i < 32; ++i) q[i] = (i * 5u + seed * 3u) & 0x1Fu;
    return q;
}

} // namespace

TEST_CASE("dequant_q5_0: offset-binary centering") {
    // q == 16 is the zero point: d * (16 - 16) == 0 for any d.
    std::array<unsigned, 32> q{};
    q.fill(16);
    auto block = encode_q5_0(3.5f, q);

    float out[32]{};
    ultima::kernels::dequant_q5_0_block(block.data(), out);
    for (unsigned i = 0; i < 32; ++i) CHECK(out[i] == doctest::Approx(0.0f));
}

TEST_CASE("dequant_q5_0: extremes of the 5-bit range") {
    std::array<unsigned, 32> lo{}, hi{};
    lo.fill(0);
    hi.fill(31);

    float out[32]{};
    auto b_lo = encode_q5_0(1.0f, lo);
    ultima::kernels::dequant_q5_0_block(b_lo.data(), out);
    for (unsigned i = 0; i < 32; ++i) CHECK(out[i] == doctest::Approx(-16.0f));

    auto b_hi = encode_q5_0(1.0f, hi);
    ultima::kernels::dequant_q5_0_block(b_hi.data(), out);
    for (unsigned i = 0; i < 32; ++i) CHECK(out[i] == doctest::Approx(15.0f));
}

TEST_CASE("dequant_q5_0: fifth bit is honored per element") {
    // Alternate between 15 (fifth bit clear) and 16 (fifth bit set). If the
    // qh plane were ignored both would decode to the same value.
    std::array<unsigned, 32> q{};
    for (unsigned i = 0; i < 32; ++i) q[i] = (i % 2 == 0) ? 15u : 16u;
    auto block = encode_q5_0(1.0f, q);

    float out[32]{};
    ultima::kernels::dequant_q5_0_block(block.data(), out);
    for (unsigned i = 0; i < 32; ++i) {
        CHECK(out[i] == doctest::Approx((i % 2 == 0) ? -1.0f : 0.0f));
    }
}

TEST_CASE("dequant_q5_0: nibble halves map to the right elements") {
    // Distinct values everywhere: swapping the low/high nibble halves would
    // permute elements 0..15 against 16..31.
    auto q = pattern(1);
    auto block = encode_q5_0(0.5f, q);

    float out[32]{};
    ultima::kernels::dequant_q5_0_block(block.data(), out);
    for (unsigned i = 0; i < 32; ++i) {
        const float want = 0.5f * (static_cast<float>(q[i]) - 16.0f);
        CHECK(out[i] == doctest::Approx(want).epsilon(1e-5));
    }
}

TEST_CASE("matvec_q5_0_f32: AVX2 path agrees with the scalar oracle") {
    constexpr std::size_t K = 256;   // eight blocks per row
    constexpr std::size_t M = 5;

    std::vector<std::uint8_t> w(M * (K / 32) * 22);
    std::vector<float> ref_rows(M * K);

    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t blk = 0; blk < K / 32; ++blk) {
            auto q = pattern(static_cast<unsigned>(m * 8 + blk));
            const float d = 0.125f + 0.05f * static_cast<float>(m) - 0.01f * static_cast<float>(blk);
            auto enc = encode_q5_0(d, q);
            std::memcpy(&w[(m * (K / 32) + blk) * 22], enc.data(), 22);
            ultima::kernels::dequant_q5_0_block(enc.data(), &ref_rows[m * K + blk * 32]);
        }
    }

    std::vector<float> x(K);
    for (std::size_t i = 0; i < K; ++i) x[i] = 0.03f * static_cast<float>(i % 11) - 0.15f;

    std::vector<float> y(M, 0.0f), y_scalar(M, 0.0f);
    ultima::kernels::matvec_q5_0_f32(w.data(), x.data(), y.data(), M, K);
    ultima::kernels::matvec_q5_0_f32_scalar(w.data(), x.data(), y_scalar.data(), M, K);

    for (std::size_t m = 0; m < M; ++m) {
        float ref = 0.0f;
        for (std::size_t i = 0; i < K; ++i) ref += ref_rows[m * K + i] * x[i];
        CHECK(y[m]        == doctest::Approx(ref).epsilon(1e-4));
        CHECK(y_scalar[m] == doctest::Approx(ref).epsilon(1e-4));
    }
}

TEST_CASE("matvec_q5_0_f32: negative and mixed-sign weights") {
    // A row that is entirely below the zero point, dotted with ones, must
    // come out negative — catches a lost offset-binary bias.
    std::array<unsigned, 32> q{};
    q.fill(0);                       // every value decodes to -16 * d
    auto row = encode_q5_0(2.0f, q);

    std::vector<float> x(32, 1.0f);
    std::vector<float> y(1, 0.0f);
    ultima::kernels::matvec_q5_0_f32(row.data(), x.data(), y.data(), 1, 32);
    CHECK(y[0] == doctest::Approx(-16.0f * 2.0f * 32.0f).epsilon(1e-4));
}

TEST_CASE("matvec_q5_0_f32: K not a multiple of 32 yields zeros, not garbage") {
    std::vector<std::uint8_t> w(64, 0);
    std::vector<float> x(33, 1.0f);
    std::vector<float> y(2, 1234.0f);
    ultima::kernels::matvec_q5_0_f32(w.data(), x.data(), y.data(), 2, 33);
    CHECK(y[0] == 0.0f);
    CHECK(y[1] == 0.0f);
}
