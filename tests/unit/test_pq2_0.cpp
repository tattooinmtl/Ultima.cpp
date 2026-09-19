#include <doctest/doctest.h>

#include "ultima/kernels/dequant_pq2_0.hpp"
#include "ultima/kernels/matvec.hpp"
#include "ultima/model/dtype.hpp"

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

using namespace ultima::kernels;

namespace {

constexpr std::size_t kBlock = 128;
constexpr std::size_t kBytes = 34;

std::uint16_t float_to_fp16(float f) {
    std::uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    std::int32_t  exp  = static_cast<std::int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    std::uint32_t mant = x & 0x007FFFFFu;
    if (exp <= 0)   return static_cast<std::uint16_t>(sign);
    if (exp >= 31)  return static_cast<std::uint16_t>(sign | 0x7C00u);
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10)
                                           | (mant >> 13));
}

// Build one PQ2_0 block from 128 codes in [0,4) and a scale.
std::vector<std::uint8_t> make_block(const std::vector<std::uint8_t>& codes, float d) {
    std::vector<std::uint8_t> blk(kBytes, 0);
    const std::uint16_t h = float_to_fp16(d);
    blk[0] = static_cast<std::uint8_t>(h & 0xFFu);
    blk[1] = static_cast<std::uint8_t>(h >> 8);
    for (std::size_t i = 0; i < kBlock; ++i) {
        blk[2 + i / 4] |= static_cast<std::uint8_t>((codes[i] & 0x03u) << (2 * (i % 4)));
    }
    return blk;
}

} // namespace

TEST_CASE("PQ2_0 block geometry matches the format") {
    using ultima::model::DataType;
    CHECK(ultima::model::block_size(DataType::PQ2_0)      == 128);
    CHECK(ultima::model::bytes_per_block(DataType::PQ2_0) == 34);
    CHECK(ultima::model::is_active(DataType::PQ2_0));
    // 34 bytes for 128 weights = 2.125 bits per weight.
    CHECK(34.0 * 8.0 / 128.0 == doctest::Approx(2.125));
}

TEST_CASE("PQ2_0 codes map to -1, 0, +1, +2") {
    std::vector<std::uint8_t> codes(kBlock);
    for (std::size_t i = 0; i < kBlock; ++i) codes[i] = static_cast<std::uint8_t>(i % 4);
    auto blk = make_block(codes, 2.0f);

    std::vector<float> out(kBlock);
    dequant_pq2_0_block(blk.data(), out.data());

    for (std::size_t i = 0; i < kBlock; ++i) {
        const float expect = (static_cast<int>(i % 4) - 1) * 2.0f;
        CHECK(out[i] == doctest::Approx(expect));
    }
    CHECK(out[0] == doctest::Approx(-2.0f));
    CHECK(out[1] == doctest::Approx( 0.0f));
    CHECK(out[2] == doctest::Approx( 2.0f));
    CHECK(out[3] == doctest::Approx( 4.0f));
}

TEST_CASE("PQ2_0 packs four codes per byte, low bits first") {
    // Only element 3 is nonzero: it lives in the top two bits of byte 0.
    std::vector<std::uint8_t> codes(kBlock, 1);   // 1 -> 0.0
    codes[3] = 2;                                 // 2 -> +d
    auto blk = make_block(codes, 1.0f);

    CHECK((blk[2] >> 6) == 2);       // top pair of the first code byte

    std::vector<float> out(kBlock);
    dequant_pq2_0_block(blk.data(), out.data());
    CHECK(out[0] == doctest::Approx(0.0f));
    CHECK(out[1] == doctest::Approx(0.0f));
    CHECK(out[2] == doctest::Approx(0.0f));
    CHECK(out[3] == doctest::Approx(1.0f));
    CHECK(out[4] == doctest::Approx(0.0f));
}

TEST_CASE("PQ2_0 applies one scale across the whole group of 128") {
    std::vector<std::uint8_t> codes(kBlock, 2);   // every weight is +1 * d
    auto blk = make_block(codes, 0.25f);

    std::vector<float> out(kBlock);
    dequant_pq2_0_block(blk.data(), out.data());
    for (float f : out) CHECK(f == doctest::Approx(0.25f));
}

TEST_CASE("PQ2_0 dequant handles several blocks back to back") {
    std::vector<std::uint8_t> bytes;
    std::vector<float> expect;
    for (int b = 0; b < 3; ++b) {
        std::vector<std::uint8_t> codes(kBlock);
        for (std::size_t i = 0; i < kBlock; ++i) {
            codes[i] = static_cast<std::uint8_t>((i + b) % 4);
        }
        const float d = 0.5f * static_cast<float>(b + 1);
        auto blk = make_block(codes, d);
        bytes.insert(bytes.end(), blk.begin(), blk.end());
        for (std::size_t i = 0; i < kBlock; ++i) {
            expect.push_back((static_cast<int>(codes[i]) - 1) * d);
        }
    }
    std::vector<float> out(3 * kBlock);
    dequant_pq2_0(bytes.data(), 3, out.data());
    for (std::size_t i = 0; i < out.size(); ++i) {
        CHECK(out[i] == doctest::Approx(expect[i]));
    }
}

TEST_CASE("PQ2_0 matvec matches dequantize-then-dot") {
    constexpr std::size_t M = 5, K = 256;   // two groups per row
    std::mt19937 rng{2026};
    std::uniform_int_distribution<int>   cd{0, 3};
    std::uniform_real_distribution<float> xd{-2.0f, 2.0f};

    std::vector<std::uint8_t> w;
    std::vector<float> dense(M * K);
    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t b = 0; b < K / kBlock; ++b) {
            std::vector<std::uint8_t> codes(kBlock);
            for (auto& c : codes) c = static_cast<std::uint8_t>(cd(rng));
            const float d = 0.1f + 0.05f * static_cast<float>(b + m);
            auto blk = make_block(codes, d);
            w.insert(w.end(), blk.begin(), blk.end());

            std::vector<float> ref(kBlock);
            dequant_pq2_0_block(blk.data(), ref.data());
            for (std::size_t i = 0; i < kBlock; ++i) {
                dense[m * K + b * kBlock + i] = ref[i];
            }
        }
    }

    std::vector<float> x(K);
    for (auto& v : x) v = xd(rng);

    std::vector<float> expect(M, 0.0f);
    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t k = 0; k < K; ++k) expect[m] += dense[m * K + k] * x[k];
    }

    std::vector<float> got(M, 0.0f);
    matvec_pq2_0_f32(w.data(), x.data(), got.data(), M, K);
    for (std::size_t m = 0; m < M; ++m) {
        CHECK(got[m] == doctest::Approx(expect[m]).epsilon(1e-5));
    }
}

TEST_CASE("PQ2_0 matvec refuses a K that is not a whole number of groups") {
    std::vector<std::uint8_t> w(kBytes, 0);
    std::vector<float> x(100, 1.0f);
    std::vector<float> y(2, 42.0f);
    matvec_pq2_0_f32(w.data(), x.data(), y.data(), 2, 100);
    // Zeroed rather than reading past the group boundary.
    CHECK(y[0] == 0.0f);
    CHECK(y[1] == 0.0f);
}

// ---- Q2_0: the same format at group 64 --------------------------------------

namespace {

std::vector<std::uint8_t> make_q2_0_block(const std::vector<std::uint8_t>& codes,
                                          float d) {
    std::vector<std::uint8_t> blk(18, 0);
    const std::uint16_t h = float_to_fp16(d);
    blk[0] = static_cast<std::uint8_t>(h & 0xFFu);
    blk[1] = static_cast<std::uint8_t>(h >> 8);
    for (std::size_t i = 0; i < 64; ++i) {
        blk[2 + i / 4] |= static_cast<std::uint8_t>((codes[i] & 0x03u) << (2 * (i % 4)));
    }
    return blk;
}

} // namespace

TEST_CASE("Q2_0 geometry is the 64-wide sibling of PQ2_0") {
    using ultima::model::DataType;
    CHECK(ultima::model::block_size(DataType::Q2_0)      == 64);
    CHECK(ultima::model::bytes_per_block(DataType::Q2_0) == 18);
    CHECK(ultima::model::is_active(DataType::Q2_0));
    CHECK(18.0 * 8.0 / 64.0 == doctest::Approx(2.25));
}

TEST_CASE("Q2_0 uses the same code mapping as PQ2_0") {
    std::vector<std::uint8_t> codes(64);
    for (std::size_t i = 0; i < 64; ++i) codes[i] = static_cast<std::uint8_t>(i % 4);
    auto blk = make_q2_0_block(codes, 0.5f);

    std::vector<float> out(64);
    dequant_q2_0_block(blk.data(), out.data());
    for (std::size_t i = 0; i < 64; ++i) {
        CHECK(out[i] == doctest::Approx((static_cast<int>(i % 4) - 1) * 0.5f));
    }
}

TEST_CASE("Q2_0 and PQ2_0 agree on identical leading data") {
    // The formats differ only in run length, so the first 64 values of a
    // PQ2_0 block and a Q2_0 block built from the same codes must match.
    std::vector<std::uint8_t> c128(128);
    for (std::size_t i = 0; i < 128; ++i) c128[i] = static_cast<std::uint8_t>((i * 7) % 4);
    std::vector<std::uint8_t> c64(c128.begin(), c128.begin() + 64);

    auto b128 = make_block(c128, 0.75f);
    auto b64  = make_q2_0_block(c64, 0.75f);

    std::vector<float> o128(128), o64(64);
    dequant_pq2_0_block(b128.data(), o128.data());
    dequant_q2_0_block(b64.data(), o64.data());
    for (std::size_t i = 0; i < 64; ++i) CHECK(o64[i] == doctest::Approx(o128[i]));
}

TEST_CASE("Q2_0 dequant walks consecutive blocks") {
    std::vector<std::uint8_t> bytes;
    std::vector<float> expect;
    for (int b = 0; b < 3; ++b) {
        std::vector<std::uint8_t> codes(64);
        for (std::size_t i = 0; i < 64; ++i) {
            codes[i] = static_cast<std::uint8_t>((i + b) % 4);
        }
        const float d = 0.25f * static_cast<float>(b + 1);
        auto blk = make_q2_0_block(codes, d);
        bytes.insert(bytes.end(), blk.begin(), blk.end());
        for (std::size_t i = 0; i < 64; ++i) {
            expect.push_back((static_cast<int>(codes[i]) - 1) * d);
        }
    }
    std::vector<float> out(3 * 64);
    dequant_q2_0(bytes.data(), 3, out.data());
    for (std::size_t i = 0; i < out.size(); ++i) {
        CHECK(out[i] == doctest::Approx(expect[i]));
    }
}

TEST_CASE("PQ2_0 vectorized path matches the scalar one") {
    // The earlier matvec test uses M=5, below the row count at which the
    // permuted AVX2 path engages, so it exercises only the scalar kernel.
    // This one is deliberately wide enough to take the other branch, and
    // wide enough in K to span several 128-groups.
    constexpr std::size_t M = 33, K = 512;
    std::mt19937 rng{31337};
    std::uniform_int_distribution<int>    cd{0, 3};
    std::uniform_real_distribution<float> xd{-2.0f, 2.0f};

    std::vector<std::uint8_t> w;
    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t b = 0; b < K / kBlock; ++b) {
            std::vector<std::uint8_t> codes(kBlock);
            for (auto& c : codes) c = static_cast<std::uint8_t>(cd(rng));
            auto blk = make_block(codes, 0.03f + 0.01f * static_cast<float>(b));
            w.insert(w.end(), blk.begin(), blk.end());
        }
    }
    std::vector<float> x(K);
    for (auto& v : x) v = xd(rng);

    std::vector<float> fast(M, 0.0f), slow(M, 0.0f);
    matvec_pq2_0_f32(w.data(), x.data(), fast.data(), M, K);
    matvec_pq2_0_f32_scalar(w.data(), x.data(), slow.data(), M, K);

    for (std::size_t m = 0; m < M; ++m) {
        CHECK(fast[m] == doctest::Approx(slow[m]).epsilon(1e-4));
    }
}

TEST_CASE("a non-finite block scale yields zeros, not NaN") {
    // Ternary-Bonsai-27B-dspark really does carry NaN scales in about 0.08%
    // of its embedding blocks, including the row for its own mask token.
    // A ternary block whose scale is not a number carries no information,
    // so zero is the only sane reading — and it keeps the damage inside the
    // block instead of turning a whole forward pass into NaN.
    std::vector<std::uint8_t> codes(kBlock, 2);      // all +1 * d
    auto blk = make_block(codes, 1.0f);
    blk[0] = 0x88; blk[1] = 0x7C;                    // raw fp16 0x7C88 = NaN

    std::vector<float> out(kBlock, 1.0f);
    dequant_pq2_0_block(blk.data(), out.data());
    for (float f : out) CHECK(f == 0.0f);

    // Infinity is treated the same way.
    blk[0] = 0x00; blk[1] = 0x7C;                    // +inf
    dequant_pq2_0_block(blk.data(), out.data());
    for (float f : out) CHECK(f == 0.0f);

    // A finite scale is of course untouched.
    auto good = make_block(codes, 0.5f);
    dequant_pq2_0_block(good.data(), out.data());
    for (float f : out) CHECK(f == doctest::Approx(0.5f));
}
