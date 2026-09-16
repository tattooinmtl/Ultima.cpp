#include <doctest/doctest.h>

#include "ultima/kernels/dequant_q5k.hpp"
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

// Reference Q5_K encoder, written from the format description rather than
// from the decoder, so the round-trip is a real check and not a tautology.
//
//   scale[s], min[s] : 6-bit, s in [0, 8)
//   q[s][l]          : 5-bit value in [0, 32), l in [0, 32)
struct Q5kSpec {
    float d    = 1.0f;
    float dmin = 0.0f;
    std::array<unsigned, 8>                     scale{};
    std::array<unsigned, 8>                     min{};
    std::array<std::array<unsigned, 32>, 8>     q{};
};

std::array<std::uint8_t, 176> encode_q5k(const Q5kSpec& spec) {
    std::array<std::uint8_t, 176> b{};
    write_fp16_le(&b[0], spec.d);
    write_fp16_le(&b[2], spec.dmin);

    std::uint8_t* sm = &b[4];
    for (unsigned s = 0; s < 8; ++s) {
        const unsigned sc = spec.scale[s] & 0x3Fu;
        const unsigned mn = spec.min[s]   & 0x3Fu;
        if (s < 4) {
            sm[s]     = static_cast<std::uint8_t>(sm[s]     | sc);
            sm[s + 4] = static_cast<std::uint8_t>(sm[s + 4] | mn);
        } else {
            const unsigned idx = s - 4u;
            sm[8 + idx] = static_cast<std::uint8_t>((sc & 0x0Fu) | ((mn & 0x0Fu) << 4));
            sm[idx]     = static_cast<std::uint8_t>(sm[idx]     | ((sc >> 4) << 6));
            sm[4 + idx] = static_cast<std::uint8_t>(sm[4 + idx] | ((mn >> 4) << 6));
        }
    }

    std::uint8_t* qh = &b[16];
    std::uint8_t* ql = &b[48];
    for (unsigned s = 0; s < 8; ++s) {
        const unsigned j = s / 2u;
        for (unsigned l = 0; l < 32; ++l) {
            const unsigned v   = spec.q[s][l] & 0x1Fu;
            const unsigned low = v & 0x0Fu;
            if ((s % 2u) == 0) {
                ql[j * 32 + l] = static_cast<std::uint8_t>(ql[j * 32 + l] | low);
            } else {
                ql[j * 32 + l] = static_cast<std::uint8_t>(ql[j * 32 + l] | (low << 4));
            }
            // Bit s of qh[l] is the fifth bit of sub-block s, element l.
            if (v & 0x10u) qh[l] = static_cast<std::uint8_t>(qh[l] | (1u << s));
        }
    }
    return b;
}

float expected(const Q5kSpec& s, unsigned sub, unsigned l) {
    return s.d * static_cast<float>(s.scale[sub]) * static_cast<float>(s.q[sub][l])
         - s.dmin * static_cast<float>(s.min[sub]);
}

} // namespace

TEST_CASE("dequant_q5k: uniform block") {
    Q5kSpec spec;
    spec.d = 2.0f; spec.dmin = 0.5f;
    for (unsigned s = 0; s < 8; ++s) {
        spec.scale[s] = 3; spec.min[s] = 4;
        for (unsigned l = 0; l < 32; ++l) spec.q[s][l] = 5;
    }
    auto block = encode_q5k(spec);

    float out[256]{};
    ultima::kernels::dequant_q5k_block(block.data(), out);
    // 2*3*5 - 0.5*4 = 30 - 2 = 28
    for (unsigned i = 0; i < 256; ++i) {
        CHECK(out[i] == doctest::Approx(28.0f).epsilon(1e-4));
    }
}

TEST_CASE("dequant_q5k: fifth bit is honored") {
    // Values 16..31 only reachable via the qh plane. If qh were ignored the
    // decoded value would wrap to v-16 and this would fail.
    Q5kSpec spec;
    spec.d = 1.0f; spec.dmin = 0.0f;
    for (unsigned s = 0; s < 8; ++s) {
        spec.scale[s] = 1; spec.min[s] = 0;
        for (unsigned l = 0; l < 32; ++l) spec.q[s][l] = 31;
    }
    auto block = encode_q5k(spec);

    float out[256]{};
    ultima::kernels::dequant_q5k_block(block.data(), out);
    for (unsigned i = 0; i < 256; ++i) CHECK(out[i] == doctest::Approx(31.0f));
}

TEST_CASE("dequant_q5k: per-sub-block scales, mins and values round-trip") {
    // Every sub-block and every element distinct, so a transposed sub-block
    // order or a swapped low/high nibble shows up immediately.
    Q5kSpec spec;
    spec.d = 0.25f; spec.dmin = 0.125f;
    for (unsigned s = 0; s < 8; ++s) {
        spec.scale[s] = 7u + s * 5u;          // spans the 6-bit range
        spec.min[s]   = 61u - s * 7u;
        for (unsigned l = 0; l < 32; ++l) {
            spec.q[s][l] = (s * 7u + l * 3u) & 0x1Fu;
        }
    }
    auto block = encode_q5k(spec);

    float out[256]{};
    ultima::kernels::dequant_q5k_block(block.data(), out);
    for (unsigned s = 0; s < 8; ++s) {
        for (unsigned l = 0; l < 32; ++l) {
            CHECK(out[s * 32 + l] == doctest::Approx(expected(spec, s, l)).epsilon(1e-4));
        }
    }
}

TEST_CASE("matvec_q5k_f32: agrees with dequant-then-dot") {
    Q5kSpec spec;
    spec.d = 0.25f; spec.dmin = 0.125f;
    for (unsigned s = 0; s < 8; ++s) {
        spec.scale[s] = 7u + s * 5u;
        spec.min[s]   = 61u - s * 7u;
        for (unsigned l = 0; l < 32; ++l) spec.q[s][l] = (s * 7u + l * 3u) & 0x1Fu;
    }
    auto row = encode_q5k(spec);

    std::vector<float> x(256);
    for (unsigned i = 0; i < 256; ++i) x[i] = 0.01f * static_cast<float>(i % 17) - 0.05f;

    float deq[256]{};
    ultima::kernels::dequant_q5k_block(row.data(), deq);
    float ref = 0.0f;
    for (unsigned i = 0; i < 256; ++i) ref += deq[i] * x[i];

    std::vector<float> y(1, 0.0f);
    ultima::kernels::matvec_q5k_f32(row.data(), x.data(), y.data(), 1, 256);
    CHECK(y[0] == doctest::Approx(ref).epsilon(1e-4));

    std::vector<float> y_scalar(1, 0.0f);
    ultima::kernels::matvec_q5k_f32_scalar(row.data(), x.data(), y_scalar.data(), 1, 256);
    CHECK(y_scalar[0] == doctest::Approx(ref).epsilon(1e-4));
}

TEST_CASE("matvec_q5k_f32: multi-row, multi-block rows stay independent") {
    constexpr std::size_t K = 512;   // two super-blocks per row
    constexpr std::size_t M = 3;

    std::vector<std::uint8_t> w(M * 2 * 176);
    std::vector<float> ref_rows(M * K);

    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t blk = 0; blk < 2; ++blk) {
            Q5kSpec spec;
            spec.d = 0.5f + 0.1f * static_cast<float>(m);
            spec.dmin = 0.05f * static_cast<float>(blk + 1);
            for (unsigned s = 0; s < 8; ++s) {
                spec.scale[s] = (3u + s + static_cast<unsigned>(m)) & 0x3Fu;
                spec.min[s]   = (11u + s * 2u) & 0x3Fu;
                for (unsigned l = 0; l < 32; ++l) {
                    spec.q[s][l] = (s + l + static_cast<unsigned>(m) + blk) & 0x1Fu;
                }
            }
            auto enc = encode_q5k(spec);
            std::memcpy(&w[(m * 2 + blk) * 176], enc.data(), 176);
            ultima::kernels::dequant_q5k_block(enc.data(), &ref_rows[m * K + blk * 256]);
        }
    }

    std::vector<float> x(K);
    for (std::size_t i = 0; i < K; ++i) x[i] = 0.02f * static_cast<float>((i % 13)) - 0.1f;

    std::vector<float> y(M, 0.0f);
    ultima::kernels::matvec_q5k_f32(w.data(), x.data(), y.data(), M, K);

    for (std::size_t m = 0; m < M; ++m) {
        float ref = 0.0f;
        for (std::size_t i = 0; i < K; ++i) ref += ref_rows[m * K + i] * x[i];
        CHECK(y[m] == doctest::Approx(ref).epsilon(1e-4));
    }
}
