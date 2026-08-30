#include <doctest/doctest.h>

#include "ultima/kernels/matvec.hpp"
#include "ultima/runtime/thread_pool.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

// Decision 05 §5.6: every SIMD kernel must round-trip against its scalar
// oracle on fixed-seed random inputs. F32 kernels: max abs err < 1e-4.
// Threaded variants must match the sequential SIMD variant exactly.

namespace {

constexpr std::uint32_t kSeed = 0xA5A5F00Du;

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    REQUIRE(a.size() == b.size());
    float m = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float d = std::abs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

// Reduction-order rounding: scalar sums element-by-element, SIMD sums in
// 8 parallel lanes then horizontally. For random inputs whose partial sums
// can reach O(1e5)+, an absolute tolerance is meaningless — a bug shows up
// as a large *relative* deviation.
//
// Per-element scaling (not per-vector max): dividing every deviation by
// the single largest ref value lets a big-magnitude row mask a small-
// magnitude row that's numerically wrong. Comparing each element against
// its own scale (floored at 1.0 so a near-zero ref doesn't blow up the
// ratio) catches per-row regressions.
float max_rel_diff(const std::vector<float>& ref, const std::vector<float>& got) {
    REQUIRE(ref.size() == got.size());
    float max_rel = 0.0f;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const float scale = std::max(std::abs(ref[i]), 1.0f);
        max_rel = std::max(max_rel, std::abs(ref[i] - got[i]) / scale);
    }
    return max_rel;
}

std::vector<float> random_f32(std::size_t n, std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

std::vector<std::uint8_t> random_bytes(std::size_t n, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<std::uint8_t> v(n);
    for (auto& b : v) b = static_cast<std::uint8_t>(dist(rng));
    return v;
}

// Q8_0: force per-block fp16 scale bytes to a well-behaved fp16 (avoid
// generating NaN/Inf from raw random bytes). Byte pattern: exponent field
// in the "normal" range so d is a modest finite float.
void sanitize_q8_0_scales(std::vector<std::uint8_t>& bytes,
                          std::size_t bytes_per_block = 34) {
    for (std::size_t off = 0; off + 2 <= bytes.size(); off += bytes_per_block) {
        // fp16 layout: sign(1)|exp(5)|mant(10). Clamp exp into [10, 18] so d
        // is finite and O(1)-ish regardless of RNG output.
        std::uint16_t h = static_cast<std::uint16_t>(bytes[off])
                        | (static_cast<std::uint16_t>(bytes[off + 1]) << 8);
        std::uint16_t exp5 = (h >> 10) & 0x1Fu;
        if (exp5 < 10) exp5 = 10;
        if (exp5 > 18) exp5 = 18;
        h = (h & 0x83FFu) | static_cast<std::uint16_t>(exp5 << 10);
        bytes[off + 0] = static_cast<std::uint8_t>(h & 0xFFu);
        bytes[off + 1] = static_cast<std::uint8_t>((h >> 8) & 0xFFu);
    }
}

// Q4_K: sanitize the two leading fp16 scales (d, dmin) per 144-byte block.
void sanitize_q4k_scales(std::vector<std::uint8_t>& bytes) {
    constexpr std::size_t B = 144;
    for (std::size_t off = 0; off + B <= bytes.size(); off += B) {
        for (std::size_t slot : {std::size_t{0}, std::size_t{2}}) {
            std::uint16_t h = static_cast<std::uint16_t>(bytes[off + slot])
                            | (static_cast<std::uint16_t>(bytes[off + slot + 1]) << 8);
            std::uint16_t exp5 = (h >> 10) & 0x1Fu;
            if (exp5 < 10) exp5 = 10;
            if (exp5 > 18) exp5 = 18;
            h = (h & 0x83FFu) | static_cast<std::uint16_t>(exp5 << 10);
            bytes[off + slot + 0] = static_cast<std::uint8_t>(h & 0xFFu);
            bytes[off + slot + 1] = static_cast<std::uint8_t>((h >> 8) & 0xFFu);
        }
    }
}

// Q6_K: sanitize fp16 d at bytes[208..209] of each 210-byte block.
void sanitize_q6k_scales(std::vector<std::uint8_t>& bytes) {
    constexpr std::size_t B = 210;
    for (std::size_t off = 0; off + B <= bytes.size(); off += B) {
        std::uint16_t h = static_cast<std::uint16_t>(bytes[off + 208])
                        | (static_cast<std::uint16_t>(bytes[off + 209]) << 8);
        std::uint16_t exp5 = (h >> 10) & 0x1Fu;
        if (exp5 < 10) exp5 = 10;
        if (exp5 > 18) exp5 = 18;
        h = (h & 0x83FFu) | static_cast<std::uint16_t>(exp5 << 10);
        bytes[off + 208] = static_cast<std::uint8_t>(h & 0xFFu);
        bytes[off + 209] = static_cast<std::uint8_t>((h >> 8) & 0xFFu);
    }
}

// Tolerance sized to the magnitudes we generate (K up to a few thousand,
// each product O(1)). 1e-3 absolute passes even with Q4_K/Q6_K where the
// SIMD path reorders sums into hsum-of-8 lanes.
constexpr float kTol = 1e-3f;

} // namespace

TEST_CASE("kernels correctness: matvec_f32_f32 AVX2 vs scalar") {
    std::mt19937 rng(kSeed);
    const std::size_t M = 37;
    const std::size_t K = 512 + 5;   // deliberately not a multiple of 8

    auto w = random_f32(M * K, rng, -1.0f, 1.0f);
    auto x = random_f32(K, rng, -1.0f, 1.0f);

    std::vector<float> y_scalar(M, 0.0f), y_simd(M, 0.0f);
    ultima::kernels::matvec_f32_f32_scalar(w.data(), x.data(), y_scalar.data(), M, K);
    ultima::kernels::matvec_f32_f32_avx2  (w.data(), x.data(), y_simd.data(),   M, K);

    CHECK(max_rel_diff(y_scalar, y_simd) < kTol);
}

TEST_CASE("kernels correctness: matvec_q8_0 AVX2 vs scalar") {
    std::mt19937 rng(kSeed + 1);
    const std::size_t M = 17;
    const std::size_t K = 32 * 8;                // 8 blocks/row
    const std::size_t bytes_per_row = (K / 32) * 34;

    auto w = random_bytes(M * bytes_per_row, rng);
    for (std::size_t row = 0; row < M; ++row) {
        std::vector<std::uint8_t> chunk(
            w.begin() + row * bytes_per_row,
            w.begin() + (row + 1) * bytes_per_row);
        sanitize_q8_0_scales(chunk);
        std::copy(chunk.begin(), chunk.end(), w.begin() + row * bytes_per_row);
    }
    auto x = random_f32(K, rng, -1.0f, 1.0f);

    std::vector<float> y_scalar(M, 0.0f), y_simd(M, 0.0f);
    ultima::kernels::matvec_q8_0_f32_scalar(w.data(), x.data(), y_scalar.data(), M, K);
    ultima::kernels::matvec_q8_0_f32       (w.data(), x.data(), y_simd.data(),   M, K);

    CHECK(max_rel_diff(y_scalar, y_simd) < kTol);
}

TEST_CASE("kernels correctness: matvec_q4k AVX2 vs scalar") {
    std::mt19937 rng(kSeed + 2);
    const std::size_t M = 13;
    const std::size_t K = 256 * 3;               // 3 super-blocks per row
    const std::size_t bytes_per_row = (K / 256) * 144;

    auto w = random_bytes(M * bytes_per_row, rng);
    sanitize_q4k_scales(w);
    auto x = random_f32(K, rng, -1.0f, 1.0f);

    std::vector<float> y_scalar(M, 0.0f), y_simd(M, 0.0f);
    ultima::kernels::matvec_q4k_f32_scalar(w.data(), x.data(), y_scalar.data(), M, K);
    ultima::kernels::matvec_q4k_f32       (w.data(), x.data(), y_simd.data(),   M, K);

    CHECK(max_rel_diff(y_scalar, y_simd) < kTol);
}

TEST_CASE("kernels correctness: matvec_q6k AVX2 vs scalar") {
    std::mt19937 rng(kSeed + 3);
    const std::size_t M = 11;
    const std::size_t K = 256 * 2;               // 2 super-blocks per row
    const std::size_t bytes_per_row = (K / 256) * 210;

    auto w = random_bytes(M * bytes_per_row, rng);
    sanitize_q6k_scales(w);
    auto x = random_f32(K, rng, -1.0f, 1.0f);

    std::vector<float> y_scalar(M, 0.0f), y_simd(M, 0.0f);
    ultima::kernels::matvec_q6k_f32_scalar(w.data(), x.data(), y_scalar.data(), M, K);
    ultima::kernels::matvec_q6k_f32       (w.data(), x.data(), y_simd.data(),   M, K);

    CHECK(max_rel_diff(y_scalar, y_simd) < kTol);
}

TEST_CASE("kernels correctness: matvec_f32_f32_threaded matches sequential") {
    std::mt19937 rng(kSeed + 10);
    const std::size_t M = 128;                    // enough to actually split
    const std::size_t K = 512;
    ultima::runtime::ThreadPool pool(4);

    auto w = random_f32(M * K, rng, -1.0f, 1.0f);
    auto x = random_f32(K, rng, -1.0f, 1.0f);

    std::vector<float> y_seq(M, 0.0f), y_par(M, 0.0f);
    ultima::kernels::matvec_f32_f32          (w.data(), x.data(), y_seq.data(), M, K);
    ultima::kernels::matvec_f32_f32_threaded (pool, w.data(), x.data(), y_par.data(), M, K);

    CHECK(max_abs_diff(y_seq, y_par) == doctest::Approx(0.0f));
}

TEST_CASE("kernels correctness: matvec_q4k_f32_threaded matches sequential") {
    std::mt19937 rng(kSeed + 11);
    const std::size_t M = 96;
    const std::size_t K = 256 * 2;
    const std::size_t bytes_per_row = (K / 256) * 144;
    ultima::runtime::ThreadPool pool(4);

    auto w = random_bytes(M * bytes_per_row, rng);
    sanitize_q4k_scales(w);
    auto x = random_f32(K, rng, -1.0f, 1.0f);

    std::vector<float> y_seq(M, 0.0f), y_par(M, 0.0f);
    ultima::kernels::matvec_q4k_f32          (w.data(), x.data(), y_seq.data(), M, K);
    ultima::kernels::matvec_q4k_f32_threaded (pool, w.data(), x.data(), y_par.data(), M, K);

    CHECK(max_abs_diff(y_seq, y_par) == doctest::Approx(0.0f));
}

TEST_CASE("kernels correctness: matvec_q6k_f32_threaded matches sequential") {
    std::mt19937 rng(kSeed + 12);
    const std::size_t M = 64;
    const std::size_t K = 256;
    const std::size_t bytes_per_row = (K / 256) * 210;
    ultima::runtime::ThreadPool pool(4);

    auto w = random_bytes(M * bytes_per_row, rng);
    sanitize_q6k_scales(w);
    auto x = random_f32(K, rng, -1.0f, 1.0f);

    std::vector<float> y_seq(M, 0.0f), y_par(M, 0.0f);
    ultima::kernels::matvec_q6k_f32          (w.data(), x.data(), y_seq.data(), M, K);
    ultima::kernels::matvec_q6k_f32_threaded (pool, w.data(), x.data(), y_par.data(), M, K);

    CHECK(max_abs_diff(y_seq, y_par) == doctest::Approx(0.0f));
}

TEST_CASE("kernels correctness: matvec_q8_0_f32_threaded matches sequential") {
    std::mt19937 rng(kSeed + 13);
    const std::size_t M = 80;
    const std::size_t K = 32 * 4;
    const std::size_t bytes_per_row = (K / 32) * 34;
    ultima::runtime::ThreadPool pool(4);

    auto w = random_bytes(M * bytes_per_row, rng);
    for (std::size_t row = 0; row < M; ++row) {
        std::vector<std::uint8_t> chunk(
            w.begin() + row * bytes_per_row,
            w.begin() + (row + 1) * bytes_per_row);
        sanitize_q8_0_scales(chunk);
        std::copy(chunk.begin(), chunk.end(), w.begin() + row * bytes_per_row);
    }
    auto x = random_f32(K, rng, -1.0f, 1.0f);

    std::vector<float> y_seq(M, 0.0f), y_par(M, 0.0f);
    ultima::kernels::matvec_q8_0_f32          (w.data(), x.data(), y_seq.data(), M, K);
    ultima::kernels::matvec_q8_0_f32_threaded (pool, w.data(), x.data(), y_par.data(), M, K);

    CHECK(max_abs_diff(y_seq, y_par) == doctest::Approx(0.0f));
}
