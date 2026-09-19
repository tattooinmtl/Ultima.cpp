#include <doctest/doctest.h>

#include "ultima/kernels/hadamard.hpp"
#include "ultima/model/hadamard.hpp"

#include <cmath>
#include <random>
#include <vector>

using namespace ultima::kernels;

namespace {

// The rotation as the reference materializes it: H[r][c] = +/-1/sqrt(n)
// by the parity of popcount(r & c).
std::vector<float> dense_hadamard(std::size_t n) {
    std::vector<float> h(n * n);
    const float scale = 1.0f / std::sqrt(static_cast<float>(n));
    for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t c = 0; c < n; ++c) {
            unsigned parity = 0;
            for (std::size_t bit = r & c; bit; bit >>= 1) parity ^= (bit & 1u);
            h[r * n + c] = parity ? -scale : scale;
        }
    }
    return h;
}

} // namespace

TEST_CASE("the butterfly computes the same thing as the dense rotation") {
    // This is the whole point of the kernel: O(n log n) instead of the
    // n x n matmul the reference builds. They must agree exactly.
    for (std::size_t n : {2u, 4u, 8u, 64u, 1024u}) {
        const auto H = dense_hadamard(n);
        std::mt19937 rng{static_cast<unsigned>(n)};
        std::uniform_real_distribution<float> d{-1.0f, 1.0f};

        std::vector<float> x(n);
        for (auto& v : x) v = d(rng);

        std::vector<float> expect(n, 0.0f);
        for (std::size_t r = 0; r < n; ++r) {
            for (std::size_t c = 0; c < n; ++c) expect[r] += H[r * n + c] * x[c];
        }

        std::vector<float> got = x;
        fwht_normalized_f32(got.data(), n);
        for (std::size_t i = 0; i < n; ++i) {
            REQUIRE(got[i] == doctest::Approx(expect[i]).epsilon(1e-4));
        }
    }
}

TEST_CASE("the normalized transform is its own inverse") {
    constexpr std::size_t n = 1024;
    std::mt19937 rng{7};
    std::uniform_real_distribution<float> d{-3.0f, 3.0f};
    std::vector<float> x(n);
    for (auto& v : x) v = d(rng);
    const std::vector<float> orig = x;

    fwht_normalized_f32(x.data(), n);
    fwht_normalized_f32(x.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        CHECK(x[i] == doctest::Approx(orig[i]).epsilon(1e-4));
    }
}

TEST_CASE("the transform preserves length") {
    // Orthogonality is what makes the fold lossless; if the scale were
    // wrong every activation would be uniformly mis-scaled and the model
    // would still produce fluent text.
    constexpr std::size_t n = 256;
    std::vector<float> x(n);
    std::mt19937 rng{11};
    std::uniform_real_distribution<float> d{-1.0f, 1.0f};
    for (auto& v : x) v = d(rng);

    double before = 0.0;
    for (float v : x) before += double(v) * v;
    fwht_normalized_f32(x.data(), n);
    double after = 0.0;
    for (float v : x) after += double(v) * v;

    CHECK(after == doctest::Approx(before).epsilon(1e-4));
}

TEST_CASE("a constant vector concentrates into the first coefficient") {
    constexpr std::size_t n = 64;
    std::vector<float> x(n, 1.0f);
    fwht_normalized_f32(x.data(), n);
    CHECK(x[0] == doctest::Approx(std::sqrt(float(n))));
    for (std::size_t i = 1; i < n; ++i) CHECK(x[i] == doctest::Approx(0.0f));
}

TEST_CASE("the fold runs blockwise, not across the whole width") {
    // Width 2048 with block 1024 must give the same answer as transforming
    // the two halves separately. Running one transform over the full width
    // would mix channels that the fold kept apart.
    constexpr std::size_t width = 2048, block = 1024;
    std::mt19937 rng{3};
    std::uniform_real_distribution<float> d{-1.0f, 1.0f};
    std::vector<float> x(width);
    for (auto& v : x) v = d(rng);

    std::vector<float> got(width);
    hadamard_fold_f32(x.data(), got.data(), width, block, nullptr);

    std::vector<float> lo(x.begin(), x.begin() + block);
    std::vector<float> hi(x.begin() + block, x.end());
    fwht_normalized_f32(lo.data(), block);
    fwht_normalized_f32(hi.data(), block);

    for (std::size_t i = 0; i < block; ++i) {
        CHECK(got[i]         == doctest::Approx(lo[i]));
        CHECK(got[block + i] == doctest::Approx(hi[i]));
    }
}

TEST_CASE("signs are applied before the rotation, and unfold reverses it") {
    constexpr std::size_t width = 1024, block = 1024;
    std::mt19937 rng{23};
    std::uniform_real_distribution<float> d{-1.0f, 1.0f};
    std::vector<float> x(width), signs(width);
    for (auto& v : x) v = d(rng);
    for (std::size_t i = 0; i < width; ++i) signs[i] = (i % 3 == 0) ? -1.0f : 1.0f;

    // fold = H(s * x)
    std::vector<float> folded(width);
    hadamard_fold_f32(x.data(), folded.data(), width, block, signs.data());

    std::vector<float> manual(width);
    for (std::size_t i = 0; i < width; ++i) manual[i] = x[i] * signs[i];
    fwht_normalized_f32(manual.data(), width);
    for (std::size_t i = 0; i < width; ++i) {
        CHECK(folded[i] == doctest::Approx(manual[i]).epsilon(1e-4));
    }

    // unfold = s * H(y), which inverts it
    std::vector<float> back(width);
    hadamard_unfold_f32(folded.data(), back.data(), width, block, signs.data());
    for (std::size_t i = 0; i < width; ++i) {
        CHECK(back[i] == doctest::Approx(x[i]).epsilon(1e-4));
    }
}

TEST_CASE("fold is in-place safe") {
    constexpr std::size_t width = 512, block = 512;
    std::vector<float> x(width), signs(width, -1.0f);
    for (std::size_t i = 0; i < width; ++i) x[i] = float(i) * 0.01f;

    std::vector<float> ref(width);
    hadamard_fold_f32(x.data(), ref.data(), width, block, signs.data());

    std::vector<float> inplace = x;
    hadamard_fold_f32(inplace.data(), inplace.data(), width, block, signs.data());
    for (std::size_t i = 0; i < width; ++i) {
        CHECK(inplace[i] == doctest::Approx(ref[i]));
    }
}

TEST_CASE("the gdn v permutation moves tiled head order to grouped") {
    // head_dim 2, n_k_heads 3, rep 2: tiled [hd, nk, rep] -> grouped
    // [hd, rep, nk]. Values encode (k, r) so a wrong mapping is visible.
    constexpr std::size_t hd = 2, nk = 3, rep = 2;
    std::vector<float> src(hd * nk * rep);
    for (std::size_t r = 0; r < rep; ++r) {
        for (std::size_t k = 0; k < nk; ++k) {
            for (std::size_t h = 0; h < hd; ++h) {
                src[h + hd * (k + nk * r)] = float(10 * k + r);
            }
        }
    }
    std::vector<float> dst(src.size(), -1.0f);
    ultima::model::hadamard_permute_gdn_v(src.data(), dst.data(), hd, nk, rep);

    for (std::size_t k = 0; k < nk; ++k) {
        for (std::size_t r = 0; r < rep; ++r) {
            for (std::size_t h = 0; h < hd; ++h) {
                CHECK(dst[h + hd * (r + rep * k)] == doctest::Approx(float(10 * k + r)));
            }
        }
    }
}
