#include <doctest/doctest.h>

#include "ultima/kernels/rope.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace {
constexpr float kTol = 1e-5f;
}

TEST_CASE("rope_f32: position 0 is identity") {
    // At position 0, angle = 0, cos = 1, sin = 0, no change to x.
    std::vector<float> x{1, 2, 3, 4, 5, 6, 7, 8};
    ultima::kernels::rope_f32(x.data(), /*n_heads=*/1, /*head_dim=*/8,
                              /*rope_dim=*/8, /*position=*/0, /*base=*/10000.0f);
    for (std::size_t i = 0; i < 8; ++i) {
        CHECK(x[i] == doctest::Approx(static_cast<float>(i + 1)));
    }
}

TEST_CASE("rope_f32: 2D head, position 1, single frequency") {
    // head_dim = 2, rope_dim = 2. half = 1. Only pair (x[0], x[1]).
    // theta_0 = 1 / base^0 = 1. angle = 1 * 1 = 1.
    // x[0] = 1*cos(1) - 0*sin(1) = cos(1)
    // x[1] = 1*sin(1) + 0*cos(1) = sin(1)
    std::vector<float> x{1.0f, 0.0f};
    ultima::kernels::rope_f32(x.data(), 1, 2, 2, 1, 10000.0f);
    CHECK(x[0] == doctest::Approx(std::cos(1.0f)).epsilon(1e-5));
    CHECK(x[1] == doctest::Approx(std::sin(1.0f)).epsilon(1e-5));
}

TEST_CASE("rope_f32: pass-through beyond rope_dim") {
    // head_dim = 8, rope_dim = 4. The last 4 elements should be untouched.
    std::vector<float> x{1, 2, 3, 4, 5, 6, 7, 8};
    ultima::kernels::rope_f32(x.data(), 1, 8, 4, 3, 10000.0f);
    // The tail (indices 4..7) must equal the original values.
    for (std::size_t i = 4; i < 8; ++i) {
        CHECK(x[i] == doctest::Approx(static_cast<float>(i + 1)));
    }
}

TEST_CASE("rope_f32: rotation preserves the L2 norm of each pair") {
    // For any pair (a, b), (a*c - b*s)^2 + (a*s + b*c)^2 = a^2 + b^2.
    std::vector<float> x{0.3f, -0.8f, 1.2f, 0.5f, -0.6f, 0.4f, 0.9f, -1.1f};
    std::vector<float> orig = x;
    ultima::kernels::rope_f32(x.data(), /*n_heads=*/1, /*head_dim=*/8,
                              /*rope_dim=*/8, /*position=*/42, /*base=*/500000.0f);
    // Each pair (i, i + half) is a rotation.
    const std::size_t half = 4;
    for (std::size_t i = 0; i < half; ++i) {
        const float n_new = x[i] * x[i] + x[i + half] * x[i + half];
        const float n_old = orig[i] * orig[i] + orig[i + half] * orig[i + half];
        CHECK(n_new == doctest::Approx(n_old).epsilon(1e-5));
    }
}

TEST_CASE("rope_f32: multi-head applies independently") {
    // Two identical heads at the same position must produce identical outputs.
    constexpr std::size_t H = 2, D = 4;
    std::vector<float> x{1, 2, 3, 4,  1, 2, 3, 4};
    ultima::kernels::rope_f32(x.data(), H, D, D, 7, 1000000.0f);
    for (std::size_t i = 0; i < D; ++i) {
        CHECK(x[i] == doctest::Approx(x[D + i]));
    }
}
