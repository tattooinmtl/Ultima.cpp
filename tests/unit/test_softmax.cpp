#include <doctest/doctest.h>

#include "ultima/kernels/softmax.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace {

struct Prng {
    std::mt19937 gen;
    std::uniform_real_distribution<float> dist{-5.0f, 5.0f};
    explicit Prng(std::uint32_t seed) : gen{seed} {}
    float operator()() { return dist(gen); }
};

} // namespace

TEST_CASE("softmax_f32: uniform input is uniform output") {
    std::vector<float> x(16, 3.14159f);
    std::vector<float> y(16, 0.0f);
    ultima::kernels::softmax_f32(x.data(), y.data(), x.size());
    for (float v : y) CHECK(v == doctest::Approx(1.0f / 16.0f).epsilon(1e-6));
}

TEST_CASE("softmax_f32: known 3-way split") {
    // Peak at index 1: expect the largest weight there.
    std::vector<float> x{0.0f, 2.0f, 0.0f};
    std::vector<float> y(3, 0.0f);
    ultima::kernels::softmax_f32(x.data(), y.data(), 3);

    CHECK(y[1] > y[0]);
    CHECK(y[1] > y[2]);
    CHECK(y[0] == doctest::Approx(y[2]).epsilon(1e-5));
    float sum = y[0] + y[1] + y[2];
    CHECK(sum == doctest::Approx(1.0f).epsilon(1e-6));
}

TEST_CASE("softmax_f32: output sums to 1 on random inputs") {
    Prng rng{9001};
    constexpr std::size_t N = 512;
    std::vector<float> x(N), y(N);
    for (auto& v : x) v = rng();
    ultima::kernels::softmax_f32(x.data(), y.data(), N);
    double sum = 0.0;
    for (float v : y) {
        CHECK(v >= 0.0f);
        sum += v;
    }
    CHECK(sum == doctest::Approx(1.0).epsilon(1e-5));
}

TEST_CASE("softmax_f32: numerically stable for huge inputs") {
    // Without the max-shift, exp(100) overflows. Stable softmax handles it.
    std::vector<float> x{100.0f, 101.0f, 102.0f};
    std::vector<float> y(3, 0.0f);
    ultima::kernels::softmax_f32(x.data(), y.data(), 3);
    double sum = double(y[0]) + double(y[1]) + double(y[2]);
    CHECK(sum == doctest::Approx(1.0).epsilon(1e-6));
    CHECK(y[2] > y[1]);
    CHECK(y[1] > y[0]);
    for (float v : y) CHECK(std::isfinite(v));
}

TEST_CASE("softmax_f32: in-place safe") {
    std::vector<float> x{1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> ref(4);
    ultima::kernels::softmax_f32(x.data(), ref.data(), 4);
    ultima::kernels::softmax_f32(x.data(), x.data(),   4);
    for (std::size_t i = 0; i < 4; ++i) CHECK(x[i] == doctest::Approx(ref[i]));
}

TEST_CASE("softmax_f32: n == 0 is a no-op") {
    ultima::kernels::softmax_f32(nullptr, nullptr, 0);
    CHECK(true);
}
