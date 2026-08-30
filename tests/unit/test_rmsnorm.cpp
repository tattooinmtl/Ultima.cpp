#include <doctest/doctest.h>

#include "ultima/kernels/norms.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace {

struct Prng {
    std::mt19937 gen;
    std::uniform_real_distribution<float> dist{-1.5f, 1.5f};
    explicit Prng(std::uint32_t seed) : gen{seed} {}
    float operator()() { return dist(gen); }
};

} // namespace

TEST_CASE("rmsnorm_f32: known small case") {
    // x = [1, 2, 3, 4], scale = [1, 1, 1, 1], eps = 0
    // ms = (1+4+9+16)/4 = 7.5, rms = sqrt(7.5) ~ 2.7386
    // y[i] = x[i] / rms  =>  [0.365..., 0.730..., 1.095..., 1.461...]
    std::vector<float> x{1, 2, 3, 4};
    std::vector<float> s{1, 1, 1, 1};
    std::vector<float> y(4, 0.0f);
    ultima::kernels::rmsnorm_f32(x.data(), s.data(), y.data(), 4, 0.0f);

    const float rms = std::sqrt(7.5f);
    for (std::size_t i = 0; i < 4; ++i) {
        CHECK(y[i] == doctest::Approx(x[i] / rms).epsilon(1e-5));
    }
}

TEST_CASE("rmsnorm_f32: scalar and AVX2 agree on random inputs") {
    Prng rng{555};
    constexpr std::size_t N = 896;   // matches Qwen2.5-0.5B embedding dim
    std::vector<float> x(N), sc(N), y_ref(N), y_avx(N);
    for (auto& v : x)  v = rng();
    for (auto& v : sc) v = rng();

    const float eps = 1e-6f;
    ultima::kernels::rmsnorm_f32_scalar(x.data(), sc.data(), y_ref.data(), N, eps);
    ultima::kernels::rmsnorm_f32_avx2  (x.data(), sc.data(), y_avx.data(), N, eps);

    for (std::size_t i = 0; i < N; ++i) {
        CHECK(std::fabs(y_ref[i] - y_avx[i]) < 1e-4f);
    }
}

TEST_CASE("rmsnorm_f32: unaligned length exercises scalar tail") {
    Prng rng{777};
    constexpr std::size_t N = 131;
    std::vector<float> x(N), sc(N, 1.0f), y_ref(N), y_avx(N);
    for (auto& v : x) v = rng();

    ultima::kernels::rmsnorm_f32_scalar(x.data(), sc.data(), y_ref.data(), N, 1e-6f);
    ultima::kernels::rmsnorm_f32_avx2  (x.data(), sc.data(), y_avx.data(), N, 1e-6f);

    for (std::size_t i = 0; i < N; ++i) {
        CHECK(std::fabs(y_ref[i] - y_avx[i]) < 1e-4f);
    }
}

TEST_CASE("rmsnorm_f32: in-place safe") {
    std::vector<float> x{1, 2, 3, 4};
    std::vector<float> s{1, 1, 1, 1};
    std::vector<float> ref(4);

    ultima::kernels::rmsnorm_f32(x.data(), s.data(), ref.data(), 4, 0.0f);
    ultima::kernels::rmsnorm_f32(x.data(), s.data(), x.data(),   4, 0.0f);

    for (std::size_t i = 0; i < 4; ++i) CHECK(x[i] == doctest::Approx(ref[i]));
}
