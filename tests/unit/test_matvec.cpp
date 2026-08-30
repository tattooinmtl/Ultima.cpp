#include <doctest/doctest.h>

#include "ultima/kernels/matvec.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace {

struct Prng {
    std::mt19937 gen;
    std::uniform_real_distribution<float> dist{-1.0f, 1.0f};
    explicit Prng(std::uint32_t seed) : gen{seed} {}
    float operator()() { return dist(gen); }
};

} // namespace

TEST_CASE("matvec_f32_f32: identity matrix returns input") {
    constexpr std::size_t N = 8;
    std::vector<float> W(N * N, 0.0f);
    for (std::size_t i = 0; i < N; ++i) W[i * N + i] = 1.0f;

    std::vector<float> x{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> y(N, 0.0f);

    ultima::kernels::matvec_f32_f32(W.data(), x.data(), y.data(), N, N);
    for (std::size_t i = 0; i < N; ++i) CHECK(y[i] == doctest::Approx(x[i]));
}

TEST_CASE("matvec_f32_f32: scalar and AVX2 variants agree on random inputs") {
    constexpr std::size_t M = 37;
    constexpr std::size_t K = 131;   // deliberately unaligned to force scalar tail

    Prng rng{12345};
    std::vector<float> W(M * K), x(K), y_scalar(M, 0.0f), y_avx2(M, 0.0f);
    for (auto& v : W) v = rng();
    for (auto& v : x) v = rng();

    ultima::kernels::matvec_f32_f32_scalar(W.data(), x.data(), y_scalar.data(), M, K);
    ultima::kernels::matvec_f32_f32_avx2  (W.data(), x.data(), y_avx2.data(),   M, K);

    for (std::size_t m = 0; m < M; ++m) {
        CHECK(std::fabs(y_scalar[m] - y_avx2[m]) < 1e-4f);
    }
}

TEST_CASE("matvec_f32_f32: dispatcher matches scalar (spot check)") {
    Prng rng{9999};
    constexpr std::size_t M = 16;
    constexpr std::size_t K = 64;
    std::vector<float> W(M * K), x(K), y_pub(M, 0.0f), y_ref(M, 0.0f);
    for (auto& v : W) v = rng();
    for (auto& v : x) v = rng();

    ultima::kernels::matvec_f32_f32       (W.data(), x.data(), y_pub.data(), M, K);
    ultima::kernels::matvec_f32_f32_scalar(W.data(), x.data(), y_ref.data(), M, K);

    for (std::size_t m = 0; m < M; ++m) {
        CHECK(std::fabs(y_pub[m] - y_ref[m]) < 1e-4f);
    }
}
