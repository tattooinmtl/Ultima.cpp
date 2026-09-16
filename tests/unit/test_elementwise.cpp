#include <doctest/doctest.h>

#include "ultima/kernels/elementwise.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace {

struct Prng {
    std::mt19937 gen;
    std::uniform_real_distribution<float> dist{-3.0f, 3.0f};
    explicit Prng(std::uint32_t seed) : gen{seed} {}
    float operator()() { return dist(gen); }
};

constexpr float kTol = 1e-4f;

} // namespace

TEST_CASE("add_f32: scalar and AVX2 agree on random unaligned length") {
    Prng rng{101};
    constexpr std::size_t N = 131;
    std::vector<float> a(N), b(N), y_ref(N), y_avx(N);
    for (auto& v : a) v = rng();
    for (auto& v : b) v = rng();

    ultima::kernels::add_f32_scalar(a.data(), b.data(), y_ref.data(), N);
    ultima::kernels::add_f32_avx2  (a.data(), b.data(), y_avx.data(), N);
    for (std::size_t i = 0; i < N; ++i) {
        CHECK(std::fabs(y_ref[i] - y_avx[i]) < kTol);
    }
}

TEST_CASE("add_f32: in-place add (y == a) works") {
    std::vector<float> a{1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<float> b{9, 8, 7, 6, 5, 4, 3, 2, 1};
    ultima::kernels::add_f32(a.data(), b.data(), a.data(), a.size());
    for (float v : a) CHECK(v == doctest::Approx(10.0f));
}

TEST_CASE("mul_f32: scalar and AVX2 agree on random unaligned length") {
    Prng rng{202};
    constexpr std::size_t N = 97;
    std::vector<float> a(N), b(N), y_ref(N), y_avx(N);
    for (auto& v : a) v = rng();
    for (auto& v : b) v = rng();

    ultima::kernels::mul_f32_scalar(a.data(), b.data(), y_ref.data(), N);
    ultima::kernels::mul_f32_avx2  (a.data(), b.data(), y_avx.data(), N);
    for (std::size_t i = 0; i < N; ++i) {
        CHECK(std::fabs(y_ref[i] - y_avx[i]) < kTol);
    }
}

TEST_CASE("silu_f32: known values") {
    // silu(0) = 0; silu(x) approaches x for large positive x; silu -> 0 for large negative x.
    std::vector<float> x{-10.0f, 0.0f, 1.0f, 10.0f};
    std::vector<float> y(x.size(), 0.0f);
    ultima::kernels::silu_f32(x.data(), y.data(), x.size());
    CHECK(y[0] == doctest::Approx(0.0f).epsilon(1e-3));
    CHECK(y[1] == doctest::Approx(0.0f));
    CHECK(y[2] == doctest::Approx(0.7310585786f).epsilon(1e-4));   // 1 * sigmoid(1)
    CHECK(y[3] == doctest::Approx(10.0f).epsilon(1e-3));
}

TEST_CASE("silu_f32: in-place matches out-of-place") {
    Prng rng{303};
    constexpr std::size_t N = 64;
    std::vector<float> x(N), y_out(N), y_in(N);
    for (auto& v : x) v = rng();
    y_in = x;

    ultima::kernels::silu_f32(x.data(),  y_out.data(), N);
    ultima::kernels::silu_f32(y_in.data(), y_in.data(), N);
    for (std::size_t i = 0; i < N; ++i) {
        CHECK(y_out[i] == doctest::Approx(y_in[i]));
    }
}

TEST_CASE("swiglu_f32: equivalent to silu(gate) * up") {
    Prng rng{404};
    constexpr std::size_t N = 128;
    std::vector<float> g(N), u(N), y(N), sg(N), y_ref(N);
    for (auto& v : g) v = rng();
    for (auto& v : u) v = rng();

    ultima::kernels::swiglu_f32(g.data(), u.data(), y.data(), N);

    ultima::kernels::silu_f32(g.data(), sg.data(), N);
    ultima::kernels::mul_f32(sg.data(), u.data(), y_ref.data(), N);

    for (std::size_t i = 0; i < N; ++i) {
        CHECK(std::fabs(y[i] - y_ref[i]) < kTol);
    }
}

// ---- GEGLU ------------------------------------------------------------------

TEST_CASE("geglu_f32: matches the tanh GELU formula times up") {
    const std::vector<float> gate{-4.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 4.0f};
    const std::vector<float> up  { 2.0f,  1.0f, -1.0f, 3.0f, 0.5f, 2.0f, 1.0f};
    std::vector<float> y(gate.size(), 0.0f);

    ultima::kernels::geglu_f32(gate.data(), up.data(), y.data(), gate.size());

    for (std::size_t i = 0; i < gate.size(); ++i) {
        const float g = gate[i];
        const float inner = 0.7978845608028654f * (g + 0.044715f * g * g * g);
        const float want = 0.5f * g * (1.0f + std::tanh(inner)) * up[i];
        CHECK(y[i] == doctest::Approx(want).epsilon(1e-6));
    }
}

TEST_CASE("geglu_f32: gelu(0) is 0 and large positives pass through") {
    const std::vector<float> gate{0.0f, 20.0f, -20.0f};
    const std::vector<float> up  {5.0f,  1.0f,   1.0f};
    std::vector<float> y(3, 0.0f);
    ultima::kernels::geglu_f32(gate.data(), up.data(), y.data(), 3);
    CHECK(y[0] == doctest::Approx(0.0f));
    CHECK(y[1] == doctest::Approx(20.0f).epsilon(1e-5));
    CHECK(y[2] == doctest::Approx(0.0f).epsilon(1e-5));
}

TEST_CASE("geglu_f32: differs from swiglu, and is in-place safe") {
    std::vector<float> gate{1.0f, -1.0f, 2.5f, -2.5f};
    const std::vector<float> up{1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> ge(4), sw(4);
    ultima::kernels::geglu_f32 (gate.data(), up.data(), ge.data(), 4);
    ultima::kernels::swiglu_f32(gate.data(), up.data(), sw.data(), 4);
    bool any_diff = false;
    for (std::size_t i = 0; i < 4; ++i) {
        if (std::fabs(ge[i] - sw[i]) > 1e-4f) any_diff = true;
    }
    CHECK(any_diff);   // picking the wrong one must not go unnoticed

    // Writing over the gate buffer gives the same answer.
    ultima::kernels::geglu_f32(gate.data(), up.data(), gate.data(), 4);
    for (std::size_t i = 0; i < 4; ++i) CHECK(gate[i] == doctest::Approx(ge[i]));
}
