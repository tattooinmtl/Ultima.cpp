#include <doctest/doctest.h>

#include "ultima/kernels/delta_net.hpp"

#include <cmath>
#include <numeric>
#include <random>
#include <vector>

using namespace ultima::kernels;

namespace {

// The delta rule, written the slow obvious way from the reference's index
// form. The kernel fuses the decay into the first sweep and the rank-1
// update into the output sweep; this does each step separately so a fusion
// bug has somewhere to show up.
void delta_reference(std::vector<float>& S, const std::vector<float>& q,
                     const std::vector<float>& k, const std::vector<float>& v,
                     float decay, float beta, std::vector<float>& out,
                     std::size_t n) {
    for (auto& s : S) s *= decay;

    std::vector<float> sk(n, 0.0f);
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) sk[j] += S[i * n + j] * k[i];
    }
    std::vector<float> d(n);
    for (std::size_t j = 0; j < n; ++j) d[j] = beta * (v[j] - sk[j]);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) S[i * n + j] += k[i] * d[j];
    }
    out.assign(n, 0.0f);
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) out[j] += S[i * n + j] * q[i];
    }
}

} // namespace

TEST_CASE("l2_norm scales a vector to unit length") {
    std::vector<float> x{3.0f, 4.0f};
    std::vector<float> y(2);
    l2_norm_f32(x.data(), y.data(), 2, 0.0f);
    CHECK(y[0] == doctest::Approx(0.6f));
    CHECK(y[1] == doctest::Approx(0.8f));

    const float len = std::sqrt(y[0] * y[0] + y[1] * y[1]);
    CHECK(len == doctest::Approx(1.0f));
}

TEST_CASE("l2_norm is not rms_norm") {
    // The distinction matters: rms_norm would leave length sqrt(n), not 1.
    // Getting this wrong scales every q and k by sqrt(128) and the model
    // still produces fluent-looking output, so it is worth pinning.
    std::vector<float> x(16, 1.0f);
    std::vector<float> y(16);
    l2_norm_f32(x.data(), y.data(), 16, 0.0f);
    float ss = 0.0f;
    for (float f : y) ss += f * f;
    CHECK(ss == doctest::Approx(1.0f));
    CHECK(y[0] == doctest::Approx(0.25f));   // 1/sqrt(16)
}

TEST_CASE("l2_norm survives an all-zero head") {
    std::vector<float> x(8, 0.0f);
    std::vector<float> y(8, 99.0f);
    l2_norm_f32(x.data(), y.data(), 8, 1e-6f);
    for (float f : y) CHECK(f == doctest::Approx(0.0f));
}

TEST_CASE("softplus matches log1p(exp(x)) and does not overflow") {
    std::vector<float> x{-10.0f, -1.0f, 0.0f, 1.0f, 10.0f, 50.0f, 200.0f};
    std::vector<float> y(x.size());
    softplus_f32(x.data(), y.data(), x.size());

    CHECK(y[2] == doctest::Approx(std::log(2.0f)));
    for (std::size_t i = 0; i < 5; ++i) {
        CHECK(y[i] == doctest::Approx(std::log1p(std::exp(x[i]))).epsilon(1e-5));
    }
    // Past the cutoff softplus(x) == x to f32 precision, and must be finite.
    CHECK(std::isfinite(y[5]));
    CHECK(y[5] == doctest::Approx(50.0f));
    CHECK(y[6] == doctest::Approx(200.0f));
    // Softplus is strictly positive everywhere.
    for (float f : y) CHECK(f > 0.0f);
}

TEST_CASE("causal conv1d sees only the current and previous inputs") {
    // One channel, width 3, kernel [1, 10, 100] over inputs 1, 2, 3.
    // Starting from a zero state the taps line up oldest-first, so the
    // outputs are 100, 10*1+100*2 = 210, and 1*1+10*2+100*3 = 321.
    constexpr std::size_t ch = 1, w = 3;
    std::vector<float> state(ch * (w - 1), 0.0f);
    const std::vector<float> kern{1.0f, 10.0f, 100.0f};
    std::vector<float> y(ch);

    float x0 = 1.0f;
    causal_conv1d_step_f32(state.data(), &x0, kern.data(), y.data(), ch, w);
    CHECK(y[0] == doctest::Approx(100.0f));

    float x1 = 2.0f;
    causal_conv1d_step_f32(state.data(), &x1, kern.data(), y.data(), ch, w);
    CHECK(y[0] == doctest::Approx(210.0f));

    float x2 = 3.0f;
    causal_conv1d_step_f32(state.data(), &x2, kern.data(), y.data(), ch, w);
    CHECK(y[0] == doctest::Approx(321.0f));

    // The state now holds the last two inputs, oldest first.
    CHECK(state[0] == doctest::Approx(2.0f));
    CHECK(state[1] == doctest::Approx(3.0f));
}

TEST_CASE("causal conv1d keeps channels independent") {
    constexpr std::size_t ch = 3, w = 4;
    std::vector<float> state(ch * (w - 1), 0.0f);
    std::vector<float> kern(ch * w, 0.0f);
    // Channel c responds only to its own most recent input, scaled by c+1.
    for (std::size_t c = 0; c < ch; ++c) kern[c * w + (w - 1)] = float(c + 1);

    const std::vector<float> x{5.0f, 7.0f, 11.0f};
    std::vector<float> y(ch);
    causal_conv1d_step_f32(state.data(), x.data(), kern.data(), y.data(), ch, w);

    CHECK(y[0] == doctest::Approx(5.0f));
    CHECK(y[1] == doctest::Approx(14.0f));
    CHECK(y[2] == doctest::Approx(33.0f));
}

TEST_CASE("delta_net_step matches the reference recurrence") {
    constexpr std::size_t n = 16;
    std::mt19937 rng{1234};
    std::uniform_real_distribution<float> d{-1.0f, 1.0f};

    std::vector<float> S(n * n), q(n), k(n), v(n);
    for (auto& x : S) x = d(rng);
    for (auto& x : q) x = d(rng);
    for (auto& x : k) x = d(rng);
    for (auto& x : v) x = d(rng);

    const float decay = 0.9f, beta = 0.3f;

    std::vector<float> S_ref = S, out_ref;
    delta_reference(S_ref, q, k, v, decay, beta, out_ref, n);

    std::vector<float> out(n), scratch(n);
    delta_net_step_f32(S.data(), q.data(), k.data(), v.data(),
                       decay, beta, out.data(), scratch.data(), n);

    for (std::size_t i = 0; i < n * n; ++i) {
        CHECK(S[i] == doctest::Approx(S_ref[i]).epsilon(1e-5));
    }
    for (std::size_t j = 0; j < n; ++j) {
        CHECK(out[j] == doctest::Approx(out_ref[j]).epsilon(1e-5));
    }
}

TEST_CASE("delta_net_step stays consistent over many steps") {
    // One step agreeing could hide an error that only compounds. The state
    // is the whole memory of the layer, so drift is the failure that
    // matters.
    constexpr std::size_t n = 8;
    std::mt19937 rng{99};
    std::uniform_real_distribution<float> d{-1.0f, 1.0f};

    std::vector<float> S(n * n, 0.0f), S_ref(n * n, 0.0f);
    std::vector<float> out(n), out_ref, scratch(n);

    for (int t = 0; t < 64; ++t) {
        std::vector<float> q(n), k(n), v(n);
        for (auto& x : q) x = d(rng);
        for (auto& x : k) x = d(rng);
        for (auto& x : v) x = d(rng);
        const float decay = 0.5f + 0.49f * std::abs(d(rng));
        const float beta  = 0.5f + 0.5f * d(rng);

        delta_reference(S_ref, q, k, v, decay, beta, out_ref, n);
        delta_net_step_f32(S.data(), q.data(), k.data(), v.data(),
                           decay, beta, out.data(), scratch.data(), n);

        for (std::size_t j = 0; j < n; ++j) {
            REQUIRE(out[j] == doctest::Approx(out_ref[j]).epsilon(1e-4));
        }
    }
    for (std::size_t i = 0; i < n * n; ++i) {
        CHECK(S[i] == doctest::Approx(S_ref[i]).epsilon(1e-4));
    }
}

TEST_CASE("delta_net_step with beta 1 writes v straight into the readout") {
    // beta = 1 makes the update exact: after writing key k, querying with
    // the same k must return v. This is the property the delta rule exists
    // for, and it holds independently of how the kernel is fused.
    constexpr std::size_t n = 32;
    std::vector<float> S(n * n, 0.0f), out(n), scratch(n);

    std::vector<float> k(n, 0.0f), v(n);
    k[3] = 1.0f;                                  // a unit key
    for (std::size_t j = 0; j < n; ++j) v[j] = float(j) * 0.1f;

    delta_net_step_f32(S.data(), k.data(), k.data(), v.data(),
                       /*decay=*/1.0f, /*beta=*/1.0f,
                       out.data(), scratch.data(), n);

    for (std::size_t j = 0; j < n; ++j) {
        CHECK(out[j] == doctest::Approx(v[j]).epsilon(1e-5));
    }
}

TEST_CASE("delta_net_step decay of zero forgets everything") {
    constexpr std::size_t n = 8;
    std::vector<float> S(n * n, 7.0f), out(n), scratch(n);
    std::vector<float> q(n, 0.0f), k(n, 0.0f), v(n, 0.0f);

    delta_net_step_f32(S.data(), q.data(), k.data(), v.data(),
                       /*decay=*/0.0f, /*beta=*/0.0f,
                       out.data(), scratch.data(), n);

    for (float s : S)   CHECK(s == doctest::Approx(0.0f));
    for (float o : out) CHECK(o == doctest::Approx(0.0f));
}
