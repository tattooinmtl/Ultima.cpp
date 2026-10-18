#include <doctest/doctest.h>

#include "ultima/model/weight_dispatch.hpp"
#include "ultima/runtime/thread_pool.hpp"

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

using ultima::model::DataType;
using ultima::model::matmul_dispatch_threaded;
using ultima::model::matvec_dispatch_threaded;

namespace {

std::uint16_t f32_to_fp16(float f) {
    std::uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    std::int32_t  exp  = static_cast<std::int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    std::uint32_t mant = x & 0x007FFFFFu;
    if (exp <= 0)  return static_cast<std::uint16_t>(sign);
    if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7C00u);
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10)
                                           | (mant >> 13));
}

// A Q8_0 matrix of [M, K]: 32 values per block, fp16 scale + 32 int8.
std::vector<std::uint8_t> make_q8_0(std::size_t M, std::size_t K, unsigned seed) {
    std::mt19937 rng{seed};
    std::uniform_int_distribution<int> qd{-127, 127};
    std::vector<std::uint8_t> w;
    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t b = 0; b < K / 32; ++b) {
            const float d = 0.01f + 0.001f * static_cast<float>(b + m);
            const std::uint16_t h = f32_to_fp16(d);
            w.push_back(static_cast<std::uint8_t>(h & 0xFF));
            w.push_back(static_cast<std::uint8_t>(h >> 8));
            for (int i = 0; i < 32; ++i) {
                w.push_back(static_cast<std::uint8_t>(static_cast<std::int8_t>(qd(rng))));
            }
        }
    }
    return w;
}

} // namespace

TEST_CASE("batched matmul agrees with one matvec per token") {
    // This is the property that matters: the GEMM path exists purely to move
    // less weight traffic, so it must be numerically interchangeable with
    // the matvec path it replaces.
    constexpr std::size_t M = 37, K = 128, N = 6;
    ultima::runtime::ThreadPool pool{4};

    const auto w = make_q8_0(M, K, 4242);
    std::mt19937 rng{7};
    std::uniform_real_distribution<float> xd{-1.5f, 1.5f};
    std::vector<float> X(N * K);
    for (auto& v : X) v = xd(rng);

    std::vector<float> batched(N * M, 0.0f);
    REQUIRE(matmul_dispatch_threaded(pool, DataType::Q8_0, w.data(),
                                     X.data(), batched.data(), M, K, N).has_value());

    for (std::size_t t = 0; t < N; ++t) {
        std::vector<float> one(M, 0.0f);
        REQUIRE(matvec_dispatch_threaded(pool, DataType::Q8_0, w.data(),
                                         X.data() + t * K, one.data(), M, K).has_value());
        for (std::size_t m = 0; m < M; ++m) {
            REQUIRE(batched[t * M + m] == doctest::Approx(one[m]).epsilon(1e-5));
        }
    }
}

TEST_CASE("batched matmul keeps tokens independent") {
    // A token's output must depend only on its own row of X. Getting the
    // striding wrong would blend tokens together and still look plausible.
    constexpr std::size_t M = 16, K = 64, N = 4;
    ultima::runtime::ThreadPool pool{2};
    const auto w = make_q8_0(M, K, 99);

    std::vector<float> X(N * K, 0.0f);
    for (std::size_t k = 0; k < K; ++k) X[1 * K + k] = 1.0f;   // only token 1 is nonzero

    std::vector<float> Y(N * M, -1.0f);
    REQUIRE(matmul_dispatch_threaded(pool, DataType::Q8_0, w.data(),
                                     X.data(), Y.data(), M, K, N).has_value());

    for (std::size_t t : {0u, 2u, 3u}) {
        for (std::size_t m = 0; m < M; ++m) CHECK(Y[t * M + m] == doctest::Approx(0.0f));
    }
    bool any_nonzero = false;
    for (std::size_t m = 0; m < M; ++m) any_nonzero |= (Y[1 * M + m] != 0.0f);
    CHECK(any_nonzero);
}

TEST_CASE("a single token falls through to the matvec path") {
    constexpr std::size_t M = 24, K = 96;
    ultima::runtime::ThreadPool pool{2};
    const auto w = make_q8_0(M, K, 5);
    std::vector<float> x(K);
    for (std::size_t k = 0; k < K; ++k) x[k] = 0.01f * static_cast<float>(k % 17);

    std::vector<float> a(M, 0.0f), b(M, 0.0f);
    REQUIRE(matmul_dispatch_threaded(pool, DataType::Q8_0, w.data(),
                                     x.data(), a.data(), M, K, 1).has_value());
    REQUIRE(matvec_dispatch_threaded(pool, DataType::Q8_0, w.data(),
                                     x.data(), b.data(), M, K).has_value());
    for (std::size_t m = 0; m < M; ++m) CHECK(a[m] == doctest::Approx(b[m]));
}

TEST_CASE("batched matmul handles f32 weights") {
    constexpr std::size_t M = 12, K = 32, N = 3;
    ultima::runtime::ThreadPool pool{2};
    std::vector<float> w(M * K);
    std::mt19937 rng{1};
    std::uniform_real_distribution<float> d{-1.0f, 1.0f};
    for (auto& v : w) v = d(rng);
    std::vector<float> X(N * K);
    for (auto& v : X) v = d(rng);

    std::vector<float> Y(N * M, 0.0f);
    REQUIRE(matmul_dispatch_threaded(pool, DataType::F32, w.data(),
                                     X.data(), Y.data(), M, K, N).has_value());

    for (std::size_t t = 0; t < N; ++t) {
        for (std::size_t m = 0; m < M; ++m) {
            float acc = 0.0f;
            for (std::size_t k = 0; k < K; ++k) acc += w[m * K + k] * X[t * K + k];
            CHECK(Y[t * M + m] == doctest::Approx(acc).epsilon(1e-5));
        }
    }
}

TEST_CASE("an unsupported dtype reports failure rather than writing garbage") {
    ultima::runtime::ThreadPool pool{2};
    std::vector<std::uint8_t> w(1024, 0);
    std::vector<float> X(64, 1.0f), Y(32, 0.0f);
    auto r = matmul_dispatch_threaded(pool, DataType::Q3_K, w.data(),
                                      X.data(), Y.data(), 2, 32, 2);
    CHECK_FALSE(r.has_value());
}
