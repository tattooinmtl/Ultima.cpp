// Measures what batching a projection is actually worth on this CPU.
//
// Decode is memory-bound on the weights: one matvec streams the whole
// matrix to produce a single token. Batching N tokens against the same
// matrix should move the same bytes once instead of N times, so the
// interesting number is not absolute throughput but the ratio between
// N separate matvecs and one batched call.
//
// Shapes are taken from the models actually in use rather than round
// numbers, because the win depends on whether a dequantized row stays in
// cache while the N dots run over it.

#include "ultima/model/weight_dispatch.hpp"
#include "ultima/runtime/thread_pool.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using ultima::model::DataType;

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

std::vector<std::uint8_t> make_q8_0(std::size_t M, std::size_t K) {
    std::mt19937 rng{1234};
    std::uniform_int_distribution<int> qd{-127, 127};
    std::vector<std::uint8_t> w;
    w.reserve(M * (K / 32) * 34);
    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t b = 0; b < K / 32; ++b) {
            const std::uint16_t h = f32_to_fp16(0.01f);
            w.push_back(static_cast<std::uint8_t>(h & 0xFF));
            w.push_back(static_cast<std::uint8_t>(h >> 8));
            for (int i = 0; i < 32; ++i) {
                w.push_back(static_cast<std::uint8_t>(static_cast<std::int8_t>(qd(rng))));
            }
        }
    }
    return w;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void run_case(ultima::runtime::ThreadPool& pool, const char* label,
              std::size_t M, std::size_t K, std::size_t N) {
    const auto w = make_q8_0(M, K);
    std::mt19937 rng{7};
    std::uniform_real_distribution<float> d{-1.0f, 1.0f};
    std::vector<float> X(N * K);
    for (auto& v : X) v = d(rng);
    std::vector<float> Y(N * M, 0.0f);

    constexpr int reps = 3;

    // N separate matvecs: the current prefill path.
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) {
        for (std::size_t t = 0; t < N; ++t) {
            (void)ultima::model::matvec_dispatch_threaded(
                pool, DataType::Q8_0, w.data(), X.data() + t * K,
                Y.data() + t * M, M, K);
        }
    }
    const double seq = seconds_since(t0) / reps;

    // One batched call.
    t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) {
        (void)ultima::model::matmul_dispatch_threaded(
            pool, DataType::Q8_0, w.data(), X.data(), Y.data(), M, K, N);
    }
    const double bat = seconds_since(t0) / reps;

    std::printf("  %-22s M=%-6zu K=%-6zu N=%-3zu  seq %7.1f ms  batched %7.1f ms  %5.2fx\n",
                label, M, K, N, seq * 1e3, bat * 1e3, bat > 0 ? seq / bat : 0.0);
}

} // namespace

int main() {
    ultima::runtime::ThreadPool pool{0};   // physical cores
    std::printf("batched projection vs one matvec per token (Q8_0, %zu threads)\n",
                pool.size());

    // Qwopus3.5-9B shapes.
    for (std::size_t n : {2u, 4u, 8u, 16u, 32u}) {
        run_case(pool, "qwen35 ffn_gate", 12288, 4096, n);
    }
    std::printf("\n");
    for (std::size_t n : {4u, 16u}) {
        run_case(pool, "qwen35 attn_q",    4096, 4096, n);
        run_case(pool, "qwen35 ffn_down",  4096, 12288, n);
    }
    return 0;
}
