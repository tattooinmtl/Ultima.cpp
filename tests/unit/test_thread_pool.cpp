#include <doctest/doctest.h>

#include "ultima/kernels/matvec.hpp"
#include "ultima/runtime/thread_pool.hpp"

#include <atomic>
#include <numeric>
#include <vector>

using ultima::runtime::ThreadPool;

TEST_CASE("thread_pool: parallel_for visits every index exactly once") {
    ThreadPool pool{4};
    constexpr std::size_t N = 10'000;
    std::vector<std::atomic<int>> hits(N);
    for (auto& a : hits) a.store(0);

    pool.parallel_for(N, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) hits[i].fetch_add(1);
    });

    for (std::size_t i = 0; i < N; ++i) {
        REQUIRE(hits[i].load() == 1);
    }
}

TEST_CASE("thread_pool: work stays inside its assigned range") {
    ThreadPool pool{2};
    std::atomic<std::size_t> min_seen{~std::size_t{0}};
    std::atomic<std::size_t> max_seen{0};

    pool.parallel_for(1000, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            std::size_t prev = min_seen.load();
            while (i < prev && !min_seen.compare_exchange_weak(prev, i)) {}
            prev = max_seen.load();
            while (i > prev && !max_seen.compare_exchange_weak(prev, i)) {}
        }
    });

    CHECK(min_seen.load() == 0u);
    CHECK(max_seen.load() == 999u);
}

TEST_CASE("thread_pool: n == 0 is a no-op") {
    ThreadPool pool{4};
    std::atomic<int> counter{0};
    pool.parallel_for(0, [&](std::size_t, std::size_t) { counter.fetch_add(1); });
    CHECK(counter.load() == 0);
}

TEST_CASE("thread_pool: default thread count is at least 1") {
    ThreadPool pool{};
    CHECK(pool.size() >= 1u);
}

TEST_CASE("thread_pool: stress — repeated parallel_for does not deadlock") {
    ThreadPool pool{16};
    std::atomic<std::size_t> total{0};
    constexpr std::size_t iters = 20'000;
    for (std::size_t iter = 0; iter < iters; ++iter) {
        pool.parallel_for(1536, [&](std::size_t begin, std::size_t end) {
            total.fetch_add(end - begin, std::memory_order_relaxed);
        });
    }
    CHECK(total.load() == iters * 1536u);
}

TEST_CASE("thread_pool: stress — back-to-back heavy matvec chunks") {
    // Mimics the generation loop: ~ms-scale chunks, sequential dispatches,
    // caller-participating pool. Regression test for an intermittent deadlock
    // seen with 8-16 workers under load.
    ThreadPool pool{16};
    std::vector<std::uint8_t> w(1536 * 6 * 144, 0x55);
    std::vector<float> x(1536, 0.5f);
    std::vector<float> y(1536, 0.0f);
    constexpr int iters = 2'000;
    for (int iter = 0; iter < iters; ++iter) {
        ultima::kernels::matvec_q4k_f32_threaded(pool, w.data(), x.data(),
                                                 y.data(), 1536, 1536);
    }
    CHECK(y[0] == y[0]);   // reachability is the assertion
}
