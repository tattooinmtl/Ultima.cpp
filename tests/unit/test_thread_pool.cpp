#include <doctest/doctest.h>

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
