#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace ultima::runtime {

// Fixed-size worker pool. Not a general task graph — supports a blocking
// parallel_for over a contiguous integer range. See Decision 05 §5.3.
class ThreadPool {
public:
    // n_threads = 0 => std::thread::hardware_concurrency() (logical cores).
    // Pass the physical-core count from cpu_features for our intended default.
    explicit ThreadPool(std::size_t n_threads = 0);
    ~ThreadPool();

    ThreadPool(const ThreadPool&)            = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&)                 = delete;
    ThreadPool& operator=(ThreadPool&&)      = delete;

    std::size_t size() const noexcept { return workers_.size(); }

    // Splits [0, n) into contiguous chunks, one per worker. Blocks until all
    // chunks complete. `work_fn` receives [begin, end) for its chunk.
    //
    // For n < size() the pool is under-subscribed; only ceil(n/1) tasks run.
    // For n == 0 this is a no-op.
    void parallel_for(std::size_t n,
                      const std::function<void(std::size_t, std::size_t)>& work_fn);

private:
    void worker_loop_();

    std::vector<std::thread>          workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex                        mtx_;
    std::condition_variable           cv_task_;
    bool                              stopping_ = false;
};

} // namespace ultima::runtime
