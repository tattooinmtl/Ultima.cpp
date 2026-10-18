#include "ultima/runtime/thread_pool.hpp"

#include <atomic>
#include <memory>
#include <utility>

namespace ultima::runtime {

ThreadPool::ThreadPool(std::size_t n_threads) {
    if (n_threads == 0) n_threads = std::thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 1;
    workers_.reserve(n_threads);
    for (std::size_t i = 0; i < n_threads; ++i) {
        workers_.emplace_back([this] { worker_loop_(); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lk{mtx_};
        stopping_ = true;
    }
    cv_task_.notify_all();
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
}

void ThreadPool::worker_loop_() {
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lk{mtx_};
            cv_task_.wait(lk, [this] { return stopping_ || !tasks_.empty(); });
            if (stopping_ && tasks_.empty()) return;
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        task();
    }
}

void ThreadPool::parallel_for(std::size_t n,
                              const std::function<void(std::size_t, std::size_t)>& work_fn) {
    if (n == 0) return;

    // Caller participates: workers get chunks 1..chunks-1, the calling thread
    // runs chunk 0 inline. This cuts the wake/wait latency per dispatch and
    // adds one more worker's worth of compute.
    const std::size_t nw = workers_.size();
    const std::size_t chunks = (n < nw + 1) ? n : nw + 1;
    const std::size_t chunk_size = (n + chunks - 1) / chunks;

    if (chunks <= 1) {
        work_fn(0, n);
        return;
    }

    // Completion state is heap-allocated and shared with the task closures:
    // the last worker signals done_cv AFTER the decrement that releases the
    // caller, so stack locals here could be destroyed while that notify is
    // still in flight (use-after-destroy race -> intermittent 500s, hangs,
    // and crashes under load).
    struct DoneState {
        std::atomic<std::size_t> remaining{0};
        std::mutex               mtx;
        std::condition_variable  cv;
    };
    const auto done = std::make_shared<DoneState>();
    done->remaining.store(chunks - 1, std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> lk{mtx_};
        for (std::size_t c = 1; c < chunks; ++c) {
            const std::size_t begin = c * chunk_size;
            const std::size_t end   = (begin + chunk_size > n) ? n : begin + chunk_size;
            tasks_.emplace([begin, end, &work_fn, done]() {
                work_fn(begin, end);
                if (done->remaining.fetch_sub(1) == 1) {
                    std::lock_guard<std::mutex> dl{done->mtx};
                    done->cv.notify_one();
                }
            });
        }
    }
    cv_task_.notify_all();

    // Chunk 0 runs on the caller.
    work_fn(0, chunk_size > n ? n : chunk_size);

    std::unique_lock<std::mutex> dl{done->mtx};
    done->cv.wait(dl, [&done] { return done->remaining.load() == 0; });
}

} // namespace ultima::runtime
