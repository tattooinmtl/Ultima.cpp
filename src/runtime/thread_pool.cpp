#include "ultima/runtime/thread_pool.hpp"

#include <atomic>
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

    const std::size_t nw = workers_.size();
    const std::size_t chunks = (n < nw) ? n : nw;
    const std::size_t chunk_size = (n + chunks - 1) / chunks;

    std::atomic<std::size_t> remaining{chunks};
    std::mutex done_mtx;
    std::condition_variable done_cv;

    {
        std::lock_guard<std::mutex> lk{mtx_};
        for (std::size_t c = 0; c < chunks; ++c) {
            const std::size_t begin = c * chunk_size;
            const std::size_t end   = (begin + chunk_size > n) ? n : begin + chunk_size;
            tasks_.emplace([begin, end, &work_fn, &remaining, &done_mtx, &done_cv]() {
                work_fn(begin, end);
                if (remaining.fetch_sub(1) == 1) {
                    std::lock_guard<std::mutex> dl{done_mtx};
                    done_cv.notify_one();
                }
            });
        }
    }
    cv_task_.notify_all();

    std::unique_lock<std::mutex> dl{done_mtx};
    done_cv.wait(dl, [&remaining] { return remaining.load() == 0; });
}

} // namespace ultima::runtime
