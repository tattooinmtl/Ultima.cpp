#include "ultima/kernels/matvec.hpp"
#include "ultima/runtime/thread_pool.hpp"

#include <cstddef>
#include <cstdint>

namespace ultima::kernels {

namespace {

// Below this M/thread ratio the sync overhead of parallel_for outweighs the
// FMA savings on the reference box. Serial matvec is a memory-bound sweep;
// starting workers only pays off when each gets a real amount of rows.
constexpr std::size_t kMinRowsPerWorker = 2;

template <class Fn>
inline void dispatch_rows(ultima::runtime::ThreadPool& pool,
                          std::size_t M, Fn&& per_range) noexcept {
    if (M == 0) return;
    const std::size_t nt = pool.size();
    if (nt <= 1 || M < nt * kMinRowsPerWorker) {
        per_range(std::size_t{0}, M);
        return;
    }
    pool.parallel_for(M, [&](std::size_t begin, std::size_t end) {
        per_range(begin, end);
    });
}

} // namespace

void matvec_f32_f32_threaded(ultima::runtime::ThreadPool& pool,
                             const float* w, const float* x, float* y,
                             std::size_t M, std::size_t K) noexcept {
    dispatch_rows(pool, M, [&](std::size_t begin, std::size_t end) {
        matvec_f32_f32_avx2(w + begin * K, x, y + begin, end - begin, K);
    });
}

void matvec_q4k_f32_threaded(ultima::runtime::ThreadPool& pool,
                             const std::uint8_t* w, const float* x, float* y,
                             std::size_t M, std::size_t K) noexcept {
    if (K == 0 || (K % 256u) != 0) {
        for (std::size_t m = 0; m < M; ++m) y[m] = 0.0f;
        return;
    }
    const std::size_t bytes_per_row = (K / 256u) * 144u;
    dispatch_rows(pool, M, [&](std::size_t begin, std::size_t end) {
        matvec_q4k_f32(w + begin * bytes_per_row, x, y + begin, end - begin, K);
    });
}

void matvec_q6k_f32_threaded(ultima::runtime::ThreadPool& pool,
                             const std::uint8_t* w, const float* x, float* y,
                             std::size_t M, std::size_t K) noexcept {
    if (K == 0 || (K % 256u) != 0) {
        for (std::size_t m = 0; m < M; ++m) y[m] = 0.0f;
        return;
    }
    const std::size_t bytes_per_row = (K / 256u) * 210u;
    dispatch_rows(pool, M, [&](std::size_t begin, std::size_t end) {
        matvec_q6k_f32(w + begin * bytes_per_row, x, y + begin, end - begin, K);
    });
}

void matvec_q8_0_f32_threaded(ultima::runtime::ThreadPool& pool,
                              const std::uint8_t* w, const float* x, float* y,
                              std::size_t M, std::size_t K) noexcept {
    if (K == 0 || (K % 32u) != 0) {
        for (std::size_t m = 0; m < M; ++m) y[m] = 0.0f;
        return;
    }
    const std::size_t bytes_per_row = (K / 32u) * 34u;
    dispatch_rows(pool, M, [&](std::size_t begin, std::size_t end) {
        matvec_q8_0_f32(w + begin * bytes_per_row, x, y + begin, end - begin, K);
    });
}

} // namespace ultima::kernels
