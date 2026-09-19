#include "ultima/model/weight_dispatch.hpp"

#include "ultima/kernels/dequant_q4_1.hpp"
#include "ultima/kernels/dequant_q4k.hpp"
#include "ultima/kernels/dequant_pq2_0.hpp"
#include "ultima/kernels/dequant_q5_0.hpp"
#include "ultima/kernels/dequant_q5k.hpp"
#include "ultima/kernels/dequant_q6k.hpp"
#include "ultima/kernels/dequant_q8_0.hpp"
#include "ultima/kernels/dot_batched.hpp"
#include "ultima/kernels/matvec.hpp"
#include "ultima/runtime/thread_pool.hpp"

#include <fmt/format.h>
#include <vector>
#include <atomic>

#include <cstdint>
#include <cstring>

namespace ultima::model {

namespace {

using core::ErrorCode;
using core::Failure;
using core::Result;
using core::fail;

} // namespace

float fp16_to_float(std::uint16_t h) noexcept {
    const std::uint32_t sign = (h & 0x8000u) << 16;
    std::uint32_t exp  = (h & 0x7C00u) >> 10;
    std::uint32_t mant = (h & 0x03FFu);
    std::uint32_t out_bits;
    if (exp == 0) {
        if (mant == 0) {
            out_bits = sign;
        } else {
            unsigned shift = 0;
            while ((mant & 0x0400u) == 0) { mant <<= 1; ++shift; }
            mant &= 0x03FFu;
            exp   = 127u - 14u - shift;
            out_bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        out_bits = sign | 0x7F800000u | (mant << 13);
    } else {
        exp = exp - 15u + 127u;
        out_bits = sign | (exp << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out_bits, sizeof(f));
    return f;
}

float bf16_to_float(std::uint16_t h) noexcept {
    // bfloat16 is simply the top 16 bits of an IEEE 754 float32.
    const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

Result<void>
matvec_dispatch(DataType dtype, const void* w, const float* x, float* y,
                std::size_t M, std::size_t K) {
    const auto* wb = static_cast<const std::uint8_t*>(w);
    switch (dtype) {
        case DataType::F32:
            kernels::matvec_f32_f32(static_cast<const float*>(w), x, y, M, K);
            return {};
        case DataType::Q4_K:
            kernels::matvec_q4k_f32(wb, x, y, M, K);
            return {};
        case DataType::Q6_K:
            kernels::matvec_q6k_f32(wb, x, y, M, K);
            return {};
        case DataType::Q8_0:
            kernels::matvec_q8_0_f32(wb, x, y, M, K);
            return {};
        case DataType::Q5_0:
            kernels::matvec_q5_0_f32(wb, x, y, M, K);
            return {};
        case DataType::PQ2_0:
            kernels::matvec_pq2_0_f32(wb, x, y, M, K);
            return {};
        case DataType::BF16:
            kernels::matvec_bf16_f32(wb, x, y, M, K);
            return {};
        case DataType::Q5_K:
            kernels::matvec_q5k_f32(wb, x, y, M, K);
            return {};
        case DataType::Q4_1:
            kernels::matvec_q4_1_f32(wb, x, y, M, K);
            return {};
        default:
            return fail(ErrorCode::InvalidModel,
                        fmt::format("matvec dtype {} not supported",
                                    to_string(dtype)),
                        "weight_dispatch");
    }
}

Result<void>
matvec_dispatch_threaded(runtime::ThreadPool& pool,
                         DataType dtype, const void* w, const float* x, float* y,
                         std::size_t M, std::size_t K) {
    const auto* wb = static_cast<const std::uint8_t*>(w);
    switch (dtype) {
        case DataType::F32:
            kernels::matvec_f32_f32_threaded(pool, static_cast<const float*>(w), x, y, M, K);
            return {};
        case DataType::Q4_K:
            kernels::matvec_q4k_f32_threaded(pool, wb, x, y, M, K);
            return {};
        case DataType::Q6_K:
            kernels::matvec_q6k_f32_threaded(pool, wb, x, y, M, K);
            return {};
        case DataType::Q8_0:
            kernels::matvec_q8_0_f32_threaded(pool, wb, x, y, M, K);
            return {};
        case DataType::Q5_K:
            kernels::matvec_q5k_f32_threaded(pool, wb, x, y, M, K);
            return {};
        case DataType::Q4_1:
            kernels::matvec_q4_1_f32_threaded(pool, wb, x, y, M, K);
            return {};
        case DataType::Q5_0:
            kernels::matvec_q5_0_f32_threaded(pool, wb, x, y, M, K);
            return {};
        case DataType::PQ2_0:
            kernels::matvec_pq2_0_f32_threaded(pool, wb, x, y, M, K);
            return {};
        case DataType::BF16:
            kernels::matvec_bf16_f32_threaded(pool, wb, x, y, M, K);
            return {};
        default:
            return fail(ErrorCode::InvalidModel,
                        fmt::format("matvec_threaded dtype {} not supported",
                                    to_string(dtype)),
                        "weight_dispatch");
    }
}

Result<void>
dequant_1d(DataType dtype, const void* src, float* dst, std::size_t n) {
    if (n == 0) return {};
    const auto* sb = static_cast<const std::uint8_t*>(src);
    switch (dtype) {
        case DataType::F32:
            std::memcpy(dst, src, n * sizeof(float));
            return {};
        case DataType::F16: {
            const auto* h = static_cast<const std::uint16_t*>(src);
            for (std::size_t i = 0; i < n; ++i) dst[i] = fp16_to_float(h[i]);
            return {};
        }
        case DataType::BF16: {
            const auto* h = static_cast<const std::uint16_t*>(src);
            for (std::size_t i = 0; i < n; ++i) dst[i] = bf16_to_float(h[i]);
            return {};
        }
        case DataType::Q8_0: {
            if ((n % 32u) != 0) {
                return fail(ErrorCode::InvalidModel,
                            fmt::format("Q8_0 dequant_1d requires n%32==0, got {}", n),
                            "weight_dispatch");
            }
            kernels::dequant_q8_0(sb, n / 32u, dst);
            return {};
        }
        case DataType::Q5_0: {
            if ((n % 32u) != 0) {
                return fail(ErrorCode::InvalidModel,
                            fmt::format("Q5_0 dequant_1d requires n%32==0, got {}", n),
                            "weight_dispatch");
            }
            kernels::dequant_q5_0(sb, n / 32u, dst);
            return {};
        }
        case DataType::PQ2_0: {
            if ((n % 128u) != 0) {
                return fail(ErrorCode::InvalidModel,
                            fmt::format("PQ2_0 dequant_1d requires n%128==0, got {}", n),
                            "weight_dispatch");
            }
            kernels::dequant_pq2_0(sb, n / 128u, dst);
            return {};
        }
        case DataType::Q2_0: {
            if ((n % 64u) != 0) {
                return fail(ErrorCode::InvalidModel,
                            fmt::format("Q2_0 dequant_1d requires n%64==0, got {}", n),
                            "weight_dispatch");
            }
            kernels::dequant_q2_0(sb, n / 64u, dst);
            return {};
        }
        case DataType::Q4_K: {
            if ((n % 256u) != 0) {
                return fail(ErrorCode::InvalidModel,
                            fmt::format("Q4_K dequant_1d requires n%256==0, got {}", n),
                            "weight_dispatch");
            }
            kernels::dequant_q4k(sb, n / 256u, dst);
            return {};
        }
        case DataType::Q6_K: {
            if ((n % 256u) != 0) {
                return fail(ErrorCode::InvalidModel,
                            fmt::format("Q6_K dequant_1d requires n%256==0, got {}", n),
                            "weight_dispatch");
            }
            kernels::dequant_q6k(sb, n / 256u, dst);
            return {};
        }
        case DataType::Q5_K: {
            if ((n % 256u) != 0) {
                return fail(ErrorCode::InvalidModel,
                            fmt::format("Q5_K dequant_1d requires n%256==0, got {}", n),
                            "weight_dispatch");
            }
            kernels::dequant_q5k(sb, n / 256u, dst);
            return {};
        }
        case DataType::Q4_1: {
            if ((n % 32u) != 0) {
                return fail(ErrorCode::InvalidModel,
                            fmt::format("Q4_1 dequant_1d requires n%32==0, got {}", n),
                            "weight_dispatch");
            }
            kernels::dequant_q4_1(sb, n / 32u, dst);
            return {};
        }
        default:
            return fail(ErrorCode::InvalidModel,
                        fmt::format("dequant_1d dtype {} not supported", to_string(dtype)),
                        "weight_dispatch");
    }
}

Result<void>
read_row(DataType dtype, const void* base, std::size_t row, std::size_t K,
         float* dst) {
    const auto* bb = static_cast<const std::uint8_t*>(base);

    switch (dtype) {
        case DataType::F32: {
            const auto* rows = static_cast<const float*>(base);
            std::memcpy(dst, rows + row * K, K * sizeof(float));
            return {};
        }
        case DataType::F16: {
            const auto* rows = static_cast<const std::uint16_t*>(base);
            const std::uint16_t* r = rows + row * K;
            for (std::size_t i = 0; i < K; ++i) dst[i] = fp16_to_float(r[i]);
            return {};
        }
        case DataType::BF16: {
            const auto* rows = static_cast<const std::uint16_t*>(base);
            const std::uint16_t* r = rows + row * K;
            for (std::size_t i = 0; i < K; ++i) dst[i] = bf16_to_float(r[i]);
            return {};
        }
        case DataType::Q8_0: {
            if ((K % 32u) != 0) {
                return fail(ErrorCode::InvalidModel,
                            fmt::format("Q8_0 read_row requires K%32==0, got {}", K),
                            "weight_dispatch");
            }
            const std::size_t bytes_per_row = (K / 32u) * 34u;
            kernels::dequant_q8_0(bb + row * bytes_per_row, K / 32u, dst);
            return {};
        }
        case DataType::Q5_0: {
            if ((K % 32u) != 0) {
                return fail(ErrorCode::InvalidModel,
                            fmt::format("Q5_0 read_row requires K%32==0, got {}", K),
                            "weight_dispatch");
            }
            const std::size_t bytes_per_row = (K / 32u) * 22u;
            kernels::dequant_q5_0(bb + row * bytes_per_row, K / 32u, dst);
            return {};
        }
        case DataType::PQ2_0: {
            if ((K % 128u) != 0) {
                return fail(ErrorCode::InvalidModel,
                            fmt::format("PQ2_0 read_row requires K%128==0, got {}", K),
                            "weight_dispatch");
            }
            const std::size_t bytes_per_row = (K / 128u) * 34u;
            kernels::dequant_pq2_0(bb + row * bytes_per_row, K / 128u, dst);
            return {};
        }
        case DataType::Q2_0: {
            if ((K % 64u) != 0) {
                return fail(ErrorCode::InvalidModel,
                            fmt::format("Q2_0 read_row requires K%64==0, got {}", K),
                            "weight_dispatch");
            }
            const std::size_t bytes_per_row = (K / 64u) * 18u;
            kernels::dequant_q2_0(bb + row * bytes_per_row, K / 64u, dst);
            return {};
        }
        case DataType::Q4_K: {
            if ((K % 256u) != 0) {
                return fail(ErrorCode::InvalidModel,
                            fmt::format("Q4_K read_row requires K%256==0, got {}", K),
                            "weight_dispatch");
            }
            const std::size_t bytes_per_row = (K / 256u) * 144u;
            kernels::dequant_q4k(bb + row * bytes_per_row, K / 256u, dst);
            return {};
        }
        case DataType::Q6_K: {
            if ((K % 256u) != 0) {
                return fail(ErrorCode::InvalidModel,
                            fmt::format("Q6_K read_row requires K%256==0, got {}", K),
                            "weight_dispatch");
            }
            const std::size_t bytes_per_row = (K / 256u) * 210u;
            kernels::dequant_q6k(bb + row * bytes_per_row, K / 256u, dst);
            return {};
        }
        case DataType::Q5_K: {
            if ((K % 256u) != 0) {
                return fail(ErrorCode::InvalidModel,
                            fmt::format("Q5_K read_row requires K%256==0, got {}", K),
                            "weight_dispatch");
            }
            const std::size_t bytes_per_row = (K / 256u) * 176u;
            kernels::dequant_q5k(bb + row * bytes_per_row, K / 256u, dst);
            return {};
        }
        case DataType::Q4_1: {
            if ((K % 32u) != 0) {
                return fail(ErrorCode::InvalidModel,
                            fmt::format("Q4_1 read_row requires K%32==0, got {}", K),
                            "weight_dispatch");
            }
            const std::size_t bytes_per_row = (K / 32u) * 20u;
            kernels::dequant_q4_1(bb + row * bytes_per_row, K / 32u, dst);
            return {};
        }
        default:
            return fail(ErrorCode::InvalidModel,
                        fmt::format("read_row dtype {} not supported", to_string(dtype)),
                        "weight_dispatch");
    }
}


namespace {

// Same row-splitting rule the matvec kernels use.
constexpr std::size_t kMinRowsPerWorker = 2;

template <class Fn>
inline void dispatch_rows(runtime::ThreadPool& pool, std::size_t M, Fn&& per_range) {
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

Result<void>
matmul_dispatch_threaded(runtime::ThreadPool& pool,
                         DataType dtype, const void* w,
                         const float* X, float* Y,
                         std::size_t M, std::size_t K, std::size_t n_tok) {
    if (n_tok == 0 || M == 0) return {};

    // Below the crossover the fused matvec wins: it decodes quants straight
    // into FMAs and never materializes the row, so there is nothing for
    // batching to amortize yet. Measured on the reference box with
    // benchmarks/bench_gemm (Q8_0, 12288x4096): N=2 is 0.62x, N=4 is 1.17x,
    // N=8 2.19x, N=32 3.39x. Four is where it turns over.
    constexpr std::size_t k_batch_crossover = 4;
    if (n_tok < k_batch_crossover) {
        for (std::size_t t = 0; t < n_tok; ++t) {
            if (auto r = matvec_dispatch_threaded(pool, dtype, w, X + t * K,
                                                  Y + t * M, M, K); !r) {
                return r;
            }
        }
        return {};
    }

    // F32 weights need no dequant, so the row is already in memory and the
    // generic path below would copy it for nothing.
    if (dtype == DataType::F32) {
        const float* wf = static_cast<const float*>(w);
        std::atomic<bool> ok{true};
        dispatch_rows(pool, M, [&](std::size_t begin, std::size_t end) {
            for (std::size_t m = begin; m < end; ++m) {
                kernels::dot_rows_f32(wf + m * K, X, K, n_tok, Y + m, M);
            }
        });
        (void)ok;
        return {};
    }

    // Generic path: dequantize each weight row once into a per-thread
    // scratch, then dot it against every token. Built on read_row so it
    // covers every dtype the loader can read, rather than needing one
    // hand-written GEMM per quantization format.
    //
    // The scratch is K floats — 68 KiB at the widest projection here — so
    // it stays in L2 while the n_tok dots run over it.
    std::atomic<bool> failed{false};
    std::atomic<ErrorCode> err{ErrorCode::InvalidModel};

    dispatch_rows(pool, M, [&](std::size_t begin, std::size_t end) {
        std::vector<float> row(K);
        for (std::size_t m = begin; m < end; ++m) {
            if (failed.load(std::memory_order_relaxed)) return;
            if (auto r = read_row(dtype, w, m, K, row.data()); !r) {
                err.store(r.error().code, std::memory_order_relaxed);
                failed.store(true, std::memory_order_relaxed);
                return;
            }
            kernels::dot_rows_f32(row.data(), X, K, n_tok, Y + m, M);
        }
    });

    if (failed.load(std::memory_order_relaxed)) {
        return fail(err.load(std::memory_order_relaxed),
                    fmt::format("matmul_dispatch dtype {} not supported",
                                to_string(dtype)),
                    "weight_dispatch");
    }
    return {};
}

} // namespace ultima::model
