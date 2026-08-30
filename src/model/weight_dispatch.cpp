#include "ultima/model/weight_dispatch.hpp"

#include "ultima/kernels/dequant_q4k.hpp"
#include "ultima/kernels/dequant_q6k.hpp"
#include "ultima/kernels/dequant_q8_0.hpp"
#include "ultima/kernels/matvec.hpp"
#include "ultima/runtime/thread_pool.hpp"

#include <fmt/format.h>

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
        default:
            return fail(ErrorCode::InvalidModel,
                        fmt::format("matvec dtype {} not supported in v0.1",
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
        default:
            return fail(ErrorCode::InvalidModel,
                        fmt::format("matvec_threaded dtype {} not supported in v0.1",
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
        case DataType::Q8_0: {
            if ((n % 32u) != 0) {
                return fail(ErrorCode::InvalidModel,
                            fmt::format("Q8_0 dequant_1d requires n%32==0, got {}", n),
                            "weight_dispatch");
            }
            kernels::dequant_q8_0(sb, n / 32u, dst);
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
        default:
            return fail(ErrorCode::InvalidModel,
                        fmt::format("read_row dtype {} not supported", to_string(dtype)),
                        "weight_dispatch");
    }
}

} // namespace ultima::model
