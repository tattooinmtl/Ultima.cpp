#pragma once

#include <cstdint>

namespace ultima::model {

// Data types Ultima recognizes. v0.1 implements dequant/loading for the
// subset marked ACTIVE. Others are enumerated so we can report their
// presence in a model without crashing, but tensors of those types are
// skipped for inference.
enum class DataType : std::uint32_t {
    // ACTIVE in v0.1
    F32  = 0,
    F16  = 1,
    Q8_0 = 8,
    Q4_K = 12,
    Q6_K = 14,

    // ENUMERATED but skipped (values match the public GGUF spec's type IDs)
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q2_K = 10,
    Q3_K = 11,
    Q5_K = 13,
    Q8_K = 15,
    BF16 = 30,

    Unknown = 0xFFFF'FFFFu,
};

const char* to_string(DataType t) noexcept;

// True if v0.1 knows how to load and dequantize this type.
bool is_active(DataType t) noexcept;

// K-quants are stored in super-blocks of 256 elements. Non-K quants are
// stored in blocks of 32.
std::uint32_t block_size(DataType t) noexcept;

// Bytes-per-block for computing tensor size in bytes given element count.
std::uint32_t bytes_per_block(DataType t) noexcept;

} // namespace ultima::model
