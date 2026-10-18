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
    Q2_0 = 42,   // upstream 2-bit; the dspark drafter stores its embedding in it

    // Prism-private ternary formats, used by the Ternary-Bonsai builds.
    // Not in the public GGUF spec — these ids come from the PrismML-Eng
    // llama.cpp fork, and stock llama.cpp rejects such files outright
    // rather than misreading them.
    PQ2_0  = 142,   // {-1, 0, +1, +2} at 2 bits, one fp16 scale per 128
    PTQ1_0 = 143,   // base-3 trits, one fp16 scale per 128

    Unknown = 0xFFFF'FFFFu,
};

const char* to_string(DataType t) noexcept;

// True if Ultima can dequantize this type (dequant_1d / read_row), which is
// what norms, biases and the embedding row lookup need.
//
// NOT the same as "usable as a projection weight": matvec_dispatch covers a
// subset (F32, Q4_1, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K). F16 and BF16 dequantize
// but have no matvec kernel yet, which is fine because no supported
// architecture stores a projection matrix in them.
bool is_active(DataType t) noexcept;

// K-quants are stored in super-blocks of 256 elements, non-K quants in
// blocks of 32, and the Prism ternary formats in groups of 128.
std::uint32_t block_size(DataType t) noexcept;

// Bytes-per-block for computing tensor size in bytes given element count.
std::uint32_t bytes_per_block(DataType t) noexcept;

} // namespace ultima::model
