#pragma once

#include "ultima/model/dtype.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace ultima::model {

struct TensorInfo {
    std::string                name;
    std::vector<std::uint64_t> dims;         // n_dims <= 4 in practice
    DataType                   dtype;
    std::uint64_t              offset;       // absolute byte offset in file
    std::uint64_t              size_bytes;   // computed from dtype + dims
};

// Compute size in bytes for a tensor of the given dtype and dims.
// Returns 0 if the dims are incompatible with the dtype's block size.
std::uint64_t compute_tensor_bytes(DataType dtype,
                                   const std::vector<std::uint64_t>& dims) noexcept;

// Total element count (product of dims).
std::uint64_t element_count(const std::vector<std::uint64_t>& dims) noexcept;

} // namespace ultima::model
