#include "ultima/model/tensor_info.hpp"

namespace ultima::model {

std::uint64_t element_count(const std::vector<std::uint64_t>& dims) noexcept {
    if (dims.empty()) return 0;
    std::uint64_t n = 1;
    for (auto d : dims) {
        if (d == 0) return 0;
        n *= d;
    }
    return n;
}

std::uint64_t compute_tensor_bytes(DataType dtype,
                                   const std::vector<std::uint64_t>& dims) noexcept {
    const std::uint64_t n = element_count(dims);
    if (n == 0) return 0;

    const std::uint32_t bs = block_size(dtype);
    const std::uint32_t bpb = bytes_per_block(dtype);
    if (bs == 0 || bpb == 0) return 0;
    if (n % bs != 0) return 0;

    return (n / bs) * bpb;
}

} // namespace ultima::model
