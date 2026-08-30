#include "gguf_reader.hpp"

#include <fmt/format.h>

namespace ultima::model {

using core::ErrorCode;
using core::Result;
using core::fail;

Result<void> GgufReader::ensure_(std::size_t n) const {
    if (cursor_ + n > size_) {
        return fail(ErrorCode::InvalidModel,
                    fmt::format("read past end of file (offset {} + {} > size {})",
                                cursor_, n, size_),
                    "gguf_reader");
    }
    return {};
}

Result<void> GgufReader::skip(std::size_t n) {
    if (auto r = ensure_(n); !r) return core::Failure{r.error()};
    cursor_ += n;
    return {};
}

Result<void> GgufReader::seek(std::size_t abs_offset) {
    if (abs_offset > size_) {
        return fail(ErrorCode::InvalidModel,
                    fmt::format("seek past end of file ({} > size {})", abs_offset, size_),
                    "gguf_reader");
    }
    cursor_ = abs_offset;
    return {};
}

Result<void> GgufReader::align_to(std::uint64_t alignment) {
    if (alignment == 0) return {};
    const std::size_t remainder = cursor_ % alignment;
    if (remainder == 0) return {};
    return skip(static_cast<std::size_t>(alignment - remainder));
}

Result<std::string> GgufReader::read_string() {
    auto len_r = read_scalar<std::uint64_t>();
    if (!len_r) return core::Failure{len_r.error()};

    const std::uint64_t len = *len_r;
    if (len > (1ull << 30)) {
        return fail(ErrorCode::InvalidModel,
                    fmt::format("string length {} exceeds 1 GiB sanity cap", len),
                    "gguf_reader");
    }
    if (auto r = ensure_(static_cast<std::size_t>(len)); !r)
        return core::Failure{r.error()};

    std::string s(reinterpret_cast<const char*>(base_ + cursor_),
                  static_cast<std::size_t>(len));
    cursor_ += static_cast<std::size_t>(len);
    return s;
}

} // namespace ultima::model
