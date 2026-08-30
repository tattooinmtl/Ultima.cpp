#pragma once

#include "ultima/core/error.hpp"

#include <cstdint>
#include <cstring>
#include <string>

namespace ultima::model {

// Bounds-checked forward reader over an mmap'd byte range. Little-endian
// scalars. Every read returns Failure on out-of-bounds.
class GgufReader {
public:
    GgufReader(const std::uint8_t* base, std::size_t size) noexcept
        : base_{base}, size_{size} {}

    std::size_t offset() const noexcept { return cursor_; }
    std::size_t size()   const noexcept { return size_; }
    bool        eof()    const noexcept { return cursor_ >= size_; }
    const std::uint8_t* data_at(std::size_t off) const noexcept { return base_ + off; }

    core::Result<void> skip(std::size_t n);
    core::Result<void> seek(std::size_t abs_offset);
    core::Result<void> align_to(std::uint64_t alignment);

    template <typename T>
    core::Result<T> read_scalar() {
        static_assert(std::is_trivially_copyable_v<T>);
        if (auto r = ensure_(sizeof(T)); !r) return core::Failure{r.error()};
        T value{};
        std::memcpy(&value, base_ + cursor_, sizeof(T));
        cursor_ += sizeof(T);
        return value;
    }

    core::Result<std::string> read_string();

private:
    core::Result<void> ensure_(std::size_t n) const;

    const std::uint8_t* base_   = nullptr;
    std::size_t         size_   = 0;
    mutable std::size_t cursor_ = 0;
};

} // namespace ultima::model
