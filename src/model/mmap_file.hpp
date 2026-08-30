#pragma once

#include "ultima/core/error.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace ultima::model {

// RAII memory-mapped, read-only, private view of a file. Owns the OS handle
// and address range; unmaps on destruction. Non-copyable, movable.
class MmapFile {
public:
    MmapFile() = default;
    ~MmapFile();

    MmapFile(const MmapFile&)            = delete;
    MmapFile& operator=(const MmapFile&) = delete;

    MmapFile(MmapFile&& other) noexcept;
    MmapFile& operator=(MmapFile&& other) noexcept;

    static core::Result<MmapFile> open(const std::filesystem::path& path);

    const std::uint8_t* data() const noexcept { return static_cast<const std::uint8_t*>(base_); }
    std::size_t         size() const noexcept { return size_; }
    bool                is_open() const noexcept { return base_ != nullptr; }

private:
    void  close_() noexcept;

    void*       base_ = nullptr;
    std::size_t size_ = 0;

    // Platform-specific handles, void* to keep the header portable.
    void*       platform_handle_a_ = nullptr;   // file handle
    void*       platform_handle_b_ = nullptr;   // mapping handle
};

} // namespace ultima::model
