#include "mmap_file.hpp"

#include <fmt/format.h>

#ifdef _WIN32

// clang-format off
#include <windows.h>
// clang-format on

namespace ultima::model {

using core::ErrorCode;
using core::Result;
using core::fail;

MmapFile::~MmapFile() {
    close_();
}

MmapFile::MmapFile(MmapFile&& other) noexcept
    : base_{other.base_},
      size_{other.size_},
      platform_handle_a_{other.platform_handle_a_},
      platform_handle_b_{other.platform_handle_b_} {
    other.base_              = nullptr;
    other.size_              = 0;
    other.platform_handle_a_ = nullptr;
    other.platform_handle_b_ = nullptr;
}

MmapFile& MmapFile::operator=(MmapFile&& other) noexcept {
    if (this != &other) {
        close_();
        base_              = other.base_;
        size_              = other.size_;
        platform_handle_a_ = other.platform_handle_a_;
        platform_handle_b_ = other.platform_handle_b_;
        other.base_              = nullptr;
        other.size_              = 0;
        other.platform_handle_a_ = nullptr;
        other.platform_handle_b_ = nullptr;
    }
    return *this;
}

void MmapFile::close_() noexcept {
    if (base_) {
        UnmapViewOfFile(base_);
        base_ = nullptr;
    }
    if (platform_handle_b_) {
        CloseHandle(platform_handle_b_);
        platform_handle_b_ = nullptr;
    }
    if (platform_handle_a_ && platform_handle_a_ != INVALID_HANDLE_VALUE) {
        CloseHandle(platform_handle_a_);
        platform_handle_a_ = nullptr;
    }
    size_ = 0;
}

Result<MmapFile> MmapFile::open(const std::filesystem::path& path) {
    const auto wpath = path.wstring();

    HANDLE file = CreateFileW(
        wpath.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS,
        nullptr
    );
    if (file == INVALID_HANDLE_VALUE) {
        const DWORD err = GetLastError();
        return fail(
            err == ERROR_FILE_NOT_FOUND ? ErrorCode::FileNotFound : ErrorCode::IoFailure,
            fmt::format("CreateFileW failed: error {} for path '{}'", err, path.string()),
            "mmap"
        );
    }

    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(file, &sz)) {
        const DWORD err = GetLastError();
        CloseHandle(file);
        return fail(ErrorCode::IoFailure,
                    fmt::format("GetFileSizeEx failed: error {}", err),
                    "mmap");
    }
    if (sz.QuadPart == 0) {
        CloseHandle(file);
        return fail(ErrorCode::FileTooSmall,
                    fmt::format("file is empty: '{}'", path.string()),
                    "mmap");
    }

    HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping) {
        const DWORD err = GetLastError();
        CloseHandle(file);
        return fail(ErrorCode::MmapFailure,
                    fmt::format("CreateFileMappingW failed: error {}", err),
                    "mmap");
    }

    void* base = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (!base) {
        const DWORD err = GetLastError();
        CloseHandle(mapping);
        CloseHandle(file);
        return fail(ErrorCode::MmapFailure,
                    fmt::format("MapViewOfFile failed: error {}", err),
                    "mmap");
    }

    // Pull the whole mapping into the page cache up front.
    //
    // Without this the first forward pass faults the file in one 4 KiB page
    // at a time, in whatever order the graph happens to touch weights — and
    // for a sparse-MoE model that order is driven by expert routing, so it
    // is close to random over tens of GB. FILE_FLAG_RANDOM_ACCESS above
    // (correct for steady state, where a token touches 8 of 256 experts)
    // also tells Windows not to read ahead, so nothing coalesces those
    // faults. The result is a model that decodes at a fraction of its real
    // speed until the cache incidentally fills: measured 0.29 tok/s on the
    // first request against 4.1 tok/s warm, same process, same prompt.
    //
    // One PrefetchVirtualMemory call turns that into large sequential I/O.
    // It is advisory — the OS may prefetch less under memory pressure, and
    // a failure is not an error — so the return value is deliberately
    // ignored. Win8+ only, hence the runtime lookup rather than a link-time
    // import. This is what llama.cpp does (src/llama-mmap.cpp), which is why
    // it reaches a stable rate immediately and Ultima did not.
    if (HMODULE k32 = GetModuleHandleW(L"kernel32.dll")) {
        using prefetch_fn = BOOL(WINAPI*)(HANDLE, ULONG_PTR,
                                          PWIN32_MEMORY_RANGE_ENTRY, ULONG);
        auto prefetch = reinterpret_cast<prefetch_fn>(
            reinterpret_cast<void*>(GetProcAddress(k32, "PrefetchVirtualMemory")));
        if (prefetch) {
            WIN32_MEMORY_RANGE_ENTRY range{};
            range.VirtualAddress = base;
            range.NumberOfBytes  = static_cast<SIZE_T>(sz.QuadPart);
            prefetch(GetCurrentProcess(), 1, &range, 0);
        }
    }

    MmapFile out;
    out.base_              = base;
    out.size_              = static_cast<std::size_t>(sz.QuadPart);
    out.platform_handle_a_ = file;
    out.platform_handle_b_ = mapping;
    return out;
}

} // namespace ultima::model

#endif // _WIN32
