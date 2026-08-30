#pragma once

#include <cstddef>
#include <memory>
#include <new>

namespace ultima::tensor {

// Aligned allocation. Cache-line aligned (64 bytes) covers AVX2 (32) with
// headroom for AVX-512 (64) and future NEON. Bytes are uninitialized.
void* aligned_alloc_bytes(std::size_t size_bytes, std::size_t alignment = 64);

// Zero-initialized variant. Slightly slower but predictable for scratch.
void* aligned_calloc_bytes(std::size_t size_bytes, std::size_t alignment = 64);

// Companion free. Do not call std::free / delete on pointers from this API.
void aligned_free(void* p) noexcept;

// RAII deleter usable with std::unique_ptr<T, AlignedDeleter>.
struct AlignedDeleter {
    void operator()(void* p) const noexcept { aligned_free(p); }
};

template <typename T>
using AlignedUniquePtr = std::unique_ptr<T[], AlignedDeleter>;

// Typed allocation helper: allocates count elements, uninitialized, aligned.
template <typename T>
inline AlignedUniquePtr<T> aligned_alloc_n(std::size_t count,
                                           std::size_t alignment = 64) {
    void* p = aligned_alloc_bytes(count * sizeof(T), alignment);
    return AlignedUniquePtr<T>{static_cast<T*>(p)};
}

} // namespace ultima::tensor
