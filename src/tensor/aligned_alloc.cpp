#include "ultima/tensor/aligned_alloc.hpp"

#include <cstdlib>
#include <cstring>
#include <new>

#ifdef _WIN32
#  include <malloc.h>
#endif

namespace ultima::tensor {

void* aligned_alloc_bytes(std::size_t size_bytes, std::size_t alignment) {
    if (size_bytes == 0) return nullptr;
#ifdef _WIN32
    void* p = _aligned_malloc(size_bytes, alignment);
    if (!p) throw std::bad_alloc{};
    return p;
#else
    void* p = nullptr;
    if (posix_memalign(&p, alignment, size_bytes) != 0) throw std::bad_alloc{};
    return p;
#endif
}

void* aligned_calloc_bytes(std::size_t size_bytes, std::size_t alignment) {
    void* p = aligned_alloc_bytes(size_bytes, alignment);
    if (p) std::memset(p, 0, size_bytes);
    return p;
}

void aligned_free(void* p) noexcept {
    if (!p) return;
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
}

} // namespace ultima::tensor
