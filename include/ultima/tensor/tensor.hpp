#pragma once

#include "ultima/model/dtype.hpp"
#include "ultima/tensor/aligned_alloc.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <utility>
#include <vector>

namespace ultima::tensor {

using ultima::model::DataType;

// A computational tensor. Two storage modes:
//   * OWNED    — heap-allocated, 64-byte aligned, mutable
//   * VIEW     — non-owning wrapper over external memory (e.g. mmap'd
//                weights). Immutable through this Tensor.
//
// The distinction lets kernels take Tensor uniformly whether the input is
// activation scratch (owned) or a weight matrix (view).
class Tensor {
public:
    enum class Storage : std::uint8_t { Owned, View };

    Tensor() = default;

    // Owned allocation. Uninitialized bytes; use `zero_init=true` for calloc.
    static Tensor allocate(DataType dtype, std::vector<std::uint64_t> shape,
                           bool zero_init = false);

    // Non-owning view. Caller guarantees `data` remains valid for the lifetime
    // of this Tensor and all Tensors constructed from it via move.
    static Tensor view(DataType dtype, std::vector<std::uint64_t> shape,
                       const void* data);

    // Move-only
    Tensor(const Tensor&)            = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor();

    // Accessors
    DataType                          dtype()      const noexcept { return dtype_; }
    const std::vector<std::uint64_t>& shape()      const noexcept { return shape_; }
    Storage                           storage()    const noexcept { return storage_; }
    std::uint64_t                     size_bytes() const noexcept { return size_bytes_; }
    std::uint64_t                     element_count() const noexcept;

    const void* data() const noexcept { return data_; }

    // Returns nullptr if this Tensor is a View (immutable). Owned tensors
    // return their writable pointer.
    void* data_mut() noexcept;

    // Convenience typed accessors — no dtype check at runtime; caller asserts.
    template <typename T> const T* as() const noexcept {
        return static_cast<const T*>(data_);
    }
    template <typename T> T* as_mut() noexcept {
        return static_cast<T*>(data_mut());
    }

private:
    Tensor(DataType dtype, std::vector<std::uint64_t> shape, Storage storage,
           void* owned_ptr, const void* data_ptr, std::uint64_t size_bytes)
        : dtype_{dtype}, shape_{std::move(shape)}, storage_{storage},
          owned_ptr_{owned_ptr}, data_{data_ptr}, size_bytes_{size_bytes} {}

    void release_() noexcept;

    DataType                    dtype_      = DataType::F32;
    std::vector<std::uint64_t>  shape_;
    Storage                     storage_    = Storage::View;
    void*                       owned_ptr_  = nullptr;   // aligned_free target; nullptr for views
    const void*                 data_       = nullptr;   // always the read pointer
    std::uint64_t               size_bytes_ = 0;
};

} // namespace ultima::tensor
