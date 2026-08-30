#include "ultima/tensor/tensor.hpp"

#include "ultima/model/tensor_info.hpp"

namespace ultima::tensor {

using ultima::model::DataType;

Tensor Tensor::allocate(DataType dtype, std::vector<std::uint64_t> shape,
                        bool zero_init) {
    const std::uint64_t bytes = ultima::model::compute_tensor_bytes(dtype, shape);
    void* buf = nullptr;
    if (bytes > 0) {
        buf = zero_init ? aligned_calloc_bytes(static_cast<std::size_t>(bytes))
                        : aligned_alloc_bytes (static_cast<std::size_t>(bytes));
    }
    return Tensor{dtype, std::move(shape), Storage::Owned, buf, buf, bytes};
}

Tensor Tensor::view(DataType dtype, std::vector<std::uint64_t> shape,
                    const void* data) {
    const std::uint64_t bytes = ultima::model::compute_tensor_bytes(dtype, shape);
    return Tensor{dtype, std::move(shape), Storage::View, nullptr, data, bytes};
}

Tensor::Tensor(Tensor&& other) noexcept
    : dtype_{other.dtype_},
      shape_{std::move(other.shape_)},
      storage_{other.storage_},
      owned_ptr_{other.owned_ptr_},
      data_{other.data_},
      size_bytes_{other.size_bytes_} {
    other.owned_ptr_  = nullptr;
    other.data_       = nullptr;
    other.size_bytes_ = 0;
    other.storage_    = Storage::View;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        release_();
        dtype_      = other.dtype_;
        shape_      = std::move(other.shape_);
        storage_    = other.storage_;
        owned_ptr_  = other.owned_ptr_;
        data_       = other.data_;
        size_bytes_ = other.size_bytes_;
        other.owned_ptr_  = nullptr;
        other.data_       = nullptr;
        other.size_bytes_ = 0;
        other.storage_    = Storage::View;
    }
    return *this;
}

Tensor::~Tensor() {
    release_();
}

void Tensor::release_() noexcept {
    if (storage_ == Storage::Owned && owned_ptr_) {
        aligned_free(owned_ptr_);
    }
    owned_ptr_  = nullptr;
    data_       = nullptr;
    size_bytes_ = 0;
}

void* Tensor::data_mut() noexcept {
    return (storage_ == Storage::Owned) ? owned_ptr_ : nullptr;
}

std::uint64_t Tensor::element_count() const noexcept {
    return ultima::model::element_count(shape_);
}

} // namespace ultima::tensor
