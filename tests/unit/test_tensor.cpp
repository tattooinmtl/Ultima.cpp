#include <doctest/doctest.h>

#include "ultima/model/dtype.hpp"
#include "ultima/tensor/tensor.hpp"

using ultima::model::DataType;
using ultima::tensor::Tensor;

TEST_CASE("tensor: allocate F32 uninitialized") {
    Tensor t = Tensor::allocate(DataType::F32, {3, 4});
    CHECK(t.dtype() == DataType::F32);
    CHECK(t.shape().size() == 2);
    CHECK(t.shape()[0] == 3u);
    CHECK(t.shape()[1] == 4u);
    CHECK(t.element_count() == 12u);
    CHECK(t.size_bytes() == 48u);
    CHECK(t.storage() == Tensor::Storage::Owned);
    REQUIRE(t.data() != nullptr);
    REQUIRE(t.data_mut() != nullptr);
}

TEST_CASE("tensor: allocate zero-initialized") {
    Tensor t = Tensor::allocate(DataType::F32, {16}, /*zero_init=*/true);
    const float* p = t.as<float>();
    for (std::size_t i = 0; i < 16; ++i) CHECK(p[i] == 0.0f);
}

TEST_CASE("tensor: view is immutable through the API") {
    alignas(64) float external[8]{1, 2, 3, 4, 5, 6, 7, 8};
    Tensor v = Tensor::view(DataType::F32, {8}, external);

    CHECK(v.storage() == Tensor::Storage::View);
    CHECK(v.data() == external);
    CHECK(v.data_mut() == nullptr);          // Views refuse write access
    CHECK(v.as<float>()[3] == 4.0f);
}

TEST_CASE("tensor: move transfers ownership without leak") {
    Tensor a = Tensor::allocate(DataType::F32, {4});
    const void* p = a.data();
    Tensor b = std::move(a);
    CHECK(b.data() == p);
    CHECK(a.data() == nullptr);
    CHECK(a.size_bytes() == 0u);
    CHECK(a.storage() == Tensor::Storage::View);   // moved-from resets to safe state
}

TEST_CASE("tensor: F16 tensor byte-count is elements*2") {
    Tensor t = Tensor::allocate(DataType::F16, {10, 10});
    CHECK(t.size_bytes() == 200u);
}
