#include <doctest/doctest.h>

#include "ultima/tensor/aligned_alloc.hpp"

#include <cstdint>

TEST_CASE("aligned_alloc: 64-byte alignment") {
    void* p = ultima::tensor::aligned_alloc_bytes(1024);
    REQUIRE(p != nullptr);
    CHECK((reinterpret_cast<std::uintptr_t>(p) & 63u) == 0u);
    ultima::tensor::aligned_free(p);
}

TEST_CASE("aligned_alloc: custom alignment") {
    void* p = ultima::tensor::aligned_alloc_bytes(1024, 128);
    REQUIRE(p != nullptr);
    CHECK((reinterpret_cast<std::uintptr_t>(p) & 127u) == 0u);
    ultima::tensor::aligned_free(p);
}

TEST_CASE("aligned_calloc: zero-initialized") {
    constexpr std::size_t N = 256;
    auto* p = static_cast<std::uint8_t*>(ultima::tensor::aligned_calloc_bytes(N));
    REQUIRE(p != nullptr);
    for (std::size_t i = 0; i < N; ++i) {
        CHECK(p[i] == 0);
    }
    ultima::tensor::aligned_free(p);
}

TEST_CASE("aligned_free: nullptr is safe") {
    ultima::tensor::aligned_free(nullptr);
    CHECK(true);
}

TEST_CASE("aligned_alloc_n: typed helper") {
    auto up = ultima::tensor::aligned_alloc_n<float>(1024);
    REQUIRE(up.get() != nullptr);
    CHECK((reinterpret_cast<std::uintptr_t>(up.get()) & 63u) == 0u);
    up[0] = 1.0f;
    up[1023] = 42.0f;
    CHECK(up[0] == 1.0f);
    CHECK(up[1023] == 42.0f);
}
