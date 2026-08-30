#include <doctest/doctest.h>

#include "ultima/kernels/cpu_features.hpp"

TEST_CASE("cpu_features: detection returns something sensible") {
    const auto& f = ultima::kernels::cpu_features();
    // Any x86-64 CPU built after ~2014 has AVX2 + FMA. We only build for x86-64.
    CHECK(f.has_sse2);
    CHECK(f.has_avx2);
    CHECK(f.has_fma);
    CHECK(f.logical_cores >= 1u);
    CHECK_FALSE(f.cpu_brand.empty());
}

TEST_CASE("cpu_features: same reference across calls (cached)") {
    const auto* p1 = &ultima::kernels::cpu_features();
    const auto* p2 = &ultima::kernels::cpu_features();
    CHECK(p1 == p2);
}
