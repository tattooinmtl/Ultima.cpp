#include <doctest/doctest.h>

#include "ultima/core/version.hpp"

#include <cstring>

TEST_CASE("version: numeric constants are populated") {
    CHECK(ultima::core::version_major == 0);
    CHECK(ultima::core::version_minor == 1);
    CHECK(ultima::core::version_patch == 0);
}

TEST_CASE("version: string is non-empty and starts with '0.1'") {
    const char* v = ultima::core::version();
    REQUIRE(v != nullptr);
    CHECK(std::strlen(v) > 0);
    CHECK(std::strncmp(v, "0.1", 3) == 0);
}
