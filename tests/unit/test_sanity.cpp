#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

TEST_CASE("sanity: 1 + 1 == 2") {
    CHECK(1 + 1 == 2);
}

TEST_CASE("sanity: doctest string matcher works") {
    const char* greeting = "hello ultima";
    CHECK(doctest::String(greeting) == doctest::String("hello ultima"));
}
