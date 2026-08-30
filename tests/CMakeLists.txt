include(CTest)
include("${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake")

# ---- ultima_test — one binary per broad area, sanity test for now -----------
add_executable(ultima_test
    unit/test_sanity.cpp
    unit/test_version.cpp
)

target_link_libraries(ultima_test
    PRIVATE
        ultima_core
        ultima_warnings
        doctest::doctest
)

doctest_discover_tests(ultima_test)
