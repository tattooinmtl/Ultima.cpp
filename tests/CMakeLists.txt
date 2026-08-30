include(CTest)
include("${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake")

# ---- ultima_test — one binary per broad area, sanity test for now -----------
add_executable(ultima_test
    unit/test_sanity.cpp
    unit/test_version.cpp
    unit/test_gguf_loader.cpp
    unit/test_aligned_alloc.cpp
    unit/test_tensor.cpp
    unit/test_thread_pool.cpp
    unit/test_cpu_features.cpp
    unit/test_matvec.cpp
    unit/test_elementwise.cpp
    unit/test_rmsnorm.cpp
    unit/test_softmax.cpp
    unit/test_rope.cpp
    unit/test_dequant_q4k.cpp
    unit/test_matvec_q4k.cpp
    unit/test_q8_0.cpp
    unit/test_q6k.cpp
    unit/test_kernels_correctness.cpp
    unit/test_tokenizer.cpp
)

target_link_libraries(ultima_test
    PRIVATE
        ultima_core
        ultima_model
        ultima_tensor
        ultima_runtime
        ultima_kernels
        ultima_tokenizer
        ultima_warnings
        doctest::doctest
)

doctest_discover_tests(ultima_test)
