add_library(ultima_kernels STATIC
    cpu_features.cpp
    matvec_f32_f32.cpp
    matvec_q4k_f32.cpp
    matvec_q8_0_f32.cpp
    dequant_q4k.cpp
    dequant_q8_0.cpp
    elementwise.cpp
    rmsnorm.cpp
    softmax.cpp
    rope.cpp
)

target_include_directories(ultima_kernels
    PUBLIC "${CMAKE_SOURCE_DIR}/include"
)

target_link_libraries(ultima_kernels
    PUBLIC
        ultima_core
        ultima_flags
    PRIVATE
        ultima_warnings
)

# AVX2 + FMA required for kernel translation units only. This flag does NOT
# leak to consumers — general code still compiles at the project baseline.
if(MSVC)
    target_compile_options(ultima_kernels PRIVATE /arch:AVX2)
else()
    target_compile_options(ultima_kernels PRIVATE -mavx2 -mfma)
endif()
