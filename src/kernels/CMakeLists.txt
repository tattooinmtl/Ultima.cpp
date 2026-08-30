add_library(ultima_kernels STATIC
    cpu_features.cpp
    matvec_f32_f32.cpp
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
