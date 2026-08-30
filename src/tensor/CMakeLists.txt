add_library(ultima_tensor STATIC
    aligned_alloc.cpp
    tensor.cpp
)

target_include_directories(ultima_tensor
    PUBLIC "${CMAKE_SOURCE_DIR}/include"
)

target_link_libraries(ultima_tensor
    PUBLIC
        ultima_core
        ultima_model
        ultima_flags
    PRIVATE
        ultima_warnings
)
