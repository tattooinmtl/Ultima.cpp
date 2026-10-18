add_library(ultima_sampler STATIC
    sampler.cpp
)

target_include_directories(ultima_sampler
    PUBLIC "${CMAKE_SOURCE_DIR}/include"
)

target_link_libraries(ultima_sampler
    PUBLIC
        ultima_core
        ultima_flags
    PRIVATE
        ultima_warnings
)
