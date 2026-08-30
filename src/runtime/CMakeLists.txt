add_library(ultima_runtime STATIC
    thread_pool.cpp
)

target_include_directories(ultima_runtime
    PUBLIC "${CMAKE_SOURCE_DIR}/include"
)

target_link_libraries(ultima_runtime
    PUBLIC
        ultima_core
        ultima_flags
    PRIVATE
        ultima_warnings
)
