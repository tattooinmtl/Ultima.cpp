add_library(ultima_kv_cache STATIC
    kv_cache.cpp
)

target_include_directories(ultima_kv_cache
    PUBLIC "${CMAKE_SOURCE_DIR}/include"
)

target_link_libraries(ultima_kv_cache
    PUBLIC
        ultima_core
        ultima_model
        ultima_flags
    PRIVATE
        ultima_warnings
)
