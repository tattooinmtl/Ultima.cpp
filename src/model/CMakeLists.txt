add_library(ultima_model STATIC
    dtype.cpp
    tensor_info.cpp
    metadata_store.cpp
    gguf_reader.cpp
    gguf_loader.cpp
    mmap_file_windows.cpp
    qwen_config.cpp
    qwen_tensor_slots.cpp
    weight_dispatch.cpp
    qwen_block.cpp
    qwen_model.cpp
)

target_include_directories(ultima_model
    PUBLIC
        "${CMAKE_SOURCE_DIR}/include"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(ultima_model
    PUBLIC
        ultima_core
        ultima_flags
        ultima_kernels
        ultima_kv_cache
        ultima_runtime
        nonstd::expected-lite
    PRIVATE
        ultima_warnings
        fmt::fmt
)
