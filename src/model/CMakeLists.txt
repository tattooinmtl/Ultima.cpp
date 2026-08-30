add_library(ultima_model STATIC
    dtype.cpp
    tensor_info.cpp
    metadata_store.cpp
    gguf_reader.cpp
    gguf_loader.cpp
    mmap_file_windows.cpp
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
        nonstd::expected-lite
    PRIVATE
        ultima_warnings
        fmt::fmt
)
