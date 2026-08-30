add_library(ultima_tokenizer STATIC
    bpe_tokenizer.cpp
    chat_template.cpp
)

target_include_directories(ultima_tokenizer
    PUBLIC "${CMAKE_SOURCE_DIR}/include"
)

target_link_libraries(ultima_tokenizer
    PUBLIC
        ultima_core
        ultima_model
        ultima_flags
    PRIVATE
        ultima_warnings
        fmt::fmt
)
