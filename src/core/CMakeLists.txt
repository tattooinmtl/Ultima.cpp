add_library(ultima_core STATIC
    version.cpp
)

target_include_directories(ultima_core
    PUBLIC
        "${CMAKE_SOURCE_DIR}/include"
        "${CMAKE_BINARY_DIR}/generated"
)

target_link_libraries(ultima_core
    PUBLIC  ultima_flags
    PRIVATE ultima_warnings
)
