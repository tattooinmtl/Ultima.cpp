# Warnings-as-errors preset. Attach to a target with:
#   target_link_libraries(<target> PRIVATE ultima_warnings)

add_library(ultima_warnings INTERFACE)

if(MSVC)
    target_compile_options(ultima_warnings INTERFACE
        /W4
        /WX
        /wd4324     # padding due to alignas — expected for SIMD types
    )
else()
    target_compile_options(ultima_warnings INTERFACE
        -Wall
        -Wextra
        -Wpedantic
        -Werror
        -Wconversion
        -Wshadow
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Wcast-align
        -Wunused
        -Woverloaded-virtual
        -Wdouble-promotion
        -Wformat=2
    )
endif()
