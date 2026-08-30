add_executable(ultima
    main.cpp
)

set_target_properties(ultima PROPERTIES
    OUTPUT_NAME ultima
)

target_link_libraries(ultima
    PRIVATE
        ultima_core
        ultima_model
        ultima_warnings
        fmt::fmt
)
