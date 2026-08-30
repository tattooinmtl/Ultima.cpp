# Shared compiler / linker configuration applied via an INTERFACE target.
# Every ultima target links to `ultima_flags` to inherit these settings.

add_library(ultima_flags INTERFACE)

# C++20, no compiler extensions
target_compile_features(ultima_flags INTERFACE cxx_std_20)

# --- MSVC ---------------------------------------------------------------------
if(MSVC)
    target_compile_options(ultima_flags INTERFACE
        /permissive-        # standards-conformant
        /Zc:__cplusplus     # correct __cplusplus macro
        /Zc:preprocessor    # standards-conformant preprocessor
        /Zc:inline          # remove unreferenced COMDATs
        /EHsc               # C++ exceptions, extern "C" is nothrow
        /GR-                # disable RTTI (see Decision 02)
        /utf-8              # source and execution character set
        /MP                 # multi-processor compile
    )
    target_compile_definitions(ultima_flags INTERFACE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )

# --- GCC / Clang --------------------------------------------------------------
else()
    target_compile_options(ultima_flags INTERFACE
        -fno-rtti
        -fno-exceptions     # exceptions allowed only via project-scoped opt-in
        -fvisibility=hidden
        -fvisibility-inlines-hidden
    )
endif()
