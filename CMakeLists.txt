cmake_minimum_required(VERSION 3.24)

project(ultima
    VERSION      0.1.0
    DESCRIPTION  "Independent C++ inference runtime with persistent-memory subsystem"
    LANGUAGES    CXX
)

# ---- Language ----------------------------------------------------------------
set(CMAKE_CXX_STANDARD          20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)

# ---- Position of built binaries ---------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# ---- Module path & shared config --------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(UltimaCompilerFlags)
include(UltimaWarnings)
include(UltimaDeps)

# ---- Version header generation ----------------------------------------------
configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/version.hpp.in"
    "${CMAKE_BINARY_DIR}/generated/ultima/core/version_generated.hpp"
    @ONLY
)

# ---- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
add_subdirectory(apps/ultima-cli)

# ---- Tests -------------------------------------------------------------------
option(ULTIMA_BUILD_TESTS "Build unit and integration tests" ON)
if(ULTIMA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---- Status banner -----------------------------------------------------------
message(STATUS "")
message(STATUS "Ultima.cpp ${PROJECT_VERSION}")
message(STATUS "  C++ standard : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type   : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler     : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Tests        : ${ULTIMA_BUILD_TESTS}")
message(STATUS "")
