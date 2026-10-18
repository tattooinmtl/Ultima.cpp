# All third-party dependencies. Each is fetched via FetchContent, pinned to a
# specific tag, marked SYSTEM so their warnings do not pollute our build, and
# excluded from the ALL target so their own tests/examples do not build.

include(FetchContent)

# CMake 4.x removed compatibility with pre-3.5 cmake_minimum_required. Many
# widely-used deps still declare older minimums. This tells CMake to accept
# them under 3.5 policy behavior. Scoped to this file's FetchContent block.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)

# ---- doctest — unit test framework ------------------------------------------
FetchContent_Declare(doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG        v2.4.11
    GIT_SHALLOW    ON
    SYSTEM
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(doctest)

# ---- fmt — string formatting (bridge until std::format is universal) --------
FetchContent_Declare(fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        11.1.1
    GIT_SHALLOW    ON
    SYSTEM
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(fmt)

# ---- expected-lite — C++20-compatible std::expected ------------------------
FetchContent_Declare(expected_lite
    GIT_REPOSITORY https://github.com/martinmoene/expected-lite.git
    GIT_TAG        v0.9.0
    GIT_SHALLOW    ON
    SYSTEM
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(expected_lite)

# ---- cpp-httplib — header-only HTTP server ---------------------------------
# Decision 14: cpp-httplib chosen over Crow/Drogon/Beast. We consume the
# single-header directly. Disable optional TLS/zlib/brotli integrations so
# a bare Windows / Linux install without OpenSSL/zlib/brotli builds clean.
set(HTTPLIB_REQUIRE_OPENSSL OFF CACHE INTERNAL "")
set(HTTPLIB_REQUIRE_ZLIB    OFF CACHE INTERNAL "")
set(HTTPLIB_REQUIRE_BROTLI  OFF CACHE INTERNAL "")
set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF CACHE INTERNAL "")
set(HTTPLIB_USE_ZLIB_IF_AVAILABLE    OFF CACHE INTERNAL "")
set(HTTPLIB_USE_BROTLI_IF_AVAILABLE  OFF CACHE INTERNAL "")
FetchContent_Declare(httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.15.3
    GIT_SHALLOW    ON
    SYSTEM
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(httplib)

# ---- nlohmann/json — JSON parser + serializer ------------------------------
# OpenAI-compatible request/response bodies need a JSON layer. Single-header,
# MIT, standard choice for C++ servers.
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install    OFF CACHE INTERNAL "")
FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    ON
    SYSTEM
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(nlohmann_json)
