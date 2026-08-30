# --- Bundle ui/testpad/index.html into a generated .cpp at configure time ---
set(TESTPAD_HTML "${CMAKE_SOURCE_DIR}/ui/testpad/index.html")
set(TESTPAD_GEN  "${CMAKE_CURRENT_BINARY_DIR}/testpad_bundle_generated.cpp")

# Convert the HTML file to a C++ string_view constant. Using file(READ) +
# configure_file dodges any file-embedding CMake version dance.
if(EXISTS "${TESTPAD_HTML}")
    file(READ "${TESTPAD_HTML}" TESTPAD_RAW HEX)
    # Split hex into 2-char groups joined by ",0x".
    string(REGEX REPLACE "(..)" "0x\\1," TESTPAD_C_ARRAY "${TESTPAD_RAW}")
    file(WRITE "${TESTPAD_GEN}"
"// GENERATED — do not edit. Source: ui/testpad/index.html
#define ULTIMA_TESTPAD_HTML_EMBEDDED 1
#include <string_view>
namespace ultima::server {
static const unsigned char kTestpadBytes[] = { ${TESTPAD_C_ARRAY} 0x00 };
extern const std::string_view kTestpadHtml;
const std::string_view kTestpadHtml{
    reinterpret_cast<const char*>(kTestpadBytes),
    sizeof(kTestpadBytes) - 1
};
} // namespace ultima::server
")
else()
    message(WARNING "Testpad HTML not found at ${TESTPAD_HTML}; server will serve a placeholder page")
    file(WRITE "${TESTPAD_GEN}"
"// GENERATED — placeholder (source HTML missing)
#include <string_view>
namespace ultima::server {
extern const std::string_view kTestpadHtml;
const std::string_view kTestpadHtml = \"<!doctype html><body>testpad HTML missing at build time</body>\";
} // namespace ultima::server
")
endif()

add_library(ultima_server STATIC
    http_server.cpp
    "${TESTPAD_GEN}"
)

target_include_directories(ultima_server
    PUBLIC "${CMAKE_SOURCE_DIR}/include"
)

target_link_libraries(ultima_server
    PUBLIC
        ultima_core
        ultima_flags
        ultima_model
        ultima_kv_cache
        ultima_sampler
        ultima_tokenizer
        ultima_runtime
        nlohmann_json::nlohmann_json
    PRIVATE
        ultima_warnings
)

# Consume cpp-httplib's include path directly rather than linking to its
# INTERFACE target — the target propagates a CPPHTTPLIB_OPENSSL_SUPPORT
# define whenever find_package(OpenSSL) succeeds on the host, which pulls
# <openssl/*> into our TU. Adding only the include dir keeps HTTP-only
# behavior (v0.1 is 127.0.0.1 by default; TLS is a v0.2 concern).
FetchContent_GetProperties(httplib SOURCE_DIR HTTPLIB_SRC_DIR)
target_include_directories(ultima_server SYSTEM PRIVATE "${HTTPLIB_SRC_DIR}")

# httplib links against Threads::Threads when built via its own CMake; since
# we're bypassing that target, add the platform threads dep ourselves.
find_package(Threads REQUIRED)
target_link_libraries(ultima_server PRIVATE Threads::Threads)
