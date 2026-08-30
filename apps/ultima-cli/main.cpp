#include "ultima/core/version.hpp"

#include <cstdio>
#include <cstring>
#include <string_view>

namespace {

int print_version() {
    std::printf("ultima %s\n", ultima::core::version());
    return 0;
}

int print_help() {
    std::printf(
        "ultima - independent local LLM runtime\n"
        "\n"
        "Usage:\n"
        "  ultima --version              print version and exit\n"
        "  ultima --help                 print this help and exit\n"
        "\n"
        "Runtime commands (Decision 04+):\n"
        "  ultima -m <model.gguf> -p <prompt>\n"
        "  ultima serve --port 7777\n"
        "  ultima models list | download | use | verify | remove\n"
    );
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        return print_help();
    }

    const std::string_view arg{argv[1]};

    if (arg == "--version" || arg == "-v") {
        return print_version();
    }
    if (arg == "--help" || arg == "-h") {
        return print_help();
    }

    std::fprintf(stderr, "ultima: unknown argument '%s'\n", argv[1]);
    std::fprintf(stderr, "Run 'ultima --help' for usage.\n");
    return 2;
}
