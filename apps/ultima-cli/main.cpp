#include "ultima/core/error.hpp"
#include "ultima/core/version.hpp"
#include "ultima/kv_cache/kv_cache.hpp"
#include "ultima/model/i_model_loader.hpp"
#include "ultima/model/imodel.hpp"
#include "ultima/model/metadata_store.hpp"
#include "ultima/runtime/thread_pool.hpp"
#include "ultima/server/http_server.hpp"
#include "ultima/tokenizer/tokenizer.hpp"

#include <fmt/format.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>
#include <variant>

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
        "  ultima --version                              print version and exit\n"
        "  ultima --help                                 print this help and exit\n"
        "  ultima --inspect <path.gguf>                  parse a GGUF file, print metadata + tensors\n"
        "  ultima serve <path.gguf> [--port N] [--host H]  load a Qwen2/Qwen3 GGUF and serve OpenAI-compatible HTTP\n"
        "\n"
        "Defaults for `serve`:\n"
        "  --host 127.0.0.1                              (LAN mode not enabled in v0.1)\n"
        "  --port 11434                                  (Ollama-mnemonic; existing clients Just Work)\n"
        "  --n-ctx 4096                                  (KV cache context length per slot)\n"
        "  --n-threads = physical cores                  (from cpu_features)\n"
        "\n"
        "Testpad UI is served at http://<host>:<port>/ .\n"
    );
    return 0;
}

std::string format_bytes(std::uint64_t n) {
    constexpr std::uint64_t kib = 1024;
    constexpr std::uint64_t mib = kib * 1024;
    constexpr std::uint64_t gib = mib * 1024;
    if (n >= gib) return fmt::format("{:.2f} GiB", static_cast<double>(n) / static_cast<double>(gib));
    if (n >= mib) return fmt::format("{:.2f} MiB", static_cast<double>(n) / static_cast<double>(mib));
    if (n >= kib) return fmt::format("{:.2f} KiB", static_cast<double>(n) / static_cast<double>(kib));
    return fmt::format("{} B", n);
}

std::string format_dims(const std::vector<std::uint64_t>& dims) {
    std::string out = "[";
    for (std::size_t i = 0; i < dims.size(); ++i) {
        out += fmt::format("{}", dims[i]);
        if (i + 1 < dims.size()) out += ", ";
    }
    out += "]";
    return out;
}

std::string format_value(const ultima::model::MetadataValue& v) {
    using namespace ultima::model;
    struct Visitor {
        std::string operator()(std::monostate) const                     { return "(null)"; }
        std::string operator()(bool b) const                             { return b ? "true" : "false"; }
        std::string operator()(std::uint8_t x)  const                    { return fmt::format("{}", x); }
        std::string operator()(std::int8_t x)   const                    { return fmt::format("{}", x); }
        std::string operator()(std::uint16_t x) const                    { return fmt::format("{}", x); }
        std::string operator()(std::int16_t x)  const                    { return fmt::format("{}", x); }
        std::string operator()(std::uint32_t x) const                    { return fmt::format("{}", x); }
        std::string operator()(std::int32_t x)  const                    { return fmt::format("{}", x); }
        std::string operator()(std::uint64_t x) const                    { return fmt::format("{}", x); }
        std::string operator()(std::int64_t x)  const                    { return fmt::format("{}", x); }
        std::string operator()(float x)  const                           { return fmt::format("{}", x); }
        std::string operator()(double x) const                           { return fmt::format("{}", x); }
        std::string operator()(const std::string& s) const {
            if (s.size() > 80) return fmt::format("\"{}...\" ({} chars)", std::string_view{s}.substr(0, 77), s.size());
            return fmt::format("\"{}\"", s);
        }
        std::string operator()(const std::vector<std::string>& a) const {
            return fmt::format("<{} strings>", a.size());
        }
        std::string operator()(const MetadataArrayView& a) const {
            return fmt::format("<{} array, type {}>", a.count, a.element_type);
        }
    };
    return std::visit(Visitor{}, v);
}

int inspect(const char* path_cstr) {
    std::filesystem::path path{path_cstr};

    auto loader = ultima::model::make_gguf_loader();
    auto model_r = loader->load(path);
    if (!model_r) {
        const auto& e = model_r.error();
        std::fprintf(stderr, "ultima --inspect failed: [%s/%s] %s\n",
                     e.component.empty() ? "?" : e.component.c_str(),
                     ultima::core::to_string(e.code),
                     e.message.c_str());
        return 1;
    }
    const auto& model = **model_r;

    std::printf("File:         %s\n", path.filename().string().c_str());
    std::printf("Size:         %s\n", format_bytes(model.file_size_bytes()).c_str());
    std::printf("Alignment:    %llu bytes\n", static_cast<unsigned long long>(model.alignment()));
    std::printf("GGUF version: %u\n", model.gguf_version());
    std::printf("\n");

    const auto& md = model.metadata();
    std::printf("Metadata (%zu keys):\n", md.size());
    // Sort keys for stable output
    std::vector<std::string> keys;
    keys.reserve(md.all().size());
    for (const auto& kv : md.all()) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
    std::size_t printed = 0;
    for (const auto& k : keys) {
        if (printed >= 30) {
            std::printf("  ... (%zu more)\n", keys.size() - printed);
            break;
        }
        const auto* v = md.find(k);
        std::printf("  %-40s = %s\n", k.c_str(), format_value(*v).c_str());
        ++printed;
    }
    std::printf("\n");

    const auto& tensors = model.tensor_infos();
    std::printf("Tensors (%zu total):\n", tensors.size());
    const std::size_t show = std::min<std::size_t>(tensors.size(), 12);
    for (std::size_t i = 0; i < show; ++i) {
        const auto& t = tensors[i];
        std::printf("  %-40s %-16s %-6s %s\n",
                    t.name.c_str(),
                    format_dims(t.dims).c_str(),
                    ultima::model::to_string(t.dtype),
                    format_bytes(t.size_bytes).c_str());
    }
    if (tensors.size() > show) {
        std::printf("  ... (%zu more)\n", tensors.size() - show);
    }
    std::printf("\n");

    std::printf("Summary:\n");
    std::printf("  Architecture:      %s\n", model.architecture().c_str());
    std::uint64_t total_tensor_bytes = 0;
    for (const auto& t : tensors) total_tensor_bytes += t.size_bytes;
    std::printf("  Total tensor size: %s\n", format_bytes(total_tensor_bytes).c_str());

    if (auto ctx = md.get_uint("general.file_type"))
        std::printf("  file_type:         %llu\n", static_cast<unsigned long long>(*ctx));

    return 0;
}

int serve(int argc, char** argv) {
    // argv[0] = "ultima", argv[1] = "serve", argv[2] = <path.gguf>, then flags.
    if (argc < 3) {
        std::fprintf(stderr, "ultima serve: missing GGUF path\n");
        return 2;
    }
    std::filesystem::path model_path{argv[2]};
    std::string host   = "127.0.0.1";
    std::uint16_t port = 11434;
    std::size_t   n_ctx = 4096;
    std::size_t   n_threads = 0;   // 0 => pool default

    for (int i = 3; i < argc; ++i) {
        std::string_view a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "ultima serve: %s requires a value\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--host") host = next("--host");
        else if (a == "--port")      port     = static_cast<std::uint16_t>(std::stoi(next("--port")));
        else if (a == "--n-ctx")     n_ctx    = static_cast<std::size_t>(std::stoi(next("--n-ctx")));
        else if (a == "--n-threads") n_threads = static_cast<std::size_t>(std::stoi(next("--n-threads")));
        else {
            std::fprintf(stderr, "ultima serve: unknown flag '%.*s'\n",
                         static_cast<int>(a.size()), a.data());
            return 2;
        }
    }

    // 1. Load model.
    std::printf("Loading %s ...\n", model_path.string().c_str());
    auto loader = ultima::model::make_gguf_loader();
    auto loaded_r = loader->load(model_path);
    if (!loaded_r) {
        const auto& e = loaded_r.error();
        std::fprintf(stderr, "load failed: [%s/%s] %s\n",
                     e.component.c_str(), ultima::core::to_string(e.code), e.message.c_str());
        return 1;
    }
    auto& loaded = **loaded_r;
    std::printf("  architecture: %s\n", loaded.architecture().c_str());
    std::printf("  tensors:      %zu\n", loaded.tensor_infos().size());

    // 2. Tokenizer.
    auto tok_r = ultima::tokenizer::BpeTokenizer::load(loaded);
    if (!tok_r) {
        const auto& e = tok_r.error();
        std::fprintf(stderr, "tokenizer load failed: %s\n", e.message.c_str());
        return 1;
    }
    auto& tok = **tok_r;
    std::printf("  vocab:        %zu\n", tok.vocab_size());

    // 3. Model runtime.
    ultima::runtime::ThreadPool pool{n_threads};
    auto model_r = ultima::model::load_model(loaded, pool);
    if (!model_r) {
        const auto& e = model_r.error();
        std::fprintf(stderr, "model init failed: %s\n", e.message.c_str());
        return 1;
    }
    auto& model = **model_r;
    const auto& d = model.dims();
    std::printf("  n_layers:     %zu\n", d.n_layers);
    std::printf("  hidden:       %zu\n", d.hidden);
    std::printf("  n_heads:      %zu (KV %zu, head_dim %zu)\n",
                d.n_heads, d.n_kv_heads, d.head_dim);
    std::printf("  ffn_hidden:   %zu\n", d.ffn_hidden);
    std::printf("  n_ctx_train:  %zu (requested %zu)\n", d.n_ctx_train, n_ctx);

    // 4. KV cache (single slot for the CLI; server allocates its own multi-slot
    //    cache in production per Decision 09).
    ultima::kv_cache::KVCache::Config kvc;
    kvc.n_slots    = 1;
    kvc.n_ctx      = std::min(n_ctx, d.n_ctx_train);
    kvc.n_layers   = d.n_layers;
    kvc.n_kv_heads = d.n_kv_heads;
    kvc.head_dim   = d.head_dim;
    kvc.dtype      = ultima::model::DataType::F32;   // Q8_0 KV lands in v0.2
    ultima::kv_cache::KVCache kv{kvc};

    // 5. Server.
    ultima::server::RuntimeContext ctx{
        .model      = model,
        .tokenizer  = tok,
        .kv         = kv,
        .pool       = pool,
        .model_name = model_path.filename().string(),
    };
    ultima::server::ServerConfig cfg;
    cfg.bind_host = host;
    cfg.bind_port = port;

    std::printf("\nServing on http://%s:%u/  (testpad at /)\n", host.c_str(), port);
    ultima::server::HttpServer srv{std::move(ctx), cfg};
    auto ok = srv.listen_blocking();
    if (!ok) {
        std::fprintf(stderr, "server failed: %s\n", ok.error().message.c_str());
        return 1;
    }
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
    if (arg == "--inspect") {
        if (argc < 3) {
            std::fprintf(stderr, "ultima --inspect: missing path\n");
            return 2;
        }
        return inspect(argv[2]);
    }
    if (arg == "serve") {
        return serve(argc, argv);
    }

    std::fprintf(stderr, "ultima: unknown argument '%s'\n", argv[1]);
    std::fprintf(stderr, "Run 'ultima --help' for usage.\n");
    return 2;
}
