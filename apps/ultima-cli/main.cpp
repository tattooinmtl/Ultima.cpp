#include "ultima/core/error.hpp"
#include "ultima/core/version.hpp"
#include "ultima/kv_cache/kv_cache.hpp"
#include "ultima/model/arch_registry.hpp"
#include "ultima/model/i_model_loader.hpp"
#include "ultima/model/qwen_config.hpp"
#include "ultima/model/qwen_tensor_slots.hpp"
#include "ultima/model/imodel.hpp"
#include "ultima/model/metadata_store.hpp"
#include "ultima/runtime/thread_pool.hpp"
#include "ultima/sampler/sampler.hpp"
#include "ultima/server/http_server.hpp"
#include "ultima/tokenizer/tokenizer.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

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
        "  ultima --inspect <path.gguf> [filter]         parse a GGUF file, print metadata + tensors\n"
        "                                                filter: case-insensitive substring, shows all matches\n"
        "  ultima archs                                  list known architectures and their status\n"
        "  ultima scan <dir>                             report which GGUFs in <dir> Ultima can run\n"
        "  ultima run <path.gguf> [--prompt S] [--n-predict N]  one-shot completion, then exit\n"
        "  ultima serve <path.gguf> [--port N] [--host H] [--workspace DIR] [--permissions auto|ask|yolo]  HTTP API + Coder IDE\n"
        "\n"
        "Defaults for `run`:\n"
        "  --prompt \"def fibonacci(n):\"                greedy (--temp 0) so runs are reproducible\n"
        "  --n-predict 48    --n-ctx 512\n"
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

// Short numeric arrays are almost always structural parameters — MRoPE
// dimension sections, per-layer flags — so print them rather than just their
// length. Long ones stay summarised; nobody needs 28672 Hadamard signs inline.
std::string format_array_value(const ultima::model::LoadedModel& model,
                               const ultima::model::MetadataValue& v) {
    using namespace ultima::model;
    const auto* a = std::get_if<MetadataArrayView>(&v);
    if (!a || a->count == 0 || a->count > 16) return format_value(v);

    std::vector<std::string> parts;
    switch (a->element_type) {
    case 4: case 5: case 10: case 11: {           // u32 / i32 / u64 / i64
        for (auto x : read_i32_array(model, *a)) parts.push_back(fmt::format("{}", x));
        break;
    }
    case 6: {                                     // f32
        for (auto x : read_f32_array(model, *a)) parts.push_back(fmt::format("{}", x));
        break;
    }
    case 0: case 1: case 7: {                     // u8 / i8 / bool
        for (auto x : read_u8_array(model, *a)) parts.push_back(fmt::format("{}", x));
        break;
    }
    default: return format_value(v);
    }
    if (parts.empty()) return format_value(v);
    return "[" + fmt::format("{}", fmt::join(parts, ", ")) + "]";
}

std::string to_lower(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    return out;
}

// `filter` is an optional case-insensitive substring. When given, every
// matching key and tensor is printed with no truncation — the default
// 30-key/12-tensor summary is for orientation, but tracking down one
// architecture's parameters means seeing all of them.
int inspect(const char* path_cstr, const char* filter_cstr) {
    const std::string filter = filter_cstr ? to_lower(filter_cstr) : std::string{};
    const bool filtering = !filter.empty();
    auto matches = [&](const std::string& s) {
        return !filtering || to_lower(s).find(filter) != std::string::npos;
    };
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
    std::size_t printed = 0, skipped = 0;
    for (const auto& k : keys) {
        if (!matches(k)) { ++skipped; continue; }
        if (!filtering && printed >= 30) {
            std::printf("  ... (%zu more; pass a filter to see them)\n",
                        keys.size() - printed);
            break;
        }
        const auto* v = md.find(k);
        std::printf("  %-40s = %s\n", k.c_str(), format_array_value(model, *v).c_str());
        ++printed;
    }
    if (filtering) {
        std::printf("  (%zu matched '%s', %zu hidden)\n", printed, filter.c_str(), skipped);
    }
    std::printf("\n");

    const auto& tensors = model.tensor_infos();
    std::printf("Tensors (%zu total):\n", tensors.size());
    std::size_t t_printed = 0;
    for (const auto& t : tensors) {
        if (!matches(t.name)) continue;
        if (!filtering && t_printed >= 12) {
            std::printf("  ... (%zu more; pass a filter to see them)\n",
                        tensors.size() - t_printed);
            break;
        }
        // Name the raw GGUF id for a type we do not know, so an unsupported
        // file can be identified rather than only rejected.
        char dts[32];
        if (t.dtype == ultima::model::DataType::Unknown) {
            std::snprintf(dts, sizeof(dts), "id:%u", t.raw_dtype);
        } else {
            std::snprintf(dts, sizeof(dts), "%s", ultima::model::to_string(t.dtype));
        }
        std::printf("  %-40s %-16s %-8s %s\n",
                    t.name.c_str(),
                    format_dims(t.dims).c_str(),
                    dts,
                    format_bytes(t.size_bytes).c_str());
        ++t_printed;
    }
    if (filtering) {
        std::printf("  (%zu matched '%s')\n", t_printed, filter.c_str());
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
    std::string   workspace = ".";   // Coder IDE file root (Decision 18)
    std::string   permissions = "auto";  // auto | ask | yolo (PERMISSIONS.md)

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
        else if (a == "--workspace")  workspace = next("--workspace");
        else if (a == "--permissions") permissions = next("--permissions");
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
    cfg.workspace_root = workspace;
    cfg.permission_mode = permissions;

    std::printf("\nServing on http://%s:%u/  (testpad at /)\n", host.c_str(), port);
    ultima::server::HttpServer srv{std::move(ctx), cfg};
    auto ok = srv.listen_blocking();
    if (!ok) {
        std::fprintf(stderr, "server failed: %s\n", ok.error().message.c_str());
        return 1;
    }
    return 0;
}

// ---- archs ------------------------------------------------------------------

int list_archs() {
    std::printf("%-14s  %-11s  %s\n", "ARCHITECTURE", "STATUS", "NOTES");
    std::printf("%-14s  %-11s  %s\n", "------------", "-----------", "-----");
    for (const auto& a : ultima::model::all_archs()) {
        std::printf("%-14.*s  %-11s  %.*s\n",
                    static_cast<int>(a.name.size()), a.name.data(),
                    a.supported ? "runnable" : "not yet",
                    static_cast<int>(a.notes.size()), a.notes.data());
    }
    return 0;
}

// ---- archinfo ---------------------------------------------------------------

// Run the two load-time steps that come before any arithmetic — config
// parsing and tensor resolution — and print what they produced. An
// architecture under construction reaches this long before `run` works, so
// this is how its layout gets checked against the reference.
int archinfo(const char* path_cstr) {
    using namespace ultima::model;

    auto loader = make_gguf_loader();
    auto model_r = loader->load(std::filesystem::path{path_cstr});
    if (!model_r) {
        const auto& e = model_r.error();
        std::fprintf(stderr, "ultima archinfo: %s\n", e.message.c_str());
        return 1;
    }
    const auto& gguf = **model_r;

    const auto arch_v = gguf.metadata().get<std::string>("general.architecture");
    if (!arch_v) {
        std::fprintf(stderr, "ultima archinfo: no general.architecture\n");
        return 1;
    }
    const std::string arch = *arch_v;
    const ArchSpec* spec = find_arch(arch);
    if (!spec) {
        std::fprintf(stderr, "ultima archinfo: '%s' is not in the registry\n", arch.c_str());
        return 1;
    }

    ModelDims  d{};
    RopeConfig rp{};
    ArchParams p{};
    if (auto r = parse_arch_config(gguf, *spec, arch, d, rp, p); !r) {
        std::fprintf(stderr, "ultima archinfo: config failed: %s\n", r.error().message.c_str());
        return 1;
    }

    std::printf("Architecture: %s  (%s)\n", arch.c_str(),
                spec->supported ? "runnable" : "not runnable yet");
    std::printf("  layers %zu   hidden %zu   ffn %zu   vocab %zu   ctx_train %zu\n",
                d.n_layers, d.hidden, d.ffn_hidden, d.vocab, d.n_ctx_train);
    std::printf("  heads %zu   kv_heads %zu   head_dim %zu   q_dim %zu   kv_dim %zu\n",
                d.n_heads, d.n_kv_heads, d.head_dim, d.q_dim, d.kv_dim);
    std::printf("  rope: dim %zu of %zu   base %g\n", rp.rope_dim, d.head_dim, rp.freq_base);

    if (p.delta.active()) {
        const auto& dn = p.delta;
        std::printf("  delta-net: d_conv %zu  d_inner %zu  d_state %zu  "
                    "k_heads %zu  v_heads %zu\n",
                    dn.d_conv, dn.d_inner, dn.d_state, dn.n_k_heads, dn.n_v_heads);
        std::printf("             key_dim %zu  value_dim %zu  head_v_dim %zu  conv_dim %zu\n",
                    dn.key_dim(), dn.value_dim(), dn.head_v_dim(), dn.conv_dim());
        const std::size_t n_attn = p.n_attention_layers(d.n_layers);
        std::printf("  layer mix: %zu recurrent, %zu attention (KV only for those)\n",
                    d.n_layers - n_attn, n_attn);
        std::printf("             ");
        for (std::size_t i = 0; i < d.n_layers; ++i) {
            std::printf("%c", p.is_recurrent_layer(i) ? 'r' : 'A');
        }
        std::printf("\n");
    }

    auto t = resolve_model_tensors(gguf, arch, d, p);
    if (!t) {
        std::fprintf(stderr, "  TENSORS FAILED: %s\n", t.error().message.c_str());
        return 1;
    }
    std::size_t resolved = 3;   // token_embd, output_norm, output_head
    for (const auto& b : t->blocks) {
        for (const WeightRef* w : {&b.attn_norm, &b.attn_q, &b.attn_k, &b.attn_v,
                                   &b.attn_qkv, &b.attn_out, &b.attn_gate, &b.ffn_norm,
                                   &b.ffn_gate, &b.ffn_up, &b.ffn_down,
                                   &b.ffn_gate_inp, &b.ffn_gate_exps, &b.ffn_up_exps,
                                   &b.ffn_down_exps,
                                   &b.ssm_in, &b.ssm_gate, &b.ssm_conv1d, &b.ssm_dt_bias,
                                   &b.ssm_a, &b.ssm_beta, &b.ssm_alpha, &b.ssm_norm,
                                   &b.ssm_out}) {
            resolved += w->valid();
        }
        resolved += b.attn_q_norm.has_value() + b.attn_k_norm.has_value();
        resolved += b.attn_q_bias.has_value() + b.attn_k_bias.has_value()
                  + b.attn_v_bias.has_value();
    }
    std::printf("  tensors: all required slots resolved (%zu of %zu in file)\n",
                resolved, gguf.tensor_infos().size());
    return 0;
}

// ---- scan -------------------------------------------------------------------

// Report, for every GGUF in a directory, whether Ultima can run it and — when
// it cannot — precisely what is missing. This is the "can we read all of
// these?" answer, and it never needs the weights, only the header.
int scan(const char* dir_path) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (!fs::is_directory(dir_path, ec)) {
        std::fprintf(stderr, "ultima scan: '%s' is not a directory\n", dir_path);
        return 2;
    }

    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(dir_path, ec)) {
        if (!e.is_regular_file(ec)) continue;
        auto p = e.path();
        auto ext = p.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext == ".gguf") files.push_back(p);
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        std::printf("No .gguf files found in %s\n", dir_path);
        return 0;
    }

    auto loader = ultima::model::make_gguf_loader();
    std::size_t runnable = 0;

    for (const auto& p : files) {
        const std::string name = p.filename().string();
        std::printf("%s\n", name.c_str());

        auto loaded_r = loader->load(p);
        if (!loaded_r) {
            const auto& e = loaded_r.error();
            std::printf("    UNREADABLE  [%s] %s\n",
                        ultima::core::to_string(e.code), e.message.c_str());
            continue;
        }
        auto& loaded = **loaded_r;

        // Quant mix, most common first.
        std::map<std::string, std::size_t> types;
        for (const auto& t : loaded.tensor_infos()) types[ultima::model::to_string(t.dtype)]++;
        std::string mix;
        for (const auto& [k, v] : types) {
            if (!mix.empty()) mix += ", ";
            mix += fmt::format("{}x{}", k, v);
        }

        std::printf("    arch=%-12s tensors=%-5zu  %s\n",
                    loaded.architecture().c_str(),
                    loaded.tensor_infos().size(),
                    format_bytes(loaded.file_size_bytes()).c_str());
        std::printf("    quants: %s\n", mix.c_str());

        const auto* spec = ultima::model::find_arch(loaded.architecture());
        if (!spec) {
            std::printf("    NOT RUNNABLE  architecture is not in the registry\n");
            continue;
        }
        if (!spec->supported) {
            std::printf("    NOT RUNNABLE  needs: %.*s\n",
                        static_cast<int>(spec->notes.size()), spec->notes.data());
            continue;
        }

        // Architecture is runnable in principle — confirm this file actually
        // resolves (config keys present, tensor shapes as expected, dtypes
        // covered) rather than claiming support we have not checked.
        ultima::runtime::ThreadPool pool{1};
        auto model_r = ultima::model::load_model(loaded, pool);
        if (!model_r) {
            std::printf("    NOT RUNNABLE  %s\n", model_r.error().message.c_str());
            continue;
        }
        const auto& d = (*model_r)->dims();
        std::printf("    RUNNABLE      %zu layers, hidden %zu, %zu heads "
                    "(KV %zu, head_dim %zu), vocab %zu, train ctx %zu\n",
                    d.n_layers, d.hidden, d.n_heads, d.n_kv_heads,
                    d.head_dim, d.vocab, d.n_ctx_train);
        ++runnable;
    }

    std::printf("\n%zu of %zu file(s) runnable.\n", runnable, files.size());
    return 0;
}

// ---- run --------------------------------------------------------------------

int run(int argc, char** argv) {
    // argv[0] = "ultima", argv[1] = "run", argv[2] = <path.gguf>, then flags.
    if (argc < 3) {
        std::fprintf(stderr, "ultima run: missing GGUF path\n");
        return 2;
    }
    std::filesystem::path model_path{argv[2]};
    std::string prompt     = "def fibonacci(n):";
    std::size_t n_predict  = 48;
    std::size_t n_ctx      = 512;
    std::size_t n_threads  = 0;
    float       temperature = 0.0f;   // greedy by default: reproducible checks
    std::uint64_t seed     = 0;
    bool        show_tokens = false;

    for (int i = 3; i < argc; ++i) {
        std::string_view a = argv[i];
        auto next = [&](const char* nm) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "ultima run: %s requires a value\n", nm);
                std::exit(2);
            }
            return argv[++i];
        };
        if      (a == "--prompt")     prompt     = next("--prompt");
        else if (a == "--n-predict")  n_predict  = static_cast<std::size_t>(std::stoi(next("--n-predict")));
        else if (a == "--n-ctx")      n_ctx      = static_cast<std::size_t>(std::stoi(next("--n-ctx")));
        else if (a == "--n-threads")  n_threads  = static_cast<std::size_t>(std::stoi(next("--n-threads")));
        else if (a == "--temp")       temperature = std::stof(next("--temp"));
        else if (a == "--seed")       seed       = static_cast<std::uint64_t>(std::stoull(next("--seed")));
        else if (a == "--show-tokens") show_tokens = true;
        else {
            std::fprintf(stderr, "ultima run: unknown flag '%.*s'\n",
                         static_cast<int>(a.size()), a.data());
            return 2;
        }
    }

    auto loader = ultima::model::make_gguf_loader();
    auto loaded_r = loader->load(model_path);
    if (!loaded_r) {
        const auto& e = loaded_r.error();
        std::fprintf(stderr, "load failed: [%s/%s] %s\n",
                     e.component.c_str(), ultima::core::to_string(e.code), e.message.c_str());
        return 1;
    }
    auto& loaded = **loaded_r;

    auto tok_r = ultima::tokenizer::BpeTokenizer::load(loaded);
    if (!tok_r) {
        std::fprintf(stderr, "tokenizer load failed: %s\n", tok_r.error().message.c_str());
        return 1;
    }
    auto& tok = **tok_r;

    ultima::runtime::ThreadPool pool{n_threads};
    auto model_r = ultima::model::load_model(loaded, pool);
    if (!model_r) {
        std::fprintf(stderr, "model init failed: %s\n", model_r.error().message.c_str());
        return 1;
    }
    auto& model = **model_r;
    const auto& d = model.dims();

    std::printf("arch=%s  layers=%zu hidden=%zu heads=%zu/%zu head_dim=%zu vocab=%zu\n",
                model.arch(), d.n_layers, d.hidden, d.n_heads, d.n_kv_heads,
                d.head_dim, d.vocab);

    ultima::kv_cache::KVCache::Config kvc;
    kvc.n_slots    = 1;
    kvc.n_ctx      = std::min(n_ctx, d.n_ctx_train);
    kvc.n_layers   = d.n_layers;
    kvc.n_kv_heads = d.n_kv_heads;
    kvc.head_dim   = d.head_dim;
    kvc.dtype      = ultima::model::DataType::F32;
    ultima::kv_cache::KVCache kv{kvc};
    const auto slot = kv.acquire(/*session=*/1);

    auto tokens = tok.encode(prompt);
    if (tokens.empty()) {
        std::fprintf(stderr, "prompt encoded to zero tokens\n");
        return 1;
    }
    if (tokens.size() + n_predict > kvc.n_ctx) {
        std::fprintf(stderr, "prompt (%zu) + n_predict (%zu) exceeds n_ctx (%zu)\n",
                     tokens.size(), n_predict, kvc.n_ctx);
        return 1;
    }

    if (show_tokens) {
        std::printf("tokens (%zu):", tokens.size());
        for (auto t : tokens) {
            std::printf(" [%d]%s", t,
                        tok.decode(std::span<const ultima::tokenizer::TokenId>{&t, 1}).c_str());
        }
        std::printf("\n");
    }

    std::vector<float> logits(d.vocab, 0.0f);

    const auto t_prefill_start = std::chrono::steady_clock::now();
    if (auto r = model.prefill(kv, slot, tokens, logits.data()); !r) {
        std::fprintf(stderr, "prefill failed: %s\n", r.error().message.c_str());
        return 1;
    }
    const auto t_prefill_end = std::chrono::steady_clock::now();

    ultima::sampler::SamplerParams sp = ultima::sampler::SamplerParams::identity();
    sp.temperature = temperature;
    ultima::sampler::SamplerContext sctx{seed};
    for (auto t : tokens) sctx.push_recent(t);

    std::printf("\n%s", prompt.c_str());
    std::fflush(stdout);

    std::size_t produced = 0;
    for (std::size_t i = 0; i < n_predict; ++i) {
        const ultima::tokenizer::TokenId next =
            (temperature <= 0.0f)
                ? ultima::sampler::argmax(logits)
                : ultima::sampler::sample(logits, sp, sctx);
        if (next == tok.eos_id() || next == tok.im_end_id()) break;

        std::printf("%s", tok.decode(std::span<const ultima::tokenizer::TokenId>{&next, 1}).c_str());
        std::fflush(stdout);
        sctx.push_recent(next);
        ++produced;

        if (auto r = model.decode(kv, slot, next, logits.data()); !r) {
            std::fprintf(stderr, "\ndecode failed: %s\n", r.error().message.c_str());
            return 1;
        }
    }
    const auto t_end = std::chrono::steady_clock::now();

    const double prefill_ms = std::chrono::duration<double, std::milli>(
        t_prefill_end - t_prefill_start).count();
    const double decode_ms = std::chrono::duration<double, std::milli>(
        t_end - t_prefill_end).count();
    std::printf("\n\n[%zu prompt tok in %.0f ms (%.1f tok/s) | "
                "%zu gen tok in %.0f ms (%.1f tok/s)]\n",
                tokens.size(), prefill_ms,
                prefill_ms > 0 ? static_cast<double>(tokens.size()) * 1000.0 / prefill_ms : 0.0,
                produced, decode_ms,
                decode_ms > 0 ? static_cast<double>(produced) * 1000.0 / decode_ms : 0.0);
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
        return inspect(argv[2], argc > 3 ? argv[3] : nullptr);
    }
    if (arg == "serve") {
        return serve(argc, argv);
    }
    if (arg == "run") {
        return run(argc, argv);
    }
    if (arg == "archs") {
        return list_archs();
    }
    if (arg == "archinfo") {
        if (argc < 3) {
            std::fprintf(stderr, "ultima archinfo: missing path\n");
            return 2;
        }
        return archinfo(argv[2]);
    }
    if (arg == "scan") {
        if (argc < 3) {
            std::fprintf(stderr, "ultima scan: missing directory\n");
            return 2;
        }
        return scan(argv[2]);
    }

    std::fprintf(stderr, "ultima: unknown argument '%s'\n", argv[1]);
    std::fprintf(stderr, "Run 'ultima --help' for usage.\n");
    return 2;
}
