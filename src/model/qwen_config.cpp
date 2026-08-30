#include "ultima/model/qwen_config.hpp"

#include <fmt/format.h>

namespace ultima::model {

namespace {

using core::ErrorCode;
using core::Failure;
using core::Result;
using core::fail;

std::optional<std::uint64_t>
get_uint_any(const MetadataStore& md, const std::vector<std::string>& keys) {
    for (const auto& k : keys) {
        if (auto v = md.get_uint(k)) return v;
    }
    return std::nullopt;
}

std::optional<float>
get_float_any(const MetadataStore& md, const std::vector<std::string>& keys) {
    for (const auto& k : keys) {
        if (auto v = md.get<float>(k))  return v;
        if (auto v = md.get<double>(k)) return static_cast<float>(*v);
    }
    return std::nullopt;
}

} // namespace

Result<void>
parse_qwen_config(const LoadedModel& gguf,
                  const std::string& arch,
                  ModelDims&  out_dims,
                  RopeConfig& out_rope) {
    const auto& md = gguf.metadata();

    // Required ints. Each entry keeps its own std::vector<std::string> so
    // key strings outlive the request (initializer_list-of-string would
    // dangle after the full expression that built the Req).
    struct Req { const char* label; std::size_t* out; std::vector<std::string> keys; };
    ModelDims d{};
    std::vector<Req> reqs;
    reqs.push_back({"context_length",         &d.n_ctx_train, {arch + ".context_length"}});
    reqs.push_back({"embedding_length",       &d.hidden,      {arch + ".embedding_length"}});
    reqs.push_back({"block_count",            &d.n_layers,    {arch + ".block_count"}});
    reqs.push_back({"feed_forward_length",    &d.ffn_hidden,  {arch + ".feed_forward_length"}});
    reqs.push_back({"attention.head_count",   &d.n_heads,     {arch + ".attention.head_count"}});
    reqs.push_back({"attention.head_count_kv",&d.n_kv_heads,  {arch + ".attention.head_count_kv",
                                                                arch + ".attention.head_count"}});
    for (const auto& r : reqs) {
        auto v = get_uint_any(md, r.keys);
        if (!v) {
            return fail(ErrorCode::MissingRequiredMetadata,
                        fmt::format("missing '{}.{}'", arch, r.label),
                        "qwen_config");
        }
        *r.out = static_cast<std::size_t>(*v);
    }

    // Vocab from tokenizer.
    if (auto tokens = md.get<std::vector<std::string>>("tokenizer.ggml.tokens")) {
        d.vocab = tokens->size();
    } else {
        return fail(ErrorCode::MissingRequiredMetadata,
                    "missing 'tokenizer.ggml.tokens'", "qwen_config");
    }

    // head_dim derives from hidden / n_heads.
    if (d.n_heads == 0 || d.hidden == 0 || (d.hidden % d.n_heads) != 0) {
        return fail(ErrorCode::InvalidModel,
                    fmt::format("hidden {} not divisible by n_heads {}",
                                d.hidden, d.n_heads),
                    "qwen_config");
    }
    d.head_dim = d.hidden / d.n_heads;

    // RMSNorm epsilon — optional; default from Qwen2 configs.
    if (auto v = get_float_any(md, {arch + ".attention.layer_norm_rms_epsilon"})) {
        d.rms_eps = *v;
    }

    // RoPE base frequency.
    RopeConfig rp{};
    rp.rope_dim  = d.head_dim;
    if (auto v = get_float_any(md, {arch + ".rope.freq_base"})) rp.freq_base = *v;

    // YaRN scaling (Qwen3 mainly).
    if (auto s = md.get<std::string>(arch + ".rope.scaling.type")) {
        if (*s == "yarn") rp.scaling = RopeConfig::Scaling::YaRN;
    }
    if (rp.scaling == RopeConfig::Scaling::YaRN) {
        if (auto v = get_float_any(md, {arch + ".rope.scaling.factor"}))            rp.yarn_factor      = *v;
        if (auto v = get_uint_any (md, {arch + ".rope.scaling.original_context_length"}))
            rp.yarn_ctx_train = static_cast<float>(*v);
        if (auto v = get_float_any(md, {arch + ".rope.scaling.attn_factor"}))        rp.yarn_attn_factor = *v;
        if (auto v = get_float_any(md, {arch + ".rope.scaling.beta_fast"}))          rp.yarn_beta_fast   = *v;
        if (auto v = get_float_any(md, {arch + ".rope.scaling.beta_slow"}))          rp.yarn_beta_slow   = *v;
    }

    out_dims = d;
    out_rope = rp;
    return {};
}

} // namespace ultima::model
