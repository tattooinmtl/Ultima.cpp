#include "ultima/model/qwen_config.hpp"

#include "ultima/model/weight_dispatch.hpp"

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

// Read a 1D f32-ish tensor into `out`. Returns false when absent.
bool read_vector_tensor(const LoadedModel& gguf, const std::string& name,
                        std::vector<float>& out) {
    auto tv = gguf.tensor(name);
    if (!tv) return false;
    const TensorInfo& ti = tv->info();
    if (ti.dims.size() != 1 || ti.dims[0] == 0) return false;
    const std::size_t n = static_cast<std::size_t>(ti.dims[0]);
    out.assign(n, 0.0f);
    return static_cast<bool>(dequant_1d(ti.dtype, tv->data(), out.data(), n));
}

} // namespace

Result<void>
parse_arch_config(const LoadedModel& gguf,
                  const ArchSpec&    spec,
                  const std::string& arch,
                  ModelDims&  out_dims,
                  RopeConfig& out_rope,
                  ArchParams& out_params) {
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
                        "arch_config");
        }
        *r.out = static_cast<std::size_t>(*v);
    }

    // Vocab from tokenizer.
    if (auto tokens = md.get<std::vector<std::string>>("tokenizer.ggml.tokens")) {
        d.vocab = tokens->size();
    } else {
        return fail(ErrorCode::MissingRequiredMetadata,
                    "missing 'tokenizer.ggml.tokens'", "arch_config");
    }

    // head_dim: an explicit attention.key_length wins. Qwen3-MoE runs 32
    // heads of 128 off a 2048-wide residual, so hidden/n_heads (64) would be
    // wrong there — only the fallback path may assume they agree.
    if (auto kl = get_uint_any(md, {arch + ".attention.key_length"})) {
        d.head_dim = static_cast<std::size_t>(*kl);
    } else {
        if (d.n_heads == 0 || d.hidden == 0 || (d.hidden % d.n_heads) != 0) {
            return fail(ErrorCode::InvalidModel,
                        fmt::format("hidden {} not divisible by n_heads {}",
                                    d.hidden, d.n_heads),
                        "arch_config");
        }
        d.head_dim = d.hidden / d.n_heads;
    }
    if (d.head_dim == 0) {
        return fail(ErrorCode::InvalidModel, "head_dim resolved to 0", "arch_config");
    }
    d.q_dim  = d.n_heads    * d.head_dim;
    d.kv_dim = d.n_kv_heads * d.head_dim;

    // RMSNorm epsilon — optional; default from Qwen2 configs.
    if (auto v = get_float_any(md, {arch + ".attention.layer_norm_rms_epsilon"})) {
        d.rms_eps = *v;
    }

    // ---- RoPE ---------------------------------------------------------------
    RopeConfig rp{};
    rp.rope_dim = d.head_dim;
    if (auto v = get_uint_any(md, {arch + ".rope.dimension_count"})) {
        const auto rd = static_cast<std::size_t>(*v);
        if (rd > 0 && rd <= d.head_dim) rp.rope_dim = rd;
    }
    if (auto v = get_float_any(md, {arch + ".rope.freq_base"})) rp.freq_base = *v;

    // Magnitude correction (Phi-3 LongRoPE sets this; most archs do not).
    if (auto v = get_float_any(md, {arch + ".rope.scaling.attn_factor"})) {
        rp.attn_factor = *v;
    }

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

    // LongRoPE frequency factors live in tensors, not metadata. llama.cpp
    // swaps to the "long" set once the sequence passes the original training
    // context; Ultima resolves the choice once at load time and takes the
    // short set, which is correct up to original_context_length tokens.
    {
        std::vector<float> ff;
        if (read_vector_tensor(gguf, "rope_factors_short.weight", ff) ||
            read_vector_tensor(gguf, "rope_factors_long.weight",  ff)) {
            if (ff.size() == rp.rope_dim / 2) {
                rp.freq_factors = std::move(ff);
            }
        }
    }

    // ---- Structural parameters ---------------------------------------------
    ArchParams p{};
    p.ffn        = spec.ffn;
    p.qkv        = spec.qkv;
    p.rope_style = spec.rope_style;

    if (p.ffn == FfnKind::SparseMoE) {
        auto n_exp  = get_uint_any(md, {arch + ".expert_count"});
        auto n_used = get_uint_any(md, {arch + ".expert_used_count"});
        auto e_ffn  = get_uint_any(md, {arch + ".expert_feed_forward_length"});
        if (!n_exp || !n_used || !e_ffn) {
            return fail(ErrorCode::MissingRequiredMetadata,
                        fmt::format("'{}' is MoE but expert_count / "
                                    "expert_used_count / expert_feed_forward_length "
                                    "is missing", arch),
                        "arch_config");
        }
        p.n_expert      = static_cast<std::size_t>(*n_exp);
        p.n_expert_used = static_cast<std::size_t>(*n_used);
        p.expert_ffn    = static_cast<std::size_t>(*e_ffn);
        if (p.n_expert == 0 || p.n_expert_used == 0 || p.expert_ffn == 0) {
            return fail(ErrorCode::InvalidModel,
                        "MoE expert counts must be non-zero", "arch_config");
        }
        if (p.n_expert_used > p.n_expert) {
            return fail(ErrorCode::InvalidModel,
                        fmt::format("expert_used_count {} exceeds expert_count {}",
                                    p.n_expert_used, p.n_expert),
                        "arch_config");
        }
        // A shared expert alongside the routed ones is a separate code path
        // (ffn_*_shexp); reject rather than silently dropping its contribution.
        if (auto se = get_uint_any(md, {arch + ".expert_shared_feed_forward_length"});
            se && *se > 0) {
            return fail(ErrorCode::InvalidModel,
                        fmt::format("'{}' declares a shared expert "
                                    "(expert_shared_feed_forward_length={}), "
                                    "which is not implemented", arch, *se),
                        "arch_config");
        }
    }

    // Granite's scalar corrections; identity when absent.
    if (auto v = get_float_any(md, {arch + ".embedding_scale"})) p.embedding_scale = *v;
    if (auto v = get_float_any(md, {arch + ".residual_scale"}))  p.residual_scale  = *v;
    if (auto v = get_float_any(md, {arch + ".logit_scale"}))     p.logit_scale     = *v;
    if (auto v = get_float_any(md, {arch + ".attention.scale"})) p.attn_scale      = *v;

    out_dims   = d;
    out_rope   = std::move(rp);
    out_params = p;
    return {};
}

Result<void>
parse_qwen_config(const LoadedModel& gguf,
                  const std::string& arch,
                  ModelDims&  out_dims,
                  RopeConfig& out_rope) {
    const ArchSpec* spec = find_arch(arch);
    // Callers that only want dims get the qwen-family structural defaults
    // when the arch is not in the registry.
    static constexpr ArchSpec k_default{
        "", true, FfnKind::DenseSwiGLU, QkvKind::Separate, RopeStyle::NeoX, ""};
    ArchParams ignored{};
    return parse_arch_config(gguf, spec ? *spec : k_default, arch,
                             out_dims, out_rope, ignored);
}

} // namespace ultima::model
