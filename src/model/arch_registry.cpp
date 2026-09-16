#include "ultima/model/arch_registry.hpp"

#include <array>

namespace ultima::model {

namespace {

// Ordered supported-first, then the architectures we can name but not yet
// execute. The `notes` on an unsupported entry is what the user sees when
// they point Ultima at such a file, so it names the missing engine work
// rather than just saying no.
constexpr std::array<ArchSpec, 14> k_archs{{
    // ---- Supported ---------------------------------------------------------
    {"qwen2", true, FfnKind::DenseSwiGLU, QkvKind::Separate, RopeStyle::NeoX,
     "Qwen2 / Qwen2.5 dense, separate QKV with biases"},

    {"qwen3", true, FfnKind::DenseSwiGLU, QkvKind::Separate, RopeStyle::NeoX,
     "Qwen3 dense, Q/K RMSNorm, no attention biases"},

    {"qwen3moe", true, FfnKind::SparseMoE, QkvKind::Separate, RopeStyle::NeoX,
     "Qwen3 MoE, Q/K RMSNorm, top-k routed experts"},

    {"granite", true, FfnKind::DenseSwiGLU, QkvKind::Separate, RopeStyle::Norm,
     "IBM Granite dense, embedding/residual/logit scales, explicit attn scale"},

    {"phi3", true, FfnKind::FusedSwiGLU, QkvKind::Fused, RopeStyle::NeoX,
     "Phi-3, fused QKV and fused gate/up, LongRoPE frequency factors"},

    // ---- Named but not yet executable --------------------------------------
    {"llama", false, {}, {}, {},
     "needs Norm-style RoPE validation and llama-specific tensor aliases"},

    {"gemma4", false, {}, {}, {},
     "needs per-layer KV head counts, alternating sliding-window attention, "
     "dual RoPE bases, post-attention/post-FFN norms, final logit softcapping"},

    {"spark2_5", false, {}, {}, {},
     "needs gated attention (attn_gate), sliding-window pattern, dual RoPE bases"},

    {"qwen35", false, {}, {}, {},
     "needs hybrid SSM/Mamba layers (ssm_* tensors, 3 of every 4 blocks), "
     "multimodal RoPE sections, and an SSM state cache alongside the KV cache"},

    {"qwen35moe", false, {}, {}, {},
     "needs the qwen35 SSM hybrid plus MoE routing with shared experts"},

    {"deepseek2", false, {}, {}, {},
     "needs multi-head latent attention (compressed KV, q_a/q_b/kv_a/kv_b "
     "projections) and MoE with grouped routing, shared experts and expert bias"},

    {"dspark", false, {}, {}, {},
     "diffusion language model, not autoregressive: block denoising does not "
     "fit prefill/decode. Also needs ternary weight kernels"},

    {"dflash-draft", false, {}, {}, {},
     "speculative-decoding draft head, not a standalone model: the file has no "
     "tokenizer, no token_embd and no output head"},

    {"bert", false, {}, {}, {},
     "encoder-only embedding model; needs a pooling path rather than "
     "causal prefill/decode"},
}};

} // namespace

const ArchSpec* find_arch(std::string_view name) noexcept {
    for (const auto& a : k_archs) {
        if (a.name == name) return &a;
    }
    return nullptr;
}

std::span<const ArchSpec> all_archs() noexcept {
    return std::span<const ArchSpec>{k_archs.data(), k_archs.size()};
}

} // namespace ultima::model
