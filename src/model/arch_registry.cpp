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
    {"qwen2", true, FfnKind::DenseSwiGLU, FfnAct::SiLU, QkvKind::Separate, RopeStyle::NeoX, false, false,
     "Qwen2 / Qwen2.5 dense, separate QKV with biases"},

    {"qwen3", true, FfnKind::DenseSwiGLU, FfnAct::SiLU, QkvKind::Separate, RopeStyle::NeoX, false, false,
     "Qwen3 dense, Q/K RMSNorm, no attention biases"},

    {"qwen3moe", true, FfnKind::SparseMoE, FfnAct::SiLU, QkvKind::Separate, RopeStyle::NeoX, false, false,
     "Qwen3 MoE, Q/K RMSNorm, top-k routed experts"},

    {"granite", true, FfnKind::DenseSwiGLU, FfnAct::SiLU, QkvKind::Separate, RopeStyle::Norm, false, false,
     "IBM Granite dense, embedding/residual/logit scales, explicit attn scale"},

    {"phi3", true, FfnKind::FusedSwiGLU, FfnAct::SiLU, QkvKind::Fused, RopeStyle::NeoX, false, false,
     "Phi-3, fused QKV and fused gate/up, LongRoPE frequency factors"},

    {"spark2_5", true, FfnKind::DenseSwiGLU, FfnAct::GELU, QkvKind::Fused, RopeStyle::NeoX, true, false,
     "Spark-X2.5, fused QKV, per-head sigmoid attention gate, GEGLU feed-forward, "
     "3-of-4 sliding-window layers with their own RoPE dim and base"},

    // ---- Named but not yet executable --------------------------------------

    {"llama", false, {}, {}, {}, {}, {}, false,
     "needs Norm-style RoPE validation and llama-specific tensor aliases"},

    {"gemma4", false, {}, {}, {}, {}, {}, false,
     "needs per-layer KV head counts, alternating sliding-window attention, "
     "dual RoPE bases, post-attention/post-FFN norms, final logit softcapping"},

    // Qwen3-Next / Qwen3.5 shape: gated delta-net linear attention on 3 of
    // every 4 blocks, full attention on the fourth. Supported upstream in
    // llama.cpp, so this is knowable work rather than guesswork.
    // The ternary Bonsai builds of this arch need two further things on top:
    // a PQ2_0/PTQ1_0 ternary weight format and a blockwise Hadamard transform
    // applied to activations at runtime (both in the PrismML-Eng fork).
    // Structural fields are filled in even though `supported` is still false:
    // config parsing and tensor resolution (Q2) land before the forward pass
    // (Q3/Q4), and they read this entry. Flip `supported` when decode matches
    // the reference, not before.
    {"qwen35", false, FfnKind::DenseSwiGLU, FfnAct::SiLU, QkvKind::SeparateQGate,
     RopeStyle::NeoX, /*attn_gate=*/false, /*hybrid_recurrent=*/true,
     "gated delta-net linear attention on 3 of every 4 blocks: config and "
     "tensors resolve, the recurrence and its state cache do not exist yet"},

    {"qwen35moe", false, FfnKind::SparseMoE, FfnAct::SiLU, QkvKind::SeparateQGate,
     RopeStyle::NeoX, /*attn_gate=*/false, /*hybrid_recurrent=*/true,
     "the qwen35 delta-net hybrid plus MoE routing with shared experts"},

    {"deepseek2", false, {}, {}, {}, {}, {}, false,
     "needs multi-head latent attention (compressed KV, q_a/q_b/kv_a/kv_b "
     "projections) and MoE with grouped routing, shared experts and expert bias"},

    {"dspark", false, {}, {}, {}, {}, {}, false,
     "diffusion language model, not autoregressive: block denoising does not "
     "fit prefill/decode. Also needs ternary weight kernels"},

    {"dflash-draft", false, {}, {}, {}, {}, {}, false,
     "speculative-decoding draft head, not a standalone model: the file has no "
     "tokenizer, no token_embd and no output head"},

    {"bert", false, {}, {}, {}, {}, {}, false,
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
