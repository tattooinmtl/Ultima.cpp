#pragma once

#include "ultima/core/error.hpp"
#include "ultima/kv_cache/kv_cache.hpp"
#include "ultima/model/arch_registry.hpp"
#include "ultima/model/loaded_model.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ultima::runtime { class ThreadPool; }

namespace ultima::model {

// Shape / config summary the runtime pulls once at load time.
struct ModelDims {
    std::size_t n_layers    = 0;
    std::size_t n_heads     = 0;
    std::size_t n_kv_heads  = 0;   // grouped-query
    std::size_t head_dim    = 0;
    std::size_t hidden      = 0;   // residual stream width
    std::size_t ffn_hidden  = 0;
    std::size_t vocab       = 0;
    std::size_t n_ctx_train = 0;
    float       rms_eps     = 1e-6f;

    // Attention projection widths. These are NOT always derivable from
    // hidden: Qwen3-MoE runs 32 heads of 128 off a 2048-wide residual, so
    // q_dim (4096) is twice hidden. Always size Q/K/V buffers from these.
    std::size_t q_dim       = 0;   // n_heads    * head_dim
    std::size_t kv_dim      = 0;   // n_kv_heads * head_dim
};

// Per-architecture behaviour that the shared block forward branches on.
// Everything here is read from GGUF metadata plus the arch registry, so a
// new architecture in the same family is a table entry, not a new code path.
struct ArchParams {
    FfnKind   ffn        = FfnKind::DenseSwiGLU;
    FfnAct    ffn_act    = FfnAct::SiLU;
    QkvKind   qkv        = QkvKind::Separate;
    RopeStyle rope_style = RopeStyle::NeoX;

    // Sparse MoE (qwen3moe and relatives).
    std::size_t n_expert      = 0;   // total experts per layer
    std::size_t n_expert_used = 0;   // top-k actually evaluated
    std::size_t expert_ffn    = 0;   // per-expert FFN width

    // Granite's explicit scalar corrections. Defaults are the identity, so
    // architectures that do not set them behave exactly as before.
    float embedding_scale = 1.0f;   // multiplies the token embedding
    float residual_scale  = 1.0f;   // multiplies each residual branch output
    float logit_scale     = 1.0f;   // divides the final logits
    float attn_scale      = 0.0f;   // 0 => use the usual 1/sqrt(head_dim)

    // Sliding-window attention. `layer_is_swa` is empty when the model
    // attends globally everywhere; otherwise it has one entry per layer and
    // the true ones attend only to the last `sliding_window` positions.
    // Those layers also use the *_swa RoPE settings.
    std::size_t       sliding_window = 0;
    std::vector<bool> layer_is_swa;

    // Per-head sigmoid gate on the attention output, applied before the
    // output projection (Spark-X2.5). The gate tensor is [n_heads, hidden].
    bool attn_gate = false;

    bool is_swa_layer(std::size_t layer) const noexcept {
        return layer < layer_is_swa.size() && layer_is_swa[layer];
    }
};

// RoPE parameters. Qwen3 optionally sets YaRN; Phi-3 sets LongRoPE frequency
// factors plus a magnitude scale.
struct RopeConfig {
    enum class Scaling { None, YaRN };

    float       freq_base       = 10000.0f;
    std::size_t rope_dim        = 0;      // == head_dim for most architectures
    Scaling     scaling         = Scaling::None;
    float       yarn_factor     = 1.0f;   // set from GGUF when scaling==YaRN
    float       yarn_ctx_train  = 0.0f;
    float       yarn_attn_factor= 1.0f;
    float       yarn_beta_fast  = 32.0f;
    float       yarn_beta_slow  = 1.0f;

    // LongRoPE / NTK-by-parts: per-dimension divisors for theta, length
    // rope_dim/2. Empty means "no correction". Phi-3 carries two sets and
    // selects by context length; we resolve that choice at load time.
    std::vector<float> freq_factors;

    // Multiplies cos/sin. 1.0 for everything that does not set it.
    float       attn_factor     = 1.0f;

    // Sliding-window layers can rotate a different number of dimensions at a
    // different base than the global-attention layers. Spark-X2.5 rotates 256
    // dims at base 1e4 on its windowed layers and only 64 at base 5e6 on the
    // global ones. Both default to the non-SWA values when a model does not
    // distinguish them.
    float       freq_base_swa   = 0.0f;   // 0 => same as freq_base
    std::size_t rope_dim_swa    = 0;      // 0 => same as rope_dim
};

// Per Decision 06 — the seam every future architecture speaks. Neither
// prefill nor decode allocates KV; the caller owns the KVCache and hands in
// a slot handle.
class IModel {
public:
    virtual ~IModel() = default;

    virtual const ModelDims&  dims()   const noexcept = 0;
    virtual const RopeConfig& rope()   const noexcept = 0;
    virtual const ArchParams& params() const noexcept = 0;
    virtual const char*       arch()   const noexcept = 0;   // general.architecture

    // Prefill a run of tokens starting at kv.pos(slot). Writes ONLY the
    // last token's logits into out_logits[0..vocab). Advances slot.pos.
    // Returns Ok on success; caller errors on shape/OOB.
    virtual core::Result<void> prefill(kv_cache::KVCache& kv,
                                       kv_cache::SlotHandle slot,
                                       std::span<const std::int32_t> tokens,
                                       float* out_logits) = 0;

    // Single-token decode using the slot's current state; appends to slot.
    virtual core::Result<void> decode(kv_cache::KVCache& kv,
                                      kv_cache::SlotHandle slot,
                                      std::int32_t token,
                                      float* out_logits) = 0;
};

// Factory: dispatches by GGUF general.architecture through the arch registry.
// An architecture the registry knows but cannot execute yet fails with the
// specific engine work it still needs; see arch_registry.hpp.
core::Result<std::unique_ptr<IModel>>
load_model(const LoadedModel& gguf, runtime::ThreadPool& pool);

} // namespace ultima::model
