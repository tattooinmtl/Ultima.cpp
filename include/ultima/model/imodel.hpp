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

    // Gated delta-net linear attention (qwen35 / Qwen3-Next). Three of every
    // four blocks replace attention with this recurrence, so there is no KV
    // for them at all — they carry a fixed-size state instead, which is why
    // the architecture is cheap at long context.
    struct DeltaNet {
        std::size_t d_conv    = 0;   // causal conv width       (ssm.conv_kernel)
        std::size_t d_inner   = 0;   // value width             (ssm.inner_size)
        std::size_t d_state   = 0;   // per-head k/v width      (ssm.state_size)
        std::size_t n_k_heads = 0;   // key heads               (ssm.group_count)
        std::size_t n_v_heads = 0;   // value heads             (ssm.time_step_rank)

        std::size_t key_dim()    const noexcept { return d_state * n_k_heads; }
        std::size_t value_dim()  const noexcept { return d_state * n_v_heads; }
        std::size_t head_v_dim() const noexcept {
            return n_v_heads ? d_inner / n_v_heads : 0;
        }
        // Width of the fused qkv projection feeding the causal conv.
        std::size_t conv_dim()   const noexcept { return key_dim() * 2 + value_dim(); }

        bool active() const noexcept { return d_state != 0 && n_v_heads != 0; }
    };
    DeltaNet delta;

    // One entry per layer when the architecture is hybrid; empty otherwise.
    // True means the block uses the delta-net mixer instead of attention.
    std::vector<bool> layer_is_recurrent;

    bool is_swa_layer(std::size_t layer) const noexcept {
        return layer < layer_is_swa.size() && layer_is_swa[layer];
    }

    bool is_recurrent_layer(std::size_t layer) const noexcept {
        return layer < layer_is_recurrent.size() && layer_is_recurrent[layer];
    }

    // Layers that actually allocate KV. For a hybrid model this is a quarter
    // of them, which is the whole point of the architecture.
    std::size_t n_attention_layers(std::size_t n_layers) const noexcept {
        if (layer_is_recurrent.empty()) return n_layers;
        std::size_t n = 0;
        for (std::size_t i = 0; i < n_layers; ++i) n += !is_recurrent_layer(i);
        return n;
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

// A request to capture the residual stream as it *enters* selected blocks.
//
// This is what a speculative drafter feeds on: the dspark head reads the
// target's hidden state at a handful of depths and projects the
// concatenation down to one embedding width. Note it is the layer *input*,
// matching llama.cpp's t_layer_inp — the output of block L is the input of
// block L+1, so the distinction only matters at the ends.
//
// `out` is filled token-major, each token's row holding the taps
// concatenated in `layers` order:
//
//   out[t * (layers.size() * hidden) + i * hidden + j]
//       = hidden[j] entering layers[i], for token t
//
// The caller owns the buffer and must size it for the tokens it passes.
struct LayerTaps {
    std::vector<std::size_t> layers;   // block indices, ascending
    float*      out    = nullptr;
    std::size_t stride = 0;            // floats per token; 0 => layers.size()*hidden

    bool active() const noexcept { return out != nullptr && !layers.empty(); }
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

    // Capture layer inputs on subsequent prefill/decode calls. Pass nullptr
    // to stop. The pointer is borrowed, not owned, and must outlive the
    // calls it applies to. Architectures that do not implement it simply
    // ignore the request, which a drafter must check for rather than
    // silently reading a zeroed buffer.
    virtual void set_layer_taps(const LayerTaps* taps) { (void)taps; }
    virtual bool supports_layer_taps() const noexcept { return false; }

    // Whether a slot's cache can be truncated to a shared prefix and only
    // the suffix re-run. True for ordinary transformers, where every KV
    // entry is indexed by position and keeping the first N is meaningful.
    //
    // False for a hybrid model with recurrent layers: their state after N
    // tokens cannot be recovered from the state after M > N, so a caller
    // that truncates and prefills only the suffix would be feeding the
    // recurrence a prompt it never saw the start of. Such a model must be
    // re-prefilled from position zero on any prompt change.
    virtual bool supports_prefix_reuse() const noexcept { return true; }

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
