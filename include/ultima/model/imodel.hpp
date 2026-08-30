#pragma once

#include "ultima/core/error.hpp"
#include "ultima/kv_cache/kv_cache.hpp"
#include "ultima/model/loaded_model.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace ultima::runtime { class ThreadPool; }

namespace ultima::model {

// Shape / config summary the runtime pulls once at load time.
struct ModelDims {
    std::size_t n_layers    = 0;
    std::size_t n_heads     = 0;
    std::size_t n_kv_heads  = 0;   // grouped-query
    std::size_t head_dim    = 0;
    std::size_t hidden      = 0;   // n_heads * head_dim
    std::size_t ffn_hidden  = 0;
    std::size_t vocab       = 0;
    std::size_t n_ctx_train = 0;
    float       rms_eps     = 1e-6f;
};

// RoPE parameters shared by Qwen2 and Qwen3. Qwen3 optionally sets YaRN.
struct RopeConfig {
    enum class Scaling { None, YaRN };

    float       freq_base       = 10000.0f;
    std::size_t rope_dim        = 0;      // == head_dim in Qwen2/3
    Scaling     scaling         = Scaling::None;
    float       yarn_factor     = 1.0f;   // set from GGUF when scaling==YaRN
    float       yarn_ctx_train  = 0.0f;
    float       yarn_attn_factor= 1.0f;
    float       yarn_beta_fast  = 32.0f;
    float       yarn_beta_slow  = 1.0f;
};

// Per Decision 06 — the seam every future architecture speaks. v0.1 ships
// Qwen2Model + Qwen3Model. Neither prefill nor decode allocates KV; the
// caller owns the KVCache and hands in a slot handle.
class IModel {
public:
    virtual ~IModel() = default;

    virtual const ModelDims&  dims() const noexcept = 0;
    virtual const RopeConfig& rope() const noexcept = 0;
    virtual const char*       arch() const noexcept = 0;   // "qwen2" | "qwen3"

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

// Factory: dispatches by GGUF general.architecture. Errors on unknown arch;
// v0.1 supports "qwen2" and "qwen3".
core::Result<std::unique_ptr<IModel>>
load_model(const LoadedModel& gguf, runtime::ThreadPool& pool);

} // namespace ultima::model
