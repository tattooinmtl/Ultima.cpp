#pragma once

#include "ultima/core/error.hpp"
#include "ultima/kv_cache/kv_cache.hpp"
#include "ultima/model/imodel.hpp"
#include "ultima/model/qwen_tensor_slots.hpp"

#include <cstddef>
#include <vector>

namespace ultima::runtime { class ThreadPool; }

namespace ultima::model {

// Per-model scratch: sized once from ModelDims + reused across every
// forward-pass call. All buffers are aligned f32 arrays.
struct BlockScratch {
    std::vector<float> x;            // hidden — residual stream
    std::vector<float> x_norm;       // hidden — pre-norm output
    std::vector<float> q_full;       // n_heads * head_dim = hidden
    std::vector<float> k_full;       // n_kv_heads * head_dim
    std::vector<float> v_full;       // n_kv_heads * head_dim
    std::vector<float> attn_out;     // hidden
    std::vector<float> attn_scores;  // n_ctx (per-head, reused across heads)
    std::vector<float> gate;         // ffn_hidden
    std::vector<float> up;           // ffn_hidden
    std::vector<float> logits_tmp;   // vocab (only allocated for LM head)

    // Dequant scratch for norm weights (loaded once at model construction).
    std::vector<float> attn_norm;    // per layer, hidden
    std::vector<float> ffn_norm;     // per layer, hidden
    std::vector<float> output_norm_f32; // hidden

    std::vector<float> attn_q_bias;  // hidden (may be zero-filled if absent)
    std::vector<float> attn_k_bias;  // kv_dim
    std::vector<float> attn_v_bias;  // kv_dim
    // Bias-presence flags per layer (avoid conditionals in hot path).
    std::vector<bool>  has_biases;

    // Qwen3 Q/K norm (per layer, per head_dim each). Optional.
    std::vector<float> attn_q_norm;
    std::vector<float> attn_k_norm;
    std::vector<bool>  has_qk_norm;

    void init(const ModelDims& d);
};

// One layer's forward pass. Consumes and produces the residual stream in
// scratch.x (in-place). Writes the layer's K/V into the KV cache at the
// given absolute position; attention reads positions [0, position+1).
core::Result<void>
qwen_block_forward(BlockScratch& s,
                   const QwenTensors::Block& block,
                   const ModelDims&  dims,
                   const RopeConfig& rope,
                   kv_cache::KVCache& kv,
                   kv_cache::SlotHandle slot,
                   std::size_t layer,
                   std::size_t absolute_position,
                   runtime::ThreadPool& pool);

// Loads norm + bias tensors from the QwenTensors into the scratch's
// pre-dequantized f32 buffers. Call once at model construction after
// resolve_qwen_tensors returns.
core::Result<void>
preload_scratch_from_tensors(BlockScratch& s,
                             const QwenTensors& t,
                             const ModelDims& d);

} // namespace ultima::model
