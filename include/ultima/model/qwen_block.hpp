#pragma once

#include "ultima/core/error.hpp"
#include "ultima/kv_cache/kv_cache.hpp"
#include "ultima/model/imodel.hpp"
#include "ultima/model/qwen_tensor_slots.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ultima::runtime { class ThreadPool; }

namespace ultima::model {

// Per-model scratch: sized once from ModelDims + ArchParams and reused across
// every forward-pass call. Nothing here is allocated in the hot path.
struct BlockScratch {
    std::vector<float> x;            // hidden — residual stream
    std::vector<float> residual;     // hidden — saved branch input
    std::vector<float> x_norm;       // hidden — pre-norm output
    std::vector<float> qkv_full;     // q_dim + 2*kv_dim (fused QKV only)
    std::vector<float> q_full;       // q_dim
    std::vector<float> k_full;       // kv_dim
    std::vector<float> v_full;       // kv_dim
    std::vector<float> attn_out;     // q_dim
    std::vector<float> attn_scores;  // n_ctx (per-head, reused across heads)
    std::vector<float> gate;         // ffn width (see init)
    std::vector<float> up;           // ffn width (see init)
    std::vector<float> logits_tmp;   // vocab (only allocated for LM head)

    // Sparse-MoE working set. Empty for dense architectures.
    std::vector<float>        router_probs;    // n_expert
    std::vector<std::uint32_t> expert_ids;     // n_expert_used
    std::vector<float>        expert_weights;  // n_expert_used
    std::vector<float>        expert_out;      // hidden
    std::vector<float>        moe_accum;       // hidden

    // Dequant scratch for norm weights (loaded once at model construction).
    std::vector<float> attn_norm;    // per layer, hidden
    std::vector<float> ffn_norm;     // per layer, hidden
    std::vector<float> output_norm_f32; // hidden

    std::vector<float> attn_q_bias;  // per layer, q_dim
    std::vector<float> attn_k_bias;  // per layer, kv_dim
    std::vector<float> attn_v_bias;  // per layer, kv_dim
    // Bias-presence flags per layer (avoid conditionals in hot path).
    std::vector<bool>  has_biases;

    // Qwen3 Q/K norm (per layer, per head_dim each). Optional.
    std::vector<float> attn_q_norm;
    std::vector<float> attn_k_norm;
    std::vector<bool>  has_qk_norm;

    void init(const ModelDims& d, const ArchParams& p);
};

// One layer's forward pass. Consumes and produces the residual stream in
// scratch.x (in-place). Writes the layer's K/V into the KV cache at the
// given absolute position; attention reads positions [0, position+1).
//
// Covers every llama-family layout in the arch registry: `params` selects
// between separate/fused QKV, dense/fused/MoE feed-forward, the RoPE
// convention and the scalar corrections.
core::Result<void>
block_forward(BlockScratch& s,
              const ModelTensors::Block& block,
              const ModelDims&  dims,
              const ArchParams& params,
              const RopeConfig& rope,
              kv_cache::KVCache& kv,
              kv_cache::SlotHandle slot,
              std::size_t layer,
              std::size_t absolute_position,
              runtime::ThreadPool& pool);

// Loads norm + bias tensors from the ModelTensors into the scratch's
// pre-dequantized f32 buffers. Call once at model construction after
// resolve_model_tensors returns.
core::Result<void>
preload_scratch_from_tensors(BlockScratch& s,
                             const ModelTensors& t,
                             const ModelDims& d);

} // namespace ultima::model
