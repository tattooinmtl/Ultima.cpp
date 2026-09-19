#pragma once

#include "ultima/core/error.hpp"
#include "ultima/kv_cache/kv_cache.hpp"
#include "ultima/kv_cache/recurrent_state.hpp"
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
    std::vector<float> head_gate;    // n_heads (gated attention only)
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

    // QkvKind::SeparateQGate: attn_q emits query and gate interleaved per
    // head, so it needs a 2*q_dim landing buffer before the two are split.
    std::vector<float> qg_full;
    std::vector<float> elem_gate;    // q_dim — the de-interleaved gate half

    // ---- Gated delta-net working set (qwen35) ---------------------------
    // Empty for architectures with no recurrent layers.
    std::vector<float> dn_mixed;     // conv_dim  — the fused q||k||v projection
    std::vector<float> dn_conv_out;  // conv_dim  — after the causal conv + SiLU
    std::vector<float> dn_z;         // value_dim — the output gate branch
    std::vector<float> dn_beta;      // n_v_heads
    std::vector<float> dn_alpha;     // n_v_heads
    std::vector<float> dn_decay;     // n_v_heads
    std::vector<float> dn_out;       // value_dim
    std::vector<float> dn_qh;        // d_state — one L2-normed, scaled q head
    std::vector<float> dn_kh;        // d_state — one L2-normed k head
    std::vector<float> dn_scratch;   // d_state — delta_net_step_f32 workspace

    // Per-layer delta-net weights that are small enough to keep dequantized.
    // The conv kernel is the big one: conv_dim * d_conv floats per layer.
    std::vector<float> ssm_conv_w;   // n_layers * conv_dim * d_conv
    std::vector<float> ssm_dt_bias;  // n_layers * n_v_heads
    std::vector<float> ssm_a;        // n_layers * n_v_heads
    std::vector<float> ssm_norm;     // n_layers * head_v_dim

    // Hadamard fold, when the file declares one. Non-owning: the model
    // outlives the scratch. fold_buf holds one rotated activation; it is
    // consumed by the matvec immediately after, so one is enough.
    const HadamardFold* fold = nullptr;
    std::vector<float>  fold_buf;
    std::vector<float>  fold_perm;   // gdn v reorder, ssm_out only

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
              runtime::ThreadPool& pool,
              kv_cache::RecurrentState* rs = nullptr);

// Loads norm + bias tensors from the ModelTensors into the scratch's
// pre-dequantized f32 buffers. Call once at model construction after
// resolve_model_tensors returns.
core::Result<void>
preload_scratch_from_tensors(BlockScratch& s,
                             const ModelTensors& t,
                             const ModelDims& d,
                             const ArchParams& p);

} // namespace ultima::model
