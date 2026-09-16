#pragma once

#include "ultima/core/error.hpp"
#include "ultima/model/imodel.hpp"
#include "ultima/model/loaded_model.hpp"
#include "ultima/model/tensor_info.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ultima::model {

// Non-owning view into a weight tensor: dtype + shape + byte pointer into
// the LoadedModel's mmap. Same lifetime as the LoadedModel.
struct WeightRef {
    DataType    dtype   = DataType::Unknown;
    const void* data    = nullptr;
    std::size_t rows    = 0;   // outer dim of a 2D projection, or vocab
    std::size_t cols    = 0;   // inner dim; 0 for 1D vectors
    // For 1D tensors (norms, biases): rows = length, cols = 0.

    // For a stack of per-expert matrices: bytes between consecutive experts.
    // Zero for ordinary 2D weights.
    std::size_t expert_stride = 0;

    bool valid() const noexcept { return data != nullptr; }

    // Byte pointer to expert `e` within a 3D expert stack.
    const void* expert(std::size_t e) const noexcept {
        return static_cast<const std::uint8_t*>(data) + e * expert_stride;
    }
};

// Resolved once at model construction and cached. Non-owning refs into the
// LoadedModel's mmap; do not outlive it.
//
// Covers the llama-family layouts Ultima executes: separate or fused QKV,
// and dense / fused / sparse-MoE feed-forward. Which members are populated
// is decided by ArchParams, not by probing.
struct ModelTensors {
    WeightRef token_embd;      // [vocab, hidden]
    WeightRef output_norm;     // [hidden]
    WeightRef output_head;     // [vocab, hidden]  (may alias token_embd)
    bool      output_head_is_tied = false;

    struct Block {
        WeightRef attn_norm;   // [hidden]

        // QkvKind::Separate
        WeightRef attn_q;      // [q_dim,  hidden]
        WeightRef attn_k;      // [kv_dim, hidden]
        WeightRef attn_v;      // [kv_dim, hidden]
        // QkvKind::Fused
        WeightRef attn_qkv;    // [q_dim + 2*kv_dim, hidden]

        WeightRef attn_out;    // [hidden, q_dim]
        WeightRef attn_gate;   // [n_heads, hidden] — one scalar per head
        WeightRef ffn_norm;    // [hidden]

        // FfnKind::DenseSwiGLU uses gate + up + down.
        // FfnKind::FusedSwiGLU leaves gate empty; ffn_up is [2*ffn_hidden, hidden]
        //   holding [gate || up].
        WeightRef ffn_gate;    // [ffn_hidden, hidden]
        WeightRef ffn_up;      // [ffn_hidden, hidden] or [2*ffn_hidden, hidden]
        WeightRef ffn_down;    // [hidden, ffn_hidden]

        // FfnKind::SparseMoE
        WeightRef ffn_gate_inp;    // router: [n_expert, hidden]
        WeightRef ffn_gate_exps;   // per expert: [expert_ffn, hidden]
        WeightRef ffn_up_exps;     // per expert: [expert_ffn, hidden]
        WeightRef ffn_down_exps;   // per expert: [hidden, expert_ffn]

        // Optional biases (present in some Qwen2 variants; absent in Qwen3).
        std::optional<WeightRef> attn_q_bias;
        std::optional<WeightRef> attn_k_bias;
        std::optional<WeightRef> attn_v_bias;

        // Qwen3 / Qwen3-MoE per-head Q/K normalization.
        std::optional<WeightRef> attn_q_norm;   // [head_dim]
        std::optional<WeightRef> attn_k_norm;   // [head_dim]
    };
    std::vector<Block> blocks;
};

// Load the tensor table for the given architecture. Errors if any required
// tensor is missing or has the wrong dims.
core::Result<ModelTensors>
resolve_model_tensors(const LoadedModel& gguf,
                      const std::string& arch,
                      const ModelDims&   dims,
                      const ArchParams&  params);

} // namespace ultima::model
