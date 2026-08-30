#pragma once

#include "ultima/core/error.hpp"
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
};

// Resolves once at model construction and cached. Non-owning refs into the
// LoadedModel's mmap; do not outlive it.
struct QwenTensors {
    WeightRef token_embd;      // [vocab, hidden]
    WeightRef output_norm;     // [hidden]
    WeightRef output_head;     // [vocab, hidden]  (may alias token_embd)
    bool      output_head_is_tied = false;

    struct Block {
        WeightRef attn_norm;   // [hidden]
        WeightRef attn_q;      // [hidden, hidden]
        WeightRef attn_k;      // [n_kv_heads*head_dim, hidden]
        WeightRef attn_v;      // [n_kv_heads*head_dim, hidden]
        WeightRef attn_out;    // [hidden, hidden]
        WeightRef ffn_norm;    // [hidden]
        WeightRef ffn_gate;    // [ffn_hidden, hidden]
        WeightRef ffn_up;      // [ffn_hidden, hidden]
        WeightRef ffn_down;    // [hidden, ffn_hidden]

        // Optional biases (present in some Qwen2 variants; absent in Qwen3).
        std::optional<WeightRef> attn_q_bias;
        std::optional<WeightRef> attn_k_bias;
        std::optional<WeightRef> attn_v_bias;

        // Qwen3 only.
        std::optional<WeightRef> attn_q_norm;   // [head_dim]
        std::optional<WeightRef> attn_k_norm;   // [head_dim]
    };
    std::vector<Block> blocks;
};

// Load the tensor table for the given architecture. Errors if any required
// tensor is missing or has the wrong dims.
core::Result<QwenTensors>
resolve_qwen_tensors(const LoadedModel& gguf,
                     const std::string& arch,        // "qwen2" or "qwen3"
                     std::size_t n_layers,
                     std::size_t hidden,
                     std::size_t n_kv_heads,
                     std::size_t head_dim,
                     std::size_t ffn_hidden,
                     std::size_t vocab);

} // namespace ultima::model
