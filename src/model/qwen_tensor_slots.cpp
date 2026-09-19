#include "ultima/model/qwen_tensor_slots.hpp"

#include <fmt/format.h>

namespace ultima::model {

namespace {

using core::ErrorCode;
using core::Failure;
using core::Result;
using core::fail;

// Bytes occupied by one [rows, cols] matrix of `dtype`, laid out row-major
// with `cols` as the quantization axis.
std::size_t matrix_bytes(DataType dtype, std::size_t rows, std::size_t cols) noexcept {
    const std::uint32_t bs = block_size(dtype);
    const std::uint32_t bb = bytes_per_block(dtype);
    if (bs == 0 || bb == 0 || (cols % bs) != 0) return 0;
    return rows * (cols / bs) * bb;
}

Result<WeightRef>
resolve(const LoadedModel& gguf, const std::string& name,
        std::size_t expected_rows, std::size_t expected_cols) {
    auto tv = gguf.tensor(name);
    if (!tv) {
        return fail(ErrorCode::MissingRequiredMetadata,
                    fmt::format("missing tensor '{}'", name),
                    "tensor_slots");
    }
    const TensorInfo& ti = tv->info();
    WeightRef w{};
    w.dtype = ti.dtype;
    w.data  = tv->data();

    // GGUF stores 2D projection weights as [cols, rows] in row-major (i.e.
    // inner dim first). llama.cpp exposes them semantically as [rows, cols]
    // where rows = output dim. dims[0] = cols (K in matvec), dims[1] = rows (M).
    if (expected_cols == 0) {
        // 1D expected (norm, bias): dims must be [expected_rows].
        if (ti.dims.size() != 1 || ti.dims[0] != expected_rows) {
            return fail(ErrorCode::InvalidModel,
                        fmt::format("tensor '{}' shape mismatch: expected [{}], got {}D",
                                    name, expected_rows, ti.dims.size()),
                        "tensor_slots");
        }
        w.rows = expected_rows;
        w.cols = 0;
    } else {
        if (ti.dims.size() != 2 ||
            ti.dims[0] != expected_cols || ti.dims[1] != expected_rows) {
            return fail(ErrorCode::InvalidModel,
                        fmt::format("tensor '{}' shape mismatch: expected [{},{}], got [{}{}]",
                                    name, expected_rows, expected_cols,
                                    ti.dims.size() > 0 ? std::to_string(ti.dims[0]) : "",
                                    ti.dims.size() > 1 ? "," + std::to_string(ti.dims[1]) : ""),
                        "tensor_slots");
        }
        w.rows = expected_rows;
        w.cols = expected_cols;
    }
    return w;
}

// A stack of `n_expert` matrices of [rows, cols], stored as a 3D GGUF tensor
// with dims [cols, rows, n_expert].
Result<WeightRef>
resolve_experts(const LoadedModel& gguf, const std::string& name,
                std::size_t rows, std::size_t cols, std::size_t n_expert) {
    auto tv = gguf.tensor(name);
    if (!tv) {
        return fail(ErrorCode::MissingRequiredMetadata,
                    fmt::format("missing expert tensor '{}'", name),
                    "tensor_slots");
    }
    const TensorInfo& ti = tv->info();
    if (ti.dims.size() != 3 ||
        ti.dims[0] != cols || ti.dims[1] != rows || ti.dims[2] != n_expert) {
        return fail(ErrorCode::InvalidModel,
                    fmt::format("expert tensor '{}' shape mismatch: "
                                "expected [{},{},{}], got {}D",
                                name, cols, rows, n_expert, ti.dims.size()),
                    "tensor_slots");
    }

    const std::size_t stride = matrix_bytes(ti.dtype, rows, cols);
    if (stride == 0) {
        return fail(ErrorCode::InvalidModel,
                    fmt::format("expert tensor '{}': dtype {} cannot tile "
                                "[{},{}] (cols not a multiple of the block size)",
                                name, to_string(ti.dtype), rows, cols),
                    "tensor_slots");
    }
    // Guard the stride arithmetic against a directory that disagrees with the
    // dtype's own packing, since expert() indexes raw bytes off it.
    if (stride * n_expert != ti.size_bytes) {
        return fail(ErrorCode::CorruptTensorDirectory,
                    fmt::format("expert tensor '{}': computed {} bytes "
                                "({} x {}) but directory says {}",
                                name, stride * n_expert, stride, n_expert,
                                ti.size_bytes),
                    "tensor_slots");
    }

    WeightRef w{};
    w.dtype         = ti.dtype;
    w.data          = tv->data();
    w.rows          = rows;
    w.cols          = cols;
    w.expert_stride = stride;
    return w;
}

std::optional<WeightRef>
resolve_optional(const LoadedModel& gguf, const std::string& name,
                 std::size_t expected_rows, std::size_t expected_cols) {
    auto tv = gguf.tensor(name);
    if (!tv) return std::nullopt;
    auto r = resolve(gguf, name, expected_rows, expected_cols);
    if (!r) return std::nullopt;
    return *r;
}

} // namespace

Result<ModelTensors>
resolve_model_tensors(const LoadedModel& gguf,
                      const std::string& arch,
                      const ModelDims&   d,
                      const ArchParams&  p) {
    (void)arch;   // reserved for future per-arch renames
    ModelTensors t{};

    // Top-level tensors.
    if (auto r = resolve(gguf, "token_embd.weight", d.vocab, d.hidden); r) t.token_embd = *r;
    else return Failure{r.error()};

    if (auto r = resolve(gguf, "output_norm.weight", d.hidden, 0); r) t.output_norm = *r;
    else return Failure{r.error()};

    // Output head: try output.weight first, fall back to token_embd (tied).
    if (auto opt = resolve_optional(gguf, "output.weight", d.vocab, d.hidden); opt) {
        t.output_head = *opt;
        t.output_head_is_tied = false;
    } else {
        t.output_head = t.token_embd;
        t.output_head_is_tied = true;
    }

    t.blocks.resize(d.n_layers);
    for (std::size_t i = 0; i < d.n_layers; ++i) {
        const std::string pfx = fmt::format("blk.{}.", i);
        ModelTensors::Block& b = t.blocks[i];

        auto get = [&](const char* suffix, std::size_t rr, std::size_t cc,
                       WeightRef& out) -> Result<void> {
            auto r = resolve(gguf, pfx + suffix, rr, cc);
            if (!r) return Failure{r.error()};
            out = *r;
            return {};
        };

        if (auto r = get("attn_norm.weight", d.hidden, 0, b.attn_norm); !r)
            return Failure{r.error()};

        // A delta-net block has no attention at all: no Q/K/V, no output
        // projection, and no KV cache entry. It is a different mixer sharing
        // the same residual stream and the same FFN.
        b.is_recurrent = p.is_recurrent_layer(i);
        if (b.is_recurrent) {
            const auto& dn = p.delta;
            if (auto r = get("attn_qkv.weight",   dn.conv_dim(),  d.hidden, b.ssm_in);   !r) return Failure{r.error()};
            if (auto r = get("attn_gate.weight",  dn.value_dim(), d.hidden, b.ssm_gate); !r) return Failure{r.error()};
            // GGUF stores the conv kernel as [d_conv, conv_dim]; resolve()
            // takes (rows, cols) with cols the inner/first GGUF dim.
            if (auto r = get("ssm_conv1d.weight", dn.conv_dim(), dn.d_conv, b.ssm_conv1d); !r)
                return Failure{r.error()};
            if (auto r = get("ssm_dt.bias",   dn.n_v_heads, 0,        b.ssm_dt_bias); !r) return Failure{r.error()};
            if (auto r = get("ssm_a",         dn.n_v_heads, 0,        b.ssm_a);       !r) return Failure{r.error()};
            if (auto r = get("ssm_beta.weight",  dn.n_v_heads, d.hidden, b.ssm_beta);  !r) return Failure{r.error()};
            if (auto r = get("ssm_alpha.weight", dn.n_v_heads, d.hidden, b.ssm_alpha); !r) return Failure{r.error()};
            if (auto r = get("ssm_norm.weight",  dn.head_v_dim(), 0,     b.ssm_norm);  !r) return Failure{r.error()};
            if (auto r = get("ssm_out.weight",   d.hidden, dn.value_dim(), b.ssm_out); !r) return Failure{r.error()};
        } else if (p.qkv == QkvKind::SeparateQGate) {
            // Q carries its own output gate at double width; K and V are
            // ordinary.
            if (auto r = get("attn_q.weight", 2 * d.q_dim, d.hidden, b.attn_q); !r) return Failure{r.error()};
            if (auto r = get("attn_k.weight", d.kv_dim,    d.hidden, b.attn_k); !r) return Failure{r.error()};
            if (auto r = get("attn_v.weight", d.kv_dim,    d.hidden, b.attn_v); !r) return Failure{r.error()};
        } else if (p.qkv == QkvKind::Fused) {
            // One matrix producing [Q || K || V] in that order.
            const std::size_t qkv_rows = d.q_dim + 2 * d.kv_dim;
            if (auto r = get("attn_qkv.weight", qkv_rows, d.hidden, b.attn_qkv); !r)
                return Failure{r.error()};
        } else {
            if (auto r = get("attn_q.weight", d.q_dim,  d.hidden, b.attn_q); !r) return Failure{r.error()};
            if (auto r = get("attn_k.weight", d.kv_dim, d.hidden, b.attn_k); !r) return Failure{r.error()};
            if (auto r = get("attn_v.weight", d.kv_dim, d.hidden, b.attn_v); !r) return Failure{r.error()};
        }

        if (!b.is_recurrent) {
            if (auto r = get("attn_output.weight", d.hidden, d.q_dim, b.attn_out); !r)
                return Failure{r.error()};
            if (p.attn_gate) {
                if (auto r = get("attn_gate.weight", d.n_heads, d.hidden, b.attn_gate); !r)
                    return Failure{r.error()};
            }
        }

        // The norm on the way into the FFN. qwen35 names it
        // post_attention_norm because it sits after the attention residual
        // has been added, but that is the same position in the block.
        if (auto r = get("ffn_norm.weight", d.hidden, 0, b.ffn_norm); !r) {
            if (auto r2 = get("post_attention_norm.weight", d.hidden, 0, b.ffn_norm); !r2)
                return Failure{r.error()};
        }

        switch (p.ffn) {
        case FfnKind::DenseSwiGLU:
            if (auto r = get("ffn_gate.weight", d.ffn_hidden, d.hidden, b.ffn_gate); !r) return Failure{r.error()};
            if (auto r = get("ffn_up.weight",   d.ffn_hidden, d.hidden, b.ffn_up);   !r) return Failure{r.error()};
            if (auto r = get("ffn_down.weight", d.hidden, d.ffn_hidden, b.ffn_down); !r) return Failure{r.error()};
            break;

        case FfnKind::FusedSwiGLU:
            // ffn_up holds [gate || up] stacked on the output axis.
            if (auto r = get("ffn_up.weight",   2 * d.ffn_hidden, d.hidden, b.ffn_up); !r) return Failure{r.error()};
            if (auto r = get("ffn_down.weight", d.hidden, d.ffn_hidden, b.ffn_down);   !r) return Failure{r.error()};
            break;

        case FfnKind::SparseMoE: {
            if (auto r = get("ffn_gate_inp.weight", p.n_expert, d.hidden, b.ffn_gate_inp); !r)
                return Failure{r.error()};
            if (auto r = resolve_experts(gguf, pfx + "ffn_gate_exps.weight",
                                         p.expert_ffn, d.hidden, p.n_expert); r) b.ffn_gate_exps = *r;
            else return Failure{r.error()};
            if (auto r = resolve_experts(gguf, pfx + "ffn_up_exps.weight",
                                         p.expert_ffn, d.hidden, p.n_expert); r) b.ffn_up_exps = *r;
            else return Failure{r.error()};
            if (auto r = resolve_experts(gguf, pfx + "ffn_down_exps.weight",
                                         d.hidden, p.expert_ffn, p.n_expert); r) b.ffn_down_exps = *r;
            else return Failure{r.error()};
            break;
        }
        }

        // Optional biases (Qwen2 sometimes has them).
        b.attn_q_bias = resolve_optional(gguf, pfx + "attn_q.bias", d.q_dim,  0);
        b.attn_k_bias = resolve_optional(gguf, pfx + "attn_k.bias", d.kv_dim, 0);
        b.attn_v_bias = resolve_optional(gguf, pfx + "attn_v.bias", d.kv_dim, 0);

        // Qwen3 Q/K normalization.
        b.attn_q_norm = resolve_optional(gguf, pfx + "attn_q_norm.weight", d.head_dim, 0);
        b.attn_k_norm = resolve_optional(gguf, pfx + "attn_k_norm.weight", d.head_dim, 0);
    }
    return t;
}

} // namespace ultima::model
