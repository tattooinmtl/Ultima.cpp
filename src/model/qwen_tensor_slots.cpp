#include "ultima/model/qwen_tensor_slots.hpp"

#include <fmt/format.h>

namespace ultima::model {

namespace {

using core::ErrorCode;
using core::Failure;
using core::Result;
using core::fail;

Result<WeightRef>
resolve(const LoadedModel& gguf, const std::string& name,
        std::size_t expected_rows, std::size_t expected_cols) {
    auto tv = gguf.tensor(name);
    if (!tv) {
        return fail(ErrorCode::MissingRequiredMetadata,
                    fmt::format("missing tensor '{}'", name),
                    "qwen_tensor_slots");
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
                        "qwen_tensor_slots");
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
                        "qwen_tensor_slots");
        }
        w.rows = expected_rows;
        w.cols = expected_cols;
    }
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

Result<QwenTensors>
resolve_qwen_tensors(const LoadedModel& gguf,
                     const std::string& arch,
                     std::size_t n_layers,
                     std::size_t hidden,
                     std::size_t n_kv_heads,
                     std::size_t head_dim,
                     std::size_t ffn_hidden,
                     std::size_t vocab) {
    (void)arch;   // reserved for future per-arch renames
    QwenTensors t{};

    // Top-level tensors.
    if (auto r = resolve(gguf, "token_embd.weight", vocab, hidden); r) t.token_embd = *r;
    else return Failure{r.error()};

    if (auto r = resolve(gguf, "output_norm.weight", hidden, 0); r) t.output_norm = *r;
    else return Failure{r.error()};

    // Output head: try output.weight first, fall back to token_embd (tied).
    if (auto opt = resolve_optional(gguf, "output.weight", vocab, hidden); opt) {
        t.output_head = *opt;
        t.output_head_is_tied = false;
    } else {
        t.output_head = t.token_embd;
        t.output_head_is_tied = true;
    }

    const std::size_t kv_dim = n_kv_heads * head_dim;

    t.blocks.resize(n_layers);
    for (std::size_t i = 0; i < n_layers; ++i) {
        const std::string pfx = fmt::format("blk.{}.", i);
        QwenTensors::Block& b = t.blocks[i];

        auto get = [&](const char* suffix, std::size_t rr, std::size_t cc, WeightRef& out) -> Result<void> {
            auto r = resolve(gguf, pfx + suffix, rr, cc);
            if (!r) return Failure{r.error()};
            out = *r;
            return {};
        };

        if (auto r = get("attn_norm.weight",    hidden, 0,       b.attn_norm);   !r) return Failure{r.error()};
        if (auto r = get("attn_q.weight",       hidden, hidden,  b.attn_q);      !r) return Failure{r.error()};
        if (auto r = get("attn_k.weight",       kv_dim, hidden,  b.attn_k);      !r) return Failure{r.error()};
        if (auto r = get("attn_v.weight",       kv_dim, hidden,  b.attn_v);      !r) return Failure{r.error()};
        if (auto r = get("attn_output.weight",  hidden, hidden,  b.attn_out);    !r) return Failure{r.error()};
        if (auto r = get("ffn_norm.weight",     hidden, 0,       b.ffn_norm);    !r) return Failure{r.error()};
        if (auto r = get("ffn_gate.weight",     ffn_hidden, hidden, b.ffn_gate); !r) return Failure{r.error()};
        if (auto r = get("ffn_up.weight",       ffn_hidden, hidden, b.ffn_up);   !r) return Failure{r.error()};
        if (auto r = get("ffn_down.weight",     hidden, ffn_hidden, b.ffn_down); !r) return Failure{r.error()};

        // Optional biases (Qwen2 sometimes has them).
        b.attn_q_bias = resolve_optional(gguf, pfx + "attn_q.bias", hidden, 0);
        b.attn_k_bias = resolve_optional(gguf, pfx + "attn_k.bias", kv_dim, 0);
        b.attn_v_bias = resolve_optional(gguf, pfx + "attn_v.bias", kv_dim, 0);

        // Qwen3 Q/K normalization.
        b.attn_q_norm = resolve_optional(gguf, pfx + "attn_q_norm.weight", head_dim, 0);
        b.attn_k_norm = resolve_optional(gguf, pfx + "attn_k_norm.weight", head_dim, 0);
    }
    return t;
}

} // namespace ultima::model
