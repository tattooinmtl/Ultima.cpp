#include "ultima/model/qwen_block.hpp"

#include "ultima/kernels/elementwise.hpp"
#include "ultima/kernels/norms.hpp"
#include "ultima/kernels/rope.hpp"
#include "ultima/kernels/softmax.hpp"
#include "ultima/model/weight_dispatch.hpp"
#include "ultima/runtime/thread_pool.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ultima::model {

namespace {

using core::ErrorCode;
using core::Failure;
using core::Result;
using core::fail;

// Add an optional bias vector to `y` in place; no-op if `bias` is null.
inline void add_bias(float* y, const float* bias, std::size_t n) {
    if (!bias) return;
    for (std::size_t i = 0; i < n; ++i) y[i] += bias[i];
}

// RMSNorm each head of a [n_heads, head_dim] tensor with a per-head-dim
// weight vector `w` of length head_dim (Qwen3 Q/K norm variant).
void rmsnorm_per_head(float* x, std::size_t n_heads, std::size_t head_dim,
                      const float* w, float eps) {
    for (std::size_t h = 0; h < n_heads; ++h) {
        kernels::rmsnorm_f32(x + h * head_dim, w,
                             x + h * head_dim, head_dim, eps);
    }
}

// Scaled dot-product attention with GQA + causal (implicit via pos_end).
// - q: [n_heads, head_dim] contiguous
// - K, V: [n_kv_heads, n_ctx, head_dim] (f32 storage in v0.1); reader is
//   the KVCache LayerView whose stride_head == n_ctx * head_dim.
// - out: [n_heads, head_dim] contiguous
// - scratch: at least pos_end floats
void gqa_attention(const float* q,
                   const float* K, const float* V,
                   std::size_t stride_head,   // element stride between heads
                   std::size_t pos_end,
                   std::size_t n_heads,
                   std::size_t n_kv_heads,
                   std::size_t head_dim,
                   float*      out,
                   float*      scratch)
{
    const float inv_sqrt_hd = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const std::size_t group = n_heads / n_kv_heads;

    for (std::size_t h = 0; h < n_heads; ++h) {
        const std::size_t hkv = h / group;
        const float* q_h = q + h * head_dim;
        const float* K_h = K + hkv * stride_head;
        const float* V_h = V + hkv * stride_head;

        // Scores: dot(q_h, K_h[t]) * inv_sqrt_hd for t in [0, pos_end)
        for (std::size_t t = 0; t < pos_end; ++t) {
            float acc = 0.0f;
            const float* k_t = K_h + t * head_dim;
            for (std::size_t i = 0; i < head_dim; ++i) acc += q_h[i] * k_t[i];
            scratch[t] = acc * inv_sqrt_hd;
        }

        // Softmax in place over pos_end.
        kernels::softmax_f32(scratch, scratch, pos_end);

        // Weighted sum of V.
        float* out_h = out + h * head_dim;
        for (std::size_t i = 0; i < head_dim; ++i) out_h[i] = 0.0f;
        for (std::size_t t = 0; t < pos_end; ++t) {
            const float w = scratch[t];
            const float* v_t = V_h + t * head_dim;
            for (std::size_t i = 0; i < head_dim; ++i) out_h[i] += w * v_t[i];
        }
    }
}

} // namespace

void BlockScratch::init(const ModelDims& d) {
    x.assign(d.hidden, 0.0f);
    x_norm.assign(d.hidden, 0.0f);
    q_full.assign(d.hidden, 0.0f);
    const std::size_t kv_dim = d.n_kv_heads * d.head_dim;
    k_full.assign(kv_dim, 0.0f);
    v_full.assign(kv_dim, 0.0f);
    attn_out.assign(d.hidden, 0.0f);
    attn_scores.assign(std::max<std::size_t>(d.n_ctx_train, 1), 0.0f);
    gate.assign(d.ffn_hidden, 0.0f);
    up.assign(d.ffn_hidden, 0.0f);
    logits_tmp.assign(d.vocab, 0.0f);

    attn_norm.assign(d.n_layers * d.hidden, 0.0f);
    ffn_norm.assign(d.n_layers * d.hidden, 0.0f);
    output_norm_f32.assign(d.hidden, 0.0f);

    attn_q_bias.assign(d.n_layers * d.hidden,  0.0f);
    attn_k_bias.assign(d.n_layers * kv_dim,    0.0f);
    attn_v_bias.assign(d.n_layers * kv_dim,    0.0f);
    has_biases.assign(d.n_layers, false);

    attn_q_norm.assign(d.n_layers * d.head_dim, 0.0f);
    attn_k_norm.assign(d.n_layers * d.head_dim, 0.0f);
    has_qk_norm.assign(d.n_layers, false);
}

Result<void>
preload_scratch_from_tensors(BlockScratch& s,
                             const QwenTensors& t,
                             const ModelDims& d) {
    // Output norm.
    if (auto r = dequant_1d(t.output_norm.dtype, t.output_norm.data,
                            s.output_norm_f32.data(), d.hidden); !r) return r;

    const std::size_t kv_dim = d.n_kv_heads * d.head_dim;
    for (std::size_t i = 0; i < d.n_layers; ++i) {
        const auto& b = t.blocks[i];
        if (auto r = dequant_1d(b.attn_norm.dtype, b.attn_norm.data,
                                s.attn_norm.data() + i * d.hidden, d.hidden); !r) return r;
        if (auto r = dequant_1d(b.ffn_norm.dtype, b.ffn_norm.data,
                                s.ffn_norm.data() + i * d.hidden, d.hidden); !r) return r;

        const bool any_bias = b.attn_q_bias || b.attn_k_bias || b.attn_v_bias;
        s.has_biases[i] = any_bias;
        if (b.attn_q_bias) {
            if (auto r = dequant_1d(b.attn_q_bias->dtype, b.attn_q_bias->data,
                                    s.attn_q_bias.data() + i * d.hidden, d.hidden); !r) return r;
        }
        if (b.attn_k_bias) {
            if (auto r = dequant_1d(b.attn_k_bias->dtype, b.attn_k_bias->data,
                                    s.attn_k_bias.data() + i * kv_dim, kv_dim); !r) return r;
        }
        if (b.attn_v_bias) {
            if (auto r = dequant_1d(b.attn_v_bias->dtype, b.attn_v_bias->data,
                                    s.attn_v_bias.data() + i * kv_dim, kv_dim); !r) return r;
        }

        const bool qk = b.attn_q_norm && b.attn_k_norm;
        s.has_qk_norm[i] = qk;
        if (qk) {
            if (auto r = dequant_1d(b.attn_q_norm->dtype, b.attn_q_norm->data,
                                    s.attn_q_norm.data() + i * d.head_dim, d.head_dim); !r) return r;
            if (auto r = dequant_1d(b.attn_k_norm->dtype, b.attn_k_norm->data,
                                    s.attn_k_norm.data() + i * d.head_dim, d.head_dim); !r) return r;
        }
    }
    return {};
}

Result<void>
qwen_block_forward(BlockScratch& s,
                   const QwenTensors::Block& block,
                   const ModelDims&  dims,
                   const RopeConfig& rope,
                   kv_cache::KVCache& kv,
                   kv_cache::SlotHandle slot,
                   std::size_t layer,
                   std::size_t absolute_position,
                   runtime::ThreadPool& pool)
{
    const std::size_t kv_dim = dims.n_kv_heads * dims.head_dim;

    // Save residual stream.
    std::vector<float> residual(s.x);

    // 1. Attention pre-norm.
    kernels::rmsnorm_f32(s.x.data(),
                         s.attn_norm.data() + layer * dims.hidden,
                         s.x_norm.data(), dims.hidden, dims.rms_eps);

    // 2. Q/K/V projections.
    if (auto r = matvec_dispatch_threaded(pool, block.attn_q.dtype, block.attn_q.data,
                                          s.x_norm.data(), s.q_full.data(),
                                          dims.hidden, dims.hidden); !r) return r;
    if (auto r = matvec_dispatch_threaded(pool, block.attn_k.dtype, block.attn_k.data,
                                          s.x_norm.data(), s.k_full.data(),
                                          kv_dim, dims.hidden); !r) return r;
    if (auto r = matvec_dispatch_threaded(pool, block.attn_v.dtype, block.attn_v.data,
                                          s.x_norm.data(), s.v_full.data(),
                                          kv_dim, dims.hidden); !r) return r;

    if (s.has_biases[layer]) {
        add_bias(s.q_full.data(),
                 block.attn_q_bias ? s.attn_q_bias.data() + layer * dims.hidden : nullptr,
                 dims.hidden);
        add_bias(s.k_full.data(),
                 block.attn_k_bias ? s.attn_k_bias.data() + layer * kv_dim : nullptr,
                 kv_dim);
        add_bias(s.v_full.data(),
                 block.attn_v_bias ? s.attn_v_bias.data() + layer * kv_dim : nullptr,
                 kv_dim);
    }

    // 3. Qwen3 Q/K RMSNorm per head.
    if (s.has_qk_norm[layer]) {
        rmsnorm_per_head(s.q_full.data(), dims.n_heads,    dims.head_dim,
                         s.attn_q_norm.data() + layer * dims.head_dim, dims.rms_eps);
        rmsnorm_per_head(s.k_full.data(), dims.n_kv_heads, dims.head_dim,
                         s.attn_k_norm.data() + layer * dims.head_dim, dims.rms_eps);
    }

    // 4. RoPE on Q and K at the absolute position.
    kernels::rope_f32(s.q_full.data(), dims.n_heads, dims.head_dim,
                      rope.rope_dim, absolute_position, rope.freq_base);
    kernels::rope_f32(s.k_full.data(), dims.n_kv_heads, dims.head_dim,
                      rope.rope_dim, absolute_position, rope.freq_base);

    // 5. Write K, V into cache at absolute position; then read up to pos+1.
    kv.write_at(slot, layer, absolute_position, s.k_full.data(), s.v_full.data());
    auto lv = kv.read_at(slot, layer, absolute_position + 1);

    // 6. Attention (K, V read as f32 from the cache).
    const float* K = static_cast<const float*>(lv.k);
    const float* V = static_cast<const float*>(lv.v);
    gqa_attention(s.q_full.data(), K, V,
                  lv.stride_head, lv.pos,
                  dims.n_heads, dims.n_kv_heads, dims.head_dim,
                  s.attn_out.data(), s.attn_scores.data());

    // 7. Output projection + residual add.
    if (auto r = matvec_dispatch_threaded(pool, block.attn_out.dtype, block.attn_out.data,
                                          s.attn_out.data(), s.x.data(),
                                          dims.hidden, dims.hidden); !r) return r;
    kernels::add_f32(s.x.data(), residual.data(), s.x.data(), dims.hidden);

    // 8. FFN pre-norm.
    std::copy(s.x.begin(), s.x.end(), residual.begin());
    kernels::rmsnorm_f32(s.x.data(),
                         s.ffn_norm.data() + layer * dims.hidden,
                         s.x_norm.data(), dims.hidden, dims.rms_eps);

    // 9. Gate, Up projections.
    if (auto r = matvec_dispatch_threaded(pool, block.ffn_gate.dtype, block.ffn_gate.data,
                                          s.x_norm.data(), s.gate.data(),
                                          dims.ffn_hidden, dims.hidden); !r) return r;
    if (auto r = matvec_dispatch_threaded(pool, block.ffn_up.dtype, block.ffn_up.data,
                                          s.x_norm.data(), s.up.data(),
                                          dims.ffn_hidden, dims.hidden); !r) return r;

    // 10. SwiGLU: silu(gate) * up -> gate scratch.
    kernels::swiglu_f32(s.gate.data(), s.up.data(), s.gate.data(), dims.ffn_hidden);

    // 11. Down projection.
    if (auto r = matvec_dispatch_threaded(pool, block.ffn_down.dtype, block.ffn_down.data,
                                          s.gate.data(), s.x.data(),
                                          dims.hidden, dims.ffn_hidden); !r) return r;

    // 12. Residual add.
    kernels::add_f32(s.x.data(), residual.data(), s.x.data(), dims.hidden);
    return {};
}

} // namespace ultima::model
