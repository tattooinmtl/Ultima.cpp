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
#include <numeric>

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

inline void scale_in_place(float* y, float k, std::size_t n) {
    if (k == 1.0f) return;
    for (std::size_t i = 0; i < n; ++i) y[i] *= k;
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
// - scratch: at least (pos_end - pos_begin) floats
// - scale: multiplies the raw dot products (1/sqrt(head_dim) for most archs;
//   Granite ships an explicit value)
// - pos_begin: first key position to attend to. Non-zero implements sliding
//   -window attention; the keys below it stay in the cache but are masked out.
void gqa_attention(const float* q,
                   const float* K, const float* V,
                   std::size_t stride_head,   // element stride between heads
                   std::size_t pos_begin,
                   std::size_t pos_end,
                   std::size_t n_heads,
                   std::size_t n_kv_heads,
                   std::size_t head_dim,
                   float       scale,
                   float*      out,
                   float*      scratch)
{
    const std::size_t group = n_heads / n_kv_heads;
    const std::size_t span  = pos_end - pos_begin;

    for (std::size_t h = 0; h < n_heads; ++h) {
        const std::size_t hkv = h / group;
        const float* q_h = q + h * head_dim;
        const float* K_h = K + hkv * stride_head;
        const float* V_h = V + hkv * stride_head;

        // Scores: dot(q_h, K_h[t]) * scale for t in [pos_begin, pos_end)
        for (std::size_t t = 0; t < span; ++t) {
            float acc = 0.0f;
            const float* k_t = K_h + (pos_begin + t) * head_dim;
            for (std::size_t i = 0; i < head_dim; ++i) acc += q_h[i] * k_t[i];
            scratch[t] = acc * scale;
        }

        // Softmax in place over the visible span.
        kernels::softmax_f32(scratch, scratch, span);

        // Weighted sum of V.
        float* out_h = out + h * head_dim;
        for (std::size_t i = 0; i < head_dim; ++i) out_h[i] = 0.0f;
        for (std::size_t t = 0; t < span; ++t) {
            const float w = scratch[t];
            const float* v_t = V_h + (pos_begin + t) * head_dim;
            for (std::size_t i = 0; i < head_dim; ++i) out_h[i] += w * v_t[i];
        }
    }
}

// Pick the `k` largest entries of `probs` (length n). Writes indices into
// `ids` and their probabilities into `weights`. Partial selection only —
// ordering among the winners is not meaningful to the caller.
void top_k_experts(const float* probs, std::size_t n, std::size_t k,
                   std::uint32_t* ids, float* weights) {
    // n_expert is 128 for Qwen3-Coder and k is 8, so a k-pass selection beats
    // sorting the whole router output.
    std::vector<bool> taken(n, false);
    for (std::size_t s = 0; s < k; ++s) {
        std::size_t best = n;
        float best_p = 0.0f;
        for (std::size_t e = 0; e < n; ++e) {
            if (taken[e]) continue;
            if (best == n || probs[e] > best_p) { best = e; best_p = probs[e]; }
        }
        taken[best] = true;
        ids[s]      = static_cast<std::uint32_t>(best);
        weights[s]  = best_p;
    }
}

Result<void>
ffn_dense(BlockScratch& s, const ModelTensors::Block& b,
          const ModelDims& d, runtime::ThreadPool& pool) {
    if (auto r = matvec_dispatch_threaded(pool, b.ffn_gate.dtype, b.ffn_gate.data,
                                          s.x_norm.data(), s.gate.data(),
                                          d.ffn_hidden, d.hidden); !r) return r;
    if (auto r = matvec_dispatch_threaded(pool, b.ffn_up.dtype, b.ffn_up.data,
                                          s.x_norm.data(), s.up.data(),
                                          d.ffn_hidden, d.hidden); !r) return r;
    kernels::swiglu_f32(s.gate.data(), s.up.data(), s.gate.data(), d.ffn_hidden);
    return matvec_dispatch_threaded(pool, b.ffn_down.dtype, b.ffn_down.data,
                                    s.gate.data(), s.x.data(),
                                    d.hidden, d.ffn_hidden);
}

// Phi-3 packs gate and up into one matrix stacked on the output axis, with
// gate first. One matvec produces both halves.
Result<void>
ffn_fused(BlockScratch& s, const ModelTensors::Block& b,
          const ModelDims& d, runtime::ThreadPool& pool) {
    if (auto r = matvec_dispatch_threaded(pool, b.ffn_up.dtype, b.ffn_up.data,
                                          s.x_norm.data(), s.gate.data(),
                                          2 * d.ffn_hidden, d.hidden); !r) return r;
    float* gate_half = s.gate.data();
    float* up_half   = s.gate.data() + d.ffn_hidden;
    kernels::swiglu_f32(gate_half, up_half, gate_half, d.ffn_hidden);
    return matvec_dispatch_threaded(pool, b.ffn_down.dtype, b.ffn_down.data,
                                    gate_half, s.x.data(),
                                    d.hidden, d.ffn_hidden);
}

// Sparse MoE: softmax the router over all experts, take the top-k, renormalize
// those k to sum to 1, then accumulate each expert's SwiGLU output weighted by
// its share. Matches llama.cpp's build_moe_ffn with norm_w=true, scale_w=false.
Result<void>
ffn_moe(BlockScratch& s, const ModelTensors::Block& b,
        const ModelDims& d, const ArchParams& p, runtime::ThreadPool& pool) {
    // Router over the full expert set.
    if (auto r = matvec_dispatch_threaded(pool, b.ffn_gate_inp.dtype, b.ffn_gate_inp.data,
                                          s.x_norm.data(), s.router_probs.data(),
                                          p.n_expert, d.hidden); !r) return r;
    kernels::softmax_f32(s.router_probs.data(), s.router_probs.data(), p.n_expert);

    top_k_experts(s.router_probs.data(), p.n_expert, p.n_expert_used,
                  s.expert_ids.data(), s.expert_weights.data());

    float wsum = 0.0f;
    for (std::size_t i = 0; i < p.n_expert_used; ++i) wsum += s.expert_weights[i];
    const float inv_wsum = (wsum > 0.0f) ? (1.0f / wsum) : 1.0f;

    std::fill(s.moe_accum.begin(), s.moe_accum.end(), 0.0f);

    for (std::size_t i = 0; i < p.n_expert_used; ++i) {
        const std::size_t e = s.expert_ids[i];
        const float w = s.expert_weights[i] * inv_wsum;

        if (auto r = matvec_dispatch_threaded(pool, b.ffn_gate_exps.dtype,
                                              b.ffn_gate_exps.expert(e),
                                              s.x_norm.data(), s.gate.data(),
                                              p.expert_ffn, d.hidden); !r) return r;
        if (auto r = matvec_dispatch_threaded(pool, b.ffn_up_exps.dtype,
                                              b.ffn_up_exps.expert(e),
                                              s.x_norm.data(), s.up.data(),
                                              p.expert_ffn, d.hidden); !r) return r;
        kernels::swiglu_f32(s.gate.data(), s.up.data(), s.gate.data(), p.expert_ffn);

        if (auto r = matvec_dispatch_threaded(pool, b.ffn_down_exps.dtype,
                                              b.ffn_down_exps.expert(e),
                                              s.gate.data(), s.expert_out.data(),
                                              d.hidden, p.expert_ffn); !r) return r;

        for (std::size_t j = 0; j < d.hidden; ++j) s.moe_accum[j] += w * s.expert_out[j];
    }

    std::copy(s.moe_accum.begin(), s.moe_accum.end(), s.x.begin());
    return {};
}

} // namespace

void BlockScratch::init(const ModelDims& d, const ArchParams& p) {
    x.assign(d.hidden, 0.0f);
    residual.assign(d.hidden, 0.0f);
    x_norm.assign(d.hidden, 0.0f);
    q_full.assign(d.q_dim, 0.0f);
    k_full.assign(d.kv_dim, 0.0f);
    v_full.assign(d.kv_dim, 0.0f);
    attn_out.assign(d.q_dim, 0.0f);
    attn_scores.assign(std::max<std::size_t>(d.n_ctx_train, 1), 0.0f);
    logits_tmp.assign(d.vocab, 0.0f);

    if (p.qkv == QkvKind::Fused) {
        qkv_full.assign(d.q_dim + 2 * d.kv_dim, 0.0f);
    }
    if (p.attn_gate) {
        head_gate.assign(d.n_heads, 0.0f);
    }

    // The FFN scratch has to cover whichever path this architecture takes:
    // the fused matvec writes gate||up into one buffer, MoE works at the
    // narrower per-expert width.
    switch (p.ffn) {
    case FfnKind::DenseSwiGLU:
        gate.assign(d.ffn_hidden, 0.0f);
        up.assign(d.ffn_hidden, 0.0f);
        break;
    case FfnKind::FusedSwiGLU:
        gate.assign(2 * d.ffn_hidden, 0.0f);
        up.clear();
        break;
    case FfnKind::SparseMoE:
        gate.assign(p.expert_ffn, 0.0f);
        up.assign(p.expert_ffn, 0.0f);
        router_probs.assign(p.n_expert, 0.0f);
        expert_ids.assign(p.n_expert_used, 0u);
        expert_weights.assign(p.n_expert_used, 0.0f);
        expert_out.assign(d.hidden, 0.0f);
        moe_accum.assign(d.hidden, 0.0f);
        break;
    }

    attn_norm.assign(d.n_layers * d.hidden, 0.0f);
    ffn_norm.assign(d.n_layers * d.hidden, 0.0f);
    output_norm_f32.assign(d.hidden, 0.0f);

    attn_q_bias.assign(d.n_layers * d.q_dim,  0.0f);
    attn_k_bias.assign(d.n_layers * d.kv_dim, 0.0f);
    attn_v_bias.assign(d.n_layers * d.kv_dim, 0.0f);
    has_biases.assign(d.n_layers, false);

    attn_q_norm.assign(d.n_layers * d.head_dim, 0.0f);
    attn_k_norm.assign(d.n_layers * d.head_dim, 0.0f);
    has_qk_norm.assign(d.n_layers, false);
}

Result<void>
preload_scratch_from_tensors(BlockScratch& s,
                             const ModelTensors& t,
                             const ModelDims& d) {
    // Output norm.
    if (auto r = dequant_1d(t.output_norm.dtype, t.output_norm.data,
                            s.output_norm_f32.data(), d.hidden); !r) return r;

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
                                    s.attn_q_bias.data() + i * d.q_dim, d.q_dim); !r) return r;
        }
        if (b.attn_k_bias) {
            if (auto r = dequant_1d(b.attn_k_bias->dtype, b.attn_k_bias->data,
                                    s.attn_k_bias.data() + i * d.kv_dim, d.kv_dim); !r) return r;
        }
        if (b.attn_v_bias) {
            if (auto r = dequant_1d(b.attn_v_bias->dtype, b.attn_v_bias->data,
                                    s.attn_v_bias.data() + i * d.kv_dim, d.kv_dim); !r) return r;
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
block_forward(BlockScratch& s,
              const ModelTensors::Block& block,
              const ModelDims&  dims,
              const ArchParams& params,
              const RopeConfig& rope,
              kv_cache::KVCache& kv,
              kv_cache::SlotHandle slot,
              std::size_t layer,
              std::size_t absolute_position,
              runtime::ThreadPool& pool)
{
    // Save residual stream.
    std::copy(s.x.begin(), s.x.end(), s.residual.begin());

    // 1. Attention pre-norm.
    kernels::rmsnorm_f32(s.x.data(),
                         s.attn_norm.data() + layer * dims.hidden,
                         s.x_norm.data(), dims.hidden, dims.rms_eps);

    // 2. Q/K/V projections.
    if (params.qkv == QkvKind::Fused) {
        // One matrix produces [Q || K || V] stacked on the output axis.
        const std::size_t qkv_rows = dims.q_dim + 2 * dims.kv_dim;
        if (auto r = matvec_dispatch_threaded(pool, block.attn_qkv.dtype, block.attn_qkv.data,
                                              s.x_norm.data(), s.qkv_full.data(),
                                              qkv_rows, dims.hidden); !r) return r;
        std::memcpy(s.q_full.data(), s.qkv_full.data(),
                    dims.q_dim * sizeof(float));
        std::memcpy(s.k_full.data(), s.qkv_full.data() + dims.q_dim,
                    dims.kv_dim * sizeof(float));
        std::memcpy(s.v_full.data(), s.qkv_full.data() + dims.q_dim + dims.kv_dim,
                    dims.kv_dim * sizeof(float));
    } else {
        if (auto r = matvec_dispatch_threaded(pool, block.attn_q.dtype, block.attn_q.data,
                                              s.x_norm.data(), s.q_full.data(),
                                              dims.q_dim, dims.hidden); !r) return r;
        if (auto r = matvec_dispatch_threaded(pool, block.attn_k.dtype, block.attn_k.data,
                                              s.x_norm.data(), s.k_full.data(),
                                              dims.kv_dim, dims.hidden); !r) return r;
        if (auto r = matvec_dispatch_threaded(pool, block.attn_v.dtype, block.attn_v.data,
                                              s.x_norm.data(), s.v_full.data(),
                                              dims.kv_dim, dims.hidden); !r) return r;
    }

    if (s.has_biases[layer]) {
        add_bias(s.q_full.data(),
                 block.attn_q_bias ? s.attn_q_bias.data() + layer * dims.q_dim : nullptr,
                 dims.q_dim);
        add_bias(s.k_full.data(),
                 block.attn_k_bias ? s.attn_k_bias.data() + layer * dims.kv_dim : nullptr,
                 dims.kv_dim);
        add_bias(s.v_full.data(),
                 block.attn_v_bias ? s.attn_v_bias.data() + layer * dims.kv_dim : nullptr,
                 dims.kv_dim);
    }

    // 3. Qwen3 Q/K RMSNorm per head.
    if (s.has_qk_norm[layer]) {
        rmsnorm_per_head(s.q_full.data(), dims.n_heads,    dims.head_dim,
                         s.attn_q_norm.data() + layer * dims.head_dim, dims.rms_eps);
        rmsnorm_per_head(s.k_full.data(), dims.n_kv_heads, dims.head_dim,
                         s.attn_k_norm.data() + layer * dims.head_dim, dims.rms_eps);
    }

    // 4. RoPE on Q and K at the absolute position. Sliding-window layers can
    //    rotate a different number of dims at a different base.
    const bool  is_swa = params.is_swa_layer(layer);
    const bool  neox   = (params.rope_style == RopeStyle::NeoX);
    const float* ff    = rope.freq_factors.empty() ? nullptr : rope.freq_factors.data();
    const std::size_t rope_dim  = is_swa ? rope.rope_dim_swa  : rope.rope_dim;
    const float       freq_base = is_swa ? rope.freq_base_swa : rope.freq_base;
    kernels::rope_f32_ext(s.q_full.data(), dims.n_heads, dims.head_dim,
                          rope_dim, absolute_position, freq_base,
                          neox, ff, rope.attn_factor);
    kernels::rope_f32_ext(s.k_full.data(), dims.n_kv_heads, dims.head_dim,
                          rope_dim, absolute_position, freq_base,
                          neox, ff, rope.attn_factor);

    // 5. Write K, V into cache at absolute position; then read up to pos+1.
    kv.write_at(slot, layer, absolute_position, s.k_full.data(), s.v_full.data());
    auto lv = kv.read_at(slot, layer, absolute_position + 1);

    // 6. Attention (K, V read as f32 from the cache).
    const float attn_scale = (params.attn_scale != 0.0f)
        ? params.attn_scale
        : 1.0f / std::sqrt(static_cast<float>(dims.head_dim));
    const float* K = static_cast<const float*>(lv.k);
    const float* V = static_cast<const float*>(lv.v);

    // Sliding-window layers see only the most recent `sliding_window` keys.
    // The older entries stay in the cache — masking here costs nothing and
    // avoids a second, ring-buffered cache geometry.
    std::size_t pos_begin = 0;
    if (is_swa && params.sliding_window > 0 && lv.pos > params.sliding_window) {
        pos_begin = lv.pos - params.sliding_window;
    }
    gqa_attention(s.q_full.data(), K, V,
                  lv.stride_head, pos_begin, lv.pos,
                  dims.n_heads, dims.n_kv_heads, dims.head_dim,
                  attn_scale,
                  s.attn_out.data(), s.attn_scores.data());

    // 6b. Per-head sigmoid gate on the attention output. The gate is one
    //     scalar per head derived from the normed residual, so it has to be
    //     applied here, while the output is still laid out per head.
    if (params.attn_gate) {
        if (auto r = matvec_dispatch_threaded(pool, block.attn_gate.dtype,
                                              block.attn_gate.data,
                                              s.x_norm.data(), s.head_gate.data(),
                                              dims.n_heads, dims.hidden); !r) return r;
        for (std::size_t h = 0; h < dims.n_heads; ++h) {
            const float g = 1.0f / (1.0f + std::exp(-s.head_gate[h]));
            float* out_h = s.attn_out.data() + h * dims.head_dim;
            for (std::size_t i = 0; i < dims.head_dim; ++i) out_h[i] *= g;
        }
    }

    // 7. Output projection + residual add.
    if (auto r = matvec_dispatch_threaded(pool, block.attn_out.dtype, block.attn_out.data,
                                          s.attn_out.data(), s.x.data(),
                                          dims.hidden, dims.q_dim); !r) return r;
    scale_in_place(s.x.data(), params.residual_scale, dims.hidden);
    kernels::add_f32(s.x.data(), s.residual.data(), s.x.data(), dims.hidden);

    // 8. FFN pre-norm.
    std::copy(s.x.begin(), s.x.end(), s.residual.begin());
    kernels::rmsnorm_f32(s.x.data(),
                         s.ffn_norm.data() + layer * dims.hidden,
                         s.x_norm.data(), dims.hidden, dims.rms_eps);

    // 9. Feed-forward, writing its output into s.x.
    switch (params.ffn) {
    case FfnKind::DenseSwiGLU:
        if (auto r = ffn_dense(s, block, dims, pool); !r) return r;
        break;
    case FfnKind::FusedSwiGLU:
        if (auto r = ffn_fused(s, block, dims, pool); !r) return r;
        break;
    case FfnKind::SparseMoE:
        if (auto r = ffn_moe(s, block, dims, params, pool); !r) return r;
        break;
    }

    // 10. Residual add.
    scale_in_place(s.x.data(), params.residual_scale, dims.hidden);
    kernels::add_f32(s.x.data(), s.residual.data(), s.x.data(), dims.hidden);
    return {};
}

} // namespace ultima::model
