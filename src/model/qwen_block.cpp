#include "ultima/model/qwen_block.hpp"

#include "ultima/kernels/delta_net.hpp"
#include "ultima/kernels/hadamard.hpp"
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
#include <cstdlib>
#include <numeric>

namespace ultima::model {

namespace {

using core::ErrorCode;
using core::Failure;
using core::Result;
using core::fail;

// Diagnostic checksum, printed only when ULTIMA_TRACE is set in the
// environment. The format mirrors llama.cpp's eval-callback output so a run
// can be diffed against the reference line by line — which is the only
// practical way to localize a fault in a 32-layer stack that produces
// fluent-looking output either way.
inline void trace(const char* name, std::size_t layer,
                  const float* x, std::size_t n) {
    static const bool on = std::getenv("ULTIMA_TRACE") != nullptr;
    if (!on) return;
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) sum += x[i];
    std::fprintf(stderr, "ultima_trace: %22s-%zu = {%zu} sum = %.6f\n",
                 name, layer, n, sum);
}

// Returns the activation to feed a matvec. For a weight stored with the
// inverse rotation folded in, that is the rotated activation; for every
// other weight it is `x` untouched.
//
// Recomputed per call rather than memoized across the projections that
// share an activation: the transform is O(width log block) against a
// matvec's O(rows * width), which at these shapes is well under a percent.
const float* folded_input(BlockScratch& s, const WeightRef& w,
                          const float* x, std::size_t width) {
    if (!w.hadamard_folded || !s.fold) return x;
    kernels::hadamard_fold_f32(x, s.fold_buf.data(), width,
                               s.fold->block, s.fold->signs_for(width));
    return s.fold_buf.data();
}

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

// gelu(gate) * up or silu(gate) * up, per the architecture.
inline void gated_act(FfnAct act, const float* gate, const float* up,
                      float* y, std::size_t n) {
    if (act == FfnAct::GELU) kernels::geglu_f32(gate, up, y, n);
    else                     kernels::swiglu_f32(gate, up, y, n);
}

Result<void>
ffn_dense(BlockScratch& s, const ModelTensors::Block& b,
          const ModelDims& d, const ArchParams& p, runtime::ThreadPool& pool) {
    if (auto r = matvec_dispatch_threaded(pool, b.ffn_gate.dtype, b.ffn_gate.data,
                                          folded_input(s, b.ffn_gate, s.x_norm.data(), d.hidden),
                                          s.gate.data(),
                                          d.ffn_hidden, d.hidden); !r) return r;
    if (auto r = matvec_dispatch_threaded(pool, b.ffn_up.dtype, b.ffn_up.data,
                                          folded_input(s, b.ffn_up, s.x_norm.data(), d.hidden),
                                          s.up.data(),
                                          d.ffn_hidden, d.hidden); !r) return r;
    gated_act(p.ffn_act, s.gate.data(), s.up.data(), s.gate.data(), d.ffn_hidden);
    return matvec_dispatch_threaded(pool, b.ffn_down.dtype, b.ffn_down.data,
                                    folded_input(s, b.ffn_down, s.gate.data(), d.ffn_hidden),
                                    s.x.data(),
                                    d.hidden, d.ffn_hidden);
}

// Phi-3 packs gate and up into one matrix stacked on the output axis, with
// gate first. One matvec produces both halves.
Result<void>
ffn_fused(BlockScratch& s, const ModelTensors::Block& b,
          const ModelDims& d, const ArchParams& p, runtime::ThreadPool& pool) {
    if (auto r = matvec_dispatch_threaded(pool, b.ffn_up.dtype, b.ffn_up.data,
                                          s.x_norm.data(), s.gate.data(),
                                          2 * d.ffn_hidden, d.hidden); !r) return r;
    float* gate_half = s.gate.data();
    float* up_half   = s.gate.data() + d.ffn_hidden;
    gated_act(p.ffn_act, gate_half, up_half, gate_half, d.ffn_hidden);
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
        gated_act(p.ffn_act, s.gate.data(), s.up.data(), s.gate.data(), p.expert_ffn);

        if (auto r = matvec_dispatch_threaded(pool, b.ffn_down_exps.dtype,
                                              b.ffn_down_exps.expert(e),
                                              s.gate.data(), s.expert_out.data(),
                                              d.hidden, p.expert_ffn); !r) return r;

        for (std::size_t j = 0; j < d.hidden; ++j) s.moe_accum[j] += w * s.expert_out[j];
    }

    std::copy(s.moe_accum.begin(), s.moe_accum.end(), s.x.begin());
    return {};
}

// The gated delta-net mixer, for one token. Replaces attention entirely on
// three of every four qwen35 blocks: there is no KV here, only the slot's
// recurrent state, which this both reads and advances.
//
// Reads s.x_norm (the attention pre-norm output) and writes s.x, so the
// caller's residual add is unchanged.
Result<void>
delta_net_forward(BlockScratch& s, const ModelTensors::Block& b,
                  const ModelDims& d, const ArchParams& p,
                  kv_cache::RecurrentState& rs, kv_cache::SlotHandle slot,
                  std::size_t layer, runtime::ThreadPool& pool) {
    const auto& dn = p.delta;
    const std::size_t key_dim   = dn.key_dim();
    const std::size_t v_dim     = dn.head_v_dim();
    const std::size_t n_v       = dn.n_v_heads;
    const std::size_t n_k       = dn.n_k_heads;
    const std::size_t d_state   = dn.d_state;

    float* conv_state = rs.conv(slot, layer);
    if (!conv_state) {
        return fail(ErrorCode::InvalidModel,
                    fmt::format("layer {} is recurrent but has no state buffer", layer),
                    "delta_net");
    }

    // 1. The fused q||k||v projection, and the output gate branch.
    if (auto r = matvec_dispatch_threaded(pool, b.ssm_in.dtype, b.ssm_in.data,
                                          folded_input(s, b.ssm_in, s.x_norm.data(), d.hidden),
                                          s.dn_mixed.data(),
                                          dn.conv_dim(), d.hidden); !r) return r;
    if (auto r = matvec_dispatch_threaded(pool, b.ssm_gate.dtype, b.ssm_gate.data,
                                          folded_input(s, b.ssm_gate, s.x_norm.data(), d.hidden),
                                          s.dn_z.data(),
                                          dn.value_dim(), d.hidden); !r) return r;

    trace("linear_attn_qkv_mixed", layer, s.dn_mixed.data(), dn.conv_dim());

    // 2. beta (the write strength) and the per-head decay.
    if (auto r = matvec_dispatch_threaded(pool, b.ssm_beta.dtype, b.ssm_beta.data,
                                          s.x_norm.data(), s.dn_beta.data(),
                                          n_v, d.hidden); !r) return r;
    if (auto r = matvec_dispatch_threaded(pool, b.ssm_alpha.dtype, b.ssm_alpha.data,
                                          s.x_norm.data(), s.dn_alpha.data(),
                                          n_v, d.hidden); !r) return r;

    const float* dt_bias = s.ssm_dt_bias.data() + layer * n_v;
    const float* a_log   = s.ssm_a.data()       + layer * n_v;
    for (std::size_t h = 0; h < n_v; ++h) {
        s.dn_beta[h] = 1.0f / (1.0f + std::exp(-s.dn_beta[h]));
        s.dn_alpha[h] += dt_bias[h];
    }
    trace("alpha", layer, s.dn_alpha.data(), n_v);
    kernels::softplus_f32(s.dn_alpha.data(), s.dn_alpha.data(), n_v);
    trace("a_softplus", layer, s.dn_alpha.data(), n_v);
    trace("beta_sigmoid", layer, s.dn_beta.data(), n_v);
    for (std::size_t h = 0; h < n_v; ++h) {
        // ssm_a is stored already negated (-exp(A_log)), so the product is
        // negative and the decay lands in (0, 1).
        s.dn_decay[h] = std::exp(s.dn_alpha[h] * a_log[h]);
    }

    // 3. Causal depthwise conv over the q||k||v mix, then SiLU. This is the
    //    only place the conv history is touched, and the kernel rolls it.
    kernels::causal_conv1d_step_f32(conv_state, s.dn_mixed.data(),
                                    s.ssm_conv_w.data() + layer * dn.conv_dim() * dn.d_conv,
                                    s.dn_conv_out.data(), dn.conv_dim(), dn.d_conv);
    trace("conv_output_raw", layer, s.dn_conv_out.data(), dn.conv_dim());
    kernels::silu_f32(s.dn_conv_out.data(), s.dn_conv_out.data(), dn.conv_dim());
    trace("conv_output_silu", layer, s.dn_conv_out.data(), dn.conv_dim());

    const float* q_all = s.dn_conv_out.data();
    const float* k_all = s.dn_conv_out.data() + key_dim;
    const float* v_all = s.dn_conv_out.data() + 2 * key_dim;

    trace("q_conv", layer, q_all, key_dim);
    trace("k_conv", layer, k_all, key_dim);
    trace("v_conv_predelta", layer, v_all, dn.value_dim());

    // 4. The recurrence, one value head at a time.
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(d_state));
    for (std::size_t h = 0; h < n_v; ++h) {
        // Value head h reads key head h % n_k, NOT h / (n_v/n_k). The
        // reference tiles q and k with ggml_repeat and the fused kernel
        // indexes `iv1 % nek1`, both of which interleave. This is the
        // opposite of the blocked mapping ordinary GQA attention uses, and
        // getting it backwards still produces fluent output.
        const std::size_t hk = h % n_k;

        kernels::l2_norm_f32(q_all + hk * d_state, s.dn_qh.data(), d_state, d.rms_eps);
        kernels::l2_norm_f32(k_all + hk * d_state, s.dn_kh.data(), d_state, d.rms_eps);
        for (std::size_t i = 0; i < d_state; ++i) s.dn_qh[i] *= q_scale;

        kernels::delta_net_step_f32(rs.delta_head(slot, layer, h),
                                    s.dn_qh.data(), s.dn_kh.data(),
                                    v_all + h * v_dim,
                                    s.dn_decay[h], s.dn_beta[h],
                                    s.dn_out.data() + h * v_dim,
                                    s.dn_scratch.data(), d_state);
    }

    trace("core_attn_out", layer, s.dn_out.data(), dn.value_dim());

    // 5. Per-head RMSNorm of the readout, gated by silu(z).
    const float* nw = s.ssm_norm.data() + layer * v_dim;
    for (std::size_t h = 0; h < n_v; ++h) {
        float* o = s.dn_out.data() + h * v_dim;
        kernels::rmsnorm_f32(o, nw, o, v_dim, d.rms_eps);
    }
    kernels::silu_f32(s.dn_z.data(), s.dn_z.data(), dn.value_dim());
    kernels::mul_f32(s.dn_out.data(), s.dn_z.data(), s.dn_out.data(), dn.value_dim());

    trace("final_output", layer, s.dn_out.data(), dn.value_dim());

    // 6. Output projection back into the residual width.
    //
    // The fold for ssm_out was computed with the feature axis in grouped
    // head order [head_dim, rep, n_k_heads], but the recurrence emits tiled
    // order [head_dim, n_k_heads, rep]. Reorder before rotating, or the
    // signs line up with the wrong channels.
    const float* out_act = s.dn_out.data();
    if (b.ssm_out.hadamard_folded && s.fold && s.fold->gdn_v_grouped) {
        hadamard_permute_gdn_v(s.dn_out.data(), s.fold_perm.data(),
                               v_dim, n_k, n_v / n_k);
        out_act = s.fold_perm.data();
    }
    return matvec_dispatch_threaded(pool, b.ssm_out.dtype, b.ssm_out.data,
                                    folded_input(s, b.ssm_out, out_act, dn.value_dim()),
                                    s.x.data(),
                                    d.hidden, dn.value_dim());
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
    if (p.qkv == QkvKind::SeparateQGate) {
        qg_full.assign(2 * d.q_dim, 0.0f);
        elem_gate.assign(d.q_dim, 0.0f);
    }
    if (p.attn_gate) {
        head_gate.assign(d.n_heads, 0.0f);
    }

    if (p.delta.active()) {
        const auto& dn = p.delta;
        dn_mixed.assign(dn.conv_dim(), 0.0f);
        dn_conv_out.assign(dn.conv_dim(), 0.0f);
        dn_z.assign(dn.value_dim(), 0.0f);
        dn_beta.assign(dn.n_v_heads, 0.0f);
        dn_alpha.assign(dn.n_v_heads, 0.0f);
        dn_decay.assign(dn.n_v_heads, 0.0f);
        dn_out.assign(dn.value_dim(), 0.0f);
        dn_qh.assign(dn.d_state, 0.0f);
        dn_kh.assign(dn.d_state, 0.0f);
        dn_scratch.assign(dn.d_state, 0.0f);

        ssm_conv_w.assign(d.n_layers * dn.conv_dim() * dn.d_conv, 0.0f);
        ssm_dt_bias.assign(d.n_layers * dn.n_v_heads, 0.0f);
        ssm_a.assign(d.n_layers * dn.n_v_heads, 0.0f);
        ssm_norm.assign(d.n_layers * dn.head_v_dim(), 0.0f);
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

    // One buffer wide enough for any folded activation in this model.
    {
        std::size_t w = std::max({d.hidden, d.q_dim, d.ffn_hidden});
        if (p.delta.active()) w = std::max(w, p.delta.value_dim());
        if (p.ffn == FfnKind::SparseMoE) w = std::max(w, p.expert_ffn);
        fold_buf.assign(w, 0.0f);
        if (p.delta.active()) fold_perm.assign(p.delta.value_dim(), 0.0f);
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
                             const ModelDims& d,
                             const ArchParams& p) {
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

        if (b.is_recurrent && p.delta.active()) {
            const auto& dn = p.delta;
            const std::size_t conv_n = dn.conv_dim() * dn.d_conv;
            if (auto r = dequant_1d(b.ssm_conv1d.dtype, b.ssm_conv1d.data,
                                    s.ssm_conv_w.data() + i * conv_n, conv_n); !r) return r;
            if (auto r = dequant_1d(b.ssm_dt_bias.dtype, b.ssm_dt_bias.data,
                                    s.ssm_dt_bias.data() + i * dn.n_v_heads, dn.n_v_heads); !r) return r;
            if (auto r = dequant_1d(b.ssm_a.dtype, b.ssm_a.data,
                                    s.ssm_a.data() + i * dn.n_v_heads, dn.n_v_heads); !r) return r;
            if (auto r = dequant_1d(b.ssm_norm.dtype, b.ssm_norm.data,
                                    s.ssm_norm.data() + i * dn.head_v_dim(),
                                    dn.head_v_dim()); !r) return r;
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
              runtime::ThreadPool& pool,
              kv_cache::RecurrentState* rs)
{
    // Save residual stream.
    std::copy(s.x.begin(), s.x.end(), s.residual.begin());

    // 1. Attention pre-norm.
    kernels::rmsnorm_f32(s.x.data(),
                         s.attn_norm.data() + layer * dims.hidden,
                         s.x_norm.data(), dims.hidden, dims.rms_eps);
    trace("attn_norm", layer, s.x_norm.data(), dims.hidden);

    // 2. Either the recurrent mixer or attention. A delta-net layer has no
    //    KV at all, so it short-circuits straight to the residual add.
    if (block.is_recurrent) {
        if (!rs) {
            return fail(ErrorCode::InvalidModel,
                        fmt::format("layer {} is recurrent but no state was supplied",
                                    layer),
                        "block_forward");
        }
        if (auto r = delta_net_forward(s, block, dims, params, *rs, slot, layer, pool); !r)
            return r;
        scale_in_place(s.x.data(), params.residual_scale, dims.hidden);
        kernels::add_f32(s.x.data(), s.residual.data(), s.x.data(), dims.hidden);

        // FFN, identical to an attention layer's.
        std::copy(s.x.begin(), s.x.end(), s.residual.begin());
        kernels::rmsnorm_f32(s.x.data(),
                             s.ffn_norm.data() + layer * dims.hidden,
                             s.x_norm.data(), dims.hidden, dims.rms_eps);
        switch (params.ffn) {
        case FfnKind::DenseSwiGLU:
            if (auto r = ffn_dense(s, block, dims, params, pool); !r) return r;
            break;
        case FfnKind::FusedSwiGLU:
            if (auto r = ffn_fused(s, block, dims, params, pool); !r) return r;
            break;
        case FfnKind::SparseMoE:
            if (auto r = ffn_moe(s, block, dims, params, pool); !r) return r;
            break;
        }
        scale_in_place(s.x.data(), params.residual_scale, dims.hidden);
        kernels::add_f32(s.x.data(), s.residual.data(), s.x.data(), dims.hidden);
        return {};
    }

    if (params.qkv == QkvKind::SeparateQGate) {
        // attn_q emits query and gate interleaved per head, so a head's
        // 2*head_dim run is [query | gate]. Split them apart here; the gate
        // is applied to the attention output further down.
        if (auto r = matvec_dispatch_threaded(pool, block.attn_q.dtype, block.attn_q.data,
                                              folded_input(s, block.attn_q, s.x_norm.data(), dims.hidden),
                                              s.qg_full.data(),
                                              2 * dims.q_dim, dims.hidden); !r) return r;
        for (std::size_t h = 0; h < dims.n_heads; ++h) {
            const float* src = s.qg_full.data() + h * 2 * dims.head_dim;
            std::memcpy(s.q_full.data()    + h * dims.head_dim, src,
                        dims.head_dim * sizeof(float));
            std::memcpy(s.elem_gate.data() + h * dims.head_dim, src + dims.head_dim,
                        dims.head_dim * sizeof(float));
        }
        if (auto r = matvec_dispatch_threaded(pool, block.attn_k.dtype, block.attn_k.data,
                                              folded_input(s, block.attn_k, s.x_norm.data(), dims.hidden),
                                              s.k_full.data(),
                                              dims.kv_dim, dims.hidden); !r) return r;
        if (auto r = matvec_dispatch_threaded(pool, block.attn_v.dtype, block.attn_v.data,
                                              folded_input(s, block.attn_v, s.x_norm.data(), dims.hidden),
                                              s.v_full.data(),
                                              dims.kv_dim, dims.hidden); !r) return r;
        trace("Qcur_reshaped", layer, s.q_full.data(), dims.q_dim);
        trace("gate_reshaped", layer, s.elem_gate.data(), dims.q_dim);
        trace("Kcur", layer, s.k_full.data(), dims.kv_dim);
        trace("Vcur", layer, s.v_full.data(), dims.kv_dim);
    } else if (params.qkv == QkvKind::Fused) {
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

    trace("Qcur_normed", layer, s.q_full.data(), dims.q_dim);
    trace("Kcur_normed", layer, s.k_full.data(), dims.kv_dim);

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

    trace("attn_pregate", layer, s.attn_out.data(), dims.q_dim);

    // 6c. qwen35's gate is per element, not per head: the sigmoid of the
    //     second half of the Q projection scales the attention output
    //     position by position.
    if (params.qkv == QkvKind::SeparateQGate) {
        for (std::size_t i = 0; i < dims.q_dim; ++i) {
            s.attn_out[i] *= 1.0f / (1.0f + std::exp(-s.elem_gate[i]));
        }
    }

    trace("attn_gated", layer, s.attn_out.data(), dims.q_dim);

    // 7. Output projection + residual add.
    if (auto r = matvec_dispatch_threaded(pool, block.attn_out.dtype, block.attn_out.data,
                                          folded_input(s, block.attn_out, s.attn_out.data(), dims.q_dim),
                                          s.x.data(),
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
        if (auto r = ffn_dense(s, block, dims, params, pool); !r) return r;
        break;
    case FfnKind::FusedSwiGLU:
        if (auto r = ffn_fused(s, block, dims, params, pool); !r) return r;
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
