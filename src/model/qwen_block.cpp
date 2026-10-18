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
                          const float* X, std::size_t width,
                          std::size_t n_tok) {
    if (!w.hadamard_folded || !s.fold) return X;
    const float* signs = s.fold->signs_for(width);
    for (std::size_t t = 0; t < n_tok; ++t) {
        kernels::hadamard_fold_f32(X + t * width, s.fold_buf.data() + t * width,
                                   width, s.fold->block, signs);
    }
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
          const ModelDims& d, const ArchParams& p, std::size_t n_tok,
          runtime::ThreadPool& pool) {
    if (auto r = matmul_dispatch_threaded(pool, b.ffn_gate.dtype, b.ffn_gate.data,
                                          folded_input(s, b.ffn_gate, s.x_norm.data(), d.hidden, n_tok),
                                          s.gate.data(),
                                          d.ffn_hidden, d.hidden, n_tok); !r) return r;
    if (auto r = matmul_dispatch_threaded(pool, b.ffn_up.dtype, b.ffn_up.data,
                                          folded_input(s, b.ffn_up, s.x_norm.data(), d.hidden, n_tok),
                                          s.up.data(),
                                          d.ffn_hidden, d.hidden, n_tok); !r) return r;
    gated_act(p.ffn_act, s.gate.data(), s.up.data(), s.gate.data(), n_tok * d.ffn_hidden);
    return matmul_dispatch_threaded(pool, b.ffn_down.dtype, b.ffn_down.data,
                                    folded_input(s, b.ffn_down, s.gate.data(), d.ffn_hidden, n_tok),
                                    s.x.data(),
                                    d.hidden, d.ffn_hidden, n_tok);
}

// Phi-3 packs gate and up into one matrix stacked on the output axis, with
// gate first. One matvec produces both halves.
Result<void>
ffn_fused(BlockScratch& s, const ModelTensors::Block& b,
          const ModelDims& d, const ArchParams& p, std::size_t n_tok,
          runtime::ThreadPool& pool) {
    if (auto r = matmul_dispatch_threaded(pool, b.ffn_up.dtype, b.ffn_up.data,
                                          s.x_norm.data(), s.gate.data(),
                                          2 * d.ffn_hidden, d.hidden, n_tok); !r) return r;
    // gate||up are stacked per token, so the halves have to be recombined
    // token by token and compacted to the narrower ffn width for ffn_down.
    for (std::size_t t = 0; t < n_tok; ++t) {
        const float* gate_half = s.gate.data() + t * 2 * d.ffn_hidden;
        const float* up_half   = gate_half + d.ffn_hidden;
        gated_act(p.ffn_act, gate_half, up_half,
                  s.up.data() + t * d.ffn_hidden, d.ffn_hidden);
    }
    return matmul_dispatch_threaded(pool, b.ffn_down.dtype, b.ffn_down.data,
                                    s.up.data(), s.x.data(),
                                    d.hidden, d.ffn_hidden, n_tok);
}

// Sparse MoE: softmax the router over all experts, take the top-k, renormalize
// those k to sum to 1, then accumulate each expert's SwiGLU output weighted by
// its share. Matches llama.cpp's build_moe_ffn with norm_w=true, scale_w=false.
// Not batched across tokens, deliberately: each token routes to its own
// top-k experts, so there is no shared weight matrix to amortize. Batching
// would mean grouping tokens by expert, which is worth doing only once a
// MoE model is actually the bottleneck.
Result<void>
ffn_moe_one(BlockScratch& s, const ModelTensors::Block& b,
            const ModelDims& d, const ArchParams& p, std::size_t t,
            std::size_t layer, runtime::ThreadPool& pool) {
    const float* x_norm_t = s.x_norm.data() + t * d.hidden;
    float*       x_t      = s.x.data()      + t * d.hidden;

    // Router over the full expert set.
    if (auto r = matvec_dispatch_threaded(pool, b.ffn_gate_inp.dtype, b.ffn_gate_inp.data,
                                          x_norm_t, s.router_probs.data(),
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
                                              x_norm_t, s.gate.data(),
                                              p.expert_ffn, d.hidden); !r) return r;
        if (auto r = matvec_dispatch_threaded(pool, b.ffn_up_exps.dtype,
                                              b.ffn_up_exps.expert(e),
                                              x_norm_t, s.up.data(),
                                              p.expert_ffn, d.hidden); !r) return r;
        gated_act(p.ffn_act, s.gate.data(), s.up.data(), s.gate.data(), p.expert_ffn);

        if (auto r = matvec_dispatch_threaded(pool, b.ffn_down_exps.dtype,
                                              b.ffn_down_exps.expert(e),
                                              s.gate.data(), s.expert_out.data(),
                                              d.hidden, p.expert_ffn); !r) return r;

        for (std::size_t j = 0; j < d.hidden; ++j) s.moe_accum[j] += w * s.expert_out[j];
    }

    trace("ffn_moe_out", layer, s.moe_accum.data(), d.hidden);

    // The shared expert sees every token. Its own router is a single row, so
    // it produces one logit; sigmoid of that scales the whole contribution
    // before it joins the routed sum. Unlike the routed weights this gate is
    // not part of the renormalized top-k mass — it is an independent scale,
    // which is why the shared expert can push the block's output past what
    // the routed experts alone would give.
    if (p.expert_shared_ffn > 0) {
        const std::size_t sf = p.expert_shared_ffn;

        if (auto r = matvec_dispatch_threaded(pool, b.ffn_gate_shexp.dtype,
                                              b.ffn_gate_shexp.data,
                                              x_norm_t, s.gate.data(),
                                              sf, d.hidden); !r) return r;
        if (auto r = matvec_dispatch_threaded(pool, b.ffn_up_shexp.dtype,
                                              b.ffn_up_shexp.data,
                                              x_norm_t, s.up.data(),
                                              sf, d.hidden); !r) return r;
        gated_act(p.ffn_act, s.gate.data(), s.up.data(), s.gate.data(), sf);

        if (auto r = matvec_dispatch_threaded(pool, b.ffn_down_shexp.dtype,
                                              b.ffn_down_shexp.data,
                                              s.gate.data(), s.expert_out.data(),
                                              d.hidden, sf); !r) return r;
        trace("ffn_shexp", layer, s.expert_out.data(), d.hidden);

        float gate_logit = 0.0f;
        if (auto r = matvec_dispatch(b.ffn_gate_inp_shexp.dtype,
                                     b.ffn_gate_inp_shexp.data,
                                     x_norm_t, &gate_logit,
                                     1, d.hidden); !r) return r;
        const float g = 1.0f / (1.0f + std::exp(-gate_logit));
        trace("shared_expert_gate_sigmoid", layer, &g, 1);

        for (std::size_t j = 0; j < d.hidden; ++j) s.moe_accum[j] += g * s.expert_out[j];
    }

    trace("ffn_out", layer, s.moe_accum.data(), d.hidden);
    std::copy_n(s.moe_accum.begin(), d.hidden, x_t);
    return {};
}

// The gated delta-net mixer. Replaces attention entirely on three of every
// four qwen35 blocks: there is no KV here, only the slot's recurrent state,
// which this both reads and advances.
//
// The projections batch across tokens, but the recurrence cannot: token t's
// state depends on t-1, so the conv and the delta rule run strictly in
// order. That split is the whole design here — the matmuls are nearly all
// of the work and they batch; the sequential part touches only a fixed-size
// state and is cheap.
//
// Reads s.x_norm (the attention pre-norm output) and writes s.x, so the
// caller's residual add is unchanged.
Result<void>
delta_net_forward(BlockScratch& s, const ModelTensors::Block& b,
                  const ModelDims& d, const ArchParams& p,
                  kv_cache::RecurrentState& rs, kv_cache::SlotHandle slot,
                  std::size_t layer, std::size_t n_tok,
                  runtime::ThreadPool& pool) {
    const auto& dn = p.delta;
    const std::size_t key_dim = dn.key_dim();
    const std::size_t v_dim   = dn.head_v_dim();
    const std::size_t n_v     = dn.n_v_heads;
    const std::size_t n_k     = dn.n_k_heads;
    const std::size_t d_state = dn.d_state;
    const std::size_t cdim    = dn.conv_dim();
    const std::size_t vdim    = dn.value_dim();

    float* conv_state = rs.conv(slot, layer);
    if (!conv_state) {
        return fail(ErrorCode::InvalidModel,
                    fmt::format("layer {} is recurrent but has no state buffer", layer),
                    "delta_net");
    }

    // ---- batched: every projection off the normed residual ---------------
    if (auto r = matmul_dispatch_threaded(pool, b.ssm_in.dtype, b.ssm_in.data,
                                          folded_input(s, b.ssm_in, s.x_norm.data(), d.hidden, n_tok),
                                          s.dn_mixed.data(), cdim, d.hidden, n_tok); !r) return r;
    if (auto r = matmul_dispatch_threaded(pool, b.ssm_gate.dtype, b.ssm_gate.data,
                                          folded_input(s, b.ssm_gate, s.x_norm.data(), d.hidden, n_tok),
                                          s.dn_z.data(), vdim, d.hidden, n_tok); !r) return r;
    if (auto r = matmul_dispatch_threaded(pool, b.ssm_beta.dtype, b.ssm_beta.data,
                                          s.x_norm.data(), s.dn_beta.data(),
                                          n_v, d.hidden, n_tok); !r) return r;
    if (auto r = matmul_dispatch_threaded(pool, b.ssm_alpha.dtype, b.ssm_alpha.data,
                                          s.x_norm.data(), s.dn_alpha.data(),
                                          n_v, d.hidden, n_tok); !r) return r;

    const float* dt_bias = s.ssm_dt_bias.data() + layer * n_v;
    const float* a_log   = s.ssm_a.data()       + layer * n_v;
    for (std::size_t i = 0; i < n_tok * n_v; ++i) {
        s.dn_beta[i] = 1.0f / (1.0f + std::exp(-s.dn_beta[i]));
        s.dn_alpha[i] += dt_bias[i % n_v];
    }
    kernels::softplus_f32(s.dn_alpha.data(), s.dn_alpha.data(), n_tok * n_v);
    for (std::size_t i = 0; i < n_tok * n_v; ++i) {
        // ssm_a is stored already negated (-exp(A_log)), so the product is
        // negative and the decay lands in (0, 1).
        s.dn_decay[i] = std::exp(s.dn_alpha[i] * a_log[i % n_v]);
    }

    trace("linear_attn_qkv_mixed", layer, s.dn_mixed.data(), cdim);
    trace("alpha", layer, s.dn_alpha.data(), n_v);
    trace("a_softplus", layer, s.dn_alpha.data(), n_v);
    trace("beta_sigmoid", layer, s.dn_beta.data(), n_v);

    // ---- sequential: conv history and the delta rule ---------------------
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(d_state));
    const float* conv_w = s.ssm_conv_w.data() + layer * cdim * dn.d_conv;

    for (std::size_t t = 0; t < n_tok; ++t) {
        float* conv_out = s.dn_conv_out.data() + t * cdim;
        kernels::causal_conv1d_step_f32(conv_state, s.dn_mixed.data() + t * cdim,
                                        conv_w, conv_out, cdim, dn.d_conv);
        kernels::silu_f32(conv_out, conv_out, cdim);
        if (t == 0) {
            trace("conv_output_silu", layer, conv_out, cdim);
        }

        const float* q_all = conv_out;
        const float* k_all = conv_out + key_dim;
        const float* v_all = conv_out + 2 * key_dim;

        for (std::size_t h = 0; h < n_v; ++h) {
            // Value head h reads key head h % n_k, NOT h / (n_v/n_k). The
            // reference tiles q and k with ggml_repeat and the fused kernel
            // indexes `iv1 % nek1`, both of which interleave. This is the
            // opposite of the blocked mapping ordinary GQA attention uses,
            // and getting it backwards still produces fluent output.
            const std::size_t hk = h % n_k;

            kernels::l2_norm_f32(q_all + hk * d_state, s.dn_qh.data(), d_state, d.rms_eps);
            kernels::l2_norm_f32(k_all + hk * d_state, s.dn_kh.data(), d_state, d.rms_eps);
            for (std::size_t i = 0; i < d_state; ++i) s.dn_qh[i] *= q_scale;

            kernels::delta_net_step_f32(rs.delta_head(slot, layer, h),
                                        s.dn_qh.data(), s.dn_kh.data(),
                                        v_all + h * v_dim,
                                        s.dn_decay[t * n_v + h],
                                        s.dn_beta[t * n_v + h],
                                        s.dn_out.data() + t * vdim + h * v_dim,
                                        s.dn_scratch.data(), d_state);
        }
    }
    trace("core_attn_out", layer, s.dn_out.data(), vdim);

    // ---- batched again: gated norm and the output projection -------------
    const float* nw = s.ssm_norm.data() + layer * v_dim;
    for (std::size_t t = 0; t < n_tok; ++t) {
        for (std::size_t h = 0; h < n_v; ++h) {
            float* o = s.dn_out.data() + t * vdim + h * v_dim;
            kernels::rmsnorm_f32(o, nw, o, v_dim, d.rms_eps);
        }
    }
    kernels::silu_f32(s.dn_z.data(), s.dn_z.data(), n_tok * vdim);
    kernels::mul_f32(s.dn_out.data(), s.dn_z.data(), s.dn_out.data(), n_tok * vdim);
    trace("final_output", layer, s.dn_out.data(), vdim);

    // The fold for ssm_out was computed with the feature axis in grouped
    // head order [head_dim, rep, n_k_heads], but the recurrence emits tiled
    // order [head_dim, n_k_heads, rep]. Reorder before rotating, or the
    // signs line up with the wrong channels.
    const float* out_act = s.dn_out.data();
    if (b.ssm_out.hadamard_folded && s.fold && s.fold->gdn_v_grouped) {
        for (std::size_t t = 0; t < n_tok; ++t) {
            hadamard_permute_gdn_v(s.dn_out.data() + t * vdim,
                                   s.fold_perm.data() + t * vdim,
                                   v_dim, n_k, n_v / n_k);
        }
        out_act = s.fold_perm.data();
    }
    return matmul_dispatch_threaded(pool, b.ssm_out.dtype, b.ssm_out.data,
                                    folded_input(s, b.ssm_out, out_act, vdim, n_tok),
                                    s.x.data(), d.hidden, vdim, n_tok);
}

} // namespace

void BlockScratch::init(const ModelDims& d, const ArchParams& p,
                        std::size_t max_batch_tokens) {
    const std::size_t B = std::max<std::size_t>(max_batch_tokens, 1);
    max_batch = B;

    x.assign(B * d.hidden, 0.0f);
    residual.assign(B * d.hidden, 0.0f);
    x_norm.assign(B * d.hidden, 0.0f);
    q_full.assign(B * d.q_dim, 0.0f);
    k_full.assign(B * d.kv_dim, 0.0f);
    v_full.assign(B * d.kv_dim, 0.0f);
    attn_out.assign(B * d.q_dim, 0.0f);
    // Scores are per head per token and consumed immediately, so one
    // token's worth is enough no matter how wide the batch is.
    attn_scores.assign(std::max<std::size_t>(d.n_ctx_train, 1), 0.0f);
    logits_tmp.assign(d.vocab, 0.0f);

    if (p.qkv == QkvKind::Fused) {
        qkv_full.assign(B * (d.q_dim + 2 * d.kv_dim), 0.0f);
    }
    if (p.qkv == QkvKind::SeparateQGate) {
        qg_full.assign(B * 2 * d.q_dim, 0.0f);
        elem_gate.assign(B * d.q_dim, 0.0f);
    }
    if (p.attn_gate) {
        head_gate.assign(B * d.n_heads, 0.0f);
    }

    if (p.delta.active()) {
        const auto& dn = p.delta;
        dn_mixed.assign(B * dn.conv_dim(), 0.0f);
        dn_conv_out.assign(B * dn.conv_dim(), 0.0f);
        dn_z.assign(B * dn.value_dim(), 0.0f);
        dn_beta.assign(B * dn.n_v_heads, 0.0f);
        dn_alpha.assign(B * dn.n_v_heads, 0.0f);
        dn_decay.assign(B * dn.n_v_heads, 0.0f);
        dn_out.assign(B * dn.value_dim(), 0.0f);
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
        gate.assign(B * d.ffn_hidden, 0.0f);
        up.assign(B * d.ffn_hidden, 0.0f);
        break;
    case FfnKind::FusedSwiGLU:
        gate.assign(B * 2 * d.ffn_hidden, 0.0f);
        // The fused path compacts gate*up down to the narrow width before
        // ffn_down, so it needs the second buffer too once batched.
        up.assign(B * d.ffn_hidden, 0.0f);
        break;
    case FfnKind::SparseMoE: {
        // One token at a time, at whichever of the two expert widths is
        // larger — the routed experts and the shared one reuse these in turn.
        const std::size_t ew = std::max(p.expert_ffn, p.expert_shared_ffn);
        gate.assign(ew, 0.0f);
        up.assign(ew, 0.0f);
        router_probs.assign(p.n_expert, 0.0f);
        expert_ids.assign(p.n_expert_used, 0u);
        expert_weights.assign(p.n_expert_used, 0.0f);
        expert_out.assign(d.hidden, 0.0f);
        moe_accum.assign(d.hidden, 0.0f);
        break;
    }
    }

    // One buffer wide enough for any folded activation in this model.
    {
        std::size_t w = std::max({d.hidden, d.q_dim, d.ffn_hidden});
        if (p.delta.active()) w = std::max(w, p.delta.value_dim());
        if (p.ffn == FfnKind::SparseMoE) {
            w = std::max({w, p.expert_ffn, p.expert_shared_ffn});
        }
        fold_buf.assign(B * w, 0.0f);
        if (p.delta.active()) fold_perm.assign(B * p.delta.value_dim(), 0.0f);
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
              std::size_t start_position,
              std::size_t n_tok,
              runtime::ThreadPool& pool,
              kv_cache::RecurrentState* rs)
{
    if (n_tok == 0) return {};
    if (n_tok > s.max_batch) {
        return fail(ErrorCode::InvalidModel,
                    fmt::format("block_forward got {} tokens but the scratch was "
                                "sized for {}", n_tok, s.max_batch),
                    "block_forward");
    }

    const std::size_t H = dims.hidden;

    // Save the residual stream for the attention branch.
    std::copy_n(s.x.begin(), n_tok * H, s.residual.begin());

    // 1. Attention pre-norm, per token.
    for (std::size_t t = 0; t < n_tok; ++t) {
        kernels::rmsnorm_f32(s.x.data() + t * H,
                             s.attn_norm.data() + layer * H,
                             s.x_norm.data() + t * H, H, dims.rms_eps);
    }
    trace("attn_norm", layer, s.x_norm.data(), H);

    // 2. Either the recurrent mixer or attention. A delta-net layer has no
    //    KV at all, so it short-circuits straight to the residual add.
    if (block.is_recurrent) {
        if (!rs) {
            return fail(ErrorCode::InvalidModel,
                        fmt::format("layer {} is recurrent but no state was supplied",
                                    layer),
                        "block_forward");
        }
        if (auto r = delta_net_forward(s, block, dims, params, *rs, slot, layer,
                                       n_tok, pool); !r) return r;
    } else {
        // ---- Q/K/V projections, batched ---------------------------------
        if (params.qkv == QkvKind::SeparateQGate) {
            // attn_q emits query and gate interleaved per head, so a head's
            // 2*head_dim run is [query | gate]. Split them apart here; the
            // gate is applied to the attention output further down.
            if (auto r = matmul_dispatch_threaded(pool, block.attn_q.dtype, block.attn_q.data,
                                                  folded_input(s, block.attn_q, s.x_norm.data(), H, n_tok),
                                                  s.qg_full.data(),
                                                  2 * dims.q_dim, H, n_tok); !r) return r;
            for (std::size_t t = 0; t < n_tok; ++t) {
                for (std::size_t h = 0; h < dims.n_heads; ++h) {
                    const float* src = s.qg_full.data() + t * 2 * dims.q_dim
                                     + h * 2 * dims.head_dim;
                    std::memcpy(s.q_full.data() + t * dims.q_dim + h * dims.head_dim,
                                src, dims.head_dim * sizeof(float));
                    std::memcpy(s.elem_gate.data() + t * dims.q_dim + h * dims.head_dim,
                                src + dims.head_dim, dims.head_dim * sizeof(float));
                }
            }
            if (auto r = matmul_dispatch_threaded(pool, block.attn_k.dtype, block.attn_k.data,
                                                  folded_input(s, block.attn_k, s.x_norm.data(), H, n_tok),
                                                  s.k_full.data(), dims.kv_dim, H, n_tok); !r) return r;
            if (auto r = matmul_dispatch_threaded(pool, block.attn_v.dtype, block.attn_v.data,
                                                  folded_input(s, block.attn_v, s.x_norm.data(), H, n_tok),
                                                  s.v_full.data(), dims.kv_dim, H, n_tok); !r) return r;
        } else if (params.qkv == QkvKind::Fused) {
            const std::size_t qkv_rows = dims.q_dim + 2 * dims.kv_dim;
            if (auto r = matmul_dispatch_threaded(pool, block.attn_qkv.dtype, block.attn_qkv.data,
                                                  folded_input(s, block.attn_qkv, s.x_norm.data(), H, n_tok),
                                                  s.qkv_full.data(), qkv_rows, H, n_tok); !r) return r;
            for (std::size_t t = 0; t < n_tok; ++t) {
                const float* src = s.qkv_full.data() + t * qkv_rows;
                std::memcpy(s.q_full.data() + t * dims.q_dim,  src, dims.q_dim * sizeof(float));
                std::memcpy(s.k_full.data() + t * dims.kv_dim, src + dims.q_dim,
                            dims.kv_dim * sizeof(float));
                std::memcpy(s.v_full.data() + t * dims.kv_dim, src + dims.q_dim + dims.kv_dim,
                            dims.kv_dim * sizeof(float));
            }
        } else {
            if (auto r = matmul_dispatch_threaded(pool, block.attn_q.dtype, block.attn_q.data,
                                                  folded_input(s, block.attn_q, s.x_norm.data(), H, n_tok),
                                                  s.q_full.data(), dims.q_dim, H, n_tok); !r) return r;
            if (auto r = matmul_dispatch_threaded(pool, block.attn_k.dtype, block.attn_k.data,
                                                  folded_input(s, block.attn_k, s.x_norm.data(), H, n_tok),
                                                  s.k_full.data(), dims.kv_dim, H, n_tok); !r) return r;
            if (auto r = matmul_dispatch_threaded(pool, block.attn_v.dtype, block.attn_v.data,
                                                  folded_input(s, block.attn_v, s.x_norm.data(), H, n_tok),
                                                  s.v_full.data(), dims.kv_dim, H, n_tok); !r) return r;
        }

        // 3. Biases, Q/K norm, RoPE — all per token.
        const bool  is_swa = params.is_swa_layer(layer);
        const bool  neox   = (params.rope_style == RopeStyle::NeoX);
        const float* ff    = rope.freq_factors.empty() ? nullptr : rope.freq_factors.data();
        const std::size_t rope_dim  = is_swa ? rope.rope_dim_swa  : rope.rope_dim;
        const float       freq_base = is_swa ? rope.freq_base_swa : rope.freq_base;

        for (std::size_t t = 0; t < n_tok; ++t) {
            float* q_t = s.q_full.data() + t * dims.q_dim;
            float* k_t = s.k_full.data() + t * dims.kv_dim;
            float* v_t = s.v_full.data() + t * dims.kv_dim;

            if (s.has_biases[layer]) {
                add_bias(q_t, block.attn_q_bias ? s.attn_q_bias.data() + layer * dims.q_dim : nullptr,
                         dims.q_dim);
                add_bias(k_t, block.attn_k_bias ? s.attn_k_bias.data() + layer * dims.kv_dim : nullptr,
                         dims.kv_dim);
                add_bias(v_t, block.attn_v_bias ? s.attn_v_bias.data() + layer * dims.kv_dim : nullptr,
                         dims.kv_dim);
            }
            if (s.has_qk_norm[layer]) {
                rmsnorm_per_head(q_t, dims.n_heads,    dims.head_dim,
                                 s.attn_q_norm.data() + layer * dims.head_dim, dims.rms_eps);
                rmsnorm_per_head(k_t, dims.n_kv_heads, dims.head_dim,
                                 s.attn_k_norm.data() + layer * dims.head_dim, dims.rms_eps);
            }
            const std::size_t pos = start_position + t;
            kernels::rope_f32_ext(q_t, dims.n_heads,    dims.head_dim, rope_dim,
                                  pos, freq_base, neox, ff, rope.attn_factor);
            kernels::rope_f32_ext(k_t, dims.n_kv_heads, dims.head_dim, rope_dim,
                                  pos, freq_base, neox, ff, rope.attn_factor);

            // Write this token's K/V before attending, so a token can see
            // itself; earlier tokens of the batch are already in place.
            kv.write_at(slot, layer, pos, k_t, v_t);
        }

        // 4. Attention, per token. Causality inside the batch is automatic:
        //    token t reads only up to its own position.
        const float attn_scale = (params.attn_scale != 0.0f)
            ? params.attn_scale
            : 1.0f / std::sqrt(static_cast<float>(dims.head_dim));

        for (std::size_t t = 0; t < n_tok; ++t) {
            const std::size_t pos = start_position + t;
            auto lv = kv.read_at(slot, layer, pos + 1);
            const float* K = static_cast<const float*>(lv.k);
            const float* V = static_cast<const float*>(lv.v);

            std::size_t pos_begin = 0;
            if (is_swa && params.sliding_window > 0 && lv.pos > params.sliding_window) {
                pos_begin = lv.pos - params.sliding_window;
            }
            gqa_attention(s.q_full.data() + t * dims.q_dim, K, V,
                          lv.stride_head, pos_begin, lv.pos,
                          dims.n_heads, dims.n_kv_heads, dims.head_dim,
                          attn_scale,
                          s.attn_out.data() + t * dims.q_dim, s.attn_scores.data());
        }

        // 5. Output gates.
        if (params.attn_gate) {
            if (auto r = matmul_dispatch_threaded(pool, block.attn_gate.dtype,
                                                  block.attn_gate.data,
                                                  s.x_norm.data(), s.head_gate.data(),
                                                  dims.n_heads, H, n_tok); !r) return r;
            for (std::size_t t = 0; t < n_tok; ++t) {
                for (std::size_t h = 0; h < dims.n_heads; ++h) {
                    const float g = 1.0f / (1.0f + std::exp(-s.head_gate[t * dims.n_heads + h]));
                    float* out_h = s.attn_out.data() + t * dims.q_dim + h * dims.head_dim;
                    for (std::size_t i = 0; i < dims.head_dim; ++i) out_h[i] *= g;
                }
            }
        }
        if (params.qkv == QkvKind::SeparateQGate) {
            // qwen35's gate is per element, not per head.
            for (std::size_t i = 0; i < n_tok * dims.q_dim; ++i) {
                s.attn_out[i] *= 1.0f / (1.0f + std::exp(-s.elem_gate[i]));
            }
        }

        // 6. Output projection.
        if (auto r = matmul_dispatch_threaded(pool, block.attn_out.dtype, block.attn_out.data,
                                              folded_input(s, block.attn_out, s.attn_out.data(),
                                                           dims.q_dim, n_tok),
                                              s.x.data(), H, dims.q_dim, n_tok); !r) return r;
    }

    // 7. Attention (or delta-net) residual add.
    scale_in_place(s.x.data(), params.residual_scale, n_tok * H);
    kernels::add_f32(s.x.data(), s.residual.data(), s.x.data(), n_tok * H);
    trace("attn_residual", layer, s.x.data(), H);

    // 8. FFN pre-norm.
    std::copy_n(s.x.begin(), n_tok * H, s.residual.begin());
    for (std::size_t t = 0; t < n_tok; ++t) {
        kernels::rmsnorm_f32(s.x.data() + t * H,
                             s.ffn_norm.data() + layer * H,
                             s.x_norm.data() + t * H, H, dims.rms_eps);
    }
    trace("attn_post_norm", layer, s.x_norm.data(), H);

    // 9. Feed-forward, writing its output into s.x.
    switch (params.ffn) {
    case FfnKind::DenseSwiGLU:
        if (auto r = ffn_dense(s, block, dims, params, n_tok, pool); !r) return r;
        break;
    case FfnKind::FusedSwiGLU:
        if (auto r = ffn_fused(s, block, dims, params, n_tok, pool); !r) return r;
        break;
    case FfnKind::SparseMoE:
        for (std::size_t t = 0; t < n_tok; ++t) {
            if (auto r = ffn_moe_one(s, block, dims, params, t, layer, pool); !r) return r;
        }
        break;
    }

    // 10. Residual add.
    scale_in_place(s.x.data(), params.residual_scale, n_tok * H);
    kernels::add_f32(s.x.data(), s.residual.data(), s.x.data(), n_tok * H);
    return {};
}

} // namespace ultima::model
