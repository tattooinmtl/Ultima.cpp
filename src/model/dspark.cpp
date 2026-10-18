#include "ultima/model/dspark.hpp"

#include "ultima/kernels/elementwise.hpp"
#include "ultima/kernels/norms.hpp"
#include "ultima/kernels/rope.hpp"
#include "ultima/kernels/softmax.hpp"
#include "ultima/model/weight_dispatch.hpp"
#include "ultima/runtime/thread_pool.hpp"

#include <fmt/format.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <variant>

namespace ultima::model {

namespace {

using core::ErrorCode;
using core::Failure;
using core::Result;
using core::fail;

// LogSnrEmbed in the reference uses 128 sinusoidal features.
constexpr std::size_t k_snr_freq = 128;

// Same idea as qwen_block's trace: under ULTIMA_TRACE, report a checksum
// and a finite-count per stage so a NaN can be located by stage instead of
// by reasoning.
void dtrace(const char* what, int il, const float* x, std::size_t n) {
    static const bool on = std::getenv("ULTIMA_TRACE") != nullptr;
    if (!on) return;
    double sum = 0.0;
    std::size_t bad = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(x[i])) ++bad; else sum += x[i];
    }
    std::fprintf(stderr,
                 "dspark_trace: %-18s il=%-3d n=%-8zu sum=%14.4f nonfinite=%zu\n",
                 what, il, n, sum, bad);
}

std::optional<std::uint64_t> get_uint(const MetadataStore& md, const std::string& k) {
    return md.get_uint(k);
}

bool get_flag(const MetadataStore& md, const std::string& k) {
    if (auto v = md.get<bool>(k)) return *v;
    if (auto v = md.get_uint(k))  return *v != 0;
    return false;
}

float get_float(const MetadataStore& md, const std::string& k, float dflt) {
    if (auto v = md.get<float>(k))  return *v;
    if (auto v = md.get<double>(k)) return static_cast<float>(*v);
    if (auto v = md.get_uint(k))    return static_cast<float>(*v);
    if (auto v = md.get<std::int32_t>(k)) return static_cast<float>(*v);
    return dflt;
}

struct Block {
    WeightRef attn_norm, attn_q, attn_k, attn_v, attn_out;
    WeightRef attn_q_norm, attn_k_norm;
    WeightRef ffn_norm, ffn_gate, ffn_up, ffn_down;
};

} // namespace

struct DsparkDrafter::Impl {
    DsparkConfig cfg{};
    runtime::ThreadPool* pool = nullptr;
    const LoadedModel*   gguf = nullptr;

    WeightRef tok_embd, output_norm, output_head;
    WeightRef fc, hidden_norm;
    WeightRef snr_fc1_w, snr_fc1_b, snr_fc2_w, snr_fc2_b;
    WeightRef conf_w, conf_b;
    WeightRef markov_a, markov_b;
    std::vector<Block> blocks;

    // Dequantized small vectors.
    std::vector<float> f_output_norm, f_hidden_norm;
    std::vector<float> f_attn_norm, f_ffn_norm, f_q_norm, f_k_norm;
    std::vector<float> f_snr_fc1_b, f_snr_fc2_b;
    float conf_bias = 0.0f;

    // Scratch, sized at load for (max context rows + block_size).
    std::vector<float> ctx_proj;    // [n_ctx][n_embd]
    std::vector<float> h;           // [n_draft][n_embd]
    std::vector<float> resid;
    std::vector<float> cur;         // [n_draft][n_embd]
    std::vector<float> attn_in;     // [(n_ctx+n_draft)][n_embd]
    std::vector<float> kbuf, vbuf;  // [(n_ctx+n_draft)][kv_dim]
    std::vector<float> qbuf;        // [n_draft][q_dim]
    std::vector<float> attn_out;    // [n_draft][q_dim]
    std::vector<float> scores;
    std::vector<float> gate, up;
    std::vector<float> snr_feat, snr_hidden, snr_embed;
    std::size_t cap_rows = 0;
};

DsparkDrafter::DsparkDrafter() : impl_{std::make_unique<Impl>()} {}
DsparkDrafter::~DsparkDrafter() = default;

const DsparkConfig& DsparkDrafter::config() const noexcept { return impl_->cfg; }

Result<std::unique_ptr<DsparkDrafter>>
DsparkDrafter::load(const LoadedModel& gguf, const ModelDims& host,
                    runtime::ThreadPool& pool) {
    const auto& md = gguf.metadata();
    const auto arch_v = md.get<std::string>("general.architecture");
    if (!arch_v || *arch_v != "dspark") {
        return fail(ErrorCode::InvalidModel,
                    "not a dspark drafter (general.architecture is not 'dspark')",
                    "dspark");
    }

    std::unique_ptr<DsparkDrafter> self{new DsparkDrafter()};
    auto& I = *self->impl_;
    I.pool = &pool;
    I.gguf = &gguf;
    DsparkConfig& c = I.cfg;

    struct Req { const char* key; std::size_t* out; };
    const Req reqs[] = {
        {"dspark.embedding_length",        &c.n_embd},
        {"dspark.block_count",             &c.n_layer},
        {"dspark.feed_forward_length",     &c.ffn_hidden},
        {"dspark.attention.head_count",    &c.n_head},
        {"dspark.attention.head_count_kv", &c.n_head_kv},
        {"dspark.dspark.block_size",       &c.block_size},
        {"dspark.dspark.mask_token_id",    &c.mask_token_id},
    };
    for (const auto& r : reqs) {
        auto v = get_uint(md, r.key);
        if (!v) {
            return fail(ErrorCode::MissingRequiredMetadata,
                        fmt::format("dspark: missing '{}'", r.key), "dspark");
        }
        *r.out = static_cast<std::size_t>(*v);
    }

    c.head_dim = static_cast<std::size_t>(
        get_uint(md, "dspark.attention.key_length").value_or(
            c.n_head ? c.n_embd / c.n_head : 0));
    c.rope_dim   = c.head_dim;
    c.rope_base  = get_float(md, "dspark.rope.freq_base", 10000.0f);
    c.rms_eps    = get_float(md, "dspark.attention.layer_norm_rms_epsilon", 1e-6f);
    c.markov_rank = static_cast<std::size_t>(
        get_uint(md, "dspark.dspark.markov_rank").value_or(0));
    c.confidence_head        = get_flag(md, "dspark.dspark.confidence_head");
    c.confidence_with_markov = get_flag(md, "dspark.dspark.confidence_head_with_markov");
    c.log_snr_conditioning   = get_flag(md, "dspark.dspark.log_snr_conditioning");
    c.min_log_snr = get_float(md, "dspark.dspark.min_log_snr", 0.0f);
    c.max_log_snr = get_float(md, "dspark.dspark.max_log_snr", 0.0f);

    if (c.log_snr_conditioning && !(c.max_log_snr > c.min_log_snr)) {
        return fail(ErrorCode::InvalidModel,
                    "dspark: log-SNR conditioning needs max_log_snr > min_log_snr; "
                    "equal bounds would divide by zero in the featurizer",
                    "dspark");
    }

    // Which host layers this drafter taps.
    if (const auto* v = md.find("dspark.dspark.target_layers")) {
        if (const auto* view = std::get_if<MetadataArrayView>(v)) {
            for (auto L : read_i32_array(gguf, *view)) {
                if (L < 0) {
                    return fail(ErrorCode::InvalidModel,
                                "dspark: negative target layer", "dspark");
                }
                c.target_layers.push_back(static_cast<std::size_t>(L));
            }
        }
    }
    if (c.target_layers.empty()) {
        return fail(ErrorCode::InvalidModel,
                    "dspark: 'dspark.dspark.target_layers' is missing or empty; "
                    "without it there is nothing to read from the host",
                    "dspark");
    }

    // The drafter has no tokenizer of its own — it ties to the host's vocab,
    // so the width only exists as token_embd's shape.
    auto te = gguf.tensor("token_embd.weight");
    if (!te || te->info().dims.size() != 2) {
        return fail(ErrorCode::InvalidModel,
                    "dspark: token_embd.weight missing or not 2D", "dspark");
    }
    c.vocab = static_cast<std::size_t>(te->info().dims[1]);

    // A drafter wired to the wrong host is the failure worth guarding: the
    // shapes would multiply cleanly and the output would be nonsense.
    if (c.n_embd != host.hidden) {
        return fail(ErrorCode::InvalidModel,
                    fmt::format("dspark: drafter width {} does not match host {}",
                                c.n_embd, host.hidden),
                    "dspark");
    }
    if (c.vocab != host.vocab) {
        return fail(ErrorCode::InvalidModel,
                    fmt::format("dspark: drafter vocab {} does not match host {}",
                                c.vocab, host.vocab),
                    "dspark");
    }
    for (auto L : c.target_layers) {
        if (L >= host.n_layers) {
            return fail(ErrorCode::InvalidModel,
                        fmt::format("dspark: taps host layer {} but the host has "
                                    "only {} layers", L, host.n_layers),
                        "dspark");
        }
    }

    const std::size_t q_dim  = c.n_head    * c.head_dim;
    const std::size_t kv_dim = c.n_head_kv * c.head_dim;

    auto grab = [&](const std::string& name, std::size_t rows, std::size_t cols,
                    WeightRef& out, bool required) -> Result<void> {
        auto tv = gguf.tensor(name);
        if (!tv) {
            if (!required) return {};
            return fail(ErrorCode::MissingRequiredMetadata,
                        fmt::format("dspark: missing tensor '{}'", name), "dspark");
        }
        const TensorInfo& ti = tv->info();
        const bool ok = (cols == 0)
            ? (ti.dims.size() == 1 && ti.dims[0] == rows)
            : (ti.dims.size() == 2 && ti.dims[0] == cols && ti.dims[1] == rows);
        if (!ok) {
            return fail(ErrorCode::InvalidModel,
                        fmt::format("dspark: '{}' has the wrong shape", name),
                        "dspark");
        }
        out.dtype = ti.dtype;
        out.data  = tv->data();
        out.rows  = rows;
        out.cols  = cols;
        return {};
    };

    if (auto r = grab("token_embd.weight",       c.vocab, c.n_embd, I.tok_embd, true); !r) return Failure{r.error()};
    if (auto r = grab("output_norm.weight",      c.n_embd, 0,       I.output_norm, true); !r) return Failure{r.error()};
    if (auto r = grab("output.weight",           c.vocab, c.n_embd, I.output_head, false); !r) return Failure{r.error()};
    if (!I.output_head.valid()) I.output_head = I.tok_embd;

    if (auto r = grab("dspark.fc.weight",          c.n_embd, c.tap_width(), I.fc, true); !r) return Failure{r.error()};
    if (auto r = grab("dspark.hidden_norm.weight", c.n_embd, 0,             I.hidden_norm, true); !r) return Failure{r.error()};

    if (c.log_snr_conditioning) {
        if (auto r = grab("dspark.log_snr_fc1.weight", c.n_embd, k_snr_freq, I.snr_fc1_w, true); !r) return Failure{r.error()};
        if (auto r = grab("dspark.log_snr_fc1.bias",   c.n_embd, 0,          I.snr_fc1_b, true); !r) return Failure{r.error()};
        if (auto r = grab("dspark.log_snr_fc2.weight", c.n_embd, c.n_embd,   I.snr_fc2_w, true); !r) return Failure{r.error()};
        if (auto r = grab("dspark.log_snr_fc2.bias",   c.n_embd, 0,          I.snr_fc2_b, true); !r) return Failure{r.error()};
    }
    if (c.confidence_head) {
        const std::size_t conf_in = c.n_embd + (c.confidence_with_markov ? c.markov_rank : 0);
        if (auto r = grab("dspark.confidence_head.weight", 1, conf_in, I.conf_w, false); !r) return Failure{r.error()};
        if (auto r = grab("dspark.confidence_head.bias",   1, 0,       I.conf_b, false); !r) return Failure{r.error()};
    }
    if (c.markov_rank > 0) {
        if (auto r = grab("dspark.markov_head_a.weight", c.vocab, c.markov_rank, I.markov_a, false); !r) return Failure{r.error()};
        if (auto r = grab("dspark.markov_head_b.weight", c.vocab, c.markov_rank, I.markov_b, false); !r) return Failure{r.error()};
    }

    I.blocks.resize(c.n_layer);
    for (std::size_t i = 0; i < c.n_layer; ++i) {
        const std::string p = fmt::format("blk.{}.", i);
        Block& b = I.blocks[i];
        if (auto r = grab(p + "attn_norm.weight",   c.n_embd, 0,        b.attn_norm, true); !r) return Failure{r.error()};
        if (auto r = grab(p + "attn_q.weight",      q_dim,    c.n_embd, b.attn_q,    true); !r) return Failure{r.error()};
        if (auto r = grab(p + "attn_k.weight",      kv_dim,   c.n_embd, b.attn_k,    true); !r) return Failure{r.error()};
        if (auto r = grab(p + "attn_v.weight",      kv_dim,   c.n_embd, b.attn_v,    true); !r) return Failure{r.error()};
        if (auto r = grab(p + "attn_output.weight", c.n_embd, q_dim,    b.attn_out,  true); !r) return Failure{r.error()};
        if (auto r = grab(p + "attn_q_norm.weight", c.head_dim, 0,      b.attn_q_norm, false); !r) return Failure{r.error()};
        if (auto r = grab(p + "attn_k_norm.weight", c.head_dim, 0,      b.attn_k_norm, false); !r) return Failure{r.error()};
        if (auto r = grab(p + "ffn_norm.weight",    c.n_embd, 0,        b.ffn_norm, true); !r) return Failure{r.error()};
        if (auto r = grab(p + "ffn_gate.weight",    c.ffn_hidden, c.n_embd, b.ffn_gate, true); !r) return Failure{r.error()};
        if (auto r = grab(p + "ffn_up.weight",      c.ffn_hidden, c.n_embd, b.ffn_up,   true); !r) return Failure{r.error()};
        if (auto r = grab(p + "ffn_down.weight",    c.n_embd, c.ffn_hidden, b.ffn_down, true); !r) return Failure{r.error()};
    }

    // Dequantize the norm vectors once.
    auto load_vec = [&](const WeightRef& w, std::vector<float>& dst, std::size_t n,
                        std::size_t off) -> Result<void> {
        if (!w.valid()) { return {}; }
        return dequant_1d(w.dtype, w.data, dst.data() + off, n);
    };
    I.f_output_norm.assign(c.n_embd, 0.0f);
    I.f_hidden_norm.assign(c.n_embd, 0.0f);
    I.f_attn_norm.assign(c.n_layer * c.n_embd, 0.0f);
    I.f_ffn_norm.assign(c.n_layer * c.n_embd, 0.0f);
    I.f_q_norm.assign(c.n_layer * c.head_dim, 0.0f);
    I.f_k_norm.assign(c.n_layer * c.head_dim, 0.0f);
    if (auto r = load_vec(I.output_norm, I.f_output_norm, c.n_embd, 0); !r) return Failure{r.error()};
    if (auto r = load_vec(I.hidden_norm, I.f_hidden_norm, c.n_embd, 0); !r) return Failure{r.error()};
    for (std::size_t i = 0; i < c.n_layer; ++i) {
        if (auto r = load_vec(I.blocks[i].attn_norm, I.f_attn_norm, c.n_embd, i * c.n_embd); !r) return Failure{r.error()};
        if (auto r = load_vec(I.blocks[i].ffn_norm,  I.f_ffn_norm,  c.n_embd, i * c.n_embd); !r) return Failure{r.error()};
        if (auto r = load_vec(I.blocks[i].attn_q_norm, I.f_q_norm, c.head_dim, i * c.head_dim); !r) return Failure{r.error()};
        if (auto r = load_vec(I.blocks[i].attn_k_norm, I.f_k_norm, c.head_dim, i * c.head_dim); !r) return Failure{r.error()};
    }
    if (c.log_snr_conditioning) {
        I.f_snr_fc1_b.assign(c.n_embd, 0.0f);
        I.f_snr_fc2_b.assign(c.n_embd, 0.0f);
        if (auto r = load_vec(I.snr_fc1_b, I.f_snr_fc1_b, c.n_embd, 0); !r) return Failure{r.error()};
        if (auto r = load_vec(I.snr_fc2_b, I.f_snr_fc2_b, c.n_embd, 0); !r) return Failure{r.error()};
    }
    if (I.conf_b.valid()) {
        float b = 0.0f;
        if (auto r = dequant_1d(I.conf_b.dtype, I.conf_b.data, &b, 1); !r) return Failure{r.error()};
        I.conf_bias = b;
    }

    return self;
}

Result<void>
DsparkDrafter::draft(std::span<const float> ctx_feat, std::size_t n_ctx,
                     std::span<const std::int32_t> positions,
                     std::span<const std::int32_t> draft_in,
                     float* out_logits, float* out_conf) {
    Impl& I = *impl_;
    const DsparkConfig& c = I.cfg;
    auto& pool = *I.pool;

    const std::size_t nd   = draft_in.size();
    const std::size_t rows = n_ctx + nd;
    const std::size_t E    = c.n_embd;
    const std::size_t q_dim  = c.n_head    * c.head_dim;
    const std::size_t kv_dim = c.n_head_kv * c.head_dim;

    if (nd == 0) return {};
    if (ctx_feat.size() < n_ctx * c.tap_width()) {
        return fail(ErrorCode::InvalidModel,
                    fmt::format("dspark: ctx_feat holds {} floats, need {}",
                                ctx_feat.size(), n_ctx * c.tap_width()),
                    "dspark");
    }
    if (positions.size() < rows) {
        return fail(ErrorCode::InvalidModel,
                    fmt::format("dspark: {} positions for {} rows",
                                positions.size(), rows),
                    "dspark");
    }

    // Grow scratch to this call's shape. Draft blocks are a fixed size and
    // the context grows slowly, so this settles after a few calls.
    if (rows > I.cap_rows) {
        I.cap_rows   = rows;
        I.ctx_proj.assign(rows * E, 0.0f);
        I.attn_in.assign(rows * E, 0.0f);
        I.kbuf.assign(rows * kv_dim, 0.0f);
        I.vbuf.assign(rows * kv_dim, 0.0f);
        I.scores.assign(rows, 0.0f);
        I.h.assign(nd * E, 0.0f);
        I.resid.assign(nd * E, 0.0f);
        I.cur.assign(nd * E, 0.0f);
        I.qbuf.assign(nd * q_dim, 0.0f);
        I.attn_out.assign(nd * q_dim, 0.0f);
        I.gate.assign(nd * c.ffn_hidden, 0.0f);
        I.up.assign(nd * c.ffn_hidden, 0.0f);
        I.snr_feat.assign(nd * k_snr_freq, 0.0f);
        I.snr_hidden.assign(nd * E, 0.0f);
        I.snr_embed.assign(nd * E, 0.0f);
    }

    // ---- 1. Project the host taps down, once for the whole call ----------
    if (n_ctx > 0) {
        if (auto r = matmul_dispatch_threaded(pool, I.fc.dtype, I.fc.data,
                                              ctx_feat.data(), I.ctx_proj.data(),
                                              E, c.tap_width(), n_ctx); !r) return r;
        for (std::size_t t = 0; t < n_ctx; ++t) {
            kernels::rmsnorm_f32(I.ctx_proj.data() + t * E, I.f_hidden_norm.data(),
                                 I.ctx_proj.data() + t * E, E, c.rms_eps);
        }
    }

    dtrace("ctx_proj", -1, I.ctx_proj.data(), n_ctx * E);

    // ---- 2. Draft-block embeddings --------------------------------------
    for (std::size_t t = 0; t < nd; ++t) {
        const std::int32_t tok = draft_in[t];
        if (tok < 0 || static_cast<std::size_t>(tok) >= c.vocab) {
            return fail(ErrorCode::InvalidModel,
                        fmt::format("dspark: draft token {} out of vocab", tok),
                        "dspark");
        }
        if (auto r = read_row(I.tok_embd.dtype, I.tok_embd.data,
                              static_cast<std::size_t>(tok), E,
                              I.h.data() + t * E); !r) return r;
    }

    dtrace("embd", -1, I.h.data(), nd * E);

    // ---- 3. Log-SNR conditioning ----------------------------------------
    // The anchor of each block sits at max_log_snr and every other (still
    // masked) position at min_log_snr; that is the fixed round-one
    // inference convention, not a learned schedule.
    if (c.log_snr_conditioning) {
        const std::size_t bsz = c.block_size ? c.block_size : nd;
        const std::size_t half = k_snr_freq / 2;
        for (std::size_t t = 0; t < nd; ++t) {
            const float log_snr = (t % bsz == 0) ? c.max_log_snr : c.min_log_snr;
            const float u = (log_snr - c.min_log_snr) / (c.max_log_snr - c.min_log_snr)
                          * 1000.0f;
            float* f = I.snr_feat.data() + t * k_snr_freq;
            for (std::size_t i = 0; i < half; ++i) {
                const float freq = std::exp(-std::log(10000.0f)
                                            * static_cast<float>(i) / static_cast<float>(half));
                const float angle = u * freq;
                f[i]        = std::sin(angle);
                f[half + i] = std::cos(angle);
            }
        }
        if (auto r = matmul_dispatch_threaded(pool, I.snr_fc1_w.dtype, I.snr_fc1_w.data,
                                              I.snr_feat.data(), I.snr_hidden.data(),
                                              E, k_snr_freq, nd); !r) return r;
        for (std::size_t t = 0; t < nd; ++t) {
            float* v = I.snr_hidden.data() + t * E;
            for (std::size_t i = 0; i < E; ++i) v[i] += I.f_snr_fc1_b[i];
            kernels::silu_f32(v, v, E);
        }
        if (auto r = matmul_dispatch_threaded(pool, I.snr_fc2_w.dtype, I.snr_fc2_w.data,
                                              I.snr_hidden.data(), I.snr_embed.data(),
                                              E, E, nd); !r) return r;
        for (std::size_t t = 0; t < nd; ++t) {
            float* v = I.snr_embed.data() + t * E;
            float* d = I.h.data() + t * E;
            for (std::size_t i = 0; i < E; ++i) d[i] += v[i] + I.f_snr_fc2_b[i];
        }
    }

    dtrace("embd+snr", -1, I.h.data(), nd * E);

    // ---- 4. The trunk ----------------------------------------------------
    const float kq_scale = 1.0f / std::sqrt(static_cast<float>(c.head_dim));

    for (std::size_t il = 0; il < c.n_layer; ++il) {
        const Block& b = I.blocks[il];
        std::copy_n(I.h.begin(), nd * E, I.resid.begin());

        for (std::size_t t = 0; t < nd; ++t) {
            kernels::rmsnorm_f32(I.h.data() + t * E,
                                 I.f_attn_norm.data() + il * E,
                                 I.cur.data() + t * E, E, c.rms_eps);
        }

        // K and V are projected from context AND draft rows together; Q only
        // from the draft rows, because the context rows never issue a query.
        // Attention output is row-independent, so skipping them is exact
        // rather than an approximation.
        std::copy_n(I.ctx_proj.begin(), n_ctx * E, I.attn_in.begin());
        std::copy_n(I.cur.begin(), nd * E, I.attn_in.begin() + n_ctx * E);

        if (auto r = matmul_dispatch_threaded(pool, b.attn_k.dtype, b.attn_k.data,
                                              I.attn_in.data(), I.kbuf.data(),
                                              kv_dim, E, rows); !r) return r;
        if (auto r = matmul_dispatch_threaded(pool, b.attn_v.dtype, b.attn_v.data,
                                              I.attn_in.data(), I.vbuf.data(),
                                              kv_dim, E, rows); !r) return r;
        if (auto r = matmul_dispatch_threaded(pool, b.attn_q.dtype, b.attn_q.data,
                                              I.cur.data(), I.qbuf.data(),
                                              q_dim, E, nd); !r) return r;

        // Q/K RMSNorm per head, then RoPE at each row's own position.
        for (std::size_t t = 0; t < nd; ++t) {
            float* q = I.qbuf.data() + t * q_dim;
            if (b.attn_q_norm.valid()) {
                for (std::size_t hh = 0; hh < c.n_head; ++hh) {
                    kernels::rmsnorm_f32(q + hh * c.head_dim,
                                         I.f_q_norm.data() + il * c.head_dim,
                                         q + hh * c.head_dim, c.head_dim, c.rms_eps);
                }
            }
            kernels::rope_f32_ext(q, c.n_head, c.head_dim, c.rope_dim,
                                  static_cast<std::size_t>(positions[n_ctx + t]),
                                  c.rope_base, /*neox=*/true, nullptr, 1.0f);
        }
        for (std::size_t t = 0; t < rows; ++t) {
            float* k = I.kbuf.data() + t * kv_dim;
            if (b.attn_k_norm.valid()) {
                for (std::size_t hh = 0; hh < c.n_head_kv; ++hh) {
                    kernels::rmsnorm_f32(k + hh * c.head_dim,
                                         I.f_k_norm.data() + il * c.head_dim,
                                         k + hh * c.head_dim, c.head_dim, c.rms_eps);
                }
            }
            kernels::rope_f32_ext(k, c.n_head_kv, c.head_dim, c.rope_dim,
                                  static_cast<std::size_t>(positions[t]),
                                  c.rope_base, /*neox=*/true, nullptr, 1.0f);
        }

        // Fully non-causal: every draft row sees the whole context and all
        // of its siblings. That bidirectionality is what lets one pass
        // propose a block instead of a single next token.
        const std::size_t group = c.n_head / c.n_head_kv;
        for (std::size_t t = 0; t < nd; ++t) {
            const float* q_row = I.qbuf.data() + t * q_dim;
            float*       o_row = I.attn_out.data() + t * q_dim;
            for (std::size_t hh = 0; hh < c.n_head; ++hh) {
                const std::size_t hkv = hh / group;
                const float* qh = q_row + hh * c.head_dim;
                for (std::size_t r = 0; r < rows; ++r) {
                    const float* kh = I.kbuf.data() + r * kv_dim + hkv * c.head_dim;
                    float acc = 0.0f;
                    for (std::size_t i = 0; i < c.head_dim; ++i) acc += qh[i] * kh[i];
                    I.scores[r] = acc * kq_scale;
                }
                kernels::softmax_f32(I.scores.data(), I.scores.data(), rows);
                float* oh = o_row + hh * c.head_dim;
                for (std::size_t i = 0; i < c.head_dim; ++i) oh[i] = 0.0f;
                for (std::size_t r = 0; r < rows; ++r) {
                    const float wgt = I.scores[r];
                    const float* vh = I.vbuf.data() + r * kv_dim + hkv * c.head_dim;
                    for (std::size_t i = 0; i < c.head_dim; ++i) oh[i] += wgt * vh[i];
                }
            }
        }

        dtrace("q", (int)il, I.qbuf.data(), nd * q_dim);
        dtrace("k", (int)il, I.kbuf.data(), rows * kv_dim);
        dtrace("attn", (int)il, I.attn_out.data(), nd * q_dim);
        if (auto r = matmul_dispatch_threaded(pool, b.attn_out.dtype, b.attn_out.data,
                                              I.attn_out.data(), I.h.data(),
                                              E, q_dim, nd); !r) return r;
        kernels::add_f32(I.h.data(), I.resid.data(), I.h.data(), nd * E);

        // FFN
        std::copy_n(I.h.begin(), nd * E, I.resid.begin());
        for (std::size_t t = 0; t < nd; ++t) {
            kernels::rmsnorm_f32(I.h.data() + t * E,
                                 I.f_ffn_norm.data() + il * E,
                                 I.cur.data() + t * E, E, c.rms_eps);
        }
        if (auto r = matmul_dispatch_threaded(pool, b.ffn_gate.dtype, b.ffn_gate.data,
                                              I.cur.data(), I.gate.data(),
                                              c.ffn_hidden, E, nd); !r) return r;
        if (auto r = matmul_dispatch_threaded(pool, b.ffn_up.dtype, b.ffn_up.data,
                                              I.cur.data(), I.up.data(),
                                              c.ffn_hidden, E, nd); !r) return r;
        kernels::swiglu_f32(I.gate.data(), I.up.data(), I.gate.data(), nd * c.ffn_hidden);
        if (auto r = matmul_dispatch_threaded(pool, b.ffn_down.dtype, b.ffn_down.data,
                                              I.gate.data(), I.h.data(),
                                              E, c.ffn_hidden, nd); !r) return r;
        kernels::add_f32(I.h.data(), I.resid.data(), I.h.data(), nd * E);
    }

    dtrace("trunk_out", -1, I.h.data(), nd * E);

    // ---- 5. Head ---------------------------------------------------------
    for (std::size_t t = 0; t < nd; ++t) {
        kernels::rmsnorm_f32(I.h.data() + t * E, I.f_output_norm.data(),
                             I.cur.data() + t * E, E, c.rms_eps);
    }
    if (auto r = matmul_dispatch_threaded(pool, I.output_head.dtype, I.output_head.data,
                                          I.cur.data(), out_logits,
                                          c.vocab, E, nd); !r) return r;

    // The confidence head scores each proposed position. Its Markov half
    // needs per-token features the host-side loop supplies; with none
    // available those inputs stay zero, which is the same as the head
    // seeing only the trunk state.
    if (out_conf && I.conf_w.valid()) {
        const std::size_t conf_in = I.conf_w.cols;
        std::vector<float> in(conf_in, 0.0f);
        for (std::size_t t = 0; t < nd; ++t) {
            std::copy_n(I.cur.data() + t * E, std::min(E, conf_in), in.begin());
            float z = 0.0f;
            if (auto r = matvec_dispatch(I.conf_w.dtype, I.conf_w.data,
                                         in.data(), &z, 1, conf_in); !r) return r;
            out_conf[t] = 1.0f / (1.0f + std::exp(-(z + I.conf_bias)));
        }
    }
    return {};
}

} // namespace ultima::model
