#include "ultima/kernels/hadamard.hpp"
#include "ultima/kernels/norms.hpp"
#include "ultima/kv_cache/kv_cache.hpp"
#include "ultima/model/arch_registry.hpp"
#include "ultima/model/imodel.hpp"
#include "ultima/model/qwen_block.hpp"
#include "ultima/model/qwen_config.hpp"
#include "ultima/model/qwen_tensor_slots.hpp"
#include "ultima/model/weight_dispatch.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <span>
#include <string>

namespace ultima::model {

namespace {

using core::ErrorCode;
using core::Failure;
using core::Result;
using core::fail;

// One concrete model covering every llama-family architecture in the registry.
// The per-architecture differences — fused vs separate QKV, dense vs fused vs
// MoE feed-forward, RoPE convention, Granite's scalar corrections — are data
// in ArchParams, resolved once at load time and branched on inside
// block_forward rather than duplicated into subclasses.
class TransformerModel final : public IModel {
public:
    TransformerModel(std::string arch,
                     const ArchSpec& spec,
                     const LoadedModel& gguf,
                     runtime::ThreadPool& pool)
        : arch_{std::move(arch)}, spec_{spec}, gguf_{gguf}, pool_{pool} {}

    Result<void> init() {
        if (auto r = parse_arch_config(gguf_, spec_, arch_, dims_, rope_, params_); !r)
            return r;
        // The Hadamard fold has to be known before tensor resolution, which
        // is what marks each weight as folded.
        auto fold = parse_hadamard_fold(gguf_);
        if (!fold) return Failure{fold.error()};
        fold_ = std::move(*fold);

        auto tensors = resolve_model_tensors(gguf_, arch_, dims_, params_,
                                             fold_ ? &*fold_ : nullptr);
        if (!tensors) return Failure{tensors.error()};
        tensors_ = std::move(*tensors);

        scratch_.init(dims_, params_, k_prefill_batch);
        scratch_.fold = fold_ ? &*fold_ : nullptr;
        if (auto r = preload_scratch_from_tensors(scratch_, tensors_, dims_, params_); !r)
            return r;

        // The recurrent layers need their own per-slot state beside the KV
        // cache. Sized here so the allocation is visible at load time rather
        // than appearing on the first token.
        rstate_ = std::make_unique<kv_cache::RecurrentState>(
            kv_cache::recurrent_config_for(params_, dims_.n_layers, k_state_slots));
        return {};
    }

    const ModelDims&  dims()   const noexcept override { return dims_; }
    const RopeConfig& rope()   const noexcept override { return rope_; }
    const ArchParams& params() const noexcept override { return params_; }
    const char*       arch()   const noexcept override { return arch_.c_str(); }

    bool supports_prefix_reuse() const noexcept override {
        return !params_.delta.active();
    }

    void set_layer_taps(const LayerTaps* taps) override { taps_ = taps; }
    bool supports_layer_taps() const noexcept override { return true; }

    Result<void> prefill(kv_cache::KVCache& kv,
                         kv_cache::SlotHandle slot,
                         std::span<const std::int32_t> tokens,
                         float* out_logits) override {
        if (tokens.empty()) return {};
        const std::size_t start_pos = kv.pos(slot);

        // The recurrent state must have absorbed exactly the tokens the KV
        // cache holds. A caller that truncated the KV to a shared prefix
        // would break that, and the recurrence has no way to rewind — so
        // this is an error rather than a silent wrong answer.
        if (params_.delta.active()) {
            if (start_pos == 0) {
                rstate_->reset(slot);
            } else if (rstate_->pos(slot) != start_pos) {
                return fail(ErrorCode::InvalidModel,
                            fmt::format("recurrent state holds {} tokens but prefill "
                                        "starts at {}: a hybrid model cannot resume "
                                        "from a truncated prefix, it must re-prefill "
                                        "from zero",
                                        rstate_->pos(slot), start_pos),
                            "transformer_model");
            }
        }
        // Run the prompt in chunks so each projection streams its weights
        // once for the whole chunk instead of once per token. Only the very
        // last token's logits are wanted, so every earlier chunk skips the
        // vocab-sized output matmul entirely.
        for (std::size_t i = 0; i < tokens.size(); i += k_prefill_batch) {
            const std::size_t n = std::min(k_prefill_batch, tokens.size() - i);
            const bool last_chunk = (i + n == tokens.size());
            if (auto r = forward_batch(kv, slot, tokens.data() + i, n, start_pos + i,
                                       last_chunk ? out_logits : nullptr); !r) {
                return r;
            }
            for (std::size_t j = 0; j < n; ++j) kv.commit_token(slot, tokens[i + j]);
            if (params_.delta.active()) rstate_->advance(slot, n);
        }
        return {};
    }

    Result<void> decode(kv_cache::KVCache& kv,
                        kv_cache::SlotHandle slot,
                        std::int32_t token,
                        float* out_logits) override {
        const std::size_t abs_pos = kv.pos(slot);
        if (auto r = forward_batch(kv, slot, &token, 1, abs_pos, out_logits); !r) return r;
        kv.commit_token(slot, token);
        if (params_.delta.active()) rstate_->advance(slot, 1);
        return {};
    }

private:
    // `n_tok` consecutive tokens starting at `start_pos`. Writes logits for
    // the LAST token only, which is all any caller needs: prefill wants the
    // token after the prompt, and decode has one token anyway.
    Result<void> forward_batch(kv_cache::KVCache& kv,
                               kv_cache::SlotHandle slot,
                               const std::int32_t* tokens,
                               std::size_t n_tok,
                               std::size_t start_pos,
                               float* out_logits) {
        if (n_tok == 0) return {};

        // 1. Embedding lookup for each token in the chunk.
        for (std::size_t t = 0; t < n_tok; ++t) {
            const std::int32_t token = tokens[t];
            if (token < 0 || static_cast<std::size_t>(token) >= dims_.vocab) {
                return fail(ErrorCode::InvalidModel,
                            fmt::format("token id {} out of vocab [0, {})",
                                        token, dims_.vocab),
                            "transformer_model");
            }
            float* dst = scratch_.x.data() + t * dims_.hidden;
            if (auto r = read_row(tensors_.token_embd.dtype, tensors_.token_embd.data,
                                  static_cast<std::size_t>(token), dims_.hidden,
                                  dst); !r) return r;

            // A Hadamard-latent embedding table stores rotated rows, so the
            // primal basis has to be restored right after the lookup:
            // h = s * (H z).
            if (fold_ && fold_->is_inverse("token_embd.weight")) {
                kernels::hadamard_unfold_f32(dst, dst, dims_.hidden, fold_->block,
                                             fold_->signs_for(dims_.hidden));
            }

            // Granite scales the embedding before the first block.
            if (params_.embedding_scale != 1.0f) {
                for (std::size_t i = 0; i < dims_.hidden; ++i)
                    dst[i] *= params_.embedding_scale;
            }
        }

        // 2. Transformer stack, capturing layer inputs on the way through.
        for (std::size_t L = 0; L < dims_.n_layers; ++L) {
            capture_taps(L, n_tok, start_pos);
            if (auto r = block_forward(scratch_, tensors_.blocks[L],
                                       dims_, params_, rope_, kv, slot, L,
                                       start_pos, n_tok, pool_, rstate_.get()); !r)
                return r;
        }

        // 3. If logits requested, finalize from the last token's row.
        if (out_logits) {
            const float* last = scratch_.x.data() + (n_tok - 1) * dims_.hidden;
            kernels::rmsnorm_f32(last,
                                 scratch_.output_norm_f32.data(),
                                 scratch_.x_norm.data(),
                                 dims_.hidden, dims_.rms_eps);
            const float* head_in = scratch_.x_norm.data();
            if (tensors_.output_head.hadamard_folded && fold_) {
                kernels::hadamard_fold_f32(scratch_.x_norm.data(),
                                           scratch_.fold_buf.data(), dims_.hidden,
                                           fold_->block,
                                           fold_->signs_for(dims_.hidden));
                head_in = scratch_.fold_buf.data();
            }
            if (auto r = matvec_dispatch_threaded(pool_,
                                                  tensors_.output_head.dtype,
                                                  tensors_.output_head.data,
                                                  head_in,
                                                  out_logits,
                                                  dims_.vocab, dims_.hidden); !r) return r;
            // Granite divides the logits by a fixed scale.
            if (params_.logit_scale != 1.0f && params_.logit_scale != 0.0f) {
                const float inv = 1.0f / params_.logit_scale;
                for (std::size_t i = 0; i < dims_.vocab; ++i) out_logits[i] *= inv;
            }
        }
        return {};
    }

    std::string             arch_;
    const ArchSpec&         spec_;
    const LoadedModel&      gguf_;
    runtime::ThreadPool&    pool_;
    // Matches the KV cache's slot count; both are indexed by the same
    // SlotHandle, so they must agree.
    static constexpr std::size_t k_state_slots = 8;

    // Prompt tokens processed per forward pass. Wider means each
    // projection streams its weights once for more tokens; the
    // measured curve (benchmarks/bench_gemm) is still climbing at 32,
    // and 32 costs about 15 MiB of scratch on the widest model here.
    static constexpr std::size_t k_prefill_batch = 32;
    std::unique_ptr<kv_cache::RecurrentState> rstate_;

    // Copy the residual stream entering block `L` into the tap buffer, for
    // every tap that names this layer. Done before the block runs, so it is
    // the input and not the output.
    void capture_taps(std::size_t L, std::size_t n_tok, std::size_t start_pos) {
        if (!taps_ || !taps_->active()) return;
        const std::size_t n_taps = taps_->layers.size();
        const std::size_t stride = taps_->stride ? taps_->stride : n_taps * dims_.hidden;

        for (std::size_t i = 0; i < n_taps; ++i) {
            if (taps_->layers[i] != L) continue;
            for (std::size_t t = 0; t < n_tok; ++t) {
                std::copy_n(scratch_.x.data() + t * dims_.hidden, dims_.hidden,
                            taps_->out + (start_pos + t) * stride + i * dims_.hidden);
            }
        }
    }

    const LayerTaps*            taps_ = nullptr;
    std::optional<HadamardFold> fold_;

    ModelDims               dims_{};
    RopeConfig              rope_{};
    ArchParams              params_{};
    ModelTensors            tensors_;
    BlockScratch            scratch_;
};

} // namespace

Result<std::unique_ptr<IModel>>
load_model(const LoadedModel& gguf, runtime::ThreadPool& pool) {
    const std::string& arch = gguf.architecture();
    const ArchSpec* spec = find_arch(arch);

    if (!spec) {
        return fail(ErrorCode::InvalidModel,
                    fmt::format("unknown architecture '{}'. Run 'ultima archs' "
                                "for the architectures Ultima knows about.", arch),
                    "load_model");
    }
    if (!spec->supported) {
        return fail(ErrorCode::InvalidModel,
                    fmt::format("architecture '{}' is recognized but not yet "
                                "runnable: {}", arch, spec->notes),
                    "load_model");
    }

    auto m = std::make_unique<TransformerModel>(arch, *spec, gguf, pool);
    if (auto r = m->init(); !r) return Failure{r.error()};
    return std::unique_ptr<IModel>{m.release()};
}

} // namespace ultima::model
