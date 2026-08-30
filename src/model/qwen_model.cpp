#include "ultima/kernels/norms.hpp"
#include "ultima/kv_cache/kv_cache.hpp"
#include "ultima/model/imodel.hpp"
#include "ultima/model/qwen_block.hpp"
#include "ultima/model/qwen_config.hpp"
#include "ultima/model/qwen_tensor_slots.hpp"
#include "ultima/model/weight_dispatch.hpp"

#include <fmt/format.h>

#include <memory>
#include <span>
#include <string>

namespace ultima::model {

namespace {

using core::ErrorCode;
using core::Failure;
using core::Result;
using core::fail;

// Concrete Qwen model. Covers both Qwen2 and Qwen3 — the Qwen3 differences
// (Q/K RMSNorm, YaRN scaling) live inside qwen_block_forward as per-flag
// branches driven by the tensor slot table + RopeConfig.
class QwenModel final : public IModel {
public:
    QwenModel(std::string arch,
              const LoadedModel& gguf,
              runtime::ThreadPool& pool)
        : arch_{std::move(arch)}, gguf_{gguf}, pool_{pool} {}

    Result<void> init() {
        if (auto r = parse_qwen_config(gguf_, arch_, dims_, rope_); !r) return r;
        auto tensors = resolve_qwen_tensors(gguf_, arch_,
                                            dims_.n_layers, dims_.hidden,
                                            dims_.n_kv_heads, dims_.head_dim,
                                            dims_.ffn_hidden, dims_.vocab);
        if (!tensors) return Failure{tensors.error()};
        tensors_ = std::move(*tensors);

        scratch_.init(dims_);
        if (auto r = preload_scratch_from_tensors(scratch_, tensors_, dims_); !r) return r;
        return {};
    }

    const ModelDims&  dims() const noexcept override { return dims_; }
    const RopeConfig& rope() const noexcept override { return rope_; }
    const char*       arch() const noexcept override { return arch_.c_str(); }

    Result<void> prefill(kv_cache::KVCache& kv,
                         kv_cache::SlotHandle slot,
                         std::span<const std::int32_t> tokens,
                         float* out_logits) override {
        if (tokens.empty()) return {};
        const std::size_t start_pos = kv.pos(slot);
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            const std::size_t abs_pos = start_pos + i;
            const bool is_last = (i + 1 == tokens.size());
            if (auto r = forward_one(kv, slot, tokens[i], abs_pos,
                                     is_last ? out_logits : nullptr); !r)
                return r;
            kv.commit_token(slot, tokens[i]);
        }
        return {};
    }

    Result<void> decode(kv_cache::KVCache& kv,
                        kv_cache::SlotHandle slot,
                        std::int32_t token,
                        float* out_logits) override {
        const std::size_t abs_pos = kv.pos(slot);
        if (auto r = forward_one(kv, slot, token, abs_pos, out_logits); !r) return r;
        kv.commit_token(slot, token);
        return {};
    }

private:
    Result<void> forward_one(kv_cache::KVCache& kv,
                             kv_cache::SlotHandle slot,
                             std::int32_t token,
                             std::size_t abs_pos,
                             float* out_logits) {
        // 1. Embedding lookup for the input token.
        if (token < 0 || static_cast<std::size_t>(token) >= dims_.vocab) {
            return fail(ErrorCode::InvalidModel,
                        fmt::format("token id {} out of vocab [0, {})",
                                    token, dims_.vocab),
                        "qwen_model");
        }
        if (auto r = read_row(tensors_.token_embd.dtype, tensors_.token_embd.data,
                              static_cast<std::size_t>(token), dims_.hidden,
                              scratch_.x.data()); !r) return r;

        // 2. Transformer stack.
        for (std::size_t L = 0; L < dims_.n_layers; ++L) {
            if (auto r = qwen_block_forward(scratch_, tensors_.blocks[L],
                                            dims_, rope_, kv, slot, L,
                                            abs_pos, pool_); !r) return r;
        }

        // 3. If logits requested, finalize.
        if (out_logits) {
            kernels::rmsnorm_f32(scratch_.x.data(),
                                 scratch_.output_norm_f32.data(),
                                 scratch_.x_norm.data(),
                                 dims_.hidden, dims_.rms_eps);
            if (auto r = matvec_dispatch_threaded(pool_,
                                                  tensors_.output_head.dtype,
                                                  tensors_.output_head.data,
                                                  scratch_.x_norm.data(),
                                                  out_logits,
                                                  dims_.vocab, dims_.hidden); !r) return r;
        }
        return {};
    }

    std::string             arch_;
    const LoadedModel&      gguf_;
    runtime::ThreadPool&    pool_;
    ModelDims               dims_{};
    RopeConfig              rope_{};
    QwenTensors             tensors_;
    BlockScratch            scratch_;
};

} // namespace

Result<std::unique_ptr<IModel>>
load_model(const LoadedModel& gguf, runtime::ThreadPool& pool) {
    const std::string& arch = gguf.architecture();
    if (arch != "qwen2" && arch != "qwen3") {
        return fail(ErrorCode::InvalidModel,
                    fmt::format("architecture '{}' not supported in v0.1 "
                                "(expected 'qwen2' or 'qwen3')", arch),
                    "load_model");
    }
    auto m = std::make_unique<QwenModel>(arch, gguf, pool);
    if (auto r = m->init(); !r) return Failure{r.error()};
    return std::unique_ptr<IModel>{m.release()};
}

} // namespace ultima::model
