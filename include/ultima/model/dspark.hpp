#pragma once

#include "ultima/core/error.hpp"
#include "ultima/model/imodel.hpp"
#include "ultima/model/loaded_model.hpp"
#include "ultima/model/qwen_tensor_slots.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ultima::runtime { class ThreadPool; }

namespace ultima::model {

// A dspark speculative drafter.
//
// This is not an IModel and deliberately does not pretend to be one: it has
// no tokenizer, it cannot be prompted, and it produces nothing on its own.
// It reads a *host* model's hidden state at a few depths and proposes a
// whole block of following tokens at once, which the host then verifies.
//
// Ternary-Bonsai-27B-dspark is 6 blocks and 3.6B parameters against a
// 64-layer host, and its dspark.target_layers names the host layers it
// taps. Its embedding width and vocabulary have to match the host exactly.
//
// Shape of one call: the tap features for `n_ctx` already-decoded positions
// arrive as the context, the draft block arrives as `block_size` token ids
// (mask_token_id where unknown), and attention is fully non-causal — every
// draft position sees the whole context and all of its siblings, which is
// what lets one pass propose several tokens instead of one.
struct DsparkConfig {
    std::size_t n_embd       = 0;
    std::size_t n_layer      = 0;
    std::size_t n_head       = 0;
    std::size_t n_head_kv    = 0;
    std::size_t head_dim     = 0;
    std::size_t ffn_hidden   = 0;
    std::size_t vocab        = 0;
    float       rms_eps      = 1e-6f;
    float       rope_base    = 10000.0f;
    std::size_t rope_dim     = 0;

    std::size_t block_size    = 0;   // tokens proposed per call
    std::size_t mask_token_id = 0;
    std::size_t markov_rank   = 0;

    std::vector<std::size_t> target_layers;   // host layer indices to tap

    bool  confidence_head       = false;
    bool  confidence_with_markov = false;
    bool  log_snr_conditioning  = false;
    float min_log_snr = 0.0f;
    float max_log_snr = 0.0f;

    // Width of one context row: one host hidden state per tapped layer,
    // concatenated. This is the fc projection's input width.
    std::size_t tap_width() const noexcept { return target_layers.size() * n_embd; }
};

class DsparkDrafter {
public:
    ~DsparkDrafter();
    DsparkDrafter(const DsparkDrafter&)            = delete;
    DsparkDrafter& operator=(const DsparkDrafter&) = delete;

    // Loads the drafter alone. `host` supplies the dimensions this must
    // agree with; a mismatch is an error rather than something to coerce,
    // because a drafter wired to the wrong host produces confident garbage.
    static core::Result<std::unique_ptr<DsparkDrafter>>
    load(const LoadedModel& gguf, const ModelDims& host, runtime::ThreadPool& pool);

    const DsparkConfig& config() const noexcept;

    // Propose one block.
    //
    //   ctx_feat   [n_ctx][tap_width()]  host taps, oldest first
    //   positions  [n_ctx + block_size]  absolute positions of every row
    //   draft_in   [block_size]          current block contents; entries the
    //                                    caller has not resolved must be
    //                                    config().mask_token_id
    //   out_logits [block_size][vocab]
    //   out_conf   [block_size]          sigmoid confidence, when the file
    //                                    carries a confidence head; left
    //                                    untouched otherwise
    core::Result<void>
    draft(std::span<const float> ctx_feat, std::size_t n_ctx,
          std::span<const std::int32_t> positions,
          std::span<const std::int32_t> draft_in,
          float* out_logits, float* out_conf);

private:
    DsparkDrafter();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ultima::model
