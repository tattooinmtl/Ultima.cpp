#pragma once

#include "ultima/kv_cache/kv_cache.hpp"
#include "ultima/model/imodel.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace ultima::kv_cache {

// Per-slot state for the recurrent (gated delta-net) layers of a hybrid
// model, living beside the KVCache and indexed by the same SlotHandle.
//
// Two buffers per recurrent layer:
//   conv  [(width-1) * channels]      the causal convolution's history
//   delta [n_v_heads][d_state][d_state]  the linear-attention state
//
// The delta state dominates: at d_state 128 and 32 heads that is 2 MiB per
// layer per slot, against 96 KiB of conv history.
//
// PREFIX REUSE DOES NOT APPLY HERE. A KV entry is indexed by position, so
// keeping the first N of them is meaningful. A recurrence has no such
// structure: the state after N tokens cannot be recovered from the state
// after M > N without replaying from the start. So a slot whose stored
// tokens are only a *prefix* of the new prompt can continue, but a slot that
// diverges must reset() and re-run the whole prompt. Callers must ask
// `can_extend()` rather than assuming the KVCache's LCP is usable.
class RecurrentState {
public:
    struct Config {
        std::size_t n_slots  = 1;
        std::size_t n_layers = 0;              // total, including attention layers
        std::vector<bool> layer_is_recurrent;  // one entry per layer

        std::size_t conv_width    = 0;   // d_conv
        std::size_t conv_channels = 0;   // conv_dim
        std::size_t d_state       = 0;
        std::size_t n_v_heads     = 0;
    };

    explicit RecurrentState(const Config& cfg);
    ~RecurrentState();

    RecurrentState(const RecurrentState&)            = delete;
    RecurrentState& operator=(const RecurrentState&) = delete;

    const Config& config() const noexcept;

    // Bytes held for all slots. Worth surfacing: on a 27B hybrid this is
    // hundreds of megabytes and it is allocated up front, like the KV cache.
    std::size_t bytes() const noexcept;

    // Conv history for one layer of one slot, [(width-1) * channels], laid
    // out per channel with the oldest sample first. Null when the layer is
    // not recurrent.
    float* conv(SlotHandle slot, std::size_t layer) noexcept;

    // Delta-net state for one layer of one slot,
    // [n_v_heads][d_state][d_state] row-major, so one head is contiguous
    // and S[i][j] has j contiguous. Null when the layer is not recurrent.
    float* delta(SlotHandle slot, std::size_t layer) noexcept;

    // Head `h` of the delta state — the pointer delta_net_step_f32 wants.
    float* delta_head(SlotHandle slot, std::size_t layer, std::size_t head) noexcept;

    // Zero every buffer for the slot and mark it as holding no tokens.
    // This is what "start this sequence over" means for a recurrent layer.
    void reset(SlotHandle slot);

    // How many tokens have been folded into this slot's state.
    std::size_t pos(SlotHandle slot) const noexcept;
    void        advance(SlotHandle slot, std::size_t n_tokens) noexcept;

    // Whether the state can be carried forward for a prompt that shares
    // `lcp` leading tokens with what this slot already absorbed. True only
    // when the stored tokens are exactly a prefix of the new prompt — any
    // divergence means the state is wrong and reset() is required.
    bool can_extend(SlotHandle slot, std::size_t lcp) const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Build a Config from the model's own parameters. Returns a Config with
// n_layers == 0 when the architecture has no recurrent layers, which the
// runtime reads as "no recurrent state needed".
RecurrentState::Config
recurrent_config_for(const model::ArchParams& params,
                     std::size_t n_layers,
                     std::size_t n_slots);

} // namespace ultima::kv_cache
