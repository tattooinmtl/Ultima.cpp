#include "ultima/kv_cache/recurrent_state.hpp"

#include <algorithm>
#include <cstring>

namespace ultima::kv_cache {

namespace {

// Dense index of a recurrent layer among the recurrent layers only, so the
// three-quarters of blocks that need storage do not pay for the quarter
// that does not. kNotRecurrent marks an attention layer.
constexpr std::size_t kNotRecurrent = static_cast<std::size_t>(-1);

} // namespace

struct RecurrentState::Impl {
    Config cfg;

    std::vector<std::size_t> slot_of_layer;   // layer -> dense index or kNotRecurrent
    std::size_t n_recurrent = 0;

    std::size_t conv_per_layer  = 0;   // floats
    std::size_t delta_per_layer = 0;   // floats
    std::size_t floats_per_slot = 0;

    std::vector<float>       buf;
    std::vector<std::size_t> tokens;   // per slot
};

RecurrentState::RecurrentState(const Config& cfg)
    : impl_{std::make_unique<Impl>()} {
    impl_->cfg = cfg;

    impl_->slot_of_layer.assign(cfg.n_layers, kNotRecurrent);
    for (std::size_t i = 0; i < cfg.n_layers; ++i) {
        const bool rec = i < cfg.layer_is_recurrent.size() && cfg.layer_is_recurrent[i];
        if (rec) impl_->slot_of_layer[i] = impl_->n_recurrent++;
    }

    if (impl_->n_recurrent > 0 && cfg.conv_width > 0) {
        impl_->conv_per_layer  = (cfg.conv_width - 1) * cfg.conv_channels;
        impl_->delta_per_layer = cfg.n_v_heads * cfg.d_state * cfg.d_state;
        impl_->floats_per_slot =
            impl_->n_recurrent * (impl_->conv_per_layer + impl_->delta_per_layer);
    }

    const std::size_t n_slots = std::max<std::size_t>(cfg.n_slots, 1);
    impl_->buf.assign(impl_->floats_per_slot * n_slots, 0.0f);
    impl_->tokens.assign(n_slots, 0);
}

RecurrentState::~RecurrentState() = default;

const RecurrentState::Config& RecurrentState::config() const noexcept {
    return impl_->cfg;
}

std::size_t RecurrentState::bytes() const noexcept {
    return impl_->buf.size() * sizeof(float);
}

float* RecurrentState::conv(SlotHandle slot, std::size_t layer) noexcept {
    if (!slot.valid() || layer >= impl_->slot_of_layer.size()) return nullptr;
    const std::size_t r = impl_->slot_of_layer[layer];
    if (r == kNotRecurrent) return nullptr;
    const std::size_t s = slot.index();
    if (s >= impl_->tokens.size()) return nullptr;

    // Each slot holds all conv buffers, then all delta buffers, so a walk
    // over the recurrence during decode touches two contiguous regions
    // rather than interleaving two very differently sized ones.
    return impl_->buf.data() + s * impl_->floats_per_slot
         + r * impl_->conv_per_layer;
}

float* RecurrentState::delta(SlotHandle slot, std::size_t layer) noexcept {
    if (!slot.valid() || layer >= impl_->slot_of_layer.size()) return nullptr;
    const std::size_t r = impl_->slot_of_layer[layer];
    if (r == kNotRecurrent) return nullptr;
    const std::size_t s = slot.index();
    if (s >= impl_->tokens.size()) return nullptr;

    return impl_->buf.data() + s * impl_->floats_per_slot
         + impl_->n_recurrent * impl_->conv_per_layer
         + r * impl_->delta_per_layer;
}

float* RecurrentState::delta_head(SlotHandle slot, std::size_t layer,
                                  std::size_t head) noexcept {
    float* base = delta(slot, layer);
    if (!base || head >= impl_->cfg.n_v_heads) return nullptr;
    return base + head * impl_->cfg.d_state * impl_->cfg.d_state;
}

void RecurrentState::reset(SlotHandle slot) {
    if (!slot.valid()) return;
    const std::size_t s = slot.index();
    if (s >= impl_->tokens.size()) return;

    if (impl_->floats_per_slot > 0) {
        std::memset(impl_->buf.data() + s * impl_->floats_per_slot, 0,
                    impl_->floats_per_slot * sizeof(float));
    }
    impl_->tokens[s] = 0;
}

std::size_t RecurrentState::pos(SlotHandle slot) const noexcept {
    if (!slot.valid()) return 0;
    const std::size_t s = slot.index();
    return s < impl_->tokens.size() ? impl_->tokens[s] : 0;
}

void RecurrentState::advance(SlotHandle slot, std::size_t n_tokens) noexcept {
    if (!slot.valid()) return;
    const std::size_t s = slot.index();
    if (s < impl_->tokens.size()) impl_->tokens[s] += n_tokens;
}

bool RecurrentState::can_extend(SlotHandle slot, std::size_t lcp) const noexcept {
    // The state encodes exactly the tokens already absorbed. It can be
    // carried forward only if every one of them is still a prefix of the new
    // prompt; keeping a shorter prefix is not possible without replaying.
    return pos(slot) <= lcp;
}

RecurrentState::Config
recurrent_config_for(const model::ArchParams& params,
                     std::size_t n_layers,
                     std::size_t n_slots) {
    RecurrentState::Config cfg{};
    if (!params.delta.active() || params.layer_is_recurrent.empty()) {
        return cfg;   // n_layers stays 0: nothing to allocate
    }
    cfg.n_slots            = std::max<std::size_t>(n_slots, 1);
    cfg.n_layers           = n_layers;
    cfg.layer_is_recurrent = params.layer_is_recurrent;
    cfg.conv_width         = params.delta.d_conv;
    cfg.conv_channels      = params.delta.conv_dim();
    cfg.d_state            = params.delta.d_state;
    cfg.n_v_heads          = params.delta.n_v_heads;
    return cfg;
}

} // namespace ultima::kv_cache
