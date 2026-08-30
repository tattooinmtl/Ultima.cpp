#include "ultima/kv_cache/kv_cache.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace ultima::kv_cache {

namespace {

std::size_t dtype_bytes(model::DataType t) noexcept {
    switch (t) {
        case model::DataType::F32:  return 4;
        case model::DataType::F16:  return 2;
        // Q8_0 storage in the KV cache is per-elt int8 + a per-block fp16
        // scale; treated here as 1 byte/elt for buffer sizing, with the
        // scale slot handled by the caller when Decision 09 §9.5 Q8_0 KV
        // wiring lands. In v0.1 the KV cache only allocates for f16.
        case model::DataType::Q8_0: return 1;
        default: return 2;
    }
}

} // namespace

struct KVCache::Impl {
    Config cfg;

    // One buffer per (slot, layer) for K and V. Layout per layer:
    //   [n_kv_heads, n_ctx, head_dim]   contiguous
    // Owned bytes.
    std::vector<std::vector<std::uint8_t>> k_buf;   // size = n_slots * n_layers
    std::vector<std::vector<std::uint8_t>> v_buf;

    struct SlotState {
        std::vector<TokenId>  tokens;
        std::size_t           pos           = 0;
        std::optional<SessionId> owner;                   // nullopt when free
        std::uint64_t         touched_at    = 0;          // monotonic
        std::vector<std::uint8_t> layers_written_at_pos;  // n_layers bitmap
    };
    std::vector<SlotState> slots;
    std::uint64_t          clock = 0;

    inline std::size_t bytes_per_layer() const noexcept {
        return cfg.n_kv_heads * cfg.n_ctx * cfg.head_dim * dtype_bytes(cfg.dtype);
    }
};

KVCache::KVCache(const Config& cfg) : impl_{std::make_unique<Impl>()} {
    if (cfg.n_slots == 0)  throw std::invalid_argument("KVCache: n_slots must be > 0");
    if (cfg.n_ctx == 0)    throw std::invalid_argument("KVCache: n_ctx must be > 0");
    if (cfg.n_layers == 0) throw std::invalid_argument("KVCache: n_layers must be > 0");
    if (cfg.n_kv_heads == 0 || cfg.head_dim == 0)
        throw std::invalid_argument("KVCache: n_kv_heads and head_dim must be > 0");

    impl_->cfg = cfg;

    const std::size_t total = cfg.n_slots * cfg.n_layers;
    impl_->k_buf.resize(total);
    impl_->v_buf.resize(total);
    const std::size_t per_layer_bytes = impl_->bytes_per_layer();
    for (std::size_t i = 0; i < total; ++i) {
        impl_->k_buf[i].assign(per_layer_bytes, 0);
        impl_->v_buf[i].assign(per_layer_bytes, 0);
    }

    impl_->slots.resize(cfg.n_slots);
    for (auto& s : impl_->slots) {
        s.tokens.reserve(cfg.n_ctx);
        s.layers_written_at_pos.assign(cfg.n_layers, 0);
    }
}

KVCache::~KVCache() = default;

const KVCache::Config& KVCache::config() const noexcept { return impl_->cfg; }

SlotHandle KVCache::acquire(SessionId session) {
    // 1) If session already owns a slot, hand it back untouched.
    for (std::size_t i = 0; i < impl_->slots.size(); ++i) {
        auto& s = impl_->slots[i];
        if (s.owner && *s.owner == session) {
            s.touched_at = ++impl_->clock;
            return SlotHandle{static_cast<std::uint32_t>(i)};
        }
    }
    // 2) Free slot.
    for (std::size_t i = 0; i < impl_->slots.size(); ++i) {
        auto& s = impl_->slots[i];
        if (!s.owner) {
            s.owner      = session;
            s.pos        = 0;
            s.tokens.clear();
            std::fill(s.layers_written_at_pos.begin(),
                      s.layers_written_at_pos.end(), 0);
            s.touched_at = ++impl_->clock;
            return SlotHandle{static_cast<std::uint32_t>(i)};
        }
    }
    // 3) LRU-steal.
    std::size_t victim = 0;
    std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t i = 0; i < impl_->slots.size(); ++i) {
        if (impl_->slots[i].touched_at < oldest) {
            oldest = impl_->slots[i].touched_at;
            victim = i;
        }
    }
    auto& s = impl_->slots[victim];
    s.owner      = session;
    s.pos        = 0;
    s.tokens.clear();
    std::fill(s.layers_written_at_pos.begin(),
              s.layers_written_at_pos.end(), 0);
    s.touched_at = ++impl_->clock;
    return SlotHandle{static_cast<std::uint32_t>(victim)};
}

void KVCache::release(SlotHandle slot) {
    if (!slot.valid() || slot.index() >= impl_->slots.size()) return;
    auto& s = impl_->slots[slot.index()];
    s.owner.reset();
    s.pos = 0;
    s.tokens.clear();
    std::fill(s.layers_written_at_pos.begin(),
              s.layers_written_at_pos.end(), 0);
}

std::size_t KVCache::pos(SlotHandle slot) const {
    if (!slot.valid() || slot.index() >= impl_->slots.size()) return 0;
    return impl_->slots[slot.index()].pos;
}

std::size_t
KVCache::reuse_prefix(SlotHandle slot, std::span<const TokenId> new_tokens) {
    if (!slot.valid() || slot.index() >= impl_->slots.size()) return 0;
    auto& s = impl_->slots[slot.index()];
    const std::size_t lcp_max = std::min(s.tokens.size(), new_tokens.size());
    std::size_t lcp = 0;
    while (lcp < lcp_max && s.tokens[lcp] == new_tokens[lcp]) ++lcp;

    // Truncate slot to the LCP. K/V beyond lcp is left in place; subsequent
    // append() calls overwrite. Re-mark layers as "not yet written at pos".
    s.tokens.resize(lcp);
    s.pos = lcp;
    std::fill(s.layers_written_at_pos.begin(),
              s.layers_written_at_pos.end(), 0);
    s.touched_at = ++impl_->clock;
    return lcp;
}

void KVCache::append(SlotHandle slot, std::size_t layer,
                     const void* k, const void* v) {
    if (!slot.valid() || slot.index() >= impl_->slots.size()) return;
    if (layer >= impl_->cfg.n_layers) return;
    auto& s = impl_->slots[slot.index()];
    if (s.pos >= impl_->cfg.n_ctx) return;   // slot full; caller should error

    const std::size_t elt_bytes = dtype_bytes(impl_->cfg.dtype);
    const std::size_t head_bytes = impl_->cfg.head_dim * elt_bytes;
    const std::size_t head_stride_bytes =
        impl_->cfg.n_ctx * impl_->cfg.head_dim * elt_bytes;

    // Buffer index (slot, layer) -> flat.
    const std::size_t bi = slot.index() * impl_->cfg.n_layers + layer;
    std::uint8_t* k_dst_base = impl_->k_buf[bi].data();
    std::uint8_t* v_dst_base = impl_->v_buf[bi].data();

    const std::uint8_t* k_src = static_cast<const std::uint8_t*>(k);
    const std::uint8_t* v_src = static_cast<const std::uint8_t*>(v);

    // For each head, scatter one head_dim slab from src to dst at position pos.
    for (std::size_t h = 0; h < impl_->cfg.n_kv_heads; ++h) {
        const std::size_t dst_off =
            h * head_stride_bytes + s.pos * head_bytes;
        std::memcpy(k_dst_base + dst_off, k_src + h * head_bytes, head_bytes);
        std::memcpy(v_dst_base + dst_off, v_src + h * head_bytes, head_bytes);
    }

    // Track whether every layer has been written at this position; only then
    // advance pos. Tokens vector grows when pos advances.
    s.layers_written_at_pos[layer] = 1;
    bool all_written = true;
    for (auto b : s.layers_written_at_pos) if (!b) { all_written = false; break; }
    if (all_written) {
        // Token id for this position is set separately via a companion API
        // in the runtime — but we still need to advance pos here for
        // subsequent append() calls to hit the next slab. The runtime will
        // patch s.tokens via reuse_prefix + repeated append flows; when
        // called in isolation (tests), pos advances without a corresponding
        // token id, which is fine because reuse_prefix is a no-op on an
        // empty tokens vector.
        s.pos += 1;
        std::fill(s.layers_written_at_pos.begin(),
                  s.layers_written_at_pos.end(), 0);
    }
    s.touched_at = ++impl_->clock;
}

LayerView KVCache::read(SlotHandle slot, std::size_t layer) const {
    LayerView v{};
    if (!slot.valid() || slot.index() >= impl_->slots.size()) return v;
    if (layer >= impl_->cfg.n_layers) return v;
    const auto& s = impl_->slots[slot.index()];
    const std::size_t bi = slot.index() * impl_->cfg.n_layers + layer;

    v.k           = impl_->k_buf[bi].data();
    v.v           = impl_->v_buf[bi].data();
    v.pos         = s.pos;
    v.head_dim    = impl_->cfg.head_dim;
    v.n_kv_heads  = impl_->cfg.n_kv_heads;
    v.stride_head = impl_->cfg.n_ctx * impl_->cfg.head_dim;
    v.dtype       = impl_->cfg.dtype;
    return v;
}

std::size_t KVCache::n_free_slots() const noexcept {
    std::size_t n = 0;
    for (const auto& s : impl_->slots) if (!s.owner) ++n;
    return n;
}
std::size_t KVCache::n_active_slots() const noexcept {
    return impl_->slots.size() - n_free_slots();
}

} // namespace ultima::kv_cache
