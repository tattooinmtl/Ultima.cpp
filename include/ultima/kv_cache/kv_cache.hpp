#pragma once

#include "ultima/model/dtype.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace ultima::kv_cache {

using SessionId = std::uint64_t;
using TokenId   = std::int32_t;

// Handle to a slot inside a KVCache. Cheap to copy; only the owning KVCache
// interprets it. An invalid handle compares equal to the default-constructed
// value.
class SlotHandle {
public:
    static constexpr std::uint32_t kInvalid = 0xFFFF'FFFFu;

    SlotHandle() noexcept = default;
    explicit SlotHandle(std::uint32_t idx) noexcept : index_{idx} {}

    bool          valid() const noexcept { return index_ != kInvalid; }
    std::uint32_t index() const noexcept { return index_; }

    friend bool operator==(SlotHandle a, SlotHandle b) noexcept {
        return a.index_ == b.index_;
    }
    friend bool operator!=(SlotHandle a, SlotHandle b) noexcept {
        return a.index_ != b.index_;
    }

private:
    std::uint32_t index_ = kInvalid;
};

// One layer's read view: pointers into the slot's K and V buffers at [0, pos),
// with the layout matching the storage from Decision 09 §9.2:
//   [n_kv_heads, n_ctx, head_dim]
// so a valid attention read is `pos` positions along the middle axis per head.
struct LayerView {
    const void* k       = nullptr;
    const void* v       = nullptr;
    std::size_t pos     = 0;
    std::size_t stride_head = 0;   // element stride between heads (n_ctx * head_dim)
    std::size_t head_dim    = 0;
    std::size_t n_kv_heads  = 0;
    model::DataType dtype   = model::DataType::F16;
};

// Fixed-size, pre-allocated KV cache. All slots are allocated up front at
// construction — no runtime growth. See Decision 09.
class KVCache {
public:
    struct Config {
        std::size_t     n_slots     = 1;
        std::size_t     n_ctx       = 4096;
        std::size_t     n_layers    = 0;
        std::size_t     n_kv_heads  = 0;
        std::size_t     head_dim    = 0;
        model::DataType dtype       = model::DataType::F16;
    };

    explicit KVCache(const Config& cfg);
    ~KVCache();

    KVCache(const KVCache&)            = delete;
    KVCache& operator=(const KVCache&) = delete;
    KVCache(KVCache&&)                 = delete;
    KVCache& operator=(KVCache&&)      = delete;

    const Config& config() const noexcept;

    // Acquire (or LRU-steal) a slot for this session. Returns the same slot on
    // repeated acquire from the same session (no eviction, tokens preserved).
    SlotHandle acquire(SessionId session);

    // Release a slot back to the free pool. Its tokens are dropped; the K/V
    // buffers are left in place (the next writer overwrites them).
    void release(SlotHandle slot);

    // Number of tokens currently valid in the slot's cache.
    std::size_t pos(SlotHandle slot) const;

    // Longest common prefix between the slot's stored token ids and
    // new_tokens. Truncates slot.pos to the LCP as a side effect so the
    // caller can prefill only tokens[lcp..].
    std::size_t reuse_prefix(SlotHandle slot, std::span<const TokenId> new_tokens);

    // Append one token's K/V for the given layer. `k` and `v` each point at
    // n_kv_heads * head_dim contiguous elements in the slot's dtype
    // (per-head contiguous). Advances slot.pos when called on the last layer.
    // (Callers should call append() for every layer at the same position, in
    // any order, exactly once per token; internal position advance is layer-
    // aware.)
    void append(SlotHandle slot, std::size_t layer,
                const void* k, const void* v);

    // Explicit-position variant used by the transformer forward pass. Writes
    // into the slot's (layer, position) slab without advancing pos or
    // toggling any per-layer "written" flags — the runtime writes at
    // `pos` first, then reads up to `pos+1` for attention, and finally
    // calls commit_token() to advance the slot.
    void write_at(SlotHandle slot, std::size_t layer, std::size_t position,
                  const void* k, const void* v);

    // Advance pos by 1 and record the token id (for prefix-reuse LCP).
    // Idempotent when the caller has already advanced via append().
    void commit_token(SlotHandle slot, TokenId token);

    // Read view for attention at this layer. positions [0, pos) are valid.
    LayerView read(SlotHandle slot, std::size_t layer) const;

    // Same as read() but returns a view whose .pos = up_to_pos, ignoring
    // slot.pos. Used by the forward pass to include the K/V just written at
    // the current position before commit_token() advances the counter.
    LayerView read_at(SlotHandle slot, std::size_t layer,
                      std::size_t up_to_pos) const;

    // Diagnostics.
    std::size_t n_free_slots()   const noexcept;
    std::size_t n_active_slots() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ultima::kv_cache
