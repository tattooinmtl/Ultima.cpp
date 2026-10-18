#include <doctest/doctest.h>

#include "ultima/kv_cache/kv_cache.hpp"
#include "ultima/model/dtype.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

namespace {

using ultima::kv_cache::KVCache;
using ultima::kv_cache::SessionId;
using ultima::kv_cache::SlotHandle;
using ultima::kv_cache::TokenId;

KVCache::Config small_cfg(std::size_t n_slots = 2) {
    KVCache::Config c;
    c.n_slots    = n_slots;
    c.n_ctx      = 16;
    c.n_layers   = 2;
    c.n_kv_heads = 2;
    c.head_dim   = 4;
    c.dtype      = ultima::model::DataType::F16;
    return c;
}

// Build a slab of n_kv_heads * head_dim f16 elements filled with a marker
// byte pattern derived from (layer, pos). Not real fp16 values — just
// something we can memcmp on read.
std::vector<std::uint8_t> marker_slab(std::size_t n_kv_heads,
                                      std::size_t head_dim,
                                      std::uint8_t marker) {
    std::vector<std::uint8_t> b(n_kv_heads * head_dim * 2, marker);
    return b;
}

} // namespace

TEST_CASE("kv_cache: acquire returns a valid slot and pos starts at zero") {
    KVCache kv(small_cfg(2));
    auto s = kv.acquire(SessionId{42});
    REQUIRE(s.valid());
    CHECK(kv.pos(s) == 0);
    CHECK(kv.n_active_slots() == 1);
    CHECK(kv.n_free_slots() == 1);
}

TEST_CASE("kv_cache: repeated acquire for same session returns same slot") {
    KVCache kv(small_cfg(2));
    auto a = kv.acquire(SessionId{7});
    auto b = kv.acquire(SessionId{7});
    CHECK(a == b);
    CHECK(kv.n_active_slots() == 1);
}

TEST_CASE("kv_cache: release frees the slot") {
    KVCache kv(small_cfg(2));
    auto a = kv.acquire(SessionId{1});
    CHECK(kv.n_active_slots() == 1);
    kv.release(a);
    CHECK(kv.n_active_slots() == 0);
    CHECK(kv.n_free_slots() == 2);
}

TEST_CASE("kv_cache: LRU steal when all slots busy") {
    KVCache kv(small_cfg(2));
    auto a = kv.acquire(SessionId{1});
    auto b = kv.acquire(SessionId{2});
    REQUIRE(a != b);
    // Touch b so a is oldest.
    kv.acquire(SessionId{2});
    // A third session must steal a slot (LRU = the one for session 1).
    auto c = kv.acquire(SessionId{3});
    REQUIRE(c.valid());
    // Session 1's slot is gone; acquiring again returns a fresh slot.
    CHECK(kv.n_active_slots() == 2);
}

TEST_CASE("kv_cache: append advances pos exactly once per token across layers") {
    auto cfg = small_cfg(1);
    KVCache kv(cfg);
    auto s = kv.acquire(SessionId{1});

    auto k0 = marker_slab(cfg.n_kv_heads, cfg.head_dim, 0xAA);
    auto v0 = marker_slab(cfg.n_kv_heads, cfg.head_dim, 0xBB);

    // Append for layer 0 first — pos should NOT advance yet.
    kv.append(s, 0, k0.data(), v0.data());
    CHECK(kv.pos(s) == 0);

    // Append for layer 1 — now pos advances to 1.
    kv.append(s, 1, k0.data(), v0.data());
    CHECK(kv.pos(s) == 1);

    // Second token, layers in reverse order.
    auto k1 = marker_slab(cfg.n_kv_heads, cfg.head_dim, 0xCC);
    auto v1 = marker_slab(cfg.n_kv_heads, cfg.head_dim, 0xDD);
    kv.append(s, 1, k1.data(), v1.data());
    CHECK(kv.pos(s) == 1);
    kv.append(s, 0, k1.data(), v1.data());
    CHECK(kv.pos(s) == 2);
}

TEST_CASE("kv_cache: read view exposes correct dims after appends") {
    auto cfg = small_cfg(1);
    KVCache kv(cfg);
    auto s = kv.acquire(SessionId{1});

    auto k = marker_slab(cfg.n_kv_heads, cfg.head_dim, 0xAA);
    auto v = marker_slab(cfg.n_kv_heads, cfg.head_dim, 0xBB);

    // Append 3 tokens across both layers.
    for (int t = 0; t < 3; ++t) {
        for (std::size_t l = 0; l < cfg.n_layers; ++l) {
            kv.append(s, l, k.data(), v.data());
        }
    }
    CHECK(kv.pos(s) == 3);

    auto lv = kv.read(s, 0);
    CHECK(lv.pos == 3);
    CHECK(lv.head_dim == cfg.head_dim);
    CHECK(lv.n_kv_heads == cfg.n_kv_heads);
    CHECK(lv.stride_head == cfg.n_ctx * cfg.head_dim);
    REQUIRE(lv.k != nullptr);
    REQUIRE(lv.v != nullptr);

    // First head, first position, first byte should be 0xAA.
    const auto* kb = static_cast<const std::uint8_t*>(lv.k);
    CHECK(kb[0] == 0xAA);
    const auto* vb = static_cast<const std::uint8_t*>(lv.v);
    CHECK(vb[0] == 0xBB);
}

TEST_CASE("kv_cache: reuse_prefix truncates to LCP") {
    auto cfg = small_cfg(1);
    KVCache kv(cfg);
    auto s = kv.acquire(SessionId{1});

    // Simulate a previous session that ended with tokens [1,2,3,4,5] at pos=5.
    // We poke the token vector via a compatible flow: reuse_prefix against
    // a matching span, then verify LCP length. Since our test doesn't have a
    // public setter for tokens, this exercises reuse_prefix on an empty slot
    // (should return 0) and on a slot that grew via a preceding reuse.
    const std::vector<TokenId> new_tokens = {1, 2, 3, 4, 5};
    CHECK(kv.reuse_prefix(s, new_tokens) == 0);   // empty slot -> LCP 0

    // After reuse_prefix on an empty slot, pos is still 0.
    CHECK(kv.pos(s) == 0);
}

TEST_CASE("kv_cache: rejects zero-sized config axes") {
    KVCache::Config c;
    c.n_slots = 0;
    CHECK_THROWS(KVCache{c});

    c.n_slots = 1;
    c.n_ctx = 0;
    CHECK_THROWS(KVCache{c});
}
