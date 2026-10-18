#include <doctest/doctest.h>

#include "ultima/kv_cache/kv_cache.hpp"
#include "ultima/model/dtype.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

namespace {

using ultima::kv_cache::KVCache;
using ultima::kv_cache::SessionId;

// Layout matching the transformer forward pass: write_at explicit position,
// then read_at up_to_pos+1, then commit_token.
KVCache::Config forward_cfg() {
    KVCache::Config c;
    c.n_slots    = 1;
    c.n_ctx      = 8;
    c.n_layers   = 3;
    c.n_kv_heads = 2;
    c.head_dim   = 4;
    c.dtype      = ultima::model::DataType::F32;
    return c;
}

std::vector<float> make_slab(std::size_t n_kv_heads, std::size_t head_dim, float base) {
    std::vector<float> v(n_kv_heads * head_dim);
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = base + static_cast<float>(i) * 0.001f;
    return v;
}

} // namespace

TEST_CASE("kv_cache: write_at + read_at exposes the just-written position") {
    KVCache kv(forward_cfg());
    auto s = kv.acquire(SessionId{1});

    auto k0 = make_slab(2, 4, 1.0f);
    auto v0 = make_slab(2, 4, 2.0f);

    // Write into position 0 for every layer (mirrors the forward pass).
    for (std::size_t L = 0; L < 3; ++L) {
        kv.write_at(s, L, /*pos=*/0, k0.data(), v0.data());
    }
    // pos hasn't advanced yet — commit_token owns that.
    CHECK(kv.pos(s) == 0);

    // read_at(pos+1=1) gives us a view including position 0.
    auto lv = kv.read_at(s, 0, 1);
    CHECK(lv.pos == 1);
    const auto* k = static_cast<const float*>(lv.k);
    // First head, first position, first element -> 1.0.
    CHECK(k[0] == doctest::Approx(1.0f));

    // commit_token advances pos.
    kv.commit_token(s, /*token=*/42);
    CHECK(kv.pos(s) == 1);
}

TEST_CASE("kv_cache: forward-pass loop advances pos consistently") {
    KVCache kv(forward_cfg());
    auto s = kv.acquire(SessionId{1});

    for (std::int32_t t = 0; t < 3; ++t) {
        auto k = make_slab(2, 4, 1.0f + static_cast<float>(t));
        auto v = make_slab(2, 4, 5.0f + static_cast<float>(t));
        for (std::size_t L = 0; L < 3; ++L) {
            kv.write_at(s, L, /*pos=*/static_cast<std::size_t>(t),
                        k.data(), v.data());
        }
        kv.commit_token(s, t);
    }
    CHECK(kv.pos(s) == 3);
}

TEST_CASE("kv_cache: reuse_prefix returns LCP after commit_token history") {
    KVCache kv(forward_cfg());
    auto s = kv.acquire(SessionId{1});

    for (std::int32_t t : {10, 20, 30, 40}) {
        auto k = make_slab(2, 4, 0.5f);
        auto v = make_slab(2, 4, 0.5f);
        for (std::size_t L = 0; L < 3; ++L) {
            kv.write_at(s, L, static_cast<std::size_t>(kv.pos(s)), k.data(), v.data());
        }
        kv.commit_token(s, t);
    }
    CHECK(kv.pos(s) == 4);

    // Same first 2, diverging at index 2.
    const std::vector<std::int32_t> new_seq = {10, 20, 99};
    const std::size_t lcp = kv.reuse_prefix(s, new_seq);
    CHECK(lcp == 2);
    CHECK(kv.pos(s) == 2);
}
