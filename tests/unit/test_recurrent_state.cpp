#include <doctest/doctest.h>

#include "ultima/kv_cache/recurrent_state.hpp"

#include <vector>

using ultima::kv_cache::RecurrentState;
using ultima::kv_cache::SlotHandle;

namespace {

// A small stand-in for the qwen35 shape: 8 layers, 3-of-4 recurrent.
RecurrentState::Config small_config(std::size_t n_slots = 2) {
    RecurrentState::Config cfg{};
    cfg.n_slots  = n_slots;
    cfg.n_layers = 8;
    cfg.layer_is_recurrent.assign(8, false);
    for (std::size_t i = 0; i < 8; ++i) cfg.layer_is_recurrent[i] = ((i + 1) % 4) != 0;
    cfg.conv_width    = 4;
    cfg.conv_channels = 16;
    cfg.d_state       = 4;
    cfg.n_v_heads     = 2;
    return cfg;
}

} // namespace

TEST_CASE("recurrent state allocates only for recurrent layers") {
    RecurrentState rs{small_config(1)};
    // 6 of 8 layers are recurrent; each holds (4-1)*16 conv + 2*4*4 delta
    // floats = 48 + 32 = 80.
    CHECK(rs.bytes() == 6 * 80 * sizeof(float));
}

TEST_CASE("attention layers have no recurrent buffers") {
    RecurrentState rs{small_config()};
    SlotHandle s{0};
    for (std::size_t i = 0; i < 8; ++i) {
        const bool recurrent = ((i + 1) % 4) != 0;
        CHECK((rs.conv(s, i)  != nullptr) == recurrent);
        CHECK((rs.delta(s, i) != nullptr) == recurrent);
    }
    // Out of range is null rather than out-of-bounds.
    CHECK(rs.conv(s, 99)  == nullptr);
    CHECK(rs.delta(s, 99) == nullptr);
}

TEST_CASE("buffers start zeroed") {
    RecurrentState rs{small_config()};
    SlotHandle s{0};
    const float* c = rs.conv(s, 0);
    const float* d = rs.delta(s, 0);
    REQUIRE(c != nullptr);
    REQUIRE(d != nullptr);
    for (std::size_t i = 0; i < 48; ++i) CHECK(c[i] == 0.0f);
    for (std::size_t i = 0; i < 32; ++i) CHECK(d[i] == 0.0f);
}

TEST_CASE("layers and slots do not overlap") {
    RecurrentState rs{small_config(2)};
    SlotHandle s0{0}, s1{1};

    // Stamp every recurrent buffer with a distinct value, then check nothing
    // bled across. Aliasing here would silently mix two sequences together.
    float tag = 1.0f;
    for (std::size_t slot = 0; slot < 2; ++slot) {
        SlotHandle s{static_cast<std::uint32_t>(slot)};
        for (std::size_t i = 0; i < 8; ++i) {
            if (float* c = rs.conv(s, i))  { for (std::size_t j = 0; j < 48; ++j) c[j] = tag; tag += 1.0f; }
            if (float* d = rs.delta(s, i)) { for (std::size_t j = 0; j < 32; ++j) d[j] = tag; tag += 1.0f; }
        }
    }
    tag = 1.0f;
    for (std::size_t slot = 0; slot < 2; ++slot) {
        SlotHandle s{static_cast<std::uint32_t>(slot)};
        for (std::size_t i = 0; i < 8; ++i) {
            if (const float* c = rs.conv(s, i))  { for (std::size_t j = 0; j < 48; ++j) REQUIRE(c[j] == tag); tag += 1.0f; }
            if (const float* d = rs.delta(s, i)) { for (std::size_t j = 0; j < 32; ++j) REQUIRE(d[j] == tag); tag += 1.0f; }
        }
    }
    (void)s0; (void)s1;
}

TEST_CASE("delta_head indexes into the right head") {
    RecurrentState rs{small_config()};
    SlotHandle s{0};
    float* base = rs.delta(s, 0);
    REQUIRE(base != nullptr);

    CHECK(rs.delta_head(s, 0, 0) == base);
    CHECK(rs.delta_head(s, 0, 1) == base + 4 * 4);
    CHECK(rs.delta_head(s, 0, 2) == nullptr);   // only 2 heads
}

TEST_CASE("reset zeroes one slot and leaves the other alone") {
    RecurrentState rs{small_config(2)};
    SlotHandle s0{0}, s1{1};

    for (std::size_t i = 0; i < 8; ++i) {
        if (float* d = rs.delta(s0, i)) d[0] = 5.0f;
        if (float* d = rs.delta(s1, i)) d[0] = 7.0f;
    }
    rs.advance(s0, 10);
    rs.advance(s1, 20);

    rs.reset(s0);

    CHECK(rs.pos(s0) == 0);
    CHECK(rs.pos(s1) == 20);
    CHECK(rs.delta(s0, 0)[0] == 0.0f);
    CHECK(rs.delta(s1, 0)[0] == 7.0f);
}

TEST_CASE("a recurrent slot can extend a prompt but never rewind into one") {
    // This is the rule that separates recurrent state from KV. A KV cache
    // can keep the first N entries of a longer history because each is
    // indexed by position. A recurrence cannot: the state after N tokens is
    // not recoverable from the state after M > N.
    RecurrentState rs{small_config()};
    SlotHandle s{0};
    rs.advance(s, 100);

    CHECK(rs.can_extend(s, 100));        // same prompt, continuing
    CHECK(rs.can_extend(s, 250));        // strictly longer, same prefix
    CHECK_FALSE(rs.can_extend(s, 99));   // diverged one token back
    CHECK_FALSE(rs.can_extend(s, 0));    // entirely different prompt

    rs.reset(s);
    CHECK(rs.can_extend(s, 0));          // after a reset anything is fine
}

TEST_CASE("a model with no recurrent layers allocates nothing") {
    ultima::model::ArchParams p{};        // delta inactive, no layer map
    auto cfg = ultima::kv_cache::recurrent_config_for(p, 32, 4);
    CHECK(cfg.n_layers == 0);

    RecurrentState rs{cfg};
    CHECK(rs.bytes() == 0);
    CHECK(rs.conv(SlotHandle{0}, 0) == nullptr);
}

TEST_CASE("recurrent_config_for mirrors the architecture parameters") {
    ultima::model::ArchParams p{};
    p.delta.d_conv    = 4;
    p.delta.d_inner   = 4096;
    p.delta.d_state   = 128;
    p.delta.n_k_heads = 16;
    p.delta.n_v_heads = 32;
    p.layer_is_recurrent.assign(32, false);
    for (std::size_t i = 0; i < 32; ++i) p.layer_is_recurrent[i] = ((i + 1) % 4) != 0;

    auto cfg = ultima::kv_cache::recurrent_config_for(p, 32, 1);
    CHECK(cfg.n_layers      == 32);
    CHECK(cfg.conv_width    == 4);
    CHECK(cfg.conv_channels == 8192);   // key_dim*2 + value_dim
    CHECK(cfg.d_state       == 128);
    CHECK(cfg.n_v_heads     == 32);

    // 24 recurrent layers x (3*8192 conv + 32*128*128 delta) floats.
    RecurrentState rs{cfg};
    const std::size_t expect = 24ull * (3ull * 8192 + 32ull * 128 * 128) * sizeof(float);
    CHECK(rs.bytes() == expect);
}
