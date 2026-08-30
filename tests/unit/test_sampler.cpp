#include <doctest/doctest.h>

#include "ultima/sampler/sampler.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

using ultima::sampler::sample;
using ultima::sampler::argmax;
using ultima::sampler::SamplerContext;
using ultima::sampler::SamplerParams;
using ultima::sampler::TokenId;

std::vector<float> logits_of(std::vector<float> v) { return v; }

} // namespace

TEST_CASE("sampler: argmax returns the max-logit index") {
    auto l = logits_of({0.1f, 0.3f, 0.2f, 0.9f, 0.5f});
    CHECK(argmax(l) == 3);
}

TEST_CASE("sampler: temperature=0 short-circuits to argmax after penalties") {
    auto l = logits_of({1.0f, 5.0f, 2.0f, 3.0f});
    SamplerParams p = SamplerParams::identity();
    p.temperature = 0.0f;
    SamplerContext ctx(/*seed=*/42);
    CHECK(sample(l, p, ctx) == 1);
}

TEST_CASE("sampler: identity params + high-temp draw doesn't crash") {
    auto l = logits_of({1.0f, 2.0f, 3.0f, 4.0f});
    SamplerParams p = SamplerParams::identity();
    p.temperature = 1.0f;
    SamplerContext ctx(/*seed=*/7);
    const TokenId t = sample(l, p, ctx);
    CHECK(t >= 0);
    CHECK(t < 4);
}

TEST_CASE("sampler: same seed reproduces the same token") {
    // Build two identical contexts + identical logits + identical params.
    // Result must match — Decision 08 §8.3 reproducibility invariant.
    SamplerParams p = SamplerParams::identity();
    p.temperature = 1.0f;

    auto l1 = logits_of({0.5f, 0.7f, 0.3f, 0.9f, 0.6f, 0.4f, 0.8f, 0.2f});
    auto l2 = l1;
    SamplerContext c1(/*seed=*/123456ull);
    SamplerContext c2(/*seed=*/123456ull);
    CHECK(sample(l1, p, c1) == sample(l2, p, c2));
}

TEST_CASE("sampler: top_k=1 collapses to argmax") {
    auto l = logits_of({0.1f, 0.4f, 0.3f, 0.9f, 0.2f});
    SamplerParams p = SamplerParams::identity();
    p.top_k       = 1;
    p.temperature = 1.0f;
    SamplerContext ctx(/*seed=*/1);
    CHECK(sample(l, p, ctx) == 3);
}

TEST_CASE("sampler: repetition penalty suppresses recent tokens") {
    // Token 2 is the argmax, but it also just appeared in the recent window
    // and rep_penalty > 1 drops its logit below token 1 (the runner-up).
    auto l = logits_of({0.1f, 2.0f, 2.5f});
    SamplerParams p = SamplerParams::identity();
    p.temperature   = 0.0f;
    p.rep_penalty   = 2.0f;
    p.repeat_last_n = 8;

    SamplerContext ctx(/*seed=*/1);
    ctx.push_recent(2);
    CHECK(sample(l, p, ctx) == 1);
}

TEST_CASE("sampler: min_p filters out very-low-probability tokens") {
    // One token dominates; a min_p above the tail forces argmax.
    auto l = logits_of({10.0f, -5.0f, -6.0f, -7.0f, -8.0f});
    SamplerParams p = SamplerParams::identity();
    p.temperature = 1.0f;
    p.min_p       = 0.5f;
    SamplerContext ctx(/*seed=*/42);
    CHECK(sample(l, p, ctx) == 0);
}

TEST_CASE("sampler: top_p=0.5 keeps only the dominant head of the distribution") {
    auto l = logits_of({10.0f, 9.0f, 0.0f, 0.0f, 0.0f});
    SamplerParams p = SamplerParams::identity();
    p.temperature = 1.0f;
    p.top_p       = 0.5f;
    SamplerContext ctx(/*seed=*/5);
    // With top_p=0.5, only tokens 0 and 1 (each ~0.5 prob after softmax at
    // temp=1) survive. Verify the drawn token is one of them.
    const TokenId t = sample(l, p, ctx);
    CHECK((t == 0 || t == 1));
}

TEST_CASE("sampler: logit_bias pushes a specific token to the top") {
    auto l = logits_of({0.1f, 0.2f, 0.3f});
    SamplerParams p = SamplerParams::identity();
    p.temperature = 0.0f;
    p.logit_bias[0] = 100.0f;
    SamplerContext ctx(/*seed=*/1);
    CHECK(sample(l, p, ctx) == 0);
}

TEST_CASE("sampler: default params (coding-tuned) return a valid token") {
    auto l = logits_of({0.1f, 0.5f, 0.9f, 0.3f, 0.7f, 0.2f, 0.4f, 0.6f, 0.8f, 0.15f});
    SamplerParams p;   // defaults from Decision 08 §8.2
    SamplerContext ctx(/*seed=*/99);
    const TokenId t = sample(l, p, ctx);
    CHECK(t >= 0);
    CHECK(t < 10);
}
