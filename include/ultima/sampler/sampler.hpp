#pragma once

#include <cstdint>
#include <optional>
#include <random>
#include <span>
#include <unordered_map>
#include <vector>

namespace ultima::sampler {

using TokenId = std::int32_t;

// Coding-tuned defaults per Decision 08 §8.2. Every knob is at its identity
// (no-op) when not specified so a caller only overrides what it needs.
struct SamplerParams {
    // Order matters — see Decision 08 §8.1 for the transform ladder.
    // 1. logit_bias
    std::unordered_map<TokenId, float> logit_bias;

    // 2. repetition penalty (applies to tokens in the recent-token window)
    float       rep_penalty      = 1.1f;
    std::size_t repeat_last_n    = 64;

    // 3. DRY (block the "keep echoing the previous line" failure)
    float       dry_multiplier   = 0.8f;
    float       dry_base         = 1.75f;
    std::size_t dry_allowed_len  = 2;
    std::size_t dry_last_n       = 512;   // DRY scans a longer window

    // 4. frequency / presence penalties (OpenAI-compatible; off by default)
    float       frequency_penalty = 0.0f;
    float       presence_penalty  = 0.0f;

    // 5. temperature — 0 short-circuits to argmax after penalties
    float       temperature      = 0.2f;

    // 6. top-k
    std::size_t top_k            = 40;

    // 7. top-p (nucleus)
    float       top_p            = 0.95f;

    // 8. min-p
    float       min_p            = 0.05f;

    // Optional RNG seed. If unset, the sampler seeds from std::random_device.
    std::optional<std::uint64_t> seed;

    // Reset every knob to its identity — chain becomes a pure argmax.
    static SamplerParams identity();
};

// Per-request state. Owns the RNG and the recent-token window used by
// penalties (which is normally supplied by the KVCache slot, but the sampler
// keeps a local ring buffer so it can run standalone in tests).
class SamplerContext {
public:
    explicit SamplerContext(std::uint64_t seed);
    SamplerContext();   // seeded from std::random_device

    // Append a token to the recent-token window used by penalty transforms.
    // Trimmed to the largest of {repeat_last_n, dry_last_n} across the calls
    // the caller intends to make; oversized appends are safely stored and
    // the transforms slice as needed.
    void push_recent(TokenId t);

    std::span<const TokenId> recent() const noexcept { return recent_; }
    void set_recent(std::span<const TokenId> tokens);

    std::uint64_t   seed() const noexcept { return seed_; }
    std::mt19937_64& rng() noexcept       { return rng_; }

private:
    std::uint64_t         seed_;
    std::mt19937_64       rng_;
    std::vector<TokenId>  recent_;
};

// Apply the full transform ladder to `logits` in place and return the sampled
// token id. `logits` must be a mutable buffer of at least `vocab_size` floats.
// The transform order and semantics are the ones locked in Decision 08.
TokenId sample(std::span<float> logits,
               const SamplerParams& params,
               SamplerContext&      ctx);

// Convenience for tests / greedy: pure argmax over `logits`, no side effects.
TokenId argmax(std::span<const float> logits) noexcept;

} // namespace ultima::sampler
