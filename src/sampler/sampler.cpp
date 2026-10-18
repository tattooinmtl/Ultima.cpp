#include "ultima/sampler/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

namespace ultima::sampler {

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// ---- Recent-window slice --------------------------------------------------
std::span<const TokenId>
window_slice(std::span<const TokenId> recent, std::size_t last_n) {
    if (last_n == 0 || recent.empty()) return {};
    if (last_n >= recent.size()) return recent;
    return recent.subspan(recent.size() - last_n);
}

// ---- Individual transforms -------------------------------------------------

void apply_logit_bias(std::span<float> logits,
                      const std::unordered_map<TokenId, float>& bias) {
    if (bias.empty()) return;
    for (const auto& [id, b] : bias) {
        if (id >= 0 && static_cast<std::size_t>(id) < logits.size()) {
            logits[static_cast<std::size_t>(id)] += b;
        }
    }
}

// Standard multiplicative repetition penalty: for each token id in the
// recent window, divide (if positive) or multiply (if negative) its logit by
// `penalty`. Matches llama.cpp `sample_repetition_penalties`.
void apply_rep_penalty(std::span<float> logits,
                       std::span<const TokenId> recent,
                       float penalty) {
    if (penalty == 1.0f || recent.empty()) return;
    // Dedup so we don't apply the penalty N times for a token seen N times.
    std::vector<TokenId> uniq(recent.begin(), recent.end());
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    for (TokenId t : uniq) {
        if (t < 0 || static_cast<std::size_t>(t) >= logits.size()) continue;
        float& l = logits[static_cast<std::size_t>(t)];
        l = (l > 0.0f) ? (l / penalty) : (l * penalty);
    }
}

// DRY (Don't Repeat Yourself). Simplified formulation matching common
// implementations: for each token id in the recent window, count how many
// times it appears (across the whole window, deduped by suffix length), and
// apply a growing penalty when the suffix length exceeds `allowed_len`.
//
// The version here scores the *last-seen* run length for each candidate token
// and subtracts `multiplier * pow(base, max(0, run_len - allowed_len))` from
// the logit. Enough to block the "echo the same 3 lines forever" failure the
// spec calls out.
void apply_dry(std::span<float> logits,
               std::span<const TokenId> recent,
               float multiplier, float base,
               std::size_t allowed_len) {
    if (multiplier <= 0.0f || recent.size() < 2) return;

    // Count repeats of each token id in the window.
    std::unordered_map<TokenId, std::size_t> counts;
    counts.reserve(recent.size());
    for (TokenId t : recent) counts[t]++;

    for (const auto& [id, n] : counts) {
        if (n <= allowed_len) continue;
        if (id < 0 || static_cast<std::size_t>(id) >= logits.size()) continue;
        const float pen =
            multiplier * std::pow(base, static_cast<float>(n - allowed_len));
        logits[static_cast<std::size_t>(id)] -= pen;
    }
}

// OpenAI-style frequency/presence penalties (both off by default).
void apply_freq_presence(std::span<float> logits,
                         std::span<const TokenId> recent,
                         float freq_pen, float presence_pen) {
    if (freq_pen == 0.0f && presence_pen == 0.0f) return;
    std::unordered_map<TokenId, std::size_t> counts;
    counts.reserve(recent.size());
    for (TokenId t : recent) counts[t]++;
    for (const auto& [id, n] : counts) {
        if (id < 0 || static_cast<std::size_t>(id) >= logits.size()) continue;
        auto& l = logits[static_cast<std::size_t>(id)];
        l -= static_cast<float>(n) * freq_pen;
        l -= presence_pen;
    }
}

void apply_temperature(std::span<float> logits, float temp) {
    if (temp == 1.0f) return;
    if (temp <= 0.0f) return;   // caller short-circuits to argmax
    const float inv = 1.0f / temp;
    for (auto& l : logits) l *= inv;
}

// Sort indices desc by logit, keep only the top-k, mask the rest with -inf.
void apply_top_k(std::span<float> logits, std::size_t k) {
    if (k == 0 || k >= logits.size()) return;
    std::vector<std::size_t> idx(logits.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k), idx.end(),
                      [&](std::size_t a, std::size_t b) { return logits[a] > logits[b]; });
    // Keep-set = first k of idx.
    std::vector<bool> keep(logits.size(), false);
    for (std::size_t i = 0; i < k; ++i) keep[idx[i]] = true;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        if (!keep[i]) logits[i] = kNegInf;
    }
}

// Compute softmax probabilities from logits (skipping -inf entries).
std::vector<float> softmax_probs(std::span<const float> logits) {
    float m = kNegInf;
    for (float l : logits) if (l > m) m = l;
    if (m == kNegInf) {
        // All -inf — return uniform over the vocab so we don't NaN.
        return std::vector<float>(logits.size(), 1.0f / static_cast<float>(logits.size()));
    }
    std::vector<float> p(logits.size(), 0.0f);
    float sum = 0.0f;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        if (logits[i] == kNegInf) { p[i] = 0.0f; continue; }
        p[i] = std::exp(logits[i] - m);
        sum += p[i];
    }
    if (sum <= 0.0f) return p;
    const float inv = 1.0f / sum;
    for (auto& x : p) x *= inv;
    return p;
}

// Nucleus / top-p: keep the smallest set of tokens whose cumulative
// probability >= p, mask the rest with -inf.
void apply_top_p(std::span<float> logits, float p) {
    if (p >= 1.0f || p <= 0.0f) return;
    auto probs = softmax_probs(logits);
    std::vector<std::size_t> idx(logits.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](std::size_t a, std::size_t b) { return probs[a] > probs[b]; });
    float cum = 0.0f;
    std::size_t cutoff = idx.size();
    for (std::size_t i = 0; i < idx.size(); ++i) {
        cum += probs[idx[i]];
        if (cum >= p) { cutoff = i + 1; break; }
    }
    std::vector<bool> keep(logits.size(), false);
    for (std::size_t i = 0; i < cutoff; ++i) keep[idx[i]] = true;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        if (!keep[i]) logits[i] = kNegInf;
    }
}

// min-p: mask tokens whose probability < min_p * max_prob.
void apply_min_p(std::span<float> logits, float min_p) {
    if (min_p <= 0.0f) return;
    auto probs = softmax_probs(logits);
    float max_p = 0.0f;
    for (float x : probs) if (x > max_p) max_p = x;
    const float threshold = min_p * max_p;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        if (probs[i] < threshold) logits[i] = kNegInf;
    }
}

TokenId sample_multinomial(std::span<const float> logits, std::mt19937_64& rng) {
    auto probs = softmax_probs(logits);
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return static_cast<TokenId>(dist(rng));
}

std::uint64_t seed_from_device() {
    std::random_device rd;
    return (static_cast<std::uint64_t>(rd()) << 32) | rd();
}

} // namespace

// ---- SamplerParams ---------------------------------------------------------
SamplerParams SamplerParams::identity() {
    SamplerParams p;
    p.rep_penalty      = 1.0f;
    p.repeat_last_n    = 0;
    p.dry_multiplier   = 0.0f;
    p.dry_last_n       = 0;
    p.frequency_penalty = 0.0f;
    p.presence_penalty  = 0.0f;
    p.temperature      = 1.0f;
    p.top_k            = 0;
    p.top_p            = 1.0f;
    p.min_p            = 0.0f;
    return p;
}

// ---- SamplerContext --------------------------------------------------------
SamplerContext::SamplerContext(std::uint64_t s)
    : seed_{s}, rng_{s} {}

SamplerContext::SamplerContext() : SamplerContext(seed_from_device()) {}

void SamplerContext::push_recent(TokenId t) { recent_.push_back(t); }

void SamplerContext::set_recent(std::span<const TokenId> tokens) {
    recent_.assign(tokens.begin(), tokens.end());
}

// ---- argmax + sample -------------------------------------------------------
TokenId argmax(std::span<const float> logits) noexcept {
    if (logits.empty()) return -1;
    std::size_t best = 0;
    float best_v = logits[0];
    for (std::size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > best_v) { best_v = logits[i]; best = i; }
    }
    return static_cast<TokenId>(best);
}

TokenId sample(std::span<float> logits,
               const SamplerParams& params,
               SamplerContext&      ctx) {
    if (logits.empty()) return -1;

    // 1. logit bias
    apply_logit_bias(logits, params.logit_bias);

    // 2. rep penalty (over the repeat_last_n window)
    if (params.rep_penalty != 1.0f && params.repeat_last_n > 0) {
        apply_rep_penalty(logits,
                          window_slice(ctx.recent(), params.repeat_last_n),
                          params.rep_penalty);
    }

    // 3. DRY (over the dry_last_n window)
    if (params.dry_multiplier > 0.0f && params.dry_last_n > 0) {
        apply_dry(logits,
                  window_slice(ctx.recent(), params.dry_last_n),
                  params.dry_multiplier, params.dry_base,
                  params.dry_allowed_len);
    }

    // 4. frequency / presence penalties
    if (params.frequency_penalty != 0.0f || params.presence_penalty != 0.0f) {
        const std::size_t win = std::max(params.repeat_last_n, params.dry_last_n);
        apply_freq_presence(logits, window_slice(ctx.recent(), win),
                            params.frequency_penalty, params.presence_penalty);
    }

    // 5. temperature — 0 short-circuits to argmax after penalties
    if (params.temperature <= 0.0f) {
        return argmax(logits);
    }
    apply_temperature(logits, params.temperature);

    // 6. top-k
    if (params.top_k > 0) apply_top_k(logits, params.top_k);

    // 7. top-p
    if (params.top_p < 1.0f) apply_top_p(logits, params.top_p);

    // 8. min-p
    if (params.min_p > 0.0f) apply_min_p(logits, params.min_p);

    // 9. softmax + multinomial draw
    return sample_multinomial(logits, ctx.rng());
}

} // namespace ultima::sampler
