// Loads the dspark drafter and runs one draft block.
//
// The host is not loaded here — only its dimensions are needed to validate
// the pairing — so this exercises config parsing, tensor resolution and the
// whole forward pass cheaply. Tap features are synthetic, so the tokens it
// proposes are meaningless; what is being checked is that the shapes line
// up and the arithmetic stays finite.

#include "ultima/model/dspark.hpp"
#include "ultima/model/i_model_loader.hpp"
#include "ultima/runtime/thread_pool.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: check_dspark <drafter.gguf>\n");
        return 2;
    }
    auto loader = ultima::model::make_gguf_loader();
    auto gguf_r = loader->load(argv[1]);
    if (!gguf_r) {
        std::fprintf(stderr, "load failed: %s\n", gguf_r.error().message.c_str());
        return 1;
    }

    // Ternary-Bonsai-2-27B's shape, which is what this drafter is built for.
    ultima::model::ModelDims host{};
    host.n_layers = 64;
    host.hidden   = 5120;
    host.vocab    = 248320;

    ultima::runtime::ThreadPool pool{0};
    auto dr = ultima::model::DsparkDrafter::load(**gguf_r, host, pool);
    if (!dr) {
        std::fprintf(stderr, "drafter load failed: %s\n", dr.error().message.c_str());
        return 1;
    }
    auto& d = **dr;
    const auto& c = d.config();

    std::printf("dspark drafter loaded\n");
    std::printf("  n_embd %zu  layers %zu  heads %zu/%zu  head_dim %zu  ffn %zu\n",
                c.n_embd, c.n_layer, c.n_head, c.n_head_kv, c.head_dim, c.ffn_hidden);
    std::printf("  vocab %zu  block_size %zu  mask_token %zu  markov_rank %zu\n",
                c.vocab, c.block_size, c.mask_token_id, c.markov_rank);
    std::printf("  taps ");
    for (auto L : c.target_layers) std::printf("%zu ", L);
    std::printf(" -> fc input width %zu\n", c.tap_width());
    std::printf("  confidence_head %d (markov %d)   log_snr %d [%.1f, %.1f]\n",
                (int)c.confidence_head, (int)c.confidence_with_markov,
                (int)c.log_snr_conditioning, c.min_log_snr, c.max_log_snr);

    // One draft block off synthetic context.
    const std::size_t n_ctx = 8;
    std::mt19937 rng{5};
    std::normal_distribution<float> nd{0.0f, 0.3f};
    std::vector<float> ctx(n_ctx * c.tap_width());
    for (auto& v : ctx) v = nd(rng);

    std::vector<std::int32_t> draft(c.block_size,
                                    static_cast<std::int32_t>(c.mask_token_id));
    std::vector<std::int32_t> pos(n_ctx + c.block_size);
    for (std::size_t i = 0; i < pos.size(); ++i) pos[i] = static_cast<std::int32_t>(i);

    std::vector<float> logits(c.block_size * c.vocab, 0.0f);
    std::vector<float> conf(c.block_size, -1.0f);

    const auto t0 = std::chrono::steady_clock::now();
    if (auto r = d.draft(ctx, n_ctx, pos, draft, logits.data(), conf.data()); !r) {
        std::fprintf(stderr, "draft failed: %s\n", r.error().message.c_str());
        return 1;
    }
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    std::printf("draft of %zu tokens in %.1f ms\n", c.block_size, ms);
    int bad = 0;
    for (std::size_t t = 0; t < c.block_size; ++t) {
        const float* row = logits.data() + t * c.vocab;
        std::size_t arg = 0;
        float best = row[0];
        for (std::size_t v = 1; v < c.vocab; ++v) {
            if (!std::isfinite(row[v])) { ++bad; break; }
            if (row[v] > best) { best = row[v]; arg = v; }
        }
        double rowsum = 0.0;
        for (std::size_t v = 0; v < c.vocab; ++v) rowsum += row[v];
        std::printf("  pos %zu: argmax %-8zu logit %.6f  conf %.6f  rowsum %.4f\n",
                    t, arg, best, conf[t], rowsum);
    }
    if (bad) { std::fprintf(stderr, "non-finite logits\n"); return 1; }
    return 0;
}
