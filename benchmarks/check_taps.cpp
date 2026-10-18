// Verifies LayerTaps against a real model.
//
// The check that matters is positional: a tap is supposed to be the
// residual stream *entering* a block, so the tap at layer 0 must equal the
// token embedding exactly, and the tap at layer L for token t must equal
// what layer L-1 produced. Plumbing that silently captured the layer
// *output*, or slid by one token, would still fill the buffer with
// plausible numbers.

#include "ultima/model/i_model_loader.hpp"
#include "ultima/model/imodel.hpp"
#include "ultima/model/weight_dispatch.hpp"
#include "ultima/kv_cache/kv_cache.hpp"
#include "ultima/runtime/thread_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: check_taps <model.gguf>\n");
        return 2;
    }
    auto loader = ultima::model::make_gguf_loader();
    auto gguf_r = loader->load(argv[1]);
    if (!gguf_r) {
        std::fprintf(stderr, "load failed: %s\n", gguf_r.error().message.c_str());
        return 1;
    }
    const auto& gguf = **gguf_r;

    ultima::runtime::ThreadPool pool{0};
    auto model_r = ultima::model::load_model(gguf, pool);
    if (!model_r) {
        std::fprintf(stderr, "model failed: %s\n", model_r.error().message.c_str());
        return 1;
    }
    auto& model = **model_r;
    const auto& d = model.dims();

    if (!model.supports_layer_taps()) {
        std::fprintf(stderr, "this model does not implement layer taps\n");
        return 1;
    }

    ultima::model::LayerTaps taps;
    taps.layers = {0, 1, d.n_layers / 2};
    const std::size_t stride = taps.layers.size() * d.hidden;

    const std::vector<std::int32_t> tokens{9707, 1879, 11, 1246, 525, 498};
    std::vector<float> buf(tokens.size() * stride, -12345.0f);
    taps.out = buf.data();
    model.set_layer_taps(&taps);

    ultima::kv_cache::KVCache kv{{
        /*n_slots  =*/1,
        /*n_ctx    =*/256,
        /*n_layers =*/d.n_layers,
        /*n_kv_heads=*/d.n_kv_heads,
        /*head_dim =*/d.head_dim,
        /*dtype    =*/ultima::model::DataType::F32,
    }};
    auto slot = kv.acquire(1);

    std::vector<float> logits(d.vocab, 0.0f);
    if (auto r = model.prefill(kv, slot, tokens, logits.data()); !r) {
        std::fprintf(stderr, "prefill failed: %s\n", r.error().message.c_str());
        return 1;
    }
    model.set_layer_taps(nullptr);

    // Every slot must have been written.
    std::size_t untouched = 0;
    for (float v : buf) untouched += (v == -12345.0f);
    std::printf("tap buffer: %zu floats, %zu left unwritten\n", buf.size(), untouched);

    // The decisive positional check: block 0's input is the embedding, so
    // tap 0 must equal a direct row read. If the capture were taking the
    // layer *output*, or were off by a token, this fails.
    auto embd = gguf.tensor("token_embd.weight");
    std::vector<float> emb(d.hidden, 0.0f);
    int bad = 0;
    double worst = 0.0;
    for (std::size_t t = 0; t < tokens.size(); ++t) {
        if (embd) {
            (void)ultima::model::read_row(embd->info().dtype, embd->data(),
                                          static_cast<std::size_t>(tokens[t]),
                                          d.hidden, emb.data());
        }
        const float* tap0 = buf.data() + t * stride;
        double diff = 0.0, mag = 0.0;
        for (std::size_t j = 0; j < d.hidden; ++j) {
            diff = std::max(diff, std::abs(double(tap0[j]) - double(emb[j])));
            mag  = std::max(mag, std::abs(double(emb[j])));
        }
        worst = std::max(worst, diff);
        if (diff > 1e-5 * std::max(mag, 1.0)) ++bad;
    }
    std::printf("layer-0 tap vs embedding row: worst abs diff %.3g\n", worst);
    std::printf("tokens matching the embedding exactly: %zu/%zu\n",
                tokens.size() - static_cast<std::size_t>(bad), tokens.size());

    // Distinct layers must not be copies of each other, and distinct tokens
    // must differ — either would mean an indexing bug.
    for (std::size_t i = 0; i + 1 < taps.layers.size(); ++i) {
        double diff = 0.0;
        for (std::size_t j = 0; j < d.hidden; ++j) {
            const double a = buf[0 * stride + i * d.hidden + j];
            const double b = buf[0 * stride + (i + 1) * d.hidden + j];
            diff += std::abs(a - b);
        }
        std::printf("  tap %zu (layer %zu) vs tap %zu (layer %zu): L1 diff %.3f\n",
                    i, taps.layers[i], i + 1, taps.layers[i + 1], diff);
    }
    double tok_diff = 0.0;
    for (std::size_t j = 0; j < d.hidden; ++j) {
        tok_diff += std::abs(double(buf[0 * stride + j]) - double(buf[1 * stride + j]));
    }
    std::printf("  token 0 vs token 1 at layer 0: L1 diff %.3f\n", tok_diff);
    return (untouched == 0 && bad == 0) ? 0 : 1;
}
