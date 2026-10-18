#include "ultima/kernels/delta_net.hpp"

#include <cmath>
#include <cstring>

namespace ultima::kernels {

void l2_norm_f32(const float* x, float* y, std::size_t n, float eps) noexcept {
    float ss = 0.0f;
    for (std::size_t i = 0; i < n; ++i) ss += x[i] * x[i];
    const float inv = 1.0f / std::sqrt(ss + eps);
    for (std::size_t i = 0; i < n; ++i) y[i] = x[i] * inv;
}

void softplus_f32(const float* x, float* y, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        const float v = x[i];
        // For large v, log1p(exp(v)) overflows in exp but equals v to well
        // within f32 precision; 20 is far past where the difference shows.
        y[i] = v > 20.0f ? v : std::log1p(std::exp(v));
    }
}

void causal_conv1d_step_f32(float* state, const float* x, const float* w,
                            float* y, std::size_t channels,
                            std::size_t width) noexcept {
    if (width == 0) return;
    const std::size_t hist = width - 1;

    for (std::size_t c = 0; c < channels; ++c) {
        float*       s  = state + c * hist;
        const float* wc = w + c * width;

        float acc = 0.0f;
        for (std::size_t k = 0; k < hist; ++k) acc += wc[k] * s[k];
        acc += wc[hist] * x[c];

        // Roll the history forward: drop the oldest, append this token. Done
        // before writing y so an aliasing y == x is still safe.
        for (std::size_t k = 0; k + 1 < hist; ++k) s[k] = s[k + 1];
        if (hist > 0) s[hist - 1] = x[c];

        y[c] = acc;
    }
}

void delta_net_step_f32(float* state, const float* q, const float* k,
                        const float* v, float decay, float beta,
                        float* out, float* scratch,
                        std::size_t d_state) noexcept {
    float* sk = scratch;

    // Decay the state and read S^T k off it in the same sweep, so the state
    // is touched once instead of twice. At d_state 128 that is a 64 KB
    // working set per head, and this runs for every head of every recurrent
    // layer of every token — the traffic is the whole cost of the layer.
    for (std::size_t j = 0; j < d_state; ++j) sk[j] = 0.0f;
    for (std::size_t i = 0; i < d_state; ++i) {
        float* row = state + i * d_state;
        const float ki = k[i];
        for (std::size_t j = 0; j < d_state; ++j) {
            const float s = row[j] * decay;
            row[j] = s;
            sk[j] += s * ki;
        }
    }

    // d[j] = beta * (v[j] - sk[j]); reuse sk to hold it.
    for (std::size_t j = 0; j < d_state; ++j) sk[j] = beta * (v[j] - sk[j]);

    // Rank-1 update, then read the output off the *updated* state.
    for (std::size_t j = 0; j < d_state; ++j) out[j] = 0.0f;
    for (std::size_t i = 0; i < d_state; ++i) {
        float* row = state + i * d_state;
        const float ki = k[i];
        const float qi = q[i];
        for (std::size_t j = 0; j < d_state; ++j) {
            const float s = row[j] + ki * sk[j];
            row[j] = s;
            out[j] += s * qi;
        }
    }
}

} // namespace ultima::kernels
