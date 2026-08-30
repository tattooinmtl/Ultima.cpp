#include "ultima/kernels/rope.hpp"

#include <cmath>
#include <cstddef>

namespace ultima::kernels {

// Half-split rotation:
//   For j in [0, rope_dim/2):
//     theta_j        = 1 / (freq_base ^ (2j / rope_dim))
//     angle          = position * theta_j
//     c = cos(angle), s = sin(angle)
//     x0            = x[j]                       (first half)
//     x1            = x[j + rope_dim/2]          (second half)
//     x[j]                   = x0 * c - x1 * s
//     x[j + rope_dim/2]      = x0 * s + x1 * c
void rope_f32_scalar(float* x,
                     std::size_t n_heads,
                     std::size_t head_dim,
                     std::size_t rope_dim,
                     std::size_t position,
                     float       freq_base) noexcept {
    if (rope_dim == 0 || (rope_dim % 2) != 0) return;
    const std::size_t half = rope_dim / 2;
    const float dim_f = static_cast<float>(rope_dim);
    const float pos_f = static_cast<float>(position);

    for (std::size_t h = 0; h < n_heads; ++h) {
        float* head = x + h * head_dim;
        for (std::size_t j = 0; j < half; ++j) {
            const float exponent = (2.0f * static_cast<float>(j)) / dim_f;
            const float theta    = 1.0f / std::pow(freq_base, exponent);
            const float angle    = pos_f * theta;
            const float c = std::cos(angle);
            const float s = std::sin(angle);
            const float x0 = head[j];
            const float x1 = head[j + half];
            head[j]        = x0 * c - x1 * s;
            head[j + half] = x0 * s + x1 * c;
        }
        // head_dim > rope_dim: trailing dims pass through unchanged.
    }
}

void rope_f32(float* x,
              std::size_t n_heads,
              std::size_t head_dim,
              std::size_t rope_dim,
              std::size_t position,
              float       freq_base) noexcept {
    // The math is dominated by sin/cos calls; AVX2 without vector trig is not
    // a clear win. Precomputed cos/sin tables per layer (M4 wiring) will be
    // the real optimization. Ship scalar for correctness now.
    rope_f32_scalar(x, n_heads, head_dim, rope_dim, position, freq_base);
}

} // namespace ultima::kernels
