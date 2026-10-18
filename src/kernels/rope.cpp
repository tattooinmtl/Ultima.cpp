#include "ultima/kernels/rope.hpp"

#include <cmath>
#include <cstddef>

namespace ultima::kernels {

// theta_j = 1 / (freq_base ^ (2j / rope_dim)), optionally divided by
// freq_factors[j]; angle = position * theta_j.
//
// NeoX pairing:  (x[j], x[j + rope_dim/2])
// Norm pairing:  (x[2j], x[2j + 1])
//
// Both then apply the same rotation:
//   x_a' = x_a * c - x_b * s
//   x_b' = x_a * s + x_b * c
void rope_f32_ext(float* x,
                  std::size_t  n_heads,
                  std::size_t  head_dim,
                  std::size_t  rope_dim,
                  std::size_t  position,
                  float        freq_base,
                  bool         neox,
                  const float* freq_factors,
                  float        attn_factor) noexcept {
    if (rope_dim == 0 || (rope_dim % 2) != 0) return;
    const std::size_t half = rope_dim / 2;
    const float dim_f = static_cast<float>(rope_dim);
    const float pos_f = static_cast<float>(position);

    for (std::size_t h = 0; h < n_heads; ++h) {
        float* head = x + h * head_dim;
        for (std::size_t j = 0; j < half; ++j) {
            const float exponent = (2.0f * static_cast<float>(j)) / dim_f;
            float theta = 1.0f / std::pow(freq_base, exponent);
            if (freq_factors) theta /= freq_factors[j];

            const float angle = pos_f * theta;
            const float c = std::cos(angle) * attn_factor;
            const float s = std::sin(angle) * attn_factor;

            const std::size_t ia = neox ? j        : (2 * j);
            const std::size_t ib = neox ? (j+half) : (2 * j + 1);

            const float x0 = head[ia];
            const float x1 = head[ib];
            head[ia] = x0 * c - x1 * s;
            head[ib] = x0 * s + x1 * c;
        }
        // head_dim > rope_dim: trailing dims pass through unchanged.
    }
}

void rope_f32_scalar(float* x,
                     std::size_t n_heads,
                     std::size_t head_dim,
                     std::size_t rope_dim,
                     std::size_t position,
                     float       freq_base) noexcept {
    rope_f32_ext(x, n_heads, head_dim, rope_dim, position, freq_base,
                 /*neox=*/true, /*freq_factors=*/nullptr, /*attn_factor=*/1.0f);
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
