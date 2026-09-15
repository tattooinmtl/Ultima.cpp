#include "ultima/kernels/dequant_q5_0.hpp"
#include "ultima/kernels/matvec.hpp"

#include <cstddef>
#include <cstdint>

namespace ultima::kernels {

// Scalar-only fused Q5_0 -> F32 matvec. Dequant one 32-elt block into a
// tiny stack scratch, dot against the matching x slice, accumulate.
// AVX2 fast-path can follow the same pattern as matvec_q8_0_f32 (int8
// unpack -> f32 FMA) with the extra 5th-bit merge from qh; for now the
// scalar variant is enough to unblock the Qwen2.5-Coder-0.5B round-trip
// (that file uses Q5_0 for most projections).
void matvec_q5_0_f32(const std::uint8_t* w, const float* x, float* y,
                     std::size_t M, std::size_t K) noexcept {
    if (K == 0 || (K % 32u) != 0) {
        for (std::size_t m = 0; m < M; ++m) y[m] = 0.0f;
        return;
    }
    const std::size_t blocks_per_row = K / 32u;
    const std::size_t bytes_per_row  = blocks_per_row * 22u;

    float scratch[32];
    for (std::size_t m = 0; m < M; ++m) {
        const std::uint8_t* row = w + m * bytes_per_row;
        float acc = 0.0f;
        for (std::size_t b = 0; b < blocks_per_row; ++b) {
            dequant_q5_0_block(row + b * 22u, scratch);
            const float* xslice = x + b * 32u;
            for (std::size_t i = 0; i < 32; ++i) acc += scratch[i] * xslice[i];
        }
        y[m] = acc;
    }
}

} // namespace ultima::kernels
