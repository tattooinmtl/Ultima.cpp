#include "ultima/kernels/dequant_q4k.hpp"
#include "ultima/kernels/matvec.hpp"

#include <cstddef>
#include <cstdint>

namespace ultima::kernels {

// Fused Q4_K -> F32 matvec.
//
// Layout: W is [M, K] row-major in Q4_K format. Each row uses K/256 super-
// blocks, each 144 bytes, contiguous. Row stride in bytes = (K/256) * 144.
//
// Strategy: for each output row m, iterate super-blocks along K. Dequant one
// super-block (256 floats) into a small stack scratch, then dot it against
// the corresponding x[k..k+256] slice. Accumulate into acc. Write acc to y[m].
//
// This is intentionally the simplest correct fused path. AVX2 in the inner
// dot is a later optimization; the win from avoiding an M*K F32 scratch
// buffer is preserved regardless.
void matvec_q4k_f32_scalar(const std::uint8_t* w, const float* x, float* y,
                           std::size_t M, std::size_t K) noexcept {
    if (K == 0 || (K % 256u) != 0) {
        for (std::size_t m = 0; m < M; ++m) y[m] = 0.0f;
        return;
    }
    const std::size_t blocks_per_row = K / 256u;
    const std::size_t bytes_per_row  = blocks_per_row * 144u;

    float scratch[256];
    for (std::size_t m = 0; m < M; ++m) {
        const std::uint8_t* row = w + m * bytes_per_row;
        float acc = 0.0f;
        for (std::size_t b = 0; b < blocks_per_row; ++b) {
            dequant_q4k_block(row + b * 144u, scratch);
            const float* xslice = x + b * 256u;
            for (std::size_t i = 0; i < 256; ++i) {
                acc += scratch[i] * xslice[i];
            }
        }
        y[m] = acc;
    }
}

void matvec_q4k_f32(const std::uint8_t* w, const float* x, float* y,
                    std::size_t M, std::size_t K) noexcept {
    matvec_q4k_f32_scalar(w, x, y, M, K);
}

} // namespace ultima::kernels
