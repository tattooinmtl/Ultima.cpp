#include "ultima/kernels/dequant_q8_0.hpp"
#include "ultima/kernels/matvec.hpp"

#include <cstddef>
#include <cstdint>

namespace ultima::kernels {

void matvec_q8_0_f32_scalar(const std::uint8_t* w, const float* x, float* y,
                            std::size_t M, std::size_t K) noexcept {
    if (K == 0 || (K % 32u) != 0) {
        for (std::size_t m = 0; m < M; ++m) y[m] = 0.0f;
        return;
    }
    const std::size_t blocks_per_row = K / 32u;
    const std::size_t bytes_per_row  = blocks_per_row * 34u;

    float scratch[32];
    for (std::size_t m = 0; m < M; ++m) {
        const std::uint8_t* row = w + m * bytes_per_row;
        float acc = 0.0f;
        for (std::size_t b = 0; b < blocks_per_row; ++b) {
            dequant_q8_0_block(row + b * 34u, scratch);
            const float* xslice = x + b * 32u;
            for (std::size_t i = 0; i < 32; ++i) acc += scratch[i] * xslice[i];
        }
        y[m] = acc;
    }
}

void matvec_q8_0_f32(const std::uint8_t* w, const float* x, float* y,
                     std::size_t M, std::size_t K) noexcept {
    matvec_q8_0_f32_scalar(w, x, y, M, K);
}

} // namespace ultima::kernels
