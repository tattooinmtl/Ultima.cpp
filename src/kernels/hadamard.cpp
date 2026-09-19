#include "ultima/kernels/hadamard.hpp"

#include <cmath>
#include <cstring>

namespace ultima::kernels {

void fwht_normalized_f32(float* x, std::size_t n) noexcept {
    if (n < 2) return;

    for (std::size_t len = 1; len < n; len <<= 1) {
        for (std::size_t i = 0; i < n; i += (len << 1)) {
            for (std::size_t j = i; j < i + len; ++j) {
                const float a = x[j];
                const float b = x[j + len];
                x[j]       = a + b;
                x[j + len] = a - b;
            }
        }
    }

    const float inv = 1.0f / std::sqrt(static_cast<float>(n));
    for (std::size_t i = 0; i < n; ++i) x[i] *= inv;
}

void hadamard_fold_f32(const float* x, float* y, std::size_t width,
                       std::size_t block, const float* signs) noexcept {
    if (block == 0 || width % block != 0) {
        if (y != x) std::memcpy(y, x, width * sizeof(float));
        return;
    }
    if (signs) {
        for (std::size_t i = 0; i < width; ++i) y[i] = x[i] * signs[i];
    } else if (y != x) {
        std::memcpy(y, x, width * sizeof(float));
    }
    for (std::size_t off = 0; off < width; off += block) {
        fwht_normalized_f32(y + off, block);
    }
}

void hadamard_unfold_f32(const float* x, float* y, std::size_t width,
                         std::size_t block, const float* signs) noexcept {
    if (block == 0 || width % block != 0) {
        if (y != x) std::memcpy(y, x, width * sizeof(float));
        return;
    }
    if (y != x) std::memcpy(y, x, width * sizeof(float));
    for (std::size_t off = 0; off < width; off += block) {
        fwht_normalized_f32(y + off, block);
    }
    if (signs) {
        for (std::size_t i = 0; i < width; ++i) y[i] *= signs[i];
    }
}

} // namespace ultima::kernels
