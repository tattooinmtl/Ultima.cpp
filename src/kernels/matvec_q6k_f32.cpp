#include "ultima/kernels/dequant_q6k.hpp"
#include "ultima/kernels/matvec.hpp"

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace ultima::kernels {

namespace {

inline float hsum256(__m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

} // namespace

void matvec_q6k_f32_scalar(const std::uint8_t* w, const float* x, float* y,
                           std::size_t M, std::size_t K) noexcept {
    if (K == 0 || (K % 256u) != 0) {
        for (std::size_t m = 0; m < M; ++m) y[m] = 0.0f;
        return;
    }
    const std::size_t blocks_per_row = K / 256u;
    const std::size_t bytes_per_row  = blocks_per_row * 210u;

    float scratch[256];
    for (std::size_t m = 0; m < M; ++m) {
        const std::uint8_t* row = w + m * bytes_per_row;
        float acc = 0.0f;
        for (std::size_t b = 0; b < blocks_per_row; ++b) {
            dequant_q6k_block(row + b * 210u, scratch);
            const float* xslice = x + b * 256u;
            for (std::size_t i = 0; i < 256; ++i) acc += scratch[i] * xslice[i];
        }
        y[m] = acc;
    }
}

// AVX2: per super-block, dequant into a 256-elt aligned scratch (the ggml
// packing stripes across the block in a way that isn't friendly to inline
// per-sub-block SIMD unpacking; a materialize-then-FMA sweep is the simple
// correct path). Then 32 AVX2 FMAs against the x slice.
void matvec_q6k_f32_avx2(const std::uint8_t* w, const float* x, float* y,
                         std::size_t M, std::size_t K) noexcept {
    if (K == 0 || (K % 256u) != 0) {
        for (std::size_t m = 0; m < M; ++m) y[m] = 0.0f;
        return;
    }
    const std::size_t blocks_per_row = K / 256u;
    const std::size_t bytes_per_row  = blocks_per_row * 210u;

    alignas(32) float scratch[256];

    for (std::size_t m = 0; m < M; ++m) {
        const std::uint8_t* row = w + m * bytes_per_row;
        __m256 row_acc = _mm256_setzero_ps();
        for (std::size_t b = 0; b < blocks_per_row; ++b) {
            dequant_q6k_block(row + b * 210u, scratch);
            const float* xslice = x + b * 256u;
            for (int c = 0; c < 32; ++c) {
                __m256 nv = _mm256_load_ps (scratch + c * 8);
                __m256 xv = _mm256_loadu_ps(xslice  + c * 8);
                row_acc = _mm256_fmadd_ps(nv, xv, row_acc);
            }
        }
        y[m] = hsum256(row_acc);
    }
}

void matvec_q6k_f32(const std::uint8_t* w, const float* x, float* y,
                    std::size_t M, std::size_t K) noexcept {
    matvec_q6k_f32_avx2(w, x, y, M, K);
}

} // namespace ultima::kernels
