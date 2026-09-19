#include "ultima/kernels/matvec.hpp"

#include <cstring>

#if defined(__AVX2__) || defined(_MSC_VER)
#include <immintrin.h>
#define ULTIMA_BF16_HAVE_AVX2 1
#endif

namespace ultima::kernels {

// BF16 is the top 16 bits of an f32, so widening is a shift — no exponent
// rebias, no table. That makes this the cheapest dequantizing matvec we
// have, and it needs no block structure: K is unconstrained.
//
// The Ternary-Bonsai files store their small delta-net projections
// (ssm_alpha, ssm_beta) this way while the large matrices are PQ2_0.

void matvec_bf16_f32_scalar(const std::uint8_t* w, const float* x, float* y,
                            std::size_t M, std::size_t K) noexcept {
    for (std::size_t m = 0; m < M; ++m) {
        const std::uint8_t* row = w + m * K * 2;
        float acc = 0.0f;
        for (std::size_t k = 0; k < K; ++k) {
            const std::uint32_t bits =
                (static_cast<std::uint32_t>(row[2 * k])
                 | (static_cast<std::uint32_t>(row[2 * k + 1]) << 8)) << 16;
            float f;
            std::memcpy(&f, &bits, sizeof(f));
            acc += f * x[k];
        }
        y[m] = acc;
    }
}

#ifdef ULTIMA_BF16_HAVE_AVX2
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

void matvec_bf16_f32_avx2(const std::uint8_t* w, const float* x, float* y,
                          std::size_t M, std::size_t K) noexcept {
    const std::size_t k8 = K & ~std::size_t{7};
    for (std::size_t m = 0; m < M; ++m) {
        const std::uint8_t* row = w + m * K * 2;
        __m256 acc = _mm256_setzero_ps();
        for (std::size_t k = 0; k < k8; k += 8) {
            // Eight bf16 -> eight f32: widen to 32 bits and shift into the
            // high half, which is the whole conversion.
            const __m128i h = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(row + 2 * k));
            const __m256i w32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16);
            acc = _mm256_fmadd_ps(_mm256_castsi256_ps(w32),
                                  _mm256_loadu_ps(x + k), acc);
        }
        float tail = 0.0f;
        for (std::size_t k = k8; k < K; ++k) {
            const std::uint32_t bits =
                (static_cast<std::uint32_t>(row[2 * k])
                 | (static_cast<std::uint32_t>(row[2 * k + 1]) << 8)) << 16;
            float f;
            std::memcpy(&f, &bits, sizeof(f));
            tail += f * x[k];
        }
        y[m] = hsum256(acc) + tail;
    }
}
#endif

void matvec_bf16_f32(const std::uint8_t* w, const float* x, float* y,
                     std::size_t M, std::size_t K) noexcept {
#ifdef ULTIMA_BF16_HAVE_AVX2
    matvec_bf16_f32_avx2(w, x, y, M, K);
#else
    matvec_bf16_f32_scalar(w, x, y, M, K);
#endif
}

} // namespace ultima::kernels
