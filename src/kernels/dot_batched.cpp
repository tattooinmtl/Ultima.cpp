#include "ultima/kernels/dot_batched.hpp"

#if defined(__AVX2__) || defined(_MSC_VER)
#include <immintrin.h>
#define ULTIMA_DOT_HAVE_AVX2 1
#endif

namespace ultima::kernels {

void dot_rows_f32_scalar(const float* row, const float* X, std::size_t K,
                         std::size_t n_tok, float* out,
                         std::size_t out_stride) noexcept {
    for (std::size_t t = 0; t < n_tok; ++t) {
        const float* x = X + t * K;
        float acc = 0.0f;
        for (std::size_t k = 0; k < K; ++k) acc += row[k] * x[k];
        out[t * out_stride] = acc;
    }
}

#ifdef ULTIMA_DOT_HAVE_AVX2
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

// Four tokens at a time: one row load feeds four FMAs, so the row streams
// through the register file a quarter as often as it would token by token.
// Four is the useful width here — it keeps eight accumulators plus the row
// live in the sixteen available ymm registers without spilling.
void dot_rows_f32_avx2(const float* row, const float* X, std::size_t K,
                       std::size_t n_tok, float* out,
                       std::size_t out_stride) noexcept {
    const std::size_t k8 = K & ~std::size_t{7};
    std::size_t t = 0;

    for (; t + 4 <= n_tok; t += 4) {
        const float* x0 = X + (t + 0) * K;
        const float* x1 = X + (t + 1) * K;
        const float* x2 = X + (t + 2) * K;
        const float* x3 = X + (t + 3) * K;

        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();

        for (std::size_t k = 0; k < k8; k += 8) {
            const __m256 r = _mm256_loadu_ps(row + k);
            a0 = _mm256_fmadd_ps(r, _mm256_loadu_ps(x0 + k), a0);
            a1 = _mm256_fmadd_ps(r, _mm256_loadu_ps(x1 + k), a1);
            a2 = _mm256_fmadd_ps(r, _mm256_loadu_ps(x2 + k), a2);
            a3 = _mm256_fmadd_ps(r, _mm256_loadu_ps(x3 + k), a3);
        }
        float s0 = hsum256(a0), s1 = hsum256(a1);
        float s2 = hsum256(a2), s3 = hsum256(a3);
        for (std::size_t k = k8; k < K; ++k) {
            const float r = row[k];
            s0 += r * x0[k]; s1 += r * x1[k];
            s2 += r * x2[k]; s3 += r * x3[k];
        }
        out[(t + 0) * out_stride] = s0;
        out[(t + 1) * out_stride] = s1;
        out[(t + 2) * out_stride] = s2;
        out[(t + 3) * out_stride] = s3;
    }

    for (; t < n_tok; ++t) {
        const float* x = X + t * K;
        __m256 acc = _mm256_setzero_ps();
        for (std::size_t k = 0; k < k8; k += 8) {
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(row + k),
                                  _mm256_loadu_ps(x + k), acc);
        }
        float s = hsum256(acc);
        for (std::size_t k = k8; k < K; ++k) s += row[k] * x[k];
        out[t * out_stride] = s;
    }
}
#endif

void dot_rows_f32(const float* row, const float* X, std::size_t K,
                  std::size_t n_tok, float* out, std::size_t out_stride) noexcept {
#ifdef ULTIMA_DOT_HAVE_AVX2
    dot_rows_f32_avx2(row, X, K, n_tok, out, out_stride);
#else
    dot_rows_f32_scalar(row, X, K, n_tok, out, out_stride);
#endif
}

} // namespace ultima::kernels
