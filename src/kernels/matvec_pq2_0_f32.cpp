#include "ultima/kernels/matvec.hpp"

#include "ultima/kernels/dequant_pq2_0.hpp"

#include <cstring>
#include <vector>

namespace ultima::kernels {

namespace {

constexpr std::size_t k_block = 128;
constexpr std::size_t k_bytes = 34;

float fp16_to_float(std::uint16_t h) noexcept {
    const std::uint32_t sign = (h & 0x8000u) << 16;
    std::uint32_t exp  = (h & 0x7C00u) >> 10;
    std::uint32_t mant = (h & 0x03FFu);
    std::uint32_t out_bits;
    if (exp == 0) {
        if (mant == 0) {
            out_bits = sign;
        } else {
            unsigned shift = 0;
            while ((mant & 0x0400u) == 0) { mant <<= 1; ++shift; }
            mant &= 0x03FFu;
            exp   = 127u - 14u - shift;
            out_bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        out_bits = sign | 0x7F800000u | (mant << 13);
    } else {
        exp = exp - 15u + 127u;
        out_bits = sign | (exp << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out_bits, sizeof(f));
    return f;
}

inline float read_fp16_le(const std::uint8_t* p) noexcept {
    const std::uint16_t h = static_cast<std::uint16_t>(p[0])
                          | (static_cast<std::uint16_t>(p[1]) << 8);
    return fp16_to_float(h);
}

} // namespace

// The activation is permuted once, which is what makes this vectorizable.
//
// PQ2_0 packs four codes per byte, so the weights in natural order sit at
// stride 4 once unpacked — the reason the first version of this kernel was
// scalar. But a dot product only requires weight[i] to meet x[i]; it does
// not care in what order the pairs are summed. So instead of gathering x to
// match the codes, x is reordered once for the whole matvec into the order
// the codes come out in, and every row then reads it contiguously.
//
// Two further savings fall out of the format:
//   sum_i (code_i - 1) * x_i  ==  sum_i code_i * x_i  -  sum_i x_i
// so the -1 never happens per element, and the per-block sum of x is also
// precomputed once. Each row then costs four shifts and sixteen FMAs per
// 128 weights, with one fp16 scale at the end.

void matvec_pq2_0_f32_scalar(const std::uint8_t* w, const float* x, float* y,
                             std::size_t M, std::size_t K) noexcept {
    if (K == 0 || (K % k_block) != 0) {
        for (std::size_t m = 0; m < M; ++m) y[m] = 0.0f;
        return;
    }
    const std::size_t blocks_per_row = K / k_block;
    const std::size_t bytes_per_row  = blocks_per_row * k_bytes;

    for (std::size_t m = 0; m < M; ++m) {
        const std::uint8_t* row = w + m * bytes_per_row;
        float acc = 0.0f;
        for (std::size_t b = 0; b < blocks_per_row; ++b) {
            const std::uint8_t* blk = row + b * k_bytes;
            const float         d   = read_fp16_le(blk);
            const std::uint8_t* qs  = blk + 2;
            const float*        xs  = x + b * k_block;

            float sub = 0.0f;
            for (std::size_t i = 0; i < k_block / 4; ++i) {
                const unsigned byte = qs[i];
                sub += (static_cast<int>( byte        & 0x03u) - 1) * xs[4 * i + 0];
                sub += (static_cast<int>((byte >> 2)  & 0x03u) - 1) * xs[4 * i + 1];
                sub += (static_cast<int>((byte >> 4)  & 0x03u) - 1) * xs[4 * i + 2];
                sub += (static_cast<int>((byte >> 6)  & 0x03u) - 1) * xs[4 * i + 3];
            }
            acc += d * sub;
        }
        y[m] = acc;
    }
}

#if defined(__AVX2__) || defined(_MSC_VER)
#include <immintrin.h>

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

// x_perm[b*128 + j*32 + i] = x[b*128 + i*4 + j], i.e. grouped by which
// bit-pair of its byte the weight occupies. Also returns each block's plain
// sum, which folds away the (code - 1) offset.
void pq2_0_prepare_activation(const float* x, std::size_t K,
                              float* x_perm, float* block_sum) noexcept {
    const std::size_t nb = K / k_block;
    for (std::size_t b = 0; b < nb; ++b) {
        const float* xs = x + b * k_block;
        float sum = 0.0f;
        for (std::size_t i = 0; i < k_block; ++i) sum += xs[i];
        block_sum[b] = sum;
        for (std::size_t j = 0; j < 4; ++j) {
            float* dst = x_perm + b * k_block + j * 32;
            for (std::size_t i = 0; i < 32; ++i) dst[i] = xs[4 * i + j];
        }
    }
}

void matvec_pq2_0_f32_avx2(const std::uint8_t* w, const float* x_perm,
                           const float* block_sum, float* y,
                           std::size_t M, std::size_t K) noexcept {
    const std::size_t nb = K / k_block;
    const std::size_t bytes_per_row = nb * k_bytes;
    const __m256i code_mask = _mm256_set1_epi8(0x03);

    for (std::size_t m = 0; m < M; ++m) {
        const std::uint8_t* row = w + m * bytes_per_row;
        float acc = 0.0f;

        for (std::size_t b = 0; b < nb; ++b) {
            const std::uint8_t* blk = row + b * k_bytes;
            const float         d   = read_fp16_le(blk);
            // 32 code bytes, four codes each.
            const __m256i codes = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(blk + 2));
            const float* xp = x_perm + b * k_block;

            __m256 a0 = _mm256_setzero_ps();
            for (int j = 0; j < 4; ++j) {
                // Shifting 16-bit lanes bleeds the high byte's low bits into
                // the low byte's top bits, but the mask discards exactly
                // those, so each byte yields its own j-th code pair.
                const __m256i cj = _mm256_and_si256(
                    _mm256_srli_epi16(codes, 2 * j), code_mask);
                const float* xj = xp + j * 32;

                // 32 codes -> four 8-wide f32 conversions.
                const __m128i lo = _mm256_castsi256_si128(cj);
                const __m128i hi = _mm256_extracti128_si256(cj, 1);
                a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo)),
                                     _mm256_loadu_ps(xj + 0), a0);
                a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
                                         _mm_srli_si128(lo, 8))),
                                     _mm256_loadu_ps(xj + 8), a0);
                a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi)),
                                     _mm256_loadu_ps(xj + 16), a0);
                a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
                                         _mm_srli_si128(hi, 8))),
                                     _mm256_loadu_ps(xj + 24), a0);
            }
            // sum (code-1)*x == sum code*x - sum x
            acc += d * (hsum256(a0) - block_sum[b]);
        }
        y[m] = acc;
    }
}
#endif

void matvec_pq2_0_f32(const std::uint8_t* w, const float* x, float* y,
                      std::size_t M, std::size_t K) noexcept {
    if (K == 0 || (K % k_block) != 0) {
        for (std::size_t m = 0; m < M; ++m) y[m] = 0.0f;
        return;
    }
#if defined(__AVX2__) || defined(_MSC_VER)
    // The permutation is paid once per call and reused by every row, so it
    // is worth it as soon as there is more than a handful of rows.
    if (M >= 8) {
        thread_local std::vector<float> perm;
        thread_local std::vector<float> sums;
        perm.resize(K);
        sums.resize(K / k_block);
        pq2_0_prepare_activation(x, K, perm.data(), sums.data());
        matvec_pq2_0_f32_avx2(w, perm.data(), sums.data(), y, M, K);
        return;
    }
#endif
    matvec_pq2_0_f32_scalar(w, x, y, M, K);
}

} // namespace ultima::kernels
