#include "ultima/kernels/dequant_q6k.hpp"
#include "ultima/kernels/matvec.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace ultima::kernels {

namespace {

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

inline float hsum256(__m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

// Dot 32 signed 6-bit values (one __m256i of int8, range [-32, 31]) against
// 32 floats, folding in the per-16 sub-scale: sA for values 0..15, sB for
// 16..31. acc_g += (q * s) * x[g*8 .. g*8+8].
inline void dot32_q6(__m256i q, const float* x, __m256 sA, __m256 sB,
                     __m256& a0, __m256& a1, __m256& a2, __m256& a3) noexcept {
    const __m128i b0 = _mm256_castsi256_si128(q);      // values  0..15
    const __m128i b1 = _mm256_extracti128_si256(q, 1); // values 16..31
    const __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b0));
    const __m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b0, 8)));
    const __m256 f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b1));
    const __m256 f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b1, 8)));
    a0 = _mm256_fmadd_ps(_mm256_mul_ps(f0, sA), _mm256_loadu_ps(x +  0), a0);
    a1 = _mm256_fmadd_ps(_mm256_mul_ps(f1, sA), _mm256_loadu_ps(x +  8), a1);
    a2 = _mm256_fmadd_ps(_mm256_mul_ps(f2, sB), _mm256_loadu_ps(x + 16), a2);
    a3 = _mm256_fmadd_ps(_mm256_mul_ps(f3, sB), _mm256_loadu_ps(x + 24), a3);
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

// AVX2: unpack one super-block half (128 values) into int8 SIMD vectors with
// shifts/masks, then convert + FMA per 8 lanes. No scalar scratch pass — the
// per-16 sub-scales are folded in as broadcast multiplies.
void matvec_q6k_f32_avx2(const std::uint8_t* w, const float* x, float* y,
                         std::size_t M, std::size_t K) noexcept {
    if (K == 0 || (K % 256u) != 0) {
        for (std::size_t m = 0; m < M; ++m) y[m] = 0.0f;
        return;
    }
    const std::size_t blocks_per_row = K / 256u;
    const std::size_t bytes_per_row  = blocks_per_row * 210u;

    const __m256i m4  = _mm256_set1_epi8(0x0F);
    const __m256i m3  = _mm256_set1_epi8(0x03);
    const __m256i m32 = _mm256_set1_epi8(32);

    for (std::size_t m = 0; m < M; ++m) {
        const std::uint8_t* row = w + m * bytes_per_row;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(),
               a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps(),
               a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps(),
               a6 = _mm256_setzero_ps(), a7 = _mm256_setzero_ps();

        for (std::size_t b = 0; b < blocks_per_row; ++b) {
            const std::uint8_t* blk = row + b * 210u;
            const std::uint8_t* ql_base = blk;
            const std::uint8_t* qh_base = blk + 128;
            const std::int8_t*  sc_base = reinterpret_cast<const std::int8_t*>(blk + 192);
            const float d = read_fp16_le(blk + 208);

            for (unsigned half = 0; half < 2; ++half) {
                const __m256i ql0 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(ql_base + half * 64u));
                const __m256i ql1 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(ql_base + half * 64u + 32u));
                const __m256i qhv = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(qh_base + half * 32u));
                const std::int8_t* sch = sc_base + half * 8u;
                const float* xh = x + b * 256u + half * 128u;

                const __m256i q1 = _mm256_sub_epi8(
                    _mm256_or_si256(_mm256_and_si256(ql0, m4),
                                    _mm256_slli_epi16(_mm256_and_si256(qhv, m3), 4)), m32);
                const __m256i q2 = _mm256_sub_epi8(
                    _mm256_or_si256(_mm256_and_si256(ql1, m4),
                                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 2), m3), 4)), m32);
                const __m256i q3 = _mm256_sub_epi8(
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql0, 4), m4),
                                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 4), m3), 4)), m32);
                const __m256i q4 = _mm256_sub_epi8(
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql1, 4), m4),
                                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 6), m3), 4)), m32);

                const __m256 s0 = _mm256_set1_ps(d * static_cast<float>(sch[0]));
                const __m256 s1 = _mm256_set1_ps(d * static_cast<float>(sch[1]));
                const __m256 s2 = _mm256_set1_ps(d * static_cast<float>(sch[2]));
                const __m256 s3 = _mm256_set1_ps(d * static_cast<float>(sch[3]));
                const __m256 s4 = _mm256_set1_ps(d * static_cast<float>(sch[4]));
                const __m256 s5 = _mm256_set1_ps(d * static_cast<float>(sch[5]));
                const __m256 s6 = _mm256_set1_ps(d * static_cast<float>(sch[6]));
                const __m256 s7 = _mm256_set1_ps(d * static_cast<float>(sch[7]));

                dot32_q6(q1, xh,      s0, s1, a0, a1, a2, a3);
                dot32_q6(q2, xh + 32, s2, s3, a4, a5, a6, a7);
                dot32_q6(q3, xh + 64, s4, s5, a0, a1, a2, a3);
                dot32_q6(q4, xh + 96, s6, s7, a4, a5, a6, a7);
            }
        }
        const __m256 s01 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        const __m256 s23 = _mm256_add_ps(_mm256_add_ps(a4, a5), _mm256_add_ps(a6, a7));
        y[m] = hsum256(_mm256_add_ps(s01, s23));
    }
}

void matvec_q6k_f32(const std::uint8_t* w, const float* x, float* y,
                    std::size_t M, std::size_t K) noexcept {
    matvec_q6k_f32_avx2(w, x, y, M, K);
}

} // namespace ultima::kernels
