#include "ultima/kernels/dequant_q4k.hpp"
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
            exp   = 127u - 14u - shift + 1u;
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

inline void unpack_scale_min_6bit(const std::uint8_t* sm, unsigned s,
                                  unsigned& out_scale, unsigned& out_min) noexcept {
    if (s < 4) {
        out_scale = sm[s]     & 0x3Fu;
        out_min   = sm[s + 4] & 0x3Fu;
    } else {
        const unsigned idx = s - 4u;
        out_scale = (sm[8u + idx] & 0x0Fu) | ((sm[    idx] >> 6) << 4);
        out_min   = (sm[8u + idx] >>   4)  | ((sm[4u + idx] >> 6) << 4);
    }
}

inline float hsum256(__m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

} // namespace

// ----- Scalar oracle --------------------------------------------------------
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
            for (std::size_t i = 0; i < 256; ++i) acc += scratch[i] * xslice[i];
        }
        y[m] = acc;
    }
}

// ----- AVX2 + FMA -----------------------------------------------------------
// Algebraic reshape avoids materializing the full 256-elt f32 scratch:
//   value[k] = scale_s * nibble[k] - min_s        (per sub-block s)
//   sum_k value[k] * x[k] = scale_s * S_wx(s) - min_s * S_x(s)
// where S_wx(s) = sum over the sub-block of nibble * x_matching,
//       S_x(s)  = sum over the sub-block of x_matching.
// Both are computed by 4 AVX2 FMAs of 8 lanes each per sub-block.
// Nibble unpack stays scalar into a 32-elt on-stack buffer (cache-hot).
void matvec_q4k_f32_avx2(const std::uint8_t* w, const float* x, float* y,
                         std::size_t M, std::size_t K) noexcept {
    if (K == 0 || (K % 256u) != 0) {
        for (std::size_t m = 0; m < M; ++m) y[m] = 0.0f;
        return;
    }
    const std::size_t blocks_per_row = K / 256u;
    const std::size_t bytes_per_row  = blocks_per_row * 144u;

    alignas(32) float sub_nib[32];

    for (std::size_t m = 0; m < M; ++m) {
        const std::uint8_t* row = w + m * bytes_per_row;
        float acc = 0.0f;

        for (std::size_t b = 0; b < blocks_per_row; ++b) {
            const std::uint8_t* blk = row + b * 144u;
            const float d    = read_fp16_le(blk + 0);
            const float dmin = read_fp16_le(blk + 2);
            const std::uint8_t* sm = blk + 4;
            const std::uint8_t* qs = blk + 16;
            const float* xblk = x + b * 256u;

            for (unsigned s = 0; s < 8; ++s) {
                unsigned scale_q = 0, min_q = 0;
                unpack_scale_min_6bit(sm, s, scale_q, min_q);
                const float scale = d    * static_cast<float>(scale_q);
                const float min_v = dmin * static_cast<float>(min_q);

                const std::uint8_t* sub_qs = qs + s * 16u;
                for (unsigned i = 0; i < 16; ++i) {
                    sub_nib[2u * i    ] = static_cast<float>(sub_qs[i] & 0x0Fu);
                    sub_nib[2u * i + 1] = static_cast<float>((sub_qs[i] >> 4) & 0x0Fu);
                }

                const float* xsub = xblk + s * 32u;
                __m256 nib_acc = _mm256_setzero_ps();
                __m256 x_acc   = _mm256_setzero_ps();
                for (int c = 0; c < 4; ++c) {
                    __m256 nv = _mm256_load_ps (sub_nib + c * 8);
                    __m256 xv = _mm256_loadu_ps(xsub    + c * 8);
                    nib_acc = _mm256_fmadd_ps(nv, xv, nib_acc);
                    x_acc   = _mm256_add_ps (x_acc, xv);
                }
                const float S_wx = hsum256(nib_acc);
                const float S_x  = hsum256(x_acc);
                acc += scale * S_wx - min_v * S_x;
            }
        }
        y[m] = acc;
    }
}

void matvec_q4k_f32(const std::uint8_t* w, const float* x, float* y,
                    std::size_t M, std::size_t K) noexcept {
    matvec_q4k_f32_avx2(w, x, y, M, K);
}

} // namespace ultima::kernels
