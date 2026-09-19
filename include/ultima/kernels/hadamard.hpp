#pragma once

#include <cstddef>

// Blockwise Hadamard rotation of activations.
//
// The Prism ternary checkpoints store each folded weight already multiplied
// by the inverse rotation, so the matmul only gives the right answer if the
// activation is rotated to match. The product is mathematically unchanged;
// the point is that the rotation spreads each weight's information across a
// block, so 2-bit quantization error lands as small noise everywhere instead
// of large error on a few outlier channels.
//
// Skipping this does not fail — it produces fluent-looking garbage, because
// the weights are still valid numbers in the wrong basis.

namespace ultima::kernels {

// In-place normalized Walsh-Hadamard transform over `n` values, n a power
// of two:
//
//   y[i] = (1/sqrt(n)) * sum_k (-1)^popcount(i & k) * x[k]
//
// Computed as the O(n log n) butterfly rather than the n x n matmul the
// reference materializes — at n = 1024 that is 10 passes instead of a
// million multiply-adds.
//
// The matrix is symmetric and orthogonal, so this is its own inverse.
void fwht_normalized_f32(float* x, std::size_t n) noexcept;

// Apply the activation-side fold to a vector of `width` values:
//
//   y = H_blockwise(signs * x)
//
// `signs` is one +/-1 per element (null for the identity sign mode), and
// the transform runs independently over each consecutive `block`-sized
// chunk. `width` must be a whole number of blocks.
// y may alias x.
void hadamard_fold_f32(const float* x, float* y, std::size_t width,
                       std::size_t block, const float* signs) noexcept;

// The reverse, for a table whose rows are stored already rotated:
//
//   y = signs * H_blockwise(x)
//
// Used on the token-embedding lookup, which holds latent rows.
// y may alias x.
void hadamard_unfold_f32(const float* x, float* y, std::size_t width,
                         std::size_t block, const float* signs) noexcept;

} // namespace ultima::kernels
