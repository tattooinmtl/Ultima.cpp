#pragma once

#include "ultima/core/error.hpp"
#include "ultima/model/loaded_model.hpp"

#include <cstddef>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace ultima::model {

// The `prism.hadamard.*` contract, read from the GGUF.
//
// These files are self-describing about the fold: the transform name, the
// block size, the explicit sign vectors, and the exact list of weights whose
// stored values already carry the inverse rotation. Nothing here is guessed.
struct HadamardFold {
    std::size_t block = 0;               // 1024 in every file seen so far

    // One +/-1 vector per activation width that some folded weight consumes.
    // Keyed by width because that is how the file stores them: sign_widths
    // lists the widths, sign_values is their concatenation.
    std::map<std::size_t, std::vector<float>> signs;

    // Tensor names whose activation must be folded before the matmul.
    std::set<std::string> folded;

    // Tensor names holding already-rotated rows, where the inverse is
    // applied to the lookup result instead. Only token_embd in practice.
    std::set<std::string> inverse;

    // The delta-net output projection was folded in grouped head order while
    // the runtime produces tiled order, so its activation needs a permute
    // first. See hadamard_permute_gdn_v.
    bool gdn_v_grouped = false;

    const float* signs_for(std::size_t width) const {
        auto it = signs.find(width);
        return it == signs.end() ? nullptr : it->second.data();
    }
    bool is_folded(const std::string& name) const {
        return folded.count(name) != 0;
    }
    bool is_inverse(const std::string& name) const {
        return inverse.count(name) != 0;
    }
};

// Read the fold description. Returns nullopt when the file declares no
// Hadamard folding, which is the ordinary case. Errors when the file
// declares one Ultima cannot honour exactly — a fold applied wrongly is
// worse than not loading, because it produces confident nonsense.
core::Result<std::optional<HadamardFold>>
parse_hadamard_fold(const LoadedModel& gguf);

// Reorder a delta-net output activation from the tiled head layout the
// recurrence produces, [head_dim, n_k_heads, rep], into the grouped layout
// the fold was computed in, [head_dim, rep, n_k_heads].
//
//   out[h + hd*(r + rep*k)] = in[h + hd*(k + nk*r)]
//
// `dst` must not alias `src`.
void hadamard_permute_gdn_v(const float* src, float* dst,
                            std::size_t head_dim, std::size_t n_k_heads,
                            std::size_t rep) noexcept;

} // namespace ultima::model
