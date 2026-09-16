#pragma once

#include <cstddef>
#include <span>
#include <string_view>

namespace ultima::model {

// What shape of FFN a block uses.
enum class FfnKind {
    DenseSwiGLU,    // separate ffn_gate / ffn_up / ffn_down
    FusedSwiGLU,    // single ffn_up holding [gate || up], then ffn_down
    SparseMoE,      // router + per-expert gate/up/down, top-k weighted
};

// How Q/K/V reach the block.
enum class QkvKind {
    Separate,       // attn_q / attn_k / attn_v
    Fused,          // one attn_qkv holding [Q || K || V]
};

// Rotary convention. NeoX rotates (i, i + rope_dim/2); Norm rotates the
// consecutive pair (2i, 2i+1). Getting this wrong produces fluent-looking
// but subtly wrong output, so it is per-arch data rather than a guess.
enum class RopeStyle {
    NeoX,
    Norm,
};

// Every architecture Ultima can name. `supported == false` entries exist so
// the loader can say precisely what a file needs instead of failing with a
// generic "unknown architecture".
struct ArchSpec {
    std::string_view name;
    bool             supported = false;

    // Only meaningful when supported.
    FfnKind   ffn        = FfnKind::DenseSwiGLU;
    QkvKind   qkv        = QkvKind::Separate;
    RopeStyle rope_style = RopeStyle::NeoX;

    // Per-head sigmoid gate on the attention output. Structural, not
    // derivable from metadata, so it lives here rather than being probed
    // from the presence of an attn_gate tensor.
    bool      attn_gate  = false;

    // Human-readable. For supported archs, a short family description; for
    // unsupported ones, the engine work the file still needs.
    std::string_view notes;
};

// Look up an architecture by its GGUF `general.architecture` value.
// Returns nullptr when the name is not in the table at all.
const ArchSpec* find_arch(std::string_view name) noexcept;

// The whole table, for `ultima archs` and for diagnostics.
std::span<const ArchSpec> all_archs() noexcept;

} // namespace ultima::model
