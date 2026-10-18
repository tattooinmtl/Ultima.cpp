#include "ultima/model/hadamard.hpp"

#include <fmt/format.h>

#include <variant>

namespace ultima::model {

namespace {

using core::ErrorCode;
using core::Result;
using core::fail;

// Only one transform and one axis convention have ever been emitted, and a
// different value would mean different math. Refuse rather than assume.
constexpr const char* k_transform = "normalized-sylvester-walsh-hadamard";
constexpr const char* k_axis      = "input-last-dimension";

} // namespace

Result<std::optional<HadamardFold>>
parse_hadamard_fold(const LoadedModel& gguf) {
    const auto& md = gguf.metadata();

    const auto version = md.get_uint("prism.hadamard.version");
    if (!version) return std::optional<HadamardFold>{};   // no fold declared

    if (*version != 1) {
        return fail(ErrorCode::InvalidModel,
                    fmt::format("unsupported prism.hadamard.version {}", *version),
                    "hadamard");
    }

    HadamardFold f{};

    const auto transform = md.get<std::string>("prism.hadamard.transform");
    if (!transform || *transform != k_transform) {
        return fail(ErrorCode::InvalidModel,
                    fmt::format("unsupported prism.hadamard.transform '{}'",
                                transform ? *transform : std::string{"(absent)"}),
                    "hadamard");
    }
    const auto axis = md.get<std::string>("prism.hadamard.axis");
    if (!axis || *axis != k_axis) {
        return fail(ErrorCode::InvalidModel,
                    fmt::format("unsupported prism.hadamard.axis '{}'",
                                axis ? *axis : std::string{"(absent)"}),
                    "hadamard");
    }

    const auto block = md.get_uint("prism.hadamard.block_size");
    if (!block || *block == 0 || (*block & (*block - 1)) != 0) {
        return fail(ErrorCode::InvalidModel,
                    "prism.hadamard.block_size must be a power of two",
                    "hadamard");
    }
    f.block = static_cast<std::size_t>(*block);

    const auto names = md.get<std::vector<std::string>>("prism.hadamard.weight_names");
    if (!names || names->empty()) {
        return fail(ErrorCode::InvalidModel,
                    "prism.hadamard.weight_names is missing or empty", "hadamard");
    }
    for (const auto& n : *names) f.folded.insert(n);

    if (auto inv = md.get<std::vector<std::string>>("prism.hadamard.inverse_weight_names")) {
        for (const auto& n : *inv) {
            // The inverse is only applied at the embedding lookup. Any other
            // latent table would load and silently stay rotated.
            if (n != "token_embd.weight") {
                return fail(ErrorCode::InvalidModel,
                            fmt::format("prism.hadamard: '{}' is not a supported "
                                        "inverse-after-lookup table", n),
                            "hadamard");
            }
            f.inverse.insert(n);
        }
    }

    const auto sign_mode = md.get<std::string>("prism.hadamard.sign_mode");
    if (sign_mode && *sign_mode == "explicit") {
        const auto* widths_v = md.find("prism.hadamard.sign_widths");
        const auto* values_v = md.find("prism.hadamard.sign_values");
        const auto* widths_a = widths_v ? std::get_if<MetadataArrayView>(widths_v) : nullptr;
        const auto* values_a = values_v ? std::get_if<MetadataArrayView>(values_v) : nullptr;
        if (!widths_a || !values_a) {
            return fail(ErrorCode::InvalidModel,
                        "prism.hadamard.sign_mode is explicit but the sign "
                        "arrays are missing; an empty sign table reads as the "
                        "identity and would silently change the model",
                        "hadamard");
        }
        const auto widths = read_i32_array(gguf, *widths_a);
        const auto values = read_i32_array(gguf, *values_a);

        std::size_t off = 0;
        for (const auto w : widths) {
            if (w <= 0 || (static_cast<std::size_t>(w) % f.block) != 0 ||
                off + static_cast<std::size_t>(w) > values.size()) {
                return fail(ErrorCode::InvalidModel,
                            fmt::format("invalid prism.hadamard sign width {}", w),
                            "hadamard");
            }
            std::vector<float> vec(static_cast<std::size_t>(w));
            for (std::size_t i = 0; i < vec.size(); ++i) {
                const auto v = values[off + i];
                if (v != 1 && v != -1) {
                    return fail(ErrorCode::InvalidModel,
                                fmt::format("prism.hadamard sign values must be "
                                            "+/-1, got {}", v),
                                "hadamard");
                }
                vec[i] = static_cast<float>(v);
            }
            f.signs.emplace(static_cast<std::size_t>(w), std::move(vec));
            off += static_cast<std::size_t>(w);
        }
        if (off != values.size()) {
            return fail(ErrorCode::InvalidModel,
                        fmt::format("prism.hadamard.sign_values length mismatch: "
                                    "widths cover {} of {} entries",
                                    off, values.size()),
                        "hadamard");
        }
    } else if (sign_mode && *sign_mode != "identity") {
        return fail(ErrorCode::InvalidModel,
                    fmt::format("unsupported prism.hadamard.sign_mode '{}'", *sign_mode),
                    "hadamard");
    }

    if (auto g = md.get_uint("prism.hadamard.gdn_v_grouped")) {
        f.gdn_v_grouped = (*g != 0);
    } else if (auto b = md.get<bool>("prism.hadamard.gdn_v_grouped")) {
        f.gdn_v_grouped = *b;
    }

    return std::optional<HadamardFold>{std::move(f)};
}

void hadamard_permute_gdn_v(const float* src, float* dst,
                            std::size_t head_dim, std::size_t n_k_heads,
                            std::size_t rep) noexcept {
    for (std::size_t k = 0; k < n_k_heads; ++k) {
        for (std::size_t r = 0; r < rep; ++r) {
            const float* in  = src + head_dim * (k + n_k_heads * r);
            float*       out = dst + head_dim * (r + rep * k);
            for (std::size_t h = 0; h < head_dim; ++h) out[h] = in[h];
        }
    }
}

} // namespace ultima::model
