#pragma once

#include "ultima/model/metadata_store.hpp"
#include "ultima/model/tensor_info.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ultima::model {

// Non-owning view into a tensor's data. Points into the LoadedModel's mmap.
// Invalidated when the owning LoadedModel is destroyed.
class TensorView {
public:
    TensorView(const TensorInfo& info, const void* data) noexcept
        : info_{&info}, data_{data} {}

    const TensorInfo& info() const noexcept { return *info_; }
    const void*       data() const noexcept { return data_; }

private:
    const TensorInfo* info_;
    const void*       data_;
};

class LoadedModel {
public:
    virtual ~LoadedModel() = default;

    virtual const std::string&             architecture()   const = 0;
    virtual const MetadataStore&           metadata()       const = 0;
    virtual const std::vector<TensorInfo>& tensor_infos()   const = 0;
    virtual std::optional<TensorView>      tensor(std::string_view name) const = 0;

    virtual std::uint64_t                  file_size_bytes() const = 0;
    virtual std::uint64_t                  alignment()       const = 0;
    virtual std::uint32_t                  gguf_version()    const = 0;

    // Base of the mapped file. Non-string metadata arrays are stored as
    // MetadataArrayView (offset + count) rather than materialized, so a
    // reader needs this to resolve one. May be null for non-file-backed
    // implementations (test shims); read_f32_array handles that.
    virtual const std::uint8_t*            file_data()       const = 0;
};

// Materialize a numeric metadata array as floats. Handles the F32 and the
// signed/unsigned 32-bit integer element types, which is what GGUF uses for
// tokenizer.ggml.scores and tokenizer.ggml.token_type. Returns an empty
// vector when the view cannot be resolved or has an unexpected element type.
std::vector<float> read_f32_array(const LoadedModel& model,
                                  const MetadataArrayView& view);

std::vector<std::int32_t> read_i32_array(const LoadedModel& model,
                                         const MetadataArrayView& view);

// Single-byte element arrays (GGUF bool / uint8), used for per-layer flag
// vectors such as attention.sliding_window_pattern.
std::vector<std::uint8_t> read_u8_array(const LoadedModel& model,
                                        const MetadataArrayView& view);

} // namespace ultima::model
