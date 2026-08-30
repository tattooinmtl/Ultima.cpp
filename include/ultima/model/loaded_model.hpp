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
};

} // namespace ultima::model
