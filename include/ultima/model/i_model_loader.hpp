#pragma once

#include "ultima/core/error.hpp"
#include "ultima/model/loaded_model.hpp"

#include <filesystem>
#include <memory>

namespace ultima::model {

// Metadata-only result of a cheap header inspection. Does not mmap the file.
struct ModelSummary {
    std::string   architecture;
    std::uint32_t gguf_version;
    std::uint64_t file_size_bytes;
    std::uint64_t metadata_kv_count;
    std::uint64_t tensor_count;
};

class IModelLoader {
public:
    virtual ~IModelLoader() = default;

    // Cheap: reads only enough of the file to fill ModelSummary. No mmap of
    // the tensor blob. Suitable for registry inspection and pre-flight.
    virtual core::Result<ModelSummary>
        inspect(const std::filesystem::path& path) = 0;

    // Full load: mmaps the file, parses metadata and tensor directory.
    // Constant time on top of the linear metadata/tensor walks.
    virtual core::Result<std::unique_ptr<LoadedModel>>
        load(const std::filesystem::path& path) = 0;
};

// Concrete loader for GGUF v3 files.
std::unique_ptr<IModelLoader> make_gguf_loader();

} // namespace ultima::model
