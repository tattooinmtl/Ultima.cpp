#include "ultima/model/i_model_loader.hpp"

#include "gguf_keys.hpp"
#include "gguf_reader.hpp"
#include "mmap_file.hpp"

#include <fmt/format.h>

#include <array>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace ultima::model {
namespace {

using core::ErrorCode;
using core::Failure;
using core::Result;
using core::fail;

constexpr std::uint32_t k_gguf_magic     = 0x46554747u;   // "GGUF"
constexpr std::uint32_t k_gguf_version_v3 = 3u;
constexpr std::uint64_t k_default_alignment = 32u;

// GGUF metadata value type enum values, per the public spec.
enum class GgufValueType : std::uint32_t {
    U8      = 0,
    I8      = 1,
    U16     = 2,
    I16     = 3,
    U32     = 4,
    I32     = 5,
    F32     = 6,
    Bool    = 7,
    String  = 8,
    Array   = 9,
    U64     = 10,
    I64     = 11,
    F64     = 12,
};

// Read a single metadata value (recursive for Array).
Result<MetadataValue> read_value(GgufReader& r, std::uint32_t type_id,
                                 const std::uint8_t* base);

Result<MetadataValue> read_array(GgufReader& r, const std::uint8_t* base) {
    auto elem_type_r = r.read_scalar<std::uint32_t>();
    if (!elem_type_r) return Failure{elem_type_r.error()};
    auto count_r = r.read_scalar<std::uint64_t>();
    if (!count_r) return Failure{count_r.error()};

    const std::uint32_t elem_type = *elem_type_r;
    const std::uint64_t count     = *count_r;

    // Selective materialization: strings (typical vocab is 100k+ entries) get
    // fully materialized into vector<string>. Other array types get stored
    // as a view (offset + count) since we do not use them at load time.
    if (elem_type == static_cast<std::uint32_t>(GgufValueType::String)) {
        std::vector<std::string> out;
        out.reserve(static_cast<std::size_t>(count));
        for (std::uint64_t i = 0; i < count; ++i) {
            auto s = r.read_string();
            if (!s) return Failure{s.error()};
            out.push_back(std::move(*s));
        }
        return MetadataValue{std::move(out)};
    }

    // For non-string arrays, we know the fixed byte-size per element.
    std::size_t elem_size = 0;
    switch (static_cast<GgufValueType>(elem_type)) {
    case GgufValueType::U8:   case GgufValueType::I8:   case GgufValueType::Bool: elem_size = 1; break;
    case GgufValueType::U16:  case GgufValueType::I16:                             elem_size = 2; break;
    case GgufValueType::U32:  case GgufValueType::I32:  case GgufValueType::F32:  elem_size = 4; break;
    case GgufValueType::U64:  case GgufValueType::I64:  case GgufValueType::F64:  elem_size = 8; break;
    default:
        return fail(ErrorCode::InvalidModel,
                    fmt::format("unsupported array element type {}", elem_type),
                    "gguf_loader");
    }

    MetadataArrayView view;
    view.element_type = elem_type;
    view.count        = count;
    view.file_offset  = static_cast<std::uint64_t>(r.offset());
    if (auto sk = r.skip(static_cast<std::size_t>(count * elem_size)); !sk)
        return Failure{sk.error()};
    (void)base;
    return MetadataValue{view};
}

Result<MetadataValue> read_value(GgufReader& r, std::uint32_t type_id,
                                 const std::uint8_t* base) {
    switch (static_cast<GgufValueType>(type_id)) {
    case GgufValueType::U8:     { auto v = r.read_scalar<std::uint8_t >(); if (!v) return Failure{v.error()}; return MetadataValue{*v}; }
    case GgufValueType::I8:     { auto v = r.read_scalar<std::int8_t  >(); if (!v) return Failure{v.error()}; return MetadataValue{*v}; }
    case GgufValueType::U16:    { auto v = r.read_scalar<std::uint16_t>(); if (!v) return Failure{v.error()}; return MetadataValue{*v}; }
    case GgufValueType::I16:    { auto v = r.read_scalar<std::int16_t >(); if (!v) return Failure{v.error()}; return MetadataValue{*v}; }
    case GgufValueType::U32:    { auto v = r.read_scalar<std::uint32_t>(); if (!v) return Failure{v.error()}; return MetadataValue{*v}; }
    case GgufValueType::I32:    { auto v = r.read_scalar<std::int32_t >(); if (!v) return Failure{v.error()}; return MetadataValue{*v}; }
    case GgufValueType::F32:    { auto v = r.read_scalar<float        >(); if (!v) return Failure{v.error()}; return MetadataValue{*v}; }
    case GgufValueType::Bool:   { auto v = r.read_scalar<std::uint8_t >(); if (!v) return Failure{v.error()}; return MetadataValue{static_cast<bool>(*v != 0)}; }
    case GgufValueType::U64:    { auto v = r.read_scalar<std::uint64_t>(); if (!v) return Failure{v.error()}; return MetadataValue{*v}; }
    case GgufValueType::I64:    { auto v = r.read_scalar<std::int64_t >(); if (!v) return Failure{v.error()}; return MetadataValue{*v}; }
    case GgufValueType::F64:    { auto v = r.read_scalar<double       >(); if (!v) return Failure{v.error()}; return MetadataValue{*v}; }
    case GgufValueType::String: { auto v = r.read_string();                if (!v) return Failure{v.error()}; return MetadataValue{std::move(*v)}; }
    case GgufValueType::Array:  return read_array(r, base);
    }
    return fail(ErrorCode::InvalidModel,
                fmt::format("unknown metadata value type {}", type_id),
                "gguf_loader");
}

DataType map_dtype(std::uint32_t raw) noexcept {
    switch (raw) {
    case 0:  return DataType::F32;
    case 1:  return DataType::F16;
    case 2:  return DataType::Q4_0;
    case 3:  return DataType::Q4_1;
    case 6:  return DataType::Q5_0;
    case 7:  return DataType::Q5_1;
    case 8:  return DataType::Q8_0;
    case 10: return DataType::Q2_K;
    case 11: return DataType::Q3_K;
    case 12: return DataType::Q4_K;
    case 13: return DataType::Q5_K;
    case 14: return DataType::Q6_K;
    case 15: return DataType::Q8_K;
    case 30: return DataType::BF16;
    default: return DataType::Unknown;
    }
}

// ----- Concrete LoadedModel --------------------------------------------------
class GgufLoadedModel final : public LoadedModel {
public:
    GgufLoadedModel() = default;

    const std::string&             architecture() const override { return architecture_; }
    const MetadataStore&           metadata()     const override { return metadata_; }
    const std::vector<TensorInfo>& tensor_infos() const override { return tensors_; }

    std::optional<TensorView> tensor(std::string_view name) const override {
        auto it = tensor_index_.find(std::string(name));
        if (it == tensor_index_.end()) return std::nullopt;
        const TensorInfo& info = tensors_[it->second];
        return TensorView{info, file_.data() + info.offset};
    }

    std::uint64_t file_size_bytes() const override { return file_.size(); }
    std::uint64_t alignment()       const override { return alignment_; }
    std::uint32_t gguf_version()    const override { return gguf_version_; }

    // Mutators used by the loader during construction:
    MmapFile                                file_;
    std::string                             architecture_;
    MetadataStore                           metadata_;
    std::vector<TensorInfo>                 tensors_;
    std::unordered_map<std::string, std::size_t> tensor_index_;
    std::uint64_t                           alignment_    = k_default_alignment;
    std::uint32_t                           gguf_version_ = 0;
};

// ----- Loader implementation -------------------------------------------------
class GgufLoader final : public IModelLoader {
public:
    Result<ModelSummary> inspect(const std::filesystem::path& path) override;
    Result<std::unique_ptr<LoadedModel>> load(const std::filesystem::path& path) override;

private:
    static Result<void> read_header_(GgufReader& r,
                                     std::uint32_t& out_version,
                                     std::uint64_t& out_tensor_count,
                                     std::uint64_t& out_metadata_count);
    static Result<void> read_metadata_(GgufReader& r, std::uint64_t kv_count,
                                       MetadataStore& out, const std::uint8_t* base);
    static Result<void> read_tensor_directory_(GgufReader& r, std::uint64_t tensor_count,
                                               std::vector<TensorInfo>& out);
    static void         detect_split_file_(const std::filesystem::path& path);
};

Result<void> GgufLoader::read_header_(GgufReader& r,
                                      std::uint32_t& out_version,
                                      std::uint64_t& out_tensor_count,
                                      std::uint64_t& out_metadata_count) {
    auto magic = r.read_scalar<std::uint32_t>();
    if (!magic) return Failure{magic.error()};
    if (*magic != k_gguf_magic) {
        return fail(ErrorCode::InvalidModel,
                    fmt::format("bad magic 0x{:08x} (expected 0x{:08x} \"GGUF\")",
                                *magic, k_gguf_magic),
                    "gguf_loader");
    }
    auto version = r.read_scalar<std::uint32_t>();
    if (!version) return Failure{version.error()};
    if (*version != k_gguf_version_v3) {
        return fail(ErrorCode::UnsupportedVersion,
                    fmt::format("GGUF v{} not supported in v0.1 (require v{})",
                                *version, k_gguf_version_v3),
                    "gguf_loader");
    }
    out_version = *version;

    auto tc = r.read_scalar<std::uint64_t>();
    if (!tc) return Failure{tc.error()};
    auto mc = r.read_scalar<std::uint64_t>();
    if (!mc) return Failure{mc.error()};
    out_tensor_count   = *tc;
    out_metadata_count = *mc;
    return {};
}

Result<void> GgufLoader::read_metadata_(GgufReader& r, std::uint64_t kv_count,
                                        MetadataStore& out, const std::uint8_t* base) {
    for (std::uint64_t i = 0; i < kv_count; ++i) {
        auto key = r.read_string();
        if (!key) return Failure{key.error()};
        auto type_id = r.read_scalar<std::uint32_t>();
        if (!type_id) return Failure{type_id.error()};
        auto value = read_value(r, *type_id, base);
        if (!value) return Failure{value.error()};
        out.insert(std::move(*key), std::move(*value));
    }
    return {};
}

Result<void> GgufLoader::read_tensor_directory_(GgufReader& r, std::uint64_t tensor_count,
                                                std::vector<TensorInfo>& out) {
    out.clear();
    out.reserve(static_cast<std::size_t>(tensor_count));
    for (std::uint64_t i = 0; i < tensor_count; ++i) {
        TensorInfo t{};
        auto name = r.read_string();
        if (!name) return Failure{name.error()};
        t.name = std::move(*name);

        auto ndims = r.read_scalar<std::uint32_t>();
        if (!ndims) return Failure{ndims.error()};
        if (*ndims == 0 || *ndims > 4) {
            return fail(ErrorCode::CorruptTensorDirectory,
                        fmt::format("tensor '{}' has bad n_dims {}", t.name, *ndims),
                        "gguf_loader");
        }
        t.dims.resize(*ndims);
        for (std::uint32_t d = 0; d < *ndims; ++d) {
            auto dim = r.read_scalar<std::uint64_t>();
            if (!dim) return Failure{dim.error()};
            if (*dim == 0) {
                return fail(ErrorCode::CorruptTensorDirectory,
                            fmt::format("tensor '{}' has zero dim at index {}", t.name, d),
                            "gguf_loader");
            }
            t.dims[d] = *dim;
        }
        auto raw_type = r.read_scalar<std::uint32_t>();
        if (!raw_type) return Failure{raw_type.error()};
        t.dtype = map_dtype(*raw_type);

        auto off = r.read_scalar<std::uint64_t>();
        if (!off) return Failure{off.error()};
        t.offset = *off;
        t.size_bytes = compute_tensor_bytes(t.dtype, t.dims);
        out.push_back(std::move(t));
    }
    return {};
}

void GgufLoader::detect_split_file_(const std::filesystem::path& path) {
    // Best-effort filename check for the "-00001-of-000NN.gguf" split pattern.
    // Purely informational at inspect; hard-fail is applied by callers.
    (void)path;
}

Result<ModelSummary> GgufLoader::inspect(const std::filesystem::path& path) {
    // Detect split-file naming and reject with a clear error.
    const auto fname = path.filename().string();
    if (fname.find("-of-") != std::string::npos && fname.find("-000") != std::string::npos) {
        return fail(ErrorCode::MultiFileNotSupported,
                    fmt::format("multi-file GGUFs not supported in v0.1: '{}'", fname),
                    "gguf_loader");
    }

    auto mm = MmapFile::open(path);
    if (!mm) return Failure{mm.error()};

    GgufReader r{mm->data(), mm->size()};
    std::uint32_t ver = 0;
    std::uint64_t tc = 0, mc = 0;
    if (auto h = read_header_(r, ver, tc, mc); !h) return Failure{h.error()};

    ModelSummary s{};
    s.gguf_version      = ver;
    s.file_size_bytes   = static_cast<std::uint64_t>(mm->size());
    s.metadata_kv_count = mc;
    s.tensor_count      = tc;
    // We do not walk the whole metadata for architecture name — a full load
    // does that. inspect leaves architecture empty; callers wanting it should
    // call load().
    return s;
}

Result<std::unique_ptr<LoadedModel>> GgufLoader::load(const std::filesystem::path& path) {
    const auto fname = path.filename().string();
    if (fname.find("-of-") != std::string::npos && fname.find("-000") != std::string::npos) {
        return fail(ErrorCode::MultiFileNotSupported,
                    fmt::format("multi-file GGUFs not supported in v0.1: '{}'", fname),
                    "gguf_loader");
    }

    auto mm = MmapFile::open(path);
    if (!mm) return Failure{mm.error()};

    auto model = std::make_unique<GgufLoadedModel>();
    model->file_ = std::move(*mm);

    GgufReader r{model->file_.data(), model->file_.size()};

    std::uint64_t tc = 0, mc = 0;
    if (auto h = read_header_(r, model->gguf_version_, tc, mc); !h)
        return Failure{h.error()};

    if (auto md = read_metadata_(r, mc, model->metadata_, model->file_.data()); !md)
        return Failure{md.error()};

    if (auto td = read_tensor_directory_(r, tc, model->tensors_); !td)
        return Failure{td.error()};

    // Pull alignment override if present
    if (auto align = model->metadata_.get_uint(gguf_keys::general_alignment)) {
        if (*align > 0) model->alignment_ = *align;
    }

    // Required: general.architecture
    auto arch = model->metadata_.get<std::string>(gguf_keys::general_architecture);
    if (!arch) {
        return fail(ErrorCode::MissingRequiredMetadata,
                    fmt::format("missing required key '{}'", gguf_keys::general_architecture),
                    "gguf_loader");
    }
    model->architecture_ = std::move(*arch);

    // Compute tensor data base: position after the tensor directory, aligned up.
    if (auto a = r.align_to(model->alignment_); !a) return Failure{a.error()};
    const std::uint64_t tensor_data_base = static_cast<std::uint64_t>(r.offset());

    // Convert per-tensor offsets from "relative to tensor_data_base" to
    // absolute file offsets, and validate they fit inside the file.
    for (auto& t : model->tensors_) {
        t.offset += tensor_data_base;
        if (t.size_bytes == 0) continue;   // unsupported dtype: leave alone
        if (t.offset + t.size_bytes > model->file_.size()) {
            return fail(ErrorCode::CorruptTensorDirectory,
                        fmt::format("tensor '{}' extends past end of file "
                                    "(off {} + size {} > file {})",
                                    t.name, t.offset, t.size_bytes,
                                    model->file_.size()),
                        "gguf_loader");
        }
    }

    // Name index for O(1) lookup
    model->tensor_index_.reserve(model->tensors_.size());
    for (std::size_t i = 0; i < model->tensors_.size(); ++i) {
        model->tensor_index_.emplace(model->tensors_[i].name, i);
    }

    return std::unique_ptr<LoadedModel>{model.release()};
}

} // namespace

std::unique_ptr<IModelLoader> make_gguf_loader() {
    return std::make_unique<GgufLoader>();
}

} // namespace ultima::model
