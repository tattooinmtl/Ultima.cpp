#include "ultima/model/loaded_model.hpp"

#include <cstring>

namespace ultima::model {

namespace {

// GGUF metadata value type ids for the element types we accept here.
constexpr std::uint32_t kI32 = 5;
constexpr std::uint32_t kF32 = 6;
constexpr std::uint32_t kU32 = 4;

// Bounds-check the view against the file before reading `count` elements of
// `elem_size` bytes at view.file_offset.
bool view_in_bounds(const LoadedModel& m, const MetadataArrayView& v,
                    std::size_t elem_size) {
    if (m.file_data() == nullptr) return false;
    const std::uint64_t bytes = v.count * elem_size;
    if (v.count != 0 && bytes / v.count != elem_size) return false;   // overflow
    return v.file_offset + bytes <= m.file_size_bytes();
}

} // namespace

std::vector<float> read_f32_array(const LoadedModel& model,
                                  const MetadataArrayView& view) {
    if (!view_in_bounds(model, view, 4)) return {};

    const std::uint8_t* p = model.file_data() + view.file_offset;
    std::vector<float> out(static_cast<std::size_t>(view.count));

    switch (view.element_type) {
    case kF32:
        std::memcpy(out.data(), p, out.size() * sizeof(float));
        return out;
    case kI32:
    case kU32:
        for (std::size_t i = 0; i < out.size(); ++i) {
            std::int32_t v;
            std::memcpy(&v, p + i * 4, sizeof(v));
            out[i] = static_cast<float>(v);
        }
        return out;
    default:
        return {};
    }
}

std::vector<std::uint8_t> read_u8_array(const LoadedModel& model,
                                        const MetadataArrayView& view) {
    // 0 = UINT8, 1 = INT8, 7 = BOOL — all one byte per element.
    if (view.element_type != 0 && view.element_type != 1 && view.element_type != 7) {
        return {};
    }
    if (!view_in_bounds(model, view, 1)) return {};

    std::vector<std::uint8_t> out(static_cast<std::size_t>(view.count));
    std::memcpy(out.data(), model.file_data() + view.file_offset, out.size());
    return out;
}

std::vector<std::int32_t> read_i32_array(const LoadedModel& model,
                                         const MetadataArrayView& view) {
    if (view.element_type != kI32 && view.element_type != kU32) return {};
    if (!view_in_bounds(model, view, 4)) return {};

    std::vector<std::int32_t> out(static_cast<std::size_t>(view.count));
    std::memcpy(out.data(), model.file_data() + view.file_offset,
                out.size() * sizeof(std::int32_t));
    return out;
}

} // namespace ultima::model
