#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace ultima::model {

// A lazy reference into the file's tensor-data section for a large array
// value (e.g. tokenizer vocab). Points into the mmap; owned by LoadedModel.
struct MetadataArrayView {
    std::uint32_t element_type;   // raw GGUF value type ID
    std::uint64_t count;
    std::uint64_t file_offset;    // absolute byte offset in the file
};

using MetadataValue = std::variant<
    std::monostate,
    bool,
    std::uint8_t,  std::int8_t,
    std::uint16_t, std::int16_t,
    std::uint32_t, std::int32_t,
    std::uint64_t, std::int64_t,
    float, double,
    std::string,
    std::vector<std::string>,
    MetadataArrayView
>;

class MetadataStore {
public:
    void insert(std::string key, MetadataValue value);

    bool contains(std::string_view key) const;

    const MetadataValue* find(std::string_view key) const;

    template <typename T>
    std::optional<T> get(std::string_view key) const {
        const MetadataValue* v = find(key);
        if (!v) return std::nullopt;
        if (const T* p = std::get_if<T>(v)) return *p;
        return std::nullopt;
    }

    // Best-effort integer coercion across the various uint/int widths.
    std::optional<std::uint64_t> get_uint(std::string_view key) const;
    std::optional<std::int64_t>  get_int(std::string_view key) const;

    std::size_t size() const noexcept { return kv_.size(); }

    const std::unordered_map<std::string, MetadataValue>& all() const noexcept {
        return kv_;
    }

private:
    std::unordered_map<std::string, MetadataValue> kv_;
};

} // namespace ultima::model
