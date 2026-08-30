#include "ultima/model/metadata_store.hpp"

namespace ultima::model {

void MetadataStore::insert(std::string key, MetadataValue value) {
    kv_.insert_or_assign(std::move(key), std::move(value));
}

bool MetadataStore::contains(std::string_view key) const {
    return kv_.find(std::string(key)) != kv_.end();
}

const MetadataValue* MetadataStore::find(std::string_view key) const {
    auto it = kv_.find(std::string(key));
    if (it == kv_.end()) return nullptr;
    return &it->second;
}

std::optional<std::uint64_t> MetadataStore::get_uint(std::string_view key) const {
    const MetadataValue* v = find(key);
    if (!v) return std::nullopt;

    if (auto p = std::get_if<std::uint8_t >(v)) return static_cast<std::uint64_t>(*p);
    if (auto p = std::get_if<std::uint16_t>(v)) return static_cast<std::uint64_t>(*p);
    if (auto p = std::get_if<std::uint32_t>(v)) return static_cast<std::uint64_t>(*p);
    if (auto p = std::get_if<std::uint64_t>(v)) return *p;

    if (auto p = std::get_if<std::int8_t >(v); p && *p >= 0)
        return static_cast<std::uint64_t>(*p);
    if (auto p = std::get_if<std::int16_t>(v); p && *p >= 0)
        return static_cast<std::uint64_t>(*p);
    if (auto p = std::get_if<std::int32_t>(v); p && *p >= 0)
        return static_cast<std::uint64_t>(*p);
    if (auto p = std::get_if<std::int64_t>(v); p && *p >= 0)
        return static_cast<std::uint64_t>(*p);

    return std::nullopt;
}

std::optional<std::int64_t> MetadataStore::get_int(std::string_view key) const {
    const MetadataValue* v = find(key);
    if (!v) return std::nullopt;

    if (auto p = std::get_if<std::int8_t >(v)) return static_cast<std::int64_t>(*p);
    if (auto p = std::get_if<std::int16_t>(v)) return static_cast<std::int64_t>(*p);
    if (auto p = std::get_if<std::int32_t>(v)) return static_cast<std::int64_t>(*p);
    if (auto p = std::get_if<std::int64_t>(v)) return *p;

    if (auto p = std::get_if<std::uint8_t >(v)) return static_cast<std::int64_t>(*p);
    if (auto p = std::get_if<std::uint16_t>(v)) return static_cast<std::int64_t>(*p);
    if (auto p = std::get_if<std::uint32_t>(v)) return static_cast<std::int64_t>(*p);
    if (auto p = std::get_if<std::uint64_t>(v))
        return (*p > 0x7fffffffffffffffull) ? std::nullopt
                                            : std::optional<std::int64_t>(static_cast<std::int64_t>(*p));

    return std::nullopt;
}

} // namespace ultima::model
