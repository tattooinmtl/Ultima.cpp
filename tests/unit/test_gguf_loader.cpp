#include <doctest/doctest.h>

#include "ultima/core/error.hpp"
#include "ultima/model/i_model_loader.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

// Helper: append raw bytes to a vector.
void put_u32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    out.push_back(static_cast<std::uint8_t>( v        & 0xff));
    out.push_back(static_cast<std::uint8_t>((v >>  8) & 0xff));
    out.push_back(static_cast<std::uint8_t>((v >> 16) & 0xff));
    out.push_back(static_cast<std::uint8_t>((v >> 24) & 0xff));
}
void put_u64(std::vector<std::uint8_t>& out, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<std::uint8_t>((v >> (i * 8)) & 0xff));
}
void put_str(std::vector<std::uint8_t>& out, const std::string& s) {
    put_u64(out, s.size());
    out.insert(out.end(), s.begin(), s.end());
}

std::filesystem::path make_fixture(const std::string& name,
                                   const std::vector<std::uint8_t>& bytes) {
    auto path = std::filesystem::temp_directory_path() / ("ultima_test_" + name);
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    return path;
}

// Build a minimal but structurally-valid GGUF v3 file with one metadata key
// (general.architecture = "test-arch") and zero tensors.
std::vector<std::uint8_t> build_minimal_valid() {
    std::vector<std::uint8_t> b;
    put_u32(b, 0x46554747);            // magic "GGUF"
    put_u32(b, 3);                     // version
    put_u64(b, 0);                     // tensor_count
    put_u64(b, 1);                     // metadata_kv_count
    put_str(b, "general.architecture");
    put_u32(b, 8);                     // value_type: String
    put_str(b, "test-arch");
    // Pad to alignment (32). tensor_data_base = align_up(current, 32)
    while (b.size() % 32 != 0) b.push_back(0);
    return b;
}

} // namespace

TEST_CASE("gguf: minimal valid file loads with expected architecture") {
    auto bytes = build_minimal_valid();
    auto path  = make_fixture("valid.gguf", bytes);

    {
        auto loader = ultima::model::make_gguf_loader();
        auto model_r = loader->load(path);
        REQUIRE(model_r.has_value());
        const auto& model = **model_r;
        CHECK(model.architecture() == "test-arch");
        CHECK(model.gguf_version() == 3u);
        CHECK(model.tensor_infos().empty());
        CHECK(model.metadata().size() == 1);
    }
    std::filesystem::remove(path);
}

TEST_CASE("gguf: bad magic rejected") {
    std::vector<std::uint8_t> b;
    put_u32(b, 0xDEADBEEF);
    put_u32(b, 3);
    put_u64(b, 0);
    put_u64(b, 0);
    while (b.size() % 32 != 0) b.push_back(0);

    auto path = make_fixture("badmagic.gguf", b);
    auto loader = ultima::model::make_gguf_loader();
    auto r = loader->load(path);

    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().code == ultima::core::ErrorCode::InvalidModel);

    std::filesystem::remove(path);
}

TEST_CASE("gguf: unsupported version rejected") {
    std::vector<std::uint8_t> b;
    put_u32(b, 0x46554747);
    put_u32(b, 99);                    // bogus version
    put_u64(b, 0);
    put_u64(b, 0);
    while (b.size() % 32 != 0) b.push_back(0);

    auto path = make_fixture("badver.gguf", b);
    auto loader = ultima::model::make_gguf_loader();
    auto r = loader->load(path);

    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().code == ultima::core::ErrorCode::UnsupportedVersion);

    std::filesystem::remove(path);
}

TEST_CASE("gguf: truncated file rejected") {
    auto bytes = build_minimal_valid();
    bytes.resize(bytes.size() / 2);    // chop in half

    auto path = make_fixture("truncated.gguf", bytes);
    auto loader = ultima::model::make_gguf_loader();
    auto r = loader->load(path);

    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().code == ultima::core::ErrorCode::InvalidModel);

    std::filesystem::remove(path);
}

TEST_CASE("gguf: split-file naming rejected") {
    auto bytes = build_minimal_valid();
    auto path  = make_fixture("model-00001-of-00003.gguf", bytes);

    auto loader = ultima::model::make_gguf_loader();
    auto r = loader->load(path);

    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().code == ultima::core::ErrorCode::MultiFileNotSupported);

    std::filesystem::remove(path);
}

TEST_CASE("gguf: missing required architecture key rejected") {
    // Zero metadata, zero tensors — no general.architecture.
    std::vector<std::uint8_t> b;
    put_u32(b, 0x46554747);
    put_u32(b, 3);
    put_u64(b, 0);
    put_u64(b, 0);
    while (b.size() % 32 != 0) b.push_back(0);

    auto path = make_fixture("noarch.gguf", b);
    auto loader = ultima::model::make_gguf_loader();
    auto r = loader->load(path);

    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().code == ultima::core::ErrorCode::MissingRequiredMetadata);

    std::filesystem::remove(path);
}

TEST_CASE("gguf: inspect returns basic summary") {
    auto bytes = build_minimal_valid();
    auto path  = make_fixture("inspect.gguf", bytes);

    auto loader = ultima::model::make_gguf_loader();
    auto s = loader->inspect(path);
    REQUIRE(s.has_value());
    CHECK(s->gguf_version == 3u);
    CHECK(s->tensor_count == 0u);
    CHECK(s->metadata_kv_count == 1u);
    CHECK(s->file_size_bytes == bytes.size());

    std::filesystem::remove(path);
}
