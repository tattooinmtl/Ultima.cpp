#include <doctest/doctest.h>

#include "ultima/model/imodel.hpp"
#include "ultima/model/loaded_model.hpp"
#include "ultima/model/qwen_config.hpp"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {

// Reuses the "just enough to satisfy the interface" LoadedModel shim from
// the tokenizer tests. Only metadata is exercised — tensor() is not called
// by parse_qwen_config.
class FakeModel : public ultima::model::LoadedModel {
public:
    explicit FakeModel(std::string arch,
                       ultima::model::MetadataStore md)
        : arch_{std::move(arch)}, meta_{std::move(md)} {}

    const std::string&                            architecture()  const override { return arch_; }
    const ultima::model::MetadataStore&           metadata()      const override { return meta_; }
    const std::vector<ultima::model::TensorInfo>& tensor_infos()  const override { return tensors_; }
    std::optional<ultima::model::TensorView>      tensor(std::string_view) const override { return std::nullopt; }

    std::uint64_t file_size_bytes() const override { return 0; }
    std::uint64_t alignment()       const override { return 32; }
    std::uint32_t gguf_version()    const override { return 3; }

private:
    std::string                                   arch_;
    ultima::model::MetadataStore                  meta_;
    std::vector<ultima::model::TensorInfo>        tensors_;
};

using ultima::model::MetadataValue;

ultima::model::MetadataStore build_qwen2_metadata() {
    ultima::model::MetadataStore md;
    md.insert("general.architecture", MetadataValue{std::string{"qwen2"}});
    md.insert("qwen2.context_length",             MetadataValue{std::uint64_t{4096}});
    md.insert("qwen2.embedding_length",           MetadataValue{std::uint64_t{2048}});
    md.insert("qwen2.block_count",                MetadataValue{std::uint64_t{24}});
    md.insert("qwen2.feed_forward_length",        MetadataValue{std::uint64_t{5632}});
    md.insert("qwen2.attention.head_count",       MetadataValue{std::uint64_t{16}});
    md.insert("qwen2.attention.head_count_kv",    MetadataValue{std::uint64_t{4}});
    md.insert("qwen2.attention.layer_norm_rms_epsilon",
              MetadataValue{1e-6f});
    md.insert("qwen2.rope.freq_base",             MetadataValue{1000000.0f});
    md.insert("tokenizer.ggml.tokens",
              MetadataValue{std::vector<std::string>(151936, std::string{"x"})});
    return md;
}

} // namespace

TEST_CASE("qwen_config: parses standard Qwen2 metadata") {
    FakeModel m{"qwen2", build_qwen2_metadata()};

    ultima::model::ModelDims  d{};
    ultima::model::RopeConfig r{};
    auto ok = ultima::model::parse_qwen_config(m, "qwen2", d, r);
    REQUIRE(ok.has_value());

    CHECK(d.n_ctx_train == 4096);
    CHECK(d.hidden      == 2048);
    CHECK(d.n_layers    == 24);
    CHECK(d.ffn_hidden  == 5632);
    CHECK(d.n_heads     == 16);
    CHECK(d.n_kv_heads  == 4);
    CHECK(d.head_dim    == 128);   // hidden / n_heads
    CHECK(d.vocab       == 151936);
    CHECK(d.rms_eps     == doctest::Approx(1e-6f));
    CHECK(r.freq_base   == doctest::Approx(1000000.0f));
    CHECK(r.rope_dim    == 128);
    CHECK(r.scaling     == ultima::model::RopeConfig::Scaling::None);
}

TEST_CASE("qwen_config: hidden not divisible by n_heads is rejected") {
    auto md = build_qwen2_metadata();
    // Force a bad shape.
    md.insert("qwen2.embedding_length", MetadataValue{std::uint64_t{2049}});
    FakeModel m{"qwen2", std::move(md)};

    ultima::model::ModelDims  d{};
    ultima::model::RopeConfig r{};
    auto ok = ultima::model::parse_qwen_config(m, "qwen2", d, r);
    REQUIRE_FALSE(ok.has_value());
    CHECK(ok.error().code == ultima::core::ErrorCode::InvalidModel);
}

TEST_CASE("qwen_config: missing required key is caught") {
    ultima::model::MetadataStore md;
    md.insert("general.architecture", MetadataValue{std::string{"qwen2"}});
    // Omit qwen2.block_count etc.
    FakeModel m{"qwen2", std::move(md)};

    ultima::model::ModelDims  d{};
    ultima::model::RopeConfig r{};
    auto ok = ultima::model::parse_qwen_config(m, "qwen2", d, r);
    REQUIRE_FALSE(ok.has_value());
    CHECK(ok.error().code == ultima::core::ErrorCode::MissingRequiredMetadata);
}

TEST_CASE("qwen_config: YaRN scaling picked up from qwen3 metadata") {
    ultima::model::MetadataStore md;
    md.insert("general.architecture",             MetadataValue{std::string{"qwen3"}});
    md.insert("qwen3.context_length",             MetadataValue{std::uint64_t{32768}});
    md.insert("qwen3.embedding_length",           MetadataValue{std::uint64_t{2560}});
    md.insert("qwen3.block_count",                MetadataValue{std::uint64_t{32}});
    md.insert("qwen3.feed_forward_length",        MetadataValue{std::uint64_t{6912}});
    md.insert("qwen3.attention.head_count",       MetadataValue{std::uint64_t{20}});
    md.insert("qwen3.attention.head_count_kv",    MetadataValue{std::uint64_t{4}});
    md.insert("qwen3.rope.freq_base",             MetadataValue{5000000.0f});
    md.insert("qwen3.rope.scaling.type",          MetadataValue{std::string{"yarn"}});
    md.insert("qwen3.rope.scaling.factor",        MetadataValue{2.5f});
    md.insert("qwen3.rope.scaling.original_context_length",
              MetadataValue{std::uint64_t{16384}});
    md.insert("tokenizer.ggml.tokens",
              MetadataValue{std::vector<std::string>(151936, std::string{"x"})});

    FakeModel m{"qwen3", std::move(md)};
    ultima::model::ModelDims  d{};
    ultima::model::RopeConfig r{};
    auto ok = ultima::model::parse_qwen_config(m, "qwen3", d, r);
    REQUIRE(ok.has_value());
    CHECK(r.scaling        == ultima::model::RopeConfig::Scaling::YaRN);
    CHECK(r.yarn_factor    == doctest::Approx(2.5f));
    CHECK(r.yarn_ctx_train == doctest::Approx(16384.0f));
    CHECK(r.freq_base      == doctest::Approx(5000000.0f));
}
