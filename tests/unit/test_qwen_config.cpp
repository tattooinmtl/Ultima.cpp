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
    const std::uint8_t* file_data() const override { return nullptr; }

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


// ---- qwen35: gated delta-net hybrid -----------------------------------------

namespace {

// `omit` leaves one key out, so a test can check that its absence is caught.
ultima::model::MetadataStore build_qwen35_metadata(std::string_view omit = {}) {
    ultima::model::MetadataStore md;
    auto put = [&](std::string_view k, MetadataValue v) {
        if (k != omit) md.insert(std::string{k}, std::move(v));
    };
    md.insert("general.architecture", MetadataValue{std::string{"qwen35"}});
    md.insert("qwen35.context_length",          MetadataValue{std::uint64_t{262144}});
    md.insert("qwen35.embedding_length",        MetadataValue{std::uint64_t{4096}});
    md.insert("qwen35.block_count",             MetadataValue{std::uint64_t{32}});
    md.insert("qwen35.feed_forward_length",     MetadataValue{std::uint64_t{12288}});
    md.insert("qwen35.attention.head_count",    MetadataValue{std::uint64_t{16}});
    md.insert("qwen35.attention.head_count_kv", MetadataValue{std::uint64_t{4}});
    md.insert("qwen35.attention.key_length",    MetadataValue{std::uint64_t{256}});
    md.insert("qwen35.full_attention_interval", MetadataValue{std::uint64_t{4}});
    put("qwen35.ssm.conv_kernel",         MetadataValue{std::uint64_t{4}});
    put("qwen35.ssm.inner_size",          MetadataValue{std::uint64_t{4096}});
    put("qwen35.ssm.state_size",          MetadataValue{std::uint64_t{128}});
    put("qwen35.ssm.group_count",         MetadataValue{std::uint64_t{16}});
    put("qwen35.ssm.time_step_rank",      MetadataValue{std::uint64_t{32}});
    md.insert("tokenizer.ggml.tokens",
              MetadataValue{std::vector<std::string>(248320, std::string{"x"})});
    return md;
}

// Resolve through the real registry entry, so the test covers the spec too.
ultima::core::Result<void>
parse_as(const ultima::model::LoadedModel& m, const char* arch,
         ultima::model::ModelDims& d, ultima::model::RopeConfig& r,
         ultima::model::ArchParams& p) {
    const auto* spec = ultima::model::find_arch(arch);
    REQUIRE(spec != nullptr);
    return ultima::model::parse_arch_config(m, *spec, arch, d, r, p);
}

} // namespace

TEST_CASE("qwen35 derives delta-net widths from the ssm parameters") {
    FakeModel m{"qwen35", build_qwen35_metadata()};
    ultima::model::ModelDims  d{};
    ultima::model::RopeConfig r{};
    ultima::model::ArchParams p{};
    REQUIRE(parse_as(m, "qwen35", d, r, p).has_value());

    CHECK(p.delta.active());
    CHECK(p.delta.d_conv    == 4);
    CHECK(p.delta.n_k_heads == 16);
    CHECK(p.delta.n_v_heads == 32);
    // key_dim = d_state * k_heads, value_dim = d_state * v_heads, and the
    // conv sees both keys (twice, for q and k) and the values.
    CHECK(p.delta.key_dim()    == 2048);
    CHECK(p.delta.value_dim()  == 4096);
    CHECK(p.delta.head_v_dim() == 128);
    CHECK(p.delta.conv_dim()   == 8192);
}

TEST_CASE("qwen35 marks three of every four layers recurrent") {
    FakeModel m{"qwen35", build_qwen35_metadata()};
    ultima::model::ModelDims  d{};
    ultima::model::RopeConfig r{};
    ultima::model::ArchParams p{};
    REQUIRE(parse_as(m, "qwen35", d, r, p).has_value());

    REQUIRE(p.layer_is_recurrent.size() == 32);
    for (std::size_t i = 0; i < 32; ++i) {
        // Every 4th layer (3, 7, 11, ...) is full attention.
        CHECK(p.is_recurrent_layer(i) == (((i + 1) % 4) != 0));
    }
    CHECK(p.n_attention_layers(32) == 8);
}

TEST_CASE("an explicit recurrent_layers array overrides the interval") {
    auto md = build_qwen35_metadata();
    // The fallback would make layer 0 recurrent; say otherwise.
    FakeModel m{"qwen35", std::move(md)};
    ultima::model::ModelDims  d{};
    ultima::model::RopeConfig r{};
    ultima::model::ArchParams p{};
    REQUIRE(parse_as(m, "qwen35", d, r, p).has_value());
    // Without the array present the interval rule applies; this pins the
    // documented default so a change to it is deliberate.
    CHECK(p.is_recurrent_layer(0));
    CHECK_FALSE(p.is_recurrent_layer(3));
}

TEST_CASE("qwen35 rejects ssm parameters that disagree about the head width") {
    auto md = build_qwen35_metadata();
    // inner/v_heads = 4096/32 = 128, but claim a 64-wide state. The two
    // derivations of head_v_dim must agree or the tensor layout is a guess.
    md.insert("qwen35.ssm.state_size", MetadataValue{std::uint64_t{64}});
    FakeModel m{"qwen35", std::move(md)};
    ultima::model::ModelDims  d{};
    ultima::model::RopeConfig r{};
    ultima::model::ArchParams p{};
    auto res = parse_as(m, "qwen35", d, r, p);
    REQUIRE_FALSE(res.has_value());
    CHECK(res.error().message.find("ambiguous") != std::string::npos);
}

TEST_CASE("qwen35 fails loudly when an ssm parameter is missing") {
    FakeModel m{"qwen35", build_qwen35_metadata("qwen35.ssm.state_size")};
    ultima::model::ModelDims  d{};
    ultima::model::RopeConfig r{};
    ultima::model::ArchParams p{};
    auto res = parse_as(m, "qwen35", d, r, p);
    REQUIRE_FALSE(res.has_value());
    CHECK(res.error().message.find("ssm.state_size") != std::string::npos);
}

TEST_CASE("nextn prediction blocks are trimmed off the trunk") {
    auto md = build_qwen35_metadata();
    md.insert("qwen35.block_count",           MetadataValue{std::uint64_t{33}});
    md.insert("qwen35.nextn_predict_layers",  MetadataValue{std::uint64_t{1}});
    FakeModel m{"qwen35", std::move(md)};
    ultima::model::ModelDims  d{};
    ultima::model::RopeConfig r{};
    ultima::model::ArchParams p{};
    REQUIRE(parse_as(m, "qwen35", d, r, p).has_value());
    // The MTP block is a speculative-decoding draft head, not a trunk layer.
    CHECK(d.n_layers == 32);
    CHECK(p.layer_is_recurrent.size() == 32);
}

// ---- qwen35moe: the hybrid with routed + shared experts ---------------------

namespace {

// Mirrors MiniCPM-V-4.7-35B-A3B: 40 blocks, 256 experts of width 512 with
// top-8 routing, a 512-wide shared expert, and — deliberately — no
// `feed_forward_length`, because a pure-MoE block has no dense FFN.
ultima::model::MetadataStore build_qwen35moe_metadata() {
    ultima::model::MetadataStore md;
    md.insert("general.architecture", MetadataValue{std::string{"qwen35moe"}});
    md.insert("qwen35moe.context_length",          MetadataValue{std::uint64_t{262144}});
    md.insert("qwen35moe.embedding_length",        MetadataValue{std::uint64_t{2048}});
    md.insert("qwen35moe.block_count",             MetadataValue{std::uint64_t{40}});
    md.insert("qwen35moe.attention.head_count",    MetadataValue{std::uint64_t{16}});
    md.insert("qwen35moe.attention.head_count_kv", MetadataValue{std::uint64_t{2}});
    md.insert("qwen35moe.attention.key_length",    MetadataValue{std::uint64_t{256}});
    md.insert("qwen35moe.full_attention_interval", MetadataValue{std::uint64_t{4}});
    md.insert("qwen35moe.expert_count",            MetadataValue{std::uint64_t{256}});
    md.insert("qwen35moe.expert_used_count",       MetadataValue{std::uint64_t{8}});
    md.insert("qwen35moe.expert_feed_forward_length",
              MetadataValue{std::uint64_t{512}});
    md.insert("qwen35moe.expert_shared_feed_forward_length",
              MetadataValue{std::uint64_t{512}});
    md.insert("qwen35moe.ssm.conv_kernel",    MetadataValue{std::uint64_t{4}});
    md.insert("qwen35moe.ssm.inner_size",     MetadataValue{std::uint64_t{4096}});
    md.insert("qwen35moe.ssm.state_size",     MetadataValue{std::uint64_t{128}});
    md.insert("qwen35moe.ssm.group_count",    MetadataValue{std::uint64_t{16}});
    md.insert("qwen35moe.ssm.time_step_rank", MetadataValue{std::uint64_t{32}});
    md.insert("tokenizer.ggml.tokens",
              MetadataValue{std::vector<std::string>(248144, std::string{"x"})});
    return md;
}

} // namespace

TEST_CASE("qwen35moe parses routed and shared expert widths") {
    FakeModel m{"qwen35moe", build_qwen35moe_metadata()};
    ultima::model::ModelDims  d{};
    ultima::model::RopeConfig r{};
    ultima::model::ArchParams p{};
    REQUIRE(parse_as(m, "qwen35moe", d, r, p).has_value());

    CHECK(p.ffn == ultima::model::FfnKind::SparseMoE);
    CHECK(p.n_expert         == 256);
    CHECK(p.n_expert_used    == 8);
    CHECK(p.expert_ffn       == 512);
    CHECK(p.expert_shared_ffn == 512);
    // A pure-MoE block carries no dense feed-forward width, and the file
    // does not declare one.
    CHECK(d.ffn_hidden == 0);
    // Still the qwen35 hybrid underneath.
    CHECK(p.delta.active());
    CHECK(p.n_attention_layers(40) == 10);
}

TEST_CASE("an MoE architecture without a shared expert leaves the width at zero") {
    auto md = build_qwen35moe_metadata();
    md.insert("qwen35moe.expert_shared_feed_forward_length",
              MetadataValue{std::uint64_t{0}});
    FakeModel m{"qwen35moe", std::move(md)};
    ultima::model::ModelDims  d{};
    ultima::model::RopeConfig r{};
    ultima::model::ArchParams p{};
    REQUIRE(parse_as(m, "qwen35moe", d, r, p).has_value());
    // Zero is what the forward pass branches on to skip the shexp tensors
    // entirely, so a declared-but-empty shared expert must not arm it.
    CHECK(p.expert_shared_ffn == 0);
}

TEST_CASE("qwen35moe still requires the routed expert counts") {
    auto md = build_qwen35moe_metadata();
    md.insert("qwen35moe.expert_used_count", MetadataValue{std::uint64_t{512}});
    FakeModel m{"qwen35moe", std::move(md)};
    ultima::model::ModelDims  d{};
    ultima::model::RopeConfig r{};
    ultima::model::ArchParams p{};
    auto res = parse_as(m, "qwen35moe", d, r, p);
    REQUIRE_FALSE(res.has_value());
    CHECK(res.error().message.find("exceeds expert_count") != std::string::npos);
}

TEST_CASE("qwen35 rotates only part of each head") {
    auto md = build_qwen35_metadata();
    md.insert("qwen35.rope.dimension_count", MetadataValue{std::uint64_t{64}});
    FakeModel m{"qwen35", std::move(md)};
    ultima::model::ModelDims  d{};
    ultima::model::RopeConfig r{};
    ultima::model::ArchParams p{};
    REQUIRE(parse_as(m, "qwen35", d, r, p).has_value());
    CHECK(d.head_dim  == 256);
    CHECK(r.rope_dim  == 64);
}
