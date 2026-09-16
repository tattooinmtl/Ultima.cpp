#include <doctest/doctest.h>

#include "ultima/model/loaded_model.hpp"
#include "ultima/tokenizer/tokenizer.hpp"

#include <cstdio>
#include <cstring>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace {

using ultima::model::MetadataArrayView;
using ultima::model::MetadataStore;
using ultima::model::MetadataValue;
using ultima::tokenizer::BpeTokenizer;
using ultima::tokenizer::TokenId;

// SentencePiece vocabs keep scores and token types in numeric metadata arrays,
// which the GGUF loader leaves as (offset, count) views into the mapped file
// rather than materializing. This shim owns a byte buffer and hands it back
// from file_data(), so the views resolve exactly as they would from a file.
class SpmFakeModel : public ultima::model::LoadedModel {
public:
    SpmFakeModel(MetadataStore md, std::vector<std::uint8_t> blob)
        : arch_{"llama"}, meta_{std::move(md)}, blob_{std::move(blob)} {}

    const std::string&                  architecture() const override { return arch_; }
    const MetadataStore&                metadata()     const override { return meta_; }
    const std::vector<ultima::model::TensorInfo>& tensor_infos() const override { return tensors_; }
    std::optional<ultima::model::TensorView> tensor(std::string_view) const override { return std::nullopt; }

    std::uint64_t file_size_bytes() const override { return blob_.size(); }
    std::uint64_t alignment()       const override { return 32; }
    std::uint32_t gguf_version()    const override { return 3; }
    const std::uint8_t* file_data() const override { return blob_.data(); }

private:
    std::string                            arch_;
    MetadataStore                          meta_;
    std::vector<ultima::model::TensorInfo> tensors_;
    std::vector<std::uint8_t>              blob_;
};

constexpr const char* kSp = "\xE2\x96\x81";   // U+2581

struct SpmVocabSpec {
    std::vector<std::string>  tokens;
    std::vector<float>        scores;
    std::vector<std::int32_t> types;   // 1 normal, 3 control, 6 byte
};

std::unique_ptr<BpeTokenizer> build(const SpmVocabSpec& spec) {
    // Lay scores then types into one blob and point two array views at them.
    std::vector<std::uint8_t> blob(spec.scores.size() * 4 + spec.types.size() * 4);
    std::memcpy(blob.data(), spec.scores.data(), spec.scores.size() * 4);
    const std::size_t types_off = spec.scores.size() * 4;
    std::memcpy(blob.data() + types_off, spec.types.data(), spec.types.size() * 4);

    MetadataStore md;
    md.insert("tokenizer.ggml.model",  MetadataValue{std::string{"llama"}});
    md.insert("tokenizer.ggml.tokens", MetadataValue{spec.tokens});

    MetadataArrayView scores_view{};
    scores_view.element_type = 6;   // F32
    scores_view.count        = spec.scores.size();
    scores_view.file_offset  = 0;
    md.insert("tokenizer.ggml.scores", MetadataValue{scores_view});

    MetadataArrayView types_view{};
    types_view.element_type = 5;    // I32
    types_view.count        = spec.types.size();
    types_view.file_offset  = types_off;
    md.insert("tokenizer.ggml.token_type", MetadataValue{types_view});

    SpmFakeModel model{std::move(md), std::move(blob)};
    auto r = BpeTokenizer::load(model);
    REQUIRE(static_cast<bool>(r));
    return std::move(*r);
}

// A vocab covering the pieces needed for "▁hi ▁there" plus byte fallback for
// characters that have no piece of their own.
SpmVocabSpec basic_spec() {
    SpmVocabSpec s;
    auto add = [&](std::string t, float sc, std::int32_t ty) {
        s.tokens.push_back(std::move(t));
        s.scores.push_back(sc);
        s.types.push_back(ty);
    };
    add("<unk>", 0.0f, 2);
    add("<s>",   0.0f, 3);
    add("</s>",  0.0f, 3);
    for (int b = 0; b < 256; ++b) {
        char name[8];
        std::snprintf(name, sizeof(name), "<0x%02X>", b);
        add(name, 0.0f, 6);
    }
    add(std::string{kSp},            -1.0f, 1);
    add("h",                         -5.0f, 1);
    add("i",                         -5.0f, 1);
    add("t",                         -5.0f, 1);
    add("e",                         -5.0f, 1);
    add("r",                         -5.0f, 1);
    // SentencePiece only ever glues two adjacent symbols, so a word is
    // reachable only if every intermediate prefix is itself a piece.
    add(std::string{kSp} + "h",      -3.0f, 1);
    add(std::string{kSp} + "hi",     -2.0f, 1);   // best merge for "▁hi"
    add("th",                        -4.0f, 1);
    add("the",                       -3.5f, 1);
    add(std::string{kSp} + "t",      -4.5f, 1);
    add(std::string{kSp} + "th",     -4.0f, 1);
    add(std::string{kSp} + "the",    -3.0f, 1);
    add(std::string{kSp} + "ther",   -2.5f, 1);
    add(std::string{kSp} + "there",  -1.5f, 1);
    add("er",                        -4.0f, 1);
    add("re",                        -4.0f, 1);
    return s;
}

} // namespace

TEST_CASE("spm: prefers the highest-scoring merge") {
    auto tok = build(basic_spec());
    REQUIRE(tok->kind() == BpeTokenizer::Kind::Spm);

    // "▁hi" (-2.0) beats gluing "▁h"(-3.0) then "i", and both beat singles.
    auto ids = tok->encode("hi");
    REQUIRE(ids.size() == 1);
    CHECK(tok->token_str(ids[0]) == std::string{kSp} + "hi");
}

TEST_CASE("spm: round-trips text through encode/decode") {
    auto tok = build(basic_spec());
    const std::string text = "hi there";
    auto ids = tok->encode(text);
    CHECK(!ids.empty());
    // SentencePiece prepends a word-boundary marker, so decoding yields the
    // original text with one leading space.
    CHECK(tok->decode(ids) == " " + text);
}

TEST_CASE("spm: spaces become U+2581 word boundaries") {
    auto tok = build(basic_spec());
    auto ids = tok->encode("hi there");
    // Exactly two pieces: "▁hi" and "▁there", both whole-word merges.
    REQUIRE(ids.size() == 2);
    CHECK(tok->token_str(ids[0]) == std::string{kSp} + "hi");
    CHECK(tok->token_str(ids[1]) == std::string{kSp} + "there");
}

TEST_CASE("spm: unknown characters fall back to byte tokens") {
    auto tok = build(basic_spec());
    // 'z' has no piece in this vocab, so it must decompose to <0x7A>.
    auto ids = tok->encode("z");
    bool saw_byte = false;
    for (auto id : ids) {
        if (tok->token_str(id) == "<0x7A>") saw_byte = true;
    }
    CHECK(saw_byte);
    CHECK(tok->decode(ids) == " z");
}

TEST_CASE("spm: multi-byte UTF-8 survives byte fallback") {
    auto tok = build(basic_spec());
    // 'é' is U+00E9 => C3 A9; neither byte has a piece, so both fall back.
    const std::string text = "\xC3\xA9";
    auto ids = tok->encode(text);
    REQUIRE(ids.size() >= 2);
    CHECK(tok->decode(ids) == " " + text);
}

TEST_CASE("spm: control tokens are hidden on decode unless asked for") {
    auto tok = build(basic_spec());
    const TokenId bos = tok->id_of("<s>");
    REQUIRE(bos >= 0);

    std::vector<TokenId> ids{bos};
    auto hi = tok->encode("hi");
    ids.insert(ids.end(), hi.begin(), hi.end());

    CHECK(tok->decode(ids, /*skip_special=*/true)  == " hi");
    CHECK(tok->decode(ids, /*skip_special=*/false) == "<s> hi");
}

TEST_CASE("spm: a vocab without scores is rejected, not silently mis-tokenized") {
    MetadataStore md;
    md.insert("tokenizer.ggml.model",  MetadataValue{std::string{"llama"}});
    md.insert("tokenizer.ggml.tokens",
              MetadataValue{std::vector<std::string>{"<unk>", "a"}});

    SpmFakeModel model{std::move(md), {}};
    auto r = BpeTokenizer::load(model);
    CHECK_FALSE(static_cast<bool>(r));
}
