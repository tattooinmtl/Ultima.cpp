#include <doctest/doctest.h>

#include "ultima/tokenizer/chat_template.hpp"
#include "ultima/tokenizer/tokenizer.hpp"

#include "ultima/model/loaded_model.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace {

// Synthetic in-memory LoadedModel that returns a tiny hand-built vocab, just
// enough to exercise the BPE algorithm end-to-end without needing a real
// GGUF file. Real-vocab correctness is a smoke-test we run against a
// downloaded Qwen2.5-Coder GGUF at M6 wiring — see roadmap Decision 07 §7.7.
class FakeModel : public ultima::model::LoadedModel {
public:
    explicit FakeModel(ultima::model::MetadataStore md,
                       std::vector<ultima::model::TensorInfo> ti = {})
        : arch_{"qwen2"}, meta_{std::move(md)}, tensors_{std::move(ti)} {}

    const std::string&                          architecture()  const override { return arch_; }
    const ultima::model::MetadataStore&         metadata()      const override { return meta_; }
    const std::vector<ultima::model::TensorInfo>& tensor_infos() const override { return tensors_; }
    std::optional<ultima::model::TensorView>    tensor(std::string_view) const override { return std::nullopt; }

    std::uint64_t file_size_bytes() const override { return 0; }
    std::uint64_t alignment()       const override { return 32; }
    std::uint32_t gguf_version()    const override { return 3; }

private:
    std::string                                   arch_;
    ultima::model::MetadataStore                  meta_;
    std::vector<ultima::model::TensorInfo>        tensors_;
};

// The 256 byte-token strings, in the same order the tokenizer builds them:
// printable ASCII bytes first as themselves, then non-printable bytes remap
// to U+0100 onward. To build a valid vocab we just have to include the exact
// same strings; the tokenizer's inverse map will resolve them.
//
// For test simplicity we build a helper that returns the tokenizer's own
// byte-to-str mapping by calling it via a throwaway load.
std::string cp_to_utf8(std::uint32_t cp) {
    std::string s;
    if (cp < 0x80)          s.push_back(static_cast<char>(cp));
    else if (cp < 0x800)   { s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                              s.push_back(static_cast<char>(0x80 | (cp & 0x3F))); }
    else if (cp < 0x10000) { s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                              s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                              s.push_back(static_cast<char>(0x80 | (cp & 0x3F))); }
    else                   { s.push_back(static_cast<char>(0xF0 | (cp >> 18)));
                              s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
                              s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                              s.push_back(static_cast<char>(0x80 | (cp & 0x3F))); }
    return s;
}
std::vector<std::string> byte_token_strings() {
    std::vector<std::string> v(256);
    auto is_printable = [](int b) {
        return (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
    };
    unsigned n = 0;
    for (int b = 0; b < 256; ++b) {
        if (is_printable(b)) v[b] = cp_to_utf8(static_cast<std::uint32_t>(b));
    }
    for (int b = 0; b < 256; ++b) {
        if (!is_printable(b)) v[b] = cp_to_utf8(256u + n++);
    }
    return v;
}

std::unique_ptr<ultima::model::LoadedModel>
build_fake_model(const std::vector<std::string>& extra_tokens,
                 const std::vector<std::string>& merges,
                 int im_start_id_hint = -1,
                 int im_end_id_hint   = -1)
{
    (void)im_start_id_hint;
    (void)im_end_id_hint;

    // Vocab layout: 256 byte tokens, then extras (each of which may include
    // specials like "<|im_start|>").
    std::vector<std::string> vocab = byte_token_strings();
    vocab.insert(vocab.end(), extra_tokens.begin(), extra_tokens.end());

    ultima::model::MetadataStore md;
    md.insert("general.architecture", ultima::model::MetadataValue{std::string{"qwen2"}});
    md.insert("tokenizer.ggml.model", ultima::model::MetadataValue{std::string{"gpt2"}});
    md.insert("tokenizer.ggml.tokens", ultima::model::MetadataValue{vocab});
    md.insert("tokenizer.ggml.merges", ultima::model::MetadataValue{merges});
    return std::make_unique<FakeModel>(std::move(md));
}

} // namespace

TEST_CASE("tokenizer: loads vocab and resolves basic byte tokens") {
    auto model = build_fake_model({}, {});
    auto tok_r = ultima::tokenizer::BpeTokenizer::load(*model);
    REQUIRE(tok_r.has_value());
    auto& tok = **tok_r;
    CHECK(tok.vocab_size() == 256);
    // ASCII 'a' is byte 0x61, and byte 0x61 == printable, so its string is "a".
    CHECK(tok.id_of("a") == 'a');
}

TEST_CASE("tokenizer: encode+decode round-trip on ASCII (no merges)") {
    auto model = build_fake_model({}, {});
    auto tok = std::move(*ultima::tokenizer::BpeTokenizer::load(*model));

    const std::string text = "hello world 123";
    auto ids = tok->encode(text);
    CHECK(!ids.empty());
    const std::string back = tok->decode(ids, /*skip_special=*/false);
    CHECK(back == text);
}

TEST_CASE("tokenizer: BPE merges reduce sequence length") {
    // Add a merged token "he" (id 256) and a merge rule "h e".
    auto model = build_fake_model({"he"}, {"h e"});
    auto tok = std::move(*ultima::tokenizer::BpeTokenizer::load(*model));

    auto ids = tok->encode("he");
    REQUIRE(ids.size() == 1);
    CHECK(ids[0] == 256);
    CHECK(tok->decode(ids, false) == "he");
}

TEST_CASE("tokenizer: special-token allow_special path emits the id directly") {
    auto model = build_fake_model({"<|im_start|>", "<|im_end|>"},
                                  {});
    auto tok = std::move(*ultima::tokenizer::BpeTokenizer::load(*model));

    // With allow_special = true, the literal string maps to a single id.
    auto ids = tok->encode("<|im_start|>", /*allow_special=*/true);
    REQUIRE(ids.size() == 1);
    CHECK(ids[0] == tok->im_start_id());

    // With allow_special = false, the same string encodes byte-by-byte
    // through the normal path, so it's several tokens.
    auto ids_lit = tok->encode("<|im_start|>", /*allow_special=*/false);
    CHECK(ids_lit.size() > 1);
}

TEST_CASE("tokenizer: skip_special decode drops specials from output") {
    auto model = build_fake_model({"<|im_start|>", "<|im_end|>"},
                                  {});
    auto tok = std::move(*ultima::tokenizer::BpeTokenizer::load(*model));

    std::vector<ultima::tokenizer::TokenId> ids = {
        tok->im_start_id(), 'h', 'i', tok->im_end_id()
    };
    CHECK(tok->decode(ids, /*skip_special=*/true)  == "hi");
    CHECK(tok->decode(ids, /*skip_special=*/false) == "<|im_start|>hi<|im_end|>");
}

TEST_CASE("chat_template: qwen2 ChatML renders roles and generation prompt") {
    using ultima::tokenizer::ChatMessage;
    std::vector<ChatMessage> msgs = {
        {"system", "You are helpful.", ""},
        {"user",   "Hi there",         ""},
    };
    auto s = ultima::tokenizer::render_qwen2_chatml(msgs, /*add_generation_prompt=*/true);
    CHECK(s ==
        "<|im_start|>system\nYou are helpful.<|im_end|>\n"
        "<|im_start|>user\nHi there<|im_end|>\n"
        "<|im_start|>assistant\n");
}

TEST_CASE("chat_template: tool role emits the name= metadata") {
    using ultima::tokenizer::ChatMessage;
    std::vector<ChatMessage> msgs = {
        {"tool", "42", "calc"},
    };
    auto s = ultima::tokenizer::render_qwen2_chatml(msgs, /*add_generation_prompt=*/false);
    CHECK(s == "<|im_start|>tool name=calc\n42<|im_end|>\n");
}

TEST_CASE("chat_template: dispatcher routes qwen3 correctly") {
    using ultima::tokenizer::ChatMessage;
    std::vector<ChatMessage> msgs = { {"user", "hi", ""} };
    auto s2 = ultima::tokenizer::render_chatml("qwen2", msgs, false);
    auto s3 = ultima::tokenizer::render_chatml("qwen3", msgs, false);
    // v0.1 renderers are identical; both should produce the same output.
    CHECK(s2 == s3);
    CHECK(s2 == "<|im_start|>user\nhi<|im_end|>\n");
}
