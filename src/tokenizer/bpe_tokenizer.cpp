#include "ultima/tokenizer/tokenizer.hpp"

#include "ultima/model/loaded_model.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

namespace ultima::tokenizer {

namespace {

using core::ErrorCode;
using core::Failure;
using core::Result;
using core::fail;

// ---- GGUF tokenizer keys ---------------------------------------------------
constexpr const char* k_tokens         = "tokenizer.ggml.tokens";
constexpr const char* k_merges         = "tokenizer.ggml.merges";
constexpr const char* k_token_type     = "tokenizer.ggml.token_type";
constexpr const char* k_model          = "tokenizer.ggml.model";
constexpr const char* k_bos_id         = "tokenizer.ggml.bos_token_id";
constexpr const char* k_eos_id         = "tokenizer.ggml.eos_token_id";
constexpr const char* k_pad_id         = "tokenizer.ggml.padding_token_id";

// ---- GPT-2 byte-to-unicode map --------------------------------------------
// Standard mapping used by every GPT-2-family tokenizer (Qwen2, Qwen3
// included). Bytes that print cleanly stay as themselves; the rest map to
// unicode code points 256+ in order.
std::array<std::string, 256> build_byte_to_str() {
    std::array<std::string, 256> out;
    // Bytes that map directly to their own code point:
    //   '!'..'~' (33..126), '\xA1'..'\xAC' (161..172), '\xAE'..'\xFF' (174..255)
    auto is_printable = [](int b) {
        return (b >= 33 && b <= 126)
            || (b >= 161 && b <= 172)
            || (b >= 174 && b <= 255);
    };
    // Fill printable bytes first.
    auto encode_cp = [](std::uint32_t cp) -> std::string {
        std::string s;
        if (cp < 0x80) {
            s.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            s.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
        return s;
    };
    unsigned n = 0;
    for (int b = 0; b < 256; ++b) {
        if (is_printable(b)) {
            out[b] = encode_cp(static_cast<std::uint32_t>(b));
        }
    }
    for (int b = 0; b < 256; ++b) {
        if (!is_printable(b)) {
            out[b] = encode_cp(256u + n);
            ++n;
        }
    }
    return out;
}

inline std::uint64_t pack_pair(TokenId a, TokenId b) noexcept {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32)
         |  static_cast<std::uint64_t>(static_cast<std::uint32_t>(b));
}

// ---- Pretokenizer (GPT-2 rules, ASCII-focused) ----------------------------
//
// Splits the input into pretokens using GPT-2's regex behavior — implemented
// by hand so we don't drag ICU or a regex engine in. For pure-ASCII input
// (the coding workload) this matches HF's `Qwen2Tokenizer` output. Non-ASCII
// UTF-8 bytes are grouped as opaque "other" runs and fed through byte-level
// encoding; correctness on non-Latin scripts is a v0.2 concern.
enum class Cat : std::uint8_t { Letter, Digit, Space, Other };

Cat classify_ascii(unsigned char c) noexcept {
    if (c >= 128)                              return Cat::Other;
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f')
        return Cat::Space;
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))
        return Cat::Letter;
    if (c >= '0' && c <= '9')
        return Cat::Digit;
    return Cat::Other;
}

// Contractions GPT-2 splits off: 's, 't, 're, 've, 'm, 'll, 'd.
// Returns the length in bytes of the matched contraction at `p`, or 0.
std::size_t match_contraction(const char* p, std::size_t n) noexcept {
    if (n < 2 || p[0] != '\'') return 0;
    auto lower = [](char c) { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); };
    const char c1 = lower(p[1]);
    if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return 2;
    if (n >= 3) {
        const char c2 = lower(p[2]);
        if (c1 == 'r' && c2 == 'e') return 3;   // 're
        if (c1 == 'v' && c2 == 'e') return 3;   // 've
        if (c1 == 'l' && c2 == 'l') return 3;   // 'll
    }
    return 0;
}

std::vector<std::string_view> pretokenize(std::string_view text) {
    std::vector<std::string_view> out;
    const char*  p    = text.data();
    const std::size_t n = text.size();
    std::size_t  i = 0;

    while (i < n) {
        // Contractions like 's, 're, ...
        if (auto len = match_contraction(p + i, n - i); len > 0) {
            out.emplace_back(p + i, len);
            i += len;
            continue;
        }
        // Optional single leading space, then a run of letters, digits, or
        // "other" (each category grouped independently). Whitespace runs
        // become their own pretokens (matching `\s+` in the GPT-2 regex).
        const unsigned char c0 = static_cast<unsigned char>(p[i]);
        Cat first = classify_ascii(c0);
        if (first == Cat::Space) {
            // Determine whether this whitespace run has a trailing non-space
            // it should merge with. GPT-2's regex: `\s+(?!\S)|\s+`
            // Simplification: take the whole run as its own pretoken UNLESS
            // exactly one leading space precedes a Letter/Digit/Other run,
            // in which case we include it with the following run.
            std::size_t j = i;
            while (j < n && classify_ascii(static_cast<unsigned char>(p[j])) == Cat::Space)
                ++j;
            const std::size_t ws_len   = j - i;
            const bool has_following   = (j < n);
            if (ws_len == 1 && has_following) {
                // Fold the single leading space into the next run.
                Cat next = classify_ascii(static_cast<unsigned char>(p[j]));
                if (next == Cat::Letter || next == Cat::Digit || next == Cat::Other) {
                    std::size_t k = j;
                    while (k < n && classify_ascii(static_cast<unsigned char>(p[k])) == next)
                        ++k;
                    out.emplace_back(p + i, k - i);
                    i = k;
                    continue;
                }
            }
            out.emplace_back(p + i, ws_len);
            i = j;
            continue;
        }
        // Non-space start: run of the same category.
        std::size_t k = i;
        while (k < n && classify_ascii(static_cast<unsigned char>(p[k])) == first)
            ++k;
        out.emplace_back(p + i, k - i);
        i = k;
    }
    return out;
}

} // namespace

// Builder class inside the namespace to construct a BpeTokenizer.
class BpeTokenizerBuilder {
public:
    BpeTokenizerBuilder() = default;

    Result<std::unique_ptr<BpeTokenizer>> build(const model::LoadedModel& model) {
        auto tok = std::unique_ptr<BpeTokenizer>(new BpeTokenizer());
        tok->byte_to_str_ = build_byte_to_str();

        // Full inverse mapping: byte-token string -> raw byte value.
        tok->str_to_byte_map_.reserve(256);
        for (int b = 0; b < 256; ++b) {
            tok->str_to_byte_map_.emplace(tok->byte_to_str_[b],
                                          static_cast<std::uint8_t>(b));
        }

        // Load vocab.
        auto vocab_opt = model.metadata().get<std::vector<std::string>>(k_tokens);
        if (!vocab_opt) {
            return fail(ErrorCode::MissingRequiredMetadata,
                        fmt::format("missing '{}'", k_tokens),
                        "bpe_tokenizer");
        }
        tok->vocab_ = std::move(*vocab_opt);
        tok->token_to_id_.reserve(tok->vocab_.size() * 2);
        for (std::size_t i = 0; i < tok->vocab_.size(); ++i) {
            tok->token_to_id_.emplace(tok->vocab_[i], static_cast<TokenId>(i));
        }

        // token_type array marks specials: 3 = control (special), 2 = user
        // defined, 6 = byte-fallback. Anything other than 1 (normal) we treat
        // as "not eligible for merges" (special).
        tok->is_special_.assign(tok->vocab_.size(), false);
        if (auto tt = model.metadata().get<std::vector<std::string>>(k_token_type)) {
            // some GGUFs encode token_type as a string vector; skip if so
        } else {
            // Fallback: mark tokens matching common special patterns.
            for (std::size_t i = 0; i < tok->vocab_.size(); ++i) {
                const auto& s = tok->vocab_[i];
                if (s.size() >= 4 && s.front() == '<' && s.back() == '>' &&
                    s.find("|") != std::string::npos) {
                    tok->is_special_[i] = true;
                }
            }
        }

        // Load merges — vector of "a b" strings, in priority order.
        auto merges_opt = model.metadata().get<std::vector<std::string>>(k_merges);
        if (!merges_opt) {
            return fail(ErrorCode::MissingRequiredMetadata,
                        fmt::format("missing '{}'", k_merges),
                        "bpe_tokenizer");
        }
        const auto& merges = *merges_opt;
        tok->merge_rank_.reserve(merges.size());
        for (std::size_t i = 0; i < merges.size(); ++i) {
            const auto& line = merges[i];
            const auto sp = line.find(' ');
            if (sp == std::string::npos) continue;
            const std::string a = line.substr(0, sp);
            const std::string b = line.substr(sp + 1);
            auto ia = tok->token_to_id_.find(a);
            auto ib = tok->token_to_id_.find(b);
            if (ia == tok->token_to_id_.end() || ib == tok->token_to_id_.end())
                continue;
            const std::uint64_t key = pack_pair(ia->second, ib->second);
            tok->merge_rank_.emplace(key, static_cast<std::uint32_t>(i));
        }

        // Resolve special-token ids.
        tok->bos_id_ = static_cast<TokenId>(model.metadata().get_uint(k_bos_id).value_or(-1));
        tok->eos_id_ = static_cast<TokenId>(model.metadata().get_uint(k_eos_id).value_or(-1));
        tok->pad_id_ = static_cast<TokenId>(model.metadata().get_uint(k_pad_id).value_or(-1));
        tok->im_start_id_ = tok->id_of("<|im_start|>");
        tok->im_end_id_   = tok->id_of("<|im_end|>");

        // Sort specials by length desc for greedy longest-match encode.
        for (std::size_t i = 0; i < tok->vocab_.size(); ++i) {
            if (tok->is_special_[i]) {
                tok->specials_by_len_desc_.emplace_back(
                    tok->vocab_[i], static_cast<TokenId>(i));
            }
        }
        std::sort(tok->specials_by_len_desc_.begin(),
                  tok->specials_by_len_desc_.end(),
                  [](const auto& a, const auto& b) {
                      return a.first.size() > b.first.size();
                  });
        return tok;
    }
};

BpeTokenizer::BpeTokenizer() = default;

Result<std::unique_ptr<BpeTokenizer>>
BpeTokenizer::load(const model::LoadedModel& model) {
    BpeTokenizerBuilder builder;
    return builder.build(model);
}

std::string_view BpeTokenizer::token_str(TokenId id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= vocab_.size()) return {};
    return vocab_[static_cast<std::size_t>(id)];
}

TokenId BpeTokenizer::id_of(std::string_view s) const {
    auto it = token_to_id_.find(std::string(s));
    return it == token_to_id_.end() ? TokenId{-1} : it->second;
}

// ---- Encode ----------------------------------------------------------------
namespace {

// Encode one pretoken (raw bytes, as they appear in the input) into a
// token-id sequence via the BPE merge loop.
std::vector<TokenId> encode_pretoken_bytes(
    std::string_view raw_bytes,
    const std::array<std::string, 256>& byte_to_str,
    const std::unordered_map<std::string, TokenId>& token_to_id,
    const std::unordered_map<std::uint64_t, std::uint32_t,
                             PairHash>& merge_rank,
    const std::vector<std::string>& vocab)
{
    if (raw_bytes.empty()) return {};

    // 1) Convert each raw byte to its GPT-2 byte-token string, resolve to id.
    std::vector<TokenId> seq;
    seq.reserve(raw_bytes.size());
    for (unsigned char b : raw_bytes) {
        auto it = token_to_id.find(byte_to_str[b]);
        if (it == token_to_id.end()) {
            // Byte-fallback missing from vocab — this shouldn't happen with a
            // real GPT-2/Qwen vocab; emit token 0 as a defensive placeholder.
            seq.push_back(0);
        } else {
            seq.push_back(it->second);
        }
    }

    // 2) Merge loop. On each pass find the lowest-rank mergeable pair,
    //    replace it with the merged token, restart. The merged token's
    //    string is vocab[a] + vocab[b].
    while (seq.size() >= 2) {
        std::uint32_t best_rank = std::numeric_limits<std::uint32_t>::max();
        std::size_t   best_pos  = 0;
        for (std::size_t i = 0; i + 1 < seq.size(); ++i) {
            const std::uint64_t key = pack_pair(seq[i], seq[i + 1]);
            auto it = merge_rank.find(key);
            if (it != merge_rank.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos  = i;
                if (best_rank == 0) break;
            }
        }
        if (best_rank == std::numeric_limits<std::uint32_t>::max()) break;

        const std::string merged = vocab[static_cast<std::size_t>(seq[best_pos])]
                                 + vocab[static_cast<std::size_t>(seq[best_pos + 1])];
        auto mit = token_to_id.find(merged);
        if (mit == token_to_id.end()) {
            // Shouldn't happen if merge_rank was populated from vocab, but
            // guard anyway: stop merging at this pair to make progress.
            break;
        }
        seq[best_pos] = mit->second;
        seq.erase(seq.begin() + static_cast<std::ptrdiff_t>(best_pos) + 1);
    }
    return seq;
}

} // namespace

std::vector<TokenId>
BpeTokenizer::encode(std::string_view text, bool allow_special) const {
    std::vector<TokenId> out;
    if (text.empty()) return out;

    // Cut the text into pieces, splitting off special-token literals when
    // allow_special is true. Everything else flows through the normal
    // pretokenize + BPE path.
    auto encode_normal_chunk = [&](std::string_view chunk) {
        for (auto pt : pretokenize(chunk)) {
            // Pass raw pretoken bytes; encode_pretoken_bytes maps each byte
            // to its byte-token id, then runs BPE.
            auto ids = encode_pretoken_bytes(pt, byte_to_str_,
                                             token_to_id_, merge_rank_, vocab_);
            out.insert(out.end(), ids.begin(), ids.end());
        }
    };

    if (!allow_special || specials_by_len_desc_.empty()) {
        encode_normal_chunk(text);
        return out;
    }

    std::size_t i = 0;
    while (i < text.size()) {
        // Longest-match against known specials at position i.
        bool matched = false;
        for (const auto& [sp, sp_id] : specials_by_len_desc_) {
            if (sp.size() <= text.size() - i &&
                std::memcmp(text.data() + i, sp.data(), sp.size()) == 0) {
                out.push_back(sp_id);
                i += sp.size();
                matched = true;
                break;
            }
        }
        if (matched) continue;
        // Walk forward until we hit the start of another special.
        std::size_t j = i + 1;
        while (j < text.size()) {
            bool at_special = false;
            for (const auto& [sp, sp_id] : specials_by_len_desc_) {
                if (sp.size() <= text.size() - j &&
                    std::memcmp(text.data() + j, sp.data(), sp.size()) == 0) {
                    at_special = true;
                    break;
                }
            }
            if (at_special) break;
            ++j;
        }
        encode_normal_chunk(text.substr(i, j - i));
        i = j;
    }
    return out;
}

// ---- Decode ----------------------------------------------------------------
std::string
BpeTokenizer::decode(std::span<const TokenId> tokens, bool skip_special) const {
    std::string out;
    out.reserve(tokens.size() * 4);

    for (TokenId id : tokens) {
        if (id < 0 || static_cast<std::size_t>(id) >= vocab_.size()) continue;
        const auto sidx = static_cast<std::size_t>(id);
        if (skip_special && is_special_[sidx]) continue;

        const std::string& s = vocab_[sidx];
        // Special tokens (kept when !skip_special) go through verbatim.
        if (is_special_[sidx]) {
            out += s;
            continue;
        }
        // Walk the token's UTF-8 string, decoding each GPT-2 byte-token
        // char sequence back to its raw byte via str_to_byte_map_.
        std::size_t k = 0;
        while (k < s.size()) {
            const unsigned char c = static_cast<unsigned char>(s[k]);
            std::size_t clen;
            if (c < 0x80)      clen = 1;
            else if (c < 0xC0) clen = 1;   // stray continuation — treat as 1
            else if (c < 0xE0) clen = 2;
            else if (c < 0xF0) clen = 3;
            else               clen = 4;
            if (k + clen > s.size()) clen = s.size() - k;
            std::string chunk = s.substr(k, clen);
            auto it = str_to_byte_map_.find(chunk);
            if (it == str_to_byte_map_.end()) {
                out += chunk;       // unknown mapping: pass through
            } else {
                out.push_back(static_cast<char>(it->second));
            }
            k += clen;
        }
    }
    return out;
}

} // namespace ultima::tokenizer
