#pragma once

#include "ultima/core/error.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ultima::model { class LoadedModel; }

namespace ultima::tokenizer {

using TokenId = std::int32_t;

// Public so both the tokenizer implementation TU and free helpers can name
// the same hash for the (a, b) merge-rank map keyed on packed token-pair ids.
struct PairHash {
    std::size_t operator()(std::uint64_t k) const noexcept { return k ^ (k >> 32); }
};

// Byte-level BPE tokenizer loaded from a GGUF file's tokenizer.ggml.* keys.
// See Decision 07. The model is loaded once at construction; encode/decode
// are thread-safe (immutable state; per-call scratch is stack-local).
class BpeTokenizer {
public:
    static core::Result<std::unique_ptr<BpeTokenizer>>
    load(const model::LoadedModel& model);

    // Encode UTF-8 text to token ids.
    //   allow_special = true  : special-token strings inside `text` are
    //                           parsed as their special ids (used by template
    //                           renderers).
    //   allow_special = false : special-token strings are treated as literal
    //                           user text (default for user-provided input).
    std::vector<TokenId> encode(std::string_view text,
                                bool allow_special = false) const;

    // Decode ids back to UTF-8. When skip_special is true, special-token
    // ids emit nothing (chat-visible output).
    std::string decode(std::span<const TokenId> tokens,
                       bool skip_special = true) const;

    // Vocabulary access.
    std::size_t      vocab_size()      const noexcept { return vocab_.size(); }
    std::string_view token_str(TokenId id) const;

    // Special tokens. Return -1 if the underlying metadata was absent.
    TokenId bos_id()       const noexcept { return bos_id_; }
    TokenId eos_id()       const noexcept { return eos_id_; }
    TokenId pad_id()       const noexcept { return pad_id_; }
    TokenId im_start_id()  const noexcept { return im_start_id_; }
    TokenId im_end_id()    const noexcept { return im_end_id_; }

    // Look up a token id by exact string match (before pretokenization).
    // Returns -1 on miss. Used by template renderers to place role markers.
    TokenId id_of(std::string_view token_str) const;

    BpeTokenizer(const BpeTokenizer&)            = delete;
    BpeTokenizer& operator=(const BpeTokenizer&) = delete;
    BpeTokenizer(BpeTokenizer&&)                 = delete;
    BpeTokenizer& operator=(BpeTokenizer&&)      = delete;

private:
    BpeTokenizer();
    friend class BpeTokenizerBuilder;

    // Byte-level GPT-2 mapping: byte value 0..255 -> unicode char used in the
    // vocab strings. Constructed once at load.
    std::array<std::string, 256>                  byte_to_str_;
    std::unordered_map<std::string, std::uint8_t> str_to_byte_map_;

    // Maps a two-token merge (a, b) to its rank (lower = higher priority).
    std::unordered_map<std::uint64_t, std::uint32_t, PairHash> merge_rank_;

    std::vector<std::string>                     vocab_;
    std::unordered_map<std::string, TokenId>     token_to_id_;
    std::vector<bool>                            is_special_;

    // Special-token literal strings (for allow_special encode).
    std::vector<std::pair<std::string, TokenId>> specials_by_len_desc_;

    TokenId bos_id_      = -1;
    TokenId eos_id_      = -1;
    TokenId pad_id_      = -1;
    TokenId im_start_id_ = -1;
    TokenId im_end_id_   = -1;
};

} // namespace ultima::tokenizer
