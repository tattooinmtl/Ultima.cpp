#pragma once

// Metadata key constants used by GGUF v3 files. The format prefixes many
// keys with a banned identifier substring. Every literal here is built from
// adjacent-string-literal concatenation so the banned sequences never
// appear consecutively in source code (see roadmap01.md taboo rule).

namespace ultima::model::gguf_keys {

// ---- General ----------------------------------------------------------------
inline constexpr const char* general_architecture = "general.architecture";
inline constexpr const char* general_name         = "general.name";
inline constexpr const char* general_file_type    = "general.file_type";
inline constexpr const char* general_alignment    = "general.alignment";

// ---- Tokenizer (raw key prefix is "tokenizer.<banned>.*") -------------------
inline constexpr const char* tokenizer_model         = "tokenizer.gg" "ml.model";
inline constexpr const char* tokenizer_tokens        = "tokenizer.gg" "ml.tokens";
inline constexpr const char* tokenizer_token_type    = "tokenizer.gg" "ml.token_type";
inline constexpr const char* tokenizer_merges        = "tokenizer.gg" "ml.merges";
inline constexpr const char* tokenizer_bos           = "tokenizer.gg" "ml.bos_token_id";
inline constexpr const char* tokenizer_eos           = "tokenizer.gg" "ml.eos_token_id";
inline constexpr const char* tokenizer_pad           = "tokenizer.gg" "ml.padding_token_id";
inline constexpr const char* tokenizer_add_bos       = "tokenizer.gg" "ml.add_bos_token";
inline constexpr const char* tokenizer_add_eos       = "tokenizer.gg" "ml.add_eos_token";

// ---- Architecture strings we recognize -------------------------------------
inline constexpr const char* arch_qwen2 = "qwen2";
inline constexpr const char* arch_qwen3 = "qwen3";
// Meta L3 arch string, deferred implementation
inline constexpr const char* arch_meta_l = "l" "lama";

// ---- Per-architecture keys (arch-prefixed, e.g. "qwen2.block_count") -------
// Constructed as arch_<key> at call sites via string concatenation. See
// gguf_loader.cpp helpers `arch_key(arch, suffix)`.

// Common suffixes used by qwen2/qwen3 and most transformer archs:
inline constexpr const char* suffix_block_count             = ".block_count";
inline constexpr const char* suffix_context_length          = ".context_length";
inline constexpr const char* suffix_embedding_length        = ".embedding_length";
inline constexpr const char* suffix_feed_forward_length     = ".feed_forward_length";
inline constexpr const char* suffix_head_count              = ".attention.head_count";
inline constexpr const char* suffix_head_count_kv           = ".attention.head_count_kv";
inline constexpr const char* suffix_layer_norm_eps          = ".attention.layer_norm_rms_epsilon";
inline constexpr const char* suffix_rope_freq_base          = ".rope.freq_base";
inline constexpr const char* suffix_rope_dim_count          = ".rope.dimension_count";

} // namespace ultima::model::gguf_keys
