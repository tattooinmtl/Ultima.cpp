#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace ultima::tokenizer {

struct ChatMessage {
    std::string role;      // "system" | "user" | "assistant" | "tool"
    std::string content;
    std::string name;      // tool name for role="tool"; empty otherwise
};

// Render a message list into the ChatML text a Qwen2/Qwen3 model expects.
// The result string is *user text* — pass it through BpeTokenizer::encode
// with allow_special=true so the <|im_start|>, <|im_end|> markers get their
// special ids instead of being encoded as literal characters.
//
// `add_generation_prompt = true` appends "<|im_start|>assistant\n" at the end
// so the model continues in the assistant slot.

std::string render_qwen2_chatml(const std::vector<ChatMessage>& msgs,
                                bool add_generation_prompt = true);

// Same shape; Qwen3 uses the same ChatML template in v0.1 (thinking-tag
// support is a per-request UX toggle wired later, not a template change).
std::string render_qwen3_chatml(const std::vector<ChatMessage>& msgs,
                                bool add_generation_prompt = true);

// Dispatch by architecture string ("qwen2" or "qwen3"); falls through to the
// Qwen2 renderer for any unknown arch with a fallback warning-worthy behavior.
std::string render_chatml(std::string_view arch,
                          const std::vector<ChatMessage>& msgs,
                          bool add_generation_prompt = true);

} // namespace ultima::tokenizer
