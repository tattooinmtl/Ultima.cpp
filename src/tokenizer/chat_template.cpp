#include "ultima/tokenizer/chat_template.hpp"

namespace ultima::tokenizer {

namespace {

// ChatML shape used by Qwen2 / Qwen2.5-Coder / Qwen3 / Qwen3-Coder.
//   <|im_start|>{role}\n{content}<|im_end|>\n
// with an optional trailing "<|im_start|>assistant\n" when the caller wants
// the model to continue in the assistant slot.
std::string render_chatml_body(const std::vector<ChatMessage>& msgs,
                               bool add_generation_prompt) {
    std::string out;
    out.reserve(64 + msgs.size() * 64);
    for (const auto& m : msgs) {
        out += "<|im_start|>";
        out += m.role;
        // "tool" role gets an extra "name=..." metadata line above the body.
        if (m.role == "tool" && !m.name.empty()) {
            out += " name=";
            out += m.name;
        }
        out.push_back('\n');
        out += m.content;
        out += "<|im_end|>\n";
    }
    if (add_generation_prompt) {
        out += "<|im_start|>assistant\n";
    }
    return out;
}

} // namespace

std::string render_qwen2_chatml(const std::vector<ChatMessage>& msgs,
                                bool add_generation_prompt) {
    return render_chatml_body(msgs, add_generation_prompt);
}

std::string render_qwen3_chatml(const std::vector<ChatMessage>& msgs,
                                bool add_generation_prompt) {
    return render_chatml_body(msgs, add_generation_prompt);
}

std::string render_chatml(std::string_view arch,
                          const std::vector<ChatMessage>& msgs,
                          bool add_generation_prompt) {
    if (arch == "qwen3") return render_qwen3_chatml(msgs, add_generation_prompt);
    // Qwen2 renderer is the safe default for anything else (including unknown
    // architectures) so we never fail to produce a prompt string.
    return render_qwen2_chatml(msgs, add_generation_prompt);
}

} // namespace ultima::tokenizer
