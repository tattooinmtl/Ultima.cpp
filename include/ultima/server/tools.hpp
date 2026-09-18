#pragma once

#include "ultima/server/workspace.hpp"

#include <nlohmann/json.hpp>

#include <string>
#include <string_view>
#include <vector>

namespace ultima::server {

// The file tools a coding session can call (Decision 18 §18.3).
//
// Six operations, deliberately no more. A small local model given twenty tools
// picks the wrong one; given six it mostly picks right.

struct ToolCall {
    std::string    name;
    nlohmann::json args = nlohmann::json::object();
    std::string    raw;      // the text this was parsed out of, for the trace
};

struct ToolResult {
    std::string name;
    bool        ok = false;
    // What goes back to the model. On failure this is the error text verbatim —
    // a small model can often recover from "no match for the text to replace"
    // but not from a generic failure.
    std::string content;
    // Set when the call changed a file, so the caller can push an editor event.
    std::string changed_path;
    std::string tab_label;
};

// JSON schemas for every tool, in the shape Qwen's chat template expects.
const nlohmann::json& tool_schemas();

// The `# Tools` section appended to the system prompt: the schemas plus the
// <tool_call> protocol the model is trained to emit.
std::string render_tools_system_prompt();

// Pull tool calls out of generated text.
//
// Accepts, in order of preference: the <tool_call>{...}</tool_call> wrapper
// Qwen is trained on, a ```json fenced block, and a bare top-level object with
// `name` and `arguments`. Small models drift between these three constantly,
// and rejecting the drift would mean failing calls the model clearly intended.
std::vector<ToolCall> parse_tool_calls(std::string_view text);

// The same text with every tool-call block removed, so what is left is the
// prose meant for the user. Leading/trailing whitespace is trimmed.
std::string strip_tool_calls(std::string_view text);

// How much of a partially-generated reply is safe to show the user right now.
//
// Streaming raw deltas would spray <tool_call> markup into the chat. This
// returns the prose with completed call blocks removed, stops at an unclosed
// block, and holds back a trailing partial "<tool_c..." so the opening tag is
// never shown one character at a time. Call it on the accumulated text after
// each delta and emit only what is new.
std::string visible_prose_prefix(std::string_view accumulated);

// Resolve what the model emitted onto a real tool name: separator and case
// drift, aliases, and a leading-verb fallback. Exposed because anything that
// reasons about a call (the permission layer, above all) must judge the tool
// that will actually run, not the string the model typed.
std::string canonical_tool_name(std::string_view raw);

// Run one call against the workspace. Never throws: a malformed argument set
// comes back as ok=false with a message telling the model what was wrong.
ToolResult dispatch_tool(Workspace& ws, const ToolCall& call);

} // namespace ultima::server
