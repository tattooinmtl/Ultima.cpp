#include "ultima/server/tools.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace ultima::server {

namespace {

using json = nlohmann::json;

json make_tool(const char* name, const char* description, json properties,
               std::vector<std::string> required) {
    return {
        {"type", "function"},
        {"function", {
            {"name", name},
            {"description", description},
            {"parameters", {
                {"type", "object"},
                {"properties", std::move(properties)},
                {"required", required},
            }},
        }},
    };
}

json str_param(const char* desc) {
    return {{"type", "string"}, {"description", desc}};
}

// Trim ASCII whitespace from both ends.
std::string_view trim(std::string_view s) {
    std::size_t b = 0, e = s.size();
    auto ws = [](char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r';
    };
    while (b < e && ws(s[b])) ++b;
    while (e > b && ws(s[e - 1])) --e;
    return s.substr(b, e - b);
}

// Find the end of the JSON object starting at `start` (which must index a '{'),
// tracking string literals and escapes so braces inside strings do not confuse
// the count. Returns npos when the object never closes.
std::size_t match_brace(std::string_view s, std::size_t start) {
    if (start >= s.size() || s[start] != '{') return std::string_view::npos;
    int depth = 0;
    bool in_str = false, esc = false;
    for (std::size_t i = start; i < s.size(); ++i) {
        const char c = s[i];
        if (in_str) {
            if (esc)            esc = false;
            else if (c == '\\') esc = true;
            else if (c == '"')  in_str = false;
            continue;
        }
        if (c == '"') { in_str = true; continue; }
        if (c == '{') ++depth;
        else if (c == '}') {
            if (--depth == 0) return i;
        }
    }
    return std::string_view::npos;
}

// Replace typographic quotes with their ASCII equivalents.
//
// Observed live from Qwen2.5-Coder-7B: it opened a tool call with a correct
// "name" then drifted into curly quotes for every later key and value, which
// makes the JSON unparseable and loses the whole call. Only ever applied after
// a strict parse has already failed, so a call whose content legitimately
// contains curly quotes is unaffected.
std::string repair_smart_quotes(std::string_view in) {
    static const struct { const char* from; char to; } subs[] = {
        {"\xE2\x80\x9C", '"'},    // U+201C left double
        {"\xE2\x80\x9D", '"'},    // U+201D right double
        {"\xE2\x80\x98", '\''},   // U+2018 left single
        {"\xE2\x80\x99", '\''},   // U+2019 right single
    };
    std::string out;
    out.reserve(in.size());
    for (std::size_t i = 0; i < in.size();) {
        bool hit = false;
        if (i + 3 <= in.size()) {
            for (const auto& sub : subs) {
                if (in.compare(i, 3, sub.from) == 0) { out.push_back(sub.to); i += 3; hit = true; break; }
            }
        }
        if (!hit) out.push_back(in[i++]);
    }
    return out;
}

// Turn one JSON blob into a ToolCall, if it looks like one.
bool call_from_json(std::string_view blob, ToolCall& out) {
    json j;
    try {
        j = json::parse(blob);
    } catch (const std::exception&) {
        // Second chance: the usual reason a call fails to parse is curly
        // quotes, not a genuinely malformed object.
        try {
            j = json::parse(repair_smart_quotes(blob));
        } catch (const std::exception&) {
            return false;
        }
    }
    if (!j.is_object() || !j.contains("name") || !j["name"].is_string()) return false;

    out.name = j["name"].get<std::string>();
    // Qwen emits "arguments"; some models emit "parameters" or inline the
    // fields. Accept all three rather than dropping a call we understood.
    if (j.contains("arguments") && j["arguments"].is_object()) {
        out.args = j["arguments"];
    } else if (j.contains("parameters") && j["parameters"].is_object()) {
        out.args = j["parameters"];
    } else if (j.contains("arguments") && j["arguments"].is_string()) {
        // Some models double-encode the argument object as a string.
        try { out.args = json::parse(j["arguments"].get<std::string>()); }
        catch (const std::exception&) { out.args = json::object(); }
    } else {
        json rest = j;
        rest.erase("name");
        out.args = rest.is_object() ? rest : json::object();
    }
    out.raw = std::string{blob};
    return true;
}

// ---- argument access, tolerant of the shapes small models produce ----------

std::string arg_str(const json& args, const char* key, bool* found = nullptr) {
    if (found) *found = false;
    if (!args.contains(key)) return {};
    const auto& v = args[key];
    if (found) *found = true;
    if (v.is_string()) return v.get<std::string>();
    if (v.is_number_integer()) return std::to_string(v.get<long long>());
    if (v.is_boolean()) return v.get<bool>() ? "true" : "false";
    if (v.is_null()) { if (found) *found = false; return {}; }
    return v.dump();
}

bool arg_bool(const json& args, const char* key, bool dflt) {
    if (!args.contains(key)) return dflt;
    const auto& v = args[key];
    if (v.is_boolean()) return v.get<bool>();
    if (v.is_string()) {
        const auto s = v.get<std::string>();
        return s == "true" || s == "1" || s == "yes";
    }
    if (v.is_number()) return v.get<double>() != 0.0;
    return dflt;
}

} // namespace

// Map what the model actually emitted onto a real tool name.
//
// Observed from Qwen2.5-Coder-7B on the first live run: it called
// "write_project-file" with a hyphen. Separator and case drift like that is
// constant with small models and costs a whole round to recover from, so it is
// absorbed here. The aliases below are names a model reaches for when it does
// not recall the exact one; they are unambiguous, so accepting them loses
// nothing.
std::string canonical_tool_name(std::string_view raw) {
    std::string n;
    n.reserve(raw.size());
    for (char c : raw) {
        if (c == '-' || c == ' ' || c == '.') n.push_back('_');
        else n.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    // Some models namespace the call, e.g. "functions.read_file".
    const auto dot = n.find_last_of('_');
    static const std::vector<std::string> real{
        "list_files", "read_file", "write_project_file",
        "edit_file", "rename_file", "delete_file",
    };
    for (const auto& r : real) {
        if (n == r) return r;
    }
    (void)dot;

    struct Alias { const char* from; const char* to; };
    static const Alias aliases[] = {
        {"create_file",     "write_project_file"},
        {"write_file",      "write_project_file"},
        {"new_file",        "write_project_file"},
        {"save_file",       "write_project_file"},
        {"writefile",       "write_project_file"},
        {"write_project",   "write_project_file"},
        {"open_file",       "read_file"},
        {"view_file",       "read_file"},
        {"cat_file",        "read_file"},
        {"readfile",        "read_file"},
        {"list_dir",        "list_files"},
        {"list_directory",  "list_files"},
        {"listfiles",       "list_files"},
        {"ls",              "list_files"},
        {"str_replace",     "edit_file"},
        {"replace_in_file", "edit_file"},
        {"modify_file",     "edit_file"},
        {"update_file",     "edit_file"},
        {"move_file",       "rename_file"},
        {"mv",              "rename_file"},
        {"remove_file",     "delete_file"},
        {"rm",              "delete_file"},
    };
    for (const auto& a : aliases) {
        if (n == a.from) return a.to;
    }
    // Trailing segment of a namespaced call: "functions_read_file".
    for (const auto& r : real) {
        if (n.size() > r.size() && n.compare(n.size() - r.size(), r.size(), r) == 0) {
            return r;
        }
    }

    // Last resort: match on the leading verb. Observed live, the 7B invented
    // "read_project" — the object was wrong but the verb was not, and there is
    // only one read tool. Resolving on the verb turns a dead end into an
    // argument error, which names exactly what to fix.
    struct Verb { const char* prefix; const char* to; };
    static const Verb verbs[] = {
        {"read_",    "read_file"},
        {"get_",     "read_file"},
        {"write_",   "write_project_file"},
        {"create_",  "write_project_file"},
        {"save_",    "write_project_file"},
        {"list_",    "list_files"},
        {"edit_",    "edit_file"},
        {"replace_", "edit_file"},
        {"update_",  "edit_file"},
        {"rename_",  "rename_file"},
        {"move_",    "rename_file"},
        {"delete_",  "delete_file"},
        {"remove_",  "delete_file"},
    };
    for (const auto& v : verbs) {
        const std::string_view pre{v.prefix};
        if (n.size() > pre.size() && n.compare(0, pre.size(), pre) == 0) return v.to;
    }
    return n;
}

namespace {

// A model writing "/hello.js" means the project root, not the filesystem root.
// resolve() rightly refuses absolute paths, so normalize here instead of
// loosening the security gate: strip a leading slash and any "./" prefix.
std::string normalize_tool_path(std::string_view raw) {
    std::string p{raw};
    std::size_t i = 0;
    while (i < p.size() && (p[i] == '/' || p[i] == '\\')) ++i;
    p.erase(0, i);
    while (p.rfind("./", 0) == 0) p.erase(0, 2);
    return p;
}

ToolResult err(const std::string& name, std::string message) {
    ToolResult r;
    r.name = name;
    r.ok = false;
    r.content = std::move(message);
    return r;
}

ToolResult ok(const std::string& name, std::string message) {
    ToolResult r;
    r.name = name;
    r.ok = true;
    r.content = std::move(message);
    return r;
}

// Very small glob: `*` matches any run of characters. Enough for "*.js" and
// "src/*", which is all a model reaches for.
bool glob_match(std::string_view pattern, std::string_view text) {
    std::size_t p = 0, t = 0, star = std::string_view::npos, mark = 0;
    while (t < text.size()) {
        if (p < pattern.size() && (pattern[p] == text[t])) { ++p; ++t; }
        else if (p < pattern.size() && pattern[p] == '*') { star = p++; mark = t; }
        else if (star != std::string_view::npos) { p = star + 1; t = ++mark; }
        else return false;
    }
    while (p < pattern.size() && pattern[p] == '*') ++p;
    return p == pattern.size();
}

} // namespace

const json& tool_schemas() {
    static const json schemas = json::array({
        make_tool("list_files",
                  "List the files in the project. Call this first when you do not "
                  "already know what the project contains.",
                  {{"glob", str_param("Optional filter such as '*.js' or 'src/*'. "
                                      "Omit to list everything.")}},
                  {}),
        make_tool("read_file",
                  "Read a file's contents. Read a file before editing it.",
                  {{"path", str_param("Project-relative path, e.g. 'src/app.js'.")}},
                  {"path"}),
        make_tool("write_project_file",
                  "Create a new file or replace an existing one entirely. Use "
                  "edit_file instead when changing part of a file you have read.",
                  {{"path",    str_param("Project-relative path.")},
                   {"content", str_param("The complete file contents.")},
                   {"tab",     str_param("Optional editor tab label; defaults to the filename.")}},
                  {"path", "content"}),
        make_tool("edit_file",
                  "Replace an exact string in a file. The text in 'find' must "
                  "appear exactly once unless 'all' is true. Prefer this over "
                  "rewriting a whole file.",
                  {{"path",    str_param("Project-relative path.")},
                   {"find",    str_param("Exact text to replace, copied from the file.")},
                   {"replace", str_param("Replacement text.")},
                   {"all",     {{"type", "boolean"},
                                {"description", "Replace every occurrence instead of "
                                                "requiring a unique match."}}}},
                  {"path", "find", "replace"}),
        make_tool("rename_file",
                  "Rename or move a file.",
                  {{"from", str_param("Existing project-relative path.")},
                   {"to",   str_param("New project-relative path.")}},
                  {"from", "to"}),
        make_tool("delete_file",
                  "Delete a file or directory.",
                  {{"path", str_param("Project-relative path.")}},
                  {"path"}),
    });
    return schemas;
}

std::string render_tools_system_prompt() {
    // The Qwen tool protocol, verbatim in the shape the models were trained
    // on. Inventing our own wrapper here would mean fighting the training.
    std::string out =
        "\n\n# Tools\n\n"
        "You can edit this project by calling functions. Function signatures are "
        "given inside <tools></tools>:\n<tools>\n";
    for (const auto& t : tool_schemas()) {
        out += t.dump();
        out.push_back('\n');
    }
    out +=
        "</tools>\n\n"
        "To call one, emit a JSON object with the function name and arguments "
        "inside <tool_call></tool_call>:\n"
        "<tool_call>\n"
        "{\"name\": \"read_file\", \"arguments\": {\"path\": \"src/app.js\"}}\n"
        "</tool_call>\n\n"
        "Rules:\n"
        "- Put file contents in write_project_file, never in your reply. The user "
        "sees files in the editor, not in chat.\n"
        "- Read a file before you edit it, so the text you pass to edit_file "
        "matches exactly.\n"
        "- You may emit several tool calls in one reply. You will be given the "
        "results and can then continue.\n"
        "- When the work is done, reply with a short plain-text summary and no "
        "tool calls.\n";
    return out;
}

std::vector<ToolCall> parse_tool_calls(std::string_view text) {
    std::vector<ToolCall> calls;

    // 1. The trained shape: <tool_call> ... </tool_call>.
    constexpr std::string_view open_tag  = "<tool_call>";
    constexpr std::string_view close_tag = "</tool_call>";
    std::size_t pos = 0;
    while ((pos = text.find(open_tag, pos)) != std::string_view::npos) {
        const std::size_t body = pos + open_tag.size();
        const std::size_t end  = text.find(close_tag, body);
        // A missing close tag usually means generation hit the token cap
        // mid-call; take the rest of the text and let the JSON parse decide.
        const std::string_view blob = (end == std::string_view::npos)
            ? text.substr(body)
            : text.substr(body, end - body);

        ToolCall c;
        const auto brace = blob.find('{');
        if (brace != std::string_view::npos) {
            const auto close = match_brace(blob, brace);
            const auto obj = (close == std::string_view::npos)
                ? blob.substr(brace) : blob.substr(brace, close - brace + 1);
            if (call_from_json(obj, c)) calls.push_back(std::move(c));
        }
        if (end == std::string_view::npos) break;
        pos = end + close_tag.size();
    }
    if (!calls.empty()) return calls;

    // 2. A fenced ```json block holding a call.
    constexpr std::string_view fence = "```";
    pos = 0;
    while ((pos = text.find(fence, pos)) != std::string_view::npos) {
        std::size_t body = pos + fence.size();
        // Skip an optional language tag on the same line.
        const std::size_t nl = text.find('\n', body);
        if (nl == std::string_view::npos) break;
        body = nl + 1;
        const std::size_t end = text.find(fence, body);
        const std::string_view blob = (end == std::string_view::npos)
            ? text.substr(body) : text.substr(body, end - body);

        const auto brace = blob.find('{');
        if (brace != std::string_view::npos) {
            const auto close = match_brace(blob, brace);
            if (close != std::string_view::npos) {
                ToolCall c;
                if (call_from_json(blob.substr(brace, close - brace + 1), c)) {
                    calls.push_back(std::move(c));
                }
            }
        }
        if (end == std::string_view::npos) break;
        pos = end + fence.size();
    }
    if (!calls.empty()) return calls;

    // 3. A bare object. Only accepted when it is the whole reply — otherwise a
    //    model quoting JSON at the user would be misread as a call.
    const auto body = trim(text);
    if (!body.empty() && body.front() == '{') {
        const auto close = match_brace(body, 0);
        if (close == body.size() - 1) {
            ToolCall c;
            if (call_from_json(body, c)) calls.push_back(std::move(c));
        }
    }
    return calls;
}

std::string strip_tool_calls(std::string_view text) {
    std::string out;
    out.reserve(text.size());

    constexpr std::string_view open_tag  = "<tool_call>";
    constexpr std::string_view close_tag = "</tool_call>";
    std::size_t pos = 0;
    while (true) {
        const std::size_t open = text.find(open_tag, pos);
        if (open == std::string_view::npos) {
            out += text.substr(pos);
            break;
        }
        out += text.substr(pos, open - pos);
        const std::size_t end = text.find(close_tag, open);
        if (end == std::string_view::npos) break;   // truncated call: drop the tail
        pos = end + close_tag.size();
    }
    return std::string{trim(out)};
}

std::string visible_prose_prefix(std::string_view acc) {
    constexpr std::string_view open_tag  = "<tool_call>";
    constexpr std::string_view close_tag = "</tool_call>";

    std::string out;
    out.reserve(acc.size());
    std::size_t pos = 0;
    while (true) {
        const std::size_t open = acc.find(open_tag, pos);
        if (open == std::string_view::npos) {
            std::string_view tail = acc.substr(pos);
            // Hold back a trailing partial open tag so it is not shown while
            // it is still arriving.
            for (std::size_t keep = std::min(open_tag.size() - 1, tail.size());
                 keep > 0; --keep) {
                if (tail.substr(tail.size() - keep) == open_tag.substr(0, keep)) {
                    tail = tail.substr(0, tail.size() - keep);
                    break;
                }
            }
            out += tail;
            break;
        }
        out += acc.substr(pos, open - pos);
        const std::size_t end = acc.find(close_tag, open);
        if (end == std::string_view::npos) break;   // still inside a call
        pos = end + close_tag.size();
    }
    return out;
}

ToolResult dispatch_tool(Workspace& ws, const ToolCall& call) {
    const std::string name = canonical_tool_name(call.name);

    // Path-valued arguments get the leading-slash normalization; everything
    // else is taken as written.
    auto require = [&](const char* key, std::string& into) -> bool {
        bool found = false;
        into = arg_str(call.args, key, &found);
        const std::string_view k{key};
        const bool is_path = (k == "path" || k == "from" || k == "to");

        // Models rename the path argument constantly - "file_path" was
        // observed live, and cost a whole round. Accept the obvious synonyms
        // rather than bouncing a call whose intent is unambiguous. The
        // permission layer accepts the same set, so the gate and the
        // dispatcher never disagree about what a call is doing.
        if (!found && is_path) {
            for (const char* alt : {"file_path", "filepath", "filename",
                                    "file", "target"}) {
                into = arg_str(call.args, alt, &found);
                if (found) break;
            }
        }
        if (found && is_path) into = normalize_tool_path(into);
        return found;
    };
    // A bare "missing argument X" leaves a small model with nowhere to go; it
    // tends to apologize and stop. Naming the retry keeps the round useful.
    auto missing = [&](const char* key) {
        return err(name, fmt::format("missing required argument '{}'. "
                                     "Call {} again with '{}' set.",
                                     key, name, key));
    };

    if (name == "list_files") {
        const std::string glob = arg_str(call.args, "glob");
        auto entries = ws.list_recursive();
        if (!entries) return err(name, entries.error().message);

        std::string body;
        std::size_t shown = 0;
        for (const auto& e : *entries) {
            if (!glob.empty() && !glob_match(glob, e.path)) continue;
            body += fmt::format("{} ({} bytes)\n", e.path, e.size);
            if (++shown >= 500) { body += "... (truncated)\n"; break; }
        }
        if (shown == 0) {
            return ok(name, glob.empty() ? "The project is empty."
                                         : fmt::format("No files match '{}'.", glob));
        }
        return ok(name, body);
    }

    if (name == "read_file") {
        std::string path;
        if (!require("path", path)) return missing("path");
        auto got = ws.read(path);
        if (!got) return err(name, got.error().message);
        std::string body = got->content;
        if (got->truncated) body += "\n... (file truncated)";
        return ok(name, body);
    }

    if (name == "write_project_file") {
        std::string path, content;
        if (!require("path", path))       return missing("path");
        if (!require("content", content)) return missing("content");
        if (auto w = ws.write(path, content); !w) return err(name, w.error().message);

        auto r = ok(name, fmt::format("Wrote {} ({} bytes).", path, content.size()));
        r.changed_path = path;
        r.tab_label = arg_str(call.args, "tab");
        if (r.tab_label.empty()) {
            const auto slash = path.find_last_of('/');
            r.tab_label = (slash == std::string::npos) ? path : path.substr(slash + 1);
        }
        return r;
    }

    if (name == "edit_file") {
        std::string path, find, replace;
        if (!require("path", path)) return missing("path");
        if (!require("find", find)) return missing("find");
        bool has_replace = false;
        replace = arg_str(call.args, "replace", &has_replace);
        if (!has_replace) return missing("replace");

        Workspace::Edit e{find, replace, arg_bool(call.args, "all", false)};
        auto n = ws.edit(path, {e});
        if (!n) return err(name, n.error().message);

        auto r = ok(name, fmt::format("Made {} replacement(s) in {}.", *n, path));
        r.changed_path = path;
        const auto slash = path.find_last_of('/');
        r.tab_label = (slash == std::string::npos) ? path : path.substr(slash + 1);
        return r;
    }

    if (name == "rename_file") {
        std::string from, to;
        if (!require("from", from)) return missing("from");
        if (!require("to", to))     return missing("to");
        if (auto rr = ws.rename(from, to); !rr) return err(name, rr.error().message);
        auto r = ok(name, fmt::format("Renamed {} to {}.", from, to));
        r.changed_path = to;
        return r;
    }

    if (name == "delete_file") {
        std::string path;
        if (!require("path", path)) return missing("path");
        if (auto rr = ws.remove(path); !rr) return err(name, rr.error().message);
        auto r = ok(name, fmt::format("Deleted {}.", path));
        r.changed_path = path;
        return r;
    }

    // Naming the available tools turns a hallucinated call into a recoverable
    // turn instead of a dead end.
    std::string known;
    for (const auto& t : tool_schemas()) {
        if (!known.empty()) known += ", ";
        known += t["function"]["name"].get<std::string>();
    }
    return err(name, fmt::format("No tool named '{}'. Available tools: {}. "
                                 "Pick the closest one and call it again.",
                                 name, known));
}

} // namespace ultima::server
