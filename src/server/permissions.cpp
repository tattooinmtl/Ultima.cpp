#include "ultima/server/permissions.hpp"

#include <algorithm>
#include <array>
#include <cctype>

namespace ultima::server {

namespace {

std::string lower(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    return out;
}

bool contains(std::string_view hay, std::string_view needle) {
    return hay.find(needle) != std::string_view::npos;
}

// A path argument that tries to leave the workspace. Workspace::resolve()
// already refuses these outright; checking here too means a future tool that
// legitimately works outside the project cannot do it silently.
//
// This has to judge the path the tool will ACTUALLY use, not the one the model
// typed. dispatch_tool strips a leading slash first, because a model writing
// "/notes.txt" means the project root — treating that as an escape would
// prompt on most ordinary writes, since models spell paths that way constantly.
// Mirrors normalize_tool_path() in tools.cpp.
bool escapes_project(std::string_view p) noexcept {
    if (p.empty()) return false;
    std::string n{p};
    std::replace(n.begin(), n.end(), '\\', '/');

    std::size_t lead = 0;
    while (lead < n.size() && n[lead] == '/') ++lead;
    n.erase(0, lead);
    while (n.rfind("./", 0) == 0) n.erase(0, 2);
    if (n.empty()) return false;

    if (n.size() >= 2 && n[1] == ':' &&
        std::isalpha(static_cast<unsigned char>(n[0]))) return true;    // drive letter
    if (n == ".." || n.rfind("../", 0) == 0) return true;               // leading ..
    return contains(n, "/../") || (n.size() >= 3 && n.compare(n.size() - 3, 3, "/..") == 0);
}

bool touches_plan(std::string_view p) noexcept {
    const auto n = lower(p);
    return contains(n, ".ultima/plan.md") || contains(n, ".ultima\\plan.md")
        || contains(n, ".ultima/todos.json") || contains(n, ".ultima\\todos.json");
}

// The single path-ish argument a call is about, if it has one. The key list
// is wide because the prompt is useless without it: observed live, the model
// passed "file_path" and the user was asked to approve deleting nothing in
// particular.
std::string subject_path(const ToolCall& call) {
    for (const char* key : {"path", "file_path", "filepath", "filename",
                            "file", "target", "from", "to"}) {
        if (call.args.contains(key) && call.args[key].is_string()) {
            return call.args[key].get<std::string>();
        }
    }
    return {};
}

std::string subject_command(const ToolCall& call) {
    for (const char* key : {"command", "cmd", "code", "script"}) {
        if (call.args.contains(key) && call.args[key].is_string()) {
            return call.args[key].get<std::string>();
        }
    }
    return {};
}

bool is_read_tool(std::string_view name) noexcept {
    // use_skill only reads a bundled reference sheet, so it is never gated.
    return name == "read_file" || name == "list_files" || name == "use_skill";
}

bool is_exec_tool(std::string_view name) noexcept {
    return name == "run_command" || name == "bash_tool"
        || name == "python_run" || name == "node_run" || name == "php_run";
}

} // namespace

const char* to_string(Risk r) noexcept {
    switch (r) {
    case Risk::Read:        return "read";
    case Risk::Mutate:      return "mutate";
    case Risk::Destructive: return "destructive";
    case Risk::Escape:      return "escape";
    case Risk::Plan:        return "plan";
    case Risk::Install:     return "install";
    case Risk::Execute:     return "execute";
    }
    return "unknown";
}

const char* to_string(PermissionMode m) noexcept {
    switch (m) {
    case PermissionMode::Auto: return "auto";
    case PermissionMode::Ask:  return "ask";
    case PermissionMode::Yolo: return "yolo";
    }
    return "auto";
}

PermissionMode parse_permission_mode(std::string_view s) noexcept {
    const auto n = lower(s);
    if (n == "ask")  return PermissionMode::Ask;
    if (n == "yolo") return PermissionMode::Yolo;
    return PermissionMode::Auto;   // unknown text falls back to the safe default
}

bool looks_destructive_command(std::string_view cmd) noexcept {
    const auto n = lower(cmd);
    static constexpr std::array<std::string_view, 10> needles{
        "rm ", "rm -", "rmdir", "del ", "erase ", "format ",
        "mkfs", "diskpart", "remove-item", "shutil.rmtree",
    };
    for (auto w : needles) {
        if (contains(n, w)) return true;
    }
    return false;
}

bool looks_install_command(std::string_view cmd) noexcept {
    const auto n = lower(cmd);
    static constexpr std::array<std::string_view, 12> needles{
        "npm install", "npm i ", "yarn add", "pnpm add", "pnpm install",
        "pip install", "pip3 install", "cargo add", "go get",
        "apt install", "brew install", "composer require",
    };
    for (auto w : needles) {
        if (contains(n, w)) return true;
    }
    return false;
}

Risk classify(const ToolCall& call) noexcept {
    // Judge the tool that will RUN, not the string the model typed. Observed
    // live: the 7B called "delete_project", which the dispatcher resolves to
    // delete_file by its leading verb. Comparing the raw name here classified
    // it as an ordinary write, so auto mode deleted the file without asking —
    // the exact failure this layer exists to prevent.
    const std::string canon = canonical_tool_name(call.name);
    const std::string_view name{canon};

    if (is_read_tool(name)) return Risk::Read;

    if (is_exec_tool(name)) {
        const auto cmd = subject_command(call);
        if (looks_destructive_command(cmd)) return Risk::Destructive;
        if (looks_install_command(cmd))     return Risk::Install;
        return Risk::Execute;
    }

    // Path-based tools. Escape outranks everything: a delete outside the
    // project is worse than a delete inside it, and the prompt should say so.
    const auto path = subject_path(call);
    if (escapes_project(path)) return Risk::Escape;
    if (call.args.contains("to") && call.args["to"].is_string() &&
        escapes_project(call.args["to"].get<std::string>())) {
        return Risk::Escape;
    }

    if (name == "delete_file") return Risk::Destructive;
    if (touches_plan(path))    return Risk::Plan;

    return Risk::Mutate;
}

Verdict evaluate(PermissionMode mode, const ToolCall& call,
                 const std::vector<Risk>& session_allowed) {
    Verdict v;
    v.risk = classify(call);

    // Reading is never gated, in any mode.
    if (v.risk == Risk::Read) return v;

    if (mode == PermissionMode::Yolo) return v;

    const bool remembered =
        std::find(session_allowed.begin(), session_allowed.end(), v.risk)
            != session_allowed.end();
    if (remembered) return v;

    const bool always_ask = (v.risk != Risk::Mutate);
    if (!always_ask && mode == PermissionMode::Auto) return v;   // ordinary write

    v.needs_prompt = true;
    v.subject = subject_path(call);
    if (v.subject.empty()) v.subject = subject_command(call);
    if (v.subject.empty() && !call.args.empty()) {
        // Last resort: show the arguments rather than an empty parenthesis.
        // Approving a blank prompt is worse than approving a noisy one.
        v.subject = call.args.dump();
        if (v.subject.size() > 160) v.subject = v.subject.substr(0, 157) + "...";
    }

    switch (v.risk) {
    case Risk::Destructive:
        v.reason = "Deleting cannot be undone from the editor.";
        break;
    case Risk::Escape:
        v.reason = "This path is outside the project folder.";
        break;
    case Risk::Plan:
        v.reason = "This rewrites the plan the agent is working from.";
        break;
    case Risk::Install:
        v.reason = "This installs third-party code on your machine.";
        break;
    case Risk::Execute:
        v.reason = "This runs a command, not just a file edit.";
        break;
    case Risk::Mutate:
        v.reason = "Permission mode is 'ask', so every change is confirmed.";
        break;
    case Risk::Read:
        break;   // unreachable: handled above
    }
    return v;
}

} // namespace ultima::server
