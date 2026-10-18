#pragma once

#include "ultima/server/tools.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace ultima::server {

// What the coding agent may do unattended. See PERMISSIONS.md for the user-
// facing description; this header is the contract the server implements.
//
// The shape is conventional on purpose: reads are never gated, writes are, and
// a small set always asks regardless of mode. Gating every call — reads
// included — trains the user to approve without reading, which is how a
// permission system stops meaning anything.
enum class PermissionMode {
    Auto,   // ordinary writes pass; the always-ask set asks
    Ask,    // every mutation asks
    Yolo,   // nothing asks
};

// Why a call is being gated. Ordered roughly by how much the user should care.
enum class Risk {
    Read,         // list_files, read_file — never gated
    Mutate,       // write / edit / rename inside the project
    Destructive,  // delete, and rm/format-style commands
    Escape,       // resolves outside the workspace
    Plan,         // rewrites .ultima/plan.md or todos.json
    Install,      // pulls third-party code onto the machine
    Execute,      // runs a command outside the project
};

const char* to_string(Risk r) noexcept;
const char* to_string(PermissionMode m) noexcept;

// Parse a mode name; unknown text yields Auto, which is the safe default.
PermissionMode parse_permission_mode(std::string_view s) noexcept;

struct Verdict {
    Risk        risk = Risk::Read;
    bool        needs_prompt = false;
    // Shown to the user verbatim when prompting. A request someone cannot
    // evaluate is a request they will rubber-stamp, so this is never empty
    // when needs_prompt is true.
    std::string reason;
    // The specific argument the risk attaches to (a path, a command), for the
    // prompt. Empty when the whole call is the subject.
    std::string subject;
};

// Classify a call without reference to the mode. Exposed for tests and so the
// UI can explain a risk it did not prompt for.
Risk classify(const ToolCall& call) noexcept;

// Decide whether `call` may proceed under `mode`.
//
// `session_allowed` is the set of risk classes the user has already approved
// for the rest of this session via "always allow this kind". Denials are
// deliberately not remembered: the reason for refusing once may not hold the
// next time.
Verdict evaluate(PermissionMode mode, const ToolCall& call,
                 const std::vector<Risk>& session_allowed);

// True when a shell command string looks destructive (rm, del, rmdir, format,
// mkfs, diskpart...). Used by the Execute path once sandbox tools land; a
// command that deletes is destructive whichever tool spells it.
bool looks_destructive_command(std::string_view cmd) noexcept;

// True when a command installs third-party code.
bool looks_install_command(std::string_view cmd) noexcept;

} // namespace ultima::server
