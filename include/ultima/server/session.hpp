#pragma once

#include "ultima/core/error.hpp"
#include "ultima/server/workspace.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ultima::server {

// The phased-planning workflow (Decision 18 §18.5).
//
// Small models drift when handed an open-ended goal, so a project-sized
// request is decomposed into phases and only one is worked at a time. The
// state lives in the workspace, not in memory, so a session resumed tomorrow
// reads its own history:
//
//   <workspace>/.ultima/plan.md    — the phase list, readable and hand-editable
//   <workspace>/.ultima/todos.json — subtasks, when a phase needs more than one
//   <workspace>/.ultima/CHANGELOG.md — append-only log of completed phases
//
// The plan is a file rather than a tool because the agent already knows how to
// write files, and six tools is as many as a small model reliably picks from.

struct Todo {
    std::size_t phase = 0;      // 1-based; 0 means "not tied to a phase"
    std::string id;
    std::string text;
    bool        done = false;
};

struct Phase {
    std::size_t       number = 0;   // 1-based, as displayed
    std::string       text;         // the outcome, without the "Phase N:" prefix
    bool              done = false;
    std::vector<Todo> todos;        // filled in by load_plan from todos.json
};

struct Plan {
    std::string        title;
    std::string        requested;
    std::string        started;
    std::vector<Phase> phases;

    bool empty() const noexcept { return phases.empty(); }

    // The phase the agent should be working: the first unchecked one.
    // Returns nullptr when every phase is done or there are none.
    const Phase* active() const noexcept;

    std::size_t done_count() const noexcept;
};

// Parse the plan.md format. Tolerant by design: the user is invited to edit
// this file by hand, so anything unrecognized is preserved as-is rather than
// treated as an error. A file with no checkbox lines parses to an empty plan.
Plan parse_plan(std::string_view markdown);

// Render a Plan back to plan.md. Round-trips what parse_plan understands.
std::string render_plan(const Plan& plan);

// Read plan.md + todos.json from <workspace>/.ultima/. A missing plan is not
// an error — it means no project is in flight — and yields an empty Plan.
core::Result<Plan> load_plan(const Workspace& ws);

// Write plan.md, creating .ultima/ if needed.
core::Result<void> save_plan(Workspace& ws, const Plan& plan);

core::Result<std::vector<Todo>> load_todos(const Workspace& ws);
core::Result<void> save_todos(Workspace& ws, const std::vector<Todo>& todos);

// Append one line to <workspace>/.ultima/CHANGELOG.md. Never rewrites.
core::Result<void> append_changelog(Workspace& ws, std::string_view line);

// The plan section injected into the agent's system prompt. Empty when there
// is no plan. Names the one phase to work and the subtasks under it, because
// handing a small model the whole list invites it to skip ahead.
std::string render_plan_for_prompt(const Plan& plan);

// Instructions for a first turn: produce a plan and nothing else.
std::string render_planning_instructions(const Workspace& ws);

// Where the workflow files live, relative to the workspace root.
constexpr std::string_view k_plan_path      = ".ultima/plan.md";
constexpr std::string_view k_todos_path     = ".ultima/todos.json";
constexpr std::string_view k_changelog_path = ".ultima/CHANGELOG.md";

} // namespace ultima::server
