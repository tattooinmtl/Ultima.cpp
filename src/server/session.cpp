#include "ultima/server/session.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <ctime>
#include <sstream>

namespace ultima::server {

namespace {

using core::ErrorCode;
using core::Failure;
using core::Result;
using core::fail;
using json = nlohmann::json;

std::string_view trim(std::string_view s) {
    auto ws = [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
    std::size_t b = 0, e = s.size();
    while (b < e && ws(s[b])) ++b;
    while (e > b && ws(s[e - 1])) --e;
    return s.substr(b, e - b);
}

bool starts_with(std::string_view s, std::string_view p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

// "Phase 3: build the nav" -> "build the nav". The number in the file is
// display sugar; the authoritative ordering is position in the list, so a user
// who reorders phases by hand does not have to renumber them.
std::string_view strip_phase_prefix(std::string_view s) {
    // "Phase 3:", "Phase3 :", "phase 3." all mean the same thing.
    if (!starts_with(s, "Phase") && !starts_with(s, "phase")) return s;
    std::size_t i = 5;
    while (i < s.size() && s[i] == ' ') ++i;
    const std::size_t digits_start = i;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i;
    if (i == digits_start) return s;
    while (i < s.size() && (s[i] == ':' || s[i] == '.' || s[i] == ' ')) ++i;
    return s.substr(i);
}

std::string now_stamp() {
    const auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M", &tm);
    return buf;
}

} // namespace

const Phase* Plan::active() const noexcept {
    for (const auto& p : phases) {
        if (!p.done) return &p;
    }
    return nullptr;
}

std::size_t Plan::done_count() const noexcept {
    std::size_t n = 0;
    for (const auto& p : phases) if (p.done) ++n;
    return n;
}

Plan parse_plan(std::string_view markdown) {
    Plan plan;
    std::istringstream in{std::string{markdown}};
    std::string raw;

    while (std::getline(in, raw)) {
        const std::string_view line = trim(raw);
        if (line.empty()) continue;

        if (starts_with(line, "## ") && plan.title.empty()) {
            plan.title = std::string{trim(line.substr(3))};
            continue;
        }
        if (starts_with(line, "**Requested:**")) {
            plan.requested = std::string{trim(line.substr(14))};
            continue;
        }
        if (starts_with(line, "**Started:**")) {
            plan.started = std::string{trim(line.substr(12))};
            continue;
        }

        // A checkbox item is a phase. Accept "- [ ]", "- [x]", "* [X]" and the
        // numbered "1. [ ]" a model might produce instead.
        std::string_view rest = line;
        if (starts_with(rest, "- ") || starts_with(rest, "* ")) {
            rest = trim(rest.substr(2));
        } else if (starts_with(rest, "-[") || starts_with(rest, "*[")) {
            // Observed live: a model emitting "-[x]Phase4 : ..." with no space.
            rest = trim(rest.substr(1));
        } else {
            std::size_t i = 0;
            while (i < rest.size() && std::isdigit(static_cast<unsigned char>(rest[i]))) ++i;
            if (i > 0 && i + 1 < rest.size() && (rest[i] == '.' || rest[i] == ')')) {
                rest = trim(rest.substr(i + 1));
            } else {
                continue;   // ordinary prose; not part of the phase list
            }
        }
        if (rest.size() < 3 || rest[0] != '[' || rest[2] != ']') continue;

        Phase ph;
        ph.done   = (rest[1] == 'x' || rest[1] == 'X');
        ph.text   = std::string{trim(strip_phase_prefix(trim(rest.substr(3))))};
        ph.number = plan.phases.size() + 1;
        plan.phases.push_back(std::move(ph));
    }
    return plan;
}

std::string render_plan(const Plan& plan) {
    std::string out;
    out += "## " + (plan.title.empty() ? std::string{"Untitled project"} : plan.title) + "\n";
    if (!plan.requested.empty()) out += "**Requested:** " + plan.requested + "\n";
    out += "**Started:** " + (plan.started.empty() ? now_stamp() : plan.started) + "\n\n";

    for (std::size_t i = 0; i < plan.phases.size(); ++i) {
        const auto& p = plan.phases[i];
        out += fmt::format("- [{}] Phase {}: {}\n", p.done ? 'x' : ' ', i + 1, p.text);
    }
    return out;
}

Result<std::vector<Todo>> load_todos(const Workspace& ws) {
    std::vector<Todo> out;
    if (!ws.exists(k_todos_path)) return out;

    auto got = ws.read(k_todos_path);
    if (!got) return Failure{got.error()};

    try {
        const auto j = json::parse(got->content);
        if (!j.is_array()) return out;
        for (const auto& e : j) {
            if (!e.is_object()) continue;
            Todo t;
            t.phase = e.value("phase", std::size_t{0});
            t.id    = e.value("id", std::string{});
            t.text  = e.value("text", std::string{});
            t.done  = e.value("done", false);
            if (!t.text.empty()) out.push_back(std::move(t));
        }
    } catch (const std::exception& e) {
        // A hand-edited file with a syntax error should say so, not silently
        // drop the user's subtasks.
        return fail(ErrorCode::InvalidArgument,
                    fmt::format("{} is not valid JSON: {}", k_todos_path, e.what()),
                    "session");
    }
    return out;
}

Result<void> save_todos(Workspace& ws, const std::vector<Todo>& todos) {
    json arr = json::array();
    for (const auto& t : todos) {
        arr.push_back({{"phase", t.phase}, {"id", t.id}, {"text", t.text}, {"done", t.done}});
    }
    return ws.write(k_todos_path, arr.dump(2) + "\n");
}

Result<Plan> load_plan(const Workspace& ws) {
    Plan plan;
    if (!ws.exists(k_plan_path)) return plan;   // no project in flight

    auto got = ws.read(k_plan_path);
    if (!got) return Failure{got.error()};
    plan = parse_plan(got->content);

    auto todos = load_todos(ws);
    if (!todos) return Failure{todos.error()};
    for (const auto& t : *todos) {
        if (t.phase >= 1 && t.phase <= plan.phases.size()) {
            plan.phases[t.phase - 1].todos.push_back(t);
        }
    }
    return plan;
}

Result<void> save_plan(Workspace& ws, const Plan& plan) {
    return ws.write(k_plan_path, render_plan(plan));
}

Result<void> append_changelog(Workspace& ws, std::string_view line) {
    std::string body;
    if (ws.exists(k_changelog_path)) {
        auto got = ws.read(k_changelog_path);
        if (!got) return Failure{got.error()};
        body = std::move(got->content);
        if (!body.empty() && body.back() != '\n') body.push_back('\n');
    } else {
        body = "# Project log\n\nAppend-only. One line per completed phase.\n\n";
    }
    body += fmt::format("{}  {}\n", now_stamp(), trim(line));
    return ws.write(k_changelog_path, body);
}

std::string render_plan_for_prompt(const Plan& plan) {
    if (plan.empty()) return {};

    std::string out = "\n\n# Current plan\n\n";
    out += fmt::format("Project: {}\n", plan.title.empty() ? "(untitled)" : plan.title);
    out += fmt::format("Progress: {} of {} phases done.\n\n",
                       plan.done_count(), plan.phases.size());

    const Phase* act = plan.active();
    if (!act) {
        out += "Every phase is checked off. Tell the user the project is complete "
               "and do not start new work unless they ask.\n";
        return out;
    }

    // Only the active phase is spelled out. Listing the whole plan invites a
    // small model to skip ahead and do three phases badly in one turn.
    out += fmt::format("You are working PHASE {} ONLY:\n  {}\n", act->number, act->text);
    if (!act->todos.empty()) {
        out += "\nSubtasks for this phase:\n";
        for (const auto& t : act->todos) {
            out += fmt::format("  [{}] {}\n", t.done ? 'x' : ' ', t.text);
        }
    }
    const std::size_t remaining = plan.phases.size() - plan.done_count();
    if (remaining > 1) {
        out += fmt::format("\n{} later phase(s) exist. Do not start them.\n", remaining - 1);
    }
    out +=
        "\nWhen this phase is done, rewrite " + std::string{k_plan_path} +
        " with its box checked ([x]) and stop, so the user can look at it "
        "before the next phase begins.\n";
    return out;
}

std::string render_planning_instructions(const Workspace& ws) {
    (void)ws;
    return
        "\n\n# Plan first\n\n"
        "This request needs more than one step, and no plan exists yet. Your ENTIRE "
        "job this turn is to write the plan. Do not write any project code.\n\n"
        "Call write_project_file once, with path \"" + std::string{k_plan_path} + "\" "
        "and content in exactly this shape:\n\n"
        "## <short project title>\n"
        "**Requested:** <the user's request in one line>\n\n"
        "- [ ] Phase 1: <outcome>\n"
        "- [ ] Phase 2: <outcome>\n"
        "- [ ] Phase 3: <outcome>\n\n"
        "Rules for the phases:\n"
        "- A phase is done when it produces something the user can look at. "
        "\"Set up the project\" is a phase; \"write the CSS variables\" is not.\n"
        "- Three to six phases. Fewer means they are too big; more means you are "
        "listing steps, not phases.\n"
        "- Order them so each one leaves the project in a working state.\n\n"
        "After that one tool call, reply with a one-line summary and stop. "
        "The user will tell you when to start Phase 1.\n";
}

} // namespace ultima::server
