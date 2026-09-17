#include <doctest/doctest.h>

#include "ultima/server/session.hpp"
#include "ultima/server/workspace.hpp"

#include <filesystem>
#include <fstream>
#include <ostream>
#include <string>

namespace fs = std::filesystem;
using namespace ultima::server;

namespace {

class TempWorkspace {
public:
    TempWorkspace() {
        static int counter = 0;
        root_ = fs::temp_directory_path() /
                ("ultima_session_test_" + std::to_string(++counter) + "_" +
                 std::to_string(reinterpret_cast<std::uintptr_t>(this)));
        fs::create_directories(root_);
    }
    ~TempWorkspace() { std::error_code ec; fs::remove_all(root_, ec); }

    void write(const std::string& rel, const std::string& c) const {
        const auto p = root_ / rel;
        fs::create_directories(p.parent_path());
        std::ofstream out{p, std::ios::binary | std::ios::trunc};
        out << c;
    }
    std::string read(const std::string& rel) const {
        std::ifstream in{root_ / rel, std::ios::binary};
        return std::string{std::istreambuf_iterator<char>{in}, {}};
    }
    Workspace open() const {
        auto ws = Workspace::open(root_);
        REQUIRE(static_cast<bool>(ws));
        return std::move(*ws);
    }

private:
    fs::path root_;
};

const char* kSamplePlan =
    "## Landing page\n"
    "**Requested:** build a landing page for the shop\n"
    "**Started:** 2026-09-17 10:00\n"
    "\n"
    "- [x] Phase 1: page skeleton with header and footer\n"
    "- [ ] Phase 2: hero section\n"
    "- [ ] Phase 3: product grid\n";

} // namespace

// ---- parsing -----------------------------------------------------------------

TEST_CASE("parse_plan: reads the title, request and phases") {
    const Plan p = parse_plan(kSamplePlan);
    CHECK(p.title == "Landing page");
    CHECK(p.requested == "build a landing page for the shop");
    CHECK(p.started == "2026-09-17 10:00");
    REQUIRE(p.phases.size() == 3);
    CHECK(p.phases[0].done);
    CHECK_FALSE(p.phases[1].done);
    CHECK(p.phases[0].text == "page skeleton with header and footer");
    CHECK(p.phases[1].number == 2);
}

TEST_CASE("parse_plan: the active phase is the first unchecked one") {
    const Plan p = parse_plan(kSamplePlan);
    const Phase* a = p.active();
    REQUIRE(a != nullptr);
    CHECK(a->number == 2);
    CHECK(a->text == "hero section");
    CHECK(p.done_count() == 1);
}

TEST_CASE("parse_plan: a fully checked plan has no active phase") {
    const Plan p = parse_plan("## X\n- [x] Phase 1: a\n- [x] Phase 2: b\n");
    CHECK(p.active() == nullptr);
    CHECK(p.done_count() == 2);
}

TEST_CASE("parse_plan: tolerates the shapes a hand-edit or a model produces") {
    // Bullet style, capital X, numbered list, and a missing "Phase N:" prefix.
    const Plan p = parse_plan(
        "## Mixed\n"
        "* [X] first thing\n"
        "- [ ] Phase 2: second thing\n"
        "3. [ ] third thing\n"
        "\n"
        "Some prose that is not a phase at all.\n");
    REQUIRE(p.phases.size() == 3);
    CHECK(p.phases[0].done);
    CHECK(p.phases[0].text == "first thing");
    CHECK(p.phases[1].text == "second thing");
    CHECK(p.phases[2].text == "third thing");
}

TEST_CASE("parse_plan: renumbers by position, so reordering by hand is safe") {
    // The user swapped two phases but left the old numbers in place.
    const Plan p = parse_plan("## X\n- [ ] Phase 2: b\n- [ ] Phase 1: a\n");
    REQUIRE(p.phases.size() == 2);
    CHECK(p.phases[0].text == "b");
    CHECK(p.phases[0].number == 1);
    CHECK(p.phases[1].text == "a");
    CHECK(p.phases[1].number == 2);
}

TEST_CASE("parse_plan: no checkboxes means no plan, not a broken one") {
    const Plan p = parse_plan("# Current task plan\n\nNo active task.\n");
    CHECK(p.empty());
    CHECK(p.active() == nullptr);
}

TEST_CASE("render_plan: round-trips what parse_plan understands") {
    const Plan a = parse_plan(kSamplePlan);
    const Plan b = parse_plan(render_plan(a));
    REQUIRE(b.phases.size() == a.phases.size());
    CHECK(b.title == a.title);
    CHECK(b.requested == a.requested);
    CHECK(b.started == a.started);
    for (std::size_t i = 0; i < a.phases.size(); ++i) {
        CHECK(b.phases[i].text == a.phases[i].text);
        CHECK(b.phases[i].done == a.phases[i].done);
    }
}

// ---- workspace round trip -----------------------------------------------------

TEST_CASE("load_plan: a workspace with no plan yields an empty one, not an error") {
    TempWorkspace tmp;
    auto ws = tmp.open();
    auto p = load_plan(ws);
    REQUIRE(static_cast<bool>(p));
    CHECK(p->empty());
}

TEST_CASE("save_plan / load_plan: survives a trip through the workspace") {
    TempWorkspace tmp;
    auto ws = tmp.open();

    Plan p = parse_plan(kSamplePlan);
    REQUIRE(save_plan(ws, p));
    CHECK(tmp.read(".ultima/plan.md").find("Phase 2: hero section") != std::string::npos);

    auto back = load_plan(ws);
    REQUIRE(static_cast<bool>(back));
    REQUIRE(back->phases.size() == 3);
    CHECK(back->active()->text == "hero section");
}

TEST_CASE("todos attach to their phase") {
    TempWorkspace tmp;
    auto ws = tmp.open();
    REQUIRE(save_plan(ws, parse_plan(kSamplePlan)));

    std::vector<Todo> todos{
        {2, "t1", "markup", true},
        {2, "t2", "styling", false},
        {3, "t3", "card component", false},
        {99, "t4", "orphan pointing at a phase that does not exist", false},
    };
    REQUIRE(save_todos(ws, todos));

    auto p = load_plan(ws);
    REQUIRE(static_cast<bool>(p));
    CHECK(p->phases[1].todos.size() == 2);
    CHECK(p->phases[2].todos.size() == 1);
    CHECK(p->phases[0].todos.empty());
    // The orphan is dropped rather than crashing or landing on a real phase.
    std::size_t total = 0;
    for (const auto& ph : p->phases) total += ph.todos.size();
    CHECK(total == 3);
}

TEST_CASE("load_todos: a hand-broken json file reports the problem") {
    TempWorkspace tmp;
    tmp.write(".ultima/todos.json", "[{ this is not json }]");
    auto ws = tmp.open();
    auto r = load_todos(ws);
    CHECK_FALSE(r);
    CHECK(r.error().message.find("valid JSON") != std::string::npos);
}

TEST_CASE("append_changelog: appends and never rewrites") {
    TempWorkspace tmp;
    auto ws = tmp.open();

    REQUIRE(append_changelog(ws, "phase 1 done - skeleton"));
    REQUIRE(append_changelog(ws, "phase 2 done - hero"));

    const auto body = tmp.read(".ultima/CHANGELOG.md");
    CHECK(body.find("phase 1 done - skeleton") != std::string::npos);
    CHECK(body.find("phase 2 done - hero") != std::string::npos);
    CHECK(body.find("phase 1") < body.find("phase 2"));   // oldest first
}

// ---- prompt rendering ---------------------------------------------------------

TEST_CASE("render_plan_for_prompt: names one phase and withholds the rest") {
    const Plan p = parse_plan(kSamplePlan);
    const auto s = render_plan_for_prompt(p);

    CHECK(s.find("PHASE 2") != std::string::npos);
    CHECK(s.find("hero section") != std::string::npos);
    // The later phase must not be spelled out, or the model tries to do it too.
    CHECK(s.find("product grid") == std::string::npos);
    CHECK(s.find("1 later phase") != std::string::npos);
}

TEST_CASE("render_plan_for_prompt: subtasks of the active phase are included") {
    TempWorkspace tmp;
    auto ws = tmp.open();
    REQUIRE(save_plan(ws, parse_plan(kSamplePlan)));
    REQUIRE(save_todos(ws, {{2, "t1", "write the markup", false}}));

    auto p = load_plan(ws);
    REQUIRE(static_cast<bool>(p));
    const auto s = render_plan_for_prompt(*p);
    CHECK(s.find("write the markup") != std::string::npos);
}

TEST_CASE("render_plan_for_prompt: a finished plan says so instead of inventing work") {
    const Plan p = parse_plan("## X\n- [x] Phase 1: a\n");
    const auto s = render_plan_for_prompt(p);
    CHECK(s.find("complete") != std::string::npos);
    CHECK(s.find("do not start new work") != std::string::npos);
}

TEST_CASE("render_plan_for_prompt: empty when there is no plan") {
    CHECK(render_plan_for_prompt(Plan{}).empty());
}

TEST_CASE("render_planning_instructions: demands a plan and forbids code") {
    TempWorkspace tmp;
    auto ws = tmp.open();
    const auto s = render_planning_instructions(ws);
    CHECK(s.find(".ultima/plan.md") != std::string::npos);
    CHECK(s.find("Do not write any project code") != std::string::npos);
    CHECK(s.find("- [ ] Phase 1:") != std::string::npos);
}

TEST_CASE("parse_plan: tolerates the malformed bullets a model actually emits") {
    // All four shapes came out of one live Qwen2.5-Coder-7B planning turn.
    const Plan p = parse_plan(
        "## Coffee Shop Landing Page\n"
        "- [ ] Phase 1: Set up HTML structure\n"
        "- [ ]Phase2: Style hero section\n"
        "- [ ]Phase3 : Add menu section\n"
        "-[x]Phase4 : Contact form\n");
    REQUIRE(p.phases.size() == 4);
    CHECK(p.phases[0].text == "Set up HTML structure");
    CHECK(p.phases[1].text == "Style hero section");
    CHECK(p.phases[2].text == "Add menu section");
    CHECK(p.phases[3].text == "Contact form");
    CHECK(p.phases[3].done);
    CHECK(p.active()->number == 1);
}
