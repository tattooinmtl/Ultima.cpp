#include <doctest/doctest.h>

#include "ultima/server/skills.hpp"
#include "ultima/server/tools.hpp"
#include "ultima/server/workspace.hpp"

#include <filesystem>
#include <fstream>
#include <ostream>
#include <string>

namespace fs = std::filesystem;
using namespace ultima::server;

namespace {

// Two throwaway skill roots, so priority and shadowing can be exercised.
class TempRoots {
public:
    TempRoots() {
        static int counter = 0;
        base_ = fs::temp_directory_path() /
                ("ultima_skills_test_" + std::to_string(++counter) + "_" +
                 std::to_string(reinterpret_cast<std::uintptr_t>(this)));
        fs::create_directories(base_ / "first");
        fs::create_directories(base_ / "second");
    }
    ~TempRoots() { std::error_code ec; fs::remove_all(base_, ec); }

    void add(const char* root, const std::string& folder, const std::string& body) const {
        const auto dir = base_ / root / folder;
        fs::create_directories(dir);
        std::ofstream out{dir / "SKILL.md", std::ios::binary | std::ios::trunc};
        out << body;
    }
    std::vector<fs::path> roots() const { return {base_ / "first", base_ / "second"}; }
    fs::path base() const { return base_; }

private:
    fs::path base_;
};

std::string skill_md(const std::string& name, const std::string& desc,
                     const std::string& tags, const std::string& body) {
    return "---\nname: " + name + "\ndescription: " + desc +
           "\ncommand: /" + name + "\ntags: [" + tags + "]\n---\n\n" + body;
}

} // namespace

// ---- frontmatter --------------------------------------------------------------

TEST_CASE("parse_skill_frontmatter: reads the fields we rely on") {
    const auto sk = parse_skill_frontmatter(
        "---\nname: html5-canvas\ndescription: 2D canvas games\n"
        "command: /canvas\ntags: [html5, canvas, game]\n---\n\n# body\n");
    CHECK(sk.name == "html5-canvas");
    CHECK(sk.description == "2D canvas games");
    CHECK(sk.command == "/canvas");
    REQUIRE(sk.tags.size() == 3);
    CHECK(sk.tags[1] == "canvas");
}

TEST_CASE("parse_skill_frontmatter: quoted values and a missing block") {
    const auto q = parse_skill_frontmatter(
        "---\nname: \"x-y\"\ndescription: 'has: a colon'\n---\nbody");
    CHECK(q.name == "x-y");
    CHECK(q.description == "has: a colon");

    // No frontmatter is not an error; discovery falls back to the folder name.
    CHECK(parse_skill_frontmatter("# just a heading\n").name.empty());
    CHECK(parse_skill_frontmatter("").name.empty());
}

// ---- discovery ----------------------------------------------------------------

TEST_CASE("discover: finds skills across roots, nested or not") {
    TempRoots t;
    t.add("first",  "alpha",            skill_md("alpha", "first root", "a", "# A"));
    t.add("second", "beta",             skill_md("beta",  "second root", "b", "# B"));
    t.add("second", "group/gamma",      skill_md("gamma", "nested",     "c", "# C"));

    const auto idx = SkillIndex::discover(t.roots());
    CHECK(idx.size() == 3);
    REQUIRE(idx.find("alpha"));
    REQUIRE(idx.find("gamma"));
    CHECK(idx.find("gamma")->description == "nested");
}

TEST_CASE("discover: an earlier root shadows a later one with the same name") {
    TempRoots t;
    t.add("first",  "shared", skill_md("shared", "the project's own", "x", "# mine"));
    t.add("second", "shared", skill_md("shared", "the shipped one",   "x", "# theirs"));

    const auto idx = SkillIndex::discover(t.roots());
    CHECK(idx.size() == 1);
    REQUIRE(idx.find("shared"));
    CHECK(idx.find("shared")->description == "the project's own");
    CHECK(idx.find("shared")->root == 0);
}

TEST_CASE("discover: a name comes from the folder when frontmatter has none") {
    TempRoots t;
    t.add("first", "folder-named", "# no frontmatter here\n");
    const auto idx = SkillIndex::discover(t.roots());
    REQUIRE(idx.find("folder-named"));
}

TEST_CASE("discover: a root that does not exist is skipped, not an error") {
    TempRoots t;
    t.add("first", "solo", skill_md("solo", "d", "t", "# S"));
    auto roots = t.roots();
    roots.push_back(t.base() / "no-such-root");
    const auto idx = SkillIndex::discover(roots);
    CHECK(idx.size() == 1);
}

TEST_CASE("find: name matching ignores case and separators") {
    TempRoots t;
    t.add("first", "html5-canvas", skill_md("html5-canvas", "games", "c", "# C"));
    const auto idx = SkillIndex::discover(t.roots());

    // Every spelling a model reaches for must land on the same skill.
    for (const char* n : {"html5-canvas", "HTML5-Canvas", "html5_canvas",
                          "html5 canvas", "HTML5CANVAS"}) {
        CHECK_MESSAGE(idx.find(n) != nullptr, "should resolve: ", n);
    }
    CHECK(idx.find("canvas") == nullptr);      // a partial name is not an exact match
}

// ---- search: the discovery path for a wrong name --------------------------------

TEST_CASE("search: ranks an exact name over a description hit") {
    TempRoots t;
    t.add("first", "threejs-3d",  skill_md("threejs-3d", "scenes and meshes", "3d", "# T"));
    t.add("first", "web-frontend", skill_md("web-frontend", "html css and threejs notes",
                                            "web", "# W"));
    const auto idx = SkillIndex::discover(t.roots());

    auto hits = idx.search("threejs-3d", 5);
    REQUIRE(!hits.empty());
    CHECK(hits[0]->name == "threejs-3d");
}

TEST_CASE("search: a near-miss name still finds the skill") {
    TempRoots t;
    t.add("first", "html5-canvas", skill_md("html5-canvas", "2D canvas games", "game", "# C"));
    const auto idx = SkillIndex::discover(t.roots());

    for (const char* q : {"canvas", "html5", "game", "2D canvas"}) {
        CHECK_MESSAGE(!idx.search(q, 5).empty(), "should suggest something for: ", q);
    }
    CHECK(idx.search("", 5).empty());
}

// ---- body loading ---------------------------------------------------------------

TEST_CASE("load_skill_body: strips the frontmatter") {
    TempRoots t;
    t.add("first", "x", skill_md("x", "d", "t", "# Heading\n\nThe patterns."));
    const auto idx = SkillIndex::discover(t.roots());

    auto body = load_skill_body(*idx.find("x"));
    REQUIRE(static_cast<bool>(body));
    CHECK(body->rfind("# Heading", 0) == 0);        // starts at the body
    CHECK(body->find("description:") == std::string::npos);
}

// ---- the prompt -----------------------------------------------------------------

TEST_CASE("render_skills_prompt: names the near skills and counts the rest") {
    TempRoots t;
    t.add("first",  "alpha", skill_md("alpha", "a", "t", "# A"));
    t.add("second", "beta",  skill_md("beta",  "b", "t", "# B"));
    const auto idx = SkillIndex::discover(t.roots());

    const auto p = render_skills_prompt(idx);
    CHECK(p.find("use_skill") != std::string::npos);
    CHECK(p.find("alpha") != std::string::npos);
    CHECK(p.find("beta")  != std::string::npos);   // root 1 is still "near"

    CHECK(render_skills_prompt(SkillIndex{}).empty());
}

// ---- the tool -------------------------------------------------------------------

TEST_CASE("use_skill: returns the body, and a miss returns suggestions") {
    TempRoots t;
    t.add("first", "threejs-3d", skill_md("threejs-3d", "scene setup", "3d",
                                          "# Three.js\n\nboilerplate here"));
    const auto idx = SkillIndex::discover(t.roots());

    const auto wsdir = t.base() / "ws";
    fs::create_directories(wsdir);
    auto wsr = Workspace::open(wsdir);
    REQUIRE(static_cast<bool>(wsr));
    auto ws = std::move(*wsr);

    ToolCall hit;
    hit.name = "use_skill";
    hit.args = {{"name", "threejs-3d"}};
    auto r = dispatch_tool(ws, hit, &idx);
    CHECK(r.ok);
    CHECK(r.content.find("boilerplate here") != std::string::npos);

    // A wrong name is where discovery happens — it must name real alternatives.
    ToolCall miss;
    miss.name = "use_skill";
    miss.args = {{"name", "threejs"}};
    auto m = dispatch_tool(ws, miss, &idx);
    CHECK_FALSE(m.ok);
    CHECK(m.content.find("threejs-3d") != std::string::npos);

    // And with no index at all it says so rather than failing opaquely.
    auto none = dispatch_tool(ws, hit, nullptr);
    CHECK_FALSE(none.ok);
    CHECK(none.content.find("No skills") != std::string::npos);
}

TEST_CASE("use_skill: the name is accepted under its common synonyms") {
    TempRoots t;
    t.add("first", "code-review", skill_md("code-review", "review checklist", "qa", "# R"));
    const auto idx = SkillIndex::discover(t.roots());

    const auto wsdir = t.base() / "ws2";
    fs::create_directories(wsdir);
    auto wsr = Workspace::open(wsdir);
    REQUIRE(static_cast<bool>(wsr));
    auto ws = std::move(*wsr);

    for (const char* key : {"name", "skill", "skill_name", "topic"}) {
        nlohmann::json args;
        args[key] = "code-review";
        ToolCall c;
        c.name = "use_skill";
        c.args = args;
        CHECK_MESSAGE(dispatch_tool(ws, c, &idx).ok, "key not accepted: ", key);
    }
}

TEST_CASE("use_skill: misspelled tool names still route to it") {
    CHECK(canonical_tool_name("use_skills") == "use_skill");
    CHECK(canonical_tool_name("load_skill") == "use_skill");
    CHECK(canonical_tool_name("Use-Skill")  == "use_skill");
    CHECK(canonical_tool_name("skill")      == "use_skill");
}
