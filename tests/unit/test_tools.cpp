#include <doctest/doctest.h>

#include "ultima/server/tools.hpp"
#include "ultima/server/workspace.hpp"

#include <filesystem>
#include <fstream>
#include <ostream>
#include <string>

namespace fs = std::filesystem;
using ultima::server::ToolCall;
using ultima::server::Workspace;
using ultima::server::dispatch_tool;
using ultima::server::parse_tool_calls;
using ultima::server::strip_tool_calls;

namespace {

class TempWorkspace {
public:
    TempWorkspace() {
        static int counter = 0;
        root_ = fs::temp_directory_path() /
                ("ultima_tools_test_" + std::to_string(++counter) + "_" +
                 std::to_string(reinterpret_cast<std::uintptr_t>(this)));
        fs::create_directories(root_ / "src");
        write("src/app.js", "const greeting = 'hi';\n");
        write("index.html", "<h1>hello</h1>\n");
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
    const fs::path& root() const { return root_; }

private:
    fs::path root_;
};

} // namespace

// ---- parsing: the shapes small models actually emit --------------------------

TEST_CASE("parse_tool_calls: the trained <tool_call> shape") {
    const std::string text =
        "I'll look at the file first.\n"
        "<tool_call>\n"
        "{\"name\": \"read_file\", \"arguments\": {\"path\": \"src/app.js\"}}\n"
        "</tool_call>";

    auto calls = parse_tool_calls(text);
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "read_file");
    CHECK(calls[0].args["path"] == "src/app.js");
}

TEST_CASE("parse_tool_calls: several calls in one reply") {
    const std::string text =
        "<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"a.js\"}}</tool_call>\n"
        "<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"b.js\"}}</tool_call>";

    auto calls = parse_tool_calls(text);
    REQUIRE(calls.size() == 2);
    CHECK(calls[0].args["path"] == "a.js");
    CHECK(calls[1].args["path"] == "b.js");
}

TEST_CASE("parse_tool_calls: braces inside string values do not end the object") {
    // A model writing JS through write_project_file emits braces in `content`.
    const std::string text =
        "<tool_call>{\"name\":\"write_project_file\",\"arguments\":"
        "{\"path\":\"a.js\",\"content\":\"function f() { return {x: 1}; }\"}}</tool_call>";

    auto calls = parse_tool_calls(text);
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].args["content"] == "function f() { return {x: 1}; }");
}

TEST_CASE("parse_tool_calls: a fenced json block is accepted") {
    const std::string text =
        "Here is the call:\n```json\n"
        "{\"name\": \"list_files\", \"arguments\": {}}\n"
        "```\n";
    auto calls = parse_tool_calls(text);
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "list_files");
}

TEST_CASE("parse_tool_calls: a bare object is accepted only as the whole reply") {
    auto whole = parse_tool_calls("{\"name\":\"list_files\",\"arguments\":{}}");
    REQUIRE(whole.size() == 1);
    CHECK(whole[0].name == "list_files");

    // Prose that merely mentions JSON must not be read as a call.
    auto prose = parse_tool_calls(
        "The config looks like {\"name\":\"list_files\"} but I won't call it yet.");
    CHECK(prose.empty());
}

TEST_CASE("parse_tool_calls: 'parameters' and double-encoded arguments") {
    auto a = parse_tool_calls(
        "<tool_call>{\"name\":\"read_file\",\"parameters\":{\"path\":\"x.js\"}}</tool_call>");
    REQUIRE(a.size() == 1);
    CHECK(a[0].args["path"] == "x.js");

    auto b = parse_tool_calls(
        "<tool_call>{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"y.js\\\"}\"}</tool_call>");
    REQUIRE(b.size() == 1);
    CHECK(b[0].args["path"] == "y.js");
}

TEST_CASE("parse_tool_calls: a call truncated by the token cap still parses") {
    // Generation stopped before </tool_call>.
    const std::string text =
        "<tool_call>\n{\"name\":\"read_file\",\"arguments\":{\"path\":\"src/app.js\"}}";
    auto calls = parse_tool_calls(text);
    REQUIRE(calls.size() == 1);
    CHECK(calls[0].name == "read_file");
}

TEST_CASE("parse_tool_calls: plain prose yields nothing") {
    CHECK(parse_tool_calls("I have finished; the landing page is in index.html.").empty());
    CHECK(parse_tool_calls("").empty());
}

TEST_CASE("strip_tool_calls: leaves only the prose") {
    const std::string text =
        "Reading the file.\n"
        "<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"a.js\"}}</tool_call>\n"
        "Then I will edit it.";
    CHECK(strip_tool_calls(text) == "Reading the file.\n\nThen I will edit it.");
}

TEST_CASE("strip_tool_calls: a truncated call does not leak markup to the user") {
    const std::string text = "Working on it.\n<tool_call>{\"name\":\"read_file\"";
    CHECK(strip_tool_calls(text) == "Working on it.");
}

// ---- dispatch ----------------------------------------------------------------

TEST_CASE("dispatch: write_project_file creates the file and names a tab") {
    TempWorkspace tmp;
    auto ws = tmp.open();

    ToolCall c;
    c.name = "write_project_file";
    c.args = {{"path", "src/new.js"}, {"content", "export const n = 1;\n"}};
    auto r = dispatch_tool(ws, c);

    CHECK(r.ok);
    CHECK(tmp.read("src/new.js") == "export const n = 1;\n");
    CHECK(r.changed_path == "src/new.js");
    CHECK(r.tab_label == "new.js");          // defaults to the basename
}

TEST_CASE("dispatch: read_file returns contents") {
    TempWorkspace tmp;
    auto ws = tmp.open();

    ToolCall c;
    c.name = "read_file";
    c.args = {{"path", "src/app.js"}};
    auto r = dispatch_tool(ws, c);
    CHECK(r.ok);
    CHECK(r.content == "const greeting = 'hi';\n");
}

TEST_CASE("dispatch: edit_file applies and reports an exact replacement") {
    TempWorkspace tmp;
    auto ws = tmp.open();

    ToolCall c;
    c.name = "edit_file";
    c.args = {{"path", "src/app.js"}, {"find", "'hi'"}, {"replace", "'hello'"}};
    auto r = dispatch_tool(ws, c);
    CHECK(r.ok);
    CHECK(tmp.read("src/app.js") == "const greeting = 'hello';\n");
}

TEST_CASE("dispatch: a failed edit returns the reason verbatim for the model to act on") {
    TempWorkspace tmp;
    auto ws = tmp.open();

    ToolCall c;
    c.name = "edit_file";
    c.args = {{"path", "src/app.js"}, {"find", "not present"}, {"replace", "x"}};
    auto r = dispatch_tool(ws, c);
    CHECK_FALSE(r.ok);
    CHECK(r.content.find("no match") != std::string::npos);
}

TEST_CASE("dispatch: rename and delete") {
    TempWorkspace tmp;
    auto ws = tmp.open();

    ToolCall mv;
    mv.name = "rename_file";
    mv.args = {{"from", "index.html"}, {"to", "public/index.html"}};
    CHECK(dispatch_tool(ws, mv).ok);
    CHECK(ws.exists("public/index.html"));

    ToolCall rm;
    rm.name = "delete_file";
    rm.args = {{"path", "public/index.html"}};
    CHECK(dispatch_tool(ws, rm).ok);
    CHECK_FALSE(ws.exists("public/index.html"));
}

TEST_CASE("dispatch: list_files honours a glob") {
    TempWorkspace tmp;
    auto ws = tmp.open();

    ToolCall all;
    all.name = "list_files";
    auto r_all = dispatch_tool(ws, all);
    CHECK(r_all.ok);
    CHECK(r_all.content.find("src/app.js") != std::string::npos);
    CHECK(r_all.content.find("index.html") != std::string::npos);

    ToolCall js;
    js.name = "list_files";
    js.args = {{"glob", "*.js"}};
    auto r_js = dispatch_tool(ws, js);
    CHECK(r_js.ok);
    CHECK(r_js.content.find("src/app.js") != std::string::npos);
    CHECK(r_js.content.find("index.html") == std::string::npos);
}

TEST_CASE("dispatch: a missing required argument says which one") {
    TempWorkspace tmp;
    auto ws = tmp.open();

    ToolCall c;
    c.name = "write_project_file";
    c.args = {{"path", "a.js"}};          // no content
    auto r = dispatch_tool(ws, c);
    CHECK_FALSE(r.ok);
    CHECK(r.content.find("content") != std::string::npos);
}

TEST_CASE("dispatch: a resolved alias reports the missing argument, not the name") {
    // "create_file" is an accepted alias, so the complaint should be about the
    // absent content rather than about the tool name.
    TempWorkspace tmp;
    auto ws = tmp.open();

    ToolCall c;
    c.name = "create_file";
    c.args = {{"path", "a.js"}};          // no content
    auto r = dispatch_tool(ws, c);
    CHECK_FALSE(r.ok);
    CHECK(r.content.find("content") != std::string::npos);
}

TEST_CASE("dispatch: tools cannot escape the workspace") {
    TempWorkspace tmp;
    auto ws = tmp.open();

    ToolCall c;
    c.name = "write_project_file";
    c.args = {{"path", "../escaped.js"}, {"content", "x"}};
    auto r = dispatch_tool(ws, c);
    CHECK_FALSE(r.ok);
    CHECK_FALSE(fs::exists(tmp.root().parent_path() / "escaped.js"));
}

TEST_CASE("tool schemas: the six tools are declared with their required args") {
    const auto& s = ultima::server::tool_schemas();
    CHECK(s.size() == 6);

    bool saw_write = false;
    for (const auto& t : s) {
        CHECK(t["function"].contains("description"));
        if (t["function"]["name"] == "write_project_file") {
            saw_write = true;
            const auto req = t["function"]["parameters"]["required"];
            CHECK(std::find(req.begin(), req.end(), "path") != req.end());
            CHECK(std::find(req.begin(), req.end(), "content") != req.end());
        }
    }
    CHECK(saw_write);
}

TEST_CASE("tools system prompt: carries the protocol the model is trained on") {
    const auto p = ultima::server::render_tools_system_prompt();
    CHECK(p.find("<tools>") != std::string::npos);
    CHECK(p.find("<tool_call>") != std::string::npos);
    CHECK(p.find("write_project_file") != std::string::npos);
}

// ---- robustness against the drift small models actually produce ---------------
// Every case below was either observed live or is one separator away from it.

TEST_CASE("dispatch: tool-name separator and case drift is absorbed") {
    TempWorkspace tmp;
    auto ws = tmp.open();

    // Observed from Qwen2.5-Coder-7B on the first live run.
    for (const char* variant : {"write_project-file", "Write_Project_File",
                                "write project file", "functions.write_project_file"}) {
        ToolCall c;
        c.name = variant;
        c.args = {{"path", std::string("v_") + "x.js"}, {"content", "1"}};
        auto r = dispatch_tool(ws, c);
        CHECK_MESSAGE(r.ok, "should resolve to write_project_file: ", variant);
    }
}

TEST_CASE("dispatch: plausible aliases resolve to the real tool") {
    TempWorkspace tmp;
    auto ws = tmp.open();

    ToolCall create;
    create.name = "create_file";
    create.args = {{"path", "aliased.js"}, {"content", "const a = 1;\n"}};
    CHECK(dispatch_tool(ws, create).ok);
    CHECK(tmp.read("aliased.js") == "const a = 1;\n");

    ToolCall ls;
    ls.name = "list_directory";
    CHECK(dispatch_tool(ws, ls).ok);

    ToolCall open;
    open.name = "open_file";
    open.args = {{"path", "aliased.js"}};
    CHECK(dispatch_tool(ws, open).ok);
}

TEST_CASE("dispatch: a leading slash means the project root, not the filesystem") {
    // Also observed live: the model wrote "/hello.js" meaning "at the top of
    // the project". resolve() rightly refuses absolute paths, so the tool layer
    // normalizes instead of the security gate loosening.
    TempWorkspace tmp;
    auto ws = tmp.open();

    ToolCall c;
    c.name = "write_project_file";
    c.args = {{"path", "/hello.js"}, {"content", "export const hi = 1;\n"}};
    auto r = dispatch_tool(ws, c);
    CHECK(r.ok);
    CHECK(tmp.read("hello.js") == "export const hi = 1;\n");
    CHECK(r.changed_path == "hello.js");
}

TEST_CASE("dispatch: normalizing a leading slash does not permit escaping") {
    TempWorkspace tmp;
    auto ws = tmp.open();

    // Stripping the slash must not turn an escape into a valid path.
    for (const char* bad : {"/../escaped.js", "//../escaped.js", "/./../escaped.js"}) {
        ToolCall c;
        c.name = "write_project_file";
        c.args = {{"path", bad}, {"content", "x"}};
        auto r = dispatch_tool(ws, c);
        CHECK_MESSAGE(!r.ok, "must still refuse: ", bad);
    }
    CHECK_FALSE(fs::exists(tmp.root().parent_path() / "escaped.js"));
}

TEST_CASE("dispatch: a genuinely unknown name still reports the real tools") {
    TempWorkspace tmp;
    auto ws = tmp.open();

    ToolCall c;
    c.name = "summon_daemon";
    auto r = dispatch_tool(ws, c);
    CHECK_FALSE(r.ok);
    CHECK(r.content.find("write_project_file") != std::string::npos);
}

// ---- streaming filter --------------------------------------------------------

TEST_CASE("visible_prose_prefix: hides tool markup while it streams in") {
    using ultima::server::visible_prose_prefix;

    CHECK(visible_prose_prefix("Working on it.") == "Working on it.");
    // A partial opening tag is held back rather than shown character by character.
    CHECK(visible_prose_prefix("Working.<tool_c") == "Working.");
    // Inside an unclosed call, nothing further is shown.
    CHECK(visible_prose_prefix("Working.<tool_call>{\"name\":\"read") == "Working.");
    // After the call closes, following prose resumes.
    CHECK(visible_prose_prefix(
        "A<tool_call>{\"name\":\"x\"}</tool_call>B") == "AB");
}

TEST_CASE("visible_prose_prefix: grows monotonically as text arrives") {
    using ultima::server::visible_prose_prefix;
    const std::string full =
        "Reading.<tool_call>{\"name\":\"read_file\"}</tool_call>Done.";
    std::size_t last = 0;
    for (std::size_t i = 1; i <= full.size(); ++i) {
        const auto vis = visible_prose_prefix(std::string_view{full}.substr(0, i));
        CHECK(vis.size() >= last);      // never retracts what was already shown
        last = vis.size();
    }
    CHECK(visible_prose_prefix(full) == "Reading.Done.");
}

TEST_CASE("dispatch: an invented name resolves on its verb when that is unambiguous") {
    // Observed live: the 7B called "read_project" with a glob. The object was
    // wrong, the verb was not. Resolving on the verb turns a dead end into an
    // argument error the model can act on.
    TempWorkspace tmp;
    auto ws = tmp.open();

    ToolCall c;
    c.name = "read_project";
    c.args = {{"glob", "*.html"}};
    auto r = dispatch_tool(ws, c);
    CHECK_FALSE(r.ok);
    CHECK(r.content.find("path") != std::string::npos);   // names the real problem

    ToolCall good;
    good.name = "read_project";
    good.args = {{"path", "src/app.js"}};
    CHECK(dispatch_tool(ws, good).ok);
}

TEST_CASE("dispatch: a name with no recognizable verb is still refused") {
    TempWorkspace tmp;
    auto ws = tmp.open();
    ToolCall c;
    c.name = "frobnicate";
    auto r = dispatch_tool(ws, c);
    CHECK_FALSE(r.ok);
    CHECK(r.content.find("Available tools") != std::string::npos);
}
