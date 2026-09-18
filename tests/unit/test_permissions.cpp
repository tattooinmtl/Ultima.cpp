#include <doctest/doctest.h>

#include "ultima/server/permissions.hpp"

#include <ostream>
#include <string>
#include <vector>

using namespace ultima::server;

namespace {

ToolCall call(const char* name, nlohmann::json args = nlohmann::json::object()) {
    ToolCall c;
    c.name = name;
    c.args = std::move(args);
    return c;
}

const std::vector<Risk> kNothingRemembered{};

} // namespace

// ---- classification ----------------------------------------------------------

TEST_CASE("classify: reads, writes and deletes land in the right class") {
    CHECK(classify(call("read_file",  {{"path", "a.js"}}))               == Risk::Read);
    CHECK(classify(call("list_files"))                                    == Risk::Read);
    CHECK(classify(call("write_project_file", {{"path", "a.js"}}))        == Risk::Mutate);
    CHECK(classify(call("edit_file",   {{"path", "a.js"}}))               == Risk::Mutate);
    CHECK(classify(call("rename_file", {{"from", "a.js"}, {"to", "b.js"}})) == Risk::Mutate);
    CHECK(classify(call("delete_file", {{"path", "a.js"}}))               == Risk::Destructive);
}

TEST_CASE("classify: writing the plan is its own class") {
    CHECK(classify(call("write_project_file", {{"path", ".ultima/plan.md"}}))   == Risk::Plan);
    CHECK(classify(call("write_project_file", {{"path", ".ultima/todos.json"}})) == Risk::Plan);
    // A file that merely mentions the plan is an ordinary write.
    CHECK(classify(call("write_project_file", {{"path", "docs/plan-notes.md"}})) == Risk::Mutate);
}

TEST_CASE("classify: escaping the project outranks everything else") {
    // Worse than an ordinary delete, and the prompt should say which it is.
    CHECK(classify(call("delete_file",        {{"path", "../secrets"}}))      == Risk::Escape);
    // A leading slash is NOT an escape: dispatch_tool reads it as the project
    // root, so this lands at <workspace>/etc/passwd. Classifying it as an
    // escape would prompt on every path a model spells with a leading slash,
    // which is most of them.
    CHECK(classify(call("write_project_file", {{"path", "/etc/passwd"}}))     == Risk::Mutate);
    CHECK(classify(call("write_project_file", {{"path", "/notes.txt"}}))      == Risk::Mutate);
    CHECK(classify(call("write_project_file", {{"path", "C:/Windows/x"}}))    == Risk::Escape);
    CHECK(classify(call("write_project_file", {{"path", "src/../../out"}}))   == Risk::Escape);
    // A rename whose destination escapes counts too, not just the source.
    CHECK(classify(call("rename_file", {{"from", "a.js"}, {"to", "../a.js"}})) == Risk::Escape);
}

TEST_CASE("classify: commands split into destructive, install and plain execute") {
    CHECK(classify(call("run_command", {{"command", "node build.js"}}))    == Risk::Execute);
    CHECK(classify(call("run_command", {{"command", "rm -rf build"}}))     == Risk::Destructive);
    CHECK(classify(call("run_command", {{"command", "npm install left-pad"}})) == Risk::Install);
    CHECK(classify(call("python_run",  {{"code", "import shutil; shutil.rmtree('x')"}}))
          == Risk::Destructive);
}

TEST_CASE("looks_destructive_command / looks_install_command") {
    CHECK(looks_destructive_command("rm -rf /"));
    CHECK(looks_destructive_command("del C:\\temp\\x"));
    CHECK(looks_destructive_command("Remove-Item foo"));
    CHECK_FALSE(looks_destructive_command("npm run format-check"));  // 'format ' needs the space
    CHECK_FALSE(looks_destructive_command("node server.js"));

    CHECK(looks_install_command("npm install"));
    CHECK(looks_install_command("pip3 install requests"));
    CHECK(looks_install_command("cargo add serde"));
    CHECK_FALSE(looks_install_command("npm run build"));
}

// ---- auto mode ----------------------------------------------------------------

TEST_CASE("auto: ordinary writes pass without a prompt") {
    for (const char* t : {"write_project_file", "edit_file", "rename_file"}) {
        const auto v = evaluate(PermissionMode::Auto,
                                call(t, {{"path", "src/a.js"}, {"from", "a"}, {"to", "b"}}),
                                kNothingRemembered);
        CHECK_MESSAGE(!v.needs_prompt, "should not prompt in auto: ", t);
    }
}

TEST_CASE("auto: reads are never gated") {
    CHECK_FALSE(evaluate(PermissionMode::Auto, call("read_file", {{"path", "a"}}),
                         kNothingRemembered).needs_prompt);
    CHECK_FALSE(evaluate(PermissionMode::Ask, call("read_file", {{"path", "a"}}),
                         kNothingRemembered).needs_prompt);
    CHECK_FALSE(evaluate(PermissionMode::Yolo, call("read_file", {{"path", "a"}}),
                         kNothingRemembered).needs_prompt);
}

TEST_CASE("auto: the always-ask set still asks") {
    struct Case { const char* tool; nlohmann::json args; Risk risk; };
    const std::vector<Case> cases{
        {"delete_file",        {{"path", "a.js"}},                Risk::Destructive},
        {"write_project_file", {{"path", "../out.js"}},           Risk::Escape},
        {"write_project_file", {{"path", ".ultima/plan.md"}},     Risk::Plan},
        {"run_command",        {{"command", "npm install x"}},    Risk::Install},
        {"run_command",        {{"command", "node build.js"}},    Risk::Execute},
    };
    for (const auto& c : cases) {
        const auto v = evaluate(PermissionMode::Auto, call(c.tool, c.args), kNothingRemembered);
        CHECK_MESSAGE(v.needs_prompt, "should prompt in auto: ", c.tool);
        CHECK(v.risk == c.risk);
    }
}

// ---- ask mode -----------------------------------------------------------------

TEST_CASE("ask: every mutation asks, including ordinary writes") {
    const auto v = evaluate(PermissionMode::Ask,
                            call("write_project_file", {{"path", "src/a.js"}}),
                            kNothingRemembered);
    CHECK(v.needs_prompt);
    CHECK(v.risk == Risk::Mutate);
    CHECK(v.reason.find("ask") != std::string::npos);
}

// ---- yolo mode ----------------------------------------------------------------

TEST_CASE("yolo: nothing is gated, including deletes") {
    // This is the point of the mode, and PERMISSIONS.md says so out loud.
    for (const auto& c : {call("delete_file",        {{"path", "a.js"}}),
                          call("write_project_file", {{"path", "../out.js"}}),
                          call("run_command",        {{"command", "rm -rf x"}})}) {
        CHECK_FALSE(evaluate(PermissionMode::Yolo, c, kNothingRemembered).needs_prompt);
    }
}

// ---- prompts carry a usable reason ---------------------------------------------

TEST_CASE("every prompt names a reason and the thing at issue") {
    const std::vector<ToolCall> gated{
        call("delete_file",        {{"path", "src/old.css"}}),
        call("write_project_file", {{"path", "../escape.js"}}),
        call("write_project_file", {{"path", ".ultima/plan.md"}}),
        call("run_command",        {{"command", "pip install flask"}}),
        call("run_command",        {{"command", "node build.js"}}),
    };
    for (const auto& c : gated) {
        const auto v = evaluate(PermissionMode::Auto, c, kNothingRemembered);
        REQUIRE(v.needs_prompt);
        CHECK_FALSE(v.reason.empty());      // a prompt without a reason gets rubber-stamped
        CHECK_FALSE(v.subject.empty());     // and the user must see what it applies to
    }
}

TEST_CASE("the delete reason mentions that it cannot be undone") {
    const auto v = evaluate(PermissionMode::Auto, call("delete_file", {{"path", "a.js"}}),
                            kNothingRemembered);
    CHECK(v.reason.find("undone") != std::string::npos);
    CHECK(v.subject == "a.js");
}

// ---- session memory -------------------------------------------------------------

TEST_CASE("always-allow remembers one risk class, not all of them") {
    const std::vector<Risk> remembered{Risk::Destructive};

    CHECK_FALSE(evaluate(PermissionMode::Auto, call("delete_file", {{"path", "a.js"}}),
                         remembered).needs_prompt);
    // Approving deletes must not quietly approve reaching outside the project.
    CHECK(evaluate(PermissionMode::Auto, call("write_project_file", {{"path", "../x"}}),
                   remembered).needs_prompt);
    CHECK(evaluate(PermissionMode::Auto, call("run_command", {{"command", "npm install x"}}),
                   remembered).needs_prompt);
}

// ---- mode parsing ----------------------------------------------------------------

TEST_CASE("parse_permission_mode: unknown text falls back to the safe default") {
    CHECK(parse_permission_mode("auto") == PermissionMode::Auto);
    CHECK(parse_permission_mode("ASK")  == PermissionMode::Ask);
    CHECK(parse_permission_mode("yolo") == PermissionMode::Yolo);
    CHECK(parse_permission_mode("")         == PermissionMode::Auto);
    CHECK(parse_permission_mode("whatever") == PermissionMode::Auto);
    // A typo must never silently mean "allow everything".
    CHECK(parse_permission_mode("yolo!") == PermissionMode::Auto);
}

// ---- the name the model typed is not the name that runs -------------------------

TEST_CASE("classify: a misspelled delete is still destructive") {
    // Observed live, and the worst bug of this phase: the 7B called
    // "delete_project". The dispatcher resolves that to delete_file by its
    // leading verb, but classify() was comparing the raw string, so auto mode
    // saw an ordinary write and deleted the file with no prompt.
    CHECK(classify(call("delete_project", {{"path", "src/old.js"}})) == Risk::Destructive);
    CHECK(classify(call("delete-file",    {{"path", "a.js"}}))       == Risk::Destructive);
    CHECK(classify(call("remove_file",    {{"path", "a.js"}}))       == Risk::Destructive);
    CHECK(classify(call("rm",             {{"path", "a.js"}}))       == Risk::Destructive);

    const auto v = evaluate(PermissionMode::Auto,
                            call("delete_project", {{"path", "/src/old.js"}}),
                            kNothingRemembered);
    CHECK(v.needs_prompt);
    CHECK(v.risk == Risk::Destructive);
}

TEST_CASE("classify: a misspelled read is still free") {
    // The same canonicalization must not accidentally gate reads.
    CHECK(classify(call("read_project", {{"path", "a.js"}}))  == Risk::Read);
    CHECK(classify(call("list_dir"))                           == Risk::Read);
}

TEST_CASE("classify: a misspelled write is still an ordinary write") {
    CHECK(classify(call("write_project-file", {{"path", "a.js"}})) == Risk::Mutate);
    CHECK(classify(call("create_file",        {{"path", "a.js"}})) == Risk::Mutate);
}

TEST_CASE("the prompt always names its subject, whatever the argument is called") {
    // Observed live: the model passed "file_path", not "path", and the prompt
    // showed the user nothing to evaluate.
    for (const char* key : {"path", "file_path", "filepath", "filename", "file", "target"}) {
        nlohmann::json args;
        args[key] = "src/old.js";
        const auto v = evaluate(PermissionMode::Auto, call("delete_file", args),
                                kNothingRemembered);
        REQUIRE(v.needs_prompt);
        CHECK_MESSAGE(v.subject == "src/old.js", "subject lost for key: ", key);
    }
}

TEST_CASE("an unrecognised argument shape still shows something to judge") {
    const auto v = evaluate(PermissionMode::Auto,
                            call("delete_file", {{"weird_key", "src/old.js"}}),
                            kNothingRemembered);
    REQUIRE(v.needs_prompt);
    CHECK_FALSE(v.subject.empty());
    CHECK(v.subject.find("src/old.js") != std::string::npos);
}
