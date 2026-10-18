#include <doctest/doctest.h>

#include "ultima/server/workspace.hpp"

#include <filesystem>
#include <fstream>
#include <ostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using ultima::server::Workspace;

namespace {

// A throwaway directory tree, removed when the fixture goes out of scope.
class TempWorkspace {
public:
    TempWorkspace() {
        static int counter = 0;
        root_ = fs::temp_directory_path() /
                ("ultima_ws_test_" + std::to_string(++counter) + "_" +
                 std::to_string(reinterpret_cast<std::uintptr_t>(this)));
        fs::create_directories(root_ / "src");
        write("README.md", "# hello\n");
        write("src/main.js", "const a = 1;\nconst b = 2;\nconst a2 = 1;\n");
    }
    ~TempWorkspace() {
        std::error_code ec;
        fs::remove_all(root_, ec);
    }

    void write(const std::string& rel, const std::string& content) const {
        const auto p = root_ / rel;
        fs::create_directories(p.parent_path());
        std::ofstream out{p, std::ios::binary | std::ios::trunc};
        out << content;
    }
    std::string read(const std::string& rel) const {
        std::ifstream in{root_ / rel, std::ios::binary};
        return std::string{std::istreambuf_iterator<char>{in}, {}};
    }

    const fs::path& root() const { return root_; }
    Workspace open() const {
        auto ws = Workspace::open(root_);
        REQUIRE(static_cast<bool>(ws));
        return std::move(*ws);
    }

private:
    fs::path root_;
};

} // namespace

// ---- resolve(): the security surface -----------------------------------------

TEST_CASE("workspace: accepts ordinary relative paths") {
    TempWorkspace tmp;
    auto ws = tmp.open();

    for (const char* good : {"README.md", "src/main.js", "src", "a/b/c.txt", ""}) {
        auto r = ws.resolve(good);
        CHECK_MESSAGE(static_cast<bool>(r), "should accept: ", good);
    }
}

TEST_CASE("workspace: refuses every spelling of escaping the root") {
    TempWorkspace tmp;
    auto ws = tmp.open();

    const std::vector<std::string> bad{
        "..",
        "../outside.txt",
        "src/../../outside.txt",
        "src/../..",
        "a/./../../b",
        "..\\outside.txt",          // Windows separator
        "src\\..\\..\\outside",
    };
    for (const auto& p : bad) {
        auto r = ws.resolve(p);
        CHECK_MESSAGE(!r, "should refuse: ", p);
    }
}

TEST_CASE("workspace: refuses absolute paths") {
    TempWorkspace tmp;
    auto ws = tmp.open();

    const std::vector<std::string> bad{
        "/etc/passwd",
        "/",
        "C:/Windows/System32/config",
        "c:\\Windows\\System32",
        "\\\\server\\share\\file",   // UNC
        "\\Windows",
    };
    for (const auto& p : bad) {
        auto r = ws.resolve(p);
        CHECK_MESSAGE(!r, "should refuse: ", p);
    }
}

TEST_CASE("workspace: refuses control characters and reserved device names") {
    TempWorkspace tmp;
    auto ws = tmp.open();

    CHECK_FALSE(ws.resolve(std::string("evil\0.txt", 9)));
    CHECK_FALSE(ws.resolve("bad\nname.txt"));
    CHECK_FALSE(ws.resolve("NUL"));
    CHECK_FALSE(ws.resolve("nul.txt"));
    CHECK_FALSE(ws.resolve("COM1"));
    CHECK_FALSE(ws.resolve("src/LPT9.log"));
    // Names that merely start the same way are fine.
    CHECK(ws.resolve("console.js"));
    CHECK(ws.resolve("COM10.txt"));
    CHECK(ws.resolve("nullable.ts"));
}

TEST_CASE("is_reserved_device_name: classifies the Windows device names") {
    using ultima::server::is_reserved_device_name;
    CHECK(is_reserved_device_name("CON"));
    CHECK(is_reserved_device_name("con"));
    CHECK(is_reserved_device_name("NUL.txt"));
    CHECK(is_reserved_device_name("COM3"));
    CHECK(is_reserved_device_name("LPT1.log"));
    CHECK_FALSE(is_reserved_device_name("COM0"));
    CHECK_FALSE(is_reserved_device_name("CONSOLE"));
    CHECK_FALSE(is_reserved_device_name("index.js"));
    CHECK_FALSE(is_reserved_device_name(""));
}

TEST_CASE("workspace: a refused path never becomes a clamped good path") {
    // The failure mode this guards: silently turning "../x" into "<root>/x".
    TempWorkspace tmp;
    auto ws = tmp.open();
    auto r = ws.resolve("../README.md");
    REQUIRE_FALSE(r);
    // And the error says why, rather than being a bare failure.
    CHECK(r.error().message.find("..") != std::string::npos);
}

// ---- read / write ------------------------------------------------------------

TEST_CASE("workspace: reads and writes text files") {
    TempWorkspace tmp;
    auto ws = tmp.open();

    auto got = ws.read("README.md");
    REQUIRE(static_cast<bool>(got));
    CHECK(got->content == "# hello\n");
    CHECK_FALSE(got->truncated);

    REQUIRE(ws.write("notes/todo.txt", "buy milk\n"));
    CHECK(tmp.read("notes/todo.txt") == "buy milk\n");   // parents auto-created
}

TEST_CASE("workspace: refuses to read a binary file rather than returning mojibake") {
    TempWorkspace tmp;
    tmp.write("logo.png", std::string("\x89PNG\r\n\x1a\n\0\0\0 binary", 20));
    auto ws = tmp.open();

    auto got = ws.read("logo.png");
    REQUIRE_FALSE(got);
    CHECK(got.error().message.find("binary") != std::string::npos);
}

TEST_CASE("workspace: reading a missing file fails, it does not return empty") {
    TempWorkspace tmp;
    auto ws = tmp.open();
    CHECK_FALSE(ws.read("nope.txt"));
}

TEST_CASE("workspace: will not write over the root itself") {
    TempWorkspace tmp;
    auto ws = tmp.open();
    CHECK_FALSE(ws.write("", "oops"));
}

// ---- edit --------------------------------------------------------------------

TEST_CASE("workspace: edit replaces a unique match") {
    TempWorkspace tmp;
    auto ws = tmp.open();

    auto n = ws.edit("src/main.js", {{"const b = 2;", "const b = 99;", false}});
    REQUIRE(static_cast<bool>(n));
    CHECK(*n == 1);
    CHECK(tmp.read("src/main.js") == "const a = 1;\nconst b = 99;\nconst a2 = 1;\n");
}

TEST_CASE("workspace: edit refuses an ambiguous match instead of guessing") {
    TempWorkspace tmp;
    auto ws = tmp.open();
    const auto before = tmp.read("src/main.js");

    // "const a" matches both "const a = 1" and "const a2 = 1".
    auto n = ws.edit("src/main.js", {{"const a", "let a", false}});
    REQUIRE_FALSE(n);
    CHECK(n.error().message.find("more than once") != std::string::npos);
    CHECK(tmp.read("src/main.js") == before);            // untouched
}

TEST_CASE("workspace: edit with all=true replaces every occurrence") {
    TempWorkspace tmp;
    auto ws = tmp.open();

    auto n = ws.edit("src/main.js", {{"const", "let", true}});
    REQUIRE(static_cast<bool>(n));
    CHECK(*n == 3);
    CHECK(tmp.read("src/main.js").find("const") == std::string::npos);
}

TEST_CASE("workspace: a failing edit leaves the file untouched") {
    TempWorkspace tmp;
    auto ws = tmp.open();
    const auto before = tmp.read("src/main.js");

    // First edit is fine, second cannot match — neither may land.
    auto n = ws.edit("src/main.js", {
        {"const a = 1;", "const a = 7;", false},
        {"this text is not in the file", "x", false},
    });
    REQUIRE_FALSE(n);
    CHECK(tmp.read("src/main.js") == before);
}

TEST_CASE("workspace: edit reports a miss rather than silently doing nothing") {
    TempWorkspace tmp;
    auto ws = tmp.open();
    auto n = ws.edit("src/main.js", {{"nonexistent", "x", false}});
    REQUIRE_FALSE(n);
    CHECK(n.error().message.find("no match") != std::string::npos);
}

// ---- rename / remove / mkdir / list -------------------------------------------

TEST_CASE("workspace: rename moves a file and refuses to clobber") {
    TempWorkspace tmp;
    auto ws = tmp.open();

    REQUIRE(ws.rename("README.md", "docs/README.md"));
    CHECK_FALSE(ws.exists("README.md"));
    CHECK(ws.exists("docs/README.md"));

    tmp.write("a.txt", "a");
    tmp.write("b.txt", "b");
    auto clobber = ws.rename("a.txt", "b.txt");
    CHECK_FALSE(clobber);
    CHECK(tmp.read("b.txt") == "b");                     // not overwritten
}

TEST_CASE("workspace: rename cannot be used to escape the workspace") {
    TempWorkspace tmp;
    auto ws = tmp.open();
    CHECK_FALSE(ws.rename("README.md", "../escaped.md"));
    CHECK_FALSE(ws.rename("../outside.md", "inside.md"));
    CHECK(ws.exists("README.md"));
}

TEST_CASE("workspace: remove deletes, and refuses the root") {
    TempWorkspace tmp;
    auto ws = tmp.open();

    REQUIRE(ws.remove("README.md"));
    CHECK_FALSE(ws.exists("README.md"));
    CHECK_FALSE(ws.remove("README.md"));                 // already gone
    CHECK_FALSE(ws.remove(""));                          // the root
    CHECK(fs::exists(tmp.root()));
}

TEST_CASE("workspace: list returns directories first, paths workspace-relative") {
    TempWorkspace tmp;
    auto ws = tmp.open();

    auto entries = ws.list("");
    REQUIRE(static_cast<bool>(entries));
    REQUIRE(entries->size() >= 2);
    CHECK(entries->front().is_dir);                      // "src" sorts first
    for (const auto& e : *entries) {
        CHECK(e.path.find(':') == std::string::npos);    // not absolute
        CHECK(e.path.find('\\') == std::string::npos);   // forward slashes
    }
}

TEST_CASE("workspace: recursive listing skips dependency and build dirs") {
    TempWorkspace tmp;
    tmp.write("node_modules/left-pad/index.js", "module.exports = 1;");
    tmp.write(".git/config", "[core]");
    tmp.write("build/out.o", "x");
    tmp.write("src/deep/nested.ts", "export {};");
    auto ws = tmp.open();

    auto all = ws.list_recursive();
    REQUIRE(static_cast<bool>(all));

    bool saw_nested = false;
    for (const auto& e : *all) {
        CHECK(e.path.find("node_modules/") == std::string::npos);
        CHECK(e.path.find(".git/") == std::string::npos);
        CHECK(e.path.find("build/") == std::string::npos);
        if (e.path == "src/deep/nested.ts") saw_nested = true;
    }
    CHECK(saw_nested);
}

TEST_CASE("workspace: mkdir creates nested directories") {
    TempWorkspace tmp;
    auto ws = tmp.open();
    REQUIRE(ws.mkdir("a/b/c"));
    CHECK(fs::is_directory(tmp.root() / "a" / "b" / "c"));
}

TEST_CASE("workspace: opening a missing or non-directory root fails") {
    CHECK_FALSE(Workspace::open(fs::temp_directory_path() / "ultima_no_such_dir_xyz"));

    TempWorkspace tmp;
    CHECK_FALSE(Workspace::open(tmp.root() / "README.md"));
}
