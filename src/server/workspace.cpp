#include "ultima/server/workspace.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <fstream>
#include <sstream>
#include <system_error>

namespace ultima::server {

namespace fs = std::filesystem;

namespace {

using core::ErrorCode;
using core::Failure;
using core::Result;
using core::fail;

// A file bigger than this is not something a local model should be fed, and
// not something the editor should try to render. Read returns the head with
// truncated=true.
constexpr std::size_t k_max_read_bytes = 2u * 1024u * 1024u;

// Directories never worth walking into for a project tree. Keeping these out
// matters more than usual here: the tree is fed to a small model, and a single
// node_modules would bury the actual source.
constexpr std::array<std::string_view, 9> k_skip_dirs{
    ".git", "node_modules", "build", "dist", "out",
    "__pycache__", ".venv", "venv", "target",
};

std::int64_t to_unix_seconds(fs::file_time_type t) {
    // file_clock's epoch is unspecified; go through the system clock.
    const auto sys = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
        t - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
    return std::chrono::duration_cast<std::chrono::seconds>(
        sys.time_since_epoch()).count();
}

std::string to_slash(const fs::path& p) {
    std::string s = p.generic_string();
    return s;
}

// Text vs binary, decided the way every editor decides it: a NUL byte in the
// first block means binary. Cheap and wrong only for UTF-16, which we do not
// claim to support.
bool looks_binary(std::string_view head) {
    const std::size_t n = std::min<std::size_t>(head.size(), 8000);
    for (std::size_t i = 0; i < n; ++i) {
        if (head[i] == '\0') return true;
    }
    return false;
}

} // namespace

bool is_reserved_device_name(std::string_view segment) noexcept {
    // Strip an extension: "NUL.txt" is still the NUL device on Windows.
    const auto dot = segment.find('.');
    std::string_view stem = (dot == std::string_view::npos)
                          ? segment : segment.substr(0, dot);
    if (stem.size() < 3 || stem.size() > 4) return false;

    std::string upper;
    upper.reserve(stem.size());
    for (char c : stem) {
        upper.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
    }
    if (upper == "CON" || upper == "PRN" || upper == "AUX" || upper == "NUL") return true;
    if (upper.size() == 4 && (upper.compare(0, 3, "COM") == 0 || upper.compare(0, 3, "LPT") == 0)) {
        return upper[3] >= '1' && upper[3] <= '9';
    }
    return false;
}

Result<Workspace> Workspace::open(const fs::path& root) {
    std::error_code ec;
    if (!fs::exists(root, ec) || ec) {
        return fail(ErrorCode::FileNotFound,
                    fmt::format("workspace '{}' does not exist", root.string()),
                    "workspace");
    }
    if (!fs::is_directory(root, ec) || ec) {
        return fail(ErrorCode::InvalidArgument,
                    fmt::format("workspace '{}' is not a directory", root.string()),
                    "workspace");
    }
    const fs::path canon = fs::canonical(root, ec);
    if (ec) {
        return fail(ErrorCode::IoFailure,
                    fmt::format("cannot canonicalize workspace '{}': {}",
                                root.string(), ec.message()),
                    "workspace");
    }
    return Workspace{canon};
}

Result<fs::path> Workspace::resolve(std::string_view rel) const {
    // 1. Reject control characters outright. A NUL would truncate the path
    //    inside any C API downstream; the rest have no business in a filename.
    for (unsigned char c : rel) {
        if (c < 0x20 || c == 0x7F) {
            return fail(ErrorCode::InvalidArgument,
                        "path contains a control character", "workspace");
        }
    }

    // 2. Normalize separators so the checks below see one spelling.
    std::string norm{rel};
    std::replace(norm.begin(), norm.end(), '\\', '/');

    // 3. Reject absolute paths in every spelling: POSIX root, UNC, drive letter.
    if (!norm.empty() && norm.front() == '/') {
        return fail(ErrorCode::InvalidArgument,
                    "path must be relative to the workspace, got an absolute path",
                    "workspace");
    }
    if (norm.size() >= 2 && norm[1] == ':' &&
        std::isalpha(static_cast<unsigned char>(norm[0]))) {
        return fail(ErrorCode::InvalidArgument,
                    "path must be relative to the workspace, got a drive letter",
                    "workspace");
    }

    // 4. Walk the segments. Any `..` is refused rather than resolved — a
    //    request that needs to climb out is a bug or an attack, and either way
    //    the caller should hear about it.
    fs::path joined = root_;
    std::istringstream ss{norm};
    std::string seg;
    while (std::getline(ss, seg, '/')) {
        if (seg.empty() || seg == ".") continue;
        if (seg == "..") {
            return fail(ErrorCode::InvalidArgument,
                        "path contains a '..' segment", "workspace");
        }
        if (is_reserved_device_name(seg)) {
            return fail(ErrorCode::InvalidArgument,
                        fmt::format("'{}' is a reserved device name", seg),
                        "workspace");
        }
        joined /= seg;
    }

    // 5. Resolve symlinks and re-check containment. Steps 3-4 handle the path
    //    *string*; this handles the filesystem, where a symlink inside the
    //    workspace can still point outside it.
    std::error_code ec;
    fs::path canon = fs::weakly_canonical(joined, ec);
    if (ec) canon = joined.lexically_normal();

    const auto rootstr = root_.generic_string();
    const auto canonstr = canon.generic_string();
    const bool inside = canonstr == rootstr ||
        (canonstr.size() > rootstr.size() &&
         canonstr.compare(0, rootstr.size(), rootstr) == 0 &&
         canonstr[rootstr.size()] == '/');
    if (!inside) {
        return fail(ErrorCode::InvalidArgument,
                    "path resolves outside the workspace", "workspace");
    }
    return canon;
}

Result<std::vector<Workspace::Entry>> Workspace::list(std::string_view rel) const {
    auto dir = resolve(rel);
    if (!dir) return Failure{dir.error()};

    std::error_code ec;
    if (!fs::is_directory(*dir, ec)) {
        return fail(ErrorCode::InvalidArgument,
                    fmt::format("'{}' is not a directory", std::string{rel}),
                    "workspace");
    }

    std::vector<Entry> out;
    for (const auto& e : fs::directory_iterator(*dir, ec)) {
        if (ec) break;
        Entry entry;
        entry.path   = to_slash(fs::relative(e.path(), root_, ec));
        entry.is_dir = e.is_directory(ec);
        entry.size   = entry.is_dir ? 0u : static_cast<std::uint64_t>(e.file_size(ec));
        const auto mt = e.last_write_time(ec);
        entry.mtime  = ec ? 0 : to_unix_seconds(mt);
        ec.clear();
        out.push_back(std::move(entry));
    }
    std::sort(out.begin(), out.end(), [](const Entry& a, const Entry& b) {
        if (a.is_dir != b.is_dir) return a.is_dir;   // directories first
        return a.path < b.path;
    });
    return out;
}

Result<std::vector<Workspace::Entry>> Workspace::list_recursive() const {
    std::vector<Entry> out;
    std::error_code ec;

    auto it = fs::recursive_directory_iterator(
        root_, fs::directory_options::skip_permission_denied, ec);
    if (ec) {
        return fail(ErrorCode::IoFailure,
                    fmt::format("cannot walk workspace: {}", ec.message()),
                    "workspace");
    }
    for (; it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (ec) { ec.clear(); continue; }
        const auto name = it->path().filename().string();
        if (it->is_directory(ec)) {
            const bool skip = std::find(k_skip_dirs.begin(), k_skip_dirs.end(),
                                        std::string_view{name}) != k_skip_dirs.end();
            if (skip) { it.disable_recursion_pending(); }
            continue;
        }
        Entry entry;
        entry.path   = to_slash(fs::relative(it->path(), root_, ec));
        entry.is_dir = false;
        entry.size   = static_cast<std::uint64_t>(it->file_size(ec));
        const auto mt = it->last_write_time(ec);
        entry.mtime  = ec ? 0 : to_unix_seconds(mt);
        ec.clear();
        out.push_back(std::move(entry));
    }
    std::sort(out.begin(), out.end(),
              [](const Entry& a, const Entry& b) { return a.path < b.path; });
    return out;
}

Result<Workspace::FileRead> Workspace::read(std::string_view rel) const {
    auto path = resolve(rel);
    if (!path) return Failure{path.error()};

    std::error_code ec;
    if (!fs::is_regular_file(*path, ec)) {
        return fail(ErrorCode::FileNotFound,
                    fmt::format("'{}' is not a file", std::string{rel}),
                    "workspace");
    }

    std::ifstream in{*path, std::ios::binary};
    if (!in) {
        return fail(ErrorCode::IoFailure,
                    fmt::format("cannot open '{}'", std::string{rel}),
                    "workspace");
    }

    FileRead out;
    out.content.resize(k_max_read_bytes);
    in.read(out.content.data(), static_cast<std::streamsize>(k_max_read_bytes));
    out.content.resize(static_cast<std::size_t>(in.gcount()));
    out.truncated = (in.gcount() == static_cast<std::streamsize>(k_max_read_bytes))
                 && (fs::file_size(*path, ec) > k_max_read_bytes);

    if (looks_binary(out.content)) {
        return fail(ErrorCode::InvalidArgument,
                    fmt::format("'{}' looks like a binary file; the editor only "
                                "handles text", std::string{rel}),
                    "workspace");
    }
    return out;
}

Result<void> Workspace::write(std::string_view rel, std::string_view content) {
    auto path = resolve(rel);
    if (!path) return Failure{path.error()};
    if (*path == root_) {
        return fail(ErrorCode::InvalidArgument,
                    "cannot write to the workspace root itself", "workspace");
    }

    std::error_code ec;
    fs::create_directories(path->parent_path(), ec);
    if (ec) {
        return fail(ErrorCode::IoFailure,
                    fmt::format("cannot create parent of '{}': {}",
                                std::string{rel}, ec.message()),
                    "workspace");
    }

    std::ofstream out{*path, std::ios::binary | std::ios::trunc};
    if (!out) {
        return fail(ErrorCode::IoFailure,
                    fmt::format("cannot write '{}'", std::string{rel}),
                    "workspace");
    }
    out.write(content.data(), static_cast<std::streamsize>(content.size()));
    if (!out) {
        return fail(ErrorCode::IoFailure,
                    fmt::format("write to '{}' failed", std::string{rel}),
                    "workspace");
    }
    return {};
}

Result<std::size_t> Workspace::edit(std::string_view rel,
                                    const std::vector<Edit>& edits) {
    auto current = read(rel);
    if (!current) return Failure{current.error()};
    if (current->truncated) {
        return fail(ErrorCode::InvalidArgument,
                    fmt::format("'{}' is too large to edit safely", std::string{rel}),
                    "workspace");
    }

    // Apply to a copy: a failure part-way through must leave the file alone,
    // otherwise a bad edit list corrupts the user's work.
    std::string text = std::move(current->content);
    std::size_t applied = 0;

    for (std::size_t i = 0; i < edits.size(); ++i) {
        const Edit& e = edits[i];
        if (e.find.empty()) {
            return fail(ErrorCode::InvalidArgument,
                        fmt::format("edit {}: 'find' is empty", i), "workspace");
        }
        const auto first = text.find(e.find);
        if (first == std::string::npos) {
            return fail(ErrorCode::NotFound,
                        fmt::format("edit {}: no match for the text to replace "
                                    "in '{}'", i, std::string{rel}),
                        "workspace");
        }
        if (e.all) {
            std::size_t pos = 0;
            while ((pos = text.find(e.find, pos)) != std::string::npos) {
                text.replace(pos, e.find.size(), e.replace);
                pos += e.replace.size();
                ++applied;
            }
        } else {
            const auto second = text.find(e.find, first + e.find.size());
            if (second != std::string::npos) {
                return fail(ErrorCode::InvalidArgument,
                            fmt::format("edit {}: the text to replace appears more "
                                        "than once in '{}'; pass all=true or "
                                        "include more surrounding context",
                                        i, std::string{rel}),
                            "workspace");
            }
            text.replace(first, e.find.size(), e.replace);
            ++applied;
        }
    }

    if (auto w = write(rel, text); !w) return Failure{w.error()};
    return applied;
}

Result<void> Workspace::rename(std::string_view from, std::string_view to) {
    auto src = resolve(from);
    if (!src) return Failure{src.error()};
    auto dst = resolve(to);
    if (!dst) return Failure{dst.error()};

    std::error_code ec;
    if (!fs::exists(*src, ec)) {
        return fail(ErrorCode::FileNotFound,
                    fmt::format("'{}' does not exist", std::string{from}),
                    "workspace");
    }
    if (fs::exists(*dst, ec)) {
        return fail(ErrorCode::InvalidArgument,
                    fmt::format("'{}' already exists", std::string{to}),
                    "workspace");
    }
    fs::create_directories(dst->parent_path(), ec);
    ec.clear();
    fs::rename(*src, *dst, ec);
    if (ec) {
        return fail(ErrorCode::IoFailure,
                    fmt::format("rename '{}' -> '{}' failed: {}",
                                std::string{from}, std::string{to}, ec.message()),
                    "workspace");
    }
    return {};
}

Result<void> Workspace::remove(std::string_view rel) {
    auto path = resolve(rel);
    if (!path) return Failure{path.error()};
    if (*path == root_) {
        return fail(ErrorCode::InvalidArgument,
                    "refusing to delete the workspace root", "workspace");
    }

    std::error_code ec;
    if (!fs::exists(*path, ec)) {
        return fail(ErrorCode::FileNotFound,
                    fmt::format("'{}' does not exist", std::string{rel}),
                    "workspace");
    }
    const auto removed = fs::remove_all(*path, ec);
    if (ec || removed == 0) {
        return fail(ErrorCode::IoFailure,
                    fmt::format("cannot delete '{}': {}", std::string{rel},
                                ec ? ec.message() : "nothing removed"),
                    "workspace");
    }
    return {};
}

Result<void> Workspace::mkdir(std::string_view rel) {
    auto path = resolve(rel);
    if (!path) return Failure{path.error()};

    std::error_code ec;
    fs::create_directories(*path, ec);
    if (ec) {
        return fail(ErrorCode::IoFailure,
                    fmt::format("cannot create '{}': {}", std::string{rel},
                                ec.message()),
                    "workspace");
    }
    return {};
}

bool Workspace::exists(std::string_view rel) const {
    auto path = resolve(rel);
    if (!path) return false;
    std::error_code ec;
    return fs::exists(*path, ec);
}

} // namespace ultima::server
