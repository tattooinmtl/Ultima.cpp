#pragma once

#include "ultima/core/error.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace ultima::server {

// A single directory the coding agent and the editor are allowed to touch,
// plus every file operation they need.
//
// The security surface of the whole Coder feature is one function: resolve().
// Nothing else in the server turns a client-supplied string into a path, so
// path-escape review starts and ends there.
//
// See roadmap01.md Decision 18 §18.2.
class Workspace {
public:
    // `root` is canonicalized on construction. A root that does not exist or
    // is not a directory is an error the caller must handle — the server
    // refuses to start rather than silently serving the wrong tree.
    static core::Result<Workspace> open(const std::filesystem::path& root);

    const std::filesystem::path& root() const noexcept { return root_; }

    // Turn a client-supplied relative path into an absolute one known to be
    // inside the workspace.
    //
    // Refuses, with an error naming the reason: absolute paths, drive letters,
    // any `..` segment, NUL or control characters, Windows reserved device
    // names, and anything that resolves (through symlinks) to a location
    // outside the root. An empty path means the workspace root itself.
    //
    // Never clamps a bad path into a good one — a refusal is always visible.
    core::Result<std::filesystem::path> resolve(std::string_view rel) const;

    struct Entry {
        std::string   path;     // workspace-relative, forward slashes
        bool          is_dir = false;
        std::uint64_t size  = 0;
        std::int64_t  mtime = 0;   // unix seconds
    };

    // Direct children of `rel`, sorted directories-first then by name.
    core::Result<std::vector<Entry>> list(std::string_view rel) const;

    // Every file under the workspace, recursively. Skips the noise a project
    // tree always has (.git, node_modules, build outputs) so a small model's
    // context is not spent on them.
    core::Result<std::vector<Entry>> list_recursive() const;

    struct FileRead {
        std::string content;
        bool        truncated = false;
    };

    // Read a text file. Binary content is refused rather than returned as
    // mojibake; oversized files come back truncated with the flag set, never
    // silently cut.
    core::Result<FileRead> read(std::string_view rel) const;

    // Create or overwrite. Parent directories are created as needed.
    core::Result<void> write(std::string_view rel, std::string_view content);

    struct Edit {
        std::string find;
        std::string replace;
        bool        all = false;   // false => the match must be unique
    };

    // Exact-string replacement, applied in order. Chosen over diffs and line
    // numbers deliberately: see Decision 18 §18.3. A `find` that does not
    // appear, or appears more than once when `all` is false, fails the whole
    // call and leaves the file untouched.
    core::Result<std::size_t> edit(std::string_view rel,
                                   const std::vector<Edit>& edits);

    core::Result<void> rename(std::string_view from, std::string_view to);
    core::Result<void> remove(std::string_view rel);
    core::Result<void> mkdir(std::string_view rel);

    // True when `rel` names something that exists.
    bool exists(std::string_view rel) const;

private:
    explicit Workspace(std::filesystem::path root) : root_{std::move(root)} {}
    std::filesystem::path root_;
};

// Exposed for tests: is this path segment a Windows reserved device name
// (CON, PRN, AUX, NUL, COM1-9, LPT1-9), with or without an extension?
bool is_reserved_device_name(std::string_view segment) noexcept;

} // namespace ultima::server
