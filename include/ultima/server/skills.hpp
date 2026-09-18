#pragma once

#include "ultima/core/error.hpp"

#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace ultima::server {

// Skill discovery (Decision 11).
//
// A skill is a SKILL.md file with YAML-ish frontmatter and a markdown body of
// canonical patterns for one domain. Small models are much better at, say,
// Three.js when handed the boilerplate than when recalling it, so this is the
// highest-leverage context the runtime can supply.
//
// Roots are walked in order and the FIRST match for a name wins, so a project
// can shadow a shipped skill without editing it.
struct Skill {
    std::string           name;          // frontmatter `name`, else the folder name
    std::string           description;   // one line, shown when suggesting
    std::string           command;       // optional slash command, e.g. "/canvas"
    std::vector<std::string> tags;
    std::filesystem::path path;          // the SKILL.md itself
    std::size_t           root = 0;      // index into the roots list
};

// Frontmatter only — exposed for tests. A file with no frontmatter yields a
// Skill whose name is empty, which discovery then fills from the folder name.
Skill parse_skill_frontmatter(std::string_view text);

class SkillIndex {
public:
    // Walk each root for SKILL.md files. Missing roots are skipped silently:
    // a machine without the cross-agent skills tree is not misconfigured.
    static SkillIndex discover(const std::vector<std::filesystem::path>& roots);

    // Exact match, case- and separator-insensitive ("html5-canvas" ==
    // "HTML5_Canvas"), because models are not careful about either.
    const Skill* find(std::string_view name) const;

    // Ranked substring search over name, description and tags. Used to turn a
    // wrong name into a useful suggestion rather than a dead end.
    std::vector<const Skill*> search(std::string_view query, std::size_t limit) const;

    std::vector<const Skill*> in_root(std::size_t root) const;

    const std::vector<Skill>& all() const noexcept { return skills_; }
    std::size_t size() const noexcept { return skills_.size(); }

private:
    std::vector<Skill> skills_;
};

// The skill's markdown body, with the frontmatter stripped. Capped: a runaway
// SKILL.md would eat the whole context window of a small model.
core::Result<std::string> load_skill_body(const Skill& skill);

// The `# Skills` section for the system prompt. Deliberately compact — names
// of the project's own skills plus a count of the rest. With 300+ skills
// available, listing every description would cost more context than the skills
// are worth; a wrong name comes back with suggestions instead.
std::string render_skills_prompt(const SkillIndex& index);

// The roots to walk, in priority order, given the current workspace.
std::vector<std::filesystem::path> default_skill_roots(const std::filesystem::path& workspace);

} // namespace ultima::server
