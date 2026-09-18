#include "ultima/server/skills.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <sstream>
#include <system_error>

namespace ultima::server {

namespace fs = std::filesystem;

namespace {

using core::ErrorCode;
using core::Result;
using core::fail;

// Measured across the installed skills: median 8.7 KB, p90 16.8 KB, largest
// 88 KB. On CPU inference every kilobyte is prefill time the user waits
// through, so the cap is well below the median on purpose — roughly 1500
// tokens, which is a page and a half of patterns. Truncation lands on a
// section boundary so a skill never ends mid-example.
constexpr std::size_t k_max_body_bytes = 6u * 1024u;

// Guards against walking a whole disk if someone points a root at C:\.
constexpr int k_max_depth = 4;

std::string_view trim(std::string_view s) {
    auto ws = [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
    std::size_t b = 0, e = s.size();
    while (b < e && ws(s[b])) ++b;
    while (e > b && ws(s[e - 1])) --e;
    return s.substr(b, e - b);
}

// Fold a skill name for comparison: lowercase, and '-' '_' ' ' all equal.
// Models reach for every spelling of a hyphenated name.
std::string fold(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '-' || c == '_' || c == ' ' || c == '.') continue;
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return out;
}

std::string lower(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    return out;
}

// Read at most `limit` bytes. Frontmatter lives at the top, so the index does
// not need the whole file — with 300+ skills that difference is the whole
// startup cost.
std::string read_head(const fs::path& p, std::size_t limit) {
    std::ifstream in{p, std::ios::binary};
    if (!in) return {};
    std::string buf(limit, '\0');
    in.read(buf.data(), static_cast<std::streamsize>(limit));
    buf.resize(static_cast<std::size_t>(in.gcount()));
    return buf;
}

// Strip one layer of [] and quotes from a frontmatter scalar or list.
std::vector<std::string> parse_tags(std::string_view v) {
    std::vector<std::string> out;
    std::string s{trim(v)};
    if (!s.empty() && s.front() == '[') s = s.substr(1);
    if (!s.empty() && s.back() == ']')  s.pop_back();
    std::istringstream ss{s};
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        std::string t{trim(tok)};
        if (!t.empty() && (t.front() == '"' || t.front() == '\'')) t = t.substr(1);
        if (!t.empty() && (t.back()  == '"' || t.back()  == '\'')) t.pop_back();
        if (!t.empty()) out.push_back(std::move(t));
    }
    return out;
}

std::string unquote(std::string_view v) {
    std::string s{trim(v)};
    if (s.size() >= 2 && (s.front() == '"' || s.front() == '\'') && s.back() == s.front()) {
        s = s.substr(1, s.size() - 2);
    }
    return s;
}

} // namespace

Skill parse_skill_frontmatter(std::string_view text) {
    Skill sk;
    const auto body = trim(text);
    if (body.rfind("---", 0) != 0) return sk;    // no frontmatter at all

    const auto end = body.find("\n---", 3);
    if (end == std::string_view::npos) return sk;

    std::istringstream in{std::string{body.substr(3, end - 3)}};
    std::string line;
    while (std::getline(in, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        const auto key = std::string{trim(std::string_view{line}.substr(0, colon))};
        const auto val = std::string_view{line}.substr(colon + 1);
        if      (key == "name")        sk.name        = unquote(val);
        else if (key == "description") sk.description = unquote(val);
        else if (key == "command")     sk.command     = unquote(val);
        else if (key == "tags")        sk.tags        = parse_tags(val);
    }
    return sk;
}

std::vector<fs::path> default_skill_roots(const fs::path& workspace) {
    // Order is priority: the project's own skills shadow everything, then what
    // Ultima ships, then the user's cross-agent tree.
    return {
        workspace / ".skills",
        fs::path{"examples"} / "knowledge" / "patterns",
        fs::path{"C:/.skills"},
    };
}

SkillIndex SkillIndex::discover(const std::vector<fs::path>& roots) {
    SkillIndex index;
    std::vector<std::string> seen;      // folded names already claimed

    for (std::size_t r = 0; r < roots.size(); ++r) {
        std::error_code ec;
        if (!fs::is_directory(roots[r], ec)) continue;   // absent root is fine

        auto it = fs::recursive_directory_iterator(
            roots[r], fs::directory_options::skip_permission_denied, ec);
        if (ec) continue;

        for (; it != fs::recursive_directory_iterator(); it.increment(ec)) {
            if (ec) { ec.clear(); continue; }
            if (it.depth() > k_max_depth) { it.disable_recursion_pending(); continue; }
            if (it->is_directory(ec)) continue;
            if (it->path().filename() != "SKILL.md") continue;

            Skill sk = parse_skill_frontmatter(read_head(it->path(), 2048));
            if (sk.name.empty()) {
                sk.name = it->path().parent_path().filename().string();
            }
            if (sk.name.empty()) continue;

            const auto folded = fold(sk.name);
            if (std::find(seen.begin(), seen.end(), folded) != seen.end()) {
                continue;   // an earlier root already owns this name
            }
            seen.push_back(folded);

            sk.path = it->path();
            sk.root = r;
            index.skills_.push_back(std::move(sk));
        }
    }

    std::sort(index.skills_.begin(), index.skills_.end(),
              [](const Skill& a, const Skill& b) {
                  if (a.root != b.root) return a.root < b.root;
                  return a.name < b.name;
              });
    return index;
}

const Skill* SkillIndex::find(std::string_view name) const {
    const auto want = fold(name);
    for (const auto& s : skills_) {
        if (fold(s.name) == want) return &s;
    }
    return nullptr;
}

std::vector<const Skill*> SkillIndex::search(std::string_view query,
                                             std::size_t limit) const {
    const auto q = lower(trim(query));
    std::vector<std::pair<int, const Skill*>> scored;
    if (q.empty()) return {};

    for (const auto& s : skills_) {
        const auto n = lower(s.name);
        int score = 0;
        if (n == q)                                   score = 100;
        else if (n.find(q) != std::string::npos)      score = 60;
        else if (fold(s.name).find(fold(q)) != std::string::npos) score = 50;
        else if (lower(s.description).find(q) != std::string::npos) score = 25;
        else {
            for (const auto& t : s.tags) {
                if (lower(t).find(q) != std::string::npos) { score = 20; break; }
            }
        }
        // Earlier roots win ties: the project's own skill beats a shipped one.
        if (score > 0) scored.emplace_back(score * 10 - static_cast<int>(s.root), &s);
    }
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<const Skill*> out;
    for (const auto& [sc, s] : scored) {
        if (out.size() >= limit) break;
        out.push_back(s);
    }
    return out;
}

std::vector<const Skill*> SkillIndex::in_root(std::size_t root) const {
    std::vector<const Skill*> out;
    for (const auto& s : skills_) {
        if (s.root == root) out.push_back(&s);
    }
    return out;
}

Result<std::string> load_skill_body(const Skill& skill) {
    std::ifstream in{skill.path, std::ios::binary};
    if (!in) {
        return fail(ErrorCode::FileNotFound,
                    fmt::format("cannot read skill '{}' at {}",
                                skill.name, skill.path.string()),
                    "skills");
    }
    std::string text{std::istreambuf_iterator<char>{in}, {}};

    // Drop the frontmatter: the model wants the patterns, not the metadata it
    // has already been shown in the index.
    std::string_view body{text};
    if (body.rfind("---", 0) == 0) {
        const auto end = body.find("\n---", 3);
        if (end != std::string_view::npos) {
            const auto nl = body.find('\n', end + 1);
            body = (nl == std::string_view::npos) ? std::string_view{} : body.substr(nl + 1);
        }
    }

    std::string out{trim(body)};
    if (out.size() > k_max_body_bytes) {
        out.resize(k_max_body_bytes);
        out += "\n\n... (skill truncated)";
    }
    return out;
}

std::string render_skills_prompt(const SkillIndex& index) {
    if (index.size() == 0) return {};

    // Only the first two roots are named individually. With 300+ skills
    // available, listing them all would cost more context than the skills are
    // worth, and a wrong name comes back with suggestions anyway.
    std::vector<const Skill*> near;
    for (std::size_t r = 0; r < 2; ++r) {
        for (auto* s : index.in_root(r)) near.push_back(s);
    }

    std::string out =
        "\n\n# Skills\n\n"
        "Reference sheets of canonical patterns. Call use_skill(name) BEFORE "
        "writing code in a domain one covers — the patterns in it are known to "
        "work, and guessing from memory is where small models go wrong.\n\n";

    if (!near.empty()) {
        out += "Available here: ";
        for (std::size_t i = 0; i < near.size(); ++i) {
            if (i) out += ", ";
            out += near[i]->name;
        }
        out += "\n";
    }
    const std::size_t rest = index.size() - near.size();
    if (rest > 0) {
        out += fmt::format("Plus {} more from the shared library — ask for one by "
                           "name (a wrong name returns the closest matches).\n", rest);
    }
    return out;
}

} // namespace ultima::server
