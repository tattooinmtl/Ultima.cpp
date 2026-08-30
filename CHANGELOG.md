# Changelog

All notable changes to Ultima.cpp are recorded here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- Initial architecture scaffold (M0):
  - CMake build system with `windows-msvc-*` presets
  - Compiler flags module, warnings-as-errors, no RTTI
  - Interface header layout under `include/ultima/`
  - Dependency policy: doctest, fmt, nlohmann/json, cpp-httplib, xxhash, expected-lite (all vendored via FetchContent, pinned)
  - `ultima-cli` executable stub that prints version
  - `doctest`-based test harness with a sanity test
  - CI taboo-check script (naming rule enforcement)
  - Directory structure: src/, include/, apps/, tests/, benchmarks/, examples/, third_party/, docs/, scripts/, launcher/, webui/
- Design roadmap (`roadmap01.md`) documenting decisions 01, 02, 03, 03b
- `start.bat` — one-click Windows launcher stopgap that auto-detects VS 2022 via `vswhere`, builds if needed, then runs `ultima --version` / `--help`. Superseded by the Go/Wails launcher (Decision 03) once implemented.
