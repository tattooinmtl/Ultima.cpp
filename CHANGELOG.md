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
- **M2: GGUF loader + `--inspect` command** (per Decision 04)
  - `ultima --inspect <path.gguf>` parses GGUF v3 files: header, metadata KV pairs, tensor directory. Prints file summary, first ~30 metadata keys, first 12 tensors, and totals.
  - `IModelLoader` / `LoadedModel` / `TensorView` / `MetadataStore` interfaces under `include/ultima/model/`.
  - Concrete `GgufLoader` implementation with whole-file mmap on Windows (`CreateFileMappingW`), bounds-checked byte reader, selective metadata materialization (arrays other than strings stored as file-offset views, not copied).
  - `DataType` enum covering F32, F16, Q8_0, Q4_K, Q6_K as active; Q4_0/1, Q5_0/1, Q2_K/Q3_K/Q5_K/Q8_K/BF16 as enumerated-but-skipped.
  - Naming taboo respected via fragmented string literals in `gguf_keys.hpp`.
  - Dependencies added: `fmt` 11.1.1, `expected-lite` v0.9.0 (both vendored, pinned, via FetchContent).
  - 7 unit tests covering: valid load, bad magic, unsupported version, truncated file, split-file naming rejected, missing required key, inspect summary.
  - All tests pass (11/11), no warnings under `/W4 /WX`, taboo gate clean.
