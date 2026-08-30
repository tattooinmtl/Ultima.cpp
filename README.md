# Ultima.cpp

Independent C++ inference runtime for local LLMs, with a persistent-memory subsystem designed in from the beginning.

**Status:** v0.1 pre-alpha. Architecture, scaffold, and build system in place. No functional inference yet.

See [`roadmap01.md`](roadmap01.md) for locked-in design decisions. See [`docs/history/original-spec.md`](docs/history/original-spec.md) for the original project vision.

## Requirements (Windows dev target)

- Windows 11
- Visual Studio 2022 (or VS Build Tools 2022) with the "Desktop development with C++" workload
- CMake 3.24 or newer
- Git

Linux and macOS builds are reserved in the build system but not gated by v0.1 CI.

## Quick start

**Easiest:** double-click `start.bat` in the project root. It auto-detects Visual Studio, builds if needed, and runs the binary. Wraps everything below in one click. Temporary stopgap until the Go/Wails launcher lands (Decision 03).

## Build from a shell

From a **Developer PowerShell for VS 2022** (or any shell if `cmake` can find MSVC via `vswhere`):

```powershell
cd C:\Ultima.cpp
cmake --preset windows-msvc-release
cmake --build --preset windows-msvc-release
.\build\windows-msvc-release\bin\Release\ultima.exe --version
```

Expected output:

```
ultima 0.1.0-alpha
```

## Test

```powershell
ctest --preset windows-msvc-release
```

## Project layout

```
ultima.cpp/
├── CMakeLists.txt          top-level build
├── CMakePresets.json       one-command builds
├── cmake/                  compiler flags, deps, warnings
├── include/ultima/         public headers (interfaces)
├── src/                    implementations
├── apps/
│   ├── ultima-cli/         the ultima executable
│   └── ultima-server/      (stub, wired up post-M8)
├── tests/                  unit + integration tests
├── benchmarks/             perf harness (stub)
├── examples/               example programs (stub)
├── scripts/                CI helpers
├── third_party/            vendored deps (via FetchContent)
├── docs/                   architecture + history
├── launcher/               Go/Wails launcher (stub)
└── webui/                  static chat UI (stub)
```

## License

MIT. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for dependency attributions.
