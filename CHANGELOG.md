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
- **M1 chunk 1: tensor engine foundations** (per Decision 05)
  - `include/ultima/tensor/aligned_alloc.hpp` + `.cpp` — 64-byte-aligned allocator (Windows `_aligned_malloc`, POSIX `posix_memalign`), typed `aligned_alloc_n<T>` helper, `AlignedDeleter` for `unique_ptr`.
  - `include/ultima/tensor/tensor.hpp` + `.cpp` — `Tensor` class with two storage modes: Owned (mutable, aligned heap) and View (immutable wrapper for mmap'd weights). Move-only, RAII.
  - `include/ultima/runtime/thread_pool.hpp` + `.cpp` — fixed-size worker pool with blocking `parallel_for(n, work_fn)`. No OpenMP/TBB dependency.
  - `include/ultima/kernels/cpu_features.hpp` + `.cpp` — `__cpuid`-based feature detection (AVX2, FMA, AVX-512F, SSE2, brand string), cached, thread-safe. `require_v01_baseline_or_die()` for startup gating.
  - `include/ultima/kernels/matvec.hpp` — kernel declarations
  - `src/kernels/matvec_f32_f32.cpp` — first kernel: scalar oracle + AVX2+FMA path (8-lane FMA with horizontal reduction, scalar tail) + public dispatcher. `/arch:AVX2` scoped to `src/kernels/` only.
  - New static libs: `ultima_tensor`, `ultima_runtime`, `ultima_kernels`
  - 19 new tests: alignment guarantees, calloc zeroing, view immutability, move semantics, parallel-for correctness, no-op safety, CPU feature detection, dispatcher/scalar/AVX2 equivalence with random inputs (max abs err <1e-4).
  - All 30 tests pass total. Warnings-clean.
- **M1 chunk 2: element-wise, RMSNorm, softmax** (continues Decision 05)
  - `include/ultima/kernels/elementwise.hpp` — `add_f32`, `mul_f32`, `silu_f32`, `swiglu_f32` (each with scalar oracle + best-available public variant).
  - `include/ultima/kernels/norms.hpp` — `rmsnorm_f32`(scalar + AVX2). Two-pass: sum-of-squares reduction with FMA, then broadcast multiply.
  - `include/ultima/kernels/softmax.hpp` — `softmax_f32` numerically stable (max-shift, exp, normalize).
  - `src/kernels/elementwise.cpp` — AVX2 8-lane loops for add/mul (in-place safe). silu/swiglu use scalar `std::exp` (AVX2 polynomial deferred to v0.2).
  - `src/kernels/rmsnorm.cpp` — AVX2 sum-of-squares + broadcast, scalar tail for `n % 8`.
  - `src/kernels/softmax.cpp` — scalar 3-pass (max, exp, normalize). Vector expf deferred.
  - 16 new tests (46 total): random unaligned equivalence, in-place safety (`y == x`), known small cases (silu(0)=0, uniform softmax=1/N), numerical stability (softmax of [100,101,102] does not overflow), SwiGLU = silu(gate) * up cross-check, RMSNorm known formula for `[1,2,3,4]`, RMSNorm on Qwen-shape `n=896`.
  - All 46 tests pass. No warnings. Taboo gate clean.
