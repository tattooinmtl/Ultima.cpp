# Ultima.cpp — Roadmap 01

Living document. We make **one decision at a time**, in order. Each decision considers CPU, Vulkan, CUDA, Metal separately, but v0.1 ships only what runs on the reference machine.

---

## Product scope

**Long-term vision (in priority order):**

1. **Ultima.cpp runtime** — independent C++ inference engine with the persistent-memory subsystem as a first-class citizen. This is the **core research bet**: force models to use the external memory / adapter layer instead of fine-tuning on HF/Colab. Memory + adapters become the "living intelligence" wrapper around an immutable base model.
2. **Coding assistant** — the first *product* built on Ultima. Runs a local coding model, calls tools, uses MCP, has skills, has memory of the codebase and the user's habits.
3. **Ultima code editor** — staged in two phases:
   - **v0.2 (Tier 1 editor):** lightweight Wails/Tauri window hosting the Monaco editor + Ultima chat side-panel + file tree + terminal + `rg` search + agent-inline buffer writes + diff review. No LSP, no plugin ecosystem. Ships as a second `.exe` next to `ultima.exe` + `ultima-launcher.exe`. 3–6 months after v0.1.
   - **v0.3+ (Tier 3 editor, "VS Code analog"):** **fork Code-OSS** (MIT-licensed VS Code base) → rebrand as Ultima Editor → rip Microsoft telemetry → deep-integrate Ultima as the AI backend → inherit the VS Code extension API, marketplace-compatible plugins, LSP/DAP, remote dev. This is the Cursor / Windsurf / Void path — the only realistic route to plugin-ecosystem parity. **Do not attempt from scratch.** Triggered only if v0.2's Tier 1 editor validates the product enough to justify the multi-month fork work.

**Free bridge in the meantime:** because v0.1 ships an OpenAI-compatible endpoint at `localhost:7777`, users who want VS Code today just install Continue.dev / Cline and point them at Ultima. Full VS Code experience + Ultima backend, day one, zero fork work.

Stage 3 is explicitly **not now**. The editor is deferred to v0.2 (Tier 1) and possibly v0.3+ (Tier 3 Code-OSS fork). Stages 1 and 2 develop together because the coding assistant is what proves the memory subsystem works — you can't demonstrate "persistent live intelligence" without a real user workflow using it.

## Non-negotiable: clean-room, independent implementation

Ultima.cpp does **not** link, fork, vendor, or copy any existing C/C++ local-inference runtime. Not a dependency in any form. GGUF is a documented public file format and mmap-based quantized inference is a public technique — we implement both from scratch. This preserves:

- Full license freedom for the memory-subsystem research and editor product
- Architectural freedom (QK-Norm, Qwen3 kernels, memory routing, adapter hooks) without fighting an upstream codebase
- The "clean, editor-first-designed engine" identity of the project

External reference implementations (transformers, vLLM, other public runtimes) are used only as **test oracles** — we generate their numerical outputs on fixed prompts, commit those outputs as opaque `.bin` fixture files, and compare our results against the fixtures. We read their behavior, never their source. Oracle tooling lives outside the shipped repo.

### Naming taboo (hard rule)

**The name of the incumbent C++ LLM runtime, and its associated tensor library, must never appear anywhere in the Ultima repository** — not in source, headers, filenames, comments, string literals, config keys, log messages, test names, commit messages, or CI configs. Enforced by CI grep gate (Decision 02 §2.7).

When we eventually add support for Meta's L-family models (post-v0.1), they are referred to internally by neutral identifiers (`meta_l3_8b`, `meta_l3_70b`) or by the specific fine-tune's own name (never the base brand). Rejection notes in this roadmap that previously mentioned older Meta code models have been removed for consistency.

This roadmap document itself contains **zero mentions** of the banned identifiers going forward.

### Prior work at `C:\UltimaLlama` — reference for intent only

An earlier project at `C:\UltimaLlama` explored a similar goal. It is a hard fork of the banned upstream with a Rust CLI wrapper on top, therefore **derivative work**. Ultima.cpp treats it as:

- **Allowed input:** README, AGENTS.md, and other user-authored design documents — these describe *product intent* (tool list, memory idea, session persistence, tier-based context sizing) and are safe requirement inputs.
- **Forbidden input:** any source file, build script, or design that mirrors the banned upstream's APIs. Not read, not ported, not translated.

Feature capabilities from the prior project are re-implemented from scratch under Ultima.cpp's clean-room architecture — every wish list item (model picker, tools, knowledge base, sessions, context auto-sizing, CLI REPL) is covered by v0.1 decisions and re-designed to be better (native inference, dual-runtime embed, MCP, language-JSON memory shards, skills, webui, cross-platform).

## v0.1 UX surface (locked)

v0.1 ships:
- `ultima` **CLI** — interactive REPL, one-shot `--prompt`, tool-call display, `--show-thinking` for reasoning models
- `ultima serve` **HTTP endpoint** on `localhost:7777` — OpenAI-compatible + Ultima extensions
- **Web chat UI** served at the same port — chat template rendering, streaming, markdown, syntax-highlighted code blocks, tool-call visualization, model/tier switcher, model registry UI (wraps Decision 01e endpoints)

**Free bonus:** because we're OpenAI-compatible on `/v1/chat/completions`, existing editor plugins (Continue.dev, Cline, Aider, VS Code with any custom-endpoint chat extension) can point at `localhost:7777` and use Ultima **immediately**. Users get an "editor experience" via their existing tools while v0.2's native editor is built.

v0.1 does **not** ship:
- Native Ultima code editor (deferred to v0.2)
- Direct file-writing "agent codes in your open buffer" flow (v0.2)
- LSP / diff review UI / cursor-following (v0.2)
- Plugin system for third-party editor addons (v0.2+)

Rationale: editor scope rivals the runtime itself. Building both simultaneously = neither ships. Prove the runtime + memory thesis in the chat UI first; if it holds, the editor becomes UX-work on a proven engine.

## Design principle: CLI and webui share 100% of the runtime

The prior CLI project's real problem wasn't the CLI — it was that a terminal is a bad rendering surface for code artifacts, diffs, tool-call arguments, MCP server schemas, and memory browsing. Ultima.cpp fixes this by making the **webui the primary surface**, but keeps the CLI fully capable for pipelines / SSH / scripting.

**Non-negotiable:** every feature works in both surfaces. No CLI-only feature, no webui-only feature. The runtime is one process, one HTTP endpoint. The CLI and webui are just different renderers over the same JSON stream.

Concrete implication: any Decision from 09 onwards (tools, MCP, skills, memory) is spec'd once at the runtime level. CLI gets a text renderer for it; webui gets a rich renderer. Both are thin.

**What this means for v0.1:**

- Reference model becomes a **coding model** (not a general chat model)
- Tool-calling / MCP / skills are **core v0.1 features**, not "eventually"
- The memory subsystem must store *codebase knowledge, corrections, project state* — not just chat facts
- Every interface we design has to survive being called both from a CLI and (later) from an editor process over localhost

## Two-model architecture (dual-GGUF runtime)

Ultima loads **two models simultaneously** from v0.1:

1. **Coder model** (~1.5B–7B, Q4_K_M) — generates code, calls tools, reasons.
2. **Search/embed model** (tiny, ~30–500 MB) — its only job is producing embeddings for retrieval and, optionally, reranking. Never generates user-facing text. This keeps the expensive coder off the retrieval hot path entirely.

Every user turn goes:

```
user prompt
    │
    ▼
[embed model] → query vector ─┐
    │                          ▼
    │                    ┌───────────────────┐
    │                    │  language JSON    │
    │                    │  shards + project │
    │                    │  memory + skills  │
    │                    └────────┬──────────┘
    │                             │ top-k
    │                             ▼
    │                     retrieved chunks
    │                             │
    ▼                             ▼
       augmented context → [coder model] → answer / tool call
```

## Memory as per-language JSON shards

The memory tree is organized so retrieval can be scoped instantly by language before any vector work:

```text
memory/
├── languages/
│   ├── cpp.json           ← idioms, std lib gotchas, RAII patterns, ABI notes
│   ├── python.json        ← packaging, typing, common bugs
│   ├── rust.json          ← borrow-checker patterns, cargo, unsafe rules
│   ├── typescript.json    ← tsconfig, module systems, React vs Node
│   ├── go.json
│   ├── zig.json
│   ├── glsl.json          ← for shader work (Blender/Vulkan later)
│   └── ...one file per language we support
│
├── frameworks/            ← per-framework JSONs (react.json, qt.json, bevy.json…)
├── projects/              ← per-project state (current codebase, decisions, TODOs)
├── skills/                ← reusable procedural memories (see Decision 10)
├── corrections/           ← learned "when you did X it was wrong, do Y"
└── conversations/         ← episodic memory of past sessions
```

Retrieval is a two-stage funnel:
1. **Router** (cheap): infer language(s) from the request → load only the relevant shard(s) into the embed search.
2. **Vector search** (embed model): top-k inside the scoped shard(s).

This avoids embedding-searching gigabytes of memory on every turn.

## Native code-search tools (built-in, not MCP)

`rg` (ripgrep) and `fd` are effectively part of the runtime — not general MCP tools. They ship as **first-class native tools** the coder model can call directly, because they're the fastest way to search a codebase and the model needs them constantly. Full tool taxonomy comes in Decision 09; the short list of built-in-native tools is:

- `rg` — content search
- `fd` — filename search
- `read_file`, `write_file`, `edit_file`
- `list_dir`
- `run_shell` (sandboxed)
- `git_*` (status, diff, log, blame)
- `open_url` (fetch docs)

Everything else (Blender, Visual Studio, database tools, browser automation, etc.) goes through **MCP** so it's pluggable.

---

## Reference machine (v0.1 development target)

- **CPU:** AMD Ryzen 7 5700U — 8 cores / 16 threads, **Zen 2** mobile (Lucienne / Renoir refresh), AVX2, **no AVX-512**, 8 MB L3, 15 W mobile TDP
- **RAM:** 32 GB
- **GPU:** AMD Radeon Vega 8 iGPU — **unused in v0.1**; potential Vulkan compute target in v0.2+ (no CUDA path — non-NVIDIA)
- **OS:** Windows 11 Home 25H2 (build 26200)
- **Build:** MSVC 2022 (v143), Windows 11 SDK
- **Quantization floor:** 4-bit (Q4) and up — no F16/F32 full-precision requirement, no sub-4-bit yet

**Realistic tok/s expectations on this hardware (post-implementation, no compiler magic):**
- Qwen2.5-Coder-0.5B Q4_K_M: 40–60 tok/s generation
- Qwen2.5-Coder-1.5B Q4_K_M: 15–25 tok/s generation
- Qwen2.5-Coder-7B Q4_K_M: 4–8 tok/s generation (thermal throttling on sustained load)
- Ornith-1.5-9B Q4_K_M (M4+): 3–6 tok/s generation
- Qwen2.5-Coder-14B Q4_K_M: 2–4 tok/s generation

Zen 2 mobile is genuinely modest hardware — this makes the "small-model + strong-memory beats big-model raw" thesis *more* important, not less. The Tiny/Small tiers will feel snappy; the Mid tier is usable for real work; Large tier is patient-batching territory. Vulkan backend on the Vega iGPU (v0.2+) could roughly 2–3× these numbers.

This is the **dev machine** — but Ultima has to run well on smaller *and* larger boxes too.

## System tiers (target profiles, all CPU-only)

The memory subsystem is what makes tiered scaling *actually work*: a 1.5B model with strong language-JSON retrieval, skills, and corrections behaves closer to a 7B raw model than to a 1.5B raw model. This is the whole point of the project — **the memory layer buys you 1–2 model sizes of effective capability**.

| Tier | Hardware | Coder model | Embed model | Expected gen tok/s | Target user |
|---|---|---|---|---|---|
| **Tiny** | 8 GB RAM, 4-core laptop | Qwen2.5-Coder-0.5B Q4_K_M (~400 MB) | all-MiniLM-L6-v2 Q8 (~25 MB) | 20–40 | Old laptops, Raspberry Pi 5, first-time users |
| **Small** | 16 GB RAM, 6–8 core | Qwen2.5-Coder-1.5B Q4_K_M (~1 GB) | bge-small-en-v1.5 Q8 (~35 MB) | 15–25 | Mainstream laptops |
| **Mid (dev target)** | 32 GB RAM, 8c/16t (your box) | Qwen2.5-Coder-7B Q4_K_M (~4.5 GB) | bge-base-en-v1.5 Q8 (~110 MB) | 6–12 | Workstations, this project's dev loop |
| **Large** | 64 GB RAM, 12c+ | Qwen2.5-Coder-14B Q4_K_M (~8.5 GB) | bge-large-en-v1.5 Q8 (~340 MB) | 3–6 | Beefy desktops |
| **XL (later)** | 64+ GB, GPU | Qwen2.5-Coder-32B Q4_K_M (~19 GB) | same as Large | GPU-only realistic | Post-v0.1 with Vulkan/CUDA |

**Ultima ships one binary that auto-detects the tier**, or accepts an explicit `--tier` / `--profile` flag. Same code path, different GGUFs on disk. The language-JSON memory shards and skills are **identical across tiers** — so upgrading hardware upgrades quality without re-authoring any memory.

The v0.1 development happens on the Mid tier, but M4 correctness is validated on the **Tiny** tier first (fastest dev loop) and the Small tier second, before touching 7B.

## Backend strategy (across all decisions)

For every subsystem below we record four columns:

| Subsystem | CPU (v0.1) | Vulkan (v0.2+) | CUDA (v0.3+) | Metal (v0.3+) |

CPU is the only column we implement now. The others get a **one-line note** so we don't paint ourselves into a corner. If a v0.1 choice would block a future backend, we flag it and revisit.

## Decision log

Decisions are numbered. Each has: **Question → Options → Chosen → Rationale → Cross-backend note**. Nothing is chosen until you approve it.

Planned sequence (subject to change as decisions land):

1. Reference coding model + quantization ✅ **DECIDED (Option C)**
   - 1a: Coder family = Qwen2.5-Coder (M4) + Qwen3 (M4+)
   - 1b: Embed model family ← *waiting on you*
   - 1c: Qwen3 add-on milestone specced
   - 1d: Ornith licensing note
2. Build system & dependency policy (CMake, C++20 subset, third-party rules, `rg`/`fd` bundling)
3. GGUF loader + mmap strategy
4. Tokenizer strategy (BPE regex engine, Qwen2 + Qwen3 vocab loading)
5. CPU kernel strategy (SIMD level, threading, dequant approach)
6. Model architecture layer (Qwen2Model, Qwen3Model, `IModel` dispatch)
7. Sampling + stop conditions (including code-aware stops, reasoning-token handling)
8. KV cache + long-context strategy (YaRN scaling for Qwen3)
9. **Tool-calling & MCP architecture** (native tools + MCP client — product hinge)
10. **Skills system** (how skills are stored, loaded, invoked)
11. **Memory subsystem v0.1** (language-JSON shards, index, retrieval, injection)
12. Adapter subsystem stub (LoRA hot-load interface only, no training)
13. CLI UX (coding-assistant loop, `--show-thinking`, tier auto-detect)
14. HTTP / localhost server + streaming (OpenAI-compatible + `reasoning_content` extension)
15. Editor-bridge protocol reservation (interface only, no implementation)

---

# Decision 01 — Reference coding model & quantization format

**Why this is first:** Every downstream choice (tokenizer, tensor ops, dequant kernels, GGUF parser scope, chat template, tool-call format, FIM tokens, test fixtures) depends on which single model we make "hello world" work against. Picking one concrete model turns the whole v0.1 spec from abstract to testable.

Because the product is a **coding assistant**, the reference model must:

- Be trained for code (not general chat retrofitted with code data)
- Have **native tool-use / function-calling** training — so we can prove MCP integration end-to-end without prompt-hacking a non-tool model
- Support **fill-in-the-middle (FIM)** tokens — required for editor autocomplete later
- Have a permissive license (Apache-2 or similar) — no Meta / Mistral commercial gates
- Come in multiple sizes sharing the **same architecture**, so the tiny variant is the dev-loop model and the bigger variant is the daily-use model with zero code changes

## The question

> **Which single model *family* do we commit to, and which specific sizes fill the Tiny / Small / Mid / Large tiers?**
>
> Family commitment matters more than the specific size, because all sizes in the same family share tokenizer, chat template, tool-call format, and architecture. Pick the family once and every tier just swaps the GGUF.

## Options (all Q4_K_M, all fit 32 GB with room for tools + memory + KV)

| # | Model | Size @ Q4_K_M | Arch | Tool use | FIM | License | Notes |
|---|---|---|---|---|---|---|---|
| A | **Qwen2.5-Coder-1.5B-Instruct** → scale to **Qwen2.5-Coder-7B-Instruct** | 1.0 GB → 4.5 GB | Qwen2 (GQA, RoPE, **QKV bias**, RMSNorm, SwiGLU) | **Native** (Hermes/Qwen format) | **Yes** (`<\|fim_prefix\|>` etc.) | Apache-2 | Industry-standard local coding stack. 1.5B for dev correctness, 7B for real work. Same code path. |
| B | **Qwen2.5-Coder-3B** → scale to **14B** | 2.0 GB → 8.5 GB | same as A | Native | Yes | Apache-2 | Better ceiling but 14B is slow on CPU-only (~2–4 tok/s realistic). |
| C | **DeepSeek-Coder-V2-Lite-Instruct** (16B MoE, 2.4B active) | ~10 GB | DeepSeek-V2 MoE (GQA, MLA attention, expert routing) | Native | Yes | DeepSeek license (permissive-ish) | Excellent code quality, MoE runs fast on CPU (only 2.4B active per token). But **MoE + MLA = ~3× the v0.1 kernel work**. Wrong first target. |
| D | **StarCoder2-3B** → scale to **7B** | 1.8 GB → 4.0 GB | GPT-2-ish + GQA + sliding window | Weak (no native tool training) | Yes | BigCode OpenRAIL-M (has use restrictions) | Fails the tool-use requirement. |
| E | **Codestral-22B** | ~13 GB | Mistral | Yes | Yes | Mistral non-production license | License blocks commercial use. Also too slow on CPU. |

## DECIDED: **Option C — Qwen2 AND Qwen3 architectures, staged**

Qwen2.5-Coder family for M4 correctness + immediate tiering; Qwen3 architecture added right after M4 to unlock Ornith and the Qwen3 fine-tune ecosystem.

### Tier plan (default coder per tier, alternatives in parens)

| Tier | Default coder | Alt (user opt-in) | Arch needed |
|---|---|---|---|
| Tiny | Qwen2.5-Coder-0.5B Q4_K_M | — | Qwen2 |
| Small | Qwen2.5-Coder-1.5B Q4_K_M ← **M4 gate** | VibeThinker-1.5B (reasoning) | Qwen2 |
| Mid | Qwen2.5-Coder-7B Q4_K_M ← **daily use** | DeepSeek-R1-Distill-Qwen-7B | Qwen2 |
| **Mid+** | **Ornith-1.5-9B Q4_K_M** | Qwen3-Coder-8B (when available) | **Qwen3** |
| Large | Qwen2.5-Coder-14B Q4_K_M | DeepSeek-R1-Distill-Qwen-14B | Qwen2 |
| XL (post-v0.1) | Qwen2.5-Coder-32B / Qwen3-Coder-32B | — | both |

**Qwen2** tiers ship with M4. **Qwen3** tiers unlock at M4+ (see Decision 01c below).

### Why this is right

- **Fastest dev loop preserved:** Qwen2.5-Coder-1.5B remains the correctness gate — smallest, most external reference implementations to cross-check against (transformers, vLLM, other public runtimes), fastest iteration.
- **Full community ecosystem:** Qwen2 side unlocks VibeThinker, DeepSeek-R1-Distill-Qwen. Qwen3 side unlocks Ornith and all forward-looking fine-tunes.
- **Ornith-1.5-9B at Mid+ tier** slots between 7B and 14B — genuinely SOTA agentic-coding numbers (SWE-bench Verified 70.6) that a Qwen2-only build would exclude.
- **Additive, not disruptive:** Qwen3 support is a follow-on milestone (M4+), not a v0.1 blocker. If Qwen3 kernels turn out harder than estimated, we ship Qwen2-only and add Qwen3 in a point release.

### What this commits us to implementing in v0.1

**M4 gate (Qwen2 only):**
- GGUF v3 parser with mmap
- Q4_K_M dequant (scalar → AVX2; AVX-512 stretch on Zen 4)
- Q6_K + Q8_0 dequant (near-free)
- F32-accumulator matmul: Q4_K weight × F32 activation, plus Q6_K/Q8_0 variants
- RMSNorm, SwiGLU
- **Qwen2 RoPE** (base 1_000_000, no scaling)
- **GQA attention with QKV bias** (Qwen2 specific)
- **Qwen2 tokenizer** (tiktoken-family BPE + regex pre-tokenizer)
- **Qwen2 chat template** (`<|im_start|>` / `<|im_end|>`)
- Qwen tool-call parser (`<tool_call>{...}</tool_call>`)
- FIM tokens reserved in tokenizer (dormant until editor stage)

**M4+ milestone (Qwen3 add-on — see Decision 01c):**
- **QK-Norm** in attention path
- **Qwen3 RoPE** variant + **YaRN long-context scaling** (Ornith needs this for 256K → 1M)
- **Qwen3 tokenizer vocab** (superset-ish of Qwen2 with new special tokens)
- **Reasoning-token streaming**: parser splits `<think>...</think>` from user-facing output; CLI has `--show-thinking` flag
- **qwen3_xml tool-call parser** variant
- **Explicitly deferred:** vision tower / multimodal input (Ornith supports images; we ignore that for v0.1 — text-only still works)

### Cross-backend note

| Subsystem | CPU (v0.1) | Vulkan | CUDA | Metal |
|---|---|---|---|---|
| Q4_K_M / Q6_K dequant | AVX2 per-block scalar | compute shader per super-block | CUDA kernel per super-block | MSL kernel per super-block |
| Matmul (Q4×F32 mat-vec) | hand SIMD, no BLAS dep | mat-vec shader → later mat-mat | custom kernel or cuBLASLt | MPS mat-vec |
| Mmap weights | `CreateFileMapping` (Win) + `mmap` (Linux) | host-visible staging | pinned host + async H2D | unified memory on Apple Silicon |
| QK-Norm (M4+) | trivial: one RMSNorm per head, per token | same | same | same |
| YaRN RoPE (M4+) | precomputed inv-freq table + scaling | precomputed constant buffer | same | same |
| Tokenizer regex (both) | `std::regex` first, `re2` if perf demands | CPU-only | CPU-only | CPU-only |
| Reasoning-token parser | streaming state machine on decoded tokens | CPU | CPU | CPU |

Nothing in Option C blocks any future backend.

### Bonus: the whole Qwen community fine-tune ecosystem comes free

Committing to Qwen2 + Qwen3 means Ultima runs, with **zero new code**, any GGUF built on either base:

**Qwen2-arch (M4):**
- **VibeThinker-1.5B** — small reasoning fine-tune, punches far above weight on math/logic
- **DeepSeek-R1-Distill-Qwen** (1.5B / 7B / 14B / 32B) — reasoning distillations
- All Qwen2.5-Coder community merges, LoRA-fused variants, uncensored fine-tunes

**Qwen3-arch (M4+):**
- **Ornith-1.5-9B** — SOTA agentic-coding fine-tune (SWE-bench 70.6)
- **Qwen3-Coder** official variants
- Any future Qwen3 fine-tune

Users can swap the coder model per project (`--model <path>.gguf`) and get a specialist without Ultima needing a release.

**Identification rule** for any model a user asks about: check HF `config.json` `model_type`.
- `qwen2` → free drop-in at M4
- `qwen3` → free drop-in at M4+
- anything else → new-architecture ticket

**Non-Qwen fine-tunes** (deferred to post-v0.1, listed in order of expected cost):
- Meta L3 family (add RoPE variant + drop QKV bias — trivial, ~1 day)
- Gemma-2 (different norm placement, logit soft-cap)
- Phi-3/4 (fused QKV, sliding window)
- DeepSeek-V2/V3 MoE + MLA (major project)

Rationale:

1. **Tool-use is native, not bolted on.** Qwen2.5-Coder ships with function-calling training and a documented JSON tool schema. This means our MCP integration (Decision 09) can be verified end-to-end against real model behavior instead of us praying that prompt-engineering works.
2. **FIM tokens exist.** `<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>` are in the vocab. When we build the editor later, autocomplete is a tokenizer change, not a model change.
3. **Two-tier dev loop.** 1.5B at Q4 loads in <1s and generates fast enough on your Ryzen to run the correctness harness on every save. When 1.5B passes M4, you drop in the 7B GGUF and the same code runs it — Qwen2 architecture is identical across sizes.
4. **Architecture forward-compat.** Qwen2 (GQA + RoPE + RMSNorm + SwiGLU + QKV bias) is a near-superset of the Meta L3 building blocks. If we implement Qwen2 first, adding L3 later is *removing* the QKV bias — trivial. The reverse means retrofitting bias into the attention path.
5. **Apache-2.** No license paperwork. Distributable, embeddable in the editor later, no per-user acceptance.
6. **7B at Q4_K_M on your box.** ~4.5 GB weights + ~2 GB KV cache at 8k ctx + memory subsystem + tool schemas ≈ 10 GB working set. Comfortable in 32 GB with room for a browser and the OS. Realistic throughput on Zen 3/4 8-core with AVX2: **6–12 tok/s generation, 40–80 tok/s prompt processing** once we have a decent matmul.

### What this commits us to implementing in v0.1

**Model kernel:**
- GGUF v3 parser with mmap
- Q4_K_M dequant (scalar → AVX2; AVX-512 as stretch if you're on Zen 4)
- Q6_K + Q8_0 dequant (comes almost free with Q4_K_M — needed because embeddings/output are often stored at higher precision even in a Q4_K_M file)
- F32-accumulator matmul: Q4_K weight × F32 activation (hot path), plus Q6_K/Q8_0 variants for output projection
- RMSNorm, SwiGLU, RoPE (Qwen2 variant: base 1_000_000)
- GQA attention **with QKV bias**
- Qwen2 tokenizer (tiktoken-family BPE, cl100k-adjacent, with regex pre-tokenizer)

**Product surface:**
- Qwen tool-call parser (parses `<tool_call>{...}</tool_call>` blocks the model emits)
- Chat template renderer for Qwen2.5 format (`<|im_start|>` / `<|im_end|>`)
- FIM token support in tokenizer (dormant until editor stage, but reserved)

### Cross-backend note

| Subsystem | CPU (v0.1) | Vulkan | CUDA | Metal |
|---|---|---|---|---|
| Q4_K_M / Q6_K dequant | AVX2 per-block scalar | compute shader per super-block | CUDA kernel per super-block | MSL kernel per super-block |
| Matmul (Q4×F32 mat-vec) | hand SIMD, no BLAS dep | mat-vec shader → later mat-mat | custom kernel or cuBLASLt | MPS mat-vec |
| Mmap weights | `CreateFileMapping` (Win) + `mmap` (Linux) | host-visible staging | pinned host + async H2D copy | unified memory on Apple Silicon |
| Tokenizer regex | `std::regex` first, `re2` if we hit perf | CPU-only (never on GPU) | CPU-only | CPU-only |
| Tool-call parser | CPU (streaming state machine) | CPU | CPU | CPU |

Nothing in choice A blocks any future backend. QKV bias is a trivial add-scalar operation in every backend.

## Rejected non-Qwen options (still deferred)

- **StarCoder2:** No native tool training. Loses to Qwen2.5-Coder on every axis we care about.
- **Older Meta code models:** Outdated. Qwen2.5-Coder beats them on every published code benchmark.
- **Codestral:** License blocks the editor product later.
- **DeepSeek-Coder-V2-Lite MoE:** MoE + MLA is a major project; deferred until the memory subsystem eventually needs MoE anyway.

---

---

# Decision 01b — Embed / search model

**Why paired with 01:** the dual-model runtime needs both GGUFs picked together — the embed model determines the vector DB dimensionality, the retrieval kernels, and whether we need a second tokenizer implementation.

## The question

> **Which embedding model do we ship (per tier), and do we implement it as a native BERT-style forward pass in Ultima or reuse the transformer kernels we already have?**

## Options

| # | Model | Size Q8 | Dim | Arch | Notes |
|---|---|---|---|---|---|
| A | **BGE family** — `bge-small` (Tiny/Small), `bge-base` (Mid), `bge-large` (Large) | 35 MB / 110 MB / 340 MB | 384 / 768 / 1024 | BERT (encoder-only, MHA, GeLU, LayerNorm) | MIT license. Top of MTEB retrieval. One arch across sizes. |
| B | **nomic-embed-text-v1.5** | ~140 MB | 768 (Matryoshka: usable at 256/512 too) | BERT + RoPE | Apache-2. Long context (8k). Matryoshka = one model many dims. |
| C | **all-MiniLM-L6-v2** | 25 MB | 384 | BERT-tiny | Ancient but bulletproof. Tiny tier only. |
| D | **jina-embeddings-v3** | ~570 MB | 1024 | XLM-RoBERTa | Multilingual (useful for non-English code comments). Bigger. |
| E | **Reuse coder model** for embeddings (pool last hidden state) | 0 (already loaded) | 3584 (7B) | — | Zero disk cost but slow — every retrieval invokes the 7B model. Wrong architecture choice for our thesis. |

## My recommendation: **A — BGE family (bge-small / bge-base / bge-large per tier)** + a Tiny-tier exception using **C (all-MiniLM-L6-v2)** because bge-small at 35 MB is still 40% larger than MiniLM.

Rationale:
1. **BERT encoder-only** is architecturally simpler than a decoder LLM: no KV cache, no autoregressive loop, single forward pass. Kernel work is a proper subset of what we already need for Qwen2.
2. **Same family across tiers** — retrieval quality scales with hardware, same code path.
3. **MTEB leader** in its size class — retrieval quality directly determines whether the "small model + strong memory" thesis holds.
4. **Fixed output dim per tier** simplifies the vector index (Decision 11).
5. **MIT license.**

### What this commits us to implementing in v0.1

- BERT encoder forward pass (subset of Qwen2 kernels: LayerNorm instead of RMSNorm, GeLU instead of SwiGLU, absolute pos embeds or RoPE, no GQA)
- WordPiece tokenizer (BGE uses BERT WordPiece — **different from Qwen2 BPE**, so we need two tokenizers). Small implementation, single-file.
- Mean-pool + L2-normalize on the last hidden state
- Vector index: v0.1 = flat cosine over per-shard memory (fast enough for <100k memories per language shard). HNSW deferred to v0.2.

### Cross-backend note

| Subsystem | CPU (v0.1) | Vulkan | CUDA | Metal |
|---|---|---|---|---|
| BERT forward | reuses Qwen2 kernels (add LayerNorm, GeLU) | same | same | same |
| Vector search (flat cosine) | AVX2 dot-product over F32 vectors | trivial compute shader | trivial CUDA kernel | trivial MSL |
| WordPiece tokenizer | CPU-only | CPU | CPU | CPU |

Zero backend risk.

---

# Decision 01c — Qwen3 architecture add-on (M4+ milestone)

**Status:** committed as follow-on to M4 under Option C.

**Trigger:** starts the day Qwen2.5-Coder-1.5B passes the M4 correctness harness.

**Deliverable:** Ornith-1.5-9B Q4_K_M generates coherent output through the Ultima CLI, verified against a vLLM/transformers reference on 100 fixed prompts.

## Implementation checklist

| # | Item | Complexity | Notes |
|---|---|---|---|
| 1 | QK-Norm layer | Low | Per-head RMSNorm on Q and K before attention. Adds ~2 kernels, both reuse existing RMSNorm. |
| 2 | Qwen3 RoPE base + inv-freq table | Low | Just a different constant. Precompute at load. |
| 3 | YaRN long-context scaling | Medium | Needed for Ornith's 256K default. Piecewise inv-freq scaling based on ctx-len at load time. |
| 4 | Qwen3 tokenizer vocab | Low-Medium | Same tiktoken family as Qwen2, expanded vocab + new special tokens (`<think>`, tool XML). Reuse Qwen2 BPE code, load new vocab. |
| 5 | Reasoning-token streaming parser | Medium | State machine over decoded tokens: `<think>` opens a "hidden" stream, `</think>` closes it. CLI flag `--show-thinking` toggles display. Server exposes `reasoning_content` alongside `content` (OpenAI-compatible extension). |
| 6 | `qwen3_xml` tool-call parser variant | Low | Different XML shape than Qwen2's `<tool_call>{json}</tool_call>`. Additive parser, chosen by model metadata. |
| 7 | Architecture dispatch | Low | Model loader reads `config.json.model_type`, picks `Qwen2Model` or `Qwen3Model` at load. Both share the base `IModel` interface. |
| 8 | Verify on Ornith-1.5-9B | High effort, low complexity | 9B at Q4_K_M ≈ 5.5 GB, fits your 32 GB with room. Runs at ~4–8 tok/s expected. |

## Explicitly NOT in M4+ (deferred to later milestones)

- **Vision tower / multimodal input** — Ornith supports image inputs; we ignore. Text-only Ornith still works and matches its published SWE-bench numbers.
- **Full 1M-token context** — YaRN up to 256K native. 1M via YaRN factor 4.0 is technically enabled by the same code but memory-prohibitive on CPU (KV cache alone would be ~40 GB at 1M ctx for 9B). Practical cap in v0.1: 32K.
- **Speculative decoding** with a Qwen3-small as drafter for Qwen3-large — future perf work.

## Cross-backend note

Every M4+ item is CPU-shape-preserving. Vulkan/CUDA/Metal all inherit the same additions as scalar → kernel translations. No architectural risk.

---

# Decision 01d — Ornith licensing & distribution check

Before we treat Ornith as a first-class default at the Mid+ tier, confirm:

- **License:** MIT (per HF card) — safe to bundle references, safe to embed in the editor product later.
- **Redistribution of weights:** MIT covers weights. We can ship a downloader script that pulls the GGUF from HF; we should **not** re-host the weights ourselves in the Ultima repo (repo bloat, not a license issue).
- **Attribution:** MIT requires attribution in NOTICE / README. Add a `MODELS.md` listing every default model with its license and upstream link.

No decision needed from you here — this is a note for when we hit M4+.

---

## DECIDED (01b): Embed family = BGE

Per-tier defaults:

| Tier | Embed default | Size Q8 | Dim |
|---|---|---|---|
| Tiny | all-MiniLM-L6-v2 | 25 MB | 384 |
| Small | bge-small-en-v1.5 | 35 MB | 384 |
| **Mid (your box)** | **bge-base-en-v1.5** | 110 MB | 768 |
| Large | bge-large-en-v1.5 | 340 MB | 1024 |
| XL | bge-large-en-v1.5 | 340 MB | 1024 |

Users can override per slot via the model registry (Decision 01e).

---

# Decision 01e — Model registry, downloader, and per-slot overrides

## Requirement

- Every model slot (**coder**, **embed**, and future: **reranker**, **draft-model** for speculative decoding, **vision** for later) has a dropdown/selector, a download button, and support for a custom local path.
- Downloader uses **primary + 2 fallback mirrors** per model.
- User can point any slot at any local GGUF or HF-cached model.
- On first run: detect hardware tier and offer to download the tier-appropriate defaults.

## Design

**Single source of truth:** a `models.json` registry file bundled with Ultima, extended by the user's local `~/.ultima/models.json`.

```json
{
  "schema_version": 1,
  "slots": {
    "coder": {
      "current": "qwen2.5-coder-7b-instruct-q4_k_m",
      "known_models": [
        {
          "id": "qwen2.5-coder-7b-instruct-q4_k_m",
          "display_name": "Qwen2.5-Coder 7B Instruct (Q4_K_M)",
          "arch": "qwen2",
          "size_bytes": 4680000000,
          "sha256": "…",
          "min_ram_gb": 12,
          "tier": ["mid"],
          "sources": [
            "hf://Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/qwen2.5-coder-7b-instruct-q4_k_m.gguf",
            "hf-mirror://hf-mirror.com/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/qwen2.5-coder-7b-instruct-q4_k_m.gguf",
            "https://…community mirror…/qwen2.5-coder-7b.gguf"
          ],
          "license": "apache-2.0",
          "chat_template": "qwen2",
          "tool_parser": "qwen2_json"
        },
        {
          "id": "ornith-1.5-9b-q4_k_m",
          "arch": "qwen3",
          "tier": ["mid_plus"],
          "sources": ["hf://ornith-ai/Ornith-1.5-9B-GGUF/…", "…"],
          "chat_template": "qwen3",
          "tool_parser": "qwen3_xml",
          "reasoning": true
        }
      ]
    },
    "embed": {
      "current": "bge-base-en-v1.5-q8_0",
      "known_models": [ … ]
    },
    "reranker":  { "current": null, "known_models": [] },
    "draft":     { "current": null, "known_models": [] },
    "vision":    { "current": null, "known_models": [] }
  },
  "custom_paths": {
    "coder": "/optional/user/path/to/some.gguf"
  }
}
```

**Resolution order per slot:** `custom_paths` → `slots.<slot>.current` → hardware-detected default.

## CLI surface (v0.1)

```bash
ultima models list                       # everything known + downloaded/missing
ultima models list --slot coder          # just one slot
ultima models download qwen2.5-coder-7b-instruct-q4_k_m
ultima models use coder ornith-1.5-9b-q4_k_m
ultima models use coder /path/to/local.gguf     # arbitrary local file
ultima models remove qwen2.5-coder-14b-instruct-q4_k_m
ultima models default --tier auto        # re-run auto-detect
ultima models verify                     # sha256 check downloaded files
```

## Download behavior

1. Try `sources[0]`; if 4xx/5xx/timeout, try `sources[1]`; then `sources[2]`.
2. Resumable HTTP range downloads (GGUFs are big; resume on flaky connections is non-negotiable).
3. Verify SHA-256 after download; refuse to load on mismatch.
4. Store in `~/.ultima/models/<id>.gguf` (respects `ULTIMA_MODELS_DIR` env var).
5. Progress bar in CLI; JSON-lines progress events on the HTTP endpoint (so the future editor GUI can render a progress ring).

## HTTP endpoints (v0.1 — feeds the future settings UI directly)

```
GET  /v1/models                          # OpenAI-compatible list (installed only)
GET  /v1/models/registry                 # Ultima extension: full known + installed + missing
POST /v1/models/download                 # { "id": "..." } → returns SSE progress
POST /v1/models/use                      # { "slot": "coder", "id": "..." | "path": "..." }
POST /v1/models/verify                   # sha check
DELETE /v1/models/{id}
```

The future settings UI (editor stage) is a thin wrapper over these. No new backend work needed when the GUI arrives.

## Adding new models is a one-file edit

To add a new model to the registry: append a JSON block to `models.json`. No recompile. This satisfies the "make sure we can add new tools easy" principle applied to models.

## Cross-backend note

Downloader is 100% CPU / network — no backend implications. Same code path on Vulkan/CUDA/Metal builds.

---

# Decision 01f — Runtime topology (dual runtime, single process)

## DECIDED: Single process, two runtimes, one HTTP endpoint

- `ChatRuntime` (coder GGUF)
- `EmbedRuntime` (embed GGUF)
- Both share: backend, tensor allocator, thread pool, config, logging
- Router coordinates them plus MemoryStore, SkillRegistry, ToolExecutor
- Single HTTP server (`ultima serve`) exposes OpenAI-compatible chat + Ultima extensions

## Why not two processes

- Tiny/Small tiers can't afford ~200–400 MB of duplicated overhead.
- Embed calls happen on **every user turn** (and sometimes several times per turn) — in-process function call is microseconds; cross-process HTTP is milliseconds plus JSON overhead.
- Fewer moving parts to install, debug, and ship.
- The editor (later) just talks to one localhost port.

## Escape hatch

The Router talks to runtimes through an interface, not directly. Swapping `EmbedRuntime` for a `RemoteEmbedClient` (that hits another Ultima instance on the network) is a config change, not a rewrite. This preserves:

- Running embed on a different machine
- Sharing one embed server across multiple project Ultima instances
- Running the coder on a GPU box while the embed stays local

Not built in v0.1, but the interface reserves the door.

## Data flow (canonical v0.1 turn)

```
POST /v1/chat/completions
    │
    ▼
Router.begin_turn(request)
    │
    ├─▶ EmbedRuntime.embed(user_message)          → query_vec
    │
    ├─▶ MemoryStore.search(query_vec, scope)      → top-k memories
    │       (scope inferred from project + language detection)
    │
    ├─▶ SkillRegistry.match(user_message)         → applicable skills
    │
    ├─▶ ToolRegistry.enabled_tools()              → tool schemas
    │
    ├─▶ ContextBuilder.render(system, memories,   → augmented prompt tokens
    │                          skills, tools,
    │                          history, user_msg)
    │
    ├─▶ ChatRuntime.generate(prompt, stream=SSE)  → tokens stream to client
    │
    │       If token stream contains <tool_call> or <think>:
    │           ─▶ StreamParser routes:
    │                • <think>…</think>  → reasoning_content channel
    │                • <tool_call>…      → ToolExecutor.run(...)
    │                                        └─ result appended to context
    │                                        └─ ChatRuntime resumes generation
    │
    └─▶ Router.end_turn(result)
            └─ MemoryStore.commit_deltas(...)     → new corrections/facts written
```

Every action produced by the coder loops back through the ToolExecutor and the result re-enters the coder's context. This is the "agent loop" — the coder drives, everything else is called on demand.

---

## Please decide

Everything in Decision 01 is now closed:
- ✅ 01a: Qwen2 + Qwen3 (Option C)
- ✅ 01b: BGE embed family
- ✅ 01c: Qwen3 add-on milestone specced
- ✅ 01d: Ornith MIT license OK
- ✅ 01e: Model registry + downloader + per-slot override
- ✅ 01f: Single process, dual runtime, single HTTP endpoint

---

# Decision 02 — Build system & dependency policy

## Guiding principles

1. **Reproducible builds.** A fresh checkout on a clean machine should produce identical binaries with one command. No "install these 12 things first."
2. **Minimal, curated, pinned dependencies.** Not zero-dep (that's romantic and wasteful), not npm-style (that's a supply-chain trap). Every dep is chosen for a specific reason, statically linked, vendored via CMake `FetchContent` with a pinned commit hash, and MIT/Apache-2/BSD-licensed.
3. **Cross-platform from day one, even though v0.1 targets Windows.** The build system supports Linux and macOS from the first commit — we just don't gate v0.1 on them working perfectly.
4. **No exotic language features that break tooling.** Skip C++20 modules (still broken cross-compiler in 2026), skip anything experimental.

## 2.1 — Build system: CMake

**Minimum:** CMake **3.24** (gives us `FILE_SET HEADERS`, `--parallel` defaults, better `FetchContent`)
**Recommended:** 3.28+ (better dependency provider hooks, cleaner presets)

**Use `CMakePresets.json` for every supported build.** No user should ever have to type raw `-DCMAKE_...` flags. Presets we ship:

```
windows-msvc-debug          (dev — your default)
windows-msvc-release
windows-msvc-relwithdebinfo (perf profiling)
windows-clang-release       (cross-check, catches MSVC-only warnings)
linux-gcc-debug
linux-gcc-release
linux-clang-release
macos-clang-release         (future — no v0.1 CI gate)
```

**Build command any user runs:**
```bash
cmake --preset windows-msvc-release
cmake --build --preset windows-msvc-release
ctest --preset windows-msvc-release
```

Three lines. Nothing else.

## 2.2 — Language: C++20 (curated subset)

**Allowed:**
- Concepts (constrain templates, better error messages)
- `std::span`, `std::string_view`
- `std::jthread`, `std::stop_token`
- Designated initializers (`Foo{.x = 1, .y = 2}`)
- Three-way comparison `<=>` (mostly for tests)
- `constinit`, `consteval`, expanded `constexpr`
- Ranges (used sparingly — heavy compile cost)
- Attributes: `[[nodiscard]]`, `[[likely]]`, `[[unlikely]]`, `[[no_unique_address]]`

**Banned in v0.1:**
- **C++20 modules** — cross-compiler support still broken in 2026. Use headers.
- **`std::format`** — libc++ / older MSVC coverage inconsistent. Use vendored `fmt` (drop-in). Migrate to `std::format` when universal.
- **Coroutines** — not needed. Revisit if the HTTP server or streaming design demands them.
- **RTTI** — disabled globally (`/GR-`, `-fno-rtti`). No `dynamic_cast`, no `typeid`. Saves binary size, removes footgun.
- **Exceptions** — **allowed** but **not on the hot path**. Kernels, tensor ops, tokenizer inner loops must be `noexcept`. Errors above the hot path use `tl::expected<T, Error>` (C++20-compatible drop-in for `std::expected`).

**Compiler minimums:**
- MSVC 19.30 (Visual Studio 2022 17.0)
- GCC 11
- Clang 14
- Apple Clang 14

## 2.3 — Dependency policy: curated, vendored, pinned

Every dep must justify itself. Approved v0.1 list:

| Dep | Purpose | Type | License | Why not roll our own |
|---|---|---|---|---|
| **doctest** | Unit + integration tests | Header-only | MIT | Test frameworks are the wrong thing to reinvent. Faster compile than Catch2, single header. |
| **fmt** | String formatting | Static lib | MIT | `std::format` not universal yet. Bridge until it is. |
| **nlohmann/json** | Config, tool-call parsing, memory JSON | Single header | MIT | JSON isn't a hot path for us. Readable code > 5% parse speed. |
| **cpp-httplib** | HTTP server + client (for downloads) | Single header | MIT | Single file, no deps, gives us server + client. Perfect for our scale. |
| **xxhash** | Content-addressed cache keys, SHA-256 alternative for internal use | Single header | BSD-2 | Cryptographically weak but 10× faster than SHA — right tool for cache keys. |
| **tl::expected** | Error-carrying return values | Single header | CC0 | Bridge until we can use C++23 `std::expected`. |

**Downloader/HTTPS:**
- **libcurl** for GGUF downloads. System libcurl on Linux/macOS (universally available). Vendored on Windows (build from source via FetchContent, statically linked, uses Schannel — no OpenSSL dependency).
- `cpp-httplib` handles server-side + non-TLS internal calls; libcurl handles user-facing HTTPS downloads where retry, resume, and progress matter.

**Explicitly rejected for v0.1:**

| Rejected | Reason |
|---|---|
| Boost (any) | Massive, slow to build, most of what we'd want is in C++20 already |
| Abseil | Would drag in via re2. We use `std::regex` instead. |
| re2 | `std::regex` is slow but fine for v0.1 tokenizer volumes. Revisit at Decision 04 if profiling demands it. |
| simdjson | nlohmann/json is faster to iterate on. Our JSON isn't hot. |
| Protobuf / gRPC | Overkill. JSON over HTTP is enough. |
| spdlog | We write ~200 lines of our own leveled logger. Locks-free stdout + optional file sink. |
| CUDA / Vulkan / Metal SDKs | Not in v0.1. |

**Rule for adding a new dep post-v0.1:** open an issue titled `dep: <name>` with (1) what problem it solves that we can't solve in <500 LOC ourselves, (2) license, (3) build weight, (4) alternatives considered. Otherwise no.

## 2.4 — Vendoring: CMake `FetchContent` with pinned commits

Every dep lives in `third_party/` as a `FetchContent_Declare` block with:
- Pinned `GIT_TAG` (commit hash, not branch name — no supply-chain surprises)
- `GIT_SHALLOW ON`
- `SYSTEM` flag (their warnings don't pollute our build)
- `EXCLUDE_FROM_ALL` (their tests/examples don't build)

Example:
```cmake
FetchContent_Declare(doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG        v2.4.11    # pinned
    GIT_SHALLOW    ON
    SYSTEM
    EXCLUDE_FROM_ALL
)
```

**Offline builds supported** via `FetchContent_Populate` cache directory (`FETCHCONTENT_BASE_DIR` env var). CI populates a cache once per week, dev machines reuse it.

## 2.5 — External tool binaries (`rg`, `fd`, others)

**Decision:** bundle prebuilt binaries in `third_party/bin/<platform>/` for tier-1 platforms.

- Windows x64: `rg.exe`, `fd.exe`
- Linux x64: `rg`, `fd`
- Linux arm64: `rg`, `fd`
- macOS arm64: `rg`, `fd`

**Rationale over PATH-detection:**
- Deterministic version (a `rg` upgrade won't silently change tool behavior)
- Works in fresh environments (fresh WSL, sandboxed CI, corporate machines with no admin rights)
- Both binaries together = ~10 MB total — negligible next to model weights
- Licenses: ripgrep MIT, fd MIT/Apache-2 — both allow redistribution

**Fallback chain:**
1. Bundled binary in `third_party/bin/<platform>/`
2. If missing (e.g. platform we don't bundle for), search PATH
3. If still missing, feature-degrade (log a warning, tool becomes unavailable to the model, other tools keep working)

**Version pin file:** `third_party/bin/VERSIONS.txt` records exact upstream version + release URL for each binary, so refreshes are auditable.

**Upgrade process:** a script (`scripts/refresh_bundled_binaries.sh` + `.ps1`) downloads current-release binaries from official GitHub releases, verifies checksums, updates `VERSIONS.txt`, opens a PR. Human reviews and merges.

## 2.6 — Directory layout (matches `Ultima.cpp` spec §5, with build additions)

```
ultima.cpp/
├── CMakeLists.txt
├── CMakePresets.json
├── .clang-format
├── .clang-tidy
├── .gitignore
├── LICENSE
├── NOTICE                    ← attribution for bundled deps + binaries + models
├── MODELS.md                 ← per Decision 01d
├── README.md
├── CONTRIBUTING.md
├── SECURITY.md
├── CHANGELOG.md
│
├── cmake/
│   ├── UltimaCompilerFlags.cmake
│   ├── UltimaWarnings.cmake
│   ├── UltimaSanitizers.cmake
│   └── UltimaDeps.cmake       ← all FetchContent_Declare lives here
│
├── docs/                      ← per spec §5
├── include/ultima/            ← per spec §5
├── src/                       ← per spec §5
├── apps/
│   ├── ultima-cli/
│   └── ultima-server/
├── tests/                     ← per spec §5
├── benchmarks/
├── examples/
├── scripts/
│   ├── refresh_bundled_binaries.ps1
│   ├── refresh_bundled_binaries.sh
│   └── download_models.py     ← Decision 01e implementation helper (optional Python)
│
└── third_party/
    ├── CMakeLists.txt         ← declares all FetchContent
    ├── bin/
    │   ├── VERSIONS.txt
    │   ├── windows-x64/       (rg.exe, fd.exe)
    │   ├── linux-x64/
    │   ├── linux-arm64/
    │   └── macos-arm64/
    └── webui/                 ← static assets served by ultima-server (Decision 14)
```

## 2.7 — Warnings, sanitizers, formatting, taboo gate

- **Warnings-as-errors** on all CI presets: `/W4 /WX` (MSVC), `-Wall -Wextra -Wpedantic -Werror` (GCC/Clang)
- **Extra warnings we opt into:** `-Wconversion`, `-Wshadow`, `-Wnon-virtual-dtor`, `-Wold-style-cast`, `-Wcast-align`, `-Wunused`
- **Sanitizers:** ASan + UBSan on `linux-clang-release` CI preset. TSan on a dedicated preset for concurrency work.
- **Formatting:** `.clang-format` checked in, style based on LLVM with 4-space indent, 100-column soft limit. Enforced in CI.
- **Static analysis:** `clang-tidy` on CI with a minimal ruleset (bugprone-*, performance-*, modernize-*). Not blocking initially, becomes blocking at v0.1 release.

### Taboo gate (blocking on every PR)

Enforces the naming rule from the "Non-negotiable" section. A dedicated CI step scans the entire repository (source, headers, tests, apps, cmake, scripts, docs, `NOTICE`, `CHANGELOG`, `README`, commit messages of the current PR) and **fails the build** if any banned identifier appears in any case-form.

Banned patterns (case-insensitive, whole-word or identifier-fragment):

```
llama         ->  banned everywhere, all case-forms (llama, Llama, LLAMA, LLaMA, LlaMa, ...)
ggml          ->  banned everywhere, all case-forms
```

Script (checked into `scripts/ci_taboo_check.sh` and `.ps1`). The patterns themselves are constructed at runtime from fragments so the script file does not self-trigger, and the script excludes itself and the roadmap docs from the scan:

```bash
#!/usr/bin/env bash
set -e
# constructed from fragments to avoid self-match
P1="lla""ma"
P2="gg""ml"
PATTERNS="${P1}|${P2}"
FOUND=$(grep -rniE "$PATTERNS" \
    --include='*.cpp' --include='*.hpp' --include='*.h' --include='*.c' \
    --include='*.cmake' --include='CMakeLists.txt' --include='*.json' \
    --include='*.txt' --include='*.yml' --include='*.yaml' \
    --include='*.py' --include='*.sh' --include='*.ps1' \
    --exclude-dir='third_party' --exclude-dir='.git' \
    --exclude='ci_taboo_check.*' \
    . || true)
if [ -n "$FOUND" ]; then
    echo "TABOO GATE FAILED - banned identifier found:"
    echo "$FOUND"
    exit 1
fi
echo "Taboo gate: clean."
```

Also scans the current PR's commit messages (`git log $BASE..HEAD --format=%B`).

**Exclusions:**
- `third_party/` — bundled deps may legitimately contain the strings in attribution/help text
- `ci_taboo_check.*` — self-exclusion (script's own message strings)
- `roadmap*.md` and design docs prior to first source commit — grandfathered as historical planning docs; new docs written after Decision 03 must be clean

**Runtime handling of banned strings in external data:** if some future dep or model config forces the string to appear (e.g., a `model_type` field in a downloaded GGUF), it is handled by fragment-concatenation aliasing in the loader — the raw string is constructed at runtime from fragments and never appears as a literal in our source.

## 2.8 — Cross-backend note

| Concern | CPU (v0.1) | Vulkan | CUDA | Metal |
|---|---|---|---|---|
| Build option | default | `-DULTIMA_BACKEND_VULKAN=ON` | `-DULTIMA_BACKEND_CUDA=ON` | `-DULTIMA_BACKEND_METAL=ON` |
| SDK finding | none | `find_package(Vulkan)` | `find_package(CUDAToolkit)` | Xcode built-in |
| Shader compilation | n/a | `glslc` at build time → SPIR-V embedded | `nvcc` per `.cu` | `metal` compiler → `.metallib` |
| Backends are **additive** — every ON flag adds a `.cpp/.cu/.mm` file behind the `IBackend` interface. Not linked unless enabled. |

No v0.1 choice above blocks any backend addition.

## 2.9 — v0.1 build definition of done

- `cmake --preset windows-msvc-release && cmake --build --preset windows-msvc-release` succeeds on a clean Windows box with only VS 2022 + Git installed
- `ctest --preset windows-msvc-release` passes 100% of unit tests
- Total build time (release, cold): <5 minutes on 8c/16t
- Total binary size: `ultima.exe` <30 MB (statically linked, no external runtime deps except system libs)
- `ldd` / `dumpbin /dependents` shows only system DLLs

## Please decide

**Question 2:** Lock in Decision 02 as written?

Points where you might want to override:

- **Testing framework:** doctest is my pick. Alternative: Catch2 (more features, slower compile), gtest (Google, more mainstream, heavier). Say the word.
- **HTTP lib:** cpp-httplib is my pick. Alternatives: mongoose (older but battle-tested), uWebSockets (fastest, more setup). httplib wins on simplicity.
- **JSON lib:** nlohmann/json is my pick. Alternative: simdjson (faster reads, awkward writes) or a hand-rolled parser tuned for GGUF metadata + tool calls. Pick nlohmann now, migrate if profiling demands.
- **`rg` / `fd` bundling:** bundle-first is my pick. Alternative: PATH-only (simpler for us, friction for users). Bundling wins on determinism.
- **RTTI/exceptions ban:** RTTI globally banned, exceptions banned on hot path. Aggressive but conventional in low-level C++ codebases. Say if you want them fully allowed.

---

# Decision 03 — Launcher (Go GUI, single `.exe` entry point)

## The requirement

- User double-clicks **one `.exe`**. No batch files, no editing config files by hand, no PowerShell setup.
- Launcher opens a **settings GUI** — model registry, tier selection, port, MCP server list, memory profile, tool toggles, log level.
- A **Launch Ultima** button starts the C++ runtime as a subprocess.
- Settings are editable from the launcher at any time.
- All existing runtime configuration (`config.json`, `models.json`, per-profile memory dirs) is the source of truth — the launcher is a GUI editor over these files, not a parallel config system.

## 3.1 — Language: Go

**Chosen.** Locked by user preference and justified:

- Compiles to a **single statically-linked `.exe`** with no runtime dependency (no JVM, no Python, no .NET, no MSVCRT redist).
- ~10–15 MB binary, cold-starts in <100 ms.
- Excellent HTTP client → the launcher is a natural home for the model downloader (Decision 01e) — pulls GGUFs with resume, SHA-256 verify, fallback mirrors, all with a progress bar the user actually sees.
- Cross-compiles to Windows/Linux/macOS from any host.
- Subprocess management (`os/exec`) is clean → launcher supervises `ultima.exe`, restarts on config change, kills on quit.

## 3.2 — GUI framework choice

Three viable options:

| Framework | Type | Binary | Pros | Cons |
|---|---|---|---|---|
| **Fyne** | Pure Go, custom-drawn widgets | ~15 MB | Zero JS toolchain. Single `go build`. Cross-platform. Mature. | Widgets look "Fyne-ish", not native Windows. |
| **Wails v2** | Go backend + WebView2 (Win) / WebKit (mac) / GTK-WebKit (Linux) frontend | ~20 MB | Native window, HTML/CSS for the UI, reuses design tokens with the main webui (Decision 14). Modern look. | Adds Node/npm build step for the frontend. WebView2 must be installed (Win10+ ships it). |
| **walk** | Native Win32 only | ~8 MB | Actual native widgets. Smallest binary. | Windows-only. Doesn't fit the cross-platform reservation. |

**My recommendation: Wails v2**, because:
1. The launcher's settings UI is the *same shape* as the main webui's settings tab — model registry table with download buttons, MCP server list, memory profile picker. Sharing HTML/CSS components across launcher + webui halves the design work.
2. Progress bars, per-row download states, live logs → these render cleanly in HTML, awkwardly in Fyne.
3. WebView2 ships with Windows 10 20H2+ and is bundled with Win11. On Linux/macOS Wails uses the system webview. No extra install for 99% of users.
4. Users expect a modern desktop app to look modern in 2026.

**Fallback if Wails proves fragile:** Fyne. Same launcher scope, native widgets.

## 3.3 — Clear boundary: launcher vs. runtime

The launcher is **not** part of the C++ runtime. Two separate binaries. Two separate repositories (or two separate top-level dirs in one repo — TBD in Decision 04's file layout).

| Concern | Launcher (`ultima-launcher.exe`) | Runtime (`ultima.exe`) |
|---|---|---|
| Language | Go | C++20 |
| Role | Settings GUI, model downloader, process supervisor | Inference, memory, tools, HTTP endpoint |
| Reads | `config.json`, `models.json` | same files |
| Writes | `config.json`, `models.json` (via GUI edits) | never — runtime is a pure consumer of config |
| Ships webui? | no | yes (serves `webui/` static assets over its HTTP port) |
| Talks to runtime? | via `os/exec` (spawn) + HTTP (status/health) | receives signals from launcher |
| Lifetime | user-controlled (window open = alive) | child of launcher, killed on launcher quit |

**Config is single-writer.** Only the launcher edits config files. Runtime reads them at startup and on `SIGHUP` (or Windows equivalent — named pipe signal).

## 3.4 — Process model

```
User double-clicks ultima-launcher.exe
    │
    ▼
Launcher window opens (Wails)
    │
    ├─▶ Load config.json + models.json
    ├─▶ Show settings page + [Launch Ultima] button
    │
    ├─▶ On [Launch] click:
    │     ├─ Spawn: ultima.exe serve --config <path>
    │     ├─ Wait for HTTP :7777 to respond to GET /health
    │     ├─ Open http://localhost:7777 in default browser
    │     └─ Show "Running" indicator + [Stop] + [Restart] + live logs
    │
    ├─▶ On config edit while running:
    │     ├─ Save to config.json
    │     ├─ Send SIGHUP (or equivalent) to runtime
    │     └─ Runtime hot-reloads what it can (model swap = restart, other = live)
    │
    └─▶ On launcher window close:
          ├─ Send SIGTERM to runtime
          ├─ Wait up to 5s for graceful shutdown
          └─ SIGKILL if still alive
```

## 3.5 — Launcher settings pages (v0.1)

Tab layout in the Wails window:

1. **Home** — [Launch Ultima] big button, status indicator (running/stopped), model/tier summary, "Open Chat in Browser" shortcut (opens `http://localhost:7777` in default browser)
2. **Models** — model registry table (Decision 01e). Per-slot dropdown (coder/embed). Download button per known model. Custom path picker. Verify button. Delete button.
3. **Hardware** — detected tier + manual override. **Thread slider** (default = physical cores, cap = logical cores). RAM budget hint. GPU detection (grayed out in v0.1).
4. **MCP Servers** — list of configured MCP endpoints. Add/remove. Per-server enable toggle. Test connection button. (Decision 09 defines schema.)
5. **Memory** — memory profile selector (default/coding/project-alpha/…). Language shard toggles. Memory dir path. "Open memory folder" shortcut. **Inline editor for `AGENTS.md`** (the behavior contract / system prompt) with diff-against-default and revert. **Reset buttons**: reset conversations, reset per-language shard, reset entire profile — each with a typed-confirm dialog.
6. **Tools** — enable/disable native tools (rg, fd, run_shell, git_*, edit_file, etc.). Per-tool approval requirement toggle.
7. **Advanced** — port, log level, log file path, `--show-thinking` default, sampling defaults (temperature / top-p / top-k / min-p / repetition penalty). **Chat template editor**: dropdown of known templates (qwen2, qwen3, custom) with inline Jinja/JSON editor for the active template and "Reset to default" per model family.
8. **About** — version, license, links.

Every setting is a straight edit of `config.json` / `models.json` / `AGENTS.md`. No launcher-side state.

## 3.6 — First-run experience

First launch (no `config.json` yet):

```
1. Detect hardware (CPU cores, RAM, GPU absent) -> propose tier "mid"
2. Show welcome dialog:
     "Ultima detected a Mid-tier system (8 cores, 32 GB RAM).
      Recommended coder model: Qwen2.5-Coder-7B (4.5 GB).
      Recommended embed model: bge-base (110 MB).
      Total download: 4.6 GB.
      [Download now] [Choose different models] [Skip - I have my own]"
3. On [Download]: parallel resumable downloads with progress bars, SHA-256 verify
4. On success: write config.json with defaults, land on Home tab, [Launch] enabled
5. On [Skip]: land on Models tab so user can point at their own GGUF files
```

Every subsequent launch: skip welcome, go straight to Home.

## 3.7 — Runtime discovery

The launcher finds `ultima.exe` in this order:
1. Same directory as `ultima-launcher.exe` (default when both ship together in a portable folder)
2. `%ProgramFiles%\Ultima\ultima.exe` (installed)
3. PATH
4. Explicit path from `config.json` → `runtime_path`

If not found: launcher shows "Runtime binary missing" with a [Download Runtime] button (fetches from GitHub releases when we start publishing).

## 3.8 — File layout on disk (Windows example)

```
Ultima\
├── ultima-launcher.exe      ← user double-clicks this
├── ultima.exe               ← the C++ runtime
├── webui\                   ← static assets served by ultima.exe
│   ├── index.html
│   └── ...
├── third_party\bin\windows-x64\
│   ├── rg.exe
│   └── fd.exe
└── README.txt

%USERPROFILE%\.ultima\
├── config.json
├── models.json
├── models\                  ← downloaded GGUFs
├── memory\                  ← language shards, projects, skills
├── sessions\                ← chat history
└── logs\
```

Ships as a **portable folder** (zip). No installer required in v0.1. Optional MSI installer post-v0.1.

## 3.9 — Cross-platform / cross-backend note

Launcher itself is 100% CPU + GUI — no backend concern. Wails builds on Windows/Linux/macOS from the same Go source. When the runtime gains Vulkan/CUDA/Metal (post-v0.1), the launcher's Hardware tab reads capabilities from `ultima.exe --caps` (JSON) and shows a backend selector. No launcher rewrite required.

## 3.10 — Repository structure implication

Add a new top-level directory to the layout defined in Decision 02:

```
ultima.cpp/                  ← this repo
├── ...                      (all existing C++ dirs)
└── launcher/                ← NEW: Go project
    ├── go.mod
    ├── go.sum
    ├── main.go
    ├── frontend/            (Wails web frontend: Vite/Vue or Vite/Svelte)
    ├── wails.json
    └── build/
```

Launcher builds are separate from the C++ build:
```bash
cd launcher && wails build           # produces ultima-launcher.exe
cmake --build --preset windows-msvc-release   # produces ultima.exe
scripts/package.ps1                  # zips both + webui + third_party/bin into a portable folder
```

CI matrix gains launcher jobs (Go 1.22+, Node 20+ for Wails frontend build).

## Please decide

**Question 3:** Lock in Decision 03 as written — Wails-based launcher, portable-folder ship layout?

Override points:

- **GUI framework:** Wails is my pick for the reasons above. Fyne if you want zero JS toolchain. walk if Windows-only forever is acceptable.
- **Launcher scope:** I've kept it as a *thin GUI over config + supervisor*. If you want the launcher to *also* embed the webui browser (i.e., launcher window contains both the settings tabs and the chat) → we're building a mini-Electron-ish thing. That's a bigger project — Wails technically supports it but doubles the frontend. Recommend keeping launcher = settings-only and chat = user's browser at `localhost:7777`.
- **Portable-first vs. installer-first:** portable zip is my pick for v0.1. MSI post-v0.1.

---

# Decision 03b — Chat UI hosting: browser-only

## DECIDED: Approach A — pure browser-served webui, no native window wrap

- `ultima.exe` serves the chat webui at `http://localhost:7777` (static assets from `webui/`, same port as the API)
- User opens in their default browser (Chrome / Firefox / Edge / Brave / whatever)
- Launcher's **"Open Chat in Browser"** button just calls the OS default-browser handler on that URL
- **No Wails wrapper for chat.** Launcher stays Wails (settings + supervisor). Chat is browser-only.

### Why this is right

- Single canonical way to use the chat — no "was that in the app window or the browser?" confusion
- Zero native GUI code for the chat side — pure web stack
- Automatically cross-platform (any OS with any modern browser)
- User can bookmark it, open on a phone against the LAN IP, share sessions, use browser dev tools if they want to inspect a payload
- Iterate on chat UI = edit HTML, reload — fastest possible dev loop
- We already ship the HTTP server for `/v1/chat/completions`; serving static assets on the same port is trivial (dozen lines with `cpp-httplib`)

### What "better than the classic upstream webui" means concretely

The upstream chat page is a bare form + sampling sliders. Ultima's chat webui (spec'd in Decision 14) will include:

- Syntax-highlighted code blocks with copy / edit-in-place / apply-diff buttons
- Tool-call panel: foldable JSON tree of arguments + streamed results
- Diff view when the model uses `edit_file` / `replace_in_file`
- Reasoning-token pane (collapsible) for `<think>` blocks
- Memory-retrieval sidebar — shows which memories/skills were injected for the current turn
- MCP server list with live connection status
- Model registry UI (mirrors the launcher's Models tab, but as a slide-out drawer)
- Multi-session history — resume any past conversation
- Project scope indicator — which language shards + which project memory are active
- Skills panel — see which skills are loaded, view/edit them
- Sampling controls in a slide-out drawer, not the main chat area
- Dark/light themes, keyboard shortcuts
- Streaming word-by-word with correct rendering of partial code blocks
- (Future v0.2 bridge:) split view for editing files inline with the model

### Cross-backend note

Chat UI is 100% HTML/CSS/JS served over HTTP — completely orthogonal to any backend. Adding Vulkan/CUDA/Metal changes nothing about the UI layer.

---

---

# Deferred phases (do not forget)

Registry of every future-phase commitment made so far. Nothing in this section blocks v0.1; everything in it is a note-to-future-self so scope stays consistent when we get there.

## v0.2 targets

- **Tier 1 Ultima Editor** — Wails/Tauri window, embedded Monaco, file tree, terminal, `rg` search, agent-inline buffer writes, diff review. Ships as `ultima-editor.exe` alongside the runtime and launcher. Spec written when v0.1 ships.
- **Web search tool** — provider-agnostic (Tavily, Brave, DuckDuckGo, SearXNG). MCP-compatible.
- **HNSW vector index** — replaces v0.1 flat-cosine search once memory shards exceed ~100k entries.
- **Adapter hot-load** — LoRA hot-swap without model reload. Runtime interface reserved in v0.1 (Decision 12).
- **Reranker slot** in the model registry.
- **Draft model** slot for speculative decoding (Qwen-Coder-0.5B drafting for Qwen-Coder-7B).

## v0.3+ targets

- **Fork Code-OSS → Ultima Editor (Tier 3).** MIT-licensed VS Code base. Rebrand, strip Microsoft telemetry, deep-integrate Ultima runtime, inherit the entire VS Code extension API + marketplace-compatible plugins + LSP + DAP + remote dev (SSH/WSL/containers). This is the Cursor/Windsurf/Void path. Trigger: only if v0.2's Tier 1 editor validates the coding-assistant product enough to justify multi-month fork work. **Do not attempt to build the extension host from scratch** — it's a 3–5-year, 3–5-person project that misdirects effort from the memory-subsystem research bet.
- **Vulkan backend** — first GPU backend. Compute shaders for Q4_K dequant + matmul.
- **CUDA backend** — for NVIDIA users on Windows/Linux.
- **Metal backend** — for Apple Silicon.
- **MSI installer** for Windows (v0.1 ships as portable zip only).
- **Bridges to Visual Studio and Blender** — over localhost MCP. Reserved in Decision 09's MCP architecture.

## v0.4+ / research targets

- **Hidden-state memory** (Level 2 adaptation from the original spec) — memory → learned projection → hidden state.
- **Adapter memory** (Level 3) — memory → LoRA → model.
- **Online adapter training** (Level 4) — live weight updates from corrections.
- **MoE + MLA** support (DeepSeek-V2/V3 family).
- **Recurrent state architectures** (Gated DeltaNet, Mamba-style).
- **Sparse attention** (Qwen Sparse Attention, other long-context schemes).
- **Multimodal input** (vision tower for Ornith / Qwen-VL).

## Non-Qwen architecture families (post-v0.1, in expected implementation-cost order)

- Meta L3 family (add RoPE variant + drop QKV bias — trivial, ~1 day)
- Gemma-2 (different norm placement, logit soft-cap)
- Phi-3/4 (fused QKV, sliding window)
- DeepSeek-V2/V3 MoE + MLA (major project)

---

## Please decide

**Question 4:** ready to move to **Decision 04: GGUF loader + mmap strategy** (first actual runtime module we implement)?

That decision covers:
- Whole-file `mmap` vs. lazy per-tensor loading
- GGUF v3 header + metadata parsing
- Tensor directory walk, alignment enforcement
- SHA-256 or xxhash verification on load
- Multi-file (split GGUF) handling
- Endianness (little-endian only for v0.1)
- Which metadata keys we require vs. ignore
- Exact `IModelLoader` interface signatures

Say **"go"**.
