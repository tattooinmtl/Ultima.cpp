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
- **Reasoning-token streaming**: parser splits `<think>...</think>` from user-facing output; CLI has `--show-thinking` flag. Ornith's template always opens the assistant turn with `<think>\n` after the generation prompt; `enable_thinking=false` short-circuits with an empty `<think>\n\n</think>\n\n` stub.
- **qwen3_xml tool-call parser** variant — Ornith's tool format is **XML-nested**, not JSON:
  ```
  <tool_call>
  <function=name>
  <parameter=key>
  value
  </parameter>
  </function>
  </tool_call>
  ```
  distinct from Qwen2's `<tool_call>{"name":..., "arguments":{...}}</tool_call>`. Requires separate parser + emitter.
- **System message merging:** up to 2 consecutive system/developer messages merged with `\n`.
- **Tools injected into system prompt** with a fixed instruction block ("# Tools\n\nYou have access to the following functions:...").
- **Reference template** committed at `docs/references/chat_templates/ornith-1.5-9b.jinja` — source of truth for the M3 C++ renderer implementation.
- **Explicitly deferred:** vision tower / multimodal input (Ornith supports `<|vision_start|><|image_pad|><|vision_end|>` and video variants; we ignore that for v0.1 — text-only still works and matches published SWE-bench numbers).

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
- **`qwen35` (Qwopus-style hybrids)** — fused QKV + gated attention + SSM/Mamba layers + custom 248k vocab. Not a Qwen fork despite the name. Requires state-space kernels, fused-QKV path, gated attention variant, custom tokenizer. Realistic cost 4–8 weeks. Correct home: v0.4+ under "recurrent state architectures", not v0.1.

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
- `qwen35` / Qwopus hybrids (SSM+attention hybrid, fused QKV, gated attention, 248k custom vocab; v0.4+)

---

---

# Decision 04 — GGUF loader + mmap strategy

## Deliverable

A **`ultima --inspect <path.gguf>`** command that opens a real GGUF v3 file, memory-maps it, parses the header and metadata, walks the tensor directory, and prints a human-readable summary. No inference yet — this proves the whole file-loading pipeline works and is the prerequisite for every downstream module.

Example expected output:

```
$ ultima --inspect qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
File:         qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
Size:         386.7 MiB
Alignment:    32 bytes
GGUF version: 3

Metadata (48 keys):
  general.architecture           = qwen2
  general.name                   = Qwen2.5-Coder-0.5B-Instruct
  general.file_type              = 15 (MOSTLY_Q4_K_M)
  qwen2.block_count              = 24
  qwen2.context_length           = 32768
  qwen2.embedding_length         = 896
  qwen2.feed_forward_length      = 4864
  qwen2.attention.head_count     = 14
  qwen2.attention.head_count_kv  = 2
  qwen2.rope.freq_base           = 1000000
  qwen2.rope.dimension_count     = 64
  tokenizer.ggml.model           = gpt2
  tokenizer.ggml.tokens          = <151936 entries>
  ... (43 more)

Tensors (291 total):
  token_embd.weight              [896, 151936]   Q6_K       81.6 MiB
  blk.0.attn_norm.weight         [896]           F32        3.5 KiB
  blk.0.attn_q.weight            [896, 896]      Q4_K       471 KiB
  blk.0.attn_q.bias              [896]           F32        3.5 KiB
  ... (287 more)
  output_norm.weight             [896]           F32        3.5 KiB

Summary:
  Architecture:      qwen2
  Parameters:        494 M (approx.)
  Quantization:      Q4_K_M
  Context length:    32768
  Vocab size:        151936
  Total tensor size: 383.4 MiB
```

## 4.1 — GGUF v3 file format (public spec)

GGUF (GPT-Generated Unified Format) is documented at https://github.com/ggerganov/gguf (spec repository — we reference the spec, not code). Layout, little-endian throughout:

```
+---------------------------------------------------+
|  Header                                           |
|    u32  magic         = 0x46554747 ("GGUF")       |
|    u32  version       = 3                         |
|    u64  tensor_count                              |
|    u64  metadata_kv_count                         |
+---------------------------------------------------+
|  Metadata KV pairs (metadata_kv_count entries)    |
|    Each: key (gguf_string) + value_type (u32)     |
|          + value (variable, per value_type)       |
+---------------------------------------------------+
|  Tensor infos (tensor_count entries)              |
|    Each: name (gguf_string)                       |
|          n_dims (u32)                             |
|          dims[n_dims] (u64 each)                  |
|          type (u32, ggml_type enum value)         |
|          offset (u64, from tensor_data_base)      |
+---------------------------------------------------+
|  Padding to `alignment` (default 32 bytes)        |
+---------------------------------------------------+
|  Tensor data blob                                 |
|    Contiguous, tensors packed at their offsets    |
+---------------------------------------------------+
```

**Value types** we support in v0.1: u8/i8/u16/i16/u32/i32/u64/i64/f32/f64, bool, string, array-of-any-of-the-above (recursive one level).

**Tensor types (`ggml_type` enum)** we recognize by numeric value — we implement handlers for these in v0.1 (mapped to our own `DataType` enum so the banned name never appears in code):

| GGUF type value | Our name       | Notes                                |
|-----------------|----------------|--------------------------------------|
| 0               | `DataType::F32`  | full precision (norms, biases)     |
| 1               | `DataType::F16`  | half precision                     |
| 8               | `DataType::Q8_0` | 32-block, F16 scale                |
| 12              | `DataType::Q4_K` | K-quants super-block (256 elements) |
| 14              | `DataType::Q6_K` | K-quants super-block               |

All other numeric type values are logged as "unsupported quant type N, tensor <name> skipped" — the loader still enumerates them, we just can't dequant them in v0.1. This lets `--inspect` succeed on models with mixed exotic quants.

**Little-endian only.** GGUF spec permits big-endian but no shipping model uses it. If magic bytes are `47 47 55 46` (byte-reversed), we reject with a clear error.

## 4.2 — mmap strategy

**Whole-file mmap, read-only, private mapping.** Windows: `CreateFileMappingW` + `MapViewOfFile`. Linux/macOS (reserved): `mmap(PROT_READ, MAP_PRIVATE)`.

**Rationale:**
- Instant "load" — OS pages in weights lazily as we touch them during forward pass
- Multiple runs share the OS page cache — warm restarts are free
- The kernel handles paging pressure; we never run out of RAM by loading a model larger than free memory
- No copy from disk to userspace — the tensor's `.data()` pointer aims into the mmap
- Works with files larger than free RAM (as long as they fit in virtual address space, which for 64-bit Windows is many TB)

**No fallback to `fread`-based loading in v0.1.** If mmap fails (network drive, exotic FS), we return an error. Post-v0.1 we can add a slow copy-into-heap path.

**Windows-specific detail:** `CreateFileMappingW` requires the file handle to be open with `FILE_SHARE_READ` so `ultima models verify` can run concurrently.

**Lifetime rule:** the `LoadedModel` owns the mmap. All `TensorView` objects it hands out are non-owning references into that mapping. Closing the `LoadedModel` unmaps everything and invalidates every `TensorView`. No dangling — enforced by returning views through a `LoadedModel&` accessor, so views can't outlive the model.

## 4.3 — Metadata parsing

Parser is a straight linear read of the metadata section. For each KV:

1. Read `key` (gguf_string: u64 length + UTF-8 bytes, no null terminator)
2. Read `value_type` (u32)
3. Dispatch on type to read the value (scalars are fixed-size; strings and arrays are length-prefixed)

Values are stored in a `MetadataStore` — a `std::unordered_map<std::string, MetadataValue>` where `MetadataValue` is a `std::variant` over the supported types. Array-of-string is a special case (used for `tokenizer.ggml.tokens` with 151K entries) — stored as `std::vector<std::string>` inside the variant.

**Selective materialization:** for very large arrays (>1 MB when materialized), we do not copy — we store an offset+count and lazily provide a view when accessed. The 151K-token vocab is the primary case. `--inspect` prints `<N entries>` for these; a future `--inspect --dump-vocab` would materialize on demand.

**Required keys** (loader fails with a clear error if missing):
- `general.architecture` — string, drives model dispatch
- `general.file_type` — u32, tells us the dominant quant
- `<arch>.block_count` — u32/u64, layer count
- `<arch>.embedding_length` — u32/u64
- `<arch>.attention.head_count` — u32/u64
- `tokenizer.ggml.model` — string

Everything else is optional. Missing optional keys are logged at INFO, not errors.

## 4.4 — Tensor directory walk

Each entry: name + ndims + dims + type + offset.

We validate:
1. Name is non-empty valid UTF-8 (no NULs, <256 chars)
2. `ndims` ≤ 8 (spec bound; realistic max is 4)
3. Each dim > 0
4. Type is in the recognized enum values
5. `offset` is aligned to `general.alignment` (default 32)
6. `offset + size_bytes` ≤ file size

Bad entries are rejected with `ErrorCode::InvalidModel` and a message naming the offending tensor.

Tensor size in bytes is computed from `type` + `dims`. For K-quants (Q4_K, Q6_K), the block is 256 elements, so the product of dims must be divisible by 256 or we reject.

Stored in a `TensorDirectory` — `std::vector<TensorInfo>` plus a `std::unordered_map<std::string, size_t>` for name-lookup.

## 4.5 — Verification

**SHA-256 verification is not the loader's job.** That belongs to the model registry (Decision 01e) — it checks the downloaded file against the registry's expected hash *before* handing the path to the loader. The loader assumes its input has already been verified.

For v0.1, the loader only does:
- Magic byte check (`GGUF`)
- Version check (must be 3)
- Structural validation (all offsets in-bounds, types recognized, dims positive)

No mmap-time hashing. No repeated hashing per load.

## 4.6 — Split (multi-file) GGUFs — deferred

Large models on Hugging Face sometimes ship as `model-00001-of-00003.gguf` etc. v0.1 supports only single-file GGUFs — files matching the split pattern are rejected with a clear message: `"multi-file GGUFs not supported in v0.1; use a single-file quant"`. Post-v0.1 adds a `SplitGgufLoader` behind the same `IModelLoader` interface.

## 4.7 — Interface

Header: `include/ultima/model/i_model_loader.hpp`

```cpp
namespace ultima::model {

enum class DataType {
    F32, F16, Q8_0, Q4_K, Q6_K,
    // Extended post-v0.1
};

struct TensorInfo {
    std::string           name;
    std::vector<uint64_t> dims;      // ndims <= 4 in practice
    DataType              dtype;
    uint64_t              offset;    // absolute byte offset in the file
    uint64_t              size_bytes;
};

class TensorView {
public:
    const TensorInfo& info() const noexcept;
    const void*       data() const noexcept;   // points into mmap; do not free
};

class LoadedModel {
public:
    virtual ~LoadedModel();

    virtual const std::string&                          architecture() const = 0;
    virtual const class MetadataStore&                  metadata()     const = 0;
    virtual const std::vector<TensorInfo>&              tensor_infos() const = 0;
    virtual std::optional<TensorView>                   tensor(std::string_view name) const = 0;

    virtual uint64_t                                    file_size_bytes() const = 0;
    virtual uint64_t                                    alignment()       const = 0;
};

class IModelLoader {
public:
    virtual ~IModelLoader() = default;

    // Cheap: reads header only. Does not mmap the tensor blob. Good for
    // registry inspection and pre-flight checks.
    virtual tl::expected<class ModelMetadata, Error>
        inspect(const std::filesystem::path& path) = 0;

    // Full load: mmaps the file, validates the tensor directory. Fast — mmap
    // is O(1); the linear walks are O(tensor_count) plus O(metadata_count).
    virtual tl::expected<std::unique_ptr<LoadedModel>, Error>
        load(const std::filesystem::path& path) = 0;
};

// Concrete implementation for v0.1
std::unique_ptr<IModelLoader> make_gguf_loader();

} // namespace ultima::model
```

Note the earlier decision to swap `tl::expected` for `nonstd::expected` (from expected-lite) — the interface uses whichever alias we settle on in the actual implementation. Signature shape is stable.

## 4.8 — Files added to the repo for M2

```
include/ultima/model/
    dtype.hpp
    tensor_info.hpp
    tensor_view.hpp
    metadata_store.hpp
    loaded_model.hpp
    i_model_loader.hpp

src/model/
    CMakeLists.txt
    gguf/
        gguf_loader.hpp       (concrete class GgufLoader : IModelLoader)
        gguf_loader.cpp
        gguf_reader.hpp       (low-level byte reader over the mmap)
        gguf_reader.cpp
        gguf_types.hpp        (spec enum values, DataType mapping)
        gguf_metadata.cpp     (metadata KV parse)
        gguf_tensors.cpp      (tensor directory parse + validate)
    mmap/
        mmap_file.hpp         (platform-neutral RAII mmap handle)
        mmap_file_windows.cpp
        mmap_file_posix.cpp   (stub for v0.1, real for Linux/mac later)

apps/ultima-cli/
    main.cpp                  (add --inspect subcommand)

tests/unit/
    test_gguf_reader.cpp      (fixture-driven: tiny hand-crafted GGUF blobs)
    test_gguf_loader.cpp      (asserts specific keys/tensors present)

tests/fixtures/gguf/
    tiny_valid.gguf           (100-byte hand-crafted, 1 tensor, no data)
    tiny_bad_magic.gguf       (magic wrong)
    tiny_bad_version.gguf     (version 99)
    tiny_truncated.gguf       (file cut mid-tensor-directory)
```

Fixtures are hex-encoded in `.gguf.hex` files checked into git and reconstructed to binary at test-configure time by a small CMake helper (keeps binary blobs out of the repo).

## 4.9 — Dependencies pulled in by M2

Adds to `cmake/UltimaDeps.cmake`:
- **nonstd::expected** (`martinmoene/expected-lite`, pinned) — error-carrying return values
- **fmt** (already planned, pinned) — used by the error/log messages
- No JSON dep yet — GGUF metadata is not JSON

xxhash and cpp-httplib remain queued for later modules (model registry, HTTP server).

## 4.10 — Cross-backend note

| Concern | CPU (v0.1) | Vulkan | CUDA | Metal |
|---|---|---|---|---|
| mmap | Win: `CreateFileMappingW` / Posix: `mmap` | same | same | same |
| Tensor data access | `TensorView.data()` returns host pointer | must be copied to device buffer before use | pinned host → async H2D | unified memory on Apple Silicon (no copy) |
| Metadata parsing | pure CPU | pure CPU | pure CPU | pure CPU |
| Loader interface | `IModelLoader` unchanged | unchanged | unchanged | unchanged |

Adding a GPU backend later touches only *tensor upload*, not the loader. `LoadedModel::tensor(name)` still returns a host `TensorView`; the GPU backend copies from it on demand.

## 4.11 — Definition of done for Decision 04 / M2

- `ultima --inspect <path.gguf>` runs on real Qwen2.5-Coder-0.5B-Instruct-Q4_K_M.gguf and prints correct metadata + tensor list
- Loader rejects bad-magic, bad-version, and truncated fixtures with clear errors
- Unit tests cover: valid parse, missing required keys, out-of-bounds tensor offset, unsupported dtype (warns not errors), string vs. array-of-string values
- No banned identifiers in any new file (CI taboo gate passes)
- Build stays warnings-clean under `/W4 /WX`

## Please decide

**Question 5:** Lock Decision 04 as written and start implementing M2 in the next reply?

Override points:
- **`--inspect` vs. `models inspect <path>`** — my pick is a top-level `--inspect` flag now, refactored into `ultima models inspect` when the models subcommand exists. Say if you'd rather skip the top-level flag.
- **Fixture format** — I picked hex-encoded blobs reconstructed at build time. Alternative: commit small binary `.gguf` fixtures directly. Hex keeps `git diff` readable and repo textual.

Say **"go"** and I write the M2 code + commit.

---

# Decision 05 — CPU kernel policy

## Scope

Every arithmetic kernel Ultima runs on the CPU flows through the choices made here: SIMD level, alignment, threading, dequant approach, tensor storage layout. Wrong choices are expensive to reverse once forward passes exist.

## 5.1 — SIMD baseline: AVX2 required, scalar fallback

**Required at runtime on x86-64:** AVX2 + FMA. Detected via `__cpuid` at startup. If missing, exit with a clear error naming the missing feature.

- Every mainstream desktop/laptop CPU from 2014 onward has AVX2.
- Reference machine (Ryzen 7 5700U, Zen 2) has AVX2, no AVX-512.
- Zen 4 has AVX-512 but we're not tuning for it in v0.1 (marginal on our reference box).

**Scalar fallback path** for every hot kernel exists at compile time. Not shipped by default (dead code stripped), but the same source-level function is compilable in a scalar variant. Two reasons:
1. Correctness oracle: golden tests run the scalar variant to cross-check the SIMD variant on the same inputs.
2. Non-x86 platforms later (ARM64 for Apple Silicon, Raspberry Pi 5) — scalar is the fallback until NEON kernels exist.

**No AVX-512 code in v0.1.** Zen 4 desktops would get a boost from it, but that's a v0.2 opt-in flag.

**Intrinsics, not inline asm.** `<immintrin.h>` for MSVC/GCC/Clang. Portable across the three compilers, debuggable, no ABI landmines.

## 5.2 — Alignment

- All tensor storage aligned to **64 bytes** (one cache line, one AVX-512 register worth — future-proof, cheap).
- Custom aligned allocator in `include/ultima/tensor/aligned_alloc.hpp`.
- The mmap'd GGUF weights are aligned per the file's `general.alignment` (default 32). We do NOT re-copy them for alignment — we read via unaligned loads (`_mm256_loadu_si256`) inside dequant kernels. Zen 2 handles unaligned AVX2 loads at full throughput; the cost is a myth for our workload.
- Activations/scratch tensors allocated by us get 64-byte alignment.

## 5.3 — Threading model

**Choice:** hand-rolled fixed-size thread pool. No OpenMP, no TBB, no `std::async`.

**Rationale:**
- One place to reason about all parallelism.
- We control thread count precisely (matches launcher's Hardware tab slider).
- Cross-platform without conditional builds.
- OpenMP dependency spread would infect every consumer of a kernel; TBB adds a linkage.

**API sketch:**

```cpp
class ThreadPool {
public:
    explicit ThreadPool(std::size_t n_threads);
    ~ThreadPool();

    // Parallel-for over [0, n). Blocking. work_fn is called with the range
    // [start, end) each thread should process. Auto-chunks contiguously.
    void parallel_for(std::size_t n, std::function<void(std::size_t, std::size_t)> work_fn);
};
```

**Default thread count:** number of *physical* cores (not logical). On a Ryzen 7 5700U that's 8. Hyperthreading rarely helps compute-bound matmul; overrides via config for the curious.

**Parallelism scope:** per-op, coarse-grained. A matmul splits its output rows across threads. No task-graph scheduling in v0.1 — that's an M6+ concern if we want operator fusion.

## 5.4 — Dequantization approach

Two possible strategies:

| Approach | Description | Verdict |
|---|---|---|
| **A. Dequant-then-matmul** | Dequant a block to F32, then FMA into accumulator | Simple, general, wasteful — writes F32 scratch that immediately gets consumed |
| **B. Fused dequant+matmul** | Dequant a block inline inside the matmul inner loop, no intermediate F32 storage | The right approach for a Q4×F32 matvec. Standard technique. Every mainstream local runtime does this. |

**Choice: B (fused).** Non-negotiable for perf. A block of Q4_K weights (256 elements, 144 bytes) is unpacked into a register, immediately dotted with the corresponding activation slice, accumulated. No F32 weight buffer ever exists.

**Kernel shapes we implement in M1:**
- `matvec_q4k_f32` — Q4_K weight matrix × F32 activation vector → F32 output. **This is the generation-time hot path** (token-by-token forward pass runs this at every layer).
- `matvec_q6k_f32` — same shape, Q6_K weight (used by embedding/output projection when file_type is Q4_K_M).
- `matvec_q8_0_f32` — same shape, Q8_0 weight (some GGUFs keep certain tensors at Q8_0).
- `matmul_f32_f32` — F32 × F32 for norms, biases, computed KV.
- No mat-mat (batched N>1) Q4×F32 kernel in v0.1 — prompt processing runs the mat-vec kernel in a loop over tokens. Slower for long prompts but simpler code. Batched matmul is a v0.2 optimization.

**Element-wise kernels:** `add`, `mul`, `silu`, `swiglu`, `rmsnorm`, `rope`, `softmax`. All F32-in, F32-out, AVX2-vectorized where the shape allows.

## 5.5 — Tensor storage layout

- **Row-major.** All dims listed outer-to-inner as they appear in the GGUF tensor directory.
- **Contiguous.** No strides in v0.1 (all our tensors are freshly allocated with tight packing).
- **Weights are read-only mmap views**, not copies. `Tensor` internally distinguishes "owned buffer" (activations, KV cache) from "view over mmap" (weights). Both types implement the same public interface.

## 5.6 — Correctness harness

Every SIMD kernel has a scalar counterpart and a golden test that:
1. Generates random inputs at a fixed seed
2. Runs the scalar variant → reference output
3. Runs the SIMD variant → tested output
4. Asserts max absolute error < 1e-4 (F32 kernels) or exact match (integer/dequant kernels)

Test suite name: `test_kernels_correctness`. Runs in <1 second in Debug.

**Later (before M4):** a second oracle test compares Ultima's per-layer outputs against reference outputs captured from `transformers` on the same GGUF, ensuring bit-close (max abs err < 1e-3) numerical parity for the whole forward pass. Captured fixtures live under `tests/fixtures/reference_outputs/`.

## 5.7 — Compiler flags impact

Adds to `cmake/UltimaCompilerFlags.cmake` (Decision 02) — kernel translation units only, not the whole project:

- MSVC: `/arch:AVX2` on kernel objects
- GCC/Clang: `-mavx2 -mfma` on kernel objects, `-march=native` NOT used (breaks portability of the built binary)
- `-ffast-math` NOT used (breaks NaN handling, reproducibility)

**Per-file flags via CMake `target_compile_options(<lib> PRIVATE ...)` scoped to `src/kernels/`.** Prevents accidental AVX2 in general code that must run on the fallback path.

## 5.8 — File layout added at M1 (before writing kernel code)

```
include/ultima/tensor/
    aligned_alloc.hpp
    tensor.hpp            (owned buffer OR mmap view)
    dtype_traits.hpp

include/ultima/kernels/
    matvec.hpp            (matvec_q4k_f32, matvec_q6k_f32, matvec_q8_0_f32, matvec_f32_f32)
    elementwise.hpp       (add, mul, silu, swiglu)
    norms.hpp             (rmsnorm)
    rope.hpp              (Qwen2 variant; Qwen3+YaRN reserved as separate entry point)
    softmax.hpp
    cpu_features.hpp      (__cpuid wrapper, feature check at startup)

src/tensor/
    CMakeLists.txt
    tensor.cpp
    aligned_alloc.cpp

src/kernels/
    CMakeLists.txt        (adds /arch:AVX2 or -mavx2 -mfma scoped here)
    cpu_features.cpp
    matvec_q4k_f32.cpp
    matvec_q6k_f32.cpp
    matvec_q8_0_f32.cpp
    matvec_f32_f32.cpp
    elementwise.cpp
    rmsnorm.cpp
    rope.cpp
    softmax.cpp
    scalar/               (compiled only when ULTIMA_KERNELS_SCALAR_ONLY=ON, for tests)
        matvec_q4k_f32_scalar.cpp
        ... etc

src/runtime/
    thread_pool.hpp
    thread_pool.cpp

tests/unit/
    test_cpu_features.cpp
    test_thread_pool.cpp
    test_kernels_correctness.cpp
    test_aligned_alloc.cpp
```

## 5.9 — Cross-backend note

| Concern | CPU (v0.1) | Vulkan (v0.2+) | CUDA (v0.3+) | Metal (v0.3+) |
|---|---|---|---|---|
| SIMD level | AVX2 + FMA | n/a (SPIR-V) | n/a (SASS) | n/a (MSL) |
| Threading | ThreadPool (fixed) | queue submits | streams | command buffers |
| Dequant approach | fused in-register | fused in-shader | fused in-kernel | fused in-shader |
| Alignment | 64-byte host | device alignment per API | pinned + aligned | unified memory |
| Kernel dispatch | function pointer per `DataType` | pipeline per shape | kernel per shape | pipeline per shape |

Every kernel we write in v0.1 has a natural Vulkan/CUDA/Metal counterpart later — the dispatch layer (Decision 06) is what routes to the right backend.

## 5.10 — What we explicitly defer

- Batched (mat-mat) Q4×F32 kernels — matvec loop is enough for v0.1
- AVX-512 code — Zen 4 opt-in, v0.2
- Operator fusion (e.g., fused attention) — v0.2+
- KV cache paged / sliding-window optimizations — needed for long context at M4+; scoped in Decision 09 (KV cache)
- Speculative decoding — v0.3+ under the draft-model slot in the registry

## Please decide

**Question 6:** Lock Decision 05 as written?

Override points:
- **Threading:** hand-rolled pool is my pick. OpenMP is 3 lines of code and 20% smaller for the parallel_for pattern — argument for it is "less code to maintain," argument against is dependency spread. Say if you want OpenMP.
- **AVX-512:** ignored in v0.1. If you know users on Zen 4 or newer Intel who'd get a real win, say so and I add a v0.2 opt-in flag now instead of later.
- **Batched matmul:** deferred. If prompt processing speed matters more than code simplicity, we could add a mat-mat kernel in v0.1 — costs an extra week of implementation and testing.

Say **"go"** and I'll start M1 implementation (aligned allocator + thread pool + Tensor + `cpu_features` + first scalar-only kernel with round-trip test) in the next reply.

# Decision 09 — KV cache & prefix reuse policy

## Scope

Every token generation call the transformer runs has to (a) find or allocate storage for the attention keys/values across all layers, (b) reuse whatever prefix already sits in that storage instead of recomputing it, and (c) release the storage when the session ends. This decision fixes the layout, the reuse model, the persistence story, and what we defer. Wrong choices here get expensive fast: the KV cache is by far the largest live allocation at runtime, and prefix reuse is the single biggest win for chat-style workloads (system prompt + history is identical turn-to-turn).

Ownership: `kv_cache/` subsystem (spec §13). Required before M5 (streaming server). Design in place before M4 (transformer runtime) so the runtime writes to the right shapes on day one.

## 9.1 — What "KV cache" means here

At each transformer layer we compute two tensors per input token:
- **K**: shape `[n_kv_heads, head_dim]` (Grouped-Query Attention — Qwen2/Qwen3 both use GQA, `n_kv_heads < n_heads`).
- **V**: shape `[n_kv_heads, head_dim]`.

Attention at position `t` reads **all** K/V from positions `0..t` in the same layer. If we recomputed them for every generated token, generation would be O(t²) per token. The KV cache is the standard fix: append K/V once when a token is first processed, then read the whole prefix on every subsequent step.

The prefix-reuse question is orthogonal but shares the same buffer: if the *user's* next request begins with the same tokens as a previous one, the K/V for those tokens is already valid — we just point `pos` at the end of the match and prefill only the suffix.

## 9.2 — Storage layout

**Per layer, per slot, two contiguous tensors:**

```
K[layer]  : shape [n_kv_heads, n_ctx, head_dim]  dtype = f16 (default) or q8_0
V[layer]  : shape [n_kv_heads, n_ctx, head_dim]  dtype = f16 (default) or q8_0
```

- **Time axis in the middle, not innermost.** Appending token at position `pos` writes one `[n_kv_heads, head_dim]` slab. Reading position range `[0, pos)` for attention reads a contiguous stripe per head. This matches how attention actually consumes the cache and gives us dense inner loops for the matmul.
- **No paging in v0.1.** Each slot reserves `n_ctx` positions at allocation time. Simpler, no fragmentation, worst-case predictable. Paged attention (vLLM-style block manager) is a v0.3+ concern under long-context work.
- **Row-major, contiguous, 64-byte aligned** (Decision 05).
- **KV storage is owned (not mmap)** — the mmap distinction in Decision 05 §5.5 is for weights only. `Tensor` in "owned buffer" mode.

**Sizing formula** (per slot):
```
bytes = 2 * n_layers * n_kv_heads * head_dim * n_ctx * dtype_bytes
```

Reference: Qwen2-7B (n_layers=28, n_kv_heads=4, head_dim=128) at n_ctx=4096, fp16 → **~117 MB / slot**. At n_ctx=32k → ~940 MB / slot. Ryzen 7 5700U reference box (32 GB DDR4) handles a few 4k slots comfortably; long-context slots need Q8_0 KV (see §9.5).

## 9.3 — Slots

**A slot is one allocated KV cache** (all layers, both K and V) plus the metadata needed to reuse or evict it. One slot per concurrent active session.

- **CLI:** 1 slot, allocated at model load, sized to the launcher's `n_ctx` setting.
- **Server (M5+):** N slots, launcher-configured. **Hard default `N = 2`** on the reference box (single-user + editor-bridge / MCP client). Sizing heuristic (`floor((RAM_budget - weights - overhead) / bytes_per_slot)`) exposed as a "compute slots" button in the launcher for users who want it. All slots pre-allocated at model load — no runtime growth. Rejecting a new session because slots are full is a valid response; oversubscribing memory is not.
- **Eviction between sessions:** LRU. When a request arrives and no slot is free, the least-recently-used slot's token-id vector is discarded and the slot is reassigned. The K/V buffers are reused in place (no memset — the new prefill overwrites what it needs).

**Metadata per slot:**
```cpp
struct KVSlot {
    std::vector<TokenId> tokens;   // exact token ids currently valid in this slot
    std::size_t          pos;      // number of valid positions (== tokens.size())
    SessionId            owner;    // opaque; nullopt when free
    std::uint64_t        touched_at; // monotonic counter for LRU
    // K, V buffers live in a side allocation, one pair per layer
};
```

## 9.4 — Prefix reuse (the "prompt cache")

**In-session reuse** (same slot, next turn):

1. Tokenize the new full prompt → `new_tokens`.
2. Find the longest common prefix between `slot.tokens` and `new_tokens`. Call its length `lcp`.
3. Truncate: `slot.pos = lcp`. The K/V beyond `lcp` is stale but not zeroed — subsequent writes overwrite it.
4. Prefill only positions `[lcp, new_tokens.size())`. Generate as normal.

This alone eliminates re-processing the system prompt + prior turns for chat workloads. Every mainstream runtime does this; it's the highest-leverage single feature in this whole decision.

**Cross-session reuse** (v0.1, config flag, **off by default**):

Persistent disk store keyed by `(model_id, tokenizer_hash, block_index, block_hash)`:
- Split the token stream into fixed 256-token blocks.
- Hash each block's token ids (xxhash64) → `block_hash`.
- On prefill, look up each leading block; if present, mmap the K/V slab and copy it into the slot. Stop at the first miss.
- On generation completion, write any new fully-formed blocks back.

**Store layout:**
```
$ULTIMA_DATA/kv_cache/<model_id>/
    index.sqlite               (rowid → (block_hash, layer, offset, len))
    blocks/00/xx/xxxx.bin      (mmap'd K/V slabs, sharded by first hash bytes)
```

**Bounds:** on-disk size capped by config (default 8 GB per model). LRU eviction at the block level. Never grows unbounded.

**Invalidation triggers** (drop the whole store for a `model_id`):
- Model file swapped (mtime/size change on the GGUF).
- Tokenizer changed (`tokenizer_hash` mismatch on load — hash of the tokenizer vocab + merges).
- KV dtype changed (fp16 ↔ q8_0 mixing is not safe).

**What does NOT invalidate reuse** (KV cache is pre-sampler):
- Sampling parameters (temperature, top_p, top_k, min_p, repetition penalty).
- RNG seed.
- Max tokens / stop sequences.
- Streaming vs. non-streaming.

## 9.5 — KV dtype: fp16 and Q8_0 both in v0.1

- **fp16:** default when the loaded weight file uses fp16 or Q8_0 tensors (small models, oracle runs). One dtype path, no per-block dequant on attention read.
- **Q8_0:** default when the loaded weight file uses **any 4-bit K-quant** (Q4_K_M, Q4_K_S). Halves KV memory (117 MB → 60 MB per slot at Qwen2-7B/4k, ~470 MB at 32k). On the reference box (32 GB, no dGPU, 4-bit quants as the primary target) this is what makes long context practical; quality cost is small and dominated by the 4-bit weight noise anyway.
- **Auto-selection** based on file quant class; launcher slider can override to `fp16` for correctness debugging.
- **No fp8** — no portable CPU fp8; revisit with GPU backends.
- Attention read path handles both dtypes via the same kernel entry point (`attn_read_kv`); the Q8_0 variant fuses dequant into the softmax-input dot product (mirror of the Decision 05 §5.4 fused-dequant policy).

## 9.6 — Long context & RoPE scaling

Qwen2 ships with `n_ctx_train = 32768`. Qwen3 ships with YaRN scaling metadata in GGUF for extending beyond `n_ctx_train`.

- **Default:** cap `n_ctx` at `n_ctx_train`. Loud error if the launcher asks for more without YaRN metadata in the GGUF — never silently clamp (silent clamping produces confusing tokens-per-second numbers and broken long-doc behavior; the user needs to know their request wasn't honored).
- **YaRN:** if the GGUF declares `rope.scaling.type = yarn`, apply the YaRN factors in the RoPE kernel (Decision 05 §5.4 reserved `rope.hpp` as an entry point; the Qwen3 variant lands with Decision 06).
- **Sliding window attention:** deferred to v0.3. Qwen2/Qwen3 base models don't use it; Mistral-style windowed attention is a per-architecture toggle we can add when we support such a model.

## 9.7 — API sketch

```cpp
namespace ultima::kv_cache {

class KVCache {
public:
    struct Config {
        std::size_t n_slots;
        std::size_t n_ctx;
        std::size_t n_layers;
        std::size_t n_kv_heads;
        std::size_t head_dim;
        DType       dtype;            // fp16 or q8_0
    };

    explicit KVCache(const Config&);   // allocates all slots up front

    // Acquire (or steal via LRU) a slot for this session, returning its handle.
    SlotHandle acquire(SessionId);

    // Longest common prefix of the slot's current tokens and `new_tokens`.
    // Truncates slot.pos to the LCP as a side effect.
    std::size_t reuse_prefix(SlotHandle, std::span<const TokenId> new_tokens);

    // Write one token's K/V into the slot at slot.pos, then increment pos.
    void append(SlotHandle, std::size_t layer, KVView k, KVView v);

    // Read positions [0, slot.pos) for attention at the given layer.
    KVReadView read(SlotHandle, std::size_t layer) const;

    void release(SlotHandle);
};

} // namespace ultima::kv_cache
```

The transformer runtime (Decision 06) is the only caller. The server (M5) never touches `KVCache` directly — it hands the runtime a `SessionId` and lets the runtime pick the slot.

## 9.8 — File layout added at M4

```
include/ultima/kv_cache/
    kv_cache.hpp
    kv_slot.hpp
    prefix_store.hpp        (persistent block store; off by default)

src/kv_cache/
    CMakeLists.txt
    kv_cache.cpp
    prefix_store.cpp

tests/unit/
    test_kv_cache_slot.cpp
    test_prefix_reuse.cpp
    test_prefix_store.cpp
```

## 9.9 — What we explicitly defer

- **Paged / block-manager KV** (vLLM-style). Requires attention-kernel changes and a memory manager. v0.3+, tied to long-context work.
- **Cross-request batching** (sharing prefill work across concurrent requests). v0.3+.
- **fp8 KV.** No portable CPU fp8; revisit with GPU backends.
- **Sliding-window attention.** Per-architecture; add with the first model that needs it.
- **Speculative decoding cache interactions.** Draft-model slot in the registry already reserved (Decision 01e); KV interactions defined when speculative decoding lands.
- **Live cache introspection / eviction API for tools.** Diagnostic dumps only in v0.1.

## 9.10 — Locked settings (reference box: Ryzen 7 5700U, 32 GB, no dGPU, 4-bit quants primary)

All four override paths **are implemented in v0.1**. Defaults are tuned for this hardware and a single primary user; every default is a launcher-editable knob.

| Setting | Default | Why this default here |
|---|---|---|
| **Cross-session persistent store** | **off** | Single user, one machine — hit rate is low across boots, and silent disk growth / stale-cache debugging outweighs the win. Flip on if disk is cheap and system prompts are stable. |
| **KV dtype** | **auto from weight file** — fp16 for fp16/Q8_0 weights, **Q8_0 for Q4_K_M/Q4_K_S** | 4-bit quants are the primary target. Halving KV to Q8_0 turns "4k context, comfortable" into "16k context, comfortable" on 32 GB. Correctness cost is small relative to the 4-bit weight noise. |
| **Server slot count** | **2** (single user + one MCP/editor client) | Sizing heuristic exists (`floor(...)`) as a launcher "compute" button for users on bigger boxes; 2 is what the reference box actually wants. |
| **`n_ctx` above `n_ctx_train` without YaRN metadata** | **loud error** | Silent clamping produces confusing tok/s numbers and broken long-doc runs — user needs to know their request wasn't honored. |

**Locked.** M4 transformer runtime targets this API. Ship-quality but tuned for the reference box; every default is one launcher toggle away from something else.

# Decision 06 — Model architecture layer (`IModel` dispatch)

## Scope

One narrow interface (`IModel`) that the transformer runtime (M4) calls to execute a forward pass and manage KV cache slots (Decision 09). Concrete implementations select themselves from GGUF metadata at load time. v0.1 ships **Qwen2** (which covers Qwen2.5-Coder — the primary coding target) and **Qwen3** (which covers Qwen3-Coder). Both are GQA transformer stacks; the differences are the RoPE flavor, the normalization placement, and a handful of tensor names — small enough to keep in one file per architecture.

Ownership: `include/ultima/model/*.hpp` + `src/model/*.cpp`. Required before M4 code lands.

## 6.1 — Why both, not just Qwen2

Qwen2.5-Coder-7B-Instruct at Q4_K_M is the ship-day coding model (Decision 01). Qwen3-Coder is the same story with slightly better output and YaRN long-context support (Decision 01c). Building the layer with two live architectures on day one is the only way to prove the abstraction is a real seam and not a decoration around one implementation. It's also how the launcher (Decision 03) can offer both without a runtime rebuild.

## 6.2 — The interface

```cpp
namespace ultima::model {

struct ModelDims {
    std::size_t n_layers;
    std::size_t n_heads;      // attention heads
    std::size_t n_kv_heads;   // grouped-query (Qwen2/3 both use GQA)
    std::size_t head_dim;
    std::size_t hidden;       // n_heads * head_dim
    std::size_t ffn_hidden;   // SwiGLU intermediate size
    std::size_t vocab;
    std::size_t n_ctx_train;
};

struct RopeConfig {
    float       freq_base;    // 10000 for Qwen2; per-file for Qwen3
    std::size_t rope_dim;     // == head_dim for both; kept explicit
    enum class Scaling { None, YaRN } scaling = Scaling::None;
    float       yarn_factor       = 1.0f;    // set from GGUF when Scaling::YaRN
    float       yarn_ctx_train    = 0.0f;
    float       yarn_attn_factor  = 1.0f;
    float       yarn_beta_fast    = 32.0f;
    float       yarn_beta_slow    = 1.0f;
};

class IModel {
public:
    virtual ~IModel() = default;

    virtual const ModelDims&  dims()  const noexcept = 0;
    virtual const RopeConfig& rope()  const noexcept = 0;
    virtual const char*       arch()  const noexcept = 0;   // "qwen2" | "qwen3"

    // Prefill a run of tokens starting at kv_slot.pos. Advances slot.pos.
    // logits (last token only) written to `out_logits` [vocab].
    virtual void prefill(kv_cache::SlotHandle slot,
                         std::span<const std::int32_t> tokens,
                         float* out_logits) noexcept = 0;

    // One-token decode using the slot's current state; appends to slot.
    virtual void decode(kv_cache::SlotHandle slot,
                        std::int32_t token,
                        float* out_logits) noexcept = 0;
};

std::unique_ptr<IModel> load_from_gguf(const model::GgufFile& file,
                                       kv_cache::KVCache& kv,
                                       runtime::ThreadPool& pool);

} // namespace ultima::model
```

Prefill/decode are the only two hot-path methods. Sampling lives outside `IModel` (Decision 08) — the runtime takes `out_logits` and feeds it to the sampler chain.

## 6.3 — Dispatch

`load_from_gguf` reads `general.architecture` from GGUF metadata:

| GGUF `general.architecture` | Concrete class      | Notes |
|---|---|---|
| `qwen2`                     | `Qwen2Model`        | Qwen2, Qwen2.5, Qwen2.5-Coder all report `qwen2` |
| `qwen3`                     | `Qwen3Model`        | Qwen3 and Qwen3-Coder both report `qwen3` |
| anything else               | error at load       | Do not silently fall back |

Different fine-tunes on the same architecture share the same class — the tokenizer + chat template distinguishes them (Decision 07).

## 6.4 — Architectural differences that matter for v0.1

| Aspect | Qwen2 / Qwen2.5-Coder | Qwen3 / Qwen3-Coder |
|---|---|---|
| Positional | RoPE (base 10000), no scaling in base checkpoints | RoPE + YaRN scaling metadata for long context |
| Norm         | RMSNorm (pre-norm on attention and MLP) | RMSNorm (pre-norm) + Q/K RMSNorm inside attention |
| Attention    | GQA (n_kv_heads < n_heads) | GQA with Q/K normalization |
| MLP          | SwiGLU (gate/up/down) | SwiGLU (gate/up/down) |
| Tie weights  | Some sizes tie embed/lm_head, some don't | Same story |
| Special toks | `<|im_start|>` / `<|im_end|>` chat markers | Same markers, extended template (thinking tags optional) |

Two concrete classes, one shared `Qwen2Block` layer routine reused with a config flag for Qwen3's Q/K RMSNorm. Tensor-name lookup uses a per-arch table so `blk.{i}.attn_q.weight` and friends resolve without string juggling in the hot path.

## 6.5 — Tensor-name mapping

Fixed table per architecture, filled once at load. Missing tensors are a hard error at load — never silently zero-fill.

```cpp
struct TensorSlot { const char* name; TensorRef ref; bool required; };
```

## 6.6 — Long-context / YaRN

RoPE kernel (Decision 05 §5.4) already reserved the entry point. Qwen2 uses the standard RoPE path. Qwen3 reads `rope.scaling.type = yarn` + factors from GGUF and passes them into a `rope_f32_yarn` variant. Behind Decision 09 §9.6: if scaling metadata is present, `n_ctx > n_ctx_train` is allowed; otherwise it's a loud error.

## 6.7 — Files

```
include/ultima/model/
    imodel.hpp           (interface + factory)
    qwen2_model.hpp
    qwen3_model.hpp
    tensor_slots.hpp     (name → TensorRef helpers)

src/model/
    CMakeLists.txt
    imodel.cpp           (factory / dispatch)
    qwen2_model.cpp
    qwen3_model.cpp
    qwen_block.cpp       (shared attention + MLP block with per-arch flags)
```

## 6.8 — Deferred

- Non-Qwen architectures (Llama, Mistral, Phi, Gemma). Each is a new concrete class; the interface is the seam.
- Batched (N>1) prefill. `prefill` takes a token span but processes one-at-a-time via the matvec loop (Decision 05 §5.4). Batched matmul is a v0.2 kernel add.
- Vision / audio heads. Text-only in v0.1.
- Adapter / LoRA hot-load (Decision 13 stub).

## 6.9 — Locked

Two architectures, one interface, dispatch by GGUF `general.architecture`. Coding-model priority is Qwen2.5-Coder (via `Qwen2Model`) and Qwen3-Coder (via `Qwen3Model`); every other Qwen fine-tune on the same architecture just works.

---

# Decision 07 — Tokenizer strategy (smallest fast path)

## Scope

Turn a UTF-8 string into a `std::vector<int32_t>` of token ids, and turn ids back into UTF-8. Load the vocabulary and merges from GGUF metadata (no external tokenizer.json). Render Qwen2/Qwen3 chat templates without pulling in Jinja. Every millisecond here is a millisecond the user waits before the first token comes back on the reference box, so this is the smallest-hardware-first design.

Ownership: `include/ultima/tokenizer/*.hpp` + `src/tokenizer/*.cpp`. Required before M3.

## 7.1 — What "fast on small hardware" means concretely

- **Encode ~200k chars/s single-threaded on the reference box.** Prompts of 4–8k tokens (typical coding session) tokenize in well under 100 ms.
- **Zero external tokenizer files.** GGUF ships the vocab and merges; we read them once at load. No Python, no HuggingFace `tokenizer.json`, no `tokenizers` crate.
- **No allocations in the encode hot loop.** Thread-local scratch (small `std::vector<int32_t>` reused across calls).
- **Byte fallback is a table, not a branch.** 256-entry precomputed array of byte-fallback token ids so the encoder never falls off the trie into a slow path.

## 7.2 — Algorithm: byte-level BPE with a trie

Qwen2 and Qwen3 both use byte-level BPE (`tokenizer.model = gpt2`-style with the `Ġ` space marker). Load steps:

1. Read `tokenizer.ggml.tokens` (string list, one per id) and `tokenizer.ggml.merges` (space-separated pairs) from GGUF.
2. Build a **hash map `token_str → id`** (one lookup for special tokens and byte-fallback).
3. Build a **merge-rank hash map `(id_a, id_b) → rank`** for the BPE loop.
4. Build a **256-entry byte-fallback array** mapping every possible input byte to its `<0xXX>` fallback token id.
5. Cache special-token ids (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, and Qwen3's thinking markers).

**Encode:**

1. Regex pre-tokenization (GPT-2 pattern, cached compiled regex).
2. For each pretoken: seed with byte-fallback ids; iteratively apply the lowest-rank merge until no merge in the sequence has a rank; append to output.
3. No string interning inside the loop — everything runs on `int32_t` pairs after the initial byte split.

Fast paths:
- Pretokens ≤ 1 byte: single byte-fallback lookup, no merge loop.
- Pretokens whose byte sequence exactly matches a full vocab entry: hash lookup shortcut before entering the merge loop.

**Decode:** id → bytes → utf-8 pass-through. Handles the `Ġ` → space conversion. Streaming decode buffers partial multi-byte UTF-8 characters until the next chunk arrives.

## 7.3 — Chat template rendering

Qwen2/Qwen3 templates are simple ChatML-style loops (`<|im_start|>{role}\n{content}<|im_end|>\n`). We do **not** ship a Jinja engine. Instead:

- Two hard-coded renderers (`render_qwen2_chatml`, `render_qwen3_chatml`) selected by GGUF `general.architecture` (dispatch in Decision 06).
- Each takes `std::span<const ChatMessage>` where `ChatMessage = {role, content, name?, tool_calls?}` and returns a `std::string`.
- Qwen3 extension: optional `<think>...</think>` block toggled per-request (thinking mode is a decode-time UX choice; the renderer just emits or suppresses).
- Tool call rendering: JSON-serialized tool descriptions in the system slot when tools are active (wiring lives in Decision 10).

If a user loads a fine-tune with a non-standard template, we fall back to raw text and log a warning. Jinja is a v0.2 concern behind a build flag.

## 7.4 — Special-token handling

Special tokens are never split by the regex or byte fallback. Encode splits the input at the first occurrence of any registered special-token string, emits its id, then continues. A boolean per-call parameter (`allow_special`) gates whether raw input strings can inject specials (default: **off** for user content, **on** for template renderers).

## 7.5 — Files

```
include/ultima/tokenizer/
    tokenizer.hpp        (encode/decode + BpeTokenizer class)
    chat_template.hpp    (render_qwen2_chatml / render_qwen3_chatml)
    special_tokens.hpp

src/tokenizer/
    CMakeLists.txt
    bpe_tokenizer.cpp
    chat_template_qwen2.cpp
    chat_template_qwen3.cpp
    pretokenize.cpp      (regex + fast-path splits)
```

## 7.6 — Deferred

- Jinja template engine (v0.2 build flag; needed only for non-Qwen fine-tunes).
- SentencePiece / Unigram tokenizers (needed for Llama-family models).
- Parallel batch encode (encode is fast enough single-threaded at v0.1 prompt sizes).
- Vocab loading from external `tokenizer.json` (GGUF is enough for the shipped models).

## 7.7 — Locked

Byte-level BPE loaded from GGUF, trie + merge-rank hashmap, thread-local scratch, hard-coded ChatML renderers for Qwen2 and Qwen3, no Jinja in v0.1. This is the shortest path to sub-100 ms prompt tokenization on the reference box.

---

# Decision 08 — Sampling

## Scope

Everything between "logits vector for the next token" and "int32 token id". Sampling is where most of the "why did the model say that" questions get answered on a small local box — bad defaults here mean the coding model repeats itself, hallucinates identifiers, or gives up early on a hard function. This decision picks the ladder of transforms, the defaults for a coding workload, and the RNG story so runs are reproducible when you want them to be.

Ownership: `include/ultima/sampler/*.hpp` + `src/sampler/*.cpp`. Needed alongside M4.

## 8.1 — The transform ladder (in order)

Each transform is a `Sampler` node. The runtime builds a chain per request and applies it left-to-right to the logits vector before argmax/multinomial.

1. **Logit bias** — add per-token biases (JSON API `logit_bias`).
2. **Repetition penalty** — divide/multiply logits of recently seen tokens (window: last 64 by default).
3. **DRY penalty** — pattern-repetition penalty (blocks the "keep echoing the previous line" failure common in code).
4. **Frequency + presence penalty** — OpenAI-compatible.
5. **Temperature** — `logits /= temp`. `temp == 0` short-circuits to greedy (no downstream transforms needed).
6. **Top-k** — keep the top-k logits, zero the rest.
7. **Top-p (nucleus)** — cumulative-prob cutoff.
8. **Min-p** — drop tokens below `min_p * max_prob`.
9. **Softmax + multinomial draw** — RNG from a seeded `std::mt19937_64`.

Chain order matters: penalties apply to raw logits, temperature scales after penalties, filters cut after temperature, softmax happens last. Every transform is a no-op when its config is at the identity.

## 8.2 — Defaults (coding-focused)

| Param | Default | Reason |
|---|---|---|
| `temperature`        | 0.2 | Coding wants determinism-ish; 0.7 chat defaults hallucinate identifiers |
| `top_k`              | 40 | Prunes the tail; cheap; standard |
| `top_p`              | 0.95 | Nucleus safety net when the model is uncertain |
| `min_p`              | 0.05 | Kills the long noise tail that top-p sometimes lets through |
| `repetition_penalty` | 1.1 | Small; larger values break formatted code (variable reuse is legit) |
| `repeat_last_n`      | 64  | Enough to catch immediate repeats without penalizing valid re-references |
| `dry_multiplier`     | 0.8 | On by default — dry blocks the "same 3 lines forever" failure mode |
| `dry_base`           | 1.75 |     |
| `dry_allowed_len`    | 2   |     |
| `frequency_penalty`  | 0.0 | Off — redundant with repetition penalty for local use |
| `presence_penalty`   | 0.0 | Off — same reason |
| `seed`               | random per request | Set explicitly for reproducibility |

Greedy short-circuit: `temperature == 0` → argmax, skip everything after step 5. All penalty transforms still apply (they only need the recent-token window).

## 8.3 — RNG

- `std::mt19937_64` seeded from the request's `seed` (or `std::random_device()` when unset).
- One RNG per request/slot (so two concurrent server requests don't interleave draws).
- Seed is captured and returned in the response metadata — reproducing a run means replaying the seed.

## 8.4 — Streaming interaction

Sampler is stateless across tokens *except* for the recent-token window (for penalties). The window lives in the slot's metadata (Decision 09 §9.3 — `KVSlot.tokens` doubles as the penalty window; no separate ring buffer).

## 8.5 — Files

```
include/ultima/sampler/
    sampler.hpp          (Sampler chain + Config)
    transforms.hpp       (penalty / temp / top-k / top-p / min-p / dry)

src/sampler/
    CMakeLists.txt
    sampler.cpp
    transforms.cpp
    dry.cpp              (kept separate — matcher is the complex one)
```

## 8.6 — Deferred

- Mirostat v1/v2 — interesting for chat, not helpful for code; add behind a flag in v0.2.
- Guided decoding / JSON-schema constrained sampling — v0.3+; useful for tool-calling reliability.
- Beam search — not applicable to chat/coding, deliberately never.
- Speculative decoding hookup — belongs to draft-model slot (Decision 01e).

## 8.7 — Locked

Nine-node chain, coding-tuned defaults, one RNG per request, `temp == 0` shortcuts to argmax. Sampler builds per request; slot owns the recent-token window; response returns the seed for reproducibility.

---

# Decision 10 — Tools + MCP

## Scope

What the model can *do* on the user's machine beyond emitting tokens. Two layers: **built-in tools** compiled into the runtime, and **MCP servers** discovered from launcher config. Tools are named, JSON-schema-typed, and offered to the model via the OpenAI-style `tools` array in the chat request. This decision fixes the built-in tool set (matching the user's requested lineup), the MCP host model, and the permission story.

Ownership: `include/ultima/tools/*.hpp` + `src/tools/*.cpp` + `src/mcp/*.cpp`. Wires into M5 (HTTP server) and the launcher (Decision 03).

## 10.1 — MCP host

**Launcher owns MCP subprocess lifetime.** Each MCP server declared in the launcher config is a long-lived stdio child process (per the MCP spec). The Ultima runtime talks to them through the launcher's IPC seam (Decision 03) — the runtime never spawns processes itself, keeping it sandboxable.

- Transport: stdio JSON-RPC (MCP 1.0). WebSocket transport reserved for v0.2.
- One process per server, restarted on crash with backoff.
- `initialize` handshake at startup; `tools/list` cached and re-fetched on `notifications/tools/list_changed`.
- Tool call flow: runtime → launcher → MCP server → response → runtime → model.

## 10.2 — Built-in tool set (v0.1)

Compiled in, no MCP hop. The list matches the user's brief; every tool has a JSON schema, a permission scope, and a rate limit knob.

### File & shell
- `bash_tool(command, cwd?, timeout_ms?)` → stdout/stderr/exit_code. Sandbox: launcher-configured allow-list of commands; default allow: read-only ops (`ls`, `cat`, `git status`), deny writes.
- `create_file(path, content, overwrite?)` — under a launcher-configured project root.
- `str_replace(path, old, new)` — exact-string replacement, refuses ambiguous matches.
- `view(path, offset?, limit?)` — file view with line numbers.
- `present_files(paths[])` — surface files back to the UI (no side effect).

### Memory (per-user, persistent — see Decision 12)
- `memory_read(key)`, `memory_write(key, value)`, `memory_str_replace(key, old, new)`, `memory_append(key, text)`, `memory_delete(key)`, `memory_list(prefix?)`.

### Web & search
- `web_search(query, limit?)`, `web_fetch(url, max_bytes?)`, `image_search(query, limit?)`.
- `conversation_search(query)`, `recent_chats(limit?)`, `read_conversation(id)` — over the runtime's own chat history store (Decision 12).

### Visualization widgets (renderers, side-effect-free)
`chart_display_v0`, `comparison_card_display_v0`, `featured_card_display_v0`, `product_carousel_display_v0`, `itinerary_display_v0`, `places_map_display_v0`, `places_list_display_v0`, `places_search`, `step_card_display_v0`, `options_card_display_v0`, `quiz_display_v0`, `recipe_display_v0`, `translation_display_v0`, `link_preview_display_v0`, `message_compose_v1`, `weather_fetch`, `fetch_sports_data`, `ask_user_input_v0`, `visualize:read_me`, `visualize:show_widget`.

These emit structured payloads the chat UI (Decision 03b) renders inline. The runtime doesn't render anything itself.

### Research / discovery
`suggest_research`, `suggest_connectors`, `search_mcp_registry`, `search_plugins`, `suggest_plugin_install`, `search_skills`, `suggest_skills`, `recommend_claude_apps`.

### 3D / creative
- `learn_threejs`, `show_threejs_scene` — the chat UI hosts a Three.js sandbox tab.
- **Blender bridge** — the Blender MCP server (external addon) exposes `execute_blender_code`, `get_object_detail_summary`, `get_screenshot_of_window_as_image`, etc. Listed here for discovery; the tool bodies live in the addon, not in Ultima. Launcher config lists the Blender socket.

### Conversation control
- `end_conversation` — signals the runtime to close the current session cleanly.

## 10.3 — Permissions

Three-tier: **allow-always** (compiled-in-safe reads, viz widgets, memory reads), **prompt-once-per-session** (file writes under project root, web fetch, MCP tool calls), **prompt-every-time** (`bash_tool`, `memory_delete`, anything outside project root). Launcher owns the prompts; the runtime blocks until the launcher responds. No token authority for the runtime — every side-effecting call is user-approved.

## 10.4 — Tool call flow (server side)

1. Model emits a tool-call JSON block per Qwen2/Qwen3 chat template.
2. Server parses; runs the local built-in synchronously, or forwards MCP calls through the launcher.
3. Result is appended as a `tool` role message; decode resumes with the augmented context (reusing the KV prefix; only the new tool-result tokens are prefilled — Decision 09 §9.4).

## 10.5 — Files

```
include/ultima/tools/
    tool.hpp             (Tool trait, JSON-schema-typed input/output)
    builtin/*.hpp        (one header per group above)

src/tools/
    CMakeLists.txt
    tool.cpp
    builtin/*.cpp

include/ultima/mcp/
    mcp_client.hpp       (stdio JSON-RPC client)
src/mcp/
    mcp_client.cpp
```

## 10.6 — Deferred

- MCP WebSocket transport (v0.2).
- Streaming tool results (v0.2 — matters for long file reads).
- Tool result caching (v0.3).
- OAuth / API-key vaulting inside the runtime — launcher handles secrets in v0.1.

## 10.7 — Locked

Launcher hosts MCP subprocesses; runtime calls tools through a single seam. Built-in tool set matches the user brief. Three-tier permission model with user-facing prompts owned by the launcher.

---

# Decision 11 — Skills

## Scope

Skills are the packaged "here is how to do X in this project" instructions the model loads on demand. This decision picks the skill file format, the discovery roots, the load model (in-context vs plan-and-dispatch), and the interop story with the user's existing `C:/.skills/` protocol.

Ownership: `include/ultima/skills/*.hpp` + `src/skills/*.cpp`. Wires into the tool layer (Decision 10) via `search_skills` / `use_skill`.

## 11.1 — Skill file format

**Reuse the format already present in `examples/knowledge/patterns/*/SKILL.md`.** YAML frontmatter + Markdown body:

```markdown
---
name: cpp-coding
command: /cpp                # optional slash-command trigger
description: One-line summary the LLM sees during discovery.
---

# Body — plain Markdown, no size cap.
```

Fields:
- `name` (required, kebab-case, unique within its root).
- `description` (required, ≤ 200 chars — this is the *only* thing the LLM sees before deciding to load).
- `command` (optional, `/`-prefixed slash command).
- `tags` (optional list, for filter).
- `roots` (optional list of subdirs — a skill with `roots: [src/kernels]` only surfaces when the current work is under that path).

Sibling files under the skill's directory (e.g. `visual-identity.md` alongside `SKILL.md`) are addressable references the body can link to via relative paths.

## 11.2 — Discovery roots (priority order)

Every startup, Ultima scans the following roots (first-match-wins for name collisions):

1. **Project skills** — `<project>/.skills/` (per-repo overrides).
2. **Project knowledge** — `<project>/examples/knowledge/patterns/` (the shipped tree).
3. **Cross-agent skills** — `C:/.skills/skills/` (the user's `INDEX.md`-managed pool — 117 skills already migrated across sources).
4. **Claude Code skills** — `~/.claude/skills/` (VS Code / Claude Code-installed skills like `bevy-voxel-*`, `mmx-cli`).
5. **Built-in** — `<install>/skills/` (Ultima defaults).

Discovery result is cached to `<user_data>/skills.index.json` and rebuilt on file-hash mismatch at startup.

## 11.3 — Load model

**v0.1 — in-context load.** When the model calls `use_skill("<name>")` or the user types `/<command>`, the skill body is injected into the system message for the current turn. Simple, works with the KV prefix cache (Decision 09 — the injected system content is stable across the session so it hits the reuse path).

**v0.2 — plan-and-dispatch bridge** (later). The user's `C:/.skills/` protocol (`PROTOCOL.md`) is a rotation-based system: plan → dispatch phase-by-phase, each phase starts a new conversation with only its needed skills. v0.2 adds a `skills.dispatch` mode that reads the same `phases.md` / `active-skills.json` files so Ultima can plug into the existing pipeline. Not shipped in v0.1 — the in-context model covers the coding workflow, and dispatching multi-conversation phases needs the server (M5) first.

## 11.4 — Skill discovery for the LLM

Two tools (registered in Decision 10):

- `search_skills(query, limit=10)` → list of `{name, description, root, path}`. Fuzzy match against `name` + `description` + `tags`.
- `use_skill(name)` → loads the body into the current turn's system context; returns a confirmation with the skill's `description`.

The launcher UI (Decision 03) also surfaces a "Skills" panel that lists everything discovered, grouped by root, with a toggle to preload favorites into every session.

## 11.5 — Adjustments for `examples/knowledge/patterns/*`

The existing SKILL.md files reference `run_shell`, `powershell`, and other-agent-specific tool names. Ultima's `search_skills` returns them verbatim; the LLM adapts to the local tool names using the descriptions in the Decision 10 built-in list. **No forced rewrite of the shipped skill files** — treating them as reference material keeps the user's existing pipeline compatible.

For the M3+ ship, we add one **audit script** (`scripts/audit_skills.py`) that:
1. Walks every discovery root.
2. Parses SKILL.md frontmatter; flags missing `name`/`description`.
3. Grep-flags references to tools not in the built-in list, so the user can rewrite or leave as-is.
4. Emits a report to `<user_data>/skills.audit.md`.

Non-blocking — audit output is informational, not a load-time gate.

## 11.6 — Files

```
include/ultima/skills/
    skill.hpp            (SkillEntry + parse)
    skill_registry.hpp   (discovery + cache)

src/skills/
    CMakeLists.txt
    skill.cpp
    skill_registry.cpp
    frontmatter.cpp

scripts/
    audit_skills.py      (informational; run manually)
```

## 11.7 — Deferred

- Plan-and-dispatch mode (bridges into `C:/.skills/PROTOCOL.md`) — v0.2.
- Skill dependencies (`requires: [other-skill]`) — v0.2.
- Auto-loading skills based on file paths in view — v0.3.
- Remote skill fetch (Git URL / registry) — v0.3+.

## 11.8 — Locked

SKILL.md frontmatter compatible with the existing `examples/knowledge/patterns/` files. Five-root discovery with priority order. In-context load in v0.1; the user's `C:/.skills/` rotation protocol interoperates in v0.2. `search_skills` + `use_skill` are the LLM entry points; launcher UI is the human entry point. Audit script is opt-in and informational.

---

# Decision 12 — Memory subsystem (fast, reliable, coding-aware)

## Scope

Persistent per-user memory the model can read and write across turns and across sessions. Six tools declared in Decision 10 (`memory_read`, `memory_write`, `memory_str_replace`, `memory_append`, `memory_delete`, `memory_list`) plus a search side-channel the launcher UI exposes. Memory is the difference between "the model forgot we already fixed this last week" and "the model recalls the last three bugs in this file" — so it has to be fast (in the tool call hot path — every message can trigger reads), reliable (survives crashes and reboots), and scoped so a note about repo A doesn't leak into repo B.

Ultima's endgame is an AI coding editor. This decision ships the runtime memory layer that the future editor UI (Decision 15's editor-bridge) will read and write; the model, tools, and eventual IDE plugin all hit the same store.

Ownership: `include/ultima/memory/*.hpp` + `src/memory/*.cpp`. Wires into the tool layer.

## 12.1 — Storage engine: SQLite in WAL mode

**Choice: bundled SQLite (amalgamation build), WAL mode, `synchronous = NORMAL`.**

- **Fast:** in-process, no IPC; WAL keeps readers non-blocking while a write is in progress. Typical local key lookup is sub-millisecond.
- **Reliable:** ACID transactions, WAL journals survive power loss up to the last fsync, and `synchronous = NORMAL` (fsync at checkpoint, not per-commit) is the right trade for a single-user desktop — a full-durability `FULL` setting adds 5–10 ms per write with no meaningful reliability win at this scale.
- **Coding-editor friendly:** FTS5 virtual tables give full-text search on `value` for "find the memory that mentions this error" without a second search engine.
- **Zero external deps:** SQLite ships as a single .c file — dropped under `third_party/sqlite/` (Decision 02 §policy).

No LevelDB, no LMDB, no Redis. Every extra dep is a launcher headache and a Windows-install-time footgun.

## 12.2 — Schema

Single `memory.db` file under `<ULTIMA_DATA>/memory.db` (Windows: `%APPDATA%\Ultima\memory.db`).

```sql
CREATE TABLE memories (
    scope       TEXT NOT NULL,           -- "global" | "session:<id>" | "project:<hash>"
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    kind        TEXT NOT NULL DEFAULT 'text',  -- "text" | "code" | "json"
    updated_at  INTEGER NOT NULL,        -- unix seconds
    PRIMARY KEY (scope, key)
) WITHOUT ROWID;

CREATE INDEX idx_memories_updated ON memories(scope, updated_at DESC);

CREATE VIRTUAL TABLE memories_fts USING fts5(
    scope UNINDEXED,
    key   UNINDEXED,
    value,
    content='memories',
    content_rowid='rowid'
);

CREATE TABLE meta (
    k TEXT PRIMARY KEY, v TEXT NOT NULL
);
```

Scope prefixes:
- **`global`** — cross-project user preferences ("prefer tabs over spaces", "always suggest ripgrep over grep").
- **`session:<id>`** — one chat session; evicted when the session slot is released (Decision 09 §9.3).
- **`project:<hash>`** — per-repo memory keyed by a stable hash of the project root path. This is the coding-editor unlock: memories about repo A never surface in repo B.

`kind` is metadata for the eventual editor UI to render (code memories get syntax highlighting, JSON pretty-printing).

## 12.3 — In-process cache: `memory.cache`

A small LRU cache in front of SQLite catches the hot working set without a DB round-trip. Sized at 4 MB by default (~a few thousand small memories).

- **Read path:** cache → SQLite; cache-miss populates on read.
- **Write path:** write-through — SQLite first (so a crash never loses a written memory), then cache update.
- **Invalidation:** every mutation invalidates its `(scope, key)` cache entry.
- **Bounds:** LRU eviction on entry count or byte budget, whichever hits first.

## 12.4 — Search index: `memory.index`

The FTS5 virtual table `memories_fts` is the search index. Rebuilt from source via `INSERT INTO memories_fts(memories_fts) VALUES('rebuild');` if `memory.db` file hash mismatches its stored `meta.fts_source_hash` at startup — self-healing without user intervention.

Backing files SQLite creates alongside:
- `memory.db`             — main file
- `memory.db-wal`         — WAL journal (auto-checkpointed)
- `memory.db-shm`         — WAL shared-memory (transient)

The launcher's Data tab surfaces the location, size, and a "compact" button that runs `VACUUM`.

## 12.5 — Concurrency

Single writer, many readers — WAL's native model. Runtime instantiates one `MemoryStore` per process; every session shares it. Prepared statements are per-thread (WAL allows concurrent reads); the write mutex serializes mutations.

## 12.6 — Reliability posture

- `PRAGMA journal_mode = WAL`
- `PRAGMA synchronous = NORMAL`
- `PRAGMA foreign_keys = ON`
- `PRAGMA busy_timeout = 5000` (waits before erroring on a lock)
- WAL auto-checkpoint at 1000 pages
- Automatic snapshot to `memory.db.bak.<yyyymmdd-hhmm>` on graceful shutdown; keep last 5 snapshots (rolling).
- Corruption recovery: if `PRAGMA integrity_check` fails at startup, rename `memory.db` → `memory.db.corrupt.<ts>`, re-open the newest good snapshot, log loud.

## 12.7 — Tool implementations

Each tool is a thin wrapper over a prepared statement plus cache update. All accept an implicit `scope` derived from the calling session (default `session:<id>`; explicit `scope` override for global/project memories).

- `memory_read(key, scope?)` — `SELECT value, kind, updated_at FROM memories WHERE scope=? AND key=?`
- `memory_write(key, value, scope?, kind?)` — `INSERT OR REPLACE`
- `memory_str_replace(key, old, new, scope?)` — read + string replace + write; refuses ambiguous matches (>1 occurrence)
- `memory_append(key, text, scope?)` — read + append + write (single transaction)
- `memory_delete(key, scope?)` — `DELETE FROM memories WHERE scope=? AND key=?`
- `memory_list(prefix?, scope?, limit=50)` — `SELECT key, updated_at FROM memories WHERE scope=? AND key LIKE prefix||'%' ORDER BY updated_at DESC LIMIT ?`

Plus one non-model-facing helper for the launcher/editor:
- `memory_search(query, scope?, limit=20)` — FTS5 `MATCH` query with snippet extraction.

## 12.8 — Files

```
include/ultima/memory/
    memory_store.hpp     (MemoryStore + Scope helpers)

src/memory/
    CMakeLists.txt
    memory_store.cpp
    lru_cache.cpp        (bounded LRU used by MemoryStore)
    scope.cpp            (project_hash from path, session id issuance)

third_party/sqlite/
    sqlite3.c
    sqlite3.h
```

## 12.9 — Deferred

- **Vector search** (embeddings-based "find similar memories"). Interesting for a coding editor but requires an embed model (Decision 01b) already wired into the runtime — Decision 12 stays lexical/FTS5 for v0.1.
- **Cross-machine sync.** Local file only in v0.1; v0.3+ could push encrypted snapshots to a user-chosen sync target.
- **Memory expiration / TTL.** No auto-purge in v0.1; the launcher's "compact" button plus manual `memory_delete` are enough.
- **Structured schemas per memory kind.** `kind` is metadata only in v0.1; v0.2 could type-validate JSON kinds.
- **Encryption at rest.** Filesystem-level trust in v0.1; SQLite SEE / SQLCipher is a v0.3 concern if the machine leaves the desk.

## 12.10 — Locked

SQLite WAL + FTS5, three scopes (global / session / project), in-process LRU cache with write-through, rolling snapshots on shutdown, corruption self-recovery from newest good snapshot. Six model-facing tools + one launcher-facing search. `memory.db`, `memory.db-wal`, `memory.db-shm`, `memory.db.bak.*` under `%APPDATA%\Ultima\` — one place, one file family, plain SQLite tooling works on it if the runtime ever isn't around.

---

# Decision 14 — HTTP server + testpad webui

## Scope

The .cpp runtime needs a way for a human to actually *use* it — send a prompt, watch tokens stream back, tweak sampling knobs, and see whether the model is any good on the reference box. That's the immediate goal. It also needs to look like a normal OpenAI-compatible server so any existing coding tool (Continue, aider, Cline, VS Code extensions) can point at it right away without a shim. And it needs to be the surface the eventual coding editor plugs into (Decision 15's editor-bridge reuses these endpoints).

Ownership: `include/ultima/server/*.hpp` + `src/server/*.cpp` + `ui/testpad/*`. Depends on M4 (runtime + KV cache); this is M5.

## 14.1 — HTTP library: cpp-httplib (header-only)

**Choice: `cpp-httplib` (Yuji Hirose, MIT).** Header-only, blocking-thread-per-connection, SSE support out of the box, ~350 lines to hello-world.

Rejected alternatives:
- **Crow / Drogon** — much larger, async-heavy. Overkill for a single-user local server with 2 slots (Decision 09 §9.3).
- **Boost.Beast** — pulls in Boost. Non-starter given the "third_party is vendored single-header when possible" thread from Decision 02.
- **Rolling our own on Winsock/BSD sockets** — the "just to get streaming right" cost erases the shipping speed.

Under `third_party/httplib/httplib.h`. Wrapped in `include/ultima/server/http_server.hpp` so we can swap it later without touching handler code.

## 14.2 — Endpoints (v0.1)

Two families: **OpenAI-compatible** (drop-in for any existing client) and **Ultima-native** (diagnostics + dev tools).

### OpenAI-compatible

| Method | Path                            | Notes |
|---|---|---|
| POST   | `/v1/chat/completions`          | Chat with tools + streaming (`stream: true` → SSE, `text/event-stream`) |
| POST   | `/v1/completions`               | Legacy text completion |
| GET    | `/v1/models`                    | List loaded model(s) with the same JSON shape OpenAI ships |

Request/response shape matches OpenAI's April-2024 schema (chat + tool-calling). Fields Ultima doesn't implement (`logprobs`, `response_format` beyond `text`/`json_object`) are accepted-and-ignored with a `X-Ultima-Ignored: field1,field2` response header so clients don't silently mis-behave.

### Ultima-native

| Method | Path                     | Notes |
|---|---|---|
| GET    | `/api/health`            | `{"status":"ok","model":"<id>","kv_slots":{"total":2,"in_use":1}}` |
| GET    | `/api/slots`             | KV slot occupancy per Decision 09 §9.3 |
| POST   | `/api/tokenize`          | `{tokens:[...], count:N}` — dev/UI use |
| POST   | `/api/detokenize`        | `{text:"..."}` |
| GET    | `/api/skills`            | Enumerates discovered skills (Decision 11) |
| GET    | `/api/tools`             | Enumerates built-in + MCP tools (Decision 10) |
| GET    | `/`                      | Serves the testpad webui (§14.5) |
| GET    | `/static/*`              | Static assets under `<install>/ui/testpad/` |

## 14.3 — Streaming

SSE (`text/event-stream`) for `stream: true` chat requests. Chunk format matches OpenAI's `data: {...}\n\n` framing with a terminating `data: [DONE]\n\n`. Per-token flush; server sets `X-Accel-Buffering: no` to defeat any reverse-proxy buffering if the user later fronts Ultima with nginx.

Under the hood: the runtime's decode loop yields tokens through a bounded channel; the HTTP handler drains and writes each frame. A slow client (channel full) blocks the decoder for that slot — bounded RAM, no runaway queue.

## 14.4 — Bind + auth

- **Default bind:** `127.0.0.1:11434` (mnemonic to Ollama, so existing clients pointed at that port just work).
- **LAN mode:** launcher toggle opens the port on `0.0.0.0`; forces bearer-token auth on.
- **Auth token:** generated at first boot (`crypto_random 32 bytes → base64url`), stored in `%APPDATA%\Ultima\auth.token`. Launcher shows a "copy token" button and a QR code. Model-request handlers check `Authorization: Bearer <tok>`; missing/wrong → 401. Local (127.0.0.1) traffic bypasses auth **only** when LAN mode is off — makes the day-one testpad frictionless.
- **CORS:** allow `http://localhost:*` and `http://127.0.0.1:*` by default (so the testpad works). LAN mode narrows to an explicit allow-list configured in the launcher.

## 14.5 — Testpad webui (v0.1)

**Not the eventual coding editor.** A single-file HTML page under `ui/testpad/index.html` — vanilla JS, no build step, no framework, embedded CSS. Bundled into the binary as a resource so a fresh install has zero external assets to serve.

What it ships with:

- **Model select** dropdown (populated from `/v1/models`).
- **Chat pane** — messages, streaming display, tokens/s counter, first-token latency, KV slot indicator.
- **Sampler panel** — temperature, top-k, top-p, min-p, rep-penalty, seed. Defaults from Decision 08 §8.2. "Reset to coding defaults" button.
- **System prompt** field with the current chat template preview underneath.
- **Session controls** — new session (releases the slot), clear history, copy-as-cURL, copy-as-OpenAI-JSON.
- **Live diagnostics** — right-side collapsible pane showing `/api/health` + `/api/slots` polled every 2 s.

Deliberately not shipping in v0.1: file tree, code editor, terminal, diff view. Those belong to the coding editor build; the testpad is the seed that proves the runtime is worth building an editor around.

The testpad's JS talks to the same `/v1/chat/completions` endpoint an external client uses — no back-door API. This keeps the "if the testpad works, external clients work" invariant.

## 14.5.1 — Benchmark support (Ultima vs llama.cpp head-to-head)

The whole point of ship-day is proving Ultima runs the same GGUFs faster and/or better than llama.cpp on the reference box. Server + testpad ship with the fixtures and knobs to make that measurement fair:

- **Bench mode** in the testpad (top-right toggle): fixes seed, temperature, top-p, top-k, min-p, and prompt content to a per-model preset. Two clicks: pick model, run bench.
- **Prompt fixtures** under `ui/testpad/bench/*.md` — a short pack that stresses the paths that matter for coding: a 512-token "explain this function" prompt, a 4k-token "refactor this file" prompt, and a 16k-token "read this whole module and find the bug" prompt (Q4_K_M + Q8_0 KV → 16k should fit on the reference box). Same fixtures the user can point llama.cpp at.
- **Reported metrics** per run: prompt tokens, generated tokens, first-token latency (ms), decode throughput (tok/s), prompt-processing throughput (tok/s), peak RSS (RAM), KV bytes used, sampler settings, seed. Emitted both in the UI and to `events.log` (§14.8) as one JSON line per run so a script can plot ten runs.
- **`GET /api/bench/last`** returns the most recent run's metrics as JSON — external harness scripts don't need to scrape the UI.
- **Warmup**: bench mode always discards the first run (loads weights + primes the OS page cache), reports run #2 onward. Otherwise the first-run number lies about steady state.

Quality comparison is by-eye in v0.1 — the testpad's "duplicate to compare" button opens a second chat pane side-by-side (same prompt, same seed) so the user can eyeball Qwen2.5-Coder output from Ultima against a llama.cpp `curl` window running the same GGUF. An automated eval harness is a v0.2 concern (needs Decision 08's `logprobs` too).



One HTTP worker thread per KV slot from Decision 09 §9.3 (default 2). Requests wait in a small bounded queue when all workers are busy; queue full → HTTP 429 with `Retry-After`. Per-slot decoder runs sequentially — no cross-request batching in v0.1 (deferred in Decision 09 §9.9).

## 14.7 — Tool call flow (server side)

1. Model emits a tool-call block per Qwen2/Qwen3 chat template (Decision 07 §7.3).
2. Handler parses; runs built-in tools synchronously in-process; forwards MCP tool calls through the launcher IPC (Decision 10 §10.1).
3. Tool result appended as `tool` role message; decode resumes reusing the KV prefix (Decision 09 §9.4).
4. Streaming response inserts `tool_calls` chunks matching OpenAI's schema so clients that render tool activity (Continue, Cline) work unmodified.

## 14.8 — Compression + logging

- Response compression: gzip when the client sends `Accept-Encoding: gzip`. Small win locally, big win over LAN.
- Access log: newline-delimited JSON to `%APPDATA%\Ultima\logs\server.log`, rotated at 10 MB, keep last 5.
- Structured event log: separate `events.log` file with `{ts, session, event, tokens_in, tokens_out, ms_first_token, ms_total, sampler}` per completed request — this is what powers the launcher's Performance tab.

## 14.9 — Files

```
include/ultima/server/
    http_server.hpp
    openai_shim.hpp        (request/response schema translation)
    sse_stream.hpp
    testpad_bundle.hpp     (embedded HTML/JS/CSS)

src/server/
    CMakeLists.txt
    http_server.cpp
    handlers_openai.cpp
    handlers_ultima.cpp
    sse_stream.cpp
    testpad_bundle.cpp     (generated at build time from ui/testpad/*)

ui/testpad/
    index.html
    app.js
    styles.css

third_party/httplib/
    httplib.h
```

Build glue: a small CMake helper (`ultima_bundle_ui.cmake`) reads `ui/testpad/*` at configure time and emits a `.cpp` with a `std::string_view` per file. No runtime filesystem dependency for the testpad.

## 14.10 — Deferred

- **`logprobs`** in chat completions — matters for evals, add in v0.2.
- **Multi-model hot-swap** in one process — v0.1 loads one model at a time; the launcher restarts the runtime to switch.
- **WebSocket transport** (some clients prefer WS over SSE) — v0.2.
- **Rate-limiting / quotas** — single-user machine doesn't need it.
- **`response_format: json_schema`** — needs Decision 08's guided-decoding v0.3+ item.
- **Editor-bridge protocol** — that's Decision 15 (separate). The HTTP surface here is the substrate; the editor bridge adds edit-aware endpoints (`/api/edits/apply`, `/api/diagnostics`) on top.

## 14.11 — Locked

`cpp-httplib` + SSE, OpenAI-compatible schema (Ollama-port `11434` for zero-config existing clients), single-file vanilla-JS testpad bundled into the binary, one worker per KV slot, bearer-token auth (auto-generated) with local bypass off by default, gzip + structured logs. Ship-quality server the future coding editor plugs into via the same endpoints.

---

# Decision 13 — Adapter stub (LoRA hot-load)

## Scope

LoRA adapters let a small delta file re-specialize a base model without retraining the full weights. Ultima's v0.1 coding models (Qwen2.5-Coder, Qwen3-Coder) don't need adapters day one — they're already fine-tuned — but the runtime API surface has to reserve a real seam so v0.2 can add per-project style adapters ("write in this codebase's idioms") without breaking every caller. This decision is a **stub**: the interface ships in v0.1, but the runtime only supports **load-time adapter merging**, not hot-swap. Hot-swap is v0.2.

Ownership: `include/ultima/adapter/*.hpp` + `src/adapter/*.cpp`. Referenced by Decision 06 §6.8 (model layer) and Decision 01e (registry).

## 13.1 — What ships in v0.1

- **File format:** GGUF LoRA files (llama.cpp compatible). Same tooling ecosystem — `convert-lora-to-gguf.py` output loads straight in.
- **Location:** `<ULTIMA_DATA>/adapters/<name>.gguf` (Windows: `%APPDATA%\Ultima\adapters\<name>.gguf`).
- **Registry hook (Decision 01e):** each entry in the model registry may declare `adapter: "<name>"`. When present, the loader merges the adapter into the base weights **at load time** — result is a base+adapter combined tensor set in RAM. No adapter mixing at inference time.
- **Interface:**

  ```cpp
  namespace ultima::adapter {

  struct AdapterSpec {
      std::string name;
      std::filesystem::path path;
      float alpha = 1.0f;   // scale factor from the GGUF; overridable per-model
  };

  class Adapter {
  public:
      // Load a single adapter into RAM. Small (typically 10–100 MB).
      static std::unique_ptr<Adapter> load(const AdapterSpec&);

      // Fold into the base tensor set at load time. Modifies the target in place.
      void merge_into(model::TensorDirectory& base) const;

      const AdapterSpec& spec() const noexcept;
  };

  } // namespace ultima::adapter
  ```

- **No runtime hot-swap.** Switching adapters means restarting the model load (launcher handles the reload). No "swap adapter mid-session" API — the interface deliberately omits it in v0.1 so we don't paint ourselves into a corner on how the swap interacts with active KV slots (Decision 09).

## 13.2 — Why stub, not full

- Qwen2.5-Coder-7B-Instruct and Qwen3-Coder cover the shipping coding workload without adapters (Decision 01, 01c).
- The engineering cost of runtime hot-swap is real: it changes weight-tensor lifetimes, complicates the `IModel` (Decision 06) prefill/decode contract, and interacts with the KV cache (an adapter change *should* invalidate the cache the same way a model swap does — that's Decision 09 §9.4's model_id invalidation trigger, but per-adapter invalidation is a new axis).
- v0.1 doesn't need any of that. v0.2 opens it with a cleaner "swap active adapter" API once we have real user demand.

## 13.3 — Adapter validity check

At load, verify:
- Adapter's `general.architecture` matches the base model's (`qwen2` / `qwen3`).
- Every adapter tensor's name maps to a real base tensor (via Decision 06 §6.5's `TensorSlot` table).
- Rank + shape sanity on each `(A, B)` pair.

Failure → loud error at load, base model still loads without the adapter (opt-in fail-open configurable in launcher, default fail-closed).

## 13.4 — Files

```
include/ultima/adapter/
    adapter.hpp

src/adapter/
    CMakeLists.txt
    adapter.cpp
    gguf_lora.cpp        (parse GGUF LoRA tensor pairs)
```

## 13.5 — Deferred (v0.2+)

- Runtime hot-swap (switch adapter without model reload).
- Multi-adapter stacking (`[adapter_a, adapter_b]` merged at load with per-adapter alphas).
- Adapter training / fine-tuning inside Ultima (that's a Decision 20+ concern if we ever go there).
- QLoRA-style quantized-base + fp16-adapter arithmetic — for v0.1 both live at the base's dtype after merge.

## 13.6 — Locked

v0.1 ships load-time GGUF LoRA merging behind `Adapter::load` + `merge_into`, gated by a per-model-registry `adapter:` field (Decision 01e). No hot-swap. Interface deliberately narrow so v0.2 can add hot-swap without breaking callers.

---

# Decision 15 — Editor-bridge reservation

## Scope

Ultima's endgame is an AI coding editor. Decision 14 ships the runtime's HTTP surface + a testpad; that surface is what every future editor plugin — VS Code, JetBrains, Zed, a bespoke Tauri front-end — talks to. This decision **reserves the URL space and JSON schemas** for the edit-aware endpoints so v0.1 external editor devs (including a future us) can code against a stable target, without Ultima having to implement them yet.

Nothing here ships as working code in v0.1. Endpoints exist in the routing table returning **HTTP 501 Not Implemented** with a body that names the reserved endpoint and points at this decision. That way `curl` against them documents the intent, and any editor plugin that starts development early against the reserved schema won't need a URL migration when v0.2 turns the lights on.

Ownership: `include/ultima/server/editor_bridge.hpp` (schemas + 501 stubs). Layered on Decision 14's HTTP server.

## 15.1 — Reserved endpoints

All under `/api/editor/*`. All return HTTP 501 in v0.1 with `{"status":"reserved","spec":"Decision 15","endpoint":"<path>"}`.

| Method | Path                              | Purpose (v0.2 implementation) |
|---|---|---|
| POST   | `/api/editor/workspace/open`      | Associate a project root with the current session; server computes `project:<hash>` scope (Decision 12 §12.2) and returns it |
| GET    | `/api/editor/workspace/state`     | Currently open files, cursor position, selection ranges (editor pushes on change) |
| POST   | `/api/editor/edits/preview`       | Server returns a unified diff for a proposed change without touching disk |
| POST   | `/api/editor/edits/apply`         | Server writes the diff to disk (respecting Decision 10 §10.3 permissions), returns applied hunks |
| POST   | `/api/editor/diagnostics`         | Editor pushes compiler / LSP diagnostics; server injects them as `tool` messages so the model can react |
| POST   | `/api/editor/completions/inline`  | Cursor-context completions ("finish this line") — separate from `/v1/chat/completions` because the request shape is completion-style (prefix + suffix), not chat |
| WS     | `/api/editor/events`              | Bidirectional stream: file change, cursor move, buffer edit; server broadcasts model events (tool call, streaming token) |

Two knobs are important upfront and belong to this decision, not the implementation:

- **Scope pinning:** every editor-bridge request carries `X-Ultima-Workspace: <project_hash>` (issued by `/api/editor/workspace/open`). Server pins the KV slot to the project scope, so memory reads (Decision 12) auto-scope, KV prefix reuse (Decision 09 §9.4) doesn't cross-contaminate, and per-project skills (Decision 11 §11.2 root #1) surface first.
- **Trust boundary:** editor-bridge endpoints are subject to Decision 14 §14.4's auth, plus an editor-specific origin allow-list (`vscode-webview://*`, `vscode-file://*`, `jetbrains://*`, `tauri://*`) so a random browser tab can't spoof edits.

## 15.2 — v0.1 stub deliverable

- `include/ultima/server/editor_bridge.hpp` declares `register_editor_bridge_stubs(HttpServer&)`.
- `src/server/editor_bridge_stubs.cpp` implements it: registers every path above with the 501 handler.
- `ui/testpad/` gains a **"Reserved endpoints"** collapsible panel listing them for discoverability.
- A short spec document `docs/editor-bridge-spec.md` (written when v0.2 starts) will formalize request/response schemas. v0.1 ships only the URL reservation + a one-line description per endpoint.

## 15.3 — Why reserve, not implement

- v0.1's job is to *prove the .cpp runtime is worth building an editor on*. That's Decision 14's testpad and the llama.cpp head-to-head bench.
- Every editor endpoint above needs a design pass wide enough to be its own decision (`edits/apply` has a whole permission + undo story; `completions/inline` has its own request shape and streaming rules; `diagnostics` needs a schema translator per LSP flavor).
- Freezing the URL space now costs nothing and unblocks parallel editor-plugin work by anyone (including future us) the moment v0.1 ships.

## 15.4 — Files

```
include/ultima/server/
    editor_bridge.hpp     (register_editor_bridge_stubs)

src/server/
    editor_bridge_stubs.cpp  (HTTP 501 handlers per §15.1)
```

## 15.5 — Deferred to v0.2 (the actual bridge)

Every endpoint in §15.1. Plus:

- Request/response JSON schemas (formalized in `docs/editor-bridge-spec.md` when work starts).
- Undo/redo journal for `edits/apply` (probably a small SQLite table alongside `memory.db`).
- LSP diagnostic translators (start with tsserver + clangd + rust-analyzer since those cover the shipping code footprint).
- Editor-side reference implementation — a thin VS Code extension in `editor-plugins/vscode/` demoing the full loop.

## 15.6 — Locked

Seven `/api/editor/*` endpoints reserved. Every one returns HTTP 501 with a discoverable body in v0.1. `X-Ultima-Workspace` header + editor-origin allow-list pinned as invariants so v0.2 doesn't have to redo the trust model. First-class implementation is v0.2.

---

## All decisions written

The design phase for v0.1 is complete. Decisions 01–15 (skipping 09-was-numbered-earlier is not a real gap — see below) cover every subsystem the runtime needs. The implementation milestones (M2 GGUF loader → M3 tokenizer → M4 transformer runtime + KV cache + sampler → M5 server + testpad) all have locked designs to target.

Written and locked: 01, 01b, 01c, 01d, 01e, 01f, 02, 03, 03b, 04, 05, 06, 07, 08, 09, 10, 11, 12, 13, 14, 15.

No unwritten decisions remain in the "Future decisions" slot; new decisions from here on out are numbered from 16.
