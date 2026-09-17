# Ultima project changelog

Append-only work log. One line per completed task. Format:

```
YYYY-MM-DD HH:MM  <task_id>  <short description>  <files>
```

`task_id` is a short mnemonic (e.g. `landing-page`, `fix-q5_0`, `add-carousel`) — reuse across turns that continue the same task.

The model is expected to append here after every completed task, in the same request that finishes the work. This file is the project's durable memory of what has been tried, what worked, and what didn't.

---

## Entries

<!-- newest at the top -->

2026-09-17 coder-ide-p4  Phase 4 done - phased planning. Also found the cause of the silent empty turns from Phase 2.
    NEW src/server/session.{hpp,cpp} - the workflow state lives in the workspace, not in memory:
      .ultima/plan.md (phase list, hand-editable), .ultima/todos.json (subtasks), .ultima/CHANGELOG.md
      (append-only). The plan is a file rather than a new tool because six tools is already as many
      as a small model reliably picks from.
      * parse_plan is deliberately tolerant - the user is invited to edit the file. Accepts "- [ ]",
        "* [X]", "1. [ ]" and, from a live run, "-[x]Phase4 :" with no spaces. Phases renumber by
        position so reordering by hand needs no renumbering.
      * render_plan_for_prompt names ONLY the active phase. Listing the whole plan invites a small
        model to skip ahead and do three phases badly in one turn.
    /api/agent/chat gained mode: "plan" | "work" | "auto". Planning mode injects the plan-first
      instructions, refuses any tool call that is not writing the plan, and stops the moment the
      plan lands (stop:"plan_written") rather than burning a second round on prose.
    NEW GET /api/plan and POST /api/plan/phase. Checking a phase off appends to the project changelog.
    ui/testpad - Plan panel in the Coder tab: phases with nested subtasks, active phase highlighted,
      click a box to toggle it, progress count. A "Plan" button beside "Run".
    FOUND THE PHASE-2 MYSTERY: rounds that produced neither prose nor a tool call were silently
      doing nothing. They now emit a `warning` event with the raw head of the generation, which
      immediately showed Qwen2.5-Coder-7B opening a call with a correct "name" and then drifting
      into typographic quotes for every key after it. Unparseable JSON, whole call dropped.
      parse_tool_calls now retries with the quotes normalized - only after a strict parse has
      already failed, so a file whose content legitimately contains curly quotes is not rewritten.
    A plan written this turn is forced all-unchecked: the model checked off Phase 4 as it wrote the
      list, which would have made the runtime skip that phase entirely.
    Verified live: "Build a landing page for a coffee shop" in plan mode produced four phases and
      NO project code - the workspace held only .ultima/plan.md and the pre-existing README. The
      panel renders it, toggling phase 1 advanced the active phase to 2 and logged the completion.
      210 unit tests pass.


2026-09-17 coder-ide-p3  Phase 3 done - the Coder tab is a real editor. Verified in a browser.
    ui/testpad/index.html - the Coder tab was placeholder markup; it is now an IDE over the
      /api/fs/* endpoints. Still one self-contained file with no external references.
      * File tree from /api/fs/tree?recursive=1, with new / rename / delete.
      * Editor tabs: one per open file, dirty dot, close button, middle-click close, and a
        confirm before discarding unsaved changes.
      * Editor: a textarea with transparent text laid over a syntax-highlighted <pre>, plus a
        line-number gutter that slides with the scroll. Tab inserts two spaces; Ctrl+S saves.
        The two layers must share font metrics and padding exactly or the caret drifts off the
        glyphs - both CSS rules say so.
      * Highlighter: one combined regex per language (js/ts, python, css, html, json, markdown),
        precedence ordered so a keyword inside a string is not recolored. A backtick cannot appear
        inside a String.raw template, so those patterns use \x60.
      * Status bar: path, language, line and char counts, saved/unsaved.
      * Terminal panel + an agent box that drives POST /api/agent/chat from Phase 2 and lands its
        `file` events as tabs. A file the model rewrites reloads in place, unless that tab has
        unsaved edits - clobbering the user's typing would be worse than a stale tab. New tabs do
        not steal focus within 5s of the user touching the editor.
    NEW scripts/test_testpad_highlight.mjs - extracts the Coder module from the HTML and checks
      every language round-trips losslessly, including an unterminated string, a trailing
      backslash, and a 4000-line file (11ms). Not in CTest: the C++ build must not depend on node.
    Verified in a browser end to end: opened src/app.js and index.html from the tree, typed into
      index.html, watched the dirty dot and char count update, Ctrl+S, confirmed the bytes on disk;
      then asked the agent box for a file and watched notes.md appear in the tree and open as a
      focused tab showing highlighted markdown.
    Two layout bugs found by measuring rather than squinting: the empty state rendered inside the
      52px gutter column (needs grid-column:1/-1), and .coder-main never stretched because it is a
      flex child without flex:1, so the grid sized to content and left 244px of dead space.
    The 1.5B emitted write_projectFile with "./notes.md" - the Phase 2 verb fallback and path
      normalization both caught it, in a live run, again.


2026-09-16 coder-ide-p2  Phase 2 done - file tool layer + agent loop. Three robustness fixes came from live runs.
    NEW src/server/tools.{hpp,cpp} - six tools (list_files, read_file, write_project_file, edit_file,
      rename_file, delete_file), their JSON schemas, the Qwen <tool_call> system prompt, a lenient
      parser and the dispatcher.
    NEW POST /api/agent/chat - the turn is a loop: generate, execute any tool calls, feed the results
      back as `tool` messages, generate again. Ends on a reply with no calls, or the round budget.
      SSE events: delta | tool_call | tool_result | file | done | error.
    Parser accepts the three shapes small models drift between: the trained <tool_call> wrapper, a
      fenced json block, and a bare object (only as the whole reply, so prose quoting JSON is not
      misread). Brace matching is string-aware so JS braces inside `content` do not end the object.
      A call truncated by the token cap still parses.
    visible_prose_prefix() keeps <tool_call> markup out of the streamed deltas, including holding
      back a partial opening tag.
    Driven by live runs against Qwen2.5-Coder-7B, not guessed:
      - it called "write_project-file" (hyphen)     -> name canonicalization + aliases
      - it passed "/hello.js" (leading slash)       -> path normalization above resolve()
      - it invented "read_project"                  -> leading-verb fallback
      - bare "missing argument X" made it give up   -> errors now name the retry
    Verified: the 7B creates a file on disk through the endpoint, reproducibly; the file event fires
      with a tab label. 190 tests pass.
    Known limitation, not papered over: on a two-step ask the 7B often stops after one failed call
      instead of recovering. Machinery is fine; that is model capability, and Phase 4 is the answer.
    Process note: tested a stale binary for one cycle - a running server held ultima.exe open so the
      relink was skipped silently while the .lib updated. Stop the server before rebuilding.


2026-09-16 coder-ide-p1  Decision 18 written; Phase 1 done - workspace file API. Also fixed pre-existing UB in the server.
    roadmap01.md Decision 18 - Coder IDE: workspace file API, six model tools, in-tree tabbed
      editor, phased planning. Studied C:/GameForgerAI (gameSimController.js + GameSimCreator.jsx)
      as the reference; its one-tab-per-file UI carries over, its whole-file regeneration does not,
      because it cannot express rename or partial edit. No Monaco: the testpad is bundled into the
      binary with zero external references.
    plan.md - five phases, one phase = one task, subtasks go to .ultima/todos.json.
    NEW src/server/workspace.{hpp,cpp} - the whole feature funnels through one resolve(), which
      refuses .., absolute paths, drive letters, UNC, control characters and Windows device names,
      then re-checks containment after symlink resolution. Never clamps a bad path into a good one.
      edit() is exact-string and all-or-nothing: a failing edit list leaves the file untouched.
    NEW /api/fs/{tree,file,rename,mkdir} over GET/PUT/PATCH/DELETE, plus a --workspace flag on serve.
    FIX pre-existing UB: check_auth was a constructor-local lambda captured by reference into every
      route handler, so every request read a destroyed object. Latent until the new fs helpers
      changed the stack layout, then it segfaulted on the first request. All request helpers are
      free functions now.
    tests/unit/test_workspace.cpp - 22 cases with the escape vectors pinned. 160 tests pass.
    Verified live: all seven endpoints work; ../, %2F-encoded, backslash, drive-letter, UNC,
      /etc/passwd and NUL variants all refused 400 with a named reason; nothing written outside root.


2026-09-16 coder-ide-p1  Decision 18 written; Phase 1 done — workspace file API. Also fixed pre-existing UB in the server.
    roadmap01.md Decision 18 — Coder IDE: workspace file API, six model tools, in-tree tabbed editor,
      phased planning. Studied C:/GameForgerAI (gameSimController.js + GameSimCreator.jsx) as the
      reference; its one-tab-per-file UI carries over, its whole-file regeneration does not, because
      it cannot express rename or partial edit. No Monaco: testpad is bundled into the binary with
      zero external refs.
    plan.md — five phases, one phase = one task.
    NEW src/server/workspace.{hpp,cpp} — the whole feature funnels through one resolve(), which
      refuses .., absolute paths, drive letters, UNC, control chars and Windows device names, then
      re-checks containment after symlink resolution. Never clamps a bad path into a good one.
      edit() is exact-string and all-or-nothing: a failing edit list leaves the file untouched.
    NEW /api/fs/{tree,file,rename,mkdir} — GET/PUT/PATCH/DELETE. --workspace flag on .
    FIX pre-existing UB: check_auth was a constructor-local lambda captured by reference into every
      route handler, so every request read a destroyed object. Latent until the new fs helpers
      changed the stack layout, then it segfaulted on the first request. All request helpers are
      free functions now.
    tests/unit/test_workspace.cpp — 22 cases, escape vectors pinned. 160 tests pass.


2026-09-16 spark2_5  spark2_5 WORKING. Root cause was GEGLU vs SwiGLU. Also fixed a pretokenizer bug affecting every model.
    User supplied the reference: llama.cpp b10988 binaries at C:/Llama/llama-b10988, and upstream PR
    ggml-org/llama.cpp#27868 (merged 2026-09-06), fetched with `gh pr diff`.
    Root cause: the feed-forward is LLM_FFN_GELU, not SILU. Everything else the 8-way ablation had
      questioned was already right - NeoX rope, contiguous [Q|K|V], per-head sigmoid gate on the
      post-attn-norm input applied before wo, sliding_window_pattern true = sliding.
      Added kernels::geglu_f32 (tanh approximation) + FfnAct in ArchSpec/ArchParams.
    Separately found and fixed a real pretokenizer bug affecting EVERY gpt2-vocab model: the
      hand-rolled splitter grouped by character category, so "(n" always split. The Llama3/Qwen2
      family allows one non-letter to prefix a letter run, and caps digit runs (1 for qwen2/spark2_5,
      3 for llama3/dbrx). Rewrote pretokenize() with the three families selected by tokenizer.ggml.pre,
      plus real UTF-8 decoding. qwen2 and granite now match llama-tokenize token-for-token; they did
      not before.
    Verified: 8 of 9 prompts byte-identical to the reference on Spark-1.7B; the one divergence is a
      near-tied logit, explained by ggml evaluating GELU through an fp16 lookup table (less precise
      than our direct tanh). Spark-4B also byte-identical, including a prompt where BOTH degenerate
      to "nnnn" - that is the model, not us. All 7 previously-working models unchanged.
    Now 9 of 21 files in C:/models runnable. 137 tests pass.


2026-09-16 swa-infra  Sliding-window attention + gated attention + per-layer RoPE. spark2_5 attempted, NOT working.
    Infrastructure (all verified inert on the 7 working models, no regression):
      gqa_attention gained pos_begin, so SWA layers mask older keys instead of needing a second
        ring-buffered cache geometry.
      ArchParams gained sliding_window + layer_is_swa + attn_gate; RopeConfig gained freq_base_swa
        and rope_dim_swa, selected per layer in block_forward.
      loaded_model gained read_u8_array for per-layer bool flag vectors.
      ArchSpec gained attn_gate; tensor slots resolve attn_gate.weight [n_heads, hidden].
      CLI: `ultima run --show-tokens`.
    spark2_5: NOT runnable. Every structural piece is implemented and all shapes/metadata resolve,
      but it decodes to degenerate token echo. Ablated 8 combinations (SWA on/off/inverted x gate
      on/off x NeoX/Norm rope x contiguous/interleaved QKV) — none coherent. Tokenizer ruled out
      by comparing against qwen2 on the same prompt: both split sensibly. Left registered as
      not-runnable with that reason rather than shipped as working.
    Finding: spark2_5 / gemma4 / qwen35 / dspark / dflash-draft have no reference implementation
      available here, so their unconstrained semantics are guesswork. deepseek2 (GLM-4.7-Flash) is
      the one remaining arch that is well specified — standard DeepSeek-V2 MLA.


2026-09-16 fix-q5_0  Q5_0 matvec: AVX2 + threaded. 0.5B model went 2.3 -> 51.5 tok/s (22x)
    src/kernels/matvec_q5_0_f32.cpp  — fused AVX2 unpack; the offset-binary -16 is folded into the
        bit pattern (OR 0xF0 where the fifth bit is clear) so the byte reads as a signed int8 directly,
        no separate subtract and no f32 scratch. Existing scalar loop kept as matvec_q5_0_f32_scalar.
    src/kernels/matvec_threaded.cpp  — matvec_q5_0_f32_threaded.
    src/model/weight_dispatch.cpp    — Q5_0 no longer falls through to the serial scalar path with
        `(void)pool`; that was the actual defect. It ran single-threaded scalar while every other
        quant used the pool + FMA, which is why the 0.5B was 8x slower than the 1.5B.
    tests/unit/test_q5_0.cpp         — new; there were no Q5_0 tests at all, which is how the
        placeholder survived. Reference encoder written from the format spec, AVX2 vs scalar oracle.
    Verified: output byte-identical before/after on qwen2.5-coder-0.5b. 1.5B/7B/unsloth/phi3 unchanged.


2026-09-16 multi-arch  Multi-architecture model support: Q5_K/Q4_1/BF16 kernels, arch registry, granite + phi3 + qwen3moe, SentencePiece tokenizer
    Kernels:   include/ultima/kernels/{dequant_q5k,dequant_q4_1}.hpp, src/kernels/{dequant_q5k,dequant_q4_1,matvec_q5k_f32,matvec_q4_1_f32}.cpp
               rope.{hpp,cpp} gained rope_f32_ext (Norm vs NeoX pairing, LongRoPE freq_factors, attn_factor)
    Registry:  include/ultima/model/arch_registry.hpp, src/model/arch_registry.cpp — 14 architectures, each either
               runnable (with its structural params) or carrying the specific engine work it still needs.
               Replaces the hardcoded qwen2/qwen3 check in load_model.
    Model:     imodel.hpp gained ModelDims::{q_dim,kv_dim} (Qwen3-MoE has q_dim != hidden) and ArchParams.
               qwen_config.cpp generalized to parse_arch_config; qwen_tensor_slots resolves fused QKV,
               fused gate/up, and 3D MoE expert stacks; qwen_block gained fused-QKV split, fused/MoE FFN,
               per-arch RoPE and Granite's embedding/residual/logit/attention scales.
               Hoisted the per-layer residual heap allocation into BlockScratch.
    Tokenizer: SentencePiece ("llama"/"spm") support alongside GPT-2 BPE — score-ranked merges over
               U+2581-marked text with <0xNN> byte fallback. Needed for phi3. LoadedModel gained
               file_data() plus read_f32_array/read_i32_array so numeric metadata arrays resolve.
    CLI:       new `ultima run` (one-shot completion), `ultima archs`, `ultima scan <dir>`.
    Verified:  generation on all 7 runnable models in C:/models. Granite's Norm RoPE confirmed by
               A/B: NeoX degenerates into newline repetition, Norm writes a correct class.
    Docs:      MODELS.md rewritten as the real support matrix (arch, quant, tokenizer, limitations).

