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

