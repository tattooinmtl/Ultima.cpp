# Ultima.cpp — Model Support

What Ultima can actually load and run, and what each unsupported architecture
still needs. `ultima archs` prints the same table from the registry in
`src/model/arch_registry.cpp`; `ultima scan <dir>` applies it to a folder of
GGUFs and reports each file individually.

## Architecture support

| Architecture | Status  | Notes |
|--------------|---------|-------|
| `qwen2`      | runnable | Qwen2 / Qwen2.5 dense, separate QKV with biases |
| `qwen3`      | runnable | Qwen3 dense, Q/K RMSNorm, no attention biases |
| `qwen3moe`   | runnable | Qwen3 MoE, top-k routed experts (Qwen3-Coder-30B-A3B) |
| `granite`    | runnable | IBM Granite dense, embedding/residual/logit scales, Norm-style RoPE |
| `phi3`       | runnable | Fused QKV and fused gate/up, LongRoPE frequency factors |
| `spark2_5`   | runnable | Fused QKV, per-head sigmoid attention gate, GEGLU FFN, 3-of-4 sliding-window layers with their own RoPE dim and base |
| `gemma4`     | not yet  | per-layer KV head counts, alternating sliding-window attention, dual RoPE bases, post-attention/post-FFN norms, final logit softcapping |
| `qwen35`     | not yet  | gated delta-net linear attention (`ssm_*`, 3 of every 4 blocks) + a recurrent state cache. **Reference exists**: supported upstream in llama.cpp as Qwen3-Next/Qwen3.5 |
| `qwen35moe`  | not yet  | the `qwen35` SSM hybrid plus MoE with shared experts |
| `deepseek2`  | not yet  | multi-head latent attention (compressed KV) and MoE with grouped routing, shared experts, expert bias |
| `dspark`     | never*   | diffusion LM — block denoising is not autoregressive, so it does not fit prefill/decode at all |
| `dflash-draft` | never* | speculative-decoding draft head: no tokenizer, no `token_embd`, no output head. Only usable attached to its target model |
| `llama`      | not yet  | Norm-style RoPE validation and llama-specific tensor aliases |
| `bert`       | not yet  | encoder-only; needs a pooling path rather than causal prefill/decode |

\* "never" means not under the current `IModel` prefill/decode seam, not that
the file is broken. Both would need a different execution model.

## Quantization support

Unless noted, a supported dtype works in matvec, row reads and 1-D dequant.

| dtype  | Block | Bytes | Notes |
|--------|-------|-------|-------|
| F32    | 1     | 4     | |
| F16    | 1     | 2     | dequant only — no matvec kernel |
| BF16   | 1     | 2     | dequant only — no matvec kernel; top 16 bits of an f32 |
| Q4_0   | 32    | 18    | **not implemented** |
| Q4_1   | 32    | 20    | affine 4-bit |
| Q5_0   | 32    | 22    | |
| Q5_1   | 32    | 24    | **not implemented** |
| Q8_0   | 32    | 34    | |
| Q2_K   | 256   | 84    | **not implemented** |
| Q3_K   | 256   | 110   | **not implemented** |
| Q4_K   | 256   | 144   | |
| Q5_K   | 256   | 176   | |
| Q6_K   | 256   | 210   | |
| Q8_K   | 256   | 292   | **not implemented** |
| PQ2_0  | g128  | ~2.13 bits | **not implemented** — ternary {−1,0,+1}, PrismML fork only |
| PTQ1_0 | g128  | ~1.75 bits | **not implemented** — dense ternary trits, PrismML fork only |

IQ-series (`IQ1_*`…`IQ4_*`) and the ternary `TQ*` formats are not implemented.

## Tokenizers

| `tokenizer.ggml.model` | Status | Used by |
|------------------------|--------|---------|
| `gpt2` | supported | Qwen, Granite, Spark-X2.5, most modern releases |
| `llama` / `spm` | supported | Phi-3, Llama, Mistral — SentencePiece with `<0xNN>` byte fallback |
| `gemma4` | not implemented | the Gemma models |
| `none` | n/a | models with no vocabulary |

## Architectures without a public reference

`gemma4`, `dspark` and `dflash-draft` do not correspond to architectures
Ultima has a reference implementation for. (`qwen35`/`qwen35moe` were listed
here until 2026-09-18; they are in fact supported upstream in llama.cpp as
Qwen3-Next/Qwen3.5, so they are knowable work.) Their GGUF metadata
uses llama.cpp-style key names, but the semantics behind those keys cannot be
settled by inspection alone.

`spark2_5` was in this list and is the cautionary tale. Eight structural
combinations were ablated and none worked; what was actually wrong was a single
bit of knowledge — the feed-forward uses GEGLU, not SwiGLU — which is one line
in llama.cpp PR #27868 and is not inferable from the GGUF. Do not try to
brute-force an unknown architecture; find its reference first.

`deepseek2` is the exception among the remaining work: it is standard
DeepSeek-V2 multi-head latent attention, which is well specified.

## Pretokenizers

`tokenizer.ggml.pre` selects how text is cut before BPE runs. Getting it wrong
silently changes the token ids a prompt produces.

| `pre` | Family | Notes |
|-------|--------|-------|
| `qwen2`, `spark2_5` | Qwen2 | one non-letter may prefix a letter run (`(n` is one token); digits one at a time |
| `llama3`, `llama-v3`, `llama-bpe`, `dbrx`, `smaug` | Llama3 | as above, but digits in runs of up to 3 |
| anything else | GPT-2 | only a space may prefix a letter run; digit runs unbounded |

Verified token-for-token against llama.cpp `llama-tokenize` for `qwen2`,
`dbrx` (Granite) and `spark2_5`.

## Known limitations

- **Phi-3 LongRoPE**: the frequency factors are resolved once at load time and
  take the `rope_factors_short` set, which is correct up to
  `rope.scaling.original_context_length` (4096) tokens. Beyond that llama.cpp
  switches to the long set; Ultima does not yet.
- **MoE shared experts** are rejected rather than silently ignored. A model
  declaring `expert_shared_feed_forward_length > 0` fails to load with that
  reason.
- **Multi-file (split) GGUFs** are rejected at load.
- KV cache is F32 only.

## Notes

- Models are never committed to the repository. Point any slot at a local GGUF
  via the launcher's Models tab or `ultima run <path>`.
- `ultima scan <dir>` needs only the GGUF header and tensor directory, so it is
  fast even over a folder of 20 GB files.
