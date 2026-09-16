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
| `gemma4`     | not yet  | per-layer KV head counts, alternating sliding-window attention, dual RoPE bases, post-attention/post-FFN norms, final logit softcapping |
| `spark2_5`   | not yet  | gated attention (`attn_gate`), sliding-window pattern, dual RoPE bases |
| `qwen35`     | not yet  | hybrid SSM/Mamba layers (`ssm_*`, 3 of every 4 blocks), MRoPE sections, SSM state cache |
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

IQ-series (`IQ1_*`…`IQ4_*`) and the ternary `TQ*` formats are not implemented.

## Tokenizers

| `tokenizer.ggml.model` | Status | Used by |
|------------------------|--------|---------|
| `gpt2` | supported | Qwen, Granite, most modern releases |
| `llama` / `spm` | supported | Phi-3, Llama, Mistral — SentencePiece with `<0xNN>` byte fallback |
| `gemma4` | not implemented | the Gemma models |
| `none` | n/a | models with no vocabulary |

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
