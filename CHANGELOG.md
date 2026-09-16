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

