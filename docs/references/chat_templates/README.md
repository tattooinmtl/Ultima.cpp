# Chat template references

Authoritative Jinja chat templates for the model families Ultima supports (or plans to). These are **reference documents only** — Ultima does not embed a Jinja engine in v0.1. We hand-implement equivalent renderers in C++ against these templates as the source of truth.

## Contents

| File | Model family | Arch string | Notes |
|---|---|---|---|
| `ornith-1.5-9b.jinja` | Ornith 1.5 9B | `qwen3` | Native `qwen3_xml` tool format, `<think>` reasoning blocks, `enable_thinking` toggle, multimodal blocks (deferred). |

## What we extract per template

For each template, our M3 chat renderer needs to reproduce:

1. **Turn markers** — `<|im_start|>role\n ... <|im_end|>\n`
2. **System message policy** — merge count, separator, order
3. **Tool schema injection** — where tools go in the system prompt, and the exact instruction text
4. **Tool-call format** — JSON (`qwen2_json`) vs. XML (`qwen3_xml`)
5. **Reasoning-token policy** — whether `<think>...</think>` is emitted, whether it can be suppressed
6. **Generation prompt** — the exact suffix appended when `add_generation_prompt=true`
7. **Multi-tool-call sequencing** — whether tool_call blocks are back-to-back or separated

## When we add a new template

1. Save the raw Jinja under this directory as `<model-family>.jinja`
2. Add a row to the table above
3. Implement or extend the C++ renderer in `src/tokenizer/chat/` (added at M3)
4. Add a round-trip test: given a fixed message sequence, our C++ renderer must produce byte-identical output to what Jinja would produce (we cross-check against Python `jinja2` in a helper script, not at runtime)
