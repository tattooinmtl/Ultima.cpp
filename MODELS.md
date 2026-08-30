# Ultima.cpp — Model Registry

Authoritative list of models Ultima ships defaults for. Full JSON registry lives in `config/models.json` (added at Decision 04+).

## Coder slot

| Tier   | Model                              | Arch   | Q4_K_M size | License     |
|--------|------------------------------------|--------|-------------|-------------|
| Tiny   | Qwen2.5-Coder-0.5B-Instruct        | qwen2  | ~0.4 GB     | Apache-2.0  |
| Small  | Qwen2.5-Coder-1.5B-Instruct        | qwen2  | ~1.0 GB     | Apache-2.0  |
| Mid    | Qwen2.5-Coder-7B-Instruct          | qwen2  | ~4.5 GB     | Apache-2.0  |
| Mid+   | Ornith-1.5-9B                      | qwen3  | ~5.5 GB     | MIT         |
| Large  | Qwen2.5-Coder-14B-Instruct         | qwen2  | ~8.5 GB     | Apache-2.0  |

## Embed slot

| Tier   | Model                              | Dim   | Q8 size | License     |
|--------|------------------------------------|-------|---------|-------------|
| Tiny   | all-MiniLM-L6-v2                   | 384   | ~25 MB  | Apache-2.0  |
| Small  | bge-small-en-v1.5                  | 384   | ~35 MB  | MIT         |
| Mid    | bge-base-en-v1.5                   | 768   | ~110 MB | MIT         |
| Large  | bge-large-en-v1.5                  | 1024  | ~340 MB | MIT         |

## Notes

- v0.1 supports **Qwen2** architecture. **Qwen3** (Ornith / Qwen3-Coder) unlocks at the M4+ milestone.
- All models are downloaded at runtime by the model registry (Decision 01e), never committed to the repository.
- Users may point any slot at any local GGUF file via the launcher's Models tab or `ultima models use`.
