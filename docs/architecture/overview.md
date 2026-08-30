# Architecture overview

Placeholder. Detailed architecture docs land as their subsystems are implemented.

For now, the authoritative design source is [`../../roadmap01.md`](../../roadmap01.md).

Planned files under this directory:

- `overview.md`   (this file)
- `runtime.md`    top-level Runtime lifecycle, module wiring
- `tensors.md`    tensor system, dtypes, allocators
- `backends.md`   IBackend interface, CPU/Vulkan/CUDA/Metal shapes
- `models.md`     IModel interface, per-architecture adapters (Qwen2, Qwen3)
- `memory.md`     memory subsystem, language shards, retrieval pipeline
- `tools.md`      native tool registry, MCP client
- `webui.md`      HTTP endpoints, static asset layout, streaming protocol
