Yes. Ultima.cpp is technically feasible, but I would separate the project into two layers:

A clean-room inference runtime, inspired by llama.cpp's role and APIs.
A persistent live-memory subsystem that modifies the model's effective behavior without modifying the base model weights.

The important distinction is that putting JSON into a folder does not constitute fine-tuning. What we can build is closer to a persistent external parameter/memory layer that the inference engine consults dynamically.

Proposed Ultima.cpp architecture
                    ┌─────────────────────┐
                    │     Base Model      │
                    │ GGUF / other format │
                    └──────────┬──────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────┐
│                    ULTIMA RUNTIME                   │
│                                                     │
│  tokenizer → transformer → sampler → output        │
│                    │                                │
│                    │                                │
│              ┌─────▼─────┐                          │
│              │   Memory   │                          │
│              │   Router   │                          │
│              └─────┬─────┘                          │
│                    │                                │
│       ┌────────────┼─────────────┐                  │
│       ▼            ▼             ▼                  │
│   memories/    adapters/      retrieval/            │
│    *.json       *.bin         index                 │
│       │            │             │                  │
│       └────────────┴─────────────┘                  │
│                    │                                │
│                    ▼                                │
│              augmented state                        │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
                  output
The key idea

Instead of:

model weights
     ↓
answer

Ultima becomes:

             ┌──────────────┐
             │ Base model   │
             └──────┬───────┘
                    │
             ┌──────▼───────┐
             │ Memory layer │
             └──────┬───────┘
                    │
        ┌───────────┴───────────┐
        │                       │
   long-term facts        behavioral data
   preferences            learned corrections
   project knowledge      agent state
        │                       │
        └───────────┬───────────┘
                    ▼
              model inference
                    │
                    ▼
                 answer
JSON could look like this

For example:

ultima/
├── models/
│   └── llama.gguf
│
├── memories/
│   ├── user.json
│   ├── project.json
│   ├── coding.json
│   └── learned.json
│
├── adapters/
│   └── ...
│
├── indexes/
│   └── memory.index
│
└── config.json

And:

{
  "id": "project_001",
  "type": "knowledge",
  "priority": 0.85,
  "content": {
    "language": "C++",
    "framework": "Ultima",
    "architecture": "MoE + recurrent memory"
  },
  "created": "2026-08-29T00:00:00Z"
}

But I wouldn't feed the raw JSON directly into the context every time.

That would eventually become:

1 GB memories
       ↓
huge prompt
       ↓
expensive inference

Instead, Ultima should turn the JSON into a memory representation.

The interesting part: make memory a first-class inference primitive

This is where your idea gets much more interesting.

We could give Ultima a native memory API:

MemoryStore memory;

memory.add(...);
memory.search(...);
memory.update(...);
memory.forget(...);
memory.rank(...);

Then during inference:

MemoryContext mem =
    memory.retrieve(tokens, state);

model.forward(tokens, mem);

Eventually the model architecture could become:

                    tokens
                      │
                      ▼
                ┌──────────┐
                │ tokenizer│
                └────┬─────┘
                     │
             ┌───────▼────────┐
             │ memory router  │
             └───────┬────────┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       episodic   semantic   procedural
       memory      memory      memory
          │          │          │
          └──────────┼──────────┘
                     ▼
              model hidden state
                     │
                     ▼
              transformer/MoE
                     │
                     ▼
                   token

That would be a genuinely interesting inference framework.

But there's an even better approach

I would not initially modify the Transformer architecture.

Build Ultima in stages.

Phase 1 — llama.cpp-style runtime

Build:

ultima.cpp
├── tensor/
├── gguf/
├── tokenizer/
├── model/
├── cpu/
├── cuda/
├── metal/
├── vulkan/
├── sampler/
├── kv_cache/
└── cli/

The goal:

ultima -m model.gguf -p "Hello"

works.

Then:

ultima-server

provides an OpenAI-compatible API.

Phase 2 — MemoryFS

Add:

memory/

with:

memory/
├── facts/
├── conversations/
├── projects/
├── skills/
├── preferences/
└── events/

Ultima watches the directory.

When:

memory/projects/ultima.json

changes:

filesystem
    ↓
memory parser
    ↓
validation
    ↓
embedding
    ↓
index

No model restart necessary.

That gives you the live information layer you're describing.

Phase 3 — Retrieval

When the model receives:

How does our tensor allocator work?

Ultima searches:

query
 ↓
embedding
 ↓
memory index
 ↓
top-k memories
 ↓
relevance scoring
 ↓
model

This is essentially native RAG, except the memory system becomes part of the inference runtime rather than something bolted onto an application.

Phase 4 — Persistent state

Now we introduce something more interesting.

The model maintains:

short-term state
       +
persistent memory
       +
base model

For example:

Base model
     │
     ├── temporary KV cache
     │
     ├── recurrent state
     │
     └── persistent memory
             │
             ├── facts
             ├── skills
             ├── experiences
             └── corrections

This starts approaching the philosophy you're describing.

Phase 5 — Actual live adaptation

This is where we can go beyond RAG.

Suppose the model repeatedly makes this mistake:

Input:
Use Ultima's tensor API.

Model:
incorrect API...

We record:

{
  "type": "correction",
  "pattern": "tensor allocation",
  "wrong": "...",
  "correct": "...",
  "confidence": 0.94
}

Now Ultima can use that correction in future inference.

But we have several levels of adaptation available:

Level 0 — prompt memory
memory → prompt

Easy.

Level 1 — embedding retrieval
memory → vector search → context

Better.

Level 2 — hidden-state memory
memory → learned projection → hidden state

Much more interesting.

Level 3 — adapter memory
memory → LoRA/adapter → model

Now we're getting close to actual fine-tuning.

Level 4 — online weight updates
input
 ↓
loss
 ↓
gradient
 ↓
weights

That's genuine live learning.

I would not make Level 4 the first implementation.

It's substantially harder to make stable and safe.

The architecture I'd actually build

I'd call the subsystem ULM — Ultima Learning Memory.

                     ULTIMA.CPP
                         │
       ┌─────────────────┼──────────────────┐
       │                 │                  │
       ▼                 ▼                  ▼
   inference          memory             adapters
     engine            engine              engine
       │                 │                  │
       │        ┌────────┼────────┐         │
       │        │        │        │         │
       │      facts   episodes  skills    LoRA
       │        │        │        │         │
       └────────┴────────┼────────┴─────────┘
                        │
                        ▼
                 Memory Router
                        │
                        ▼
                  model state

The model itself remains immutable:

model.gguf

while the living intelligence exists separately:

model.gguf
     +
ULM
     +
adapters
     +
runtime state

That is a much cleaner architecture.

And JSON is only the human interface

I'd make JSON the source format, not the runtime format.

For example:

memory.json
      │
      ▼
   parser
      │
      ▼
 normalized memory
      │
      ├── metadata
      ├── embeddings
      ├── compressed representation
      └── index

Eventually:

memories/
    *.json

        ↓

memory.db
memory.index
memory.cache

The user can still edit JSON manually.

We can also borrow the Qwen philosophy

This is where your original Qwen question becomes relevant.

We could design Ultima around:

             ULTIMA MEMORY ARCHITECTURE

                     Context
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
       Fast memory            Exact retrieval
        recurrent               sparse index
             │                     │
             │                     │
             └──────────┬──────────┘
                        ▼
                  Memory state
                        │
                        ▼
                     MoE model

So the runtime could eventually support:

fast recurrent memory + sparse long-term retrieval + MoE inference + persistent external memory.

That's much more ambitious than cloning llama.cpp.

One important legal/engineering point

If the goal is a genuinely independent project, I would not copy llama.cpp's source code and rename it.

Instead, we can implement a clean-room inference engine using public model specifications, published papers, open model formats/specifications where permitted, and independently written code.

That also gives us freedom to design Ultima's architecture properly rather than inheriting every historical constraint of llama.cpp.

I think the project should start here

I'd define the first version as:

ULTIMA.CPP v0.1

Core
├── Tensor engine
├── GGUF reader
├── tokenizer
├── Transformer inference
├── KV cache
├── CPU backend
├── CUDA backend
└── sampler

ULM
├── JSON memory format
├── filesystem watcher
├── memory database
├── embedding index
├── retrieval
├── ranking
└── memory injection

API
├── CLI
├── HTTP server
└── OpenAI-compatible endpoint

Then v0.2:

ULM
├── episodic memory
├── semantic memory
├── procedural memory
├── memory consolidation
├── forgetting
├── confidence
└── contradiction resolution

Then v0.3:

Adaptive layer
├── hidden-state adapters
├── LoRA hot-loading
├── online adapter training
└── per-project model state

And only after that:

Ultima architecture
├── recurrent memory layers
├── sparse attention
├── MoE
├── gated residual streams
└── native live-learning mechanisms

That gives us a realistic path from "open-source llama.cpp alternative" → "inference engine with persistent intelligence" → "new model/runtime architecture."


-----------------------------------------------------------------------------------------------------------------------------------------------------

# Ultima.cpp v0.1

## Technical Specification & Repository Architecture

**Project:** Ultima.cpp
**Version:** 0.1.0
**Language:** C++20
**Status:** Initial architecture
**Primary goal:** Build an independent, modular local LLM inference runtime from zero, with persistent memory designed into the architecture from the beginning.

---

# 1. Project Vision

Ultima.cpp is an independent C++ inference runtime for running modern neural language models locally.

The project is designed around a long-term architecture where:

```text
                BASE MODEL
                    +
              ULTIMA RUNTIME
                    +
            PERSISTENT MEMORY
                    +
             OPTIONAL ADAPTERS
                    =
          LIVING MODEL INSTANCE
```

The base model remains immutable.

Knowledge, memories, adaptations, and runtime state exist outside the base model and can evolve independently.

The ultimate objective is to support:

* local inference
* multiple model architectures
* CPU/GPU execution
* quantized models
* persistent memory
* retrieval
* recurrent state
* sparse attention
* mixture-of-experts
* adapters
* eventually online adaptation

---

# 2. v0.1 Scope

v0.1 is intentionally conservative.

## Included

### Runtime

* C++20 core
* tensor abstraction
* tensor operations
* model abstraction
* tokenizer abstraction
* computation graph abstraction
* execution context
* KV cache abstraction
* sampler
* model loader abstraction
* GGUF-compatible loader
* CPU backend
* basic CLI
* basic HTTP server architecture

### Memory

* JSON memory files
* memory directory
* memory parser
* memory validation
* memory metadata
* memory indexing abstraction
* memory retrieval interface
* memory injection interface

### Engineering

* CMake
* unit tests
* integration tests
* benchmarks
* logging
* error handling
* documentation
* deterministic testing infrastructure

---

# 3. Explicitly NOT in v0.1

The following are future architecture targets:

* Gated DeltaNet
* Qwen Sparse Attention
* custom sparse attention kernels
* MoE routing
* online gradient updates
* live fine-tuning
* LoRA training
* GPU training
* distributed inference
* speculative decoding
* multimodal models
* automatic memory generation
* autonomous agents

The interfaces should accommodate these features later, but v0.1 should not become blocked by them.

---

# 4. Design Principles

## 4.1 Independent implementation

Ultima.cpp is not a rename or source-code fork of another inference runtime.

The implementation should be independently written.

Public model architecture specifications and documented file formats may be implemented where their licensing/usage terms permit.

---

## 4.2 Modular backends

The model should not know whether computation is running on:

```text
CPU
CUDA
Metal
Vulkan
other accelerator
```

The backend abstraction owns hardware-specific execution.

---

## 4.3 Model/runtime separation

A model is data.

The runtime executes it.

```text
Model
 └── weights
 └── vocabulary
 └── architecture metadata

Runtime
 └── execution
 └── scheduling
 └── memory
 └── sampling
 └── backends
```

---

## 4.4 Immutable base model

The original model file is never modified by memory operations.

```text
models/model.gguf
```

is read-only.

Memory lives separately:

```text
memory/
```

Adapters eventually live separately:

```text
adapters/
```

This allows:

```text
one model
    +
multiple memory profiles
```

---

## 4.5 Everything important gets an interface

Avoid architecture-specific code spreading through the entire repository.

For example:

```cpp
class IModel;
class ITokenizer;
class IBackend;
class IMemoryStore;
class ISampler;
class IModelLoader;
```

Future implementations can be added behind these interfaces.

---

# 5. Repository Layout

```text
ultima.cpp/
│
├── CMakeLists.txt
├── CMakePresets.json
├── LICENSE
├── README.md
├── CONTRIBUTING.md
├── SECURITY.md
├── CHANGELOG.md
│
├── docs/
│   ├── architecture/
│   │   ├── overview.md
│   │   ├── runtime.md
│   │   ├── tensors.md
│   │   ├── memory.md
│   │   ├── models.md
│   │   └── backends.md
│   │
│   └── specifications/
│       └── v0.1.md
│
├── include/
│   └── ultima/
│       ├── core/
│       ├── tensor/
│       ├── model/
│       ├── tokenizer/
│       ├── runtime/
│       ├── memory/
│       ├── backend/
│       ├── sampling/
│       └── io/
│
├── src/
│   ├── core/
│   ├── tensor/
│   ├── model/
│   ├── tokenizer/
│   ├── runtime/
│   ├── memory/
│   ├── backend/
│   │   └── cpu/
│   ├── sampling/
│   └── io/
│
├── apps/
│   ├── ultima-cli/
│   └── ultima-server/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── benchmarks/
│
├── examples/
│   ├── basic_inference/
│   └── memory/
│
├── scripts/
│
└── third_party/
```

---

# 6. Core Runtime

The central object is:

```cpp
class Runtime;
```

Conceptually:

```text
Runtime
 │
 ├── Model
 ├── Tokenizer
 ├── Backend
 ├── ExecutionContext
 ├── KVCache
 ├── MemoryStore
 └── Sampler
```

Example lifecycle:

```cpp
Runtime runtime(config);

runtime.load_model("model.gguf");

runtime.load_memory("memory/");

auto result =
    runtime.generate("Explain this project.");

std::cout << result.text;
```

---

# 7. Tensor System

The tensor subsystem is the foundation.

```cpp
enum class DataType {
    F32,
    F16,
    BF16,
    I8,
    I16,
    I32,
    I64,
};
```

A tensor should contain:

```cpp
class Tensor {
public:
    DataType dtype() const;

    const std::vector<size_t>& shape() const;

    size_t size_bytes() const;

    void* data();
    const void* data() const;
};
```

The tensor object itself should not contain CPU-specific assumptions.

---

# 8. Tensor Operations

v0.1 should define an operation layer:

```text
matmul
add
mul
sub
div
reshape
transpose
permute
softmax
rms_norm
rope
embedding
argmax
```

Initially:

```text
Tensor operation
      ↓
CPU implementation
```

Later:

```text
Tensor operation
      ↓
backend dispatcher
      ├── CPU
      ├── CUDA
      ├── Metal
      └── Vulkan
```

---

# 9. Backend Architecture

```cpp
class IBackend {
public:
    virtual ~IBackend() = default;

    virtual Tensor allocate(...) = 0;

    virtual void copy(...) = 0;

    virtual void matmul(...) = 0;

    virtual void synchronize() = 0;
};
```

v0.1:

```text
IBackend
   │
   └── CPUBackend
```

Future:

```text
IBackend
 ├── CPUBackend
 ├── CUDABackend
 ├── MetalBackend
 └── VulkanBackend
```

---

# 10. Model Abstraction

Models should expose a common interface.

```cpp
class IModel {
public:
    virtual ~IModel() = default;

    virtual ModelInfo info() const = 0;

    virtual void initialize(RuntimeContext&) = 0;

    virtual void forward(
        const TokenBatch&,
        ExecutionContext&
    ) = 0;
};
```

Architecture-specific implementations eventually become:

```text
models/
 ├── llama/
 ├── qwen/
 ├── mistral/
 ├── gemma/
 └── ...
```

The runtime does not need to know architecture internals.

---

# 11. GGUF Loader

v0.1 should implement a dedicated model-file layer.

```cpp
class IModelLoader {
public:
    virtual ~IModelLoader() = default;

    virtual ModelMetadata inspect(
        const std::filesystem::path&
    ) = 0;

    virtual LoadedModel load(
        const std::filesystem::path&
    ) = 0;
};
```

The loader is responsible for:

```text
file
 ↓
header
 ↓
metadata
 ↓
tensor directory
 ↓
tensor data
 ↓
LoadedModel
```

Do not scatter file-format parsing throughout the inference engine.

---

# 12. Tokenizer

```cpp
class ITokenizer {
public:
    virtual ~ITokenizer() = default;

    virtual std::vector<Token>
    encode(std::string_view text) = 0;

    virtual std::string
    decode(const std::vector<Token>& tokens) = 0;
};
```

The tokenizer implementation is determined by model metadata.

---

# 13. KV Cache

The KV cache should be an independent subsystem.

```cpp
class KVCache {
public:
    void reset();

    void reserve(size_t tokens);

    size_t size() const;

    size_t capacity() const;
};
```

The important design decision:

**do not make the KV cache synonymous with the model.**

Future recurrent architectures may not use a traditional KV cache.

Therefore:

```text
StateStore
   ├── KVCache
   ├── RecurrentState
   └── FutureMemoryState
```

is the preferred long-term architecture.

---

# 14. State Architecture

Introduce this abstraction from v0.1:

```cpp
class IModelState {
public:
    virtual ~IModelState() = default;

    virtual void reset() = 0;
};
```

Later:

```text
IModelState
 ├── KVState
 ├── GDNState
 ├── recurrent state
 └── hybrid state
```

This is important because Ultima should eventually support architectures that aren't purely Transformer/KV-cache based.

---

# 15. Memory System

The memory system is the defining Ultima feature.

Directory:

```text
memory/
├── facts/
├── projects/
├── conversations/
├── skills/
├── preferences/
└── events/
```

Each memory is independently addressable.

Example:

```json
{
  "id": "project-001",
  "type": "knowledge",
  "version": 1,
  "created_at": "2026-08-29T00:00:00Z",
  "updated_at": "2026-08-29T00:00:00Z",
  "importance": 0.8,
  "confidence": 0.95,
  "tags": [
    "ultima",
    "cpp",
    "architecture"
  ],
  "content": {
    "title": "Ultima architecture",
    "text": "Ultima uses a modular inference runtime..."
  }
}
```

---

# 16. Memory API

```cpp
class IMemoryStore {
public:
    virtual ~IMemoryStore() = default;

    virtual MemoryId add(
        const Memory& memory
    ) = 0;

    virtual bool update(
        const Memory& memory
    ) = 0;

    virtual bool remove(
        const MemoryId&
    ) = 0;

    virtual std::vector<Memory>
    search(
        std::string_view query,
        size_t limit
    ) = 0;

    virtual std::optional<Memory>
    get(const MemoryId&) = 0;
};
```

The important point is that the model does not directly read JSON.

Instead:

```text
JSON
 ↓
MemoryStore
 ↓
Memory objects
 ↓
retrieval
 ↓
MemoryContext
 ↓
model
```

---

# 17. Memory Context

The inference engine should receive a structured memory context.

```cpp
struct MemoryContext {
    std::vector<Memory> memories;

    size_t token_budget;

    float retrieval_threshold;

    bool enabled;
};
```

Then:

```cpp
GenerationRequest request;

request.prompt = "...";
request.memory = memory_context;
```

This gives us a clean point where future memory mechanisms can be inserted.

---

# 18. Memory Injection

v0.1 should use **context injection**, not hidden-state modification.

Pipeline:

```text
user prompt
     │
     ▼
memory retrieval
     │
     ▼
relevant memories
     │
     ▼
memory formatter
     │
     ▼
augmented prompt
     │
     ▼
model
```

This is deliberately simple.

Later we can replace:

```text
Memory → Prompt
```

with:

```text
Memory → Hidden State
```

without redesigning the entire runtime.

---

# 19. Memory Priority

Every memory should support:

```text
importance
confidence
recency
relevance
```

Conceptual retrieval score:

```text
score =
    relevance
  × confidence
  × importance
  × recency_factor
```

The exact formula is not fixed in v0.1.

The scoring interface is what matters.

---

# 20. Memory Isolation

Memory profiles should be independent.

Example:

```text
profiles/
├── default/
│   └── memory/
│
├── coding/
│   └── memory/
│
└── project-alpha/
    └── memory/
```

One model can therefore run with different persistent personalities/knowledge bases without changing the model file.

---

# 21. Generation Pipeline

The v0.1 pipeline:

```text
                   REQUEST
                      │
                      ▼
                 tokenizer
                      │
                      ▼
              memory retrieval
                      │
                      ▼
              context builder
                      │
                      ▼
                 model state
                      │
                      ▼
                 model forward
                      │
                      ▼
                  logits
                      │
                      ▼
                   sampler
                      │
                      ▼
                  next token
                      │
                      └───────┐
                              │
                         repeat until
                            stop
```

---

# 22. Sampling

v0.1 should support:

```text
greedy
temperature
top-k
top-p
min-p
repetition penalty
seed
```

Interface:

```cpp
class ISampler {
public:
    virtual ~ISampler() = default;

    virtual Token sample(
        const Logits& logits
    ) = 0;
};
```

Sampling must be deterministic when a seed is supplied.

---

# 23. CLI

Initial interface:

```bash
ultima -m model.gguf
```

Prompt:

```bash
ultima -m model.gguf -p "Hello world"
```

Interactive:

```bash
ultima -m model.gguf --interactive
```

Memory:

```bash
ultima \
  -m model.gguf \
  --memory ./memory/
```

Parameters:

```bash
ultima \
  -m model.gguf \
  --ctx 8192 \
  --temperature 0.7 \
  --top-k 40 \
  --top-p 0.95
```

---

# 24. Server

v0.1 should establish an HTTP server interface.

Target:

```text
POST /v1/chat/completions
POST /v1/completions
GET  /v1/models
GET  /health
```

The server should eventually be usable by:

```text
Python
JavaScript
OpenAI-compatible clients
local applications
agents
```

Memory can eventually be selected per request:

```json
{
  "model": "ultima-model",
  "messages": [],
  "memory_profile": "coding"
}
```

---

# 25. Configuration

Example:

```json
{
  "model": "./models/model.gguf",

  "backend": "cpu",

  "context": {
    "size": 8192
  },

  "generation": {
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.95
  },

  "memory": {
    "enabled": true,
    "path": "./memory",
    "max_tokens": 2048
  }
}
```

---

# 26. Logging

Use structured logging.

Levels:

```text
TRACE
DEBUG
INFO
WARN
ERROR
FATAL
```

Example:

```text
[INFO] Loading model: llama.gguf
[INFO] Architecture: llama
[INFO] Context: 8192
[INFO] Backend: CPU
[INFO] Memory store: ./memory
[INFO] Loaded memories: 143
```

---

# 27. Error Handling

Do not use silent failures.

Errors should contain:

```text
error code
message
component
optional underlying cause
```

Example:

```cpp
enum class ErrorCode {
    FileNotFound,
    InvalidModel,
    UnsupportedArchitecture,
    UnsupportedDataType,
    OutOfMemory,
    InvalidTokenizer,
    InvalidMemory,
    BackendFailure
};
```

---

# 28. Testing Strategy

Every subsystem gets tests.

```text
tests/
├── unit/
│   ├── tensor/
│   ├── tokenizer/
│   ├── model/
│   ├── memory/
│   ├── sampler/
│   └── io/
│
├── integration/
│   ├── model_loading/
│   ├── inference/
│   └── memory_inference/
│
└── fixtures/
```

Critical tests:

### Tensor

```text
matmul correctness
reshape correctness
transpose correctness
dtype conversion
```

### Model

```text
model loads
metadata loads
weights load
forward pass completes
```

### Memory

```text
JSON parses
invalid JSON rejected
memory indexing works
search returns relevant entries
memory updates work
memory deletion works
```

### Runtime

```text
prompt → tokens → logits → token
```

---

# 29. Benchmarking

Benchmarks should measure:

```text
model loading
prompt processing
tokens/second
first-token latency
memory retrieval latency
sampling latency
RAM usage
model memory usage
```

Output:

```text
Model: example.gguf
Backend: CPU

Prompt tokens: 512
Generated tokens: 128

Prompt processing: 23.4 tok/s
Generation:        11.8 tok/s

Memory retrieval:  1.7 ms
```

---

# 30. Future Architecture

The v0.1 interfaces deliberately lead toward this:

```text
                         ULTIMA
                           │
          ┌────────────────┼────────────────┐
          │                │                │
       inference         memory           state
          │                │                │
     ┌────┼────┐       ┌───┼────┐       ┌───┼────┐
     │    │    │       │   │    │       │   │    │
   dense MoE  sparse  facts episodic  KV  GDN  future
             attention
```

Eventually:

```text
                  MODEL
                    │
        ┌───────────┼────────────┐
        │           │            │
       MoE        GDN           QSA
        │           │            │
        │      recurrent       sparse
        │       memory        retrieval
        │           │            │
        └───────────┼────────────┘
                    │
             ULTIMA MEMORY
                    │
       ┌────────────┼─────────────┐
       │            │             │
    semantic     episodic     procedural
       │            │             │
       └────────────┼─────────────┘
                    │
              persistent disk
```

This is the long-term research direction.

---

# 31. v0.1 Milestones

## M0 — Repository

```text
CMake
source tree
headers
tests
documentation
CI
```

## M1 — Tensor Engine

```text
Tensor
DataType
Shape
CPU operations
```

## M2 — Model Loading

```text
GGUF reader
metadata
tensor loading
```

## M3 — Tokenization

```text
tokenizer abstraction
model tokenizer
encode/decode
```

## M4 — First Inference

Target:

```bash
ultima -m model.gguf -p "Hello"
```

produces valid output.

## M5 — KV State

Implement:

```text
context
KV cache
incremental generation
```

## M6 — Sampling

Implement:

```text
temperature
top-k
top-p
min-p
seed
```

## M7 — Memory

Implement:

```text
JSON
MemoryStore
filesystem discovery
retrieval
context injection
```

Target:

```bash
ultima \
  -m model.gguf \
  --memory ./memory
```

## M8 — Server

Implement:

```text
HTTP
OpenAI-compatible API
streaming
```

## M9 — Benchmark & Stabilization

Before calling v0.1 complete:

```text
unit tests
integration tests
benchmarks
memory stress tests
long-context tests
documentation
```

---

# 32. Definition of Done

Ultima.cpp v0.1 is complete when:

```text
✓ Builds from a clean checkout
✓ Loads a supported model
✓ Tokenizes input
✓ Executes inference
✓ Generates tokens
✓ Supports CPU execution
✓ Maintains model state
✓ Supports configurable sampling
✓ Loads JSON memories
✓ Retrieves relevant memories
✓ Injects memories into inference
✓ Supports multiple memory profiles
✓ Provides CLI
✓ Provides HTTP API
✓ Has automated tests
✓ Has benchmarks
✓ Has architecture documentation
```

The most important demonstration is:

```bash
ultima \
  --model models/example.gguf \
  --memory profiles/project-alpha/memory
```

Then:

```text
User:
What do you know about this project?

Ultima:
[answer based on the base model + project memory]
```

The model file itself remains unchanged.

---

# 33. Long-Term Goal

Ultima.cpp should eventually evolve from:

```text
LLM inference engine
```

into:

```text
MODEL RUNTIME
     +
STATE ENGINE
     +
MEMORY ENGINE
     +
ADAPTATION ENGINE
```

The ultimate architecture should allow:

```text
Base model
    │
    ├── persistent semantic memory
    ├── episodic memory
    ├── procedural memory
    ├── recurrent state
    ├── sparse retrieval
    ├── adapters
    └── future live-learning layers
```

without requiring the base model to be permanently modified.

This means an Ultima model instance can become increasingly specialized while preserving the original model weights.

---

# 34. First Repository Commit

The first commit should contain **architecture, interfaces, build system, tests, and documentation — not a giant inference implementation**.

Recommended initial commit:

```text
feat: initialize Ultima.cpp v0.1 runtime architecture
```

Contents:

```text
CMakeLists.txt
CMakePresets.json
README.md

include/ultima/
    core/
    tensor/
    model/
    tokenizer/
    runtime/
    memory/
    backend/
    sampling/
    io/

src/
    corresponding implementations

tests/
    basic infrastructure

docs/
    architecture/
    specifications/

apps/
    ultima-cli/
```

The first executable should simply prove the architecture works:

```bash
ultima --version
```

followed by:

```bash
ultima --help
```

Then the tensor engine becomes the first real subsystem.

---

# 35. Architectural Rule

One rule should govern the entire project:

> **Do not build today's implementation in a way that prevents tomorrow's architecture.**

Specifically, never assume:

```text
Transformer = model
KV cache = state
prompt = memory
weights = intelligence
GPU = execution
```

Ultima should treat those as implementations of broader abstractions.

That is what allows the project to eventually incorporate recurrent memory, sparse attention, MoE routing, adapters, and genuine online adaptation without replacing the entire runtime.
