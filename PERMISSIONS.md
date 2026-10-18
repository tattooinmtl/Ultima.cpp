# Ultima — agent permissions

What the coding agent may do on its own, and what it has to ask about first.

The shape is deliberately conventional: **reads are never gated, writes are,
and a few things always ask no matter how permissive the mode is.** Approving
every single call — including reads — makes the agent unusable, so Ultima does
not do that.

## The three modes

| Mode | Ordinary writes | Always-ask set | Reads |
|------|-----------------|----------------|-------|
| **auto** (default) | allowed | asks | free |
| **ask** | asks | asks | free |
| **yolo** | allowed | allowed | free |

Set at startup with `ultima serve --permissions auto|ask|yolo`, or from the
Coder tab while a session is running.

- **auto** — the working default. The agent creates and edits files inside the
  project without interrupting you, and stops at anything that could lose work
  or reach outside the project.
- **ask** — every change is confirmed, including ordinary writes. Use it while
  you are still learning to trust a model.
- **yolo** — nothing is gated. This exists because sometimes you are watching
  the screen and the prompts are just friction. It is the only mode where a
  delete can happen without you seeing it first, and the UI says so.

## The always-ask set

These ask in **auto** and in **ask**. Only `yolo` skips them.

| Risk | What triggers it | Why it is here |
|------|------------------|----------------|
| `destructive` | `delete_file`, and any `rm` / `del` / `rmdir` / `format` style command | Deleting is the one file operation with no undo. |
| `escape` | a path that resolves outside the workspace | The agent has no business editing your home directory because a plan said so. |
| `plan` | writing `.ultima/plan.md` or `.ultima/todos.json` | The plan is the contract between you and the agent. It rewriting its own instructions should be visible. |
| `install` | `npm install`, `pip install`, `cargo add`, and friends | Pulls third-party code onto your machine. |
| `execute` | running a command outside the project directory | Arbitrary execution is not a file edit; it deserves its own confirmation. |

Note that `escape` is defence in depth, not the defence. Paths that leave the
workspace are already refused outright by `Workspace::resolve()` — see
roadmap01.md Decision 18 §18.2. The permission layer exists so that if a future
tool ever operates outside the workspace on purpose, it cannot do so silently.

## Every prompt says why

A request the user cannot evaluate is a request they will rubber-stamp, so each
one carries the tool, the exact argument at issue, and the reason:

```
Permission needed — destructive
  delete_file(path: "src/old-hero.css")
  Deleting a file cannot be undone from the editor.
  [Allow]  [Always allow this kind]  [Deny]
```

**Always allow this kind** remembers that risk class for the rest of the
session only. It is never written to disk: a new `ultima serve` starts from the
configured mode again. Denials are not remembered — a second attempt asks
again, because the reason you said no the first time may not hold.

## What happens on a denial

The tool call fails with `Permission denied by the user.` and that text goes
back to the model as the tool result. It is not an error the turn recovers
from by retrying the same call; a well-behaved model reads it and either tries
a different approach or stops and asks you.

A prompt left unanswered for two minutes is treated as a denial. Silence is
never consent.

## What is not gated, ever

`list_files` and `read_file`. The agent cannot usefully work blind, reading is
reversible, and gating it would train you to click Allow without looking —
which is how a permission system stops meaning anything.
