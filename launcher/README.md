# Ultima launcher

The GUI for finding GGUF files, launching the Ultima runtime, and opening the testpad. Implements Decision 03 (launcher) + Decision 03c (model paths + Load & Test flow).

## v0.1 shipping shape

Rather than the Wails-native window Decision 03 targets, v0.1 ships as a **plain Go binary that serves an HTML launcher UI over HTTP on `127.0.0.1:11435`** and opens the user's default browser to it at startup. This is significantly less code, has no Node / npm dependency, and the browser is where the chat testpad lives anyway (Decision 03b). Migration to Wails for a native window is a v0.2 concern.

## Build

```powershell
cd launcher
go build -o ultima-launcher.exe .
```

## Run

```powershell
# Default: bind 127.0.0.1:11435, open browser at startup.
# Runtime binary defaults to `ultima.exe` next to the launcher.
./ultima-launcher.exe

# Custom runtime path / port / no browser popup:
./ultima-launcher.exe --runtime "C:\Ultima\ultima.exe" --port 11435 --open=false
```

## What it does

- **Model discovery** across five roots (Decision 03c §3c.1):
  1. `%APPDATA%\Ultima\models\`
  2. `<launcher_dir>\models\` (portable co-location)
  3. `%USERPROFILE%\.ollama\models\`
  4. `%USERPROFILE%\.cache\lm-studio\models\`
  5. Any user-added path saved in `launcher-config.json`

  First-match-wins on filename dedup. Missing roots are skipped silently.

- **Load & Test flow** (Decision 03c §3c.3): pick a `.gguf` from the table, click **Load & test** — the launcher spawns `ultima serve <path> --port 11434`, polls `/api/health` until it answers, then opens the testpad in a new browser tab. Reloading a different model kills the current runtime first.

- **Settings** persisted to `%APPDATA%\Ultima\launcher-config.json`:
  runtime host / port / n_ctx / n_threads, open-testpad-on-load, autoload-default-on-launch, default model path, model roots.

## Endpoints

The launcher's own HTTP surface (separate from the runtime's `11434`):

| Method | Path            | Purpose                                                            |
|--------|-----------------|--------------------------------------------------------------------|
| GET    | `/`             | Launcher UI (embedded HTML)                                        |
| GET    | `/api/config`   | Return current config JSON                                         |
| POST   | `/api/config`   | Update + persist config                                            |
| GET    | `/api/models`   | Walk roots, return `{models, roots, loaded}`                       |
| POST   | `/api/rescan`   | Same as GET /api/models (semantic name for the UI's Rescan button) |
| POST   | `/api/load`     | `{path}` → spawn `ultima serve <path>`, return `{loaded, testpad}` |
| POST   | `/api/unload`   | Kill the running runtime                                           |
| GET    | `/api/status`   | Running runtime info (empty if none)                               |
| POST   | `/api/open`     | Open the running testpad URL in the default browser                |

## Deferred (v0.2+)

- Native Wails window (per Decision 03 §3.2 — this is the ship target once the runtime is proven out).
- HuggingFace URL paste-to-download (Decision 03c §3c.6).
- Two-models-simultaneously (coder + embed).
- MCP server management tab (Decision 10) — Ultima runtime handles MCP; the launcher UI just lists configured servers when that lands.
- Cross-conversation memory shard editor (Decision 12).
