---
name: sandbox-python
command: /py
description: Write a Python script and execute it through the python_run sandbox tool. Use when the task benefits from Python's ecosystem (data munging, image processing, quick math) rather than JavaScript.
tags: [python, sandbox, scripting]
---

# sandbox-python

You cannot reason about every Python quirk. **Write a script, run it, read the output.** The sandbox tool `python_run` gives you a fresh subprocess per call.

## When to use

- The user needs a JSON reshape, a CSV parse, an image thumbnail, a Pillow crop, a `numpy` matrix op — Python is a saner fit than JS.
- The user has a Python codebase and wants a small script to plug into it.
- Verifying: writing a snippet that prints the answer to a math or data question, then reading stdout.

## Contract

```
python_run(code: str, stdin?: str, timeout_ms?: int)
  -> { stdout: str, stderr: str, exit_code: int, duration_ms: int }
```

- Timeout defaults to 10 s, hard-capped at 60 s.
- No persistent state between calls. Each call is a fresh interpreter.
- Working dir is a temp sandbox — write files to `./out.png` etc. and read them back via `read_file`.
- stdout/stderr are truncated at 64 KB.

## Boilerplate

```python
# Always: fail loudly.
import sys, json, traceback

def main():
    # ...work...
    print(json.dumps({"result": ...}))

try:
    main()
except Exception:
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
```

## Patterns

- **Deps**: only stdlib is guaranteed. `numpy`, `Pillow`, `requests` are common but not guaranteed — try to import; on `ModuleNotFoundError`, tell the user which pip install is needed.
- **File I/O**: write output files to the CWD, then use `read_file(...)` to pull the bytes into the tool result.
- **JSON in, JSON out**: pipe complex inputs via `stdin=json.dumps(...)` and read them with `data = json.loads(sys.stdin.read())`. Cleaner than string-formatting Python code.

## Avoid

- Long-running scripts (>60 s hard cap). Break work into stages.
- Interactive prompts (`input()`) — stdin is empty unless you pass it explicitly.
- Global side effects that assume a persistent interpreter — each call starts fresh.
