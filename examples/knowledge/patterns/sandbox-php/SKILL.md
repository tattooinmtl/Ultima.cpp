---
name: sandbox-php
command: /php
description: Write a PHP script and execute it through the php_run sandbox tool. Use for PHP-specific tasks (existing PHP codebases, PHP quirks, WordPress plugin bits).
tags: [php, sandbox, scripting]
---

# sandbox-php

Same shape as `sandbox-python` — write a script, run it via the tool, read the output.

## Contract

```
php_run(code: str, stdin?: str, timeout_ms?: int)
  -> { stdout: str, stderr: str, exit_code: int, duration_ms: int }
```

- 10 s default, 60 s hard cap.
- Fresh interpreter per call — no session state.
- Working dir is a temp sandbox — write files, then `read_file` them back.

## Boilerplate

```php
<?php
declare(strict_types=1);
try {
    // work...
    echo json_encode(['result' => 42]);
} catch (Throwable $e) {
    fwrite(STDERR, $e->getMessage() . "\n");
    exit(1);
}
```

## Patterns

- **JSON I/O**: `json_decode(stream_get_contents(STDIN), true)` for input; `json_encode(...)` to stdout.
- **File writes**: `file_put_contents('out.txt', $data)` inside the sandbox CWD.
- **Composer / autoload**: usually unavailable in the sandbox — write vanilla PHP or fetch the file via `read_file` from the workspace.
- **String parsing**: `preg_match_all` for regex; PHP's regex flavor is PCRE — same as JavaScript's `RegExp` with minor differences.

## Avoid

- Assuming PHP extensions beyond the standard ones. `mbstring`, `json`, `curl` usually there; `imagick`, `gd`, `mysqli` maybe not.
- `session_start()` and cookies — no HTTP layer around the sandbox.
- `sleep()` beyond a second or two — timeout will kill the script.
