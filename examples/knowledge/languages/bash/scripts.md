---
id: bash-scripts
title: Bash Scripts
type: howto
tags: [bash, scripting, beginner]
links: []
---
# Bash

Start with `#!/usr/bin/env bash`. Use `set -euo pipefail` to fail fast. Quote every variable: `"$var"`. Use `[[ ]]` not `[ ]`. Prefer `$(cmd)` over backticks. `local` in functions. `trap 'cleanup' EXIT` for cleanup.