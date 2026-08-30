#!/usr/bin/env bash
# CI naming-taboo check (POSIX). Patterns constructed from fragments so this
# script does not self-trigger.
set -euo pipefail

P1="lla""ma"
P2="gg""ml"
PATTERNS="${P1}|${P2}"

HITS=$(grep -rniE "$PATTERNS" \
    --include='*.cpp' --include='*.hpp' --include='*.h' --include='*.c' \
    --include='*.cc' --include='*.cxx' \
    --include='*.cmake' --include='CMakeLists.txt' \
    --include='*.json' --include='*.yml' --include='*.yaml' \
    --include='*.py' --include='*.sh' --include='*.ps1' --include='*.bat' --include='*.cmd' \
    --include='*.go' \
    --exclude-dir='.git' --exclude-dir='build' --exclude-dir='out' \
    --exclude-dir='_deps' --exclude-dir='third_party' --exclude-dir='node_modules' \
    --exclude-dir='history' \
    --exclude='ci_taboo_check.*' --exclude='roadmap*.md' \
    . || true)

if [ -n "$HITS" ]; then
    echo "TABOO GATE FAILED - banned identifier found:"
    echo "$HITS"
    exit 1
fi

echo "Taboo gate: clean."
