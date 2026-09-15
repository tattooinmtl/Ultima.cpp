---
id: py-type-hints
title: Python Type Hints
type: howto
tags: [python, typesystem, intermediate]
links: []
---
# Type Hints

`def f(x: int) -> str: ...` — hints are documentation + tooling support, NOT enforcement at runtime. Use `mypy` or `pyright` to actually check them. `from __future__ import annotations` to use the new syntax in old Python.