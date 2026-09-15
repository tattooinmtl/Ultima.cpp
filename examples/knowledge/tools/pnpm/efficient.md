---
id: pnpm-efficient
title: pnpm Efficient Installs
type: howto
tags: [pnpm, node]
links: []
---
# pnpm

`pnpm install` reads `pnpm-lock.yaml` and resolves to a content-addressable store, symlinking into `node_modules`. Massive disk savings for monorepos. `pnpm dlx` is the equivalent of `npx`. Use `pnpm-workspace.yaml` for monorepos.