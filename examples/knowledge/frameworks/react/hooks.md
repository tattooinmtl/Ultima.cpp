---
id: react-hooks
title: React Hooks
type: howto
tags: [react, frontend, intermediate]
links: []
---
# Hooks

`useState` for local state. `useEffect(fn, deps)` for side effects; pass a deps array. `useMemo` / `useCallback` for stable references. Custom hooks start with `use`. Rules: only call at the top level, only from components or other hooks.