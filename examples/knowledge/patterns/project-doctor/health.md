---
id: pattern-project-doctor
title: Project Health Checks
type: pattern
tags: [process, health]
links: []
---
# Project Doctor

When a project feels sick: 1. `cargo build` / `npm run build` — does it compile? 2. `cargo test` / `npm test` — do the tests pass? 3. `cargo audit` / `npm audit` — known vulns? 4. `git log --oneline -20` — recent activity? 5. Open the README — does it match the code? 6. Look at the dir structure — anything obviously stale?