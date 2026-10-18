---
id: pattern-anyhow
title: Application Error Handling with anyhow
type: pattern
tags: [rust, errors, application]
links: []
---
# anyhow for Apps

Use `anyhow::Result` in application binaries. Use a real error type in libraries. Crate the error with `.context("...")` to add a human-readable note. `anyhow!("...")` for ad-hoc errors. Don't use anyhow in libraries — it forces a dependency on your downstream.