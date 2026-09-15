---
id: pattern-code-review
title: Code Review Checklist
type: pattern
tags: [code-review, process]
links: []
---
# Code Review

1. Does it do what the PR description says? 2. Is the design right (no premature abstraction, no leaky)? 3. Tests cover the new logic. 4. No new lint / clippy warnings. 5. Naming is honest. 6. Error paths handled. 7. Logs are useful, not noisy. 8. The diff is small enough to review.