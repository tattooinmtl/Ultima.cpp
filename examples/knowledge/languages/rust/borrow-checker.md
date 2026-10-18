---
id: rust-borrow-checker
title: Rust Borrow Checker
type: reference
tags: [rust, memory, intermediate]
links: []
---
# Borrow Checker

References (`&T`) and mutable references (`&mut T`) cannot coexist. The compiler enforces this so that data races are impossible at runtime.