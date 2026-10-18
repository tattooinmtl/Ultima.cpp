---
id: rust-ownership
title: Rust Ownership
type: reference
tags: [rust, memory, beginner]
links: [rust-borrow-checker]
---
# Rust Ownership

Every value in Rust has a single owner. When the owner goes out of scope, the value is dropped. This eliminates the need for a garbage collector and prevents a whole class of bugs at compile time.