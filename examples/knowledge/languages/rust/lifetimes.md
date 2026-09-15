---
id: rust-lifetimes
title: Lifetimes
type: reference
tags: [rust, lifetimes, intermediate]
links: [rust-borrow-checker]
---
# Lifetimes

Every reference in Rust has a lifetime — the region of code where the reference is valid. Most of the time the compiler infers it; sometimes you have to spell it out with `'a` syntax. The borrow checker's job is to prove the lifetimes line up.