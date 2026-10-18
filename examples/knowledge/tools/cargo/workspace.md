---
id: cargo-workspace
title: Cargo Workspaces
type: howto
tags: [cargo, rust, monorepo]
links: []
---
# Cargo Workspaces

A workspace is a set of crates that share a `Cargo.lock` and an output directory. Add `members = ["crates/foo"]` to the root `Cargo.toml` to opt in. Use `cargo workspace info` to inspect; `cargo run -p <name>` to target a member.