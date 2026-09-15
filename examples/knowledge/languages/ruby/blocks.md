---
id: ruby-blocks
title: Ruby Blocks, Procs, Lambdas
type: reference
tags: [ruby, closures, intermediate]
links: []
---
# Blocks

`[1,2,3].each { |x| puts x }` — the block is implicit. Use `yield` inside the method to invoke it. `Proc.new { ... }` and `lambda { ... }` create explicit closures. Lambdas check argument count; Procs don't.