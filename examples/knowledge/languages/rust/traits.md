---
id: rust-traits
title: Traits
type: reference
tags: [rust, typesystem, intermediate]
links: []
---
# Traits

Traits describe shared behaviour. `impl Trait for Type` provides an implementation; `trait Trait: Bound` declares supertrait bounds. Trait objects (`dyn Trait`) enable dynamic dispatch when the concrete type is not known at compile time.