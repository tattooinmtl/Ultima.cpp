---
id: ts-generics
title: TypeScript Generics
type: reference
tags: [typescript, typesystem, intermediate]
links: []
---
# Generics

`function identity<T>(x: T): T { return x }` — the `T` is a type parameter, inferred at the call site. Use `extends` to constrain (`<T extends { id: string }>`). Use `keyof` and `typeof` for advanced type arithmetic.