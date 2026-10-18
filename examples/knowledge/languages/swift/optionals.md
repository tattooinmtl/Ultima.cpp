---
id: swift-optionals
title: Swift Optionals
type: reference
tags: [swift, typesystem, beginner]
links: []
---
# Optionals

`var x: Int?` — either an Int or nil. `if let x = x { ... }` to unwrap. `??` for default. `guard let x = x else { return }` for early-exit. Force-unwrap (`x!`) only when you've proved it's not nil — usually a code smell.