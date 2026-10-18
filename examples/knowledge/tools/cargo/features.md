---
id: cargo-features
title: Cargo Features
type: howto
tags: [cargo, intermediate]
links: []
---
# Features

`[features] default = ["std"]; serde = ["dep:serde"];`. Features are additive — every consumer of your crate can opt in independently. Use `dep:` prefix in features to enable optional deps only when the feature is on. Avoid breaking the default feature.