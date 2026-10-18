---
id: kotlin-coroutines
title: Kotlin Coroutines
type: howto
tags: [kotlin, async, intermediate]
links: []
---
# Coroutines

`launch { ... }` in a `CoroutineScope`. `suspend fun f() { ... }`. `delay(1000)` instead of `Thread.sleep`. Structured concurrency: a parent scope cancels all its children. Use `viewModelScope` on Android, `lifecycleScope` in Activities.