---
id: csharp-async
title: C# async/await
type: howto
tags: [csharp, async, intermediate]
links: []
---
# async/await

`async Task<T> f() { ... }`. Awaiting an incomplete Task yields control. Avoid `async void` (use only for event handlers). Use `Task.WhenAll` for parallel fan-out. Be careful with `ConfigureAwait(false)` in libraries.