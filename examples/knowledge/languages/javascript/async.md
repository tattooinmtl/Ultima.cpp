---
id: js-async
title: JavaScript Async/Await
type: howto
tags: [javascript, async, intermediate]
links: []
---
# Async / Await

`async function f() { const x = await fetch(...); }` — the function always returns a Promise. Use `Promise.all` for parallel work. Never `await` in a `for` loop if the work could run in parallel.