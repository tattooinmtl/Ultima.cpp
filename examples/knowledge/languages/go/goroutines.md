---
id: go-goroutines
title: Goroutines and Channels
type: reference
tags: [go, concurrency, intermediate]
links: []
---
# Goroutines

`go f()` runs `f` concurrently. Channels (`ch := make(chan T)`) coordinate. Close from the sender. Use `select { case <-ch1: ...; case <-ch2: ... }` for multiplexed waiting.