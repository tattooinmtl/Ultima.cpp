---
id: java-streams
title: Java Streams
type: reference
tags: [java, collections, intermediate]
links: []
---
# Streams

`list.stream().filter(x -> x > 0).map(x -> x * 2).collect(toList())`. Streams are lazy — the pipeline runs only when a terminal operation is called. Use parallel streams for CPU-bound bulk work, but be aware of ordering and side-effect pitfalls.