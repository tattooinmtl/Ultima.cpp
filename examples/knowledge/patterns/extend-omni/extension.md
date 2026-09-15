---
id: pattern-extend
title: Extending the Harness
type: pattern
tags: [process, extension]
links: []
---
# Extension

The harness has a 7-crate split. To add a new tool: implement the `Tool` trait in `agnes-tools`, register it in `ToolRegistry::standard`. To add a new event: extend `HarnessEvent` in `agnes-core`, publish via `bus.publish_harness`, handle in `state.apply` and `brain.on_event`. To add a new brain widget: implement in `agnes-ui/src/brain/` and wire in `app.rs`.