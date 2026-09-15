---
id: pattern-html-game
title: HTML Game Builder
type: pattern
tags: [game, web, intermediate]
links: []
---
# HTML Games

A simple game loop: `requestAnimationFrame` → update state → render to canvas. Keep state in a plain object. For input: keyboard events, mouse events, touch events. For physics: simple Verlet or AABB. For audio: WebAudio. Canvas2D is plenty for 2D; switch to WebGL only for shader-driven work.