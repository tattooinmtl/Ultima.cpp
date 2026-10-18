---
name: html5-canvas
command: /canvas
description: 2D canvas games — game loop, sprite management, input, save/load. Use when the user asks for a small browser game (snake, breakout, platformer, etc.).
tags: [html5, canvas, game, 2d]
---

# html5-canvas

Small browser games in one HTML file. No frameworks. Fixed timestep game loop.

## When to use

- User says "make me a game", "snake", "breakout", "tetris", "pong", etc.
- User wants a 2D interactive canvas widget (a paint tool, a drawing app).

## Boilerplate loop

```html
<canvas id="c" width="640" height="360" style="background:#111; image-rendering: pixelated;"></canvas>
<script type="module">
  const canvas = document.getElementById('c');
  const ctx = canvas.getContext('2d');
  const KEYS = new Set();
  addEventListener('keydown', e => KEYS.add(e.key));
  addEventListener('keyup',   e => KEYS.delete(e.key));

  const state = { x: 100, y: 100, vx: 0, vy: 0 };

  const FIXED_DT = 1/60;
  let acc = 0, last = performance.now();

  function step(dt) {
    // Fixed-step update — deterministic physics.
    if (KEYS.has('ArrowLeft'))  state.vx -= 200 * dt;
    if (KEYS.has('ArrowRight')) state.vx += 200 * dt;
    state.vx *= 0.9;
    state.x  += state.vx * dt;
  }
  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#66d9ef';
    ctx.fillRect(state.x, state.y, 20, 20);
  }
  function frame(now) {
    acc += (now - last) / 1000;
    last = now;
    while (acc >= FIXED_DT) { step(FIXED_DT); acc -= FIXED_DT; }
    draw();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
</script>
```

## Patterns

- **Sprites from a spritesheet**: `ctx.drawImage(img, sx, sy, sw, sh, dx, dy, sw, sh)` — same source rect + destination.
- **Save/load state**: `localStorage.setItem('save', JSON.stringify(state))`; on load, hydrate then `Object.assign(state, JSON.parse(...))`.
- **Sound**: `new Audio('kick.wav').play()` — cheap and cheerful. For overlapping instances, clone the node per play.
- **Input**: `KEYS` set above handles held keys. For "just pressed", also maintain a `TAPS = new Set()` cleared each frame.
- **Off-canvas caching**: pre-render static art to a hidden `<canvas>` once, then `drawImage` it every frame — 10× faster than re-drawing paths.

## Avoid

- `setInterval` for the game loop — jitter, no vsync alignment.
- Re-computing sprite positions inside `draw` — that belongs in `step`.
- Trigonometry per frame for background parallax; precompute a table.
