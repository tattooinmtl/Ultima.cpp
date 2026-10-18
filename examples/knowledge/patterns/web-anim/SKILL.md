---
name: web-anim
command: /anim
description: Web animations — CSS transitions/keyframes, Web Animations API, scroll-linked animation. Use for hover effects, page transitions, motion UI, or reveal-on-scroll.
tags: [animation, css, motion]
---

# web-anim

CSS-first. Reach for JS (Web Animations API) only when you need scrubbing, cancellation, or synchronized sequencing.

## When to use

- Hover/focus states, page transitions, loading skeletons, reveal-on-scroll, micro-interactions.

## CSS transition (single trigger)

```css
.btn { background: #66d9ef; transform: scale(1); transition: transform .18s ease-out; }
.btn:hover { transform: scale(1.05); }
```

## CSS keyframes (looping)

```css
@keyframes pulse { 0%, 100% { opacity: 1 } 50% { opacity: .5 } }
.badge { animation: pulse 1.6s ease-in-out infinite; }
```

## Web Animations API (scriptable)

```js
const anim = el.animate(
  [{ transform: 'translateY(0)' }, { transform: 'translateY(-10px)' }],
  { duration: 400, easing: 'ease-out', fill: 'forwards' }
);
anim.finished.then(() => el.classList.add('done'));
```

## Scroll-linked (native)

```css
@keyframes fade-in { from { opacity: 0 } to { opacity: 1 } }
.reveal {
  animation: fade-in linear both;
  animation-timeline: view();
  animation-range: entry 0% cover 40%;
}
```

Falls back to IntersectionObserver in older browsers.

## Patterns

- **prefers-reduced-motion**: wrap every non-essential animation in `@media (prefers-reduced-motion: no-preference) { … }` — accessibility win.
- **Compositing**: transform/opacity are GPU-cheap; layout-affecting props (top/left/width) are expensive. Use `transform: translate()` instead of `top`.
- **Sequenced reveals**: stagger by index with `animation-delay: calc(var(--i) * 60ms)` and set `--i` per element.
- **Spring-ish easing**: `cubic-bezier(.34, 1.56, .64, 1)` for a small overshoot bounce.

## Avoid

- GSAP unless the user asks. The Web Animations API + CSS covers ~95% of cases at zero deps.
- Animating `box-shadow` — layout thrash. Use a pseudo-element with transform instead.
- `setTimeout` chains for sequenced motion — use `animation-delay` or `anim.finished.then(...)`.
