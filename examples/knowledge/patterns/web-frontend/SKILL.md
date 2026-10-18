---
name: web-frontend
command: /web
description: Modern HTML5 semantics, CSS grid/flex/animations, and vanilla ES2023 JavaScript. Use for landing pages, forms, dashboards, or anything browser-native without a framework.
tags: [html, css, javascript, frontend, web]
---

# web-frontend

Vanilla web dev — no framework. Modern browsers (2024+), no polyfills unless the user asks.

## When to use

- The user asks for a landing page, form, form validation, a static site, a widget, or "some HTML".
- The user's project has no `package.json` or has one but the deps list is short (vanilla).

## Boilerplate to start from

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Project title</title>
  <style>
    :root { --bg:#0e0f13; --fg:#e6e8ee; --accent:#66d9ef; }
    * { box-sizing: border-box; }
    html,body { margin:0; height:100%; background:var(--bg); color:var(--fg);
                 font-family: system-ui, -apple-system, sans-serif; }
    main { max-width: 960px; margin: 0 auto; padding: 24px; }
  </style>
</head>
<body>
  <main>
    <h1>Hello</h1>
  </main>
  <script type="module">
    // Entry point — top-level await is fine in module scripts.
  </script>
</body>
</html>
```

## Patterns worth reaching for

- **Grid layouts**: `display:grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap:16px;` — one line responsive card grid.
- **Aspect ratios**: `aspect-ratio: 16/9` (no more padding-bottom hack).
- **Form validation**: use `required`, `pattern`, `type="email"` etc. — no JS needed for basic UX. Add JS only for cross-field rules.
- **Dark mode**: `@media (prefers-color-scheme: dark)` on a `:root` custom-property block — the whole page swaps colors from one place.
- **Fetch + async**: `const r = await fetch(url); if (!r.ok) throw new Error(...); const j = await r.json();` — no jQuery patterns.

## Avoid

- jQuery, moment.js, lodash, bootstrap. All modern browser APIs cover their v2010 use cases.
- Inline event handlers (`onclick="..."`). Bind via `.addEventListener` in a module script.
- CSS resets — the browser defaults are fine after `box-sizing: border-box` and a body margin zero.
