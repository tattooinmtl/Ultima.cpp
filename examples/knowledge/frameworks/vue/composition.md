---
id: vue-composition
title: Vue Composition API
type: howto
tags: [vue, frontend, intermediate]
links: []
---
# Composition API

`const count = ref(0)`; mutate via `count.value`. `computed(() => ...)` for derived state. `watch(count, (new, old) => ...)` for side effects. The `<script setup>` macro is the idiomatic form. Composables (functions starting with `use`) replace mixins.