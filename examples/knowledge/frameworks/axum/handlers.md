---
id: axum-handlers
title: Axum Handlers
type: howto
tags: [axum, rust, http]
links: []
---
# Axum

`async fn handler(State(s): State<AppState>) -> impl IntoResponse { ... }`. Compose with `Router::new().route("/", get(handler)).with_state(state)`. Extractors: `Path`, `Query`, `Json`, `Form`. Use `tower::ServiceBuilder` for cross-cutting middleware.