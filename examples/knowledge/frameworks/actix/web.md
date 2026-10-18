---
id: actix-web
title: Actix-web
type: howto
tags: [actix, rust, http]
links: []
---
# Actix-web

`#[get("/")] async fn index() -> impl Responder { ... }`. Register with `.service(index)`. Use `web::Data<T>` for shared state. `web::Json<T>` for typed JSON. Extractors in handler signatures do the heavy lifting.