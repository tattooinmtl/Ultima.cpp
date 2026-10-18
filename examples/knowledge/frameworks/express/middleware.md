---
id: express-middleware
title: Express Middleware
type: howto
tags: [express, node, http]
links: []
---
# Express Middleware

`app.use((req, res, next) => { ...; next() })`. Order matters: middleware runs in registration order. `app.use(express.json())` parses JSON bodies. Error handlers have 4 args: `(err, req, res, next)`. For TypeScript, augment the `Request` type via module declaration.