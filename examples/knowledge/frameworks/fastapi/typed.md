---
id: fastapi-typed
title: FastAPI Type-Driven
type: howto
tags: [fastapi, python, http]
links: []
---
# FastAPI

`@app.get("/items/{id}") def get(id: int) -> Item: ...`. Pydantic models for request/response. Path/query/body parameters are auto-extracted from the function signature. Auto-generated OpenAPI at `/docs`. Use `Depends` for DI.