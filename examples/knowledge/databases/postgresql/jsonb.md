---
id: db-pg-jsonb
title: PostgreSQL JSONB
type: reference
tags: [postgresql, json, intermediate]
links: []
---
# JSONB

JSONB stores JSON in a binary, indexed-friendly form. `data->>'key'` returns text, `data->'key'` returns JSON. GIN index on the column for containment queries: `CREATE INDEX ON events USING gin(data);`. `jsonb_path_*` for SQL/JSON path expressions.