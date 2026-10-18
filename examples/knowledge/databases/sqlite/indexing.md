---
id: db-sqlite-indexing
title: SQLite Indexing
type: reference
tags: [sqlite, performance]
links: []
---
# SQLite Indexes

`CREATE INDEX idx_email ON users(email);`. Use `WITHOUT ROWID` for composite-key tables. `ANALYZE` to update planner stats. Covering index: include all queried columns. WAL mode (`PRAGMA journal_mode=WAL`) for concurrent readers.