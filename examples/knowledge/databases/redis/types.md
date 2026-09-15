---
id: db-redis-types
title: Redis Data Types
type: reference
tags: [redis, beginner]
links: []
---
# Redis

`STRING` for simple values, `HASH` for object fields, `LIST` for queues, `SET` / `ZSET` for membership / ranking, `STREAM` for log-like. `EXPIRE key 60` to TTL. Use `MULTI`/`EXEC` for transactions. Watch out for `KEYS *` in production — use `SCAN`.