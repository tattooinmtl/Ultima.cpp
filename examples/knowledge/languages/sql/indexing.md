---
id: sql-indexing
title: SQL Indexes
type: reference
tags: [sql, performance, intermediate]
links: []
---
# Indexes

B-tree indexes (default) speed up `=` and range queries. Composite indexes follow leftmost-prefix. `EXPLAIN ANALYZE` to see the planner's plan. Partial indexes (`WHERE deleted_at IS NULL`) are tiny. Don't index columns with low cardinality on huge tables.