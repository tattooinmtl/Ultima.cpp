---
id: db-mongo-aggregation
title: MongoDB Aggregation
type: howto
tags: [mongodb, intermediate]
links: []
---
# Aggregation

`db.orders.aggregate([{$match: {status: "paid"}}, {$group: {_id: "$userId", total: {$sum: "$amount"}}}])`. Stages: `$match`, `$group`, `$project`, `$sort`, `$lookup` (LEFT JOIN equivalent), `$unwind`. Index the fields you `$match` on.