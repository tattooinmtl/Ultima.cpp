---
id: django-orm
title: Django ORM
type: howto
tags: [django, python, orm]
links: []
---
# Django ORM

`User.objects.filter(email__icontains="@x.com")` for queries. `User.objects.create(...)` for inserts. `qs.select_related('profile')` to avoid N+1. `F('count') + 1` for atomic updates. Use `.iterator()` for huge querysets.