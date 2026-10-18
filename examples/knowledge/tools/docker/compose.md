---
id: docker-compose
title: Docker Compose
type: howto
tags: [docker, intermediate]
links: []
---
# Compose

`docker-compose.yml` (or `compose.yaml`) describes services. `docker compose up -d` to start detached. `depends_on` for ordering, `healthcheck` for readiness, `volumes` for persistence, `networks` for isolation. `docker compose down -v` to nuke everything.