---
id: py-async
title: Python Asyncio
type: howto
tags: [python, async, intermediate]
links: []
---
# Asyncio

`async def f(): ...` for coroutines; `await g()` to wait. Run multiple with `asyncio.gather(*coros)`. CPU-bound work goes in `loop.run_in_executor`, not directly in a coroutine.