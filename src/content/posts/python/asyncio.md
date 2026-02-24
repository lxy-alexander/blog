---
title: "asyncio"
published: 2026-02-18
description: "asyncio"
image: ""
tags: ["python","asyncio"]
category: python
draft: false
lang: ""
---



## **async def**

Defines a coroutine function that returns an awaitable object instead of running immediately.

```python
async def stream_response():
    ...
```

------



## **await**

Suspends the current coroutine until the awaited operation completes, without blocking the event loop.

```python
await stream_response()
await asyncio.sleep(0.5)
```

------



## **event loop (`asyncio.run`)**

Schedules and switches between coroutines, enabling concurrency in a single thread.

```python
asyncio.run(main())
```

------



## **async for**

Iterates over an asynchronous iterator, receiving data incrementally as it becomes available.

```python
async for output in engine.generate(...):
    print(output)
```
