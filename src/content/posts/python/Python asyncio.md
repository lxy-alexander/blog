---
title: "Python asyncio"
published: 2026-03-08
description: "Python asyncio"
image: ""
tags: ["python","Python asyncio"]
category: python
draft: false
lang: ""
---


# **I. Python `asyncio` — Principles & Core Mechanics**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
<strong>Overview:</strong> <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">asyncio</code> is Python's <strong>asynchronous concurrency framework (异步并发框架)</strong> that uses an <strong>Event Loop (事件循环)</strong>, <strong>Coroutines (协程)</strong>, and <strong>non-blocking I/O (非阻塞 I/O)</strong> to efficiently handle many I/O-bound tasks within a single thread. The core principle: when a coroutine reaches an <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">await</code>, it <em>yields control</em> back to the event loop, which uses OS-level I/O multiplexing to resume the coroutine once the I/O operation is ready.
</div>

---

## 1. How `asyncio` Works — The Big Picture

The three pillars of asyncio:

- <span style="color:#E8600A;font-weight:700">Event Loop (事件循环)</span> — schedules and runs coroutines, monitors I/O events, resumes tasks when they are ready
- <span style="color:#E8600A;font-weight:700">Coroutines (协程)</span> — define asynchronous tasks using `async/await`; can pause and resume execution
- <span style="color:#E8600A;font-weight:700">Non-blocking I/O (非阻塞 I/O)</span> — allows the program to perform other work while waiting for I/O operations to complete

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> The principle of asyncio is: use an event loop to schedule coroutines (使用事件循环调度协程). When a coroutine reaches <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">await</code>, it <strong>yields control (让出执行权)</strong> back to the event loop, which then uses non-blocking I/O and OS I/O multiplexing (I/O 多路复用) to <strong>resume the coroutine (恢复协程执行)</strong> once the I/O operation is ready.</div>

---

## 2. Event Loop (事件循环)

### 1) What the Event Loop Does

The <span style="color:#E8600A;font-weight:700">Event Loop</span> is the <span style="color:#2980B9">core scheduler (核心调度器)</span> of asyncio. It is responsible for:

- <span style="color:#2980B9">Running</span> coroutines and tasks
- <span style="color:#2980B9">Monitoring</span> I/O events (sockets, file descriptors, timers)
- <span style="color:#2980B9">Resuming</span> coroutines when their awaited operation completes

### 2) Starting the Event Loop

The standard way is via <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">asyncio.run()</code>:

```python
import asyncio

async def main():
    print("hello")

asyncio.run(main())
```

What happens here:

- <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">asyncio.run()</code> <span style="color:#2980B9">creates and starts</span> the event loop
- The event loop <span style="color:#2980B9">executes</span> the <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">main()</code> coroutine to completion
- The event loop is <span style="color:#2980B9">closed</span> when the coroutine returns

---

## 3. Coroutines (协程)

### 1) What Is a Coroutine?

A <span style="color:#E8600A;font-weight:700">Coroutine</span> is a special function that can <span style="color:#2980B9">pause and resume execution</span>. When it encounters an I/O wait, it pauses and returns control to the event loop, allowing other tasks to run in the meantime.

### 2) How to Define and Use Coroutines

Define with <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">async def</code>, await with <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">await</code>:

```python
import asyncio

async def task():
    await asyncio.sleep(1)
    print("done")

asyncio.run(task())
```

What happens here:

- <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">async def</code> <span style="color:#2980B9">defines</span> the coroutine
- <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">await</code> <span style="color:#2980B9">pauses</span> the coroutine until the operation completes, yielding control back to the event loop

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Calling an <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">async def</code> function does <strong>not</strong> execute it — it returns a <strong>Coroutine Object (协程对象)</strong>. The coroutine only runs when it is <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">await</code>-ed or wrapped in a Task via <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">asyncio.create_task()</code>.</div>

---

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> asyncio achieves concurrency on a <strong>single thread</strong> by having the <strong>Event Loop (事件循环)</strong> continuously schedule <strong>Coroutines (协程)</strong> — each coroutine runs until it hits an <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">await</code>, yields control, and is resumed by the event loop once its I/O is ready.</div>



# **II. Python `asyncio` — Complete API Reference & Usage Scenarios**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <strong>Overview:</strong> This note is a complete API reference for Python's <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">asyncio</code> library, organized by category. Every interface is paired with a real-world usage scenario so you can immediately see <em>when</em> and <em>why</em> to use it. The library is built on a single-threaded <strong>Event Loop (事件循环)</strong> that schedules <strong>Coroutines (协程)</strong> cooperatively, making it ideal for <strong>I/O-bound (I/O 密集型)</strong> workloads. </div>

------

## 1. Entry Points — Running Coroutines

### 1) `asyncio.run(coro)`

<span style="color:#2980B9">The top-level entry point</span> for running an async program. Creates a new Event Loop (事件循环), runs the coroutine to completion, then closes the loop.

```python
import asyncio

async def main():
    print("Hello from asyncio!")

asyncio.run(main())
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">Never call <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">asyncio.run()</code> inside an already-running event loop</span> (e.g., inside Jupyter Notebook). Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">await coro</code> directly instead, or apply <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">nest_asyncio</code>.</div>

**Scenario:** Entry point of any standalone async application — CLI tools, scripts, servers.

------

### 2) `asyncio.get_event_loop()` / `asyncio.get_running_loop()`

```python
async def main():
    loop = asyncio.get_running_loop()   # Preferred inside async context
    print(loop)

loop = asyncio.get_event_loop()         # Can be used outside async context
```

| API                                                          | When to Use                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">get_running_loop()</code> | Inside a coroutine — raises `RuntimeError` if no loop is running |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">get_event_loop()</code> | Outside a coroutine — may create a new loop if none exists   |

------

## 2. Coroutines & Tasks (协程与任务)

### 1) `async def` / `await` — Defining and Awaiting Coroutines

```python
async def fetch(url: str) -> str:
    await asyncio.sleep(1)      # Yield control to event loop
    return f"data from {url}"

async def main():
    result = await fetch("https://api.example.com")
    print(result)
```

**Scenario:** Any function that performs I/O — HTTP requests, DB queries, file reads.

------

### 2) `asyncio.create_task(coro)` — Schedule Concurrently

<span style="color:#E8600A;font-weight:700">Wraps a coroutine into a Task (任务)</span> and schedules it to run on the current event loop immediately — without blocking the caller.

```python
async def worker(name: str, delay: float):
    await asyncio.sleep(delay)
    print(f"{name} done")

async def main():
    t1 = asyncio.create_task(worker("A", 2.0))
    t2 = asyncio.create_task(worker("B", 1.0))
    await t1
    await t2
    # Total time ≈ 2s, not 3s
```

**Scenario:** Fire multiple independent I/O operations simultaneously (parallel API calls, parallel DB queries).

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">A task that is created but never awaited will still run, but any exception it raises will be silently discarded.</span> Always await your tasks or attach a <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">add_done_callback</code>.</div>

------

### 3) `asyncio.Task` Methods

```python
async def main():
    task = asyncio.create_task(worker("A", 5.0))

    task.cancel()                    # Request cancellation
    print(task.done())               # True if finished/cancelled/errored
    print(task.cancelled())          # True if cancelled
    print(task.result())             # Returns result (raises if not done)
    task.add_done_callback(lambda t: print("finished:", t))
```

| Method                                                       | Purpose                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cancel()</code> | Request cancellation — injects `CancelledError` at next `await` |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">done()</code> | True if completed, cancelled, or raised an exception         |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">result()</code> | Returns the return value, or re-raises the exception         |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">exception()</code> | Returns the exception if one was raised, else `None`         |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">add_done_callback(fn)</code> | Register a callback to run when the task finishes            |

------

## 3. Concurrency Helpers (并发工具)

### 1) `asyncio.gather(*coros, return_exceptions=False)`

Runs multiple awaitables (可等待对象) <span style="color:#2980B9">concurrently</span>, returns a list of results in the **same order** as input.

```python
async def main():
    results = await asyncio.gather(
        fetch("url1"),
        fetch("url2"),
        fetch("url3"),
    )
    print(results)   # ["data from url1", "data from url2", "data from url3"]
```

With exception handling:

```python
results = await asyncio.gather(
    fetch("url1"),
    failing_fetch(),
    return_exceptions=True   # Exceptions returned as values, not raised
)
for r in results:
    if isinstance(r, Exception):
        print(f"Error: {r}")
```

**Scenario:** Batch HTTP requests, parallel DB lookups, loading multiple config files simultaneously.

------

### 2) `asyncio.wait(tasks, return_when=...)`

Returns two sets: <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">done</code> and <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">pending</code>. Gives fine-grained control over when to stop waiting.

```python
async def main():
    tasks = {asyncio.create_task(fetch(url)) for url in urls}

    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()
```

| `return_when`                                                | Behavior                            |
| ------------------------------------------------------------ | ----------------------------------- |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">ALL_COMPLETED</code> | Wait for all tasks (default)        |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">FIRST_COMPLETED</code> | Return as soon as any task finishes |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">FIRST_EXCEPTION</code> | Return as soon as any task raises   |

**Scenario:** Race condition (竞态) — take the first successful result and cancel the rest (e.g., querying multiple replicas, use whichever responds first).

------

### 3) `asyncio.as_completed(coros)`

Yields tasks <span style="color:#2980B9">in completion order</span> (not submission order).

```python
async def main():
    coros = [fetch(url) for url in urls]
    for future in asyncio.as_completed(coros):
        result = await future
        print(f"Got: {result}")   # Processed as each one finishes
```

**Scenario:** Show results to the user as they arrive, without waiting for the slowest request.

------

### 4) `asyncio.TaskGroup` (Python 3.11+) — Structured Concurrency (结构化并发)

If **any** task raises, all remaining tasks are <span style="color:#E8600A;font-weight:700">automatically cancelled</span>.

```python
async def main():
    async with asyncio.TaskGroup() as tg:
        t1 = tg.create_task(fetch("url1"))
        t2 = tg.create_task(fetch("url2"))
    # All done here — or all cancelled if one failed
    print(t1.result(), t2.result())
```

**Scenario:** Any workflow where subtasks are all required — if one fails, the whole group should abort (e.g., a multi-step pipeline).

------

## 4. Timeouts & Cancellation (超时与取消)

### 1) `asyncio.wait_for(coro, timeout)`

```python
async def main():
    try:
        result = await asyncio.wait_for(fetch("url"), timeout=3.0)
    except asyncio.TimeoutError:
        print("Request timed out after 3s")
```

**Scenario:** Any network call that must complete within a deadline (SLA enforcement, user-facing APIs).

------

### 2) `asyncio.timeout(seconds)` (Python 3.11+)

A context-manager (上下文管理器) version of timeout — more composable than `wait_for`.

```python
async def main():
    try:
        async with asyncio.timeout(5.0):
            result = await fetch("url")
            await process(result)
    except TimeoutError:
        print("Entire block timed out")
```

**Scenario:** Apply a single deadline across multiple awaits inside a block.

------

### 3) `asyncio.shield(coro)` — Protect from Cancellation

Prevents the inner coroutine from being cancelled when the outer task is cancelled.

```python
async def important_cleanup():
    await asyncio.sleep(1)
    print("Cleanup done")

async def main():
    task = asyncio.create_task(important_cleanup())
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        print("Outer cancelled, but cleanup still runs!")
        await task   # Wait for it to actually finish
```

**Scenario:** Protect a critical cleanup/commit operation from being interrupted by a cancellation signal.

------

### 4) Handling `CancelledError`

```python
async def worker():
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        print("Cleaning up before cancel...")
        await do_cleanup()
        raise   # Always re-raise!
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">Always re-raise <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">CancelledError</code> after cleanup.</span> Swallowing it breaks the cancellation chain. In Python 3.8+, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">CancelledError</code> is a subclass of <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">BaseException</code>, not <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Exception</code>.</div>

------

## 5. Synchronization Primitives (同步原语)

### 1) `asyncio.Lock` — Mutual Exclusion (互斥锁)

```python
lock = asyncio.Lock()

async def safe_write(db, data):
    async with lock:
        await db.write(data)   # Only one coroutine at a time
```

**Scenario:** Protecting shared in-memory state (counters, caches, connection pools) from concurrent modification.

------

### 2) `asyncio.Semaphore` — Concurrency Limiter (并发限制)

```python
sem = asyncio.Semaphore(10)   # Max 10 concurrent requests

async def rate_limited_fetch(session, url):
    async with sem:
        return await session.get(url)

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [rate_limited_fetch(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
```

**Scenario:** Rate-limiting API calls, capping DB connection count, controlling concurrent file handles.

------

### 3) `asyncio.BoundedSemaphore`

Same as `Semaphore` but raises `ValueError` if `release()` is called more times than `acquire()`.

**Scenario:** Safety-critical resource pools where over-releasing would be a bug.

------

### 4) `asyncio.Event` — Signal Between Coroutines (协程间信号)

```python
event = asyncio.Event()

async def producer():
    await asyncio.sleep(2)
    print("Data ready")
    event.set()            # Signal the consumer

async def consumer():
    await event.wait()     # Block until set
    print("Processing data")

async def main():
    await asyncio.gather(producer(), consumer())
```

**Scenario:** One-shot notification — signal consumers when data/resource becomes available.

------

### 5) `asyncio.Condition` — Wait + Notify Pattern

```python
condition = asyncio.Condition()
buffer = []

async def producer():
    async with condition:
        buffer.append("item")
        condition.notify_all()   # Wake all waiting consumers

async def consumer():
    async with condition:
        await condition.wait_for(lambda: len(buffer) > 0)
        item = buffer.pop()
```

**Scenario:** Multiple consumers waiting on a shared resource to reach a specific state.

------

### 6) `asyncio.Queue` — Producer-Consumer (生产者-消费者)

```python
async def producer(q: asyncio.Queue):
    for i in range(10):
        await q.put(i)
    await q.put(None)   # Sentinel

async def consumer(q: asyncio.Queue):
    while True:
        item = await q.get()
        if item is None:
            break
        await process(item)
        q.task_done()

async def main():
    q = asyncio.Queue(maxsize=5)
    await asyncio.gather(producer(q), consumer(q))
```

| Queue Type                                                   | Behavior                                |
| ------------------------------------------------------------ | --------------------------------------- |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Queue</code> | FIFO (先进先出)                         |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">LifoQueue</code> | LIFO / stack (后进先出)                 |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">PriorityQueue</code> | Smallest item dequeued first (优先队列) |

**Scenario:** Pipeline architectures — web crawlers, log processors, streaming data ingestion.

------

## 6. Async Context Managers & Iterators (异步上下文管理器与迭代器)

### 1) `async with` — Async Context Manager (异步上下文管理器)

Implement <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">**aenter**</code> and <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">**aexit**</code>:

```python
class AsyncDB:
    async def __aenter__(self):
        self.conn = await connect_db()
        return self.conn

    async def __aexit__(self, *args):
        await self.conn.close()

async def main():
    async with AsyncDB() as conn:
        result = await conn.query("SELECT 1")
```

**Scenario:** Any resource requiring async setup/teardown — DB connections, HTTP sessions, file handles, locks.

------

### 2) `@asynccontextmanager` — Decorator Shortcut

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def managed_connection():
    conn = await connect_db()
    try:
        yield conn
    finally:
        await conn.close()

async def main():
    async with managed_connection() as conn:
        await conn.query("SELECT 1")
```

**Scenario:** Simpler alternative to writing a full class when you need a one-off async context manager.

------

### 3) `async for` — Async Iterator (异步迭代器)

Implement <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">**aiter**</code> and <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">**anext**</code>, or use an async generator:

```python
async def paginated_api(base_url: str):
    page = 1
    while True:
        data = await fetch(f"{base_url}?page={page}")
        if not data:
            break
        yield data
        page += 1

async def main():
    async for page in paginated_api("https://api.example.com/items"):
        await process(page)
```

**Scenario:** Paginated APIs, streaming database cursors, real-time event streams (WebSocket, SSE).

------

## 7. Running Blocking Code (在异步中运行阻塞代码)

### 1) `asyncio.to_thread(func, *args)` (Python 3.9+)

```python
import time

async def main():
    # Run blocking I/O in a thread without freezing the event loop
    result = await asyncio.to_thread(time.sleep, 2)
```

**Scenario:** Legacy blocking libraries (e.g., `requests`, `psycopg2`, `time.sleep`), file system operations.

------

### 2) `loop.run_in_executor(executor, func, *args)`

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

async def main():
    loop = asyncio.get_running_loop()

    # Thread pool — for blocking I/O
    result = await loop.run_in_executor(None, blocking_io_func, arg)

    # Process pool — for CPU-bound work
    with ProcessPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, cpu_bound_func, arg)
```

| Executor                                                     | Use Case                       |
| ------------------------------------------------------------ | ------------------------------ |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">None</code> (default ThreadPool) | Blocking I/O, legacy libraries |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">ThreadPoolExecutor</code> | Explicit thread pool sizing    |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">ProcessPoolExecutor</code> | CPU-bound tasks (bypasses GIL) |

**Scenario:** Image processing, ML inference on CPU, compression, encryption — any heavy computation alongside async I/O.

------

## 8. Streams — High-Level Network I/O (高层网络 I/O)

### 1) `asyncio.open_connection(host, port)` — TCP Client

```python
async def tcp_client():
    reader, writer = await asyncio.open_connection("127.0.0.1", 8888)

    writer.write(b"Hello\n")
    await writer.drain()

    data = await reader.readline()
    print(f"Received: {data.decode()}")

    writer.close()
    await writer.wait_closed()
```

**Scenario:** Custom TCP clients — talking to Redis, custom protocols, game servers.

------

### 2) `asyncio.start_server(handler, host, port)` — TCP Server

```python
async def handle_client(reader, writer):
    data = await reader.read(1024)
    writer.write(data)         # Echo back
    await writer.drain()
    writer.close()

async def main():
    server = await asyncio.start_server(handle_client, "127.0.0.1", 8888)
    async with server:
        await server.serve_forever()
```

**Scenario:** Building lightweight TCP/protocol servers (chat, telnet, custom RPC).

------

## 9. Subprocesses (异步子进程)

### 1) `asyncio.create_subprocess_exec()` / `asyncio.create_subprocess_shell()`

```python
async def run_command():
    proc = await asyncio.create_subprocess_exec(
        "ls", "-la",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    print(stdout.decode())

async def run_shell():
    proc = await asyncio.create_subprocess_shell(
        "echo hello && sleep 1 && echo world",
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    print(stdout.decode())
```

| API                                                          | Use Case                                               |
| ------------------------------------------------------------ | ------------------------------------------------------ |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">create_subprocess_exec</code> | Safe — no shell injection, explicit args               |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">create_subprocess_shell</code> | Convenient — supports pipes/redirects, shell expansion |

**Scenario:** Running external tools (ffmpeg, git, compilers) without blocking the event loop.

------

## 10. Utilities & Introspection (工具与内省)

### 1) `asyncio.sleep(delay, result=None)`

```python
async def main():
    await asyncio.sleep(0)    # Yield control without waiting (common pattern)
    await asyncio.sleep(1.5)  # Wait 1.5 seconds
    val = await asyncio.sleep(2, result="done")  # Returns result after delay
    print(val)   # "done"
```

**Scenario:** `sleep(0)` is used to yield control voluntarily in tight loops, preventing event loop starvation (事件循环饥饿).

------

### 2) `asyncio.current_task()` / `asyncio.all_tasks()`

```python
async def main():
    me = asyncio.current_task()
    me.set_name("main-task")

    all_running = asyncio.all_tasks()
    print(f"Running tasks: {len(all_running)}")
```

**Scenario:** Debugging, logging task names, graceful shutdown (cancel all tasks on SIGINT).

------

### 3) `asyncio.ensure_future(coro_or_future)`

Schedules a coroutine or wraps a Future (期约) into a Task. Largely superseded by `create_task()` in modern code.

```python
task = asyncio.ensure_future(my_coro())   # Legacy — prefer create_task()
```

------

### 4) `asyncio.wrap_future(future)` — Bridge with `concurrent.futures`

Wraps a <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">concurrent.futures.Future</code> into an asyncio-compatible <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">asyncio.Future</code>.

```python
import concurrent.futures

def blocking():
    return 42

async def main():
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(blocking)
        result = await asyncio.wrap_future(future)
        print(result)   # 42
```

**Scenario:** Integrating existing `concurrent.futures`-based code into an asyncio application.

------

### 5) `asyncio.isfuture()` / `asyncio.iscoroutine()` / `asyncio.iscoroutinefunction()`

```python
import asyncio

async def my_coro(): pass

print(asyncio.iscoroutinefunction(my_coro))   # True
print(asyncio.iscoroutine(my_coro()))         # True
print(asyncio.isfuture(asyncio.Future()))     # True
```

**Scenario:** Writing framework code or decorators that need to handle both sync and async callables.

------

## 11. Low-level Event Loop APIs (底层事件循环接口)

### 1) `loop.call_soon(callback, *args)` / `loop.call_later(delay, callback)`

Schedule a plain (non-coroutine) callback:

```python
loop = asyncio.get_event_loop()
loop.call_soon(print, "scheduled immediately")
loop.call_later(2.0, print, "scheduled in 2s")
loop.call_at(loop.time() + 5.0, print, "scheduled at absolute time")
```

**Scenario:** Integrating callback-based legacy code into an asyncio event loop.

------

### 2) `loop.add_reader(fd, callback)` / `loop.add_writer(fd, callback)`

Register a callback to fire when a file descriptor (文件描述符) becomes readable/writable.

```python
loop.add_reader(sock.fileno(), on_data_received)
loop.remove_reader(sock.fileno())
```

**Scenario:** Building low-level protocol handlers — custom socket management, raw I/O multiplexing.

------

### 3) `asyncio.Protocol` / `asyncio.DatagramProtocol`

The low-level <span style="color:#2980B9">callback-based protocol interface</span>, underlying `StreamReader`/`StreamWriter`.

```python
class EchoProtocol(asyncio.Protocol):
    def connection_made(self, transport):
        self.transport = transport

    def data_received(self, data: bytes):
        self.transport.write(data)   # Echo

    def connection_lost(self, exc):
        print("Connection closed")

async def main():
    loop = asyncio.get_running_loop()
    server = await loop.create_server(EchoProtocol, "127.0.0.1", 8888)
    async with server:
        await server.serve_forever()
```

**Scenario:** High-performance servers where the overhead of `StreamReader`/`StreamWriter` is unacceptable, or when implementing a custom protocol (e.g., custom binary framing).

------

## 12. Graceful Shutdown Pattern (优雅关闭模式)

```python
import asyncio
import signal

async def main():
    loop = asyncio.get_running_loop()

    stop = loop.create_future()
    loop.add_signal_handler(signal.SIGINT, stop.set_result, None)
    loop.add_signal_handler(signal.SIGTERM, stop.set_result, None)

    tasks = [asyncio.create_task(worker(i)) for i in range(5)]

    await stop   # Block until SIGINT/SIGTERM

    print("Shutting down...")
    for t in tasks:
        t.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)
    print("All tasks cancelled. Bye.")

asyncio.run(main())
```

**Scenario:** Any long-running async service (web server, bot, background processor) that must clean up gracefully on `Ctrl+C` or a system signal.

------

## 13. Quick API Comparison Table

| API                                                          | Category     | Key Trait                                         |
| ------------------------------------------------------------ | ------------ | ------------------------------------------------- |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">asyncio.run()</code> | Entry point  | Creates + closes loop; top-level only             |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">create_task()</code> | Scheduling   | Non-blocking schedule; returns Task               |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">gather()</code> | Concurrency  | All results in order; short-circuits on exception |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">wait()</code> | Concurrency  | Returns done/pending sets; fine control           |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">as_completed()</code> | Concurrency  | Yields in completion order                        |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">TaskGroup</code> | Structured   | Auto-cancel on failure (3.11+)                    |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">wait_for()</code> | Timeout      | Cancels task on timeout                           |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">timeout()</code> | Timeout      | Context manager; covers multiple awaits (3.11+)   |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">shield()</code> | Cancellation | Protects inner task from outer cancel             |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Lock</code> | Sync         | Mutex; one at a time                              |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Semaphore</code> | Sync         | N at a time; rate limiting                        |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Event</code> | Sync         | One-shot signal                                   |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Condition</code> | Sync         | Wait-for-state with notify                        |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Queue</code> | Sync         | FIFO producer-consumer                            |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">to_thread()</code> | Blocking     | Offload to thread pool (3.9+)                     |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">run_in_executor()</code> | Blocking     | Thread or process pool                            |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">open_connection()</code> | Streams      | High-level TCP client                             |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">start_server()</code> | Streams      | High-level TCP server                             |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Protocol</code> | Low-level    | Callback-based; max performance                   |

------

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> Master the <strong>three tiers</strong>: use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">gather</code> / <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">TaskGroup</code> for everyday concurrency, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Lock</code> / <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Semaphore</code> / <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Queue</code> for coordination, and <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">to_thread</code> / <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">run_in_executor</code> to escape blocking code — everything else (streams, protocols, signals) builds on these foundations.</div>



