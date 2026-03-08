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


# **I. Python `asyncio` — Asynchronous Programming Complete Guide**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
<strong>Overview:</strong> <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">asyncio</code> is Python's built-in library for writing <strong>concurrent (并发)</strong> code using the <strong>async/await</strong> syntax. It is built on an <strong>Event Loop (事件循环)</strong> that schedules and runs <strong>Coroutines (协程)</strong> without creating multiple threads or processes. This makes it ideal for <strong>I/O-bound (I/O 密集型)</strong> tasks such as network requests, file I/O, and database queries — where the program spends most of its time <em>waiting</em>, not computing.
</div>

---

## 1. Core Concepts

### 1) Coroutine (协程)

A <span style="color:#E8600A;font-weight:700">Coroutine</span> is a special function defined with <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">async def</code>. Unlike a regular function, calling a coroutine does **not** execute it immediately — it returns a <strong>Coroutine Object (协程对象)</strong> that must be awaited or scheduled.

```python
import asyncio

async def greet(name: str) -> str:
    await asyncio.sleep(1)   # Simulate I/O wait (non-blocking)
    return f"Hello, {name}!"

# This does NOT run the coroutine:
coro = greet("Alice")        # → <coroutine object greet at 0x...>

# This DOES run it:
result = asyncio.run(greet("Alice"))
print(result)                # → Hello, Alice!
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">asyncio.run()</code> is the <strong>top-level entry point</strong> introduced in Python 3.7+. It creates a new Event Loop, runs the given coroutine to completion, then closes the loop. <span style="color:#C0392B;font-weight:600">Never call <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">asyncio.run()</code> inside an already-running event loop</span> (e.g., inside Jupyter Notebook — use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">await</code> directly instead).</div>

---

### 2) Event Loop (事件循环)

The <span style="color:#E8600A;font-weight:700">Event Loop</span> is the core scheduler of asyncio. It:

- <span style="color:#2980B9">Registers</span> coroutines and callbacks
- <span style="color:#2980B9">Monitors</span> I/O events (sockets, files, etc.)
- <span style="color:#2980B9">Resumes</span> suspended coroutines when their awaited task completes

```python
import asyncio

async def main():
    loop = asyncio.get_event_loop()
    print(f"Running loop: {loop}")

asyncio.run(main())
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Python's asyncio uses a <strong>single-threaded (单线程)</strong> event loop. All coroutines run on the <strong>same thread</strong> — concurrency is achieved by <em>interleaving</em> execution during <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">await</code> points, not by parallelism. This means <span style="color:#C0392B;font-weight:600">CPU-bound (CPU 密集型) tasks will block the entire loop</span> and should use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">run_in_executor</code> instead.</div>

---

### 3) `await` Keyword

<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">await</code> <span style="color:#2980B9">suspends</span> the current coroutine and <span style="color:#2980B9">yields control</span> back to the event loop until the awaited object completes. It can only be used inside an <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">async def</code> function.

**Awaitable objects (可等待对象) include:**

| Type | Description |
|---|---|
| `Coroutine` | Result of calling an `async def` function |
| `Task` | Wraps a coroutine, scheduled immediately |
| `Future` | Low-level placeholder for a result |

```python
async def fetch_data():
    print("Fetching...")
    await asyncio.sleep(2)   # Yields control here for 2 seconds
    print("Done!")
    return {"data": 42}

async def main():
    result = await fetch_data()
    print(result)

asyncio.run(main())
```

---

## 2. Tasks and Concurrency (并发)

### 1) `asyncio.create_task()` — Running Coroutines Concurrently

<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">create_task()</code> wraps a coroutine in a <span style="color:#E8600A;font-weight:700">Task (任务)</span> and schedules it to run **concurrently** on the event loop. The task starts immediately — you don't need to `await` it right away.

```python
import asyncio
import time

async def worker(name: str, delay: float):
    print(f"[{name}] Starting...")
    await asyncio.sleep(delay)
    print(f"[{name}] Done after {delay}s")
    return name

async def main():
    start = time.perf_counter()

    # Schedule both tasks — they run concurrently
    task1 = asyncio.create_task(worker("A", 2.0))
    task2 = asyncio.create_task(worker("B", 1.0))

    result1 = await task1
    result2 = await task2

    elapsed = time.perf_counter() - start
    print(f"Total time: {elapsed:.2f}s")   # ~2.0s, NOT 3.0s

asyncio.run(main())
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">Do NOT simply <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">await coroutine_a(); await coroutine_b()</code> sequentially</span> — that runs them one after another (total ≈ 3s). Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">create_task()</code> or <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">gather()</code> to achieve true concurrency.</div>

---

### 2) `asyncio.gather()` — Collecting Multiple Results

<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">asyncio.gather()</code> runs multiple awaitables <span style="color:#2980B9">concurrently</span> and returns their results as a list in the **same order** they were passed.

```python
async def main():
    results = await asyncio.gather(
        worker("A", 2.0),
        worker("B", 1.0),
        worker("C", 1.5),
    )
    print(results)   # ["A", "B", "C"] — order preserved!

asyncio.run(main())
```

**Handling exceptions in `gather`:**

```python
async def failing():
    raise ValueError("Something went wrong!")

async def main():
    # return_exceptions=True: exceptions are returned as results, not raised
    results = await asyncio.gather(
        worker("A", 1.0),
        failing(),
        return_exceptions=True
    )
    for r in results:
        if isinstance(r, Exception):
            print(f"Error: {r}")
        else:
            print(f"OK: {r}")

asyncio.run(main())
```

---

### 3) `asyncio.wait()` — Fine-grained Control

<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">asyncio.wait()</code> gives you more control — it returns two sets: <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">done</code> and <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">pending</code>.

```python
import asyncio

async def main():
    tasks = [
        asyncio.create_task(worker("A", 3.0)),
        asyncio.create_task(worker("B", 1.0)),
        asyncio.create_task(worker("C", 2.0)),
    ]

    # FIRST_COMPLETED: return as soon as any task finishes
    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_COMPLETED
    )

    print(f"Finished: {len(done)}, Still running: {len(pending)}")

    # Cancel remaining tasks
    for task in pending:
        task.cancel()

asyncio.run(main())
```

| Parameter | Options | Meaning |
|---|---|---|
| `return_when` | `ALL_COMPLETED` | Wait for all tasks (default) |
| | `FIRST_COMPLETED` | Return when any task finishes |
| | `FIRST_EXCEPTION` | Return when any task raises an exception |

---

### 4) `asyncio.as_completed()` — Process Results as They Arrive

```python
async def main():
    coros = [worker("A", 3.0), worker("B", 1.0), worker("C", 2.0)]

    # Yields tasks in completion order (B → C → A)
    for coro in asyncio.as_completed(coros):
        result = await coro
        print(f"Completed: {result}")

asyncio.run(main())
```

---

## 3. Timeouts and Cancellation (超时与取消)

### 1) `asyncio.wait_for()` — Add a Timeout

<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">asyncio.wait_for()</code> wraps a coroutine with a <span style="color:#E8600A;font-weight:700">timeout (超时)</span>. If the task doesn't finish in time, it raises <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">asyncio.TimeoutError</code> and **cancels** the task.

```python
async def slow_operation():
    await asyncio.sleep(10)
    return "finished"

async def main():
    try:
        result = await asyncio.wait_for(slow_operation(), timeout=3.0)
    except asyncio.TimeoutError:
        print("Operation timed out after 3 seconds!")

asyncio.run(main())
```

---

### 2) Task Cancellation (任务取消)

```python
async def long_task():
    try:
        print("Task running...")
        await asyncio.sleep(100)
    except asyncio.CancelledError:
        print("Task was cancelled! Cleaning up...")
        raise   # Always re-raise CancelledError!

async def main():
    task = asyncio.create_task(long_task())
    await asyncio.sleep(1)

    task.cancel()            # Send cancellation signal

    try:
        await task           # Wait for cancellation to complete
    except asyncio.CancelledError:
        print("Main: confirmed task cancelled")

asyncio.run(main())
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">Always re-raise <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">asyncio.CancelledError</code></span> after cleanup. Swallowing it prevents proper cancellation propagation. In Python 3.8+, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">CancelledError</code> is a subclass of <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">BaseException</code>, not <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Exception</code>, so bare <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">except Exception</code> won't catch it.</div>

---

## 4. Synchronization Primitives (同步原语)

### 1) `asyncio.Lock` — Mutual Exclusion (互斥锁)

Prevents multiple coroutines from accessing a shared resource simultaneously.

```python
import asyncio

lock = asyncio.Lock()
shared_counter = 0

async def increment():
    global shared_counter
    async with lock:               # Acquire lock
        temp = shared_counter
        await asyncio.sleep(0)     # Simulate context switch
        shared_counter = temp + 1  # Safe: protected by lock

async def main():
    tasks = [increment() for _ in range(1000)]
    await asyncio.gather(*tasks)
    print(f"Counter: {shared_counter}")  # Always 1000

asyncio.run(main())
```

---

### 2) `asyncio.Semaphore` — Limit Concurrency (并发限制)

A <span style="color:#E8600A;font-weight:700">Semaphore (信号量)</span> limits how many coroutines can run at the same time. Perfect for rate-limiting API calls.

```python
async def fetch_url(session, url: str, semaphore: asyncio.Semaphore):
    async with semaphore:           # Only N coroutines enter at once
        print(f"Fetching {url}")
        await asyncio.sleep(0.5)    # Simulate HTTP request
        return f"Result from {url}"

async def main():
    semaphore = asyncio.Semaphore(3)  # Max 3 concurrent requests
    urls = [f"https://api.example.com/item/{i}" for i in range(10)]

    tasks = [fetch_url(None, url, semaphore) for url in urls]
    results = await asyncio.gather(*tasks)
    print(f"Got {len(results)} results")

asyncio.run(main())
```

---

### 3) `asyncio.Queue` — Producer-Consumer Pattern (生产者-消费者模式)

```python
import asyncio
import random

async def producer(queue: asyncio.Queue, n: int):
    for i in range(n):
        item = random.randint(1, 100)
        await queue.put(item)
        print(f"Produced: {item}")
        await asyncio.sleep(0.1)
    await queue.put(None)   # Sentinel value to signal done

async def consumer(queue: asyncio.Queue):
    while True:
        item = await queue.get()
        if item is None:
            break
        print(f"Consumed: {item}")
        await asyncio.sleep(0.3)
        queue.task_done()

async def main():
    queue = asyncio.Queue(maxsize=5)
    await asyncio.gather(
        producer(queue, 10),
        consumer(queue),
    )

asyncio.run(main())
```

| Primitive | Purpose | Key Methods |
|---|---|---|
| `Lock` | Mutual exclusion | `acquire()`, `release()`, `async with` |
| `Semaphore` | Limit concurrency to N | `acquire()`, `release()`, `async with` |
| `Event` | Signal between coroutines | `set()`, `clear()`, `wait()` |
| `Condition` | Complex coordination | `wait()`, `notify()`, `notify_all()` |
| `Queue` | Producer-consumer | `put()`, `get()`, `task_done()` |

---

## 5. Async Context Managers & Iterators (异步上下文管理器与迭代器)

### 1) Async Context Manager (异步上下文管理器)

Implement <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">__aenter__</code> and <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">__aexit__</code> to use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">async with</code>.

```python
class AsyncDBConnection:
    async def __aenter__(self):
        print("Opening DB connection...")
        await asyncio.sleep(0.1)   # Simulate connection setup
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Closing DB connection...")
        await asyncio.sleep(0.05)  # Simulate cleanup

    async def query(self, sql: str):
        await asyncio.sleep(0.2)
        return f"Result of: {sql}"

async def main():
    async with AsyncDBConnection() as db:
        result = await db.query("SELECT * FROM users")
        print(result)

asyncio.run(main())
```

---

### 2) Async Generator (异步生成器)

Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">async def</code> with <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">yield</code> to create an <span style="color:#E8600A;font-weight:700">Async Generator (异步生成器)</span>. Iterate with <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">async for</code>.

```python
async def paginated_fetch(base_url: str, pages: int):
    for page in range(1, pages + 1):
        await asyncio.sleep(0.2)   # Simulate API call
        yield {"page": page, "data": f"content_{page}"}

async def main():
    async for item in paginated_fetch("https://api.example.com", 5):
        print(f"Page {item['page']}: {item['data']}")

asyncio.run(main())
```

---

## 6. Running Blocking Code (运行阻塞代码)

### 1) `loop.run_in_executor()` — Offload to Thread/Process Pool

<span style="color:#C0392B;font-weight:600">CPU-bound tasks or blocking I/O (e.g., `time.sleep`, file reads with no async support) will freeze the event loop</span>. Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">run_in_executor()</code> to run them in a separate thread or process.

```python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def blocking_io(filename: str) -> str:
    """Blocking file read — would freeze the event loop if called directly."""
    time.sleep(1)
    return f"Content of {filename}"

def cpu_intensive(n: int) -> int:
    """CPU-bound computation."""
    return sum(i * i for i in range(n))

async def main():
    loop = asyncio.get_event_loop()

    # Run blocking I/O in a thread pool (default executor)
    result = await loop.run_in_executor(None, blocking_io, "data.txt")
    print(result)

    # Run CPU-bound work in a process pool
    with ProcessPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, cpu_intensive, 10_000_000)
        print(f"Sum of squares: {result}")

asyncio.run(main())
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Python 3.9+ offers the convenience wrapper <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">asyncio.to_thread(func, *args)</code> which internally uses <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">run_in_executor</code> with the default thread pool — much cleaner for simple cases:
<br><br>
<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">result = await asyncio.to_thread(blocking_io, "data.txt")</code>
</div>

---

## 7. Real-World Patterns (实战模式)

### 1) Async HTTP Requests with `aiohttp`

```python
import asyncio
import aiohttp

async def fetch(session: aiohttp.ClientSession, url: str) -> dict:
    async with session.get(url) as response:
        return await response.json()

async def main():
    urls = [
        "https://jsonplaceholder.typicode.com/posts/1",
        "https://jsonplaceholder.typicode.com/posts/2",
        "https://jsonplaceholder.typicode.com/posts/3",
    ]

    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        results = await asyncio.gather(*tasks)

    for r in results:
        print(r["title"])

asyncio.run(main())
```

---

### 2) Async Server with `asyncio.start_server()`

```python
import asyncio

async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    addr = writer.get_extra_info("peername")
    print(f"Connection from {addr}")

    while True:
        data = await reader.read(1024)
        if not data:
            break
        message = data.decode().strip()
        print(f"Received: {message!r} from {addr}")

        response = f"Echo: {message}\n"
        writer.write(response.encode())
        await writer.drain()   # Ensure data is sent

    writer.close()
    await writer.wait_closed()
    print(f"Connection closed: {addr}")

async def main():
    server = await asyncio.start_server(handle_client, "127.0.0.1", 8888)
    print("Serving on 127.0.0.1:8888")

    async with server:
        await server.serve_forever()

asyncio.run(main())
```

---

### 3) Structured Concurrency with `asyncio.TaskGroup` (Python 3.11+)

<span style="color:#E8600A;font-weight:700">TaskGroup (任务组)</span> is the modern, structured approach. If **any** task fails, all remaining tasks are cancelled automatically.

```python
import asyncio

async def main():
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(worker("A", 1.0))
        task2 = tg.create_task(worker("B", 2.0))
        task3 = tg.create_task(worker("C", 1.5))
    # All tasks complete here — or all are cancelled if one fails
    print(task1.result(), task2.result(), task3.result())

asyncio.run(main())
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">TaskGroup</code> raises an <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">ExceptionGroup</code> (Python 3.11+) if multiple tasks fail. Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">except*</code> syntax to handle them individually. This pattern is preferred over <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">gather()</code> for new code.</div>

---

## 8. Comparison: Threading vs asyncio vs multiprocessing

| Aspect | `threading` | `asyncio` | `multiprocessing` |
|---|---|---|---|
| **Concurrency model** | OS threads | Cooperative (event loop) | Separate processes |
| **Best for** | Blocking I/O, legacy libs | Non-blocking I/O | CPU-bound tasks |
| **GIL (全局解释器锁)** | Limited by GIL | Not relevant | Bypasses GIL |
| **Memory overhead** | High (per thread) | Very low | High (per process) |
| **Complexity** | Medium | Medium | High |
| **Shared state** | Shared (with locks) | Shared (event loop) | Not shared (IPC needed) |
| **Typical use** | File I/O, DB calls | HTTP servers, web scrapers | ML training, data processing |

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> A common pattern is to combine all three: use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">asyncio</code> for the main server loop, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">run_in_executor(ThreadPool)</code> for legacy blocking I/O, and <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">run_in_executor(ProcessPool)</code> for CPU-intensive work.</div>

---

## 9. Common Pitfalls (常见陷阱)

### 1) Forgetting `await`

```python
async def main():
    # ❌ WRONG: coroutine is never awaited — silently does nothing
    asyncio.sleep(1)   # RuntimeWarning: coroutine 'sleep' was never awaited

    # ✅ CORRECT
    await asyncio.sleep(1)
```

### 2) Using Blocking Calls Inside Async Code

```python
import time

async def main():
    # ❌ WRONG: blocks the entire event loop for 5 seconds!
    time.sleep(5)

    # ✅ CORRECT: yields control during the wait
    await asyncio.sleep(5)

    # ✅ For truly blocking operations:
    await asyncio.to_thread(time.sleep, 5)
```

### 3) Running `asyncio.run()` Inside an Event Loop

```python
# ❌ WRONG (e.g., in Jupyter Notebook):
asyncio.run(main())   # RuntimeError: This event loop is already running

# ✅ CORRECT in Jupyter (event loop already running):
await main()

# ✅ Or use nest_asyncio:
# import nest_asyncio; nest_asyncio.apply()
```

### 4) Not Handling Task Exceptions

```python
async def main():
    # ❌ WRONG: exception in task is silently lost!
    task = asyncio.create_task(failing())
    await asyncio.sleep(1)   # Never awaits the task

    # ✅ CORRECT: always await your tasks or add a callback
    task = asyncio.create_task(failing())
    try:
        await task
    except ValueError as e:
        print(f"Caught: {e}")
```

---

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">asyncio</code> achieves high concurrency on a <strong>single thread</strong> by cooperatively switching between coroutines at every <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">await</code> point — master <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">create_task()</code> / <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">gather()</code> for I/O-bound concurrency, use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">run_in_executor()</code> to escape blocking calls, and prefer <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">TaskGroup</code> (3.11+) for structured, fail-safe task management.</div>