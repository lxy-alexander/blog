---
title: "Python threading"
published: 2026-03-09
description: "Python threading"
image: ""
tags: ["python","Python threading"]
category: python
draft: false
lang: ""
---

# **I. Python Multithreading — Complete API Reference Manual**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Python's <span style="color:#E8600A;font-weight:700">threading</span> module provides a high-level interface for <span style="color:#E8600A;font-weight:700">Multithreading (多线程编程)</span> built on top of the lower-level <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">_thread</code> module. Because of the <span style="color:#E8600A;font-weight:700">GIL (Global Interpreter Lock, 全局解释器锁)</span>, threads do not achieve true CPU parallelism for pure Python code — but they excel at <span style="color:#2980B9">IO-bound tasks (IO密集型任务)</span> such as network requests, file operations, and database calls. This manual covers every public API with runnable examples. </div>

------

## 1. Thread — Core Thread Object (核心线程对象)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">threading.Thread</code> is the fundamental building block. A thread can be created by passing a <strong>callable target</strong> or by <strong>subclassing</strong> and overriding <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">run()</code>. </div>

### 1) Constructor (构造函数)

```python
threading.Thread(
    group=None,      # reserved, always None
    target=None,     # callable to run in thread
    name=None,       # thread name string
    args=(),         # positional args tuple for target
    kwargs=None,     # keyword args dict for target
    daemon=None      # True → daemon thread (守护线程)
)
```

### 2) `Thread.start()` — Launch the thread

<span style="color:#2980B9">Schedules</span> the thread for execution. Must be called exactly once per Thread object.

```python
import threading
import time

def worker(name, delay):
    time.sleep(delay)
    print(f"[{name}] finished after {delay}s")

t1 = threading.Thread(target=worker, args=("Alpha", 1))
t2 = threading.Thread(target=worker, args=("Beta",  2))

t1.start()   # ← launches t1
t2.start()   # ← launches t2 concurrently

print("Main thread continues immediately")
# Output order (non-deterministic):
# Main thread continues immediately
# [Alpha] finished after 1s
# [Beta]  finished after 2s
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">Calling <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">start()</code> twice on the same Thread raises <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">RuntimeError</code>.</span> If you need to rerun a task, create a new Thread instance.</div>

### 3) `Thread.join(timeout=None)` — Wait for completion (等待线程结束)

Blocks the calling thread until the target thread terminates, or ==until `timeout` seconds elapse.==

```python
import threading, time

def slow_task():
    print("Task started")
    time.sleep(3)
    print("Task done")

t = threading.Thread(target=slow_task)
t.start()

t.join(timeout=5)   # wait up to 5 seconds

if t.is_alive():
    print("Thread still running after timeout!")
else:
    print("Thread completed successfully")
# → Task started
# → Task done
# → Thread completed successfully
```

### 4) `Thread.is_alive()` — Check thread status (检查线程状态)

Returns `True` between `start()` and thread termination.

```python
import threading, time

def task():
    time.sleep(2)

t = threading.Thread(target=task)
print(t.is_alive())   # → False  (not started yet)
t.start()
print(t.is_alive())   # → True   (running)
t.join()
print(t.is_alive())   # → False  (terminated)
```

### 5) `Thread.name` / `Thread.getName()` / `Thread.setName()` — Thread name (线程名)

```python
import threading

def task():
    # Access name inside the thread
    print(f"Running as: {threading.current_thread().name}")

t = threading.Thread(target=task, name="WorkerThread-1")
print(t.name)          # → WorkerThread-1
t.setName("Renamed")
print(t.getName())     # → Renamed
t.start()
t.join()
# → Running as: Renamed
```

### 6) `Thread.daemon` — Daemon threads (守护线程)

<span style="color:#C0392B;font-weight:600">A daemon thread is automatically killed when ALL non-daemon threads exit — it does NOT block program shutdown.</span>

```python
import threading, time

def background_monitor():
    while True:
        print("[Monitor] heartbeat")
        time.sleep(1)

# Must set daemon BEFORE start()
monitor = threading.Thread(target=background_monitor, daemon=True)
monitor.start()

print("Main: doing work")
time.sleep(2.5)
print("Main: exiting — monitor will be killed automatically")
# → [Monitor] heartbeat
# → Main: doing work
# → [Monitor] heartbeat
# → [Monitor] heartbeat
# → Main: exiting — monitor will be killed automatically
```

### 7) `Thread.ident` / `Thread.native_id` — Thread identifiers (线程标识符)

```python
import threading

def show_ids():
    t = threading.current_thread()
    print(f"ident={t.ident}, native_id={t.native_id}")

t = threading.Thread(target=show_ids)
t.start()
t.join()
# → ident=140234567890, native_id=12345

print(f"Main ident: {threading.main_thread().ident}")
```

### 8) Subclass Pattern — Override `run()` (子类模式)

```python
import threading, time

class DownloadThread(threading.Thread):
    """Custom thread that downloads a resource."""

    def __init__(self, url: str):
        super().__init__(name=f"Download-{url}")
        self.url    = url
        self.result = None

    def run(self):
        # Simulate download
        time.sleep(0.5)
        self.result = f"<html from {self.url}>"
        print(f"Downloaded: {self.url}")

threads = [DownloadThread(f"http://example.com/page{i}") for i in range(3)]

for t in threads:
    t.start()

for t in threads:
    t.join()
    print(f"Result: {t.result}")
```

------

## 2. Lock — Mutual Exclusion (互斥锁)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> A <span style="color:#E8600A;font-weight:700">Lock (互斥锁)</span> ensures only ONE thread accesses a critical section (临界区) at a time. It has two states: <span style="color:#2980B9">locked</span> and <span style="color:#2980B9">unlocked</span>. </div>

### 1) `Lock.acquire(blocking=True, timeout=-1)` / `Lock.release()`

```python
import threading

counter = 0
lock    = threading.Lock()

def increment(n):
    global counter
    for _ in range(n):
        lock.acquire()     # ← blocks until lock is free
        counter += 1       # critical section (临界区)
        lock.release()     # ← always release!

threads = [threading.Thread(target=increment, args=(100_000,)) for _ in range(5)]
for t in threads: t.start()
for t in threads: t.join()

print(f"Counter: {counter}")   # → Counter: 500000  (always correct)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">Never call <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">release()</code> without a matching <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">acquire()</code> — raises <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">RuntimeError</code>. Always prefer the <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">with</code> context manager to guarantee release on exceptions.</span></div>

### 2) Context Manager — `with lock` (上下文管理器)

```python
import threading

shared_list = []
lock = threading.Lock()

def safe_append(value):
    with lock:                     # ← acquire on entry, release on exit (even on exception)
        shared_list.append(value)

threads = [threading.Thread(target=safe_append, args=(i,)) for i in range(10)]
for t in threads: t.start()
for t in threads: t.join()

print(sorted(shared_list))   # → [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

### 3) `Lock.acquire(blocking=False)` — Non-blocking try (非阻塞尝试)

```python
import threading, time

lock = threading.Lock()

def try_lock(name):
    acquired = lock.acquire(blocking=False)
    if acquired:
        print(f"[{name}] acquired the lock")
        time.sleep(2)
        lock.release()
    else:
        print(f"[{name}] could not acquire — skipping")

t1 = threading.Thread(target=try_lock, args=("T1",))
t2 = threading.Thread(target=try_lock, args=("T2",))
t1.start(); t2.start()
t1.join();  t2.join()
# → [T1] acquired the lock
# → [T2] could not acquire — skipping
```

### 4) `Lock.acquire(timeout=N)` — Timed wait (超时等待)

```python
import threading, time

lock = threading.Lock()
lock.acquire()   # pre-lock it

def worker():
    result = lock.acquire(timeout=1.5)   # wait max 1.5s
    if result:
        print("Got the lock")
        lock.release()
    else:
        print("Timed out waiting for lock")

t = threading.Thread(target=worker)
t.start()
t.join()
# → Timed out waiting for lock   (lock was never released)
```

### 5) `Lock.locked()` — Query state (查询状态)

```python
import threading

lock = threading.Lock()
print(lock.locked())   # → False

lock.acquire()
print(lock.locked())   # → True

lock.release()
print(lock.locked())   # → False
```

------

## 3. RLock — Reentrant Lock (可重入锁)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> A <span style="color:#E8600A;font-weight:700">RLock (可重入锁)</span> can be acquired multiple times by the <em>same thread</em> without deadlocking. It tracks an internal <span style="color:#2980B9">recursion count (递归计数)</span> — the lock is only released when the count reaches zero. </div>

### 1) Basic RLock usage

```python
import threading

rlock = threading.RLock()

def outer():
    with rlock:                   # recursion count → 1
        print("outer acquired")
        inner()                   # same thread acquires again
        print("outer releasing")
    # recursion count → 0 (fully released)

def inner():
    with rlock:                   # recursion count → 2
        print("inner acquired")
    # recursion count → 1

t = threading.Thread(target=outer)
t.start(); t.join()
# → outer acquired
# → inner acquired
# → outer releasing
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">A plain <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Lock</code> would DEADLOCK in the above pattern</span> because the same thread tries to acquire an already-locked lock. Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">RLock</code> whenever a method holding the lock may call another method that also needs the lock.</div>

### 2) RLock in a class (类中使用RLock)

```python
import threading

class BankAccount:
    def __init__(self, balance: float):
        self.balance = balance
        self._lock   = threading.RLock()

    def deposit(self, amount: float):
        with self._lock:
            self.balance += amount
            print(f"Deposited {amount:.2f} → balance={self.balance:.2f}")

    def withdraw(self, amount: float):
        with self._lock:
            self.balance -= amount
            print(f"Withdrew  {amount:.2f} → balance={self.balance:.2f}")

    def transfer_in(self, amount: float):
        with self._lock:            # outer acquire
            self.deposit(amount)   # inner acquire (reentrant!)
            print(f"Transfer complete")

account = BankAccount(1000.0)
t = threading.Thread(target=account.transfer_in, args=(250.0,))
t.start(); t.join()
# → Deposited 250.00 → balance=1250.00
# → Transfer complete
```

------

## 4. Condition — Wait/Notify Pattern (条件变量)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> A <span style="color:#E8600A;font-weight:700">Condition (条件变量)</span> allows threads to <span style="color:#2980B9">wait</span> for a specific condition to become true and <span style="color:#2980B9">notify</span> other threads when it does. It wraps an underlying lock. </div>

### 1) `Condition.wait()` / `notify()` / `notify_all()`

```python
import threading, time, collections

# Classic Producer-Consumer (生产者-消费者) pattern
buffer    = collections.deque()
MAX_SIZE  = 3
condition = threading.Condition()

def producer():
    for i in range(6):
        with condition:
            while len(buffer) >= MAX_SIZE:
                print(f"Producer waiting — buffer full")
                condition.wait()           # ← releases lock, blocks
            buffer.append(i)
            print(f"Produced {i}  | buffer={list(buffer)}")
            condition.notify_all()        # ← wake waiting consumers
        time.sleep(0.3)

def consumer(name):
    for _ in range(3):
        with condition:
            while not buffer:
                print(f"[{name}] waiting — buffer empty")
                condition.wait()           # ← releases lock, blocks
            item = buffer.popleft()
            print(f"[{name}] consumed {item} | buffer={list(buffer)}")
            condition.notify_all()        # ← wake waiting producer

threads = [
    threading.Thread(target=producer),
    threading.Thread(target=consumer, args=("C1",)),
    threading.Thread(target=consumer, args=("C2",)),
]
for t in threads: t.start()
for t in threads: t.join()
```

### 2) `Condition.wait(timeout=N)` — Timed wait

```python
import threading, time

condition = threading.Condition()
data_ready = False

def waiter():
    with condition:
        result = condition.wait(timeout=2.0)   # wait max 2 seconds
        if result:
            print("Condition met!")
        else:
            print("Timed out — condition never triggered")

def notifier():
    time.sleep(5)   # too slow
    with condition:
        condition.notify()

t1 = threading.Thread(target=waiter)
t2 = threading.Thread(target=notifier)
t1.start(); t2.start()
t1.join();  t2.join()
# → Timed out — condition never triggered
```

### 3) `Condition.wait_for(predicate, timeout=None)` — Predicate wait

```python
import threading, time

items  = []
cond   = threading.Condition()

def consumer():
    with cond:
        # Block until at least 3 items are available
        cond.wait_for(lambda: len(items) >= 3)
        print(f"Got items: {items}")

def producer():
    for i in range(5):
        time.sleep(0.5)
        with cond:
            items.append(i)
            print(f"Added item {i}")
            cond.notify_all()

t1 = threading.Thread(target=consumer)
t2 = threading.Thread(target=producer)
t1.start(); t2.start()
t1.join();  t2.join()
# → Added item 0
# → Added item 1
# → Added item 2
# → Got items: [0, 1, 2]
```

------

## 5. Semaphore & BoundedSemaphore (信号量)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> A <span style="color:#E8600A;font-weight:700">Semaphore (信号量)</span> maintains an internal counter. <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">acquire()</code> decrements it (blocks at zero); <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">release()</code> increments it. Perfect for limiting concurrent access to a resource pool. </div>

### 1) `Semaphore(value=1)` — Connection pool simulation (连接池模拟)

```python
import threading, time, random

# Allow max 3 simultaneous DB connections
db_semaphore = threading.Semaphore(3)

def use_db_connection(thread_id):
    print(f"Thread {thread_id}: waiting for DB connection")
    with db_semaphore:                      # acquire (count -1)
        print(f"Thread {thread_id}: got connection")
        time.sleep(random.uniform(0.5, 1.5))
        print(f"Thread {thread_id}: released connection")
                                            # release (count +1) on exit

threads = [threading.Thread(target=use_db_connection, args=(i,)) for i in range(7)]
for t in threads: t.start()
for t in threads: t.join()
# At most 3 "got connection" lines active at any time
```

### 2) `BoundedSemaphore` — Prevent over-release (防止超额释放)

<span style="color:#C0392B;font-weight:600">Warning: a plain `Semaphore` allows `release()` beyond the initial value — this is usually a bug. `BoundedSemaphore` raises `ValueError` if the count would exceed the initial value.</span>

```python
import threading

sem   = threading.Semaphore(2)
bsem  = threading.BoundedSemaphore(2)

# Plain Semaphore — silently over-releases
sem.release()   # count goes to 3 — no error (潜在bug)
print(f"Semaphore value after over-release: OK (silent)")

# BoundedSemaphore — raises ValueError
try:
    bsem.release()   # count would exceed 2
except ValueError as e:
    print(f"BoundedSemaphore caught: {e}")
# → BoundedSemaphore caught: Semaphore released too many times
```

### 3) Rate limiter pattern (限速器模式)

```python
import threading, time

# Limit to 2 concurrent API calls
api_semaphore = threading.BoundedSemaphore(2)

def call_api(endpoint):
    with api_semaphore:
        print(f"Calling {endpoint}")
        time.sleep(1)   # simulate API latency
        print(f"Done    {endpoint}")

endpoints = [f"/api/resource/{i}" for i in range(6)]
threads   = [threading.Thread(target=call_api, args=(ep,)) for ep in endpoints]

for t in threads: t.start()
for t in threads: t.join()
```

------

## 6. Event — Simple Flag Signaling (事件信号)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> An <span style="color:#E8600A;font-weight:700">Event (事件)</span> is a simple boolean flag. Threads can <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">wait()</code> until the flag is set, and any thread can <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">set()</code> or <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">clear()</code> it. </div>

### 1) `Event.set()` / `Event.clear()` / `Event.wait()` / `Event.is_set()`

```python
import threading, time

start_event = threading.Event()

def worker(name):
    print(f"[{name}] waiting for start signal...")
    start_event.wait()               # blocks until event is set
    print(f"[{name}] GO! Starting work")

workers = [threading.Thread(target=worker, args=(f"W{i}",)) for i in range(4)]
for w in workers: w.start()

print("Main: preparing...")
time.sleep(2)
print("Main: firing start signal!")
start_event.set()                    # wake ALL waiting threads at once

for w in workers: w.join()
# → [W0] waiting for start signal...
# → [W1] waiting for start signal...
# → [W2] waiting for start signal...
# → [W3] waiting for start signal...
# (2s pause)
# → Main: firing start signal!
# → [W0] GO! Starting work    (all 4 unblock simultaneously)
```

### 2) `Event.wait(timeout=N)` — Timed wait

```python
import threading, time

ready = threading.Event()

def service():
    print("Service: initializing (takes 3s)...")
    time.sleep(3)
    ready.set()
    print("Service: ready!")

def client():
    if ready.wait(timeout=1.5):    # only wait 1.5s
        print("Client: connected!")
    else:
        print("Client: service not ready in time, aborting")

t1 = threading.Thread(target=service)
t2 = threading.Thread(target=client)
t1.start(); t2.start()
t1.join();  t2.join()
# → Service: initializing (takes 3s)...
# → Client: service not ready in time, aborting
# → Service: ready!
```

### 3) Stop signal pattern (停止信号模式)

```python
import threading, time

stop_event = threading.Event()

def background_worker():
    count = 0
    while not stop_event.is_set():    # check flag each iteration
        print(f"Working... iteration {count}")
        count += 1
        time.sleep(0.5)
    print("Worker: received stop signal, exiting cleanly")

t = threading.Thread(target=background_worker)
t.start()

time.sleep(2)
print("Main: sending stop signal")
stop_event.set()
t.join()
```

------

## 7. Timer — Delayed Execution (延迟执行)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">threading.Timer</code> is a subclass of <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Thread</code> that executes a function after a specified delay. It can be <span style="color:#2980B9">cancelled</span> before firing. </div>

### 1) Basic Timer

```python
import threading

def reminder(message):
    print(f"⏰ Reminder: {message}")

# Fire after 3 seconds
t = threading.Timer(3.0, reminder, args=("Meeting at 3pm!",))
t.start()

print("Timer set. Waiting...")
t.join()
# → Timer set. Waiting...
# (3s pause)
# → ⏰ Reminder: Meeting at 3pm!
```

### 2) `Timer.cancel()` — Cancel before firing

```python
import threading, time

fired = False

def action():
    global fired
    fired = True
    print("Action fired!")

t = threading.Timer(5.0, action)
t.start()

time.sleep(1)
t.cancel()    # ← cancel within the window
t.join()

print(f"Action fired: {fired}")   # → Action fired: False
```

### 3) Repeating timer pattern (重复定时器模式)

```python
import threading

class RepeatingTimer:
    """Fires a function every `interval` seconds."""

    def __init__(self, interval: float, func, *args):
        self.interval = interval
        self.func     = func
        self.args     = args
        self._timer   = None
        self._running = False

    def _run(self):
        self.func(*self.args)
        if self._running:
            self._schedule()

    def _schedule(self):
        self._timer = threading.Timer(self.interval, self._run)
        self._timer.daemon = True
        self._timer.start()

    def start(self):
        self._running = True
        self._schedule()

    def stop(self):
        self._running = False
        if self._timer:
            self._timer.cancel()

import time

counter = [0]
def tick():
    counter[0] += 1
    print(f"Tick #{counter[0]}")

rt = RepeatingTimer(0.5, tick)
rt.start()
time.sleep(2.5)
rt.stop()
print(f"Total ticks: {counter[0]}")   # → Total ticks: 5
```

------

## 8. Barrier — Thread Synchronization Point (屏障同步点)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> A <span style="color:#E8600A;font-weight:700">Barrier (屏障)</span> makes a fixed number of threads wait at a rendezvous point until ALL of them arrive — then releases all of them simultaneously. </div>

### 1) `Barrier(parties, action=None, timeout=None)`

```python
import threading, time, random

NUM_WORKERS = 4
barrier = threading.Barrier(NUM_WORKERS)

def phase_worker(name):
    # Phase 1
    duration = random.uniform(0.5, 2.0)
    print(f"[{name}] phase 1 working for {duration:.1f}s")
    time.sleep(duration)
    print(f"[{name}] phase 1 done — waiting at barrier")

    barrier.wait()     # ← all threads block here until all 4 arrive

    print(f"[{name}] phase 2 starting (all threads released together)")

threads = [threading.Thread(target=phase_worker, args=(f"W{i}",))
           for i in range(NUM_WORKERS)]
for t in threads: t.start()
for t in threads: t.join()
```

### 2) `Barrier` with `action` callback

```python
import threading, time

def setup_phase():
    """Runs ONCE when all threads reach the barrier, before release."""
    print(">>> All threads ready — running barrier action <<<")

barrier = threading.Barrier(3, action=setup_phase)

def worker(name):
    time.sleep(0.1)
    print(f"[{name}] arrived at barrier")
    barrier.wait()
    print(f"[{name}] past barrier")

threads = [threading.Thread(target=worker, args=(f"T{i}",)) for i in range(3)]
for t in threads: t.start()
for t in threads: t.join()
```

### 3) `Barrier.abort()` / `BrokenBarrierError`

```python
import threading, time

barrier = threading.Barrier(3)

def risky_worker(name, should_abort):
    try:
        if should_abort:
            time.sleep(0.2)
            print(f"[{name}] aborting barrier!")
            barrier.abort()          # breaks the barrier for everyone
        else:
            print(f"[{name}] waiting at barrier...")
            barrier.wait(timeout=2)
            print(f"[{name}] passed!")
    except threading.BrokenBarrierError:
        print(f"[{name}] barrier was broken — handling gracefully")

threads = [
    threading.Thread(target=risky_worker, args=("T0", False)),
    threading.Thread(target=risky_worker, args=("T1", False)),
    threading.Thread(target=risky_worker, args=("T2", True)),   # aborts
]
for t in threads: t.start()
for t in threads: t.join()
```

### 4) Barrier properties

```python
import threading

b = threading.Barrier(5)
print(b.parties)    # → 5   (total threads needed)
print(b.n_waiting)  # → 0   (currently waiting)
print(b.broken)     # → False
```

------

## 9. local — Thread-local Storage (线程本地存储)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">threading.local()</code> creates an object where each thread has its <span style="color:#E8600A;font-weight:700">own independent copy</span> of every attribute. Ideal for thread-specific state like database connections or request contexts. </div>

### 1) Basic thread-local usage

```python
import threading

local_data = threading.local()

def worker(value):
    local_data.x = value              # each thread sets its own .x
    import time; time.sleep(0.1)      # let other threads run
    print(f"Thread {threading.current_thread().name}: x = {local_data.x}")

threads = [threading.Thread(target=worker, args=(i*10,), name=f"T{i}")
           for i in range(4)]
for t in threads: t.start()
for t in threads: t.join()
# → Thread T0: x = 0
# → Thread T1: x = 10
# → Thread T2: x = 20
# → Thread T3: x = 30
# (each thread sees only its own value — no interference)
```

### 2) Thread-local DB connection pattern

```python
import threading
import sqlite3

_local = threading.local()

def get_connection(db_path: str) -> sqlite3.Connection:
    """Return a per-thread DB connection (创建线程私有数据库连接)."""
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(db_path)
        print(f"[{threading.current_thread().name}] created new connection")
    return _local.conn

def db_worker(db_path: str):
    conn = get_connection(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS t (v INTEGER)")
    conn.execute("INSERT INTO t VALUES (?)", (threading.get_ident(),))
    conn.commit()
    print(f"[{threading.current_thread().name}] inserted row")

threads = [threading.Thread(target=db_worker, args=(":memory:",), name=f"DB-{i}")
           for i in range(3)]
for t in threads: t.start()
for t in threads: t.join()
```

### 3) Subclass `local` for initialization

```python
import threading

class RequestContext(threading.local):
    """Thread-local request context with defaults."""
    def __init__(self):
        super().__init__()
        self.user_id   = None
        self.request_id = None

ctx = RequestContext()

def handle_request(user_id, req_id):
    ctx.user_id    = user_id
    ctx.request_id = req_id
    import time; time.sleep(0.05)
    print(f"Processing request {ctx.request_id} for user {ctx.user_id}")

threads = [threading.Thread(target=handle_request, args=(f"user{i}", f"req-{i:03}"))
           for i in range(4)]
for t in threads: t.start()
for t in threads: t.join()
```

------

## 10. Module-level Functions (模块级函数)

### 1) `threading.current_thread()` — Get the current thread

```python
import threading

def show_self():
    t = threading.current_thread()
    print(f"name={t.name}, ident={t.ident}, daemon={t.daemon}")

main_t = threading.current_thread()
print(f"Main thread: {main_t.name}")

t = threading.Thread(target=show_self, name="MyWorker")
t.start(); t.join()
# → Main thread: MainThread
# → name=MyWorker, ident=140..., daemon=False
```

### 2) `threading.main_thread()` — Get the main thread

```python
import threading

def check_main():
    mt = threading.main_thread()
    ct = threading.current_thread()
    print(f"Main thread: {mt.name}")
    print(f"This thread: {ct.name}")
    print(f"Am I main?  {ct is mt}")

t = threading.Thread(target=check_main)
t.start(); t.join()
# → Main thread: MainThread
# → This thread: Thread-1
# → Am I main?  False
```

### 3) `threading.active_count()` — Count live threads

```python
import threading, time

def slow():
    time.sleep(2)

print(threading.active_count())   # → 1  (main only)

threads = [threading.Thread(target=slow) for _ in range(3)]
for t in threads: t.start()

print(threading.active_count())   # → 4  (main + 3 workers)
for t in threads: t.join()
print(threading.active_count())   # → 1
```

### 4) `threading.enumerate()` — List all live threads

```python
import threading, time

def task(n):
    time.sleep(n)

threads = [threading.Thread(target=task, args=(i,), name=f"T{i}") for i in range(1,4)]
for t in threads: t.start()

for t in threading.enumerate():
    print(f"  alive: {t.name} | daemon={t.daemon}")
# → alive: MainThread | daemon=False
# → alive: T1        | daemon=False
# → alive: T2        | daemon=False
# → alive: T3        | daemon=False

for t in threads: t.join()
```

### 5) `threading.settrace(func)` / `threading.setprofile(func)` — Thread hooks

```python
import threading, sys

def my_tracer(frame, event, arg):
    if event == "call":
        print(f"[TRACE] calling {frame.f_code.co_name}")
    return my_tracer

def task():
    x = 1 + 1
    return x

threading.settrace(my_tracer)    # set trace for ALL future threads
t = threading.Thread(target=task)
t.start(); t.join()
threading.settrace(None)         # remove tracer
```

### 6) `threading.stack_size(size=0)` — Set thread stack size

```python
import threading

# Set stack size to 512 KB for all future threads
threading.stack_size(512 * 1024)
print(f"Stack size: {threading.stack_size()} bytes")

def task():
    print(f"Running with custom stack size")

t = threading.Thread(target=task)
t.start(); t.join()

threading.stack_size(0)   # reset to default
```

### 7) `threading.excepthook` — Handle uncaught thread exceptions (未捕获异常处理)

```python
import threading

def custom_excepthook(args):
    print(f"Uncaught exception in thread [{args.thread.name}]:")
    print(f"  Type:    {args.exc_type.__name__}")
    print(f"  Message: {args.exc_value}")

threading.excepthook = custom_excepthook

def buggy_task():
    raise ValueError("Something went wrong in thread!")

t = threading.Thread(target=buggy_task, name="BuggyThread")
t.start(); t.join()
# → Uncaught exception in thread [BuggyThread]:
# →   Type:    ValueError
# →   Message: Something went wrong in thread!
```

### 8) `threading.get_ident()` / `threading.get_native_id()`

```python
import threading

def show_ids():
    print(f"Python ident:    {threading.get_ident()}")
    print(f"OS native id:    {threading.get_native_id()}")

t = threading.Thread(target=show_ids)
t.start(); t.join()
```

------

## 11. queue Module — Thread-safe Queues (线程安全队列)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> The <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">queue</code> module provides three thread-safe queue classes: <span style="color:#E8600A;font-weight:700">Queue (FIFO)</span>, <span style="color:#E8600A;font-weight:700">LifoQueue (LIFO/stack)</span>, and <span style="color:#E8600A;font-weight:700">PriorityQueue (优先队列)</span>. All use internal locks, so no external synchronization is needed. </div>

### 1) `Queue(maxsize=0)` — FIFO Queue

```python
from queue import Queue
import threading, time

q = Queue(maxsize=3)

def producer():
    for i in range(6):
        q.put(i)          # blocks if queue is full (maxsize reached)
        print(f"Put {i}  | qsize={q.qsize()}")
        time.sleep(0.2)

def consumer():
    for _ in range(6):
        item = q.get()    # blocks if queue is empty
        print(f"Got {item}")
        q.task_done()
        time.sleep(0.5)

t1 = threading.Thread(target=producer)
t2 = threading.Thread(target=consumer)
t1.start(); t2.start()
t1.join();  t2.join()
```

### 2) `Queue.put_nowait()` / `Queue.get_nowait()` — Non-blocking

```python
from queue import Queue, Full, Empty

q = Queue(maxsize=2)
q.put("item1")
q.put("item2")

try:
    q.put_nowait("item3")     # queue full!
except Full:
    print("Queue full — item3 dropped")

try:
    while True:
        print(q.get_nowait())
except Empty:
    print("Queue emptied")
# → Queue full — item3 dropped
# → item1
# → item2
# → Queue emptied
```

### 3) `Queue.join()` / `Queue.task_done()` — Work tracking

```python
from queue import Queue
import threading

work_queue = Queue()

def worker():
    while True:
        task = work_queue.get()
        if task is None:
            break
        print(f"Processing: {task}")
        work_queue.task_done()   # signal this task is complete

# Start 3 workers
workers = [threading.Thread(target=worker, daemon=True) for _ in range(3)]
for w in workers: w.start()

# Enqueue tasks
for task in ["task_A", "task_B", "task_C", "task_D", "task_E"]:
    work_queue.put(task)

work_queue.join()   # blocks until ALL task_done() called
print("All tasks completed!")
```

### 4) `LifoQueue` — Stack (栈/后进先出)

```python
from queue import LifoQueue

stack = LifoQueue()
stack.put("first")
stack.put("second")
stack.put("third")

while not stack.empty():
    print(stack.get())
# → third
# → second
# → first
```

### 5) `PriorityQueue` — Priority-based processing (优先级队列)

```python
from queue import PriorityQueue
import threading, time

pq = PriorityQueue()

# (priority, task_name) — lower number = higher priority
pq.put((3, "low-priority task"))
pq.put((1, "URGENT task"))
pq.put((2, "medium task"))
pq.put((1, "another URGENT task"))

while not pq.empty():
    priority, task = pq.get()
    print(f"[priority={priority}] Processing: {task}")
# → [priority=1] Processing: URGENT task
# → [priority=1] Processing: another URGENT task
# → [priority=2] Processing: medium task
# → [priority=3] Processing: low-priority task
```

------

## 12. ThreadPoolExecutor — High-level Thread Pool (高级线程池)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">concurrent.futures.ThreadPoolExecutor</code> provides a high-level, <span style="color:#2980B9">Future-based (Future对象)</span> interface for thread pools. It is the <span style="color:#E8600A;font-weight:700">recommended way</span> to run IO-bound tasks in modern Python. </div>

### 1) `submit()` → Future

```python
from concurrent.futures import ThreadPoolExecutor
import time

def fetch_data(url: str) -> str:
    time.sleep(1)   # simulate network call
    return f"<data from {url}>"

urls = [f"http://example.com/page{i}" for i in range(5)]

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(fetch_data, url) for url in urls]

    for future in futures:
        result = future.result()   # blocks until this future completes
        print(result)
```

### 2) `map()` — Parallel map (并行映射)

```python
from concurrent.futures import ThreadPoolExecutor
import time

def square(n):
    time.sleep(0.2)
    return n * n

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(square, range(10)))

print(results)   # → [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

### 3) `Future` API — `done()`, `cancel()`, `add_done_callback()`

```python
from concurrent.futures import ThreadPoolExecutor
import time

def slow_task(n):
    time.sleep(n)
    return f"result-{n}"

def on_done(future):
    print(f"Callback: task finished → {future.result()}")

with ThreadPoolExecutor(max_workers=2) as executor:
    f1 = executor.submit(slow_task, 1)
    f2 = executor.submit(slow_task, 2)

    f1.add_done_callback(on_done)    # register callback
    f2.add_done_callback(on_done)

    print(f"f1 done: {f1.done()}")   # likely False (still running)
    time.sleep(1.5)
    print(f"f1 done: {f1.done()}")   # → True
```

### 4) `as_completed()` — Process in completion order (按完成顺序处理)

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import time, random

def task(n):
    delay = random.uniform(0.1, 1.0)
    time.sleep(delay)
    return (n, delay)

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(task, i): i for i in range(8)}

    for future in as_completed(futures):
        task_id = futures[future]
        n, delay = future.result()
        print(f"Task {n} finished in {delay:.2f}s")
# Tasks print in the order they complete, not submission order
```

### 5) Exception handling in futures (Future异常处理)

```python
from concurrent.futures import ThreadPoolExecutor

def risky(x):
    if x == 3:
        raise ValueError(f"Bad input: {x}")
    return x * 2

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(risky, i) for i in range(5)]

for i, f in enumerate(futures):
    try:
        print(f"Result {i}: {f.result()}")
    except ValueError as e:
        print(f"Result {i}: ERROR — {e}")
# → Result 0: 0
# → Result 1: 2
# → Result 2: 4
# → Result 3: ERROR — Bad input: 3
# → Result 4: 8
```

------

## 13. Common Patterns & Pitfalls (常见模式与陷阱)

### 1) Race condition example (竞态条件示例)

```python
import threading

counter = 0   # UNSAFE shared state

def unsafe_increment():
    global counter
    for _ in range(100_000):
        counter += 1   # NOT atomic! (read-modify-write)

threads = [threading.Thread(target=unsafe_increment) for _ in range(5)]
for t in threads: t.start()
for t in threads: t.join()

print(f"Expected: 500000")
print(f"Actual:   {counter}")   # likely LESS than 500000 — data race!
```

### 2) Deadlock example + fix (死锁示例及修复)

```python
import threading

lock_a = threading.Lock()
lock_b = threading.Lock()

# ─── DEADLOCK version ────────────────────────────────
def thread1_deadlock():
    with lock_a:
        import time; time.sleep(0.1)
        with lock_b:                  # waits for lock_b
            print("T1: got both locks")

def thread2_deadlock():
    with lock_b:
        import time; time.sleep(0.1)
        with lock_a:                  # waits for lock_a → DEADLOCK
            print("T2: got both locks")

# ─── FIXED version: always acquire locks in the same order ──
def thread1_safe():
    with lock_a:                      # acquire A first
        with lock_b:                  # then B
            print("T1 safe: got both locks")

def thread2_safe():
    with lock_a:                      # acquire A first (same order!)
        with lock_b:
            print("T2 safe: got both locks")

t1 = threading.Thread(target=thread1_safe)
t2 = threading.Thread(target=thread2_safe)
t1.start(); t2.start()
t1.join();  t2.join()
# → T1 safe: got both locks
# → T2 safe: got both locks
```

### 3) Thread-safe singleton (线程安全单例)

```python
import threading

class Singleton:
    _instance = None
    _lock     = threading.Lock()

    def __new__(cls):
        if cls._instance is None:              # first check (no lock)
            with cls._lock:
                if cls._instance is None:      # second check (with lock)
                    cls._instance = super().__new__(cls)
                    print("Singleton created")
        return cls._instance

def get_instance():
    s = Singleton()
    print(f"Got instance: {id(s)}")

threads = [threading.Thread(target=get_instance) for _ in range(5)]
for t in threads: t.start()
for t in threads: t.join()
# → Singleton created   (exactly once)
# → Got instance: 140...  (same id for all 5 threads)
```

------

## 14. Full API Quick Reference (API速查表)

| Class / Function                                             | Key Methods                                     | Purpose                           |
| ------------------------------------------------------------ | ----------------------------------------------- | --------------------------------- |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Thread</code> | `start()` `join()` `is_alive()`                 | Create and manage threads         |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Lock</code> | `acquire()` `release()` `locked()`              | Mutual exclusion                  |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">RLock</code> | `acquire()` `release()`                         | Reentrant mutual exclusion        |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Condition</code> | `wait()` `wait_for()` `notify()` `notify_all()` | Wait/notify synchronization       |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Semaphore</code> | `acquire()` `release()`                         | Limit concurrent access           |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">BoundedSemaphore</code> | `acquire()` `release()`                         | Semaphore with over-release guard |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Event</code> | `set()` `clear()` `wait()` `is_set()`           | Boolean flag signaling            |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Timer</code> | `start()` `cancel()`                            | Delayed / cancellable execution   |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Barrier</code> | `wait()` `abort()` `reset()`                    | N-thread rendezvous point         |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">local</code> | attribute access                                | Per-thread storage                |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Queue</code> | `put()` `get()` `task_done()` `join()`          | Thread-safe FIFO queue            |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">LifoQueue</code> | `put()` `get()`                                 | Thread-safe stack                 |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">PriorityQueue</code> | `put()` `get()`                                 | Thread-safe priority queue        |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">ThreadPoolExecutor</code> | `submit()` `map()` `shutdown()`                 | High-level thread pool            |
| <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">current_thread()</code> | —                                               | Get current Thread object         |
| <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">active_count()</code> | —                                               | Count live threads                |
| <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">enumerate()</code> | —                                               | List all live threads             |
| <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">excepthook</code> | —                                               | Handle uncaught thread exceptions |

------

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> Python threading excels at <span style="color:#E8600A;font-weight:700">IO-bound concurrency (IO密集型并发)</span>: use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">ThreadPoolExecutor</code> for simple task pools, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Queue</code> for producer-consumer pipelines, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Lock</code>/<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">RLock</code> for shared state, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Event</code> for signaling, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Semaphore</code> for resource pools, and <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Barrier</code> for multi-phase synchronization — always protect shared mutable state to avoid <span style="color:#C0392B;font-weight:600">Race Conditions (竞态条件)</span> and <span style="color:#C0392B;font-weight:600">Deadlocks (死锁)</span>. </div>

---

# **I. Python Multithreading — Complete API Reference Manual**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Python's <span style="color:#E8600A;font-weight:700">threading</span> module provides a high-level interface for <span style="color:#E8600A;font-weight:700">Multithreading (多线程编程)</span> built on top of the lower-level <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">_thread</code> module. Because of the <span style="color:#E8600A;font-weight:700">GIL (Global Interpreter Lock, 全局解释器锁)</span>, threads do not achieve true CPU parallelism for pure Python code — but they excel at <span style="color:#2980B9">IO-bound tasks (IO密集型任务)</span> such as network requests, file operations, and database calls. This manual covers every public API with runnable examples. </div>

------

## 1. Thread — Core Thread Object (核心线程对象)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">threading.Thread</code> is the fundamental building block. A thread can be created by passing a <strong>callable target</strong> or by <strong>subclassing</strong> and overriding <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">run()</code>. </div>

### 1) Constructor (构造函数)

```python
threading.Thread(
    group=None,      # reserved, always None
    target=None,     # callable to run in thread
    name=None,       # thread name string
    args=(),         # positional args tuple for target
    kwargs=None,     # keyword args dict for target
    daemon=None      # True → daemon thread (守护线程)
)
```

### 2) `Thread.start()` — Launch the thread

<span style="color:#2980B9">Schedules</span> the thread for execution. Must be called exactly once per Thread object.

```python
import threading
import time

def worker(name, delay):
    time.sleep(delay)
    print(f"[{name}] finished after {delay}s")

t1 = threading.Thread(target=worker, args=("Alpha", 1))
t2 = threading.Thread(target=worker, args=("Beta",  2))

t1.start()   # ← launches t1
t2.start()   # ← launches t2 concurrently

print("Main thread continues immediately")
# Output order (non-deterministic):
# Main thread continues immediately
# [Alpha] finished after 1s
# [Beta]  finished after 2s
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">Calling <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">start()</code> twice on the same Thread raises <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">RuntimeError</code>.</span> If you need to rerun a task, create a new Thread instance.</div>

### 3) `Thread.join(timeout=None)` — Wait for completion (等待线程结束)

Blocks the calling thread until the target thread terminates, or until `timeout` seconds elapse.

```python
import threading, time

def slow_task():
    print("Task started")
    time.sleep(3)
    print("Task done")

t = threading.Thread(target=slow_task)
t.start()

t.join(timeout=5)   # wait up to 5 seconds

if t.is_alive():
    print("Thread still running after timeout!")
else:
    print("Thread completed successfully")
# → Task started
# → Task done
# → Thread completed successfully
```

### 4) `Thread.is_alive()` — Check thread status (检查线程状态)

Returns `True` between `start()` and thread termination.

```python
import threading, time

def task():
    time.sleep(2)

t = threading.Thread(target=task)
print(t.is_alive())   # → False  (not started yet)
t.start()
print(t.is_alive())   # → True   (running)
t.join()
print(t.is_alive())   # → False  (terminated)
```

### 5) `Thread.name` / `Thread.getName()` / `Thread.setName()` — Thread name (线程名)

```python
import threading

def task():
    # Access name inside the thread
    print(f"Running as: {threading.current_thread().name}")

t = threading.Thread(target=task, name="WorkerThread-1")
print(t.name)          # → WorkerThread-1
t.setName("Renamed")
print(t.getName())     # → Renamed
t.start()
t.join()
# → Running as: Renamed
```

### 6) `Thread.daemon` — Daemon threads (守护线程)

<span style="color:#C0392B;font-weight:600">A daemon thread is automatically killed when ALL non-daemon threads exit — it does NOT block program shutdown.</span>

```python
import threading, time

def background_monitor():
    while True:
        print("[Monitor] heartbeat")
        time.sleep(1)

# Must set daemon BEFORE start()
monitor = threading.Thread(target=background_monitor, daemon=True)
monitor.start()

print("Main: doing work")
time.sleep(2.5)
print("Main: exiting — monitor will be killed automatically")
# → [Monitor] heartbeat
# → Main: doing work
# → [Monitor] heartbeat
# → [Monitor] heartbeat
# → Main: exiting — monitor will be killed automatically
```

### 7) `Thread.ident` / `Thread.native_id` — Thread identifiers (线程标识符)

```python
import threading

def show_ids():
    t = threading.current_thread()
    print(f"ident={t.ident}, native_id={t.native_id}")

t = threading.Thread(target=show_ids)
t.start()
t.join()
# → ident=140234567890, native_id=12345

print(f"Main ident: {threading.main_thread().ident}")
```

### 8) Subclass Pattern — Override `run()` (子类模式)

```python
import threading, time

class DownloadThread(threading.Thread):
    """Custom thread that downloads a resource."""

    def __init__(self, url: str):
        super().__init__(name=f"Download-{url}")
        self.url    = url
        self.result = None

    def run(self):
        # Simulate download
        time.sleep(0.5)
        self.result = f"<html from {self.url}>"
        print(f"Downloaded: {self.url}")

threads = [DownloadThread(f"http://example.com/page{i}") for i in range(3)]

for t in threads:
    t.start()

for t in threads:
    t.join()
    print(f"Result: {t.result}")
```

------

## 2. Lock — Mutual Exclusion (互斥锁)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> A <span style="color:#E8600A;font-weight:700">Lock (互斥锁)</span> ensures only ONE thread accesses a critical section (临界区) at a time. It has two states: <span style="color:#2980B9">locked</span> and <span style="color:#2980B9">unlocked</span>. </div>

### 1) `Lock.acquire(blocking=True, timeout=-1)` / `Lock.release()`

```python
import threading

counter = 0
lock    = threading.Lock()

def increment(n):
    global counter
    for _ in range(n):
        lock.acquire()     # ← blocks until lock is free
        counter += 1       # critical section (临界区)
        lock.release()     # ← always release!

threads = [threading.Thread(target=increment, args=(100_000,)) for _ in range(5)]
for t in threads: t.start()
for t in threads: t.join()

print(f"Counter: {counter}")   # → Counter: 500000  (always correct)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">Never call <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">release()</code> without a matching <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">acquire()</code> — raises <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">RuntimeError</code>. Always prefer the <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">with</code> context manager to guarantee release on exceptions.</span></div>

### 2) Context Manager — `with lock` (上下文管理器)

```python
import threading

shared_list = []
lock = threading.Lock()

def safe_append(value):
    with lock:                     # ← acquire on entry, release on exit (even on exception)
        shared_list.append(value)

threads = [threading.Thread(target=safe_append, args=(i,)) for i in range(10)]
for t in threads: t.start()
for t in threads: t.join()

print(sorted(shared_list))   # → [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

### 3) `Lock.acquire(blocking=False)` — Non-blocking try (非阻塞尝试)

```python
import threading, time

lock = threading.Lock()

def try_lock(name):
    acquired = lock.acquire(blocking=False)
    if acquired:
        print(f"[{name}] acquired the lock")
        time.sleep(2)
        lock.release()
    else:
        print(f"[{name}] could not acquire — skipping")

t1 = threading.Thread(target=try_lock, args=("T1",))
t2 = threading.Thread(target=try_lock, args=("T2",))
t1.start(); t2.start()
t1.join();  t2.join()
# → [T1] acquired the lock
# → [T2] could not acquire — skipping
```

### 4) `Lock.acquire(timeout=N)` — Timed wait (超时等待)

```python
import threading, time

lock = threading.Lock()
lock.acquire()   # pre-lock it

def worker():
    result = lock.acquire(timeout=1.5)   # wait max 1.5s
    if result:
        print("Got the lock")
        lock.release()
    else:
        print("Timed out waiting for lock")

t = threading.Thread(target=worker)
t.start()
t.join()
# → Timed out waiting for lock   (lock was never released)
```

### 5) `Lock.locked()` — Query state (查询状态)

```python
import threading

lock = threading.Lock()
print(lock.locked())   # → False

lock.acquire()
print(lock.locked())   # → True

lock.release()
print(lock.locked())   # → False
```

------

## 3. RLock — Reentrant Lock (可重入锁)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> A <span style="color:#E8600A;font-weight:700">RLock (可重入锁)</span> can be acquired multiple times by the <em>same thread</em> without deadlocking. It tracks an internal <span style="color:#2980B9">recursion count (递归计数)</span> — the lock is only released when the count reaches zero. </div>

### 1) Basic RLock usage

```python
import threading

rlock = threading.RLock()

def outer():
    with rlock:                   # recursion count → 1
        print("outer acquired")
        inner()                   # same thread acquires again
        print("outer releasing")
    # recursion count → 0 (fully released)

def inner():
    with rlock:                   # recursion count → 2
        print("inner acquired")
    # recursion count → 1

t = threading.Thread(target=outer)
t.start(); t.join()
# → outer acquired
# → inner acquired
# → outer releasing
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">A plain <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Lock</code> would DEADLOCK in the above pattern</span> because the same thread tries to acquire an already-locked lock. Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">RLock</code> whenever a method holding the lock may call another method that also needs the lock.</div>

### 2) RLock in a class (类中使用RLock)

```python
import threading

class BankAccount:
    def __init__(self, balance: float):
        self.balance = balance
        self._lock   = threading.RLock()

    def deposit(self, amount: float):
        with self._lock:
            self.balance += amount
            print(f"Deposited {amount:.2f} → balance={self.balance:.2f}")

    def withdraw(self, amount: float):
        with self._lock:
            self.balance -= amount
            print(f"Withdrew  {amount:.2f} → balance={self.balance:.2f}")

    def transfer_in(self, amount: float):
        with self._lock:            # outer acquire
            self.deposit(amount)   # inner acquire (reentrant!)
            print(f"Transfer complete")

account = BankAccount(1000.0)
t = threading.Thread(target=account.transfer_in, args=(250.0,))
t.start(); t.join()
# → Deposited 250.00 → balance=1250.00
# → Transfer complete
```

------

## 4. Condition — Wait/Notify Pattern (条件变量)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> A <span style="color:#E8600A;font-weight:700">Condition (条件变量)</span> allows threads to <span style="color:#2980B9">wait</span> for a specific condition to become true and <span style="color:#2980B9">notify</span> other threads when it does. It wraps an underlying lock. </div>

### 1) `Condition.wait()` / `notify()` / `notify_all()`

```python
import threading, time, collections

# Classic Producer-Consumer (生产者-消费者) pattern
buffer    = collections.deque()
MAX_SIZE  = 3
condition = threading.Condition()

def producer():
    for i in range(6):
        with condition:
            while len(buffer) >= MAX_SIZE:
                print(f"Producer waiting — buffer full")
                condition.wait()           # ← releases lock, blocks
            buffer.append(i)
            print(f"Produced {i}  | buffer={list(buffer)}")
            condition.notify_all()        # ← wake waiting consumers
        time.sleep(0.3)

def consumer(name):
    for _ in range(3):
        with condition:
            while not buffer:
                print(f"[{name}] waiting — buffer empty")
                condition.wait()           # ← releases lock, blocks
            item = buffer.popleft()
            print(f"[{name}] consumed {item} | buffer={list(buffer)}")
            condition.notify_all()        # ← wake waiting producer

threads = [
    threading.Thread(target=producer),
    threading.Thread(target=consumer, args=("C1",)),
    threading.Thread(target=consumer, args=("C2",)),
]
for t in threads: t.start()
for t in threads: t.join()
```

### 2) `Condition.wait(timeout=N)` — Timed wait

```python
import threading, time

condition = threading.Condition()
data_ready = False

def waiter():
    with condition:
        result = condition.wait(timeout=2.0)   # wait max 2 seconds
        if result:
            print("Condition met!")
        else:
            print("Timed out — condition never triggered")

def notifier():
    time.sleep(5)   # too slow
    with condition:
        condition.notify()

t1 = threading.Thread(target=waiter)
t2 = threading.Thread(target=notifier)
t1.start(); t2.start()
t1.join();  t2.join()
# → Timed out — condition never triggered
```

### 3) `Condition.wait_for(predicate, timeout=None)` — Predicate wait

```python
import threading, time

items  = []
cond   = threading.Condition()

def consumer():
    with cond:
        # Block until at least 3 items are available
        cond.wait_for(lambda: len(items) >= 3)
        print(f"Got items: {items}")

def producer():
    for i in range(5):
        time.sleep(0.5)
        with cond:
            items.append(i)
            print(f"Added item {i}")
            cond.notify_all()

t1 = threading.Thread(target=consumer)
t2 = threading.Thread(target=producer)
t1.start(); t2.start()
t1.join();  t2.join()
# → Added item 0
# → Added item 1
# → Added item 2
# → Got items: [0, 1, 2]
```

------

## 5. Semaphore & BoundedSemaphore (信号量)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> A <span style="color:#E8600A;font-weight:700">Semaphore (信号量)</span> maintains an internal counter. <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">acquire()</code> decrements it (blocks at zero); <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">release()</code> increments it. Perfect for limiting concurrent access to a resource pool. </div>

### 1) `Semaphore(value=1)` — Connection pool simulation (连接池模拟)

```python
import threading, time, random

# Allow max 3 simultaneous DB connections
db_semaphore = threading.Semaphore(3)

def use_db_connection(thread_id):
    print(f"Thread {thread_id}: waiting for DB connection")
    with db_semaphore:                      # acquire (count -1)
        print(f"Thread {thread_id}: got connection")
        time.sleep(random.uniform(0.5, 1.5))
        print(f"Thread {thread_id}: released connection")
                                            # release (count +1) on exit

threads = [threading.Thread(target=use_db_connection, args=(i,)) for i in range(7)]
for t in threads: t.start()
for t in threads: t.join()
# At most 3 "got connection" lines active at any time
```

### 2) `BoundedSemaphore` — Prevent over-release (防止超额释放)

<span style="color:#C0392B;font-weight:600">Warning: a plain `Semaphore` allows `release()` beyond the initial value — this is usually a bug. `BoundedSemaphore` raises `ValueError` if the count would exceed the initial value.</span>

```python
import threading

sem   = threading.Semaphore(2)
bsem  = threading.BoundedSemaphore(2)

# Plain Semaphore — silently over-releases
sem.release()   # count goes to 3 — no error (潜在bug)
print(f"Semaphore value after over-release: OK (silent)")

# BoundedSemaphore — raises ValueError
try:
    bsem.release()   # count would exceed 2
except ValueError as e:
    print(f"BoundedSemaphore caught: {e}")
# → BoundedSemaphore caught: Semaphore released too many times
```

### 3) Rate limiter pattern (限速器模式)

```python
import threading, time

# Limit to 2 concurrent API calls
api_semaphore = threading.BoundedSemaphore(2)

def call_api(endpoint):
    with api_semaphore:
        print(f"Calling {endpoint}")
        time.sleep(1)   # simulate API latency
        print(f"Done    {endpoint}")

endpoints = [f"/api/resource/{i}" for i in range(6)]
threads   = [threading.Thread(target=call_api, args=(ep,)) for ep in endpoints]

for t in threads: t.start()
for t in threads: t.join()
```

------

## 6. Event — Simple Flag Signaling (事件信号)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> An <span style="color:#E8600A;font-weight:700">Event (事件)</span> is a simple boolean flag. Threads can <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">wait()</code> until the flag is set, and any thread can <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">set()</code> or <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">clear()</code> it. </div>

### 1) `Event.set()` / `Event.clear()` / `Event.wait()` / `Event.is_set()`

```python
import threading, time

start_event = threading.Event()

def worker(name):
    print(f"[{name}] waiting for start signal...")
    start_event.wait()               # blocks until event is set
    print(f"[{name}] GO! Starting work")

workers = [threading.Thread(target=worker, args=(f"W{i}",)) for i in range(4)]
for w in workers: w.start()

print("Main: preparing...")
time.sleep(2)
print("Main: firing start signal!")
start_event.set()                    # wake ALL waiting threads at once

for w in workers: w.join()
# → [W0] waiting for start signal...
# → [W1] waiting for start signal...
# → [W2] waiting for start signal...
# → [W3] waiting for start signal...
# (2s pause)
# → Main: firing start signal!
# → [W0] GO! Starting work    (all 4 unblock simultaneously)
```

### 2) `Event.wait(timeout=N)` — Timed wait

```python
import threading, time

ready = threading.Event()

def service():
    print("Service: initializing (takes 3s)...")
    time.sleep(3)
    ready.set()
    print("Service: ready!")

def client():
    if ready.wait(timeout=1.5):    # only wait 1.5s
        print("Client: connected!")
    else:
        print("Client: service not ready in time, aborting")

t1 = threading.Thread(target=service)
t2 = threading.Thread(target=client)
t1.start(); t2.start()
t1.join();  t2.join()
# → Service: initializing (takes 3s)...
# → Client: service not ready in time, aborting
# → Service: ready!
```

### 3) Stop signal pattern (停止信号模式)

```python
import threading, time

stop_event = threading.Event()

def background_worker():
    count = 0
    while not stop_event.is_set():    # check flag each iteration
        print(f"Working... iteration {count}")
        count += 1
        time.sleep(0.5)
    print("Worker: received stop signal, exiting cleanly")

t = threading.Thread(target=background_worker)
t.start()

time.sleep(2)
print("Main: sending stop signal")
stop_event.set()
t.join()
```

------

## 7. Timer — Delayed Execution (延迟执行)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">threading.Timer</code> is a subclass of <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Thread</code> that executes a function after a specified delay. It can be <span style="color:#2980B9">cancelled</span> before firing. </div>

### 1) Basic Timer

```python
import threading

def reminder(message):
    print(f"⏰ Reminder: {message}")

# Fire after 3 seconds
t = threading.Timer(3.0, reminder, args=("Meeting at 3pm!",))
t.start()

print("Timer set. Waiting...")
t.join()
# → Timer set. Waiting...
# (3s pause)
# → ⏰ Reminder: Meeting at 3pm!
```

### 2) `Timer.cancel()` — Cancel before firing

```python
import threading, time

fired = False

def action():
    global fired
    fired = True
    print("Action fired!")

t = threading.Timer(5.0, action)
t.start()

time.sleep(1)
t.cancel()    # ← cancel within the window
t.join()

print(f"Action fired: {fired}")   # → Action fired: False
```

### 3) Repeating timer pattern (重复定时器模式)

```python
import threading

class RepeatingTimer:
    """Fires a function every `interval` seconds."""

    def __init__(self, interval: float, func, *args):
        self.interval = interval
        self.func     = func
        self.args     = args
        self._timer   = None
        self._running = False

    def _run(self):
        self.func(*self.args)
        if self._running:
            self._schedule()

    def _schedule(self):
        self._timer = threading.Timer(self.interval, self._run)
        self._timer.daemon = True
        self._timer.start()

    def start(self):
        self._running = True
        self._schedule()

    def stop(self):
        self._running = False
        if self._timer:
            self._timer.cancel()

import time

counter = [0]
def tick():
    counter[0] += 1
    print(f"Tick #{counter[0]}")

rt = RepeatingTimer(0.5, tick)
rt.start()
time.sleep(2.5)
rt.stop()
print(f"Total ticks: {counter[0]}")   # → Total ticks: 5
```

------

## 8. Barrier — Thread Synchronization Point (屏障同步点)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> A <span style="color:#E8600A;font-weight:700">Barrier (屏障)</span> makes a fixed number of threads wait at a rendezvous point until ALL of them arrive — then releases all of them simultaneously. </div>

### 1) `Barrier(parties, action=None, timeout=None)`

```python
import threading, time, random

NUM_WORKERS = 4
barrier = threading.Barrier(NUM_WORKERS)

def phase_worker(name):
    # Phase 1
    duration = random.uniform(0.5, 2.0)
    print(f"[{name}] phase 1 working for {duration:.1f}s")
    time.sleep(duration)
    print(f"[{name}] phase 1 done — waiting at barrier")

    barrier.wait()     # ← all threads block here until all 4 arrive

    print(f"[{name}] phase 2 starting (all threads released together)")

threads = [threading.Thread(target=phase_worker, args=(f"W{i}",))
           for i in range(NUM_WORKERS)]
for t in threads: t.start()
for t in threads: t.join()
```

### 2) `Barrier` with `action` callback

```python
import threading, time

def setup_phase():
    """Runs ONCE when all threads reach the barrier, before release."""
    print(">>> All threads ready — running barrier action <<<")

barrier = threading.Barrier(3, action=setup_phase)

def worker(name):
    time.sleep(0.1)
    print(f"[{name}] arrived at barrier")
    barrier.wait()
    print(f"[{name}] past barrier")

threads = [threading.Thread(target=worker, args=(f"T{i}",)) for i in range(3)]
for t in threads: t.start()
for t in threads: t.join()
```

### 3) `Barrier.abort()` / `BrokenBarrierError`

```python
import threading, time

barrier = threading.Barrier(3)

def risky_worker(name, should_abort):
    try:
        if should_abort:
            time.sleep(0.2)
            print(f"[{name}] aborting barrier!")
            barrier.abort()          # breaks the barrier for everyone
        else:
            print(f"[{name}] waiting at barrier...")
            barrier.wait(timeout=2)
            print(f"[{name}] passed!")
    except threading.BrokenBarrierError:
        print(f"[{name}] barrier was broken — handling gracefully")

threads = [
    threading.Thread(target=risky_worker, args=("T0", False)),
    threading.Thread(target=risky_worker, args=("T1", False)),
    threading.Thread(target=risky_worker, args=("T2", True)),   # aborts
]
for t in threads: t.start()
for t in threads: t.join()
```

### 4) Barrier properties

```python
import threading

b = threading.Barrier(5)
print(b.parties)    # → 5   (total threads needed)
print(b.n_waiting)  # → 0   (currently waiting)
print(b.broken)     # → False
```

------

## 9. local — Thread-local Storage (线程本地存储)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">threading.local()</code> creates an object where each thread has its <span style="color:#E8600A;font-weight:700">own independent copy</span> of every attribute. Ideal for thread-specific state like database connections or request contexts. </div>

### 1) Basic thread-local usage

```python
import threading

local_data = threading.local()

def worker(value):
    local_data.x = value              # each thread sets its own .x
    import time; time.sleep(0.1)      # let other threads run
    print(f"Thread {threading.current_thread().name}: x = {local_data.x}")

threads = [threading.Thread(target=worker, args=(i*10,), name=f"T{i}")
           for i in range(4)]
for t in threads: t.start()
for t in threads: t.join()
# → Thread T0: x = 0
# → Thread T1: x = 10
# → Thread T2: x = 20
# → Thread T3: x = 30
# (each thread sees only its own value — no interference)
```

### 2) Thread-local DB connection pattern

```python
import threading
import sqlite3

_local = threading.local()

def get_connection(db_path: str) -> sqlite3.Connection:
    """Return a per-thread DB connection (创建线程私有数据库连接)."""
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(db_path)
        print(f"[{threading.current_thread().name}] created new connection")
    return _local.conn

def db_worker(db_path: str):
    conn = get_connection(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS t (v INTEGER)")
    conn.execute("INSERT INTO t VALUES (?)", (threading.get_ident(),))
    conn.commit()
    print(f"[{threading.current_thread().name}] inserted row")

threads = [threading.Thread(target=db_worker, args=(":memory:",), name=f"DB-{i}")
           for i in range(3)]
for t in threads: t.start()
for t in threads: t.join()
```

### 3) Subclass `local` for initialization

```python
import threading

class RequestContext(threading.local):
    """Thread-local request context with defaults."""
    def __init__(self):
        super().__init__()
        self.user_id   = None
        self.request_id = None

ctx = RequestContext()

def handle_request(user_id, req_id):
    ctx.user_id    = user_id
    ctx.request_id = req_id
    import time; time.sleep(0.05)
    print(f"Processing request {ctx.request_id} for user {ctx.user_id}")

threads = [threading.Thread(target=handle_request, args=(f"user{i}", f"req-{i:03}"))
           for i in range(4)]
for t in threads: t.start()
for t in threads: t.join()
```

------

## 10. Module-level Functions (模块级函数)

### 1) `threading.current_thread()` — Get the current thread

```python
import threading

def show_self():
    t = threading.current_thread()
    print(f"name={t.name}, ident={t.ident}, daemon={t.daemon}")

main_t = threading.current_thread()
print(f"Main thread: {main_t.name}")

t = threading.Thread(target=show_self, name="MyWorker")
t.start(); t.join()
# → Main thread: MainThread
# → name=MyWorker, ident=140..., daemon=False
```

### 2) `threading.main_thread()` — Get the main thread

```python
import threading

def check_main():
    mt = threading.main_thread()
    ct = threading.current_thread()
    print(f"Main thread: {mt.name}")
    print(f"This thread: {ct.name}")
    print(f"Am I main?  {ct is mt}")

t = threading.Thread(target=check_main)
t.start(); t.join()
# → Main thread: MainThread
# → This thread: Thread-1
# → Am I main?  False
```

### 3) `threading.active_count()` — Count live threads

```python
import threading, time

def slow():
    time.sleep(2)

print(threading.active_count())   # → 1  (main only)

threads = [threading.Thread(target=slow) for _ in range(3)]
for t in threads: t.start()

print(threading.active_count())   # → 4  (main + 3 workers)
for t in threads: t.join()
print(threading.active_count())   # → 1
```

### 4) `threading.enumerate()` — List all live threads

```python
import threading, time

def task(n):
    time.sleep(n)

threads = [threading.Thread(target=task, args=(i,), name=f"T{i}") for i in range(1,4)]
for t in threads: t.start()

for t in threading.enumerate():
    print(f"  alive: {t.name} | daemon={t.daemon}")
# → alive: MainThread | daemon=False
# → alive: T1        | daemon=False
# → alive: T2        | daemon=False
# → alive: T3        | daemon=False

for t in threads: t.join()
```

### 5) `threading.settrace(func)` / `threading.setprofile(func)` — Thread hooks

```python
import threading, sys

def my_tracer(frame, event, arg):
    if event == "call":
        print(f"[TRACE] calling {frame.f_code.co_name}")
    return my_tracer

def task():
    x = 1 + 1
    return x

threading.settrace(my_tracer)    # set trace for ALL future threads
t = threading.Thread(target=task)
t.start(); t.join()
threading.settrace(None)         # remove tracer
```

### 6) `threading.stack_size(size=0)` — Set thread stack size

```python
import threading

# Set stack size to 512 KB for all future threads
threading.stack_size(512 * 1024)
print(f"Stack size: {threading.stack_size()} bytes")

def task():
    print(f"Running with custom stack size")

t = threading.Thread(target=task)
t.start(); t.join()

threading.stack_size(0)   # reset to default
```

### 7) `threading.excepthook` — Handle uncaught thread exceptions (未捕获异常处理)

```python
import threading

def custom_excepthook(args):
    print(f"Uncaught exception in thread [{args.thread.name}]:")
    print(f"  Type:    {args.exc_type.__name__}")
    print(f"  Message: {args.exc_value}")

threading.excepthook = custom_excepthook

def buggy_task():
    raise ValueError("Something went wrong in thread!")

t = threading.Thread(target=buggy_task, name="BuggyThread")
t.start(); t.join()
# → Uncaught exception in thread [BuggyThread]:
# →   Type:    ValueError
# →   Message: Something went wrong in thread!
```

### 8) `threading.get_ident()` / `threading.get_native_id()`

```python
import threading

def show_ids():
    print(f"Python ident:    {threading.get_ident()}")
    print(f"OS native id:    {threading.get_native_id()}")

t = threading.Thread(target=show_ids)
t.start(); t.join()
```

------

## 11. queue Module — Thread-safe Queues (线程安全队列)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> The <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">queue</code> module provides three thread-safe queue classes: <span style="color:#E8600A;font-weight:700">Queue (FIFO)</span>, <span style="color:#E8600A;font-weight:700">LifoQueue (LIFO/stack)</span>, and <span style="color:#E8600A;font-weight:700">PriorityQueue (优先队列)</span>. All use internal locks, so no external synchronization is needed. </div>

### 1) `Queue(maxsize=0)` — FIFO Queue

```python
from queue import Queue
import threading, time

q = Queue(maxsize=3)

def producer():
    for i in range(6):
        q.put(i)          # blocks if queue is full (maxsize reached)
        print(f"Put {i}  | qsize={q.qsize()}")
        time.sleep(0.2)

def consumer():
    for _ in range(6):
        item = q.get()    # blocks if queue is empty
        print(f"Got {item}")
        q.task_done()
        time.sleep(0.5)

t1 = threading.Thread(target=producer)
t2 = threading.Thread(target=consumer)
t1.start(); t2.start()
t1.join();  t2.join()
```

### 2) `Queue.put_nowait()` / `Queue.get_nowait()` — Non-blocking

```python
from queue import Queue, Full, Empty

q = Queue(maxsize=2)
q.put("item1")
q.put("item2")

try:
    q.put_nowait("item3")     # queue full!
except Full:
    print("Queue full — item3 dropped")

try:
    while True:
        print(q.get_nowait())
except Empty:
    print("Queue emptied")
# → Queue full — item3 dropped
# → item1
# → item2
# → Queue emptied
```

### 3) `Queue.join()` / `Queue.task_done()` — Work tracking

```python
from queue import Queue
import threading

work_queue = Queue()

def worker():
    while True:
        task = work_queue.get()
        if task is None:
            break
        print(f"Processing: {task}")
        work_queue.task_done()   # signal this task is complete

# Start 3 workers
workers = [threading.Thread(target=worker, daemon=True) for _ in range(3)]
for w in workers: w.start()

# Enqueue tasks
for task in ["task_A", "task_B", "task_C", "task_D", "task_E"]:
    work_queue.put(task)

work_queue.join()   # blocks until ALL task_done() called
print("All tasks completed!")
```

### 4) `LifoQueue` — Stack (栈/后进先出)

```python
from queue import LifoQueue

stack = LifoQueue()
stack.put("first")
stack.put("second")
stack.put("third")

while not stack.empty():
    print(stack.get())
# → third
# → second
# → first
```

### 5) `PriorityQueue` — Priority-based processing (优先级队列)

```python
from queue import PriorityQueue
import threading, time

pq = PriorityQueue()

# (priority, task_name) — lower number = higher priority
pq.put((3, "low-priority task"))
pq.put((1, "URGENT task"))
pq.put((2, "medium task"))
pq.put((1, "another URGENT task"))

while not pq.empty():
    priority, task = pq.get()
    print(f"[priority={priority}] Processing: {task}")
# → [priority=1] Processing: URGENT task
# → [priority=1] Processing: another URGENT task
# → [priority=2] Processing: medium task
# → [priority=3] Processing: low-priority task
```

------

## 12. ThreadPoolExecutor — High-level Thread Pool (高级线程池)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">concurrent.futures.ThreadPoolExecutor</code> provides a high-level, <span style="color:#2980B9">Future-based (Future对象)</span> interface for thread pools. It is the <span style="color:#E8600A;font-weight:700">recommended way</span> to run IO-bound tasks in modern Python. </div>

### 1) `submit()` → Future

```python
from concurrent.futures import ThreadPoolExecutor
import time

def fetch_data(url: str) -> str:
    time.sleep(1)   # simulate network call
    return f"<data from {url}>"

urls = [f"http://example.com/page{i}" for i in range(5)]

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(fetch_data, url) for url in urls]

    for future in futures:
        result = future.result()   # blocks until this future completes
        print(result)
```

### 2) `map()` — Parallel map (并行映射)

```python
from concurrent.futures import ThreadPoolExecutor
import time

def square(n):
    time.sleep(0.2)
    return n * n

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(square, range(10)))

print(results)   # → [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

### 3) `Future` API — `done()`, `cancel()`, `add_done_callback()`

```python
from concurrent.futures import ThreadPoolExecutor
import time

def slow_task(n):
    time.sleep(n)
    return f"result-{n}"

def on_done(future):
    print(f"Callback: task finished → {future.result()}")

with ThreadPoolExecutor(max_workers=2) as executor:
    f1 = executor.submit(slow_task, 1)
    f2 = executor.submit(slow_task, 2)

    f1.add_done_callback(on_done)    # register callback
    f2.add_done_callback(on_done)

    print(f"f1 done: {f1.done()}")   # likely False (still running)
    time.sleep(1.5)
    print(f"f1 done: {f1.done()}")   # → True
```

### 4) `as_completed()` — Process in completion order (按完成顺序处理)

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import time, random

def task(n):
    delay = random.uniform(0.1, 1.0)
    time.sleep(delay)
    return (n, delay)

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(task, i): i for i in range(8)}

    for future in as_completed(futures):
        task_id = futures[future]
        n, delay = future.result()
        print(f"Task {n} finished in {delay:.2f}s")
# Tasks print in the order they complete, not submission order
```

### 5) Exception handling in futures (Future异常处理)

```python
from concurrent.futures import ThreadPoolExecutor

def risky(x):
    if x == 3:
        raise ValueError(f"Bad input: {x}")
    return x * 2

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(risky, i) for i in range(5)]

for i, f in enumerate(futures):
    try:
        print(f"Result {i}: {f.result()}")
    except ValueError as e:
        print(f"Result {i}: ERROR — {e}")
# → Result 0: 0
# → Result 1: 2
# → Result 2: 4
# → Result 3: ERROR — Bad input: 3
# → Result 4: 8
```

------

## 13. Common Patterns & Pitfalls (常见模式与陷阱)

### 1) Race condition example (竞态条件示例)

```python
import threading

counter = 0   # UNSAFE shared state

def unsafe_increment():
    global counter
    for _ in range(100_000):
        counter += 1   # NOT atomic! (read-modify-write)

threads = [threading.Thread(target=unsafe_increment) for _ in range(5)]
for t in threads: t.start()
for t in threads: t.join()

print(f"Expected: 500000")
print(f"Actual:   {counter}")   # likely LESS than 500000 — data race!
```

### 2) Deadlock example + fix (死锁示例及修复)

```python
import threading

lock_a = threading.Lock()
lock_b = threading.Lock()

# ─── DEADLOCK version ────────────────────────────────
def thread1_deadlock():
    with lock_a:
        import time; time.sleep(0.1)
        with lock_b:                  # waits for lock_b
            print("T1: got both locks")

def thread2_deadlock():
    with lock_b:
        import time; time.sleep(0.1)
        with lock_a:                  # waits for lock_a → DEADLOCK
            print("T2: got both locks")

# ─── FIXED version: always acquire locks in the same order ──
def thread1_safe():
    with lock_a:                      # acquire A first
        with lock_b:                  # then B
            print("T1 safe: got both locks")

def thread2_safe():
    with lock_a:                      # acquire A first (same order!)
        with lock_b:
            print("T2 safe: got both locks")

t1 = threading.Thread(target=thread1_safe)
t2 = threading.Thread(target=thread2_safe)
t1.start(); t2.start()
t1.join();  t2.join()
# → T1 safe: got both locks
# → T2 safe: got both locks
```

### 3) Thread-safe singleton (线程安全单例)

```python
import threading

class Singleton:
    _instance = None
    _lock     = threading.Lock()

    def __new__(cls):
        if cls._instance is None:              # first check (no lock)
            with cls._lock:
                if cls._instance is None:      # second check (with lock)
                    cls._instance = super().__new__(cls)
                    print("Singleton created")
        return cls._instance

def get_instance():
    s = Singleton()
    print(f"Got instance: {id(s)}")

threads = [threading.Thread(target=get_instance) for _ in range(5)]
for t in threads: t.start()
for t in threads: t.join()
# → Singleton created   (exactly once)
# → Got instance: 140...  (same id for all 5 threads)
```

------

## 14. Full API Quick Reference (API速查表)

| Class / Function                                             | Key Methods                                     | Purpose                           |
| ------------------------------------------------------------ | ----------------------------------------------- | --------------------------------- |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Thread</code> | `start()` `join()` `is_alive()`                 | Create and manage threads         |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Lock</code> | `acquire()` `release()` `locked()`              | Mutual exclusion                  |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">RLock</code> | `acquire()` `release()`                         | Reentrant mutual exclusion        |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Condition</code> | `wait()` `wait_for()` `notify()` `notify_all()` | Wait/notify synchronization       |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Semaphore</code> | `acquire()` `release()`                         | Limit concurrent access           |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">BoundedSemaphore</code> | `acquire()` `release()`                         | Semaphore with over-release guard |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Event</code> | `set()` `clear()` `wait()` `is_set()`           | Boolean flag signaling            |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Timer</code> | `start()` `cancel()`                            | Delayed / cancellable execution   |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Barrier</code> | `wait()` `abort()` `reset()`                    | N-thread rendezvous point         |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">local</code> | attribute access                                | Per-thread storage                |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Queue</code> | `put()` `get()` `task_done()` `join()`          | Thread-safe FIFO queue            |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">LifoQueue</code> | `put()` `get()`                                 | Thread-safe stack                 |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">PriorityQueue</code> | `put()` `get()`                                 | Thread-safe priority queue        |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">ThreadPoolExecutor</code> | `submit()` `map()` `shutdown()`                 | High-level thread pool            |
| <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">current_thread()</code> | —                                               | Get current Thread object         |
| <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">active_count()</code> | —                                               | Count live threads                |
| <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">enumerate()</code> | —                                               | List all live threads             |
| <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">excepthook</code> | —                                               | Handle uncaught thread exceptions |

------

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> Python threading excels at <span style="color:#E8600A;font-weight:700">IO-bound concurrency (IO密集型并发)</span>: use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">ThreadPoolExecutor</code> for simple task pools, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Queue</code> for producer-consumer pipelines, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Lock</code>/<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">RLock</code> for shared state, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Event</code> for signaling, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Semaphore</code> for resource pools, and <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Barrier</code> for multi-phase synchronization — always protect shared mutable state to avoid <span style="color:#C0392B;font-weight:600">Race Conditions (竞态条件)</span> and <span style="color:#C0392B;font-weight:600">Deadlocks (死锁)</span>. </div>

------

# **II. When to Use Each API — Scenario Decision Guide (使用场景决策指南)**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Choosing the wrong synchronization primitive is a common source of bugs, deadlocks, and poor performance. This chapter maps every threading API to its <span style="color:#E8600A;font-weight:700">concrete real-world scenarios</span>, explains the <span style="color:#2980B9">decision logic</span> behind each choice, and provides a final <span style="color:#E8600A;font-weight:700">Decision Flowchart (决策流程图)</span> for quick lookup. </div>

------

## 1. Thread — When to create raw threads (何时创建原始线程)

### 1) ✅ Use `Thread` directly when

<span style="color:#E8600A">1.</span> You need **full lifecycle control** — start, monitor, join at precise moments. <span style="color:#E8600A">2.</span> The thread has **long-running, stateful logic** best expressed as a class with `run()`. <span style="color:#E8600A">3.</span> You need to store a **result on the thread object** itself (`self.result = ...`). <span style="color:#E8600A">4.</span> You're building a **daemon background service** (heartbeat, log flusher, monitor).

```python
# ✅ Scenario: long-lived stateful background service
import threading, time

class HeartbeatThread(threading.Thread):
    """Sends periodic heartbeats to a server."""
    def __init__(self, server_url, interval=5):
        super().__init__(daemon=True, name="Heartbeat")
        self.server_url = server_url
        self.interval   = interval
        self._stop      = threading.Event()

    def run(self):
        while not self._stop.is_set():
            print(f"[Heartbeat] ping → {self.server_url}")
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()

hb = HeartbeatThread("http://api.example.com/health")
hb.start()
time.sleep(12)
hb.stop()
```

### 2) ❌ Do NOT use raw `Thread` when

<span style="color:#C0392B;font-weight:600">× You just need to run many short tasks in parallel → use `ThreadPoolExecutor` instead.</span> <span style="color:#C0392B;font-weight:600">× You need return values from many tasks → `Future.result()` is cleaner than `t.result`.</span> <span style="color:#C0392B;font-weight:600">× You need CPU parallelism → use `multiprocessing` (GIL blocks true parallelism).</span>

### 3) `daemon=True` — Specifically when

Use daemon threads for tasks that should **not keep the program alive** if the main thread exits:

| Scenario                  | `daemon=True` | `daemon=False` |
| ------------------------- | ------------- | -------------- |
| Background log flusher    | ✅             | —              |
| Health monitor / watchdog | ✅             | —              |
| Worker that must finish   | —             | ✅              |
| DB write that must commit | —             | ✅              |

```python
# ✅ Scenario: log flusher that should die with the app
import threading, time

log_buffer = []

def flush_logs():
    while True:
        if log_buffer:
            print(f"[Flush] writing {len(log_buffer)} log entries")
            log_buffer.clear()
        time.sleep(1)

flusher = threading.Thread(target=flush_logs, daemon=True)
flusher.start()

# Main thread does work, flusher auto-dies when main exits
for i in range(5):
    log_buffer.append(f"event-{i}")
    time.sleep(0.5)
print("Main done — flusher daemon killed automatically")
```

------

## 2. Lock — When to use mutual exclusion (何时使用互斥锁)

### 1) ✅ Use `Lock` when

<span style="color:#E8600A">1.</span> Multiple threads **read AND write** the same variable / data structure. <span style="color:#E8600A">2.</span> An operation that looks atomic is actually **read-modify-write** (e.g. `counter += 1`). <span style="color:#E8600A">3.</span> You're updating a **shared list, dict, or custom object**. <span style="color:#E8600A">4.</span> You need to protect a **file write** or **database update**.

```python
# ✅ Scenario: shared bank account balance — MUST use Lock
import threading

class Account:
    def __init__(self, balance):
        self.balance = balance
        self._lock   = threading.Lock()

    def transfer(self, amount):
        with self._lock:                  # critical section
            if self.balance >= amount:
                time.sleep(0.001)         # simulate DB latency
                self.balance -= amount
                return True
            return False

import time
acc     = Account(1000)
results = []

def try_withdraw():
    results.append(acc.transfer(100))

threads = [threading.Thread(target=try_withdraw) for _ in range(20)]
for t in threads: t.start()
for t in threads: t.join()

print(f"Balance: {acc.balance}")         # always ≥ 0
print(f"Successful: {results.count(True)}")
```

### 2) ❌ Do NOT use `Lock` when

<span style="color:#C0392B;font-weight:600">× The same thread needs to acquire the lock twice → use `RLock` instead (plain Lock deadlocks).</span> <span style="color:#C0392B;font-weight:600">× You need to wait for a condition, not just exclusive access → use `Condition`.</span> <span style="color:#C0392B;font-weight:600">× You only need to limit concurrency to N > 1 → use `Semaphore`.</span>

### 3) Scenario matrix (场景矩阵)

| Situation                                   | Correct primitive                                            |
| ------------------------------------------- | ------------------------------------------------------------ |
| 1 thread at a time, non-reentrant           | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Lock</code> |
| 1 thread at a time, same thread re-acquires | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">RLock</code> |
| N threads at a time                         | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Semaphore(N)</code> |
| Wait until data is ready                    | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Condition</code> |
| One-time go signal                          | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Event</code> |

------

## 3. RLock — When re-entrancy is needed (何时需要可重入锁)

### 1) ✅ Use `RLock` when

<span style="color:#E8600A">1.</span> A method holding the lock **calls another method that also acquires the same lock**. <span style="color:#E8600A">2.</span> You're building a **class with multiple synchronized methods** that call each other. <span style="color:#E8600A">3.</span> You have **recursive algorithms** that need locking at each level.

```python
# ✅ Scenario: tree traversal where each node uses the same lock
import threading

class SafeTree:
    def __init__(self, value, children=None):
        self.value    = value
        self.children = children or []
        self._lock    = threading.RLock()

    def sum_values(self):
        with self._lock:                        # acquire (depth 1)
            total = self.value
            for child in self.children:
                total += child.sum_values()     # same lock, deeper (depth 2+)
            return total

tree = SafeTree(1, [SafeTree(2), SafeTree(3, [SafeTree(4)])])
t = threading.Thread(target=lambda: print(f"Sum: {tree.sum_values()}"))
t.start(); t.join()
# → Sum: 10
```

### 2) ❌ Do NOT use `RLock` when

<span style="color:#C0392B;font-weight:600">× Methods don't call each other — a plain `Lock` has slightly lower overhead.</span> <span style="color:#C0392B;font-weight:600">× You want to detect accidental re-entry as a bug — `Lock` will surface it as a deadlock.</span>

------

## 4. Condition — When threads must wait for state changes (何时等待状态变化)

### 1) ✅ Use `Condition` when

<span style="color:#E8600A">1.</span> One thread must **wait until another thread changes some data** (not just unlocks). <span style="color:#E8600A">2.</span> Implementing **producer-consumer** patterns with a bounded buffer. <span style="color:#E8600A">3.</span> Threads need to **coordinate in phases** — e.g., "wait until queue has ≥ 3 items". <span style="color:#E8600A">4.</span> You need **selective wakeup** — notify only one waiter vs. all waiters.

```python
# ✅ Scenario: order fulfillment system
# Orders must wait until inventory is restocked
import threading, time, collections

inventory = collections.defaultdict(int)
cond      = threading.Condition()

def restock_worker():
    items = [("apple", 50), ("banana", 30), ("cherry", 20)]
    for item, qty in items:
        time.sleep(1)
        with cond:
            inventory[item] += qty
            print(f"[Restock] {item} +{qty} → total={inventory[item]}")
            cond.notify_all()   # wake all waiting orders

def process_order(order_id, item, qty):
    with cond:
        cond.wait_for(lambda: inventory[item] >= qty)  # wait for stock
        inventory[item] -= qty
        print(f"[Order {order_id}] filled {qty}x {item} → remaining={inventory[item]}")

threads = [
    threading.Thread(target=restock_worker),
    threading.Thread(target=process_order, args=(1, "apple",  20)),
    threading.Thread(target=process_order, args=(2, "banana", 15)),
    threading.Thread(target=process_order, args=(3, "apple",  40)),
]
for t in threads: t.start()
for t in threads: t.join()
```

### 2) `notify()` vs `notify_all()` — When to use which

| Situation                                           | Use                                                          |
| --------------------------------------------------- | ------------------------------------------------------------ |
| Only **one** consumer can act (e.g. one slot freed) | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">notify()</code> |
| **All** consumers might now be able to proceed      | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">notify_all()</code> |
| You added multiple items to the buffer at once      | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">notify_all()</code> |
| Only one thread is waiting (guaranteed)             | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">notify()</code> |

### 3) ❌ Do NOT use `Condition` when

<span style="color:#C0392B;font-weight:600">× You just need a one-time signal → use `Event` (simpler API).</span> <span style="color:#C0392B;font-weight:600">× The data flowing between threads is the signal → use `Queue` (built-in blocking).</span>

------

## 5. Semaphore — When limiting concurrent access (何时限制并发访问数量)

### 1) ✅ Use `Semaphore` when

<span style="color:#E8600A">1.</span> You have a **resource pool** with a fixed capacity: DB connections, HTTP clients, file handles. <span style="color:#E8600A">2.</span> You need **rate limiting** — at most N concurrent API calls. <span style="color:#E8600A">3.</span> Implementing a **thread pool** from scratch (though `ThreadPoolExecutor` is preferred). <span style="color:#E8600A">4.</span> A resource requires **N permits** to use (e.g. a GPU with N memory slots).

```python
# ✅ Scenario: limit concurrent external API calls to avoid 429 Too Many Requests
import threading, time, random

MAX_CONCURRENT = 3
api_semaphore  = threading.BoundedSemaphore(MAX_CONCURRENT)

def call_external_api(request_id):
    print(f"[Req {request_id}] queued")
    with api_semaphore:
        print(f"[Req {request_id}] calling API...")
        time.sleep(random.uniform(0.5, 1.5))   # simulate API latency
        print(f"[Req {request_id}] done")

# Simulate 10 concurrent requests — only 3 run at once
threads = [threading.Thread(target=call_external_api, args=(i,)) for i in range(10)]
for t in threads: t.start()
for t in threads: t.join()
```

### 2) `Semaphore` vs `BoundedSemaphore` — When to use which

| Situation                                       | Use                                                          |
| ----------------------------------------------- | ------------------------------------------------------------ |
| Resource pool (connection pool, thread pool)    | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">BoundedSemaphore</code> — prevents logic bugs |
| Signaling between threads (producer increments) | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Semaphore</code> — counter can exceed initial |
| You want a runtime error on over-release        | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">BoundedSemaphore</code> |

### 3) ❌ Do NOT use `Semaphore` when

<span style="color:#C0392B;font-weight:600">× You only need to allow 1 thread at a time → use `Lock` (clearer intent).</span> <span style="color:#C0392B;font-weight:600">× You need workers to process tasks from a queue → use `ThreadPoolExecutor`.</span>

------

## 6. Event — When broadcasting a one-time signal (何时广播一次性信号)

### 1) ✅ Use `Event` when

<span style="color:#E8600A">1.</span> One thread needs to **signal multiple waiting threads simultaneously** (broadcast). <span style="color:#E8600A">2.</span> Implementing a **start gun** — all workers blocked until a "ready" signal fires. <span style="color:#E8600A">3.</span> A **graceful shutdown** flag — workers poll `stop_event.is_set()` each iteration. <span style="color:#E8600A">4.</span> A **service readiness probe** — clients wait until the server is initialized. <span style="color:#E8600A">5.</span> One-shot notifications where the flag **stays set** permanently after firing.

```python
# ✅ Scenario: web server workers wait for config to load before serving
import threading, time

config_loaded = threading.Event()
config        = {}

def load_config():
    print("[Config] loading from database...")
    time.sleep(2)
    config.update({"host": "0.0.0.0", "port": 8080, "debug": False})
    print("[Config] loaded!")
    config_loaded.set()           # broadcast to ALL waiting workers

def request_handler(worker_id):
    config_loaded.wait()          # block until config ready
    print(f"[Worker {worker_id}] serving on {config['host']}:{config['port']}")

threads = (
    [threading.Thread(target=load_config)] +
    [threading.Thread(target=request_handler, args=(i,)) for i in range(5)]
)
for t in threads: t.start()
for t in threads: t.join()
```

### 2) `Event` vs `Condition` — Decision rule

| Question                                              | Answer → Use                                                 |
| ----------------------------------------------------- | ------------------------------------------------------------ |
| Signal multiple threads with a permanent flag?        | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Event</code> |
| Wait for a data condition that can change repeatedly? | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Condition</code> |
| Need to reset and re-arm the signal?                  | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Event.clear()</code> or <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Condition</code> |
| One producer, many consumers woken at once?           | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Event</code> |

### 3) ❌ Do NOT use `Event` when

<span style="color:#C0392B;font-weight:600">× The condition can be true/false multiple times (e.g. buffer empty↔full) → use `Condition`.</span> <span style="color:#C0392B;font-weight:600">× You're passing data along with the signal → use `Queue`.</span>

------

## 7. Timer — When delaying or scheduling execution (何时延迟或定时执行)

### 1) ✅ Use `Timer` when

<span style="color:#E8600A">1.</span> You need to **run a function once after a delay**, without blocking the main thread. <span style="color:#E8600A">2.</span> The action might need to be **cancelled before it fires** (e.g. debouncing). <span style="color:#E8600A">3.</span> Implementing **timeouts** for external operations. <span style="color:#E8600A">4.</span> **Session expiry**, cache invalidation, or auto-logout after inactivity.

```python
# ✅ Scenario: debounce user input — only save after 500ms of inactivity
import threading

_save_timer = None

def debounced_save(content):
    global _save_timer
    if _save_timer:
        _save_timer.cancel()     # cancel previous pending save
    _save_timer = threading.Timer(0.5, do_save, args=(content,))
    _save_timer.start()

def do_save(content):
    print(f"[Save] writing: '{content}'")

# Rapid keystrokes — only the last one saves
import time
debounced_save("H")
debounced_save("He")
debounced_save("Hel")
debounced_save("Hell")
time.sleep(0.1)
debounced_save("Hello")
time.sleep(0.8)
# → [Save] writing: 'Hello'   (only once, after 500ms of quiet)
```

### 2) ❌ Do NOT use `Timer` when

<span style="color:#C0392B;font-weight:600">× You need recurring execution → build a RepeatingTimer (see §7.3 in Part I) or use `sched`.</span> <span style="color:#C0392B;font-weight:600">× You need sub-millisecond precision — `Timer` uses `time.sleep()` which is OS-dependent.</span> <span style="color:#C0392B;font-weight:600">× Complex scheduling (cron-like) → use `APScheduler` or `Celery`.</span>

------

## 8. Barrier — When threads must synchronize at a checkpoint (何时需要检查点同步)

### 1) ✅ Use `Barrier` when

<span style="color:#E8600A">1.</span> A computation has **multiple phases** and ALL threads must finish phase N before ANY starts phase N+1. <span style="color:#E8600A">2.</span> **Parallel simulation** — each timestep must complete across all worker threads before advancing. <span style="color:#E8600A">3.</span> **Test synchronization** — ensure all threads reach a certain point before asserting results. <span style="color:#E8600A">4.</span> **Coordinated startup** — all services initialized before traffic is allowed.

```python
# ✅ Scenario: parallel matrix computation with two phases
import threading, time, random

NUM_WORKERS = 4
barrier     = threading.Barrier(NUM_WORKERS)
partial_results = [0] * NUM_WORKERS
final_results   = [0] * NUM_WORKERS

def compute_worker(worker_id):
    # ── Phase 1: independent computation ──────────────
    time.sleep(random.uniform(0.3, 1.2))
    partial_results[worker_id] = random.randint(10, 100)
    print(f"[W{worker_id}] Phase 1 done: partial={partial_results[worker_id]}")

    barrier.wait()   # ← ALL workers must finish phase 1 before phase 2

    # ── Phase 2: needs ALL phase-1 results ────────────
    # e.g., normalize by global sum
    total = sum(partial_results)
    final_results[worker_id] = partial_results[worker_id] / total
    print(f"[W{worker_id}] Phase 2 done: final={final_results[worker_id]:.3f}")

threads = [threading.Thread(target=compute_worker, args=(i,)) for i in range(NUM_WORKERS)]
for t in threads: t.start()
for t in threads: t.join()

print(f"\nFinal results: {[f'{r:.3f}' for r in final_results]}")
print(f"Sum check: {sum(final_results):.6f}")   # → ~1.0
```

### 2) ❌ Do NOT use `Barrier` when

<span style="color:#C0392B;font-weight:600">× Thread count is dynamic (unknown at creation time) — Barrier requires a fixed `parties` count.</span> <span style="color:#C0392B;font-weight:600">× Only one thread needs to wait for others → use `Thread.join()` or `Event`.</span> <span style="color:#C0392B;font-weight:600">× Threads have different roles (not symmetric) → use `Condition` or `Queue`.</span>

------

## 9. threading.local — When isolating per-thread state (何时隔离线程私有状态)

### 1) ✅ Use `threading.local` when

<span style="color:#E8600A">1.</span> Each thread needs its **own copy of a connection** (DB, HTTP session, file handle). <span style="color:#E8600A">2.</span> You're building **middleware or frameworks** that attach request context per thread. <span style="color:#E8600A">3.</span> A global-looking variable must actually be **thread-specific** (e.g., current user, transaction ID). <span style="color:#E8600A">4.</span> Avoiding lock contention by giving each thread **its own cache**.

```python
# ✅ Scenario: per-thread HTTP session (connection pooling per thread)
import threading
import urllib.request

_local = threading.local()

def get_session():
    """Return a thread-local opener — no lock needed, no sharing."""
    if not hasattr(_local, "opener"):
        _local.opener = urllib.request.build_opener()
        print(f"[{threading.current_thread().name}] created new HTTP session")
    return _local.opener

def fetch(url):
    session = get_session()     # each thread gets its own
    # session.open(url) ...
    print(f"[{threading.current_thread().name}] fetching {url}")

threads = [threading.Thread(target=fetch, args=(f"http://example.com/{i}",),
                            name=f"Fetcher-{i}") for i in range(4)]
for t in threads: t.start()
for t in threads: t.join()
# Each thread creates exactly one session — no contention, no sharing
```

### 2) ❌ Do NOT use `threading.local` when

<span style="color:#C0392B;font-weight:600">× Threads need to share and pass data to each other → use `Queue` or shared objects with `Lock`.</span> <span style="color:#C0392B;font-weight:600">× Using `ThreadPoolExecutor` — threads are reused, old local state may persist unexpectedly.</span>

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">In a thread pool, worker threads are reused across tasks. If you use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">threading.local</code> inside a pool, <strong>always initialize the local value at the start of each task</strong>, not just on first access — otherwise task 2 on the same thread will see task 1's leftover state.</span></div>

------

## 10. Queue / LifoQueue / PriorityQueue — When passing data between threads (何时在线程间传递数据)

### 1) ✅ Use `Queue` when

<span style="color:#E8600A">1.</span> Implementing **producer-consumer** patterns — the queue IS the synchronization. <span style="color:#E8600A">2.</span> Work items need to be **processed in order** (FIFO). <span style="color:#E8600A">3.</span> You want **backpressure** — producers block when the buffer is full (`maxsize`). <span style="color:#E8600A">4.</span> You need **work completion tracking** via `task_done()` + `join()`.

```python
# ✅ Scenario: image processing pipeline
# Loader threads → Queue → Processor threads → Queue → Writer threads
import threading, time, queue

raw_queue       = queue.Queue(maxsize=10)
processed_queue = queue.Queue(maxsize=10)

def loader(n_images):
    for i in range(n_images):
        time.sleep(0.1)
        raw_queue.put(f"image_{i:03}.jpg")
        print(f"[Loader] queued image_{i:03}.jpg")
    raw_queue.put(None)   # sentinel (哨兵值)

def processor():
    while True:
        item = raw_queue.get()
        if item is None:
            processed_queue.put(None)
            raw_queue.task_done()
            break
        result = f"processed_{item}"
        time.sleep(0.2)   # simulate processing
        processed_queue.put(result)
        raw_queue.task_done()

def writer():
    while True:
        item = processed_queue.get()
        if item is None:
            processed_queue.task_done()
            break
        print(f"[Writer] saved {item}")
        processed_queue.task_done()

threads = [
    threading.Thread(target=loader,    args=(5,)),
    threading.Thread(target=processor),
    threading.Thread(target=writer),
]
for t in threads: t.start()
for t in threads: t.join()
```

### 2) ✅ Use `LifoQueue` when

<span style="color:#E8600A">1.</span> Most-recently-added tasks are more **cache-warm** or likely to be **more relevant**. <span style="color:#E8600A">2.</span> Implementing **depth-first search** with worker threads. <span style="color:#E8600A">3.</span> Worker threads processing **undo stacks** or **rollback operations**.

### 3) ✅ Use `PriorityQueue` when

<span style="color:#E8600A">1.</span> Tasks have different **urgency levels** — critical tasks skip the queue. <span style="color:#E8600A">2.</span> Implementing a **task scheduler** with priority (e.g., real-time vs. batch jobs). <span style="color:#E8600A">3.</span> **Retry logic** — failed tasks re-enqueued with higher priority.

```python
# ✅ Scenario: multi-tier job scheduler
import threading, queue, time

job_queue = queue.PriorityQueue()

# Priority levels (优先级级别)
CRITICAL = 1
HIGH     = 2
NORMAL   = 3
BATCH    = 4

def scheduler():
    while True:
        try:
            priority, job_id, task = job_queue.get(timeout=2)
            print(f"[Scheduler] running [{['','CRITICAL','HIGH','NORMAL','BATCH'][priority]}] {job_id}")
            task()
            job_queue.task_done()
        except queue.Empty:
            print("[Scheduler] no more jobs")
            break

# Submit jobs in arbitrary order
job_queue.put((NORMAL,   "job-001", lambda: time.sleep(0.1)))
job_queue.put((BATCH,    "job-002", lambda: time.sleep(0.1)))
job_queue.put((CRITICAL, "job-003", lambda: time.sleep(0.1)))
job_queue.put((HIGH,     "job-004", lambda: time.sleep(0.1)))
job_queue.put((NORMAL,   "job-005", lambda: time.sleep(0.1)))

t = threading.Thread(target=scheduler)
t.start(); t.join()
# Always runs: CRITICAL → HIGH → NORMAL → NORMAL → BATCH
```

------

## 11. ThreadPoolExecutor — When managing a pool of workers (何时使用线程池)

### 1) ✅ Use `ThreadPoolExecutor` when

<span style="color:#E8600A">1.</span> Running **many short-to-medium IO-bound tasks** concurrently (HTTP, DB, file IO). <span style="color:#E8600A">2.</span> You need **return values** from concurrent tasks without manual thread management. <span style="color:#E8600A">3.</span> Applying the **same function to many inputs** in parallel (`executor.map`). <span style="color:#E8600A">4.</span> You want **automatic thread lifecycle management** (creation, recycling, shutdown).

```python
# ✅ Scenario: fetch multiple URLs concurrently, collect all results
from concurrent.futures import ThreadPoolExecutor, as_completed
import time, random

def fetch_url(url):
    """Simulate network fetch with random latency."""
    latency = random.uniform(0.2, 1.5)
    time.sleep(latency)
    if "broken" in url:
        raise ConnectionError(f"Failed to connect to {url}")
    return {"url": url, "status": 200, "latency": round(latency, 3)}

urls = [
    "https://api.example.com/users",
    "https://api.example.com/products",
    "https://api.example.com/broken-endpoint",
    "https://api.example.com/orders",
    "https://api.example.com/inventory",
]

print("Starting concurrent fetches...\n")
with ThreadPoolExecutor(max_workers=4) as executor:
    future_to_url = {executor.submit(fetch_url, url): url for url in urls}

    for future in as_completed(future_to_url):
        url = future_to_url[future]
        try:
            result = future.result()
            print(f"✅ {result['url']:<40} latency={result['latency']}s")
        except ConnectionError as e:
            print(f"❌ {url:<40} ERROR: {e}")
```

### 2) `submit()` vs `map()` — When to use which

| Situation                                                   | Use                                                          |
| ----------------------------------------------------------- | ------------------------------------------------------------ |
| Need individual `Future` objects for callbacks/cancellation | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">submit()</code> |
| Simple parallel map, results in submission order            | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">map()</code> |
| Process results as they complete (not submission order)     | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">as_completed()</code> |
| Mixed inputs with different argument structures             | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">submit()</code> |

### 3) ❌ Do NOT use `ThreadPoolExecutor` when

<span style="color:#C0392B;font-weight:600">× CPU-bound tasks (image processing, ML inference) → use `ProcessPoolExecutor` instead.</span> <span style="color:#C0392B;font-weight:600">× Tasks need complex inter-thread communication → combine with `Queue`.</span> <span style="color:#C0392B;font-weight:600">× Thousands of very short tasks (< 1ms) → thread overhead dominates; use `asyncio`.</span>

------

## 12. Master Decision Flowchart (总决策流程图)

```
START: I need concurrent execution
│
├─ CPU-bound (数学计算、压缩、ML)?
│   └─ YES → use multiprocessing.Process or ProcessPoolExecutor
│
└─ IO-bound (网络、文件、数据库)?
    │
    ├─ Simple: run N tasks, collect results
    │   └─ use ThreadPoolExecutor.submit() / map()
    │
    ├─ Complex: need fine-grained control
    │   │
    │   ├─ Tasks need to exchange data?
    │   │   └─ use Queue (FIFO) / LifoQueue / PriorityQueue
    │   │
    │   ├─ Need to protect shared state?
    │   │   ├─ One thread at a time, non-reentrant → Lock
    │   │   ├─ One thread at a time, reentrant    → RLock
    │   │   └─ N threads at a time                → Semaphore(N)
    │   │
    │   ├─ Need to wait for a condition?
    │   │   ├─ One-time broadcast signal → Event
    │   │   └─ Repeated condition change → Condition
    │   │
    │   ├─ Need all threads to reach a point?
    │   │   └─ Barrier(N)
    │   │
    │   ├─ Need per-thread private state?
    │   │   └─ threading.local()
    │   │
    │   ├─ Need delayed / cancellable execution?
    │   │   └─ Timer
    │   │
    │   └─ Long-lived background service?
    │       └─ Thread(daemon=True) + Event (stop signal)
    │
    └─ Very high concurrency (1000s of tasks)?
        └─ use asyncio + aiohttp (not threading)
```

------

## 13. Real-world Scenario → API Mapping (真实场景 → API 映射)

| Real-world scenario (真实场景)                   | API to use                                                   |
| ------------------------------------------------ | ------------------------------------------------------------ |
| Fetch 100 URLs in parallel                       | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">ThreadPoolExecutor</code> |
| Download pipeline: fetch → parse → store         | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Queue</code> (3-stage pipeline) |
| Shared counter incremented by many threads       | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Lock</code> |
| Class method calls another synchronized method   | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">RLock</code> |
| Workers wait for DB to be populated              | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Condition.wait_for()</code> |
| 5 workers start simultaneously (race simulation) | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Barrier</code> |
| Max 3 concurrent DB connections                  | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">BoundedSemaphore(3)</code> |
| Server "ready" signal to all request handlers    | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Event</code> |
| Graceful shutdown of background worker           | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Event</code> (stop flag) |
| Auto-logout after 30min inactivity               | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Timer</code> + `cancel()` on activity |
| Debounce save-to-disk on rapid edits             | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Timer</code> + `cancel()` |
| Per-thread DB connection (no sharing)            | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">threading.local()</code> |
| Critical tasks skip the line                     | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">PriorityQueue</code> |
| Undo stack processed by worker thread            | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">LifoQueue</code> |
| Parallel phases: all workers finish step 1 first | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Barrier</code> |
| Background heartbeat / health monitor            | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Thread(daemon=True)</code> |
| LRU cache with thread-safe eviction              | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Lock</code> + <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">OrderedDict</code> |
| Rate-limit outgoing API requests                 | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">BoundedSemaphore</code> |

------

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway for Part II</span><br> The decision rule is simple: <span style="color:#2980B9"><strong>data flows between threads</strong></span> → <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Queue</code> &nbsp;|&nbsp; <span style="color:#2980B9"><strong>shared state needs protection</strong></span> → <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Lock</code>/<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">RLock</code> &nbsp;|&nbsp; <span style="color:#2980B9"><strong>wait for a condition</strong></span> → <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Event</code> or <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Condition</code> &nbsp;|&nbsp; <span style="color:#2980B9"><strong>limit concurrency</strong></span> → <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Semaphore</code> &nbsp;|&nbsp; <span style="color:#2980B9"><strong>just run N tasks</strong></span> → <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">ThreadPoolExecutor</code>. </div>



