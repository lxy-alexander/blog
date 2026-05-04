---
title: "queue"
published: 2026-05-04
description: "queue"
image: ""
tags: ["python","queue"]
category: python
draft: false
lang: ""
createdAt: "2026-05-04T18:38:11.575.170140516Z"
---

# Python Queue Usage Notes

Python provides different Queue (队列) tools for different scenarios, and the key interview point is to choose the right one based on Thread Safety (线程安全) and Blocking Behavior (阻塞行为).

## 1. `queue.Queue`

`queue.Queue` is a thread-safe Blocking Queue (阻塞队列) mainly used in Producer-Consumer Model (生产者消费者模型).

```
import queue

q = queue.Queue()

q.put("A")
q.put("B")

print(q.get())
print(q.get())

# Output:
# A
# B
```

<br>

## 2. `put()`

`put()` inserts an item into the Queue (队列), and it may block if the queue has a maximum size and is full.

```
import queue

q = queue.Queue(maxsize=2)

q.put("A")
q.put("B")

print(q.full())

# Output:
# True
```

<br>

## 3. `put_nowait()`

`put_nowait()` inserts an item immediately, and it raises `queue.Full` if the Queue (队列) is full.

```
import queue

q = queue.Queue(maxsize=1)

q.put_nowait("A")

try:
    q.put_nowait("B")
except queue.Full:
    print("Queue is full")

# Output:
# Queue is full
```

<br>

## 4. `get()`

`get()` removes and returns an item from the Queue (队列), and it blocks if the queue is empty.

```
import queue
import threading
import time

q = queue.Queue()

def producer():
    time.sleep(1)
    q.put("task")

def consumer():
    print("Waiting")
    item = q.get()
    print(item)

threading.Thread(target=producer).start()
consumer()

# Output:
# Waiting
# task
```

<br>

## 5. `get_nowait()`

`get_nowait()` removes an item immediately, and it raises `queue.Empty` if the Queue (队列) is empty.

```
import queue

q = queue.Queue()

try:
    item = q.get_nowait()
except queue.Empty:
    print("Queue is empty")

# Output:
# Queue is empty
```

<br>

## 6. `get(timeout=...)`

`get(timeout=...)` waits for a limited time and raises `queue.Empty` if no item arrives before the timeout.

```
import queue

q = queue.Queue()

try:
    item = q.get(timeout=1)
except queue.Empty:
    print("No item after timeout")

# Output:
# No item after timeout
```

<br>

## 7. `task_done()`

`task_done()` tells the Queue (队列) that one fetched task has been fully processed.

```
import queue

q = queue.Queue()

q.put("task")

item = q.get()
print("Processing:", item)
q.task_done()

print("Done")

# Output:
# Processing: task
# Done
```

<br>

## 8. `join()`

`join()` blocks until all tasks added to the Queue (队列) have been marked done by `task_done()`.

```
import queue
import threading

q = queue.Queue()

def worker():
    item = q.get()
    print("Worker got:", item)
    q.task_done()

q.put("task")

t = threading.Thread(target=worker)
t.start()

q.join()
print("All tasks done")

t.join()

# Output:
# Worker got: task
# All tasks done
```

<br>

## 9. Producer-Consumer Example

The Producer-Consumer Model (生产者消费者模型) uses `put()` to submit tasks and `get()` to consume tasks safely across threads.

```
import queue
import threading
import time

q = queue.Queue()

def producer():
    for i in range(3):
        q.put(i)
        print("Produced:", i)

def consumer():
    for _ in range(3):
        item = q.get()
        print("Consumed:", item)
        q.task_done()

t1 = threading.Thread(target=producer)
t2 = threading.Thread(target=consumer)

t1.start()
t2.start()

t1.join()
q.join()
t2.join()

# Output:
# Produced: 0
# Produced: 1
# Produced: 2
# Consumed: 0
# Consumed: 1
# Consumed: 2
```

<br>

## 10. `queue.LifoQueue`

`queue.LifoQueue` is a thread-safe Stack (栈), where the last inserted item is returned first.

```
import queue

stack = queue.LifoQueue()

stack.put("A")
stack.put("B")

print(stack.get())
print(stack.get())

# Output:
# B
# A
```

<br>

## 11. `queue.PriorityQueue`

`queue.PriorityQueue` is a thread-safe Priority Queue (优先队列), where the smallest priority value is returned first.

```
import queue

pq = queue.PriorityQueue()

pq.put((2, "low priority"))
pq.put((1, "high priority"))

print(pq.get())
print(pq.get())

# Output:
# (1, 'high priority')
# (2, 'low priority')
```

<br>

## 12. `collections.deque`

`collections.deque` is the preferred normal Queue (队列) for single-threaded code because `append()` and `popleft()` are efficient.

```
from collections import deque

q = deque()

q.append("A")
q.append("B")

print(q.popleft())
print(q.popleft())

# Output:
# A
# B
```

<br>

## 13. Stack with `list`

A Python `list` is a simple and efficient Stack (栈) when using `append()` and `pop()` at the end.

```
stack = []

stack.append("A")
stack.append("B")

print(stack.pop())
print(stack.pop())

# Output:
# B
# A
```

<br>

## 14. Queue Selection Rule

Use `queue.Queue` for multi-threaded Blocking Queue (阻塞队列), `collections.deque` for normal Queue (队列), `list` for normal Stack (栈), and `queue.PriorityQueue` for thread-safe Priority Queue (优先队列).

```
from collections import deque
import queue

normal_queue = deque()
thread_queue = queue.Queue()
thread_stack = queue.LifoQueue()
priority_queue = queue.PriorityQueue()

print(type(normal_queue).__name__)
print(type(thread_queue).__name__)
print(type(thread_stack).__name__)
print(type(priority_queue).__name__)

# Output:
# deque
# Queue
# LifoQueue
# PriorityQueue
```

<br> <br>
