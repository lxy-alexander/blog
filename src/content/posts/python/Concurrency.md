---
title: "Concurrency"
published: 2026-06-27
description: "Concurrency"
image: ""
tags: ["python","Concurrency"]
category: python
draft: false
lang: ""
createdAt: "2026-06-27T16:35:19.178.399610116Z"
---

## Thread Target

The target of a thread must be a callable object, such as a function, method, or lambda.

线程的 `target` 必须是可调用对象，例如函数、方法或 `lambda`。



```
import time
from threading import Thread

def countdown(n):
    while n > 0:
        print('T-minus', n)
        n -= 1
        time.sleep(1)

t = Thread(target=countdown, args=(2,))
t.start()
```





## time.sleep()

`time.sleep()` is used to pause the current thread for a given number of seconds.

`time.sleep()` 用来让当前线程暂停指定的秒数





## **Daemon thread**

Daemon threads are used for background tasks that do not need to finish before the main program exits, such as logging, monitoring, cleanup, or heartbeat checks.

守护线程用于后台任务，例如日志记录、状态监控、清理任务或心跳检测。



## join

`join()` is used to wait for a thread to finish. But daemonic threads can’t be joined. they are destroyed automatically when the main thread terminates.

`join()` 用来等待一个线程执行结束。

```python
t.start()
t.join()
print("Thread finished")
```



## high-level operations

there are no operations to terminate a thread, signal a thread, adjust its scheduling, or perform any other high-level operations. If you want these features, you need to build them yourself.

==the thread must be programmed to poll for exit at selected points.==

```python
import time
from threading import Thread

class CountDownTask:
    def __init__(self):
        self._running = True

    def terminate(self):
        self._running = False

    def run(self, n):
        while self._running and n:
            print('T-minus', n)
            n -= 1
            time.sleep(1)

c = CountDownTask()
t = Thread(target=c.run, args=(3,))
t.start()
c.terminate()
t.join()

if t.is_alive():
    print("Thread is still running")
else:
    print("finished")
```



Polling for thread termination can be tricky to coordinate if threads perform blocking operations such as I/O. 

For example, a thread blocked indefinitely on an I/O operation may never return to check if it’s been killed. To correctly deal with this case, you’ll need to carefully program thread to ==utilize timeout== loops.

Use timeout loops so a thread blocked on I/O can periodically wake up and check whether it should terminate.

```python
import socket

class IOTask:
    def __init__(self):
        self._running = True

    def terminate(self):
        self._running = False

    def run(self, sock):
        sock.settimeout(5)

        while self._running:
            try:
                data = sock.recv(8192)
                if not data:
                    break

                print("received:", data)

            except socket.timeout:
                continue
```

| I/O type      | Timeout example                  | Notes                                                   |
| ------------- | -------------------------------- | ------------------------------------------------------- |
| Socket I/O    | `sock.settimeout(5)`             | Works for `recv()`, `accept()`, `connect()`             |
| HTTP requests | `requests.get(url, timeout=5)`   | Prevents requests from hanging forever                  |
| Database I/O  | `connect_timeout=5`              | Depends on the database library                         |
| Queue         | `queue.get(timeout=5)`           | Wakes up if no item arrives                             |
| Lock          | `lock.acquire(timeout=5)`        | Avoids waiting forever for a lock                       |
| Event         | `event.wait(timeout=5)`          | Useful for thread coordination                          |
| Condition     | `condition.wait(timeout=5)`      | Useful for waiting on shared state                      |
| Subprocess    | `subprocess.run(cmd, timeout=5)` | Stops waiting for an external process                   |
| File I/O      | Usually no direct timeout        | Local file reads/writes normally do not support timeout |



## Python Global Interpreter Lock

**In CPython, the GIL allows only one thread to execute Python bytecode at a time.**

在 CPython 里，GIL 让同一时刻通常只有一个线程能执行 Python 字节码。

Because when waiting for I/O, threads usually release the GIL, and other threads can continue to run.

**In Python, we usually avoid using threads for CPU-bound computation because of the GIL. Instead, threads are more suitable for I/O-bound tasks, such as network requests, file operations, database queries, or waiting tasks.**

在 Python 中，因为 GIL 的存在，我们通常不使用多线程来做大量计算任务，而更适合用多线程处理 I/O 操作，比如网络请求、文件操作、数据库查询或等待任务。









## Event

`Event` is used to signal between threads.`Event` is used to coordinate threads by letting one thread wait until another thread sends a signal.

| Method   | Meaning                   |
| -------- | ------------------------- |
| `wait()` | Wait for the event signal |
| `set()`  | Send the event signal     |

```python
from threading import Thread, Event
import time

event = Event()

def worker():
    print("worker starting")
    time.sleep(2)
    event.set()

t = Thread(target=worker)
t.start()

print("main waiting")
event.wait()
print("main continues")
```



## Semaphore

A `Semaphore` is used to control how many threads can continue.

```python
import threading
import time

def worker(n, sema):
    print(f"Worker {n} waiting")
    sema.acquire()

    print(f"Worker {n} working")
    time.sleep(1)
    print(f"Worker {n} done")

sema = threading.Semaphore(0)

for n in range(5):
    t = threading.Thread(target=worker, args=(n, sema))
    t.start()

time.sleep(2)

sema.release()
```

Key idea:

**Each `release()` allows one waiting thread to continue.**



## Condition

A `Condition` is used when threads need to wait for a shared condition to become true.

```python
import threading
import time

ready = False
condition = threading.Condition()

def worker():
    global ready

    with condition:
        while not ready:
            print("Worker waiting")
            condition.wait()

        print("Worker working")

t = threading.Thread(target=worker)
t.start()

time.sleep(2)

with condition:
    ready = True
    condition.notify()

t.join()
```

Key idea:

**`wait()` waits for a condition, and `notify()` wakes up one waiting thread.**

------

| Tool        | Purpose                               | Wake-up behavior                               |
| ----------- | ------------------------------------- | ---------------------------------------------- |
| `Event`     | Signal that something happened        | Wakes all waiting threads                      |
| `Semaphore` | Control how many threads can continue | One `release()` wakes one thread               |
| `Condition` | Wait for shared state to change       | `notify()` wakes one, `notify_all()` wakes all |

| API                      | 人话意思                 | 谁调用       |
| ------------------------ | ------------------------ | ------------ |
| `event.wait()`           | 我等一个信号             | 等待的线程   |
| `event.set()`            | 信号来了，大家可以继续   | 发信号的线程 |
| `sema.acquire()`         | 我要一个通行证，没有就等 | 等待的线程   |
| `sema.release()`         | 发一个通行证，放一个人走 | 发信号的线程 |
| `condition.wait()`       | 条件不满足，我先等       | 等待的线程   |
| `condition.notify()`     | 条件变了，叫醒一个人     | 改条件的线程 |
| `condition.notify_all()` | 条件变了，叫醒所有人     | 改条件的线程 |

| API                   | Meaning                            |
| --------------------- | ---------------------------------- |
| `lock.acquire()`      | Try to get the lock                |
| `sema.acquire()`      | Try to get one permit              |
| `condition.acquire()` | Lock the condition before using it |





## with condition

Because a `Condition` must be used together with its internal lock.

```python
with condition:
    ready = True
    condition.notify()
```

means:

1）Lock the condition.

2）Safely change the shared variable:

```python
ready = True
```

3）Notify a waiting thread:

```python
condition.notify()
```

4）Automatically unlock when leaving the `with` block.

It is equivalent to:

```python
condition.acquire()
try:
    ready = True
    condition.notify()
    
finally:
    condition.release()
```

Why needed?

Because `ready` is shared between threads. The lock prevents one thread from reading `ready` while another thread is changing it.

Key sentence:

**`with condition:` locks the condition so shared state can be changed and notified safely.**







