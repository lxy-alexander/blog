---
title: "Python Data Structures and Algorithms"
published: 2026-06-28
description: "Python Data Structures and Algorithms"
image: ""
tags: ["python","Python Data Structures and Algorithms"]
category: python
draft: false
lang: ""
createdAt: "2026-06-28T22:36:06.319.071234293Z"
---



## yield

Each time the function reaches yield, it pauses and hands over the value. In the next loop, the execution will resume from the paused position.

每执行到一次 `yield`，函数就会暂停，把值交出去；下一次循环时，再从暂停的位置继续执行。



## `contextmanager` 

contextmanager is a decorator that transforms the yield function into a with available object

```python
from contextlib import contextmanager

@contextmanager
def open_resource():
    print('进入')
    yield '资源'
    print('退出')
```



## Finding the Largest or Smallest N Items

Maintain a min-heap of size n and use it to store the n largest elements currently encountered.



## Heaps

heapq is a module, not a class or a function, so it cannot be called like heapq().

```
import heapq

heap = []

heapq.heappush(heap, 10)
heapq.heappush(heap, 3)
heapq.heappush(heap, 7)

print(heap)              # 堆结构，不一定是排序后的列表
print(heapq.heappop(heap))  # 3，最小值先出来


nums = [[3, 3], [2, 4], [3, 5]]
heap = []
for x in nums:
    heapq.heappush(heap, (x[0], -x[1], x))
while heap:
    _, _, x = heapq.heappop(heap)
    print(x)
```



## sorted

sorted cannot change the orginal list

```
nums = [[3, 3], [2, 4], [3, 5]]
nums.sort(key=lambda x: (x[0], -x[1]))
print(nums)
```



## collections.deque

Double-Ended Queue



## PriorityQueue

If your `item` is a dictionary (e.g., `{'task': 'A'}`) or a custom object, Python does not know how to compare them using `<` or `>`. ==**to prevent code crashes** and **to ensure fairness (FIFO stability) when priorities tie**.==

```python
class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue, (-priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]
```





## defaultdict

```
d = {}
for key, value in pairs:
	if key not in d:
		d[key] = []
	d[key].append(value)
```





## OrderedDict

Custom Class need to define the repr method.

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        
        # Mark as Most Recently Used (MRU) by moving it to the end
        self.cache.move_to_end(key, last=True)  # last=True is default, added for clarity
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            # Update value and mark as MRU
            self.cache[key] = value
            self.cache.move_to_end(key, last=True)
            return

        # If key is new, check capacity limit
        if len(self.cache) >= self.capacity:
            # Evict the Least Recently Used (LRU) item from the front (FIFO)
            # last=False pops the oldest item (leftmost)
            self.cache.popitem(last=False)
            
        # Insert the new key-value pair at the end
        self.cache[key] = value
```





## `queue.LifoQueue(maxsize=N)` (Thread-Safe Stack)

```
from queue import LifoQueue, PriorityQueue

# ==========================================
# 1. LifoQueue Example (LIFO / Stack)
# ==========================================
print("--- LifoQueue Demo ---")
lifo = LifoQueue(maxsize=3)  # Caps capacity at 3

lifo.put("Task A")
lifo.put("Task B")
lifo.put("Task C")

# Pops in reverse chronological order: C -> B -> A
while not lifo.empty():
    print(f"Lifo Popped: {lifo.get()}")


# ==========================================
# 2. PriorityQueue Example (Priority Queue)
# ==========================================
print("\n--- PriorityQueue Demo ---")
pq = PriorityQueue(maxsize=3)  # Caps capacity at 3

# Insert tuples: (priority_number, item)
# Remember: Lower numbers = Higher processing priority!
pq.put((30, "Low Priority Job"))
pq.put((10, "Critical Hotfix"))
pq.put((20, "Medium Priority Review"))

# Pops strictly sorted by priority ascending: 10 -> 20 -> 30
while not pq.empty():
    priority, task = pq.get()
    print(f"Priority Popped -> Priority Score: {priority}, Task: {task}")
```







## ChainMap

 **`ChainMap` is useful when you want multiple dictionaries with priority order, without copying them into one new dictionary.**

```
from collections import ChainMap

a = {'x': 1, 'z': 3}
b = {'y': 2, 'z': 4}
c = {'y': 3, 'z': 5}

d = ChainMap(a, b, c)

print(d['x'])  # 1
print(d['y'])  # 2
print(d['z'])  # 3
```

