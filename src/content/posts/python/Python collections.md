---
title: "Python collections"
published: 2026-03-09
description: "Python collections"
image: ""
tags: ["python","Python collections"]
category: python
draft: false
lang: ""
---

# **I. Python `collections` Module — Complete Learning Manual**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Python's <span style="color:#E8600A;font-weight:700">collections</span> module provides <span style="color:#2980B9">specialized container datatypes (特殊容器数据类型)</span> that extend the built-in <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">dict</code>, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">list</code>, and <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">tuple</code>. The seven main classes are: <span style="color:#E8600A;font-weight:700">defaultdict</span>, <span style="color:#E8600A;font-weight:700">Counter</span>, <span style="color:#E8600A;font-weight:700">OrderedDict</span>, <span style="color:#E8600A;font-weight:700">deque</span>, <span style="color:#E8600A;font-weight:700">namedtuple</span>, <span style="color:#E8600A;font-weight:700">ChainMap</span>, and <span style="color:#E8600A;font-weight:700">UserDict / UserList / UserString</span>. Each solves a specific pain-point of the standard built-ins with minimal overhead. </div>

------

## 1. defaultdict — Default Value Dict (默认值字典)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> A <span style="color:#E8600A;font-weight:700">defaultdict (默认值字典)</span> behaves exactly like a regular <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">dict</code>, except that accessing a <span style="color:#2980B9">missing key (缺失键)</span> automatically creates it by calling the <span style="color:#E8600A;font-weight:700">default_factory (默认工厂函数)</span> — eliminating <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">KeyError</code> and verbose <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">setdefault()</code> boilerplate. </div>

### 1) Constructor (构造函数)

```python
collections.defaultdict(default_factory=None, **kwargs)
```

<span style="color:#E8600A;font-weight:700">default_factory</span> is any zero-argument callable: <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">int</code>, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">list</code>, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">set</code>, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">dict</code>, or a custom <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">lambda</code>.

### 2) `defaultdict(int)` — Frequency counter (频率计数)

```python
from collections import defaultdict

text = "apple banana apple cherry banana apple"

freq = defaultdict(int)        # missing key → 0

for word in text.split():
    freq[word] += 1            # no KeyError on first access

print(dict(freq))
# → {'apple': 3, 'banana': 2, 'cherry': 1}

# Compare with plain dict (verbose):
freq2 = {}
for word in text.split():
    freq2[word] = freq2.get(word, 0) + 1   # needs .get()
```

### 3) `defaultdict(list)` — Grouping (分组)

```python
from collections import defaultdict

students = [
    ("Alice", "Math"),
    ("Bob",   "Science"),
    ("Alice", "Science"),
    ("Carol", "Math"),
    ("Bob",   "Math"),
]

by_name = defaultdict(list)

for name, subject in students:
    by_name[name].append(subject)   # missing key → [] automatically

print(dict(by_name))
# → {'Alice': ['Math', 'Science'], 'Bob': ['Science', 'Math'], 'Carol': ['Math']}
```

### 4) `defaultdict(set)` — Unique grouping (去重分组)

```python
from collections import defaultdict

edges = [(1, 2), (1, 3), (2, 3), (1, 2)]   # duplicate edge (1,2)

graph = defaultdict(set)

for u, v in edges:
    graph[u].add(v)
    graph[v].add(u)

print(dict(graph))
# → {1: {2, 3}, 2: {1, 3}, 3: {1, 2}}   (no duplicates)
```

### 5) `defaultdict(dict)` — Nested dict (嵌套字典)

```python
from collections import defaultdict

# 2-level nested defaultdict
matrix = defaultdict(lambda: defaultdict(int))

matrix["row1"]["col1"] += 10
matrix["row1"]["col2"] += 20
matrix["row2"]["col1"] += 30

for row, cols in matrix.items():
    print(f"{row}: {dict(cols)}")
# → row1: {'col1': 10, 'col2': 20}
# → row2: {'col1': 30}
```

### 6) Custom `default_factory` (自定义工厂函数)

```python
from collections import defaultdict

# Factory that returns a specific default value
dd = defaultdict(lambda: "N/A")
dd["name"] = "Alice"

print(dd["name"])     # → Alice
print(dd["age"])      # → N/A   (key created with "N/A")
print(dd["city"])     # → N/A

# Factory with counter
id_counter = [0]
def next_id():
    id_counter[0] += 1
    return id_counter[0]

registry = defaultdict(next_id)
print(registry["alice"])   # → 1
print(registry["bob"])     # → 2
print(registry["alice"])   # → 1  (already exists)
```

### 7) `default_factory` attribute — Inspect and change

```python
from collections import defaultdict

dd = defaultdict(list)
print(dd.default_factory)    # → <class 'list'>

dd.default_factory = set     # change factory at runtime
dd["new_key"].add(42)
print(dict(dd))              # → {'new_key': {42}}

dd.default_factory = None    # disable factory → KeyError on missing keys
try:
    _ = dd["missing"]
except KeyError as e:
    print(f"KeyError: {e}")  # → KeyError: 'missing'
```

### 8) `__missing__` — How defaultdict works internally

```python
from collections import defaultdict

class MyDefaultDict(dict):
    """Manual implementation of defaultdict logic."""

    def __init__(self, factory):
        super().__init__()
        self.factory = factory

    def __missing__(self, key):
        # Called automatically when key is not found
        value = self.factory()
        self[key] = value
        return value

d = MyDefaultDict(list)
d["x"].append(1)
d["x"].append(2)
d["y"].append(3)
print(dict(d))   # → {'x': [1, 2], 'y': [3]}
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600"><code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">__missing__</code> is only triggered by <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">d[key]</code> access, NOT by <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">d.get(key)</code>.</span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">get()</code> always returns <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">None</code> (or the provided default) without creating the key.</div>

### 9) Inherits all `dict` methods

```python
from collections import defaultdict

dd = defaultdict(int, a=1, b=2)

# All standard dict methods work
print(dd.keys())              # → dict_keys(['a', 'b'])
print(dd.values())            # → dict_values([1, 2])
print(dd.items())             # → dict_items([('a', 1), ('b', 2)])
print(dd.get("x", 99))        # → 99  (no key created)
print("a" in dd)              # → True
dd.update({"c": 3})
print(dd.pop("a"))            # → 1
print(dict(dd))               # → {'b': 2, 'c': 3}
```

------

## 2. Counter — Multiset / Frequency Map (计数器)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> A <span style="color:#E8600A;font-weight:700">Counter (计数器)</span> is a <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">dict</code> subclass designed for <span style="color:#2980B9">counting hashable objects (统计可哈希对象)</span>. Missing keys return <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">0</code> instead of raising <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">KeyError</code>. It supports <span style="color:#E8600A;font-weight:700">arithmetic operations (算术运算)</span> between counters. </div>

### 1) Constructor — Three ways to create

```python
from collections import Counter

# From an iterable
c1 = Counter("abracadabra")
print(c1)   # → Counter({'a': 5, 'b': 2, 'r': 2, 'c': 1, 'd': 1})

# From a dict
c2 = Counter({"cats": 4, "dogs": 8})
print(c2)   # → Counter({'dogs': 8, 'cats': 4})

# From keyword arguments
c3 = Counter(red=3, blue=1, green=5)
print(c3)   # → Counter({'green': 5, 'red': 3, 'blue': 1})
```

### 2) Missing key → 0 (缺失键返回0)

```python
from collections import Counter

c = Counter("hello")
print(c["l"])    # → 2  (exists)
print(c["z"])    # → 0  (missing — no KeyError!)
print("z" in c) # → False  (not stored, just returns 0)
```

### 3) `most_common(n)` — Top N elements (最高频N个元素)

```python
from collections import Counter

words = "the quick brown fox jumps over the lazy dog the fox".split()
c = Counter(words)

print(c.most_common(3))
# → [('the', 3), ('fox', 2), ('quick', 1)]

print(c.most_common())        # all elements, sorted by frequency
print(c.most_common()[:-4:-1])# least common 3 (tail trick)
# → [('dog', 1), ('lazy', 1), ('over', 1)]
```

### 4) `elements()` — Expand back to iterable (展开为可迭代)

```python
from collections import Counter

c = Counter(a=3, b=1, c=2)

print(list(c.elements()))
# → ['a', 'a', 'a', 'b', 'c', 'c']  (ordered by insertion)

# Reconstruct a sorted list
print(sorted(c.elements()))
# → ['a', 'a', 'a', 'b', 'c', 'c']

# Elements with count ≤ 0 are excluded
c["x"] = -1
print(list(c.elements()))    # 'x' not included
```

### 5) `subtract()` / `update()` — In-place operations (就地运算)

```python
from collections import Counter

inventory = Counter(apples=10, oranges=5, bananas=8)

# subtract: reduces counts (allows negatives)
sold = Counter(apples=3, oranges=5, bananas=10)
inventory.subtract(sold)
print(inventory)
# → Counter({'apples': 7, 'bananas': -2, 'oranges': 0})

# update: adds counts (merges)
restocked = Counter(apples=5, bananas=15)
inventory.update(restocked)
print(inventory)
# → Counter({'bananas': 13, 'apples': 12, 'oranges': 0})
```

### 6) Arithmetic operators (算术运算符)

```python
from collections import Counter

a = Counter(x=4, y=2, z=0)
b = Counter(x=1, y=3, w=5)

print(a + b)    # add counts
# → Counter({'x': 5, 'w': 5, 'y': 5})

print(a - b)    # subtract, keep only positives
# → Counter({'x': 3})

print(a & b)    # intersection: min of each count
# → Counter({'x': 1, 'y': 2})

print(a | b)    # union: max of each count
# → Counter({'w': 5, 'x': 4, 'y': 3})

# Unary operators
print(+a)       # remove zero and negative counts
print(-a)       # negate — flip sign, keep negatives as positives
```

### 7) Total count and filtering (总计数与过滤)

```python
from collections import Counter

c = Counter(a=5, b=3, c=0, d=-2)

# Total of all positive counts (Python 3.10+)
print(c.total())       # → 8   (5+3+0 = 8, negatives excluded)

# Keep only positive counts
positive = +c
print(positive)        # → Counter({'a': 5, 'b': 3})

# Keep only negative counts (useful for "owed" quantities)
negative = -c
print(negative)        # → Counter({'d': 2})
```

### 8) Practical: anagram check, top-K, word frequency

```python
from collections import Counter

# ── Anagram check (变位词检测) ──
def is_anagram(s1: str, s2: str) -> bool:
    return Counter(s1.lower()) == Counter(s2.lower())

print(is_anagram("listen", "silent"))   # → True
print(is_anagram("hello",  "world"))    # → False

# ── Character frequency difference ──
def missing_chars(have: str, need: str) -> Counter:
    deficit = Counter(need) - Counter(have)
    return deficit

print(missing_chars("aab", "aaabbc"))
# → Counter({'a': 1, 'b': 1, 'c': 1})

# ── Top-K frequent words ──
import re

text = """To be or not to be that is the question
          whether tis nobler in the mind to suffer"""

words  = re.findall(r'\w+', text.lower())
top5   = Counter(words).most_common(5)
print(top5)
# → [('to', 3), ('be', 2), ('the', 2), ('or', 1), ('not', 1)]
```

### 9) Inherits all `dict` methods

```python
from collections import Counter

c = Counter("mississippi")

print(c.keys())           # → dict_keys(['m', 'i', 's', 'p'])
print(c.values())         # → dict_values([1, 4, 4, 2])
print(c.items())          # → dict_items([('m', 1), ('i', 4), ('s', 4), ('p', 2)])
print(c.get("i"))         # → 4
print(c.get("z"))         # → None   (get() returns None, not 0)

# del sets count to 0 conceptually, but removes the key
del c["m"]
print("m" in c)           # → False
print(c["m"])             # → 0  (missing key returns 0)
```

------

## 3. OrderedDict — Ordered Dictionary (有序字典)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> Since Python 3.7, plain <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">dict</code> preserves insertion order. <span style="color:#E8600A;font-weight:700">OrderedDict (有序字典)</span> still offers unique advantages: <span style="color:#2980B9">order-sensitive equality</span>, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">move_to_end()</code>, and <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">popitem(last=True/False)</code> for implementing <span style="color:#E8600A;font-weight:700">LRU Cache (LRU缓存)</span> and similar structures. </div>

### 1) Basic usage and order-sensitive equality

```python
from collections import OrderedDict

od = OrderedDict()
od["banana"] = 3
od["apple"]  = 5
od["cherry"] = 1

print(od)
# → OrderedDict([('banana', 3), ('apple', 5), ('cherry', 1)])

# Order-sensitive equality (顺序敏感的相等判断)
od1 = OrderedDict([("a", 1), ("b", 2)])
od2 = OrderedDict([("b", 2), ("a", 1)])
d1  = {"a": 1, "b": 2}

print(od1 == od2)   # → False  (same keys/values, different order)
print(od1 == d1)    # → True   (OrderedDict == dict ignores order)
```

### 2) `move_to_end(key, last=True)` — Reposition a key

```python
from collections import OrderedDict

od = OrderedDict.fromkeys("ABCDE")

od.move_to_end("B")          # move B to end (last=True default)
print(list(od))              # → ['A', 'C', 'D', 'E', 'B']

od.move_to_end("E", last=False)  # move E to front
print(list(od))              # → ['E', 'A', 'C', 'D', 'B']
```

### 3) `popitem(last=True)` — LIFO / FIFO removal

```python
from collections import OrderedDict

od = OrderedDict.fromkeys("ABCDE")

print(od.popitem(last=True))    # → ('E', None)  LIFO (like a stack)
print(od.popitem(last=False))   # → ('A', None)  FIFO (like a queue)
print(list(od))                 # → ['B', 'C', 'D']
```

### 4) LRU Cache implementation (LRU缓存实现)

```python
from collections import OrderedDict

class LRUCache:
    """
    Least Recently Used Cache (最近最少使用缓存)
    using OrderedDict for O(1) get and put.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache    = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)    # mark as recently used
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # evict least recently used

cache = LRUCache(3)
cache.put(1, 10)
cache.put(2, 20)
cache.put(3, 30)
print(cache.get(1))   # → 10  (1 moved to end)
cache.put(4, 40)      # evicts key 2 (least recently used)
print(cache.get(2))   # → -1  (evicted)
print(cache.get(3))   # → 30
print(cache.get(4))   # → 40
```

### 5) `__reversed__()` — Reverse iteration

```python
from collections import OrderedDict

od = OrderedDict([("a", 1), ("b", 2), ("c", 3)])

for key in reversed(od):
    print(key, od[key])
# → c 3
# → b 2
# → a 1
```

------

## 4. deque — Double-Ended Queue (双端队列)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> A <span style="color:#E8600A;font-weight:700">deque (双端队列)</span> supports <span style="color:#2980B9">O(1) append and pop from both ends</span>. Unlike a <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">list</code> (where <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">insert(0, x)</code> is O(n)), deque is the correct data structure for <span style="color:#E8600A;font-weight:700">queues (队列)</span>, <span style="color:#E8600A;font-weight:700">stacks (栈)</span>, and <span style="color:#E8600A;font-weight:700">sliding windows (滑动窗口)</span>. </div>

### 1) Constructor

```python
from collections import deque

d1 = deque()                         # empty
d2 = deque([1, 2, 3, 4, 5])          # from iterable
d3 = deque("abcde")                  # from string
d4 = deque(range(10), maxlen=5)      # bounded deque (固定长度)

print(d2)   # → deque([1, 2, 3, 4, 5])
print(d4)   # → deque([5, 6, 7, 8, 9], maxlen=5)  (first 5 discarded)
```

### 2) `append()` / `appendleft()` — Add to ends (两端添加)

```python
from collections import deque

d = deque([3, 4, 5])

d.append(6)         # right end  → deque([3, 4, 5, 6])
d.appendleft(2)     # left end   → deque([2, 3, 4, 5, 6])
d.appendleft(1)     #            → deque([1, 2, 3, 4, 5, 6])

print(d)            # → deque([1, 2, 3, 4, 5, 6])
```

### 3) `pop()` / `popleft()` — Remove from ends (两端弹出)

```python
from collections import deque

d = deque([1, 2, 3, 4, 5])

print(d.pop())       # → 5   (right)  → deque([1, 2, 3, 4])
print(d.popleft())   # → 1   (left)   → deque([2, 3, 4])
print(d)             # → deque([2, 3, 4])
```

### 4) `extend()` / `extendleft()` — Batch add (批量添加)

```python
from collections import deque

d = deque([3, 4])

d.extend([5, 6, 7])          # right: → deque([3, 4, 5, 6, 7])
d.extendleft([2, 1, 0])      # left, each prepended individually
                             # 2 → [2,3..], 1 → [1,2,3..], 0 → [0,1,2,3..]
print(d)   # → deque([0, 1, 2, 3, 4, 5, 6, 7])
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600"><code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">extendleft([a, b, c])</code> results in <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">[c, b, a, ...]</code> because each element is prepended one by one — the iterable is effectively reversed.</span></div>

### 5) `rotate(n)` — Circular rotation (循环旋转)

```python
from collections import deque

d = deque([1, 2, 3, 4, 5])

d.rotate(2)     # rotate RIGHT by 2
print(d)        # → deque([4, 5, 1, 2, 3])

d.rotate(-2)    # rotate LEFT by 2 (undo)
print(d)        # → deque([1, 2, 3, 4, 5])

# Circular buffer simulation (循环缓冲区)
ring = deque(range(5))
for _ in range(8):
    print(ring[0], end=" ")
    ring.rotate(-1)
# → 0 1 2 3 4 0 1 2
```

### 6) `maxlen` — Bounded / sliding window (有界滑动窗口)

```python
from collections import deque

# Keep only the last 3 elements
window = deque(maxlen=3)

for i in range(7):
    window.append(i)
    print(f"added {i}: {list(window)}")
# → added 0: [0]
# → added 1: [0, 1]
# → added 2: [0, 1, 2]
# → added 3: [1, 2, 3]   ← 0 dropped automatically
# → added 4: [2, 3, 4]
# → added 5: [3, 4, 5]
# → added 6: [4, 5, 6]

# Moving average (滑动平均)
def moving_average(data, window_size):
    w = deque(maxlen=window_size)
    result = []
    for val in data:
        w.append(val)
        result.append(sum(w) / len(w))
    return result

print(moving_average([1, 2, 3, 4, 5, 6], 3))
# → [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
```

### 7) `insert()` / `remove()` / `count()` / `index()`

```python
from collections import deque

d = deque([1, 2, 3, 2, 4])

d.insert(2, 99)       # insert 99 at position 2
print(d)              # → deque([1, 2, 99, 3, 2, 4])

d.remove(99)          # remove first occurrence
print(d)              # → deque([1, 2, 3, 2, 4])

print(d.count(2))     # → 2  (occurrences of 2)
print(d.index(3))     # → 2  (first index of 3)
```

### 8) `reverse()` / `copy()` / `clear()`

```python
from collections import deque

d = deque([1, 2, 3, 4, 5])

d.reverse()
print(d)        # → deque([5, 4, 3, 2, 1])

d2 = d.copy()   # shallow copy
d2.append(0)
print(d)        # → deque([5, 4, 3, 2, 1])  (original unchanged)

d.clear()
print(d)        # → deque([])
print(len(d))   # → 0
```

### 9) Performance comparison vs list (与list性能对比)

```python
import timeit
from collections import deque

# prepend 100_000 items
list_time  = timeit.timeit(lambda: [0] * 100_000, number=100)
deque_time = timeit.timeit(lambda: deque([0] * 100_000), number=100)

# insert at front
n = 10_000
t_list  = timeit.timeit(lambda: [None] + list(range(n)), number=1000)
t_deque = timeit.timeit(lambda: deque([None]) + deque(range(n)), number=1000)

print(f"list  front-insert: {t_list:.4f}s")
print(f"deque front-insert: {t_deque:.4f}s")
# deque is orders of magnitude faster for front operations
```

------

## 5. namedtuple — Immutable Record (具名元组)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">namedtuple</code> creates a <span style="color:#E8600A;font-weight:700">tuple subclass (元组子类)</span> whose fields can be accessed by <span style="color:#2980B9">name</span> as well as by index. It is <span style="color:#E8600A;font-weight:700">immutable (不可变)</span>, memory-efficient, and self-documenting. </div>

### 1) Factory function `namedtuple(typename, field_names)`

```python
from collections import namedtuple

# Three equivalent ways to define field names:
Point = namedtuple("Point", ["x", "y"])
Point = namedtuple("Point", "x y")
Point = namedtuple("Point", "x, y")

p = Point(3, 4)
print(p)           # → Point(x=3, y=4)
print(p.x, p.y)    # → 3 4       (by name)
print(p[0], p[1])  # → 3 4       (by index)
print(p == (3, 4)) # → True      (is a tuple subclass)
```

### 2) `_make()` — Create from iterable (从可迭代对象创建)

```python
from collections import namedtuple

Employee = namedtuple("Employee", "name age department salary")

data = ["Alice", 30, "Engineering", 95000]
emp  = Employee._make(data)
print(emp)
# → Employee(name='Alice', age=30, department='Engineering', salary=95000)

# From CSV row
import csv, io
csv_data = "Bob,25,Marketing,60000"
for row in csv.reader(io.StringIO(csv_data)):
    e = Employee._make(row)
    print(f"{e.name} in {e.department}")
# → Bob in Marketing
```

### 3) `_asdict()` — Convert to OrderedDict (转换为有序字典)

```python
from collections import namedtuple

Point3D = namedtuple("Point3D", "x y z")
p = Point3D(1, 2, 3)

d = p._asdict()
print(d)            # → {'x': 1, 'y': 2, 'z': 3}
print(type(d))      # → <class 'dict'>

# Serialize to JSON
import json
print(json.dumps(p._asdict()))   # → {"x": 1, "y": 2, "z": 3}
```

### 4) `_replace()` — Create modified copy (创建修改副本)

```python
from collections import namedtuple

# namedtuple is IMMUTABLE — _replace() returns a new instance
Person = namedtuple("Person", "name age city")
alice  = Person("Alice", 30, "NYC")

# "update" one field
older_alice = alice._replace(age=31)
print(alice)        # → Person(name='Alice', age=30, city='NYC')  (unchanged)
print(older_alice)  # → Person(name='Alice', age=31, city='NYC')
```

### 5) `_fields` / `_field_defaults` — Introspection (内省)

```python
from collections import namedtuple

Config = namedtuple("Config", "host port timeout", defaults=["localhost", 8080, 30])

print(Config._fields)          # → ('host', 'port', 'timeout')
print(Config._field_defaults)  # → {'host': 'localhost', 'port': 8080, 'timeout': 30}

c1 = Config()                  # all defaults
c2 = Config("example.com")     # override host only
print(c1)  # → Config(host='localhost', port=8080, timeout=30)
print(c2)  # → Config(host='example.com', port=8080, timeout=30)
```

### 6) `rename=True` — Auto-rename invalid field names

```python
from collections import namedtuple

# 'class' and '2bad' are invalid Python identifiers
T = namedtuple("T", ["class", "2bad", "ok"], rename=True)
print(T._fields)   # → ('_0', '_1', 'ok')  (invalid names → _index)

t = T(1, 2, 3)
print(t._0, t._1, t.ok)   # → 1 2 3
```

### 7) Subclassing namedtuple — Adding methods

```python
from collections import namedtuple
import math

class Vector(namedtuple("Vector", "x y")):
    """Extend namedtuple with custom methods."""

    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2)

    def dot(self, other: "Vector") -> float:
        return self.x * other.x + self.y * other.y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

v1 = Vector(3, 4)
v2 = Vector(1, 2)

print(v1.magnitude())   # → 5.0
print(v1.dot(v2))       # → 11.0
print(v1 + v2)          # → Vector(x=4, y=6)
```

------

## 6. ChainMap — Multi-scope Lookup (多层级查找映射)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> A <span style="color:#E8600A;font-weight:700">ChainMap (链式映射)</span> groups multiple dicts into a single, updateable view. Lookups search the dicts <span style="color:#2980B9">from first to last</span>, returning the first match. Writes always go to the <span style="color:#E8600A;font-weight:700">first map</span>. Perfect for modeling <span style="color:#2980B9">variable scopes (变量作用域)</span> like Python's own LEGB rule. </div>

### 1) Basic lookup (基本查找)

```python
from collections import ChainMap

defaults  = {"color": "red",  "user": "guest", "timeout": 30}
env_vars  = {"color": "blue", "debug": True}
cli_args  = {"timeout": 10}

# Priority: cli_args > env_vars > defaults
config = ChainMap(cli_args, env_vars, defaults)

print(config["color"])    # → blue   (from env_vars, overrides defaults)
print(config["user"])     # → guest  (only in defaults)
print(config["timeout"])  # → 10     (from cli_args, highest priority)
print(config["debug"])    # → True   (from env_vars)
```

### 2) Writes go to first map only (写入仅影响第一个映射)

```python
from collections import ChainMap

base    = {"x": 1, "y": 2}
overlay = {}

cm = ChainMap(overlay, base)

cm["x"] = 99      # written to overlay (first map)
cm["z"] = 0       # new key also goes to overlay

print(overlay)    # → {'x': 99, 'z': 0}
print(base)       # → {'x': 1, 'y': 2}   (unchanged!)
print(cm["x"])    # → 99   (overlay shadows base)
print(cm["y"])    # → 2    (from base)
```

### 3) `new_child(m=None)` — Push a new scope (推入新作用域)

```python
from collections import ChainMap

# Simulate nested scopes (模拟嵌套作用域)
global_scope = ChainMap({"x": 1, "y": 2})
local_scope  = global_scope.new_child({"x": 10, "z": 3})

print(local_scope["x"])   # → 10   (local shadows global)
print(local_scope["y"])   # → 2    (falls through to global)
print(local_scope["z"])   # → 3    (local only)

# Pop the local scope (返回父作用域)
parent_scope = local_scope.parents
print(parent_scope["x"])  # → 1    (original global value)
```

### 4) `maps` attribute — Access underlying dicts (访问底层字典列表)

```python
from collections import ChainMap

cm = ChainMap({"a": 1}, {"b": 2}, {"c": 3})

print(cm.maps)
# → [{'a': 1}, {'b': 2}, {'c': 3}]

# Modify underlying dicts directly
cm.maps[1]["b"] = 99
print(cm["b"])   # → 99
```

### 5) Practical: CLI argument + environment + defaults

```python
from collections import ChainMap
import os

def build_config(cli_args: dict) -> ChainMap:
    """Three-tier configuration (三层配置): CLI > ENV > defaults."""
    defaults = {
        "host":    "localhost",
        "port":    8080,
        "debug":   False,
        "workers": 4,
    }
    env_config = {
        k.lower().replace("app_", ""): v
        for k, v in os.environ.items()
        if k.startswith("APP_")
    }
    return ChainMap(cli_args, env_config, defaults)

config = build_config({"port": 9090, "debug": True})
print(config["host"])     # → localhost (from defaults)
print(config["port"])     # → 9090      (from cli_args)
print(config["debug"])    # → True      (from cli_args)
```

------

## 7. UserDict / UserList / UserString — Custom Containers (自定义容器基类)

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <span style="color:#E8600A;font-weight:700">UserDict</span>, <span style="color:#E8600A;font-weight:700">UserList</span>, and <span style="color:#E8600A;font-weight:700">UserString</span> are wrapper classes designed for <span style="color:#2980B9">safe subclassing</span>. Subclassing built-in <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">dict</code> / <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">list</code> directly can miss overrides because C-level methods call each other without going through Python. <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">UserDict</code> etc. route ALL operations through Python methods. </div>

### 1) `UserDict` — Custom dict with validation (带验证的自定义字典)

```python
from collections import UserDict

class TypedDict(UserDict):
    """A dict that only accepts string keys and int values."""

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError(f"Key must be str, got {type(key).__name__}")
        if not isinstance(value, int):
            raise TypeError(f"Value must be int, got {type(value).__name__}")
        super().__setitem__(key, value)   # delegate to UserDict

td = TypedDict()
td["score"] = 100
td["count"] = 42
print(td)            # → {'score': 100, 'count': 42}

try:
    td[123] = 10     # invalid key
except TypeError as e:
    print(f"Error: {e}")   # → Error: Key must be str, got int

try:
    td["x"] = "hello"  # invalid value
except TypeError as e:
    print(f"Error: {e}")   # → Error: Value must be int, got str
```

### 2) `UserList` — Custom list with constraints (带约束的自定义列表)

```python
from collections import UserList

class BoundedList(UserList):
    """A list that enforces a maximum length (最大长度限制)."""

    def __init__(self, maxlen: int, iterable=()):
        self.maxlen = maxlen
        super().__init__()
        for item in iterable:
            self.append(item)

    def append(self, item):
        if len(self.data) >= self.maxlen:
            raise OverflowError(f"List is full (max {self.maxlen})")
        self.data.append(item)

    def insert(self, index, item):
        if len(self.data) >= self.maxlen:
            raise OverflowError(f"List is full (max {self.maxlen})")
        self.data.insert(index, item)

bl = BoundedList(3, [1, 2, 3])
print(bl)   # → [1, 2, 3]

try:
    bl.append(4)
except OverflowError as e:
    print(f"Error: {e}")   # → Error: List is full (max 3)
```

### 3) `UserString` — Custom string with transforms (带转换的自定义字符串)

```python
from collections import UserString

class SlugString(UserString):
    """Auto-converts string to URL-safe slug (URL友好字符串)."""

    def __init__(self, seq=""):
        import re
        slug = re.sub(r'[^a-z0-9]+', '-', str(seq).lower()).strip('-')
        super().__init__(slug)

    def __add__(self, other):
        return SlugString(self.data + "-" + str(other))

s = SlugString("Hello World! This is a Test.")
print(s)           # → hello-world-this-is-a-test

s2 = s + "extra"
print(s2)          # → hello-world-this-is-a-test-extra
print(len(s))      # → 28   (all str methods work)
print(s.upper())   # → HELLO-WORLD-THIS-IS-A-TEST
```

------

## 8. Comparison Table (对比总结)

| Class                                                        | Based on  | Missing key    | Ordered    | Mutable   | Best use case                 |
| ------------------------------------------------------------ | --------- | -------------- | ---------- | --------- | ----------------------------- |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">defaultdict</code> | dict      | auto-creates   | insertion  | ✅         | Grouping, counting            |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Counter</code> | dict      | returns 0      | insertion  | ✅         | Frequency, multiset ops       |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">OrderedDict</code> | dict      | KeyError       | insertion  | ✅         | LRU cache, order-sensitive eq |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">deque</code> | list-like | IndexError     | yes        | ✅         | Queue, stack, sliding window  |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">namedtuple</code> | tuple     | AttributeError | yes        | ❌         | Immutable records, CSV rows   |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">ChainMap</code> | dict view | KeyError       | first-wins | ✅ (first) | Config layers, scopes         |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">UserDict</code> | dict      | KeyError       | insertion  | ✅         | Safe dict subclassing         |

------

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">defaultdict</code> to eliminate <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">KeyError</code> boilerplate, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Counter</code> for frequency analysis and multiset arithmetic, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">deque</code> when you need O(1) operations on both ends, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">namedtuple</code> for self-documenting immutable records, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">OrderedDict</code> for LRU caches and order-sensitive comparisons, and <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">ChainMap</code> for multi-tier configuration or scope simulation. </div>
