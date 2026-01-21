---
title: Python Interview
published: 2025-01-15
description: "Python Interview"
image: "./cover.jpeg"
tags: ["Blogging", "Python Interview"]
category: Guides
draft: false
---

# Python Interview 

## CPython

Python is a **language specification**, while CPython is a **concrete implementation** of that specification.

<u>Python defines *what the language is*; CPython defines *how the language runs*.</u>



### What Does CPython Do?

- Compiles source code into bytecode (`.pyc`)
- Executes bytecode using the Python Virtual Machine (PVM)



## GIL

The **GIL (Global Interpreter Lock)** is a mechanism for thread safty. It <u>ensures **only one thread executes Python bytecode at a time**.</u>



## <u>GIL is periodically switched</u>

Periodic GIL switching means that **CPython forces the currently running thread to release the GIL at regular time intervals** Without periodic switching, a CPU-bound thread could **monopolize the GIL**,
causing other threads (including I/O threads) to starve.

<u>if the execution time exceeds `switchinterval`</u>

<u>The current thread releases the GIL at the **next safe point**</u>

<u>Other threads then get a chance to acquire the GIL</u>

```python
import sys
sys.getswitchinterval()
```



## thread.join

`join()` blocks the **calling thread** until the target thread finishes.让**调用它的线程阻塞**，直到目标线程执行完毕。



## Convoy effect

In CPython, the convoy effect <u>may occur when a CPU-bound thread holds the GIL</u>, delaying I/O-bound threads that are ready to run.



## Descriptor

A descriptor is <u>an object that is stored as a **class attribute** and implements one or more of `__get__`, `__set__`, or `__delete__`.</u>When the attribute is accessed, Python invokes these methods instead of returning a value directly. 描述符（Descriptor）是一个对象，只要它作为**类属性**存在，并且实现了 `__get__`、`__set__` 或 `__delete__` 中的任意一个，Python 在访问该属性时就会调用这些方法，而不是直接取值。

```python
class MyDescriptor:
    def __get__(self, instance, owner):
        return "hello from descriptor"

class A:
    x = MyDescriptor()

a = A()
print(a.x)
```

 when Python evaluates `obj.attr`, it first checks whether the class attribute is a descriptor. A descriptor is just an **instance of a normal class**, Descriptors are not based on inheritance, but on **implementing a specific protocol**.

| 协议 / Protocol                | 方法 / Methods                     |
| ------------------------------ | ---------------------------------- |
| 描述符 / Descriptor            | `__get__`, `__set__`, `__delete__` |
| 迭代器 / Iterator              | `__iter__`, `__next__`             |
| 上下文管理器 / Context manager | `__enter__`, `__exit__`            |



## Intance

obj.attr
↓
在类 __dict__ 里找到 attr
↓
检查这个对象有没有 __get__ 方法？
↓
有 → 当描述符用
没有 → 当普通属性



## **`{}`, `[]`, and `()` in Python**

| Symbols | Name            | Common uses                       |
| ------- | --------------- | --------------------------------- |
| `{}`    | curly braces    | `dict`, `set`                     |
| `[]`    | square brackets | `list`, indexing                  |
| `()`    | parentheses     | `tuple`, function calls, grouping |

| Type  | Mutable | Ordered | Literal |
| ----- | ------- | ------- | ------- |
| list  | ✅       | ✅       | `[]`    |
| tuple | ❌       | ✅       | `()`    |
| set   | ✅       | ❌       | `{}`    |
| dict  | ✅       | ✅       | `{}`    |





## Set

`{'a','i','o','e','u'}` is a Python `set`; the semicolon is harmless but unnecessary.

Typically this question means:↳

```python
# set
vowels = {'a', 'e', 'i', 'o', 'u'}
c in vowels
```

vs

```python
# list
vowels = ['a', 'e', 'i', 'o', 'u']
c in vowels
```

| Type   | Operation `c in container` | Time complexity  |
| ------ | -------------------------- | ---------------- |
| `set`  | hash lookup                | **O(1)** average |
| `list` | linear scan                | **O(n)**         |

`"aeiou"` is fast because it’s tiny, runs in optimized C code, and avoids hashing overhead 





## Defaultdict

defaultdict ` provides default values automatically; `dict` does not.

`defaultdict(int)` comes from Python’s `collections` module. It’s a dictionary that **automatically creates missing keys** with a default value.↳

In this case, `int()` is the default factory, and `int()` → `0`.

`dict` (normal dictionary)

```python
d = {}
d['a'] += 1   # ❌ KeyError
```

You must initialize first:

```python
if 'a' not in d:
    d['a'] = 0
d['a'] += 1
```

`defaultdict`

```python
from collections import defaultdict

d = defaultdict(int)
d['a'] += 1   # ✅ starts from 0 automatically
```

Key differences

| Feature              | dict     | defaultdict  |
| -------------------- | -------- | ------------ |
| Missing key          | KeyError | Auto-created |
| Needs initialization | Yes      | No           |
| Code length          | Longer   | Shorter      |



## Range & slice

Both **`range`** and **slicing** use **left-closed, right-open intervals**:





## Counter & dict

**Use `Counter` for pure frequency counting.**

**Use `dict` when counts are part of larger logic or need control.**

| Scenario              | Use       |
| --------------------- | --------- |
| Counting frequencies  | `Counter` |
| Sliding window counts | `Counter` |
| Top-K frequent items  | `Counter` |
| Key → last index      | `dict`    |
| Config / strict data  | `dict`    |
| Missing key = bug     | `dict`    |
| Fixed small domain    | array     |
