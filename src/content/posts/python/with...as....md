---
title: "with...as..."
published: 2026-05-20
description: "with...as..."
image: ""
tags: ["python","with...as..."]
category: python
draft: false
lang: ""
createdAt: "2026-05-20T06:32:03.541.329238600Z"
---

# Python `with as`

`with as` is used to manage resources automatically and make cleanup safer.

## 1. Core Idea

`with as` uses a context manager（上下文管理器） to automatically enter, use, and clean up a resource.

```python
with open("demo.txt", "w") as f:
    f.write("hello")

# Output:
# The file is opened, written, and automatically closed.
```

<br>

## 2. Execution Flow

The object after `with` must implement `__enter__()` and `__exit__()`, while the variable after `as` receives the value returned by `__enter__()`.

```
class Demo:
    def __enter__(self):
        print("enter")
        return "resource"

    def __exit__(self, exc_type, exc_value, traceback):
        print("exit")

with Demo() as x:
    print(x)

# Output:
# enter
# resource
# exit
```

<br>

## 3. `@contextmanager`

`@contextmanager`（上下文管理器装饰器） lets a generator（生成器） function behave like a context manager. ==`yield` pauses a function, returns a value, and allows the function to continue from the same position later.==

```
from contextlib import contextmanager

@contextmanager
def demo():
    print("enter")
    try:
        yield "resource" # 
    finally:
        print("exit")

with demo() as x:
    print(x)

# Output:
# enter
# resource
# exit
```

<br>

<br>
