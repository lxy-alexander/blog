---
title: "Python typing"
published: 2026-03-08
description: "Python typing"
image: ""
tags: ["python","Python typing"]
category: python
draft: false
lang: ""
---

# **I. Python `typing` Learning Handbook**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">

<span style="color:#E8600A;font-weight:700">`typing` Module (类型注解模块)</span> is a built-in Python module used to add <span style="color:#E8600A;font-weight:700">Type Hints (类型提示)</span> to code.
Type hints improve code readability and allow <span style="color:#E8600A;font-weight:700">Static Type Checkers (静态类型检查器)</span> such as <span style="color:#E8600A;font-weight:700">mypy</span> or <span style="color:#E8600A;font-weight:700">pyright</span> to detect type errors before runtime.

The goal of Python typing is not to enforce types during execution, but to enable <span style="color:#E8600A;font-weight:700">Static Analysis (静态分析)</span> and clearer program design.

</div>

---

# **II. Core Typing Concepts**

## <span style="color:#E8600A">1.</span> **`Any` — Disable Type Checking**

### 1) Concept

<span style="color:#E8600A;font-weight:700">`Any` Type (任意类型)</span> means that any value is allowed and type checking is effectively disabled.

```python
from typing import Any

x: Any = 123
x = "abc"
```

Here:

* `x` can hold values of any type
* the type checker will not report errors

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span>
Using too much <span style="color:#C0392B;font-weight:600">`Any` weakens type safety</span> and should generally be avoided in large systems.
</div>

---

## <span style="color:#E8600A">2.</span> **`Literal` — Restrict Allowed Values**

### 1) Concept

<span style="color:#E8600A;font-weight:700">`Literal` Type (字面量类型)</span> restricts a variable to specific constant values.

```python
from typing import Literal

mode: Literal["r", "w", "a"] = "r"
```

Meaning:

`mode` can only be:

```
"r" | "w" | "a"
```

Common usage:

* configuration parameters
* API modes
* enum-like constraints

---

## <span style="color:#E8600A">3.</span> **`Callable` — Function Types**

### 1) Concept

<span style="color:#E8600A;font-weight:700">`Callable` (函数类型)</span> describes a function's parameter types and return type.

```python
from typing import Callable

f: Callable[[int, int], int]
```

Meaning:

```
(int, int) -> int
```

Example:

```python
from typing import Callable

def add(a: int, b: int) -> int:
    return a + b

f: Callable[[int, int], int] = add
```

This means `f` refers to a function that takes two integers and returns an integer.

---

# **III. Container Typing**

## <span style="color:#E8600A">1.</span> **Difference: `list` vs `Sequence` vs `Iterable`**

Three levels of abstraction exist for containers.

| Type          | Meaning                  |
| ------------- | ------------------------ |
| `list[T]`     | must be a list           |
| `Sequence[T]` | indexable container      |
| `Iterable[T]` | anything you can iterate |

Example:

```python
from typing import Sequence, Iterable

def a(x: list[int]): ...
def b(x: Sequence[int]): ...
def c(x: Iterable[int]): ...
```

Key differences:

| Type     | index access | length     | example            |
| -------- | ------------ | ---------- | ------------------ |
| list     | ✔            | ✔          | list               |
| Sequence | ✔            | ✔          | list / tuple / str |
| Iterable | ❌ required   | ❌ required | generators         |

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span>
<span style="color:#C0392B;font-weight:600">`dict` is not a `Sequence`</span> because it cannot be accessed by numeric index.
</div>

---

## <span style="color:#E8600A">2.</span> **Generic Container Types**

Since Python 3.9, built-in containers support generics directly.

```python
a: list[int]
b: dict[str, int]
c: tuple[int, str]
```

These are called <span style="color:#E8600A;font-weight:700">Parameterized Types (参数化类型)</span>.

---

# **IV. Advanced Typing Tools**

## <span style="color:#E8600A">1.</span> **`Type[T]` — Class Objects**

### 1) Concept

<span style="color:#E8600A;font-weight:700">`Type[T]` (类对象类型)</span> represents a class object that produces instances of `T`.

```python
from typing import Type

def f(cls: Type[int]):
    print(cls)
```

Usage:

```
f(int)   # correct
f(123)   # incorrect
```

---

### 2) Factory Function Example

```python
from typing import Type, TypeVar

T = TypeVar("T")

def create(cls: Type[T]) -> T:
    return cls()

x = create(list)
y = create(dict)
```

Here:

* input: class
* output: instance

The type checker correctly infers the return type.

---

### 3) Restrict Allowed Classes

```python
from typing import Type

class Animal: ...
class Dog(Animal): ...

def adopt(cls: Type[Animal]) -> Animal:
    return cls()

adopt(Dog)
```

Only subclasses of `Animal` are allowed.

---

### 4) Classmethod Typing

```python
from typing import TypeVar, Type

T = TypeVar("T", bound="Base")

class Base:
    @classmethod
    def new(cls: Type[T]) -> T:
        return cls()

class User(Base):
    pass

u = User.new()
```

Here the inferred type is:

```
User
```

not `Base`.

---

## <span style="color:#E8600A">2.</span> **`TypeAlias` — Type Aliases**

<span style="color:#E8600A;font-weight:700">Type Alias (类型别名)</span> improves readability.

```python
from typing import TypeAlias

UserId: TypeAlias = int
Names: TypeAlias = list[str]
```

Usage:

```python
def get_user_name(user_id: UserId) -> str:
    ...

def print_names(names: Names) -> None:
    for n in names:
        print(n)
```

Here:

```
UserId -> int
Names -> list[str]
```

---

## <span style="color:#E8600A">3.</span> **`TypedDict` — Typed Dictionaries**

<span style="color:#E8600A;font-weight:700">`TypedDict` (类型字典)</span> defines the expected keys and value types.

```python
from typing import TypedDict

class User(TypedDict):
    id: int
    name: str
```

Useful for:

* JSON responses
* configuration objects
* API schemas

---

## <span style="color:#E8600A">4.</span> **`Protocol` — Structural Typing**

<span style="color:#E8600A;font-weight:700">`Protocol` (结构化类型接口)</span> enables <span style="color:#E8600A;font-weight:700">Structural Typing (结构类型系统)</span>.

If an object has the required methods, it satisfies the protocol.

```python
from typing import Protocol

class HasLen(Protocol):
    def __len__(self) -> int: ...
```

Any object implementing `__len__` matches this type.

---

### Protocol vs Abstract Base Class

| Feature              | Protocol   | ABC     |
| -------------------- | ---------- | ------- |
| inheritance required | ❌          | ✔       |
| typing style         | structural | nominal |
| flexibility          | high       | strict  |

Example:

```python
from typing import Protocol

class Speaker(Protocol):
    def speak(self) -> str: ...

class Robot:
    def speak(self) -> str:
        return "beep"

def talk(x: Speaker) -> None:
    print(x.speak())

talk(Robot())
```

---

# **V. Generic Programming**

## <span style="color:#E8600A">1.</span> **`TypeVar` — Generic Type Variables**

<span style="color:#E8600A;font-weight:700">Generic Types (泛型)</span> allow writing reusable typed functions.

```python
from typing import TypeVar

T = TypeVar("T")

def first(xs: list[T]) -> T:
    return xs[0]
```

Here:

```
list[int] -> int
list[str] -> str
```

The function adapts to the input type.

---

## <span style="color:#E8600A">2.</span> **`overload` — Multiple Type Signatures**

<span style="color:#E8600A;font-weight:700">Function Overloading (函数重载)</span> allows multiple type signatures for one implementation.

```python
from typing import overload

@overload
def parse(x: int) -> str: ...

@overload
def parse(x: str) -> int: ...
```

Implementation:

```python
def parse(x):
    if isinstance(x, int):
        return str(x)
    return int(x)
```

Usage:

```
parse(123)   -> str
parse("123") -> int
```

---

## <span style="color:#E8600A">3.</span> **`Annotated` — Metadata for Types**

<span style="color:#E8600A;font-weight:700">`Annotated` Type (带元数据类型)</span> allows attaching metadata to types.

```python
from typing import Annotated

Age = Annotated[int, "must be >= 0"]
```

Common uses:

* FastAPI request validation
* Pydantic models
* runtime validators

---

# **VI. Mental Model for Mastering `typing`**

A complete understanding of Python typing requires studying it from multiple perspectives.

| Perspective | Focus                      |
| ----------- | -------------------------- |
| syntax      | how to annotate types      |
| containers  | typing for collections     |
| generics    | reusable type abstractions |
| interfaces  | protocols and class typing |
| tooling     | static type checkers       |

Understanding these layers allows developers to design more robust Python systems.

---

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>

<span style="color:#E8600A;font-weight:700">Python `typing` (类型系统)</span> introduces type hints that enable static analysis, safer APIs, and clearer program structure while keeping Python’s dynamic runtime behavior unchanged.

</div>
