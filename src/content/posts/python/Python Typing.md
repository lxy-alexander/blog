---
title: "Python Typing"
published: 2026-02-02
description: "Python Typing"
image: ""
tags: ["python","Python Typing"]
category: python
draft: false
lang: ""
---



>   `typing` is a built-in Python module that add **type hints** to make the code clearer. `typing` helps static type checkers detect type errors earlier (before runtime)

------

# Common typing concepts

## 1) `Any`

Means “any type is allowed” (no type checking).

```python
from typing import Any
x: Any = 123
x = "abc"
```

------



## 2) `Literal`

Restricts values to specific constants (common in API parameters).

```python
from typing import Literal

mode: Literal["r", "w", "a"] = "r"
```

------



## 3) `Callable`

Represents a function type: argument types + return type.

```python
from typing import Callable

f: Callable[[int, int], int]
```

```python
from typing import Callable

def add(a: int, b: int) -> int:
    return a + b

f: Callable[[int, int], int] = add

```

Meaning: takes `(int, int)` and returns `int`.

------



### 4) Difference: `Sequence` / `Iterable` / `list`

-   `list[T]`: must be a list
-   `Sequence[T]`: list/tuple/str. (indexable + has length)，==dict is not since it cannot be visited by index==
-   `Iterable[T]`: anything you can loop over (`for x in ...`)

```python
from typing import Sequence, Iterable

def a(x: list[int]): ...
def b(x: Sequence[int]): ...
def c(x: Iterable[int]): ...
```

------



### 5) Generic container types (`list` / `dict` / `tuple`)

Python 3.9+ style，==they are built-in container types==:

```python
a: list[int]
b: dict[str, int]
c: tuple[int, str]
```

------



### 6) `Type[T]`

Represents a **class object**, not an instance. This enables the type checker to make more accurate inferences and reduce misuse.

```python
from typing import Type

def f(cls: Type[int]):
    print(cls)

f(int)     # ✅ 传的是 int 这个“类”
f(123)     # ❌ 传的是实例，不是类
```

#### What is `Type[T]` used for?

-   `TypeVar("T")` creates a **type variable** called `T`.
-   `Type[T]` means **“a class object that creates instances of `T`.”**So it’s used when you want to **pass a class (not an instance)** as an argument.

------

#### 1) Factory functions (pass a class, return an instance)

```py
from typing import Type, TypeVar

T = TypeVar("T")

def create(cls: Type[T]) -> T:
    return cls()

x = create(list)   # inferred type: list
y = create(dict)   # inferred type: dict
```

✅ `Type[T]` helps the type checker infer the correct return type.

------

#### 2) Restrict allowed classes (only subclasses of something)

```py
from typing import Type

class Animal: ...
class Dog(Animal): ...

def adopt(cls: Type[Animal]) -> Animal:
    return cls()

adopt(Dog)   # ✅ OK
adopt(int)   # ❌ not allowed
```

------

#### 3) Better typing for `classmethod` (return the subclass type)

```python
from typing import TypeVar, Type

T = TypeVar("T", bound="Base")

class Base:
    @classmethod
    def new(cls: Type[T]) -> T:
        return cls()

class User(Base):
    pass

u = User.new()   # inferred as User, not Base
```

what `new` does: `cls` is the class that calls this method, and `return cls()` returns an object created from this class.





### 7) `TypeAlias`

Used to define a type alias for readability.

```python
from typing import TypeAlias

UserId: TypeAlias = int
Names: TypeAlias = list[str]
```

------

UserId is an alias for int (it's still essentially an int, just with a more descriptive name). 

Names is an alias for list[str] (a list of strings).

```python
def get_user_name(user_id: UserId) -> str:
    ...

def print_names(names: Names) -> None:
    for n in names:
        print(n)
```







### 8) `TypedDict`

Defines the expected keys + value types of a dictionary (good for JSON-like data).

```python
from typing import TypedDict

class User(TypedDict):
    id: int
    name: str
```

------



### 9) `Protocol`

Defines a structural interface (it depends on duck typing). As long as a class meets the attributes and methods defined by the protocol, it is considered to have implemented this protocol

```python
from typing import Protocol

class HasLen(Protocol):
    def __len__(self) -> int: ...
```

Any object with `__len__` matches this type.

------

```python
from abc import ABC, abstractmethod


# interface
from abc import ABC, abstractmethod

class Speaker(ABC):
    @abstractmethod
    def speak(self) -> str:
        pass

class Human(Speaker):  # ✅ 必须显式继承
    def speak(self) -> str:
        return "hello"

h = Human()
print(h.speak())


# Protocol
from typing import Protocol

class Speaker(Protocol):
    def speak(self) -> str: ...

class Robot:  # ✅ 不用继承 Speaker
    def speak(self) -> str:
        return "beep"

def talk(x: Speaker) -> None:
    print(x.speak())

talk(Robot())  # ✅ 只要有 speak() 就行

```







### 10) `Generic(泛型)` / `TypeVar`

Used for generic functions and containers.

```python
from typing import TypeVar

T = TypeVar("T")

def first(xs: list[T]) -> T:
    return xs[0]
```

------



### 11) `overload`

Allows multiple type signatures(类型签名) for one function.

```python
from typing import overload

@overload
def parse(x: int) -> str: ...
@overload
def parse(x: str) -> int: ...

def parse(x):
    if isinstance(x, int):
        return str(x)
    return int(x)


parse(123)     # inferred type: str
parse("123")   # inferred type: int

```

------



### 12) `Annotated`

Adds extra metadata to a type (often used in FastAPI / Pydantic).

```py
from typing import Annotated

Age = Annotated[int, "must be >= 0"]
```

------

