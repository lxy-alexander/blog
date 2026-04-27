---
title: "@dataclass"
published: 2026-04-27
description: "@dataclass"
image: ""
tags: ["python","@dataclass"]
category: python
draft: false
lang: ""
createdAt: "2026-04-27T15:07:38.579.271503869Z"
---

# Python Dataclass

Python `dataclass` is a decorator (装饰器) that auto-generates boilerplate code (样板代码) like `__init__`, `__repr__`, and `__eq__` for classes that mainly store data (数据).

## 1. Basic Usage

==The `@dataclass` decorator (装饰器) automatically generates the constructor (构造函数) and other dunder methods== (双下划线方法) from class-level type annotations (类型注解).

-   `__init__`: A special method called automatically when a new object is created; it initializes the object’s attributes.
-   `__repr__`: A special method that returns an official string representation of an object, mainly for debugging.
-   `__eq__`: A special method that defines how two objects are compared using `==`(double equals).

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int

p = Point(1, 2)
print(p)         # Point(x=1, y=2)
print(p == Point(1, 2))  # True
```

<br>

## 2. Default Values

Fields (字段) can have default values (默认值), but mutable defaults (可变默认值) like `list` must use `field(default_factory=...)` to avoid shared state (共享状态).

```python
from dataclasses import dataclass, field

@dataclass
class Student:
    name: str
    age: int = 18
    scores: list = field(default_factory=list)

s = Student("Tom")
s.scores.append(90)
print(s)  # Student(name='Tom', age=18, scores=[90])
```

<br>

## 3. Frozen Dataclass

Setting `frozen=True` makes instances immutable (不可变), so they can be used as dictionary keys (字典键) or in a set (集合).

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Coord:
    x: int
    y: int

c = Coord(1, 2)
print({c: "origin"})  # {Coord(x=1, y=2): 'origin'}
# c.x = 5  # raises FrozenInstanceError
```

<br>

## 4. Ordering Support

The `order=True` parameter (参数) auto-generates comparison methods (比较方法) like `<`, `<=`, `>`, `>=` based on field order (字段顺序).

```python
from dataclasses import dataclass

@dataclass(order=True)
class Item:
    priority: int
    name: str

a = Item(1, "low")
b = Item(2, "high")
print(a < b)  # True
```

<br>

## 5. Post-Init Processing

The `__post_init__` method runs after `__init__` and is used for validation (校验) or computing derived fields (派生字段).

```python
from dataclasses import dataclass

@dataclass
class Rectangle:
    width: float
    height: float

    def __post_init__(self):
        self.area = self.width * self.height

r = Rectangle(3, 4)
print(r.area)  # 12
```

<br>

## 6. Convert to Dict or Tuple

`asdict()` and `astuple()` recursively (递归地) convert a dataclass instance into a dict (字典) or tuple (元组), useful for serialization (序列化).

```python
from dataclasses import dataclass, asdict, astuple

@dataclass
class User:
    name: str
    age: int

u = User("Alice", 25)
print(asdict(u))   # {'name': 'Alice', 'age': 25}
print(astuple(u))  # ('Alice', 25)
```

<br>

## 7. Key Parameters Summary

The main parameters (参数) of `@dataclass` control which methods (方法) are auto-generated, balancing convenience (便利性) and safety (安全性).

| Parameter | Default | Function                                  |
| --------- | ------- | ----------------------------------------- |
| `init`    | `True`  | Generate `__init__`                       |
| `repr`    | `True`  | Generate `__repr__`                       |
| `eq`      | `True`  | Generate `__eq__`                         |
| `order`   | `False` | Generate comparison methods               |
| `frozen`  | `False` | Make instance immutable                   |
| `slots`   | `False` | Use `__slots__` to save memory (节省内存) |

<br> <br>
