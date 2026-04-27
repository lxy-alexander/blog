---
title: "@property"
published: 2026-04-27
description: "@property"
image: ""
tags: ["python","@property"]
category: python
draft: false
lang: ""
createdAt: "2026-04-27T15:07:31.689.752649194Z"
---

# Python `@property` Decorator

The `@property` decorator (装饰器) turns a method (方法) into a read-only attribute (只读属性), allowing controlled access (受控访问) to instance data without calling parentheses `()`.

## 1. Basic Usage

`@property` lets you access a method like an attribute (属性), which is useful for computed values (计算值) or encapsulation (封装).

```python
class Circle:
    def __init__(self, radius):
        self.radius = radius

    @property
    def area(self):
        return 3.14 * self.radius ** 2

c = Circle(5)
print(c.area)   # 78.5  (no parentheses needed)
# c.area = 100  # AttributeError: can't set attribute
```

<br>

## 2. Setter and Deleter

Pair `@property` with `@<name>.setter` and `@<name>.deleter` to allow controlled writing (写入) and deletion (删除), often with validation (校验).

```python
class Person:
    def __init__(self, age):
        self._age = age

    @property
    def age(self):
        return self._age

    @age.setter
    def age(self, value):
        if value < 0:
            raise ValueError("Age cannot be negative")
        self._age = value

    @age.deleter
    def age(self):
        del self._age

p = Person(20)
p.age = 25
print(p.age)   # 25
# p.age = -1   # ValueError
```

<br>

## 3. Encapsulation Pattern

`@property` is the Pythonic way (Python 风格) to implement getters (取值器) and setters (赋值器), replacing Java-style `get_x()` / `set_x()` methods.

```python
class Account:
    def __init__(self, balance):
        self._balance = balance  # private attribute (私有属性)

    @property
    def balance(self):
        return self._balance

    @balance.setter
    def balance(self, value):
        if value < 0:
            raise ValueError("Balance cannot be negative")
        self._balance = value

a = Account(100)
print(a.balance)   # 100
a.balance = 200
print(a.balance)   # 200
```

<br>

## 4. Read-Only Computed Property

Defining only `@property` without a setter creates a read-only property (只读属性), ideal for derived attributes (派生属性).

```python
class Rectangle:
    def __init__(self, w, h):
        self.w = w
        self.h = h

    @property
    def area(self):
        return self.w * self.h

r = Rectangle(3, 4)
print(r.area)   # 12
# r.area = 20   # AttributeError
```

<br>

## 5. Summary Table

| Decorator       | Purpose                 | Triggered By       |
| --------------- | ----------------------- | ------------------ |
| `@property`     | Define getter (取值器)  | `obj.attr`         |
| `@attr.setter`  | Define setter (赋值器)  | `obj.attr = value` |
| `@attr.deleter` | Define deleter (删除器) | `del obj.attr`     |

<br> <br>
