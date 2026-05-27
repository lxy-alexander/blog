---
title: "Python Study Notes"
published: 2026-05-26
description: "Python Study Notes"
image: ""
tags: ["python","Python Study Notes"]
category: python
draft: false
lang: ""
createdAt: "2026-05-26T22:11:20.206.617866164Z"
---

# Python Study Notes

These notes cover all core Python knowledge points in a progressive order. Each point has a one-sentence definition and a self-contained runnable example.

<br> <br>

# 1. Basics

The foundation of writing any Python program.

## 1. Variables & Assignment

A variable is a name that points to a value. (变量)

```python
x = 10
name = "Tom"
print(x, name)
# Output: 10 Tom
```

<br>

## 2. Data Types Overview

Python has built-in types like numbers, strings, lists, and dicts. (数据类型)

```python
print(type(10), type(3.14), type("hi"), type([1, 2]))
# Output: <class 'int'> <class 'float'> <class 'str'> <class 'list'>
```

<br>

## 3. Numbers: int, float, complex

Python supports integers, decimals, and complex numbers. (数字)

```python
a = 7          # int
b = 3.14       # float
c = 2 + 3j     # complex (复数)
print(a, b, c)
# Output: 7 3.14 (2+3j)
```

<br>

## 4. Boolean

A bool is either `True` or `False`. (布尔值)

```python
print(5 > 3)
print(bool(0), bool(1))
# Output: True
# Output: False True
```

<br>

## 5. Strings

A string is text wrapped in quotes. (字符串)

```python
s = "Hello"
print(s[0], s[-1], len(s))
# Output: H o 5
```

<br>

## 6. String Methods

Built-in tools to transform and search text. (字符串方法)

```python
s = "  Python  "
print(s.strip().upper())
print("a,b,c".split(","))
# Output: PYTHON
# Output: ['a', 'b', 'c']
```

<br>

## 7. String Formatting: f-string, format, %

Insert variables into a string. (字符串格式化)

```python
name, age = "Tom", 18
print(f"{name} is {age}")
print("{} is {}".format(name, age))
print("%s is %d" % (name, age))
# Output: Tom is 18
# Output: Tom is 18
# Output: Tom is 18
```

<br>

## 8. Type Conversion

Change a value from one type to another. (类型转换)

```python
print(int("10") + 5)
print(str(99) + "!")
print(float("3.5"))
# Output: 15
# Output: 99!
# Output: 3.5
```

<br>

## 9. Input & Output: input(), print()

`input()` reads text; `print()` displays text. (输入输出)

```python
# name = input("Name: ")   # reads user typing
name = "Tom"               # simulated input
print("Hi", name)
# Output: Hi Tom
```

<br>

## 10. Comments

Notes ignored by Python, written with `#`. (注释)

```python
# This is a comment
x = 5  # inline comment
print(x)
# Output: 5
```

<br>

## 11. Operators

Symbols that perform calculations or comparisons. (运算符)

```python
print(7 + 2, 7 // 2, 7 % 2, 2 ** 3)   # arithmetic
print(5 > 3, 5 == 5)                  # comparison
print(True and False, not True)       # logical
print(5 & 3, 5 | 2)                   # bitwise (位运算)
# Output: 9 3 1 8
# Output: True True
# Output: False False
# Output: 1 7
```

<br> <br>

# 2. Control Flow

Decide which code runs and how many times.

## 12. if / elif / else

Run different code based on a condition. (条件判断)

```python
x = 7
if x > 10:
    print("big")
elif x > 5:
    print("medium")
else:
    print("small")
# Output: medium
```

<br>

## 13. while Loop

Repeat while a condition stays true. (while 循环)

```python
i = 0
while i < 3:
    print(i)
    i += 1
# Output: 0
# Output: 1
# Output: 2
```

<br>

## 14. for Loop & range()

Loop over a sequence or a number range. (for 循环)

```python
for i in range(3):
    print(i)
for ch in "ab":
    print(ch)
# Output: 0
# Output: 1
# Output: 2
# Output: a
# Output: b
```

<br>

## 15. break / continue / pass

Stop, skip, or do nothing in a loop. (循环控制)

```python
for i in range(5):
    if i == 3:
        break        # stop loop
    if i == 1:
        continue     # skip this round
    print(i)
# Output: 0
# Output: 2
```

<br>

## 16. Match-Case

Pick a branch by matching a value (Python 3.10+). (匹配语句)

```python
status = 404
match status:
    case 200:
        print("OK")
    case 404:
        print("Not Found")
    case _:
        print("Other")
# Output: Not Found
```

<br> <br>

# 3. Data Structures

Containers that store and organize multiple values.

## 17. List

An ordered, changeable collection. (列表)

```python
nums = [1, 2, 3]
nums.append(4)
print(nums)
# Output: [1, 2, 3, 4]
```

<br>

## 18. List Methods & Slicing

Modify lists and grab sub-parts with `[start:end]`. (列表方法与切片)

```python
nums = [10, 20, 30, 40]
print(nums[1:3])      # slicing
nums.insert(0, 5)
nums.remove(30)
print(nums)
# Output: [20, 30]
# Output: [5, 10, 20, 40]
```

<br>

## 19. Tuple

An ordered but unchangeable collection. (元组)

```python
t = (1, 2, 3)
print(t[0], len(t))
# t[0] = 9  -> error, tuples are immutable (不可变)
# Output: 1 3
```

<br>

## 20. Set

An unordered collection with no duplicates. (集合)

```python
s = {1, 2, 2, 3}
print(s)
print({1, 2} & {2, 3})   # intersection (交集)
# Output: {1, 2, 3}
# Output: {2}
```

<br>

## 21. Dictionary

Stores data as key-value pairs. (字典)

```python
d = {"name": "Tom", "age": 18}
print(d["name"])
d["age"] = 19
print(d)
# Output: Tom
# Output: {'name': 'Tom', 'age': 19}
```

<br>

## 22. Dictionary Methods

Access keys, values, and items safely. (字典方法)

```python
d = {"a": 1, "b": 2}
print(d.get("c", 0))     # default if missing
print(list(d.keys()))
print(list(d.items()))
# Output: 0
# Output: ['a', 'b']
# Output: [('a', 1), ('b', 2)]
```

<br>

## 23. Comprehensions

Build a list, dict, or set in one line. (推导式)

```python
squares = [x**2 for x in range(4)]
evens = {x for x in range(6) if x % 2 == 0}
mapping = {x: x*10 for x in range(3)}
print(squares)
print(evens)
print(mapping)
# Output: [0, 1, 4, 9]
# Output: {0, 2, 4}
# Output: {0: 0, 1: 10, 2: 20}
```

<br>

## 24. Common Built-in Functions

Ready-made helpers for everyday tasks. (常用内置函数)

```python
nums = [3, 1, 2]
print(len(nums), sum(nums), max(nums))
print(sorted(nums))
print(list(zip([1, 2], ["a", "b"])))
print(list(enumerate(["x", "y"])))
# Output: 3 6 3
# Output: [1, 2, 3]
# Output: [(1, 'a'), (2, 'b')]
# Output: [(0, 'x'), (1, 'y')]
```

<br> <br> # 4. Functions

Reusable blocks of code that take input and give output.

## 25. Defining Functions: def

Create a named, reusable block with `def`. (定义函数)

```python
def greet():
    print("Hi")

greet()
# Output: Hi
```

<br>

## 26. Arguments: positional, keyword, default

Pass values in by position, by name, or with a fallback. (参数)

```python
def info(name, age=18):
    print(name, age)

info("Tom")              # default
info("Sue", age=20)      # keyword
# Output: Tom 18
# Output: Sue 20
```

<br>

## 27. *args & **kwargs

Accept any number of extra positional or keyword arguments. (可变参数)

```python
def show(*args, **kwargs):
    print(args, kwargs)

show(1, 2, x=3)
# Output: (1, 2) {'x': 3}
```

<br>

## 28. Return Values

Send a result back with `return`. (返回值)

```python
def add(a, b):
    return a + b

print(add(2, 3))
# Output: 5
```

<br>

## 29. Scope: local / global / nonlocal

Where a variable can be seen and used. (作用域)

```python
x = "global"

def change():
    global x
    x = "changed"

change()
print(x)
# Output: changed
```

<br>

## 30. Lambda Functions

A small anonymous function in one line. (匿名函数)

```python
square = lambda n: n * n
print(square(5))
# Output: 25
```

<br>

## 31. Higher-Order Functions: map, filter, reduce

Functions that take other functions as input. (高阶函数)

```python
from functools import reduce
print(list(map(lambda x: x*2, [1, 2, 3])))
print(list(filter(lambda x: x > 1, [1, 2, 3])))
print(reduce(lambda a, b: a + b, [1, 2, 3]))
# Output: [2, 4, 6]
# Output: [2, 3]
# Output: 6
```

<br>

## 32. Recursion

A function that calls itself to solve a problem. (递归)

```python
def fact(n):
    if n == 1:
        return 1
    return n * fact(n - 1)

print(fact(4))
# Output: 24
```

<br>

## 33. Decorators

A function that wraps another to add behavior. (装饰器)

```python
def shout(func):
    def wrapper():
        return func().upper()
    return wrapper

@shout
def hello():
    return "hi"

print(hello())
# Output: HI
```

<br>

## 34. Generators & yield

A function that produces values one at a time with `yield`. (生成器)

```python
def count_up(n):
    for i in range(n):
        yield i

for x in count_up(3):
    print(x)
# Output: 0
# Output: 1
# Output: 2
```

<br>

## 35. Iterators

An object you can loop through using `next()`. (迭代器)

```python
it = iter([10, 20])
print(next(it))
print(next(it))
# Output: 10
# Output: 20
```

<br> <br>

# 5. Modules & Packages

Organize and reuse code across files.

## 36. import & from-import

Bring code from another module into yours. (导入)

```python
import math
from math import pi

print(math.sqrt(16))
print(pi)
# Output: 4.0
# Output: 3.141592653589793
```

<br>

## 37. Creating Modules

Any `.py` file can be imported as a module. (创建模块)

```python
# file: mymod.py
#   def greet(): return "hi"
# main file:
#   import mymod
#   print(mymod.greet())   # -> would print: hi
print("A module is just a .py file")
# Output: A module is just a .py file
```

<br>

## 38. Packages & **init**.py

A folder of modules marked by `__init__.py`. (包)

```python
# folder structure:
#   mypkg/
#     __init__.py
#     tools.py
#   import mypkg.tools
print("A package is a folder of modules")
# Output: A package is a folder of modules
```

<br>

## 39. Standard Library Tour

Built-in modules ready to use. (标准库)

```python
import random, datetime
print(random.randint(1, 1))
print(datetime.date(2024, 1, 1))
# Output: 1
# Output: 2024-01-01
```

<br>

## 40. pip & Virtual Environment

Install packages and isolate project dependencies. (包管理与虚拟环境)

```python
# In terminal (命令行):
#   python -m venv venv      # create environment
#   pip install requests     # install a package
print("Use pip to install, venv to isolate")
# Output: Use pip to install, venv to isolate
```

<br> <br>

# 6. Object-Oriented Programming

Model real things as objects with data and behavior.

## 41. Class & Object

A class is a blueprint; an object is an instance of it. (类与对象)

```python
class Dog:
    pass

d = Dog()
print(type(d).__name__)
# Output: Dog
```

<br>

## 42. **init** & self

`__init__` sets up a new object; `self` is the object itself. (构造方法)

```python
class Dog:
    def __init__(self, name):
        self.name = name

d = Dog("Rex")
print(d.name)
# Output: Rex
```

<br>

## 43. Instance vs Class Attributes

Instance attributes differ per object; class attributes are shared. (实例与类属性)

```python
class Dog:
    species = "Canine"      # class attribute (shared)
    def __init__(self, name):
        self.name = name    # instance attribute

a, b = Dog("Rex"), Dog("Max")
print(a.species, b.species, a.name)
# Output: Canine Canine Rex
```

<br>

## 44. Methods: instance, class, static

Three method types bound to object, class, or nothing. (方法类型)

```python
class Calc:
    def inst(self): return "instance"
    @classmethod
    def cls_m(cls): return "class"
    @staticmethod
    def stat(): return "static"

c = Calc()
print(c.inst(), Calc.cls_m(), Calc.stat())
# Output: instance class static
```

<br>

## 45. Inheritance

A child class reuses a parent class's features. (继承)

```python
class Animal:
    def speak(self): return "sound"

class Dog(Animal):
    pass

print(Dog().speak())
# Output: sound
```

<br>

## 46. Method Overriding & super()

Replace a parent method, optionally calling it with `super()`. (方法重写)

```python
class Animal:
    def speak(self): return "sound"

class Dog(Animal):
    def speak(self): return "Woof + " + super().speak()

print(Dog().speak())
# Output: Woof + sound
```

<br>

## 47. Polymorphism

Different classes respond to the same method differently. (多态)

```python
class Cat:
    def speak(self): return "Meow"
class Dog:
    def speak(self): return "Woof"

for a in [Cat(), Dog()]:
    print(a.speak())
# Output: Meow
# Output: Woof
```

<br>

## 48. Encapsulation: private, property

Hide data and expose it safely with `@property`. (封装)

```python
class Account:
    def __init__(self):
        self._balance = 100      # _ means private (私有)
    @property
    def balance(self):
        return self._balance

print(Account().balance)
# Output: 100
```

<br>

## 49. Magic / Dunder Methods

Special `__x__` methods Python calls automatically. (魔术方法)

```python
class Book:
    def __init__(self, t): self.t = t
    def __str__(self): return f"Book: {self.t}"

print(Book("Python"))
# Output: Book: Python
```

<br>

## 50. Abstract Classes

A class that defines methods subclasses must implement. (抽象类)

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self): ...

class Square(Shape):
    def __init__(self, s): self.s = s
    def area(self): return self.s ** 2

print(Square(3).area())
# Output: 9
```

<br> <br> # 7. Error Handling

Catch and manage errors so programs don't crash.

## 51. try / except / else / finally

Try risky code, catch errors, and always clean up. (异常捕获)

```python
try:
    x = 10 / 2
except ZeroDivisionError:
    print("no divide by zero")
else:
    print("ok", x)
finally:
    print("done")
# Output: ok 5.0
# Output: done
```

<br>

## 52. Raising Exceptions: raise

Trigger an error on purpose with `raise`. (抛出异常)

```python
def check(age):
    if age < 0:
        raise ValueError("age cannot be negative")
    return age

try:
    check(-1)
except ValueError as e:
    print(e)
# Output: age cannot be negative
```

<br>

## 53. Custom Exceptions

Define your own error type by subclassing `Exception`. (自定义异常)

```python
class MyError(Exception):
    pass

try:
    raise MyError("oops")
except MyError as e:
    print("caught:", e)
# Output: caught: oops
```

<br>

## 54. Common Built-in Exceptions

Frequently seen errors like ValueError and KeyError. (常见异常)

```python
for action in ["value", "key", "index"]:
    try:
        if action == "value": int("abc")
        elif action == "key": {}["x"]
        elif action == "index": [][0]
    except Exception as e:
        print(type(e).__name__)
# Output: ValueError
# Output: KeyError
# Output: IndexError
```

<br> <br>

# 8. File & I/O

Read from and write to files on disk.

## 55. Opening Files: open(), with

`with open()` opens a file and closes it automatically. (打开文件)

```python
with open("demo.txt", "w") as f:
    f.write("hello")
print("file written")
# Output: file written
```

<br>

## 56. Reading & Writing

Write text out, then read it back in. (读写)

```python
with open("demo.txt", "w") as f:
    f.write("line1\nline2")

with open("demo.txt") as f:
    print(f.read())
# Output: line1
# Output: line2
```

<br>

## 57. File Modes

Modes control read, write, or append behavior. (文件模式)

```python
# "r" read | "w" write (overwrite) | "a" append | "x" create
with open("demo.txt", "a") as f:
    f.write("\nmore")
print("appended")
# Output: appended
```

<br>

## 58. Working with Paths: pathlib

Handle file paths in a clean, cross-platform way. (路径处理)

```python
from pathlib import Path
p = Path("demo.txt")
print(p.name, p.suffix, p.exists())
# Output: demo.txt .txt True
```

<br>

## 59. CSV & JSON Files

Read and write common data file formats. (文件格式)

```python
import json
data = {"name": "Tom", "age": 18}

with open("d.json", "w") as f:
    json.dump(data, f)

with open("d.json") as f:
    print(json.load(f))
# Output: {'name': 'Tom', 'age': 18}
```

<br> <br>

# 9. Advanced Topics

Powerful features for cleaner, smarter code.

## 60. Context Managers: with & **enter**

Objects that set up and tear down resources via `with`. (上下文管理器)

```python
class Timer:
    def __enter__(self): print("start"); return self
    def __exit__(self, *a): print("end")

with Timer():
    print("working")
# Output: start
# Output: working
# Output: end
```

<br>

## 61. Closures

An inner function that remembers outer variables. (闭包)

```python
def multiplier(n):
    def inner(x):
        return x * n
    return inner

double = multiplier(2)
print(double(5))
# Output: 10
```

<br>

## 62. Type Hints / Annotations

Optional hints showing expected types. (类型注解)

```python
def add(a: int, b: int) -> int:
    return a + b

print(add(2, 3))
# Output: 5
```

<br>

## 63. Regular Expressions

Patterns to search and match text. (正则表达式)

```python
import re
m = re.search(r"\d+", "abc123")
print(m.group())
# Output: 123
```

<br>

## 64. Comprehension vs Generator Expression

Brackets build a list; parentheses build a lazy generator. (推导式对比)

```python
lst = [x for x in range(3)]       # list (all in memory)
gen = (x for x in range(3))       # generator (lazy)
print(lst)
print(sum(gen))
# Output: [0, 1, 2]
# Output: 3
```

<br>

## 65. Unpacking & Packing

Spread or collect values with `*` and `**`. (解包与打包)

```python
a, *rest = [1, 2, 3, 4]
print(a, rest)
nums = [1, 2, 3]
print(*nums)
# Output: 1 [2, 3, 4]
# Output: 1 2 3
```

<br>

## 66. Enumerate, Zip Advanced

Loop with indexes or pair multiple sequences. (高级遍历)

```python
for i, v in enumerate(["a", "b"], start=1):
    print(i, v)
for x, y in zip([1, 2], [3, 4]):
    print(x + y)
# Output: 1 a
# Output: 2 b
# Output: 4
# Output: 6
```

<br>

## 67. Sorting with key

Sort by a custom rule using the `key` argument. (自定义排序)

```python
words = ["bbb", "a", "cc"]
print(sorted(words, key=len))
print(sorted([3, -1, 2], key=abs))
# Output: ['a', 'cc', 'bbb']
# Output: [-1, 2, 3]
```

<br> <br>

# 10. Modern & Specialized

Newer features and specialized tools.

## 68. f-string Advanced

Format numbers and expressions inside f-strings. (格式化进阶)

```python
pi = 3.14159
print(f"{pi:.2f}")
print(f"{1000000:,}")
print(f"{5 * 5 = }")
# Output: 3.14
# Output: 1,000,000
# Output: 5 * 5 = 25
```

<br>

## 69. Dataclasses

Auto-generate boilerplate for data-holding classes (3.7+). (数据类)

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int

print(Point(1, 2))
# Output: Point(x=1, y=2)
```

<br>

## 70. Enums

A set of named constant values. (枚举)

```python
from enum import Enum

class Color(Enum):
    RED = 1
    GREEN = 2

print(Color.RED, Color.RED.value)
# Output: Color.RED 1
```

<br>

## 71. Collections Module

Specialized containers like Counter and defaultdict. (集合工具)

```python
from collections import Counter, defaultdict

print(Counter("aabbbc"))
d = defaultdict(int)
d["x"] += 1
print(d)
# Output: Counter({'b': 3, 'a': 2, 'c': 1})
# Output: defaultdict(<class 'int'>, {'x': 1})
```

<br>

## 72. itertools & functools

Tools for efficient looping and function handling. (函数工具)

```python
from itertools import count, islice
from functools import lru_cache

print(list(islice(count(0, 2), 3)))   # 0, 2, 4

@lru_cache
def fib(n):
    return n if n < 2 else fib(n-1) + fib(n-2)
print(fib(10))
# Output: [0, 2, 4]
# Output: 55
```

<br>

## 73. Concurrency: threading, multiprocessing, asyncio

Run tasks at the same time for speed. (并发)

```python
import asyncio

async def hello():
    return "async hi"

print(asyncio.run(hello()))
# Output: async hi
```

<br>

## 74. Virtual Environment & Project Structure

Organize a project with isolated dependencies. (项目结构)

```python
# typical layout:
#   myproject/
#     venv/            <- isolated environment
#     src/             <- source code
#     tests/           <- test files
#     requirements.txt <- dependency list
print("Keep code, tests, and deps organized")
# Output: Keep code, tests, and deps organized
```

<br> <br> # 11. Core Language Gaps ★

Smaller but important language features worth knowing.

## 75. Walrus Operator := ★

Assign a value inside an expression in one step. (海象运算符)

```python
nums = [1, 2, 3, 4]
if (n := len(nums)) > 3:
    print(f"list has {n} items")
# Output: list has 4 items
```

<br>

## 76. Ternary Expression ★

A one-line if-else that returns a value. (三元表达式)

```python
age = 20
status = "adult" if age >= 18 else "minor"
print(status)
# Output: adult
```

<br>

## 77. Chained Comparison ★

Compare a value against two bounds at once. (链式比较)

```python
x = 5
print(1 < x < 10)
print(0 <= x <= 3)
# Output: True
# Output: False
```

<br>

## 78. for-else / while-else ★

The `else` runs only if the loop never hit `break`. (循环的 else 子句)

```python
for n in [1, 2, 3]:
    if n == 9:
        break
else:
    print("not found")
# Output: not found
```

<br>

## 79. Keyword-Only Arguments ★

Force arguments to be passed by name using `*`. (仅关键字参数)

```python
def make(name, *, color):
    return f"{name}-{color}"

print(make("car", color="red"))
# make("car", "red")  -> error, color must be keyword
# Output: car-red
```

<br>

## 80. Positional-Only Arguments ★

Force arguments to be passed by position using `/`. (仅位置参数)

```python
def power(base, exp, /):
    return base ** exp

print(power(2, 3))
# power(base=2, exp=3)  -> error, must be positional
# Output: 8
```

<br>

## 81. **slots** ★

Restrict object attributes to save memory. (限制属性)

```python
class Point:
    __slots__ = ("x", "y")
    def __init__(self, x, y):
        self.x, self.y = x, y

p = Point(1, 2)
print(p.x, p.y)
# p.z = 3  -> error, z not in __slots__
# Output: 1 2
```

<br>

## 82. Multiple Context Managers ★

Open several resources in one `with` line. (多个上下文管理器)

```python
with open("a.txt", "w") as a, open("b.txt", "w") as b:
    a.write("hi")
    b.write("yo")
print("both written")
# Output: both written
```

<br>

## 83. Dictionary Merge | ★

Combine two dicts with the `|` operator (3.9+). (字典合并)

```python
d1 = {"a": 1}
d2 = {"b": 2}
print(d1 | d2)
# Output: {'a': 1, 'b': 2}
```

<br>

## 84. Star Unpacking in Calls ★

Spread lists or dicts into function arguments. (调用时解包)

```python
def add(a, b, c):
    return a + b + c

args = [1, 2, 3]
kwargs = {"a": 1, "b": 2, "c": 3}
print(add(*args))
print(add(**kwargs))
# Output: 6
# Output: 6
```

<br>

## 85. Global vs Nonlocal ★

`global` reaches module scope; `nonlocal` reaches the enclosing function. (全局与外层变量)

```python
def outer():
    count = 0
    def inner():
        nonlocal count
        count += 1
    inner()
    return count

print(outer())
# Output: 1
```

<br>

## 86. Truthy & Falsy Values ★

Empty or zero-like values count as `False` in conditions. (真值与假值)

```python
for v in [0, "", [], None, 5, "hi"]:
    print(bool(v), end=" ")
print()
# Output: False False False False True True
```

<br> <br> # 12. More Core Features ★

Further useful built-ins and language details.

## 87. is vs == ★

`==` compares values; `is` compares identity (same object). (相等与同一)

```python
a = [1, 2]
b = [1, 2]
print(a == b)
print(a is b)
# Output: True
# Output: False
```

<br>

## 88. any() & all() ★

`any` is True if one item is truthy; `all` if every item is. (任一与全部)

```python
nums = [1, 2, 3]
print(any(n > 2 for n in nums))
print(all(n > 0 for n in nums))
# Output: True
# Output: True
```

<br>

## 89. String join() ★

Glue a list of strings together with a separator. (字符串连接)

```python
words = ["a", "b", "c"]
print("-".join(words))
# Output: a-b-c
```

<br>

## 90. zip() Unzipping ★

Use `zip(*pairs)` to split pairs back into two lists. (解压)

```python
pairs = [(1, "a"), (2, "b")]
nums, letters = zip(*pairs)
print(nums)
print(letters)
# Output: (1, 2)
# Output: ('a', 'b')
```

<br>

## 91. Dictionary get() & setdefault() ★

Read or insert a default value without errors. (安全取值)

```python
d = {"a": 1}
print(d.get("b", 0))
d.setdefault("b", 99)
print(d)
# Output: 0
# Output: {'a': 1, 'b': 99}
```

<br>

## 92. Negative & Step Slicing ★

Slice from the end or skip items with a step. (负索引与步长)

```python
nums = [0, 1, 2, 3, 4, 5]
print(nums[-2:])
print(nums[::2])
print(nums[::-1])
# Output: [4, 5]
# Output: [0, 2, 4]
# Output: [5, 4, 3, 2, 1, 0]
```

<br>

## 93. Default Arg Pitfall ★

Never use a mutable default like `[]`; it is shared across calls. (可变默认参数陷阱)

```python
def good(item, lst=None):
    if lst is None:
        lst = []
    lst.append(item)
    return lst

print(good(1))
print(good(2))
# Output: [1]
# Output: [2]
```

<br>

## 94. enumerate() with Index ★

Loop with a running index instead of a counter variable. (带索引遍历)

```python
for i, color in enumerate(["red", "green"]):
    print(i, color)
# Output: 0 red
# Output: 1 green
```

<br>

## 95. *_ Throwaway Variable ★

Use `_` to ignore values you don't need. (占位变量)

```python
first, *_, last = [1, 2, 3, 4]
print(first, last)
# Output: 1 4
```

<br>

## 96. round() & Number Formatting ★

Round numbers to a chosen number of decimals. (四舍五入)

```python
print(round(3.14159, 2))
print(round(2.5))
print(divmod(17, 5))
# Output: 3.14
# Output: 2
# Output: (3, 2)
```

<br>

## 97. sorted() reverse & Multi-key ★

Sort descending or by multiple keys at once. (多条件排序)

```python
people = [("Tom", 20), ("Sue", 20), ("Ann", 18)]
print(sorted(people, key=lambda p: (-p[1], p[0])))
# Output: [('Sue', 20), ('Tom', 20), ('Ann', 18)]
```

<br>

## 98. f-string Alignment ★

Pad and align text inside f-strings. (对齐填充)

```python
print(f"{'hi':>6}")
print(f"{'hi':<6}|")
print(f"{'hi':^6}")
# Output:     hi
# Output: hi    |
# Output:   hi
```

<br> <br>
