---
title: "closure"
published: 2026-05-02
description: "closure"
image: ""
tags: ["python","closure"]
category: python
draft: false
lang: ""
createdAt: "2026-05-02T14:01:22.443.156169816Z"
---

# Closure

A Closure (闭包) is an Inner Function (内部函数) that remembers variables from its Outer Function (外部函数, Enclosing Scope), even after the Outer Function has finished executing.

## 1. Definition

A Closure (闭包) is created when an Inner Function (内部函数) uses a variable from its Enclosing Scope (外层作用域) and is returned or used outside that scope.

```
def outer():
    name = "Tom"

    def inner():
        print(name)

    return inner


f = outer()
f()

# Output:
# Tom
```

<br>

## 2. Key Conditions

A Closure (闭包) usually requires an Outer Function (外部函数), an Inner Function (内部函数), and the Inner Function referencing a variable from the Outer Function.

```
def outer():
    message = "Hello"

    def inner():
        print(message)

    return inner


closure_func = outer()
closure_func()

# Output:
# Hello
```

<br>

## 3. Why It Works

A Closure (闭包) works because Python keeps the referenced variable alive in memory when the Inner Function (内部函数) still needs it.

```
def make_counter():
    count = 0

    def counter():
        nonlocal count
        count += 1
        print(count)

    return counter


c = make_counter()
c()
c()
c()

# Output:
# 1
# 2
# 3
```

<br>

## 4. Relationship with Decorator

A Decorator (装饰器) commonly uses a Closure (闭包) to remember the original Function (函数) and extend its behavior.

```
def log(func):
    def wrapper():
        print("Before function call")
        func()
        print("After function call")

    return wrapper


@log
def say_hello():
    print("Hello")


say_hello()

# Output:
# Before function call
# Hello
# After function call
```

<br><br>
