---
title: "decorator"
published: 2026-05-02
description: "decorator"
image: ""
tags: ["python","decorator"]
category: python
draft: false
lang: ""
createdAt: "2026-05-02T14:23:03.764.323016108Z"
---

# Decorator

==A Decorator (装饰器) is a  Function wrapper that takes a Function (函数), adds extra behavior, and returns a new Function.== 

## 1. Definition

A Decorator (装饰器) is used to extend a Function (函数) without modifying its original source code.

```
def decorator(func):
    def wrapper():
        print("Before function call")
        func()
        print("After function call")

    return wrapper


def say_hello():
    print("Hello")


say_hello = decorator(say_hello)
say_hello()

# Output:
# Before function call
# Hello
# After function call
```

<br>

## 2. `@` Syntax

==The `@` Symbol (语法糖) is shorthand(简洁的表达方式) for assigning the decorated function  to the original function.==

The `@decorator` syntax is equivalent to `function = decorator(function)`.

```
def decorator(func):
    def wrapper():
        print("Before function call")
        func()
        print("After function call")

    return wrapper


@decorator
def say_hello():
    print("Hello")


say_hello()

# Output:
# Before function call
# Hello
# After function call
```

<br>

## 3. Decorator with Arguments

A Decorator (装饰器) should use `*args` and `**kwargs` to support any Function Parameters (函数参数).

```
def log(func):
    def wrapper(*args, **kwargs):
        print(f"Calling function: {func.__name__}")
        result = func(*args, **kwargs)
        return result

    return wrapper


@log
def add(a, b):
    return a + b


print(add(3, 5))

# Output:
# Calling function: add
# 8
```

<br>

## 4. Decorator with Return Value

A Decorator (装饰器) must return the original function result if the decorated Function (函数) has a Return Value (返回值).

```
def decorator(func):
    def wrapper(*args, **kwargs):
        print("Before calculation")
        result = func(*args, **kwargs)
        print("After calculation")
        return result

    return wrapper


@decorator
def multiply(a, b):
    return a * b


print(multiply(4, 5))

# Output:
# Before calculation
# After calculation
# 20
```

<br>

## 5. Using `functools.wraps`

`functools.wraps` (函数元信息保留工具) preserves the original Function Metadata (函数元信息), such as `__name__` and `__doc__`.

```
from functools import wraps


def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@decorator
def greet():
    """This function says hello."""
    print("Hello")


print(greet.__name__)
print(greet.__doc__)

# Output:
# greet
# This function says hello.
```

<br>

## 6. Decorator with Parameters

A Parameterized Decorator (带参数装饰器) adds one more Outer Function (外层函数) to receive decorator configuration.

```
from functools import wraps


def log(level):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"[{level}] Calling {func.__name__}")
            return func(*args, **kwargs)

        return wrapper

    return decorator


@log("INFO")
def add(a, b):
    return a + b


print(add(2, 3))

# Output:
# [INFO] Calling add
# 5
```

<br>

## 7. Relationship with Closure

A Decorator (装饰器) uses a Closure (闭包) because the Wrapper Function (包装函数) remembers the original Function (原函数).

```
def decorator(func):
    def wrapper():
        print("Wrapper remembers the original function")
        func()

    return wrapper


@decorator
def hello():
    print("Hello")


hello()

# Output:
# Wrapper remembers the original function
# Hello
```

<br>

## 8. Common Use Cases

A Decorator (装饰器) is commonly used for Logging (日志记录), Authentication (身份验证), Timing (计时), Caching (缓存), and Permission Checking (权限校验).

```
import time
from functools import wraps


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Runtime: {end - start:.2f} seconds")
        return result

    return wrapper


@timer
def slow_task():
    time.sleep(1)
    print("Task finished")


slow_task()

# Output:
# Task finished
# Runtime: 1.00 seconds
```

<br> <br>
