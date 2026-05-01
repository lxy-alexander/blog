---
title: "__getattr__"
published: 2026-04-30
description: "__getattr__"
image: ""
tags: ["python","__getattr__"]
category: python
draft: false
lang: ""
createdAt: "2026-04-30T21:16:11.936.840279628Z"
---

# `__getattr__`

`__getattr__` is a ==fallback method(兜底方法)== that is called only when an attribute (属性) cannot be found normally.

## 1. Class-level `__getattr__`

Class-level `__getattr__` is triggered when an object (对象) accesses a missing attribute.

```
class A:
    def __getattr__(self, name):
        return f"{name} not found"

a = A()
print(a.x)
```

<br>

## 2. Module-level `__getattr__`

Module-level `__getattr__` is triggered when a module (模块) accesses a missing attribute.

```
# a.py
def __getattr__(name):
    return f"{name} not found"
import a
print(a.x)
```

<br>

## 3. Relationship with Lazy Initialization

`__getattr__` is not initialization (初始化) itself; it only provides a place to run lazy initialization (懒加载初始化) when a missing attribute is first accessed.

```
_current_platform = None

def __getattr__(name):
    global _current_platform
    if name == "current_platform":
        if _current_platform is None:
            _current_platform = create_platform()
        return _current_platform
    raise AttributeError(name)
```

<br>

## 4. Difference from `__init__`

`__init__` runs when an object is created, while `__getattr__` runs when a missing attribute is accessed.

```
class A:
    def __init__(self):
        print("object created")

    def __getattr__(self, name):
        print("missing attribute")
```

<br>

## 5. Interview Summary

`__getattr__` is commonly used to implement dynamic attribute lookup (动态属性查找) or lazy loading (懒加载) for missing attributes.

<br> <br>
