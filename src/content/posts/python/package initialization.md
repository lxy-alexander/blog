---
title: "package initialization"
published: 2026-04-30
description: "package initialization"
image: ""
tags: ["python","package initialization"]
category: python
draft: false
lang: ""
createdAt: "2026-04-30T04:36:33.393.380472831Z"
---

# Python Package Initialization (包初始化机制)

`__init__.py` is a special file that marks a directory as a Package（包） and controls its initialization behavior.



## 1. Core Concept

`__init__.py` is executed when a package is imported and can define package-level variables and logic.

### 1) Example

```
# folder structure:
# mypkg/
#   __init__.py
#   utils.py

# utils.py
def greet():
    return "Hello"

# __init__.py
from .utils import greet

__all__ = ["greet"]  # It controls how modules are imported using __all__.

# main.py
from mypkg import greet

print(greet())  # Output: Hello
```

<br>

