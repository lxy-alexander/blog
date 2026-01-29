---
title: "Python Imports & Packaging"
published: 2026-01-28
description: "Python Imports & Packaging"
image: ""
tags: ["python","Python Imports & Packaging"]
category: python
draft: false
lang: ""
---





# **Python module & package**

## 1) Module (模块)

==A **module** is simply **one Python file** (`.py`) that contains code (functions, classes, variables).==

### Example

File: `math_utils.py`

```python
def add(a, b):
    return a + b
```

Import it:

```python
import math_utils
print(math_utils.add(1, 2))
```

✅ A module = **one `.py` file**.

------

## 2) Package (包)

==A **package** is a **directory** that contains Python modules. Traditionally, a package contains an `__init__.py` file.==

```
mypkg/
  __init__.py
  a.py
  b.py
```

Import:

```python
import mypkg.a
from mypkg.b import foo
```

A package = **a directory of modules** (often with `__init__.py`).

What does `__init__.py` do?

-   It tells Python: “this folder is a package”
-   It can also run initialization code when the package is imported

Example:

```python
# mypkg/__init__.py
print("mypkg imported!")
```

------



# **Import system (`import`, `sys.path`)**

### What is the import system?

==Python’s **import system** is the mechanism that **finds and loads modules/packages** when you write==

```python
import something
from something import x
```

------

## 1) `import` does 3 main things

When you run:

```python
import mypkg
```

Python will:

### (1) Check cache first

It checks whether it’s already imported in:

```python
sys.modules
```

If it exists, Python reuses it (no re-loading).

------

### ✅ (2) Search where to find it (`sys.path`)

If not cached, Python searches directories in this order:

```python
import sys
print(sys.path)
```

Typical `sys.path` includes:

-   `''` (current working directory)
-   Python standard library paths
-   `site-packages` (pip-installed libraries)

Python searches from top → bottom until it finds the module/package.

------

### ✅ (3) Load and execute it

Once found:

-   Python loads the file/folder
-   Executes top-level code inside it
-   Stores it in `sys.modules`

------

## 2) What does `sys.path` mean?

`sys.path` is simply a **list of directories** Python uses to locate imports.

Example:

```python
import sys
print(sys.path[0])
```

-   If `sys.path` contains `/usr/local/lib/python3.14/site-packages`
    → Python can import packages installed by `pip`.

------

## ✅ One-sentence summary

**Python’s import system loads modules by checking `sys.modules` first, then searching paths in `sys.path`, and finally executing the module code.**







1.  **Namespace package**
2.  **Normal package vs namespace package**
3.  **`__init__.py` meaning**
4.  **PEP 420 (Implicit Namespace Packages)**
5.  **PYTHONPATH / environment variables**
6.  **`site-packages` / pip-installed distributions**
