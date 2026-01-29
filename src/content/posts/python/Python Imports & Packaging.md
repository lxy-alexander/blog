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



# import system / module import mechanism Python 的模块导入机制

==Python 导入模块时先查 `sys.modules` 缓存，没有才按 `sys.path` 查找并执行加载，然后把结果写回 `sys.modules`。Python imports by first checking `sys.modules` (cache), and if not found, it searches `sys.path`, loads and executes the module, then stores it back into `sys.modules`.==

What is the import system?

==Python’s **import system** is the mechanism that **finds and loads modules/packages** when you write==

```python
import something
from something import x
```



When you run:

```python
import mypkg
```

Python will:

## 1) Check cache first

It checks whether it’s already imported in:

```python
sys.modules
```

If it exists, Python reuses it (no re-loading).

------

## 2) Search where to find it (`sys.path`)

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

## 3) Load and execute it

Once found:

-   Python loads the file/folder
-   Executes top-level code inside it
-   Stores it in `sys.modules`

### What does `sys.path` mean?

`sys.path` is simply a **list of directories** Python uses to locate imports.

Example:

```python
import sys
print(sys.path[0])
```

-   If `sys.path` contains `/usr/local/lib/python3.14/site-packages`
    → Python can import packages installed by `pip`.





# **Namespace package**

✅ **有 `__init__.py`**：这个目录是“传统包”（regular package）
✅ **没有 `__init__.py`**：这个目录可能是“普通文件夹”，也可能是“命名空间包”（namespace package）



## 最小可运行学习用例

下面给你一个 **Namespace package（命名空间包）最小可运行学习用例** 
特点：**没有 `__init__.py` 也能 import，并且同一个包名可以分散在多个目录里。**

```
project/
  part1/
    myns/
      a.py
  part2/
    myns/
      b.py
  main.py
```

注意：`myns/` 目录下 **都没有 `__init__.py`** 

`part1/myns/a.py`

```python
def fa():
    return "from a.py"
```

`part2/myns/b.py`

```python
def fb():
    return "from b.py"
```

`main.py`

```python
import sys
from pathlib import Path

# 获取 main.py 所在目录（绝对路径）
BASE = Path(__file__).resolve().parent

# 把 part1、part2 加入 sys.path（必须是绝对路径，最稳）
sys.path.insert(0, str(BASE / "part1"))
sys.path.insert(0, str(BASE / "part2"))

from myns.a import fa
from myns.b import fb

print(fa())
print(fb())
```

运行结果

```
from a.py
from b.py
```

这个例子说明了什么？

1）`myns` 没有 `__init__.py`，依然可以 import

因为 Python 3.3+ 支持 **namespace package**。

2）同一个包 `myns` 可以由多个目录共同组成

-   `part1/myns` 提供 `a.py`
-   `part2/myns` 提供 `b.py`

Python 会把它们“合并”成一个逻辑上的包。





# **Normal package vs namespace package**

------

## 1) 定义区别（最核心）

### Normal package（普通包）

✅ **必须有 `__init__.py`**（传统意义的包）

目录例子：

```
pkg/
  __init__.py
  a.py
```

------

### Namespace package（命名空间包）

✅ **没有 `__init__.py` 也能成为包**（Python 3.3+）

目录例子：

```
pkg/
  a.py
```

并且它的最大特点是：

✅ 同一个包名 `pkg` 可以分散在多个目录里合并成一个“逻辑包”。

------

## 2) import 行为对比

### 普通包 import

```python
import pkg
```

-   Python 找到 `pkg/__init__.py` → 执行它
-   `pkg` 被创建成模块对象

✅ 会执行初始化代码

------

### 命名空间包 import

```python
import pkg
```

-   Python 找到多个 `pkg/` 目录（来自不同路径）
-   把它们拼起来当成同一个包（不执行 `__init__.py`，因为没有）

✅ **不会执行任何初始化代码**

------

## 3) 能不能写初始化逻辑？

普通包 ✅可以

`pkg/__init__.py` 里可以写：

```python
print("pkg imported")
```

### 命名空间包 ❌不行

因为根本没有 `__init__.py`
所以你没地方放“包初始化逻辑”。



## 4) 能不能暴露统一 API？

普通包 ✅可以

你可以在 `__init__.py` 写：

```python
from .a import foo
```

外部直接：

```python
from pkg import foo
```



命名空间包 ❌很难/做不到

因为没有 `__init__.py`，不能把内部内容 re-export 到顶层。

用户只能：

```python
from pkg.a import foo
```

------

## 5) `__all__` 控制 `from pkg import *`

普通包 ✅可以

`__init__.py` 写：

```python
__all__ = ["foo"]
```



命名空间包 ❌不行

没 `__init__.py`，所以没地方写 `__all__`

------

## 6) 是否允许 “一个包分布在多个路径里”？

普通包 ⚠️不适合/容易冲突

普通包通常只能来自一个确定路径。
如果多个路径都有同名 `pkg`，可能出现覆盖/冲突，难排查。



命名空间包 ✅专门干这个

这是 namespace package 的核心用途：

例如：

-   `site-packages/pkg/`（第三方提供一部分）
-   你自己项目里也有 `pkg/`（再补一部分）

Python 会合并它们。

------

## 总结区别

-   **普通包**：有 `__init__.py`，可初始化、可导出 API，行为可控
-   **命名空间包**：没 `__init__.py`，可以跨目录拼接，更适合大型拆分包

✅ **用普通包**（99% 项目都该用）

-   业务代码
-   库开发
-   需要清晰结构、初始化逻辑

✅ **用命名空间包**

-   超大型项目拆分（一个包分成多个发行包）
-   插件式扩展体系
-   一个“包名”由多个模块来源组合

------



# **`__init__.py` meaning**

`__init__.py` means **“initialize”**.

In Python, `__init__.py` is the file that **initializes a package**—it marks a directory as a package and runs when the package is imported.





# **PYTHONPATH / environment variables**

`PYTHONPATH` is an environment variable that tells Python:↳

>   “Also search these directories for modules/packages.”↳

It becomes part of `sys.path`.

## 用 PYTHONPATH 让 Python 找到你的包

假设你有：

```
/Users/alexanderlee/mycode/mylib/hello.py
```

`hello.py`:

```
def hi():
    return "hello"
```

你写一个脚本：

```
test.py
from mylib.hello import hi
print(hi())
```

如果 Python 默认找不到 `mylib`

你可以这样运行：

```
PYTHONPATH=/Users/alexanderlee/mycode python3 test.py
```

✅ 这样 Python 就能 import 到 `mylib`

# **`site-packages` / pip-installed distributions**

When you run `pip install xxx`, the package is installed into:

当您运行 pip install xxx 时，该软件包将安装到：

>   ```
>   site-packages
>   ```

This directory is automatically on `sys.path`, so Python can import it directly.
