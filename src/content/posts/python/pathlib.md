---
title: "pathlib"
published: 2026-04-27
description: "pathlib"
image: ""
tags: ["python","pathlib"]
category: python
draft: false
lang: ""
createdAt: "2026-04-27T15:08:06.269.852710339Z"
---

# Python `pathlib` vs `os`

`pathlib` is the modern object-oriented (面向对象) path library, while `os` / `os.path` is the older procedural (过程式) string-based API; `pathlib` is now the recommended (推荐) choice.

## 1. Core Difference

==`os.path` treats paths as strings (字符串), while `pathlib.Path` treats paths as objects (对象)== with methods (方法) and operators (运算符).

```python
import os
from pathlib import Path

# os style: string concatenation
path1 = os.path.join("data", "images", "cat.png")
print(path1)  # data/images/cat.png

# pathlib style: object with / operator
path2 = Path("data") / "images" / "cat.png"
print(path2)  # data/images/cat.png
```

<br>

## 2. Path Construction

`pathlib` uses the `/` operator (运算符) to join paths, which is more readable (可读) than `os.path.join()`.

```python
from pathlib import Path
import os

p = Path("/home") / "user" / "file.txt"
print(p)                              # /home/user/file.txt

q = os.path.join("/home", "user", "file.txt")
print(q)                              # /home/user/file.txt
```

<br>

## 3. Path Components

`pathlib` exposes path parts (路径组成部分) as attributes (属性), while `os.path` requires separate functions (函数).

```python
from pathlib import Path
import os

p = Path("/home/user/file.txt")
print(p.name)     # file.txt        (filename)
print(p.stem)     # file            (name without suffix)
print(p.suffix)   # .txt            (extension 扩展名)
print(p.parent)   # /home/user      (parent dir 父目录)

# os equivalent
print(os.path.basename("/home/user/file.txt"))  # file.txt
print(os.path.splitext("file.txt"))             # ('file', '.txt')
print(os.path.dirname("/home/user/file.txt"))   # /home/user
```

<br>

## 4. Existence and Type Check

Both libraries can check if a path exists (存在) or is a file/directory, but `pathlib` uses methods (方法) on the object.

```python
from pathlib import Path
import os

p = Path(".")
print(p.exists())     # True
print(p.is_dir())     # True
print(p.is_file())    # False

# os equivalent
print(os.path.exists("."))   # True
print(os.path.isdir("."))    # True
print(os.path.isfile("."))   # False
```

<br>

## 5. Read and Write Files

`pathlib` provides shortcut methods (快捷方法) like `read_text()` and `write_text()`, avoiding the need for explicit `open()`.

```python
from pathlib import Path

p = Path("hello.txt")
p.write_text("Hello, Pathlib!")        # write
content = p.read_text()                # read
print(content)                         # Hello, Pathlib!
p.unlink()                             # delete file
```

<br>

## 6. Iterate Directory

`Path.iterdir()` and `Path.glob()` replace `os.listdir()` and `glob.glob()`, returning `Path` objects directly.

```python
from pathlib import Path

# list current directory
for item in Path(".").iterdir():
    print(item.name)

# glob pattern: find all .py files recursively (递归)
for py in Path(".").rglob("*.py"):
    print(py)
```

<br>

## 7. Create and Delete

`pathlib` supports creating directories (创建目录) and deleting files in one call, with safer flags (标志).

```python
from pathlib import Path

d = Path("tmp_dir")
d.mkdir(exist_ok=True)        # create dir, no error if exists
(d / "a.txt").write_text("hi")
print(list(d.iterdir()))      # [PosixPath('tmp_dir/a.txt')]

(d / "a.txt").unlink()        # delete file
d.rmdir()                     # delete empty directory
```

<br>

## 8. Absolute and Relative Paths

`resolve()` returns the absolute path (绝对路径), and `relative_to()` returns the relative path (相对路径) from a base.

```python
from pathlib import Path

p = Path("file.txt")
print(p.resolve())                          # /current/dir/file.txt
print(Path("/a/b/c.txt").relative_to("/a")) # b/c.txt
```

<br>

## 9. Comparison Summary

| Task                     | `os` / `os.path`         | `pathlib`               |
| ------------------------ | ------------------------ | ----------------------- |
| Join paths (拼接路径)    | `os.path.join(a, b)`     | `Path(a) / b`           |
| Get filename (文件名)    | `os.path.basename(p)`    | `p.name`                |
| Get extension (扩展名)   | `os.path.splitext(p)[1]` | `p.suffix`              |
| Check exists (存在)      | `os.path.exists(p)`      | `p.exists()`            |
| List dir (列目录)        | `os.listdir(p)`          | `p.iterdir()`           |
| Glob match (通配符)      | `glob.glob(...)`         | `p.glob(...)`           |
| Make dir (建目录)        | `os.makedirs(p)`         | `p.mkdir(parents=True)` |
| Read file (读文件)       | `open(p).read()`         | `p.read_text()`         |
| Absolute path (绝对路径) | `os.path.abspath(p)`     | `p.resolve()`           |

<br>

## 10. When to Use Which

Use `pathlib` for new code (新代码) due to cleaner syntax (语法), and use `os` only when working with legacy APIs (遗留接口) or environment variables (环境变量) like `os.environ`.

```python
import os
from pathlib import Path

# pathlib: file/path operations
project = Path(__file__).parent if "__file__" in dir() else Path.cwd()

# os: still preferred for environment / process operations
home = os.environ.get("HOME", "/tmp")
print(home)  # /home/user (or similar)
```

<br> <br>
