---
title: "os"
published: 2026-05-02
description: "os"
image: ""
tags: ["python","os"]
category: python
draft: false
lang: ""
createdAt: "2026-05-02T17:23:27.925.407728635Z"
---

# Python `os` module

The `os` module (操作系统模块) provides a portable way to interact with the operating system (操作系统), such as paths, environment variables, directories, and CPU information.

## 1. `os.path.join`

`os.path.join` (路径拼接) safely combines path parts using the correct separator for the current operating system (操作系统).

```
import os

path = os.path.join("vllm", "libs", "example.so")

print(path)
# Output: vllm/libs/example.so
```

<br>

## 2. `os.getenv`

`os.getenv` (获取环境变量) reads an environment variable (环境变量) and returns a default value if it does not exist.

```
import os

value = os.getenv("VLLM_DISABLE_SCCACHE", "0")

print(value)
# Output: 0
```

<br>

## 3. `os.environ.get`

`os.environ.get` (环境变量字典读取) reads an environment variable (环境变量) from `os.environ`, which behaves like a dictionary (字典).

```
import os

cmake_args = os.environ.get("CMAKE_ARGS", "")

print(cmake_args)
# Output:
```

<br>

## 4. `os.makedirs`

`os.makedirs` (递归创建目录) creates a directory and all missing parent directories (父目录).

```
import os

os.makedirs("build/vllm/libs", exist_ok=True)

print(os.path.exists("build/vllm/libs"))
# Output: True
```

<br>

## 5. `os.path.exists`

`os.path.exists` (路径存在检查) checks whether a file path or directory path exists.

```
import os

os.makedirs("temp_dir", exist_ok=True)

print(os.path.exists("temp_dir"))
# Output: True
```

<br>

## 6. `os.path.abspath`

`os.path.abspath` (绝对路径) converts a relative path (相对路径) into an absolute path (绝对路径).

```
import os

path = os.path.abspath("setup.py")

print(path.endswith("setup.py"))
# Output: True
```

<br>

## 7. `os.path.dirname`

`os.path.dirname` (目录名获取) returns the parent directory (父目录) of a path.

```
import os

path = "build/vllm/libs/example.so"

print(os.path.dirname(path))
# Output: build/vllm/libs
```

<br>

## 8. `os.path.basename`

`os.path.basename` (文件名获取) returns the final file or directory name from a path.

```
import os

path = "build/vllm/libs/example.so"

print(os.path.basename(path))
# Output: example.so
```

<br>

## 9. `os.path.isdir`

`os.path.isdir` (目录检查) checks whether a path exists and is a directory (目录).

```
import os

os.makedirs("my_folder", exist_ok=True)

print(os.path.isdir("my_folder"))
# Output: True
```

<br>

## 10. `os.path.isfile`

`os.path.isfile` (文件检查) checks whether a path exists and is a regular file (普通文件).

```
import os

with open("demo.txt", "w") as file:
    file.write("hello")

print(os.path.isfile("demo.txt"))
# Output: True
```

<br>

## 11. `os.cpu_count`

`os.cpu_count` (CPU 数量) returns the number of logical CPUs (逻辑 CPU) available on the machine.

```
import os

count = os.cpu_count()

print(isinstance(count, int))
# Output: True
```

<br>

## 12. `os.sched_getaffinity`

`os.sched_getaffinity` (CPU 亲和性) returns the set of CPUs that the current process (当前进程) is allowed to run on.

```
import os

if hasattr(os, "sched_getaffinity"):
    cpus = os.sched_getaffinity(0)
    print(isinstance(cpus, set))
else:
    print("Not supported")

# Output: True
```

<br>

## 13. Common Build Script Pattern

A build script (构建脚本) often uses `os` to read configuration from environment variables (环境变量), create output directories (输出目录), and generate portable paths (可移植路径).

```
import os

root_dir = os.getcwd()
build_lib = os.path.join(root_dir, "build")
bundle_dir = os.path.join(build_lib, "vllm", "libs")

os.makedirs(bundle_dir, exist_ok=True)

disable_cache = int(os.getenv("VLLM_DISABLE_SCCACHE", "0"))
bundle_path = os.path.join(bundle_dir, "libexample.so")

print(os.path.isdir(bundle_dir))
# Output: True

print(disable_cache)
# Output: 0

print(bundle_path.endswith("build/vllm/libs/libexample.so"))
# Output: True
```

<br><br>
