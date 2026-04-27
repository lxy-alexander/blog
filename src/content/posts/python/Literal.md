---
title: "Literal"
published: 2026-04-27
description: "Literal"
image: ""
tags: ["python","Literal"]
category: python
draft: false
lang: ""
createdAt: "2026-04-27T15:07:01.580.819768228Z"
---

# Python `Literal` Type Hint

`Literal` is a type hint (类型提示) from the `typing` module that restricts a variable or argument to a fixed set of constant values (常量值), enabling stricter type checking (类型检查).

## 1. Basic Usage

`Literal` specifies that a value must be exactly one of the listed constants (常量), checked by static type checkers (静态类型检查器) like `mypy`.

```python
from typing import Literal

def set_mode(mode: Literal["read", "write", "append"]) -> None:
    print(f"Mode: {mode}")

set_mode("read")    # Mode: read
set_mode("write")   # Mode: write
# set_mode("delete")  # mypy error: not a valid literal
```

<br>

## 2. Multiple Types

`Literal` supports mixed types (混合类型) like `str`, `int`, `bool`, and `None`, useful for flag arguments (标志参数).

```python
from typing import Literal

def align(direction: Literal["left", "right", 0, 1]) -> None:
    print(direction)

align("left")   # left
align(1)        # 1
```

<br>

## 3. Return Type Annotation

`Literal` can also annotate return values (返回值), telling the caller exactly which constants the function may return.

```python
from typing import Literal

def get_status(code: int) -> Literal["ok", "error"]:
    return "ok" if code == 200 else "error"

print(get_status(200))  # ok
print(get_status(500))  # error
```

<br>

## 4. Combine with Union

`Literal` works with `Union` (联合类型) or `|` to mix literal values (字面量值) and regular types.

```python
from typing import Literal, Union

def parse(value: Union[Literal["auto"], int]) -> int:
    if value == "auto":
        return 0
    return value

print(parse("auto"))  # 0
print(parse(42))      # 42
```

<br>

## 5. Use Case Summary

`Literal` is mainly used for enumerated options (枚举选项), API mode flags, and improving IDE auto-completion (自动补全) and code safety (代码安全性).

| Scenario                             | Example                     |
| ------------------------------------ | --------------------------- |
| Mode selection (模式选择)            | `Literal["train", "eval"]`  |
| HTTP method (HTTP 方法)              | `Literal["GET", "POST"]`    |
| Boolean-like flag (布尔标志)         | `Literal[0, 1]`             |
| Return value constraint (返回值约束) | `-> Literal["ok", "error"]` |

<br> <br>
