---
title: "typing feature - typedDict"
published: 2026-01-26
description: "typing feature - typedDict"
image: ""
tags: ["python","typing feature - typedDict"]
category: python
draft: false
lang: ""
---

来自 `typing`（Python 3.11+）或 `typing_extensions`（旧版本），作用是：

>   在 `TypedDict` 里把某个字段标记为 **可选（不是必须提供）**

------

## 示例

```python
from typing import TypedDict, NotRequired

class User(TypedDict):
    name: str                 # 必填
    age: NotRequired[int]     # 可选
```

这样写就允许：

```python
u1: User = {"name": "Tom"}              # ✅ ok
u2: User = {"name": "Tom", "age": 18}   # ✅ ok
```



如果不用 NotRequired（默认都是必填）

```python
class User(TypedDict):
    name: str
    age: int
```

那么：

```python
{"name": "Tom"}   # ❌ 类型检查会报缺少 age
```

