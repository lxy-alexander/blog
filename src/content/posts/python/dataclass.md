---
title: "dataclass"
published: 2026-02-18
description: "dataclass"
image: ""
tags: ["python","dataclass"]
category: python
draft: false
lang: ""
---





`@dataclass` 会自动生成：

-   `__init__`
-   `__repr__`
-   `__eq__`
-   默认值处理
-   类型注解绑定

**重点：字段 = 构造参数**

```
enable_log_requests: bool = False
```

→ 自动变成：

```
def __init__(..., enable_log_requests=False):
```

------

# 为什么还能接收父类参数

因为：

```
class AsyncEngineArgs(EngineArgs)
```

且 **父类 EngineArgs 也是 dataclass**

👉 dataclass 支持 **继承字段拼接**

最终真实构造函数是：

```
def __init__(
    self,
    # 父类所有参数
    model=...,
    enforce_eager=...,
    ...
    # 子类字段
    enable_log_requests=False
)
```
