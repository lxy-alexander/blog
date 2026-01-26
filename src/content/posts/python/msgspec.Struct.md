---
title: "msgspec.Struct"
published: 2026-01-26
description: "msgspec.Struct"
image: ""
tags: ["python","msgspec.Struct"]
category: python
draft: false
lang: ""
---

>   `msgspec.Struct` 是一个 *schema-first（以 schema 为核心）*、*强类型*、*极高性能* 的 Python 数据结构，用来把结构化数据直接映射为对象，并高效地进行序列化 / 反序列化。
>
>   `msgspec.Struct` is a schema-first, strongly typed, high-performance data structure in Python for directly mapping structured data to objects with efficient serialization and deserialization.



## Mini Code

```python
pip install msgspec

import msgspec
from msgspec.json import decode

class User(msgspec.Struct):
    id: int
    name: str

user = decode(b'{"id": 1, "name": "Alice"}', type=User)

```

-   JSON **直接按 schema 解码**
-   不创建中间 `dict`
-   字段顺序、类型在 C 层校验
-   构造一个紧凑的 Struct 实例

`msgspec.Struct` = “把 JSON / MessagePack 这样的动态数据，压进一个静态、可预测、可优化的数据结构里。”



## when to use msgspec.Struct

| Question                | Yes →            | No →               |
| ----------------------- | ---------------- | ------------------ |
| Schema stable?          | `msgspec.Struct` | `dict` / dataclass |
| Performance critical?   | `msgspec.Struct` | anything           |
| Need coercion?          | pydantic         | msgspec            |
| User-facing validation? | pydantic         | msgspec            |
| Data-only object?       | msgspec          | class              |

Use `msgspec.Struct` at system boundaries with stable schemas and performance pressure; avoid it where flexibility or user-friendly validation is the priority.



## omit_defaults=True

在 **序列化（encode / to_dict）时**，如果字段的值 **等于默认值**，就**省略该字段**。

```
class User(msgspec.Struct, omit_defaults=True):
    id: int
    active: bool = True
User(id=1)  →  {"id": 1}
User(id=1, active=False) → {"id": 1, "active": false}
```

**减少 payload 大小**

-   特别是 JSON / MessagePack
-   网络 / 存储 / cache 更友好

**符合“协议默认值”语义**

-   很多协议：*missing = default*



## dict=True

**让 `Struct`：**

-   生成 `__dict__`
-   像普通 Python 对象一样存属性
-   支持 `@cached_property`

默认的 `msgspec.Struct`：❌ 没有 __dict__ ❌ 不支持 cached_property







