---
title: "python msgspec.struct"
published: 2026-03-08
description: "python msgspec.struct"
image: ""
tags: ["python","python msgspec.struct"]
category: python
draft: false
lang: ""
---



# **I. `msgspec.Struct` — High-Performance Typed Data Structures**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <strong>Overview:</strong> <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">msgspec.Struct</code> is a <strong>high-performance (高性能)</strong>, <strong>type-safe (类型安全)</strong> data class alternative from the <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">msgspec</code> library. It is designed as a faster, leaner replacement for Python <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">dataclasses</code>, <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">attrs</code>, and Pydantic models — with native support for <strong>JSON / MessagePack serialization (序列化)</strong> and <strong>validation (验证)</strong> baked in at the C level. </div>

------

## 1. Installation & Import

```bash
pip install msgspec
import msgspec
from msgspec import Struct, field
```

------

## 2. Defining a Struct (定义结构体)

### 1) Basic Definition

```python
from msgspec import Struct

class Point(Struct):
    x: float
    y: float

p = Point(x=1.0, y=2.0)
print(p)        # Point(x=1.0, y=2.0)
print(p.x)      # 1.0
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Unlike Python <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">dataclasses</code>, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Struct</code> instances are <strong>immutable by default (默认不可变)</strong> and implemented in C — construction and attribute access are significantly faster.</div>

------

### 2) Fields with Default Values (默认值)

```python
class User(Struct):
    name: str
    age: int = 0
    active: bool = True

u = User(name="Alice")
print(u)   # User(name='Alice', age=0, active=True)
```

------

### 3) `field()` — Advanced Field Configuration

```python
from msgspec import Struct, field

class Config(Struct):
    tags: list[str] = field(default_factory=list)   # Mutable default
    name: str = field(default="unnamed")
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">Never use a mutable object (list, dict) directly as a default value.</span> Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">field(default_factory=list)</code> instead — just like with Python dataclasses.</div>

------

### 4) Nested Structs (嵌套结构体)

```python
class Address(Struct):
    city: str
    country: str

class Person(Struct):
    name: str
    address: Address

p = Person(name="Bob", address=Address(city="NYC", country="US"))
print(p.address.city)   # NYC
```

------

## 3. Struct Configuration Options (结构体配置)

Pass options to the class definition via keyword arguments:

```python
class MyStruct(Struct, frozen=True, order=True, eq=True, kw_only=True):
    x: int
    y: int
```

### 1) Option Reference Table

| Option                                                       | Default  | Effect                                                       |
| ------------------------------------------------------------ | -------- | ------------------------------------------------------------ |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">frozen=True</code> | `False`  | Makes the struct <strong>immutable (不可变)</strong> — fields cannot be reassigned after creation |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">order=True</code> | `False`  | Enables `<`, `>`, `<=`, `>=` comparison operators            |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">eq=True</code> | `True`   | Enables `==` / `!=` based on field values                    |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">kw_only=True</code> | `False`  | All fields must be passed as keyword arguments               |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">array_like=True</code> | `False`  | Serializes as a JSON array `[...]` instead of object `{...}` |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">gc=False</code> | `True`   | Disables garbage collector tracking — faster for structs with no reference cycles |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">weakref=True</code> | `False`  | Enables weak references to the struct instance               |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">rename</code> | `None`   | Rename fields during (de)serialization — e.g., `rename="camel"` |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">tag</code> | `None`   | Adds a type tag for <strong>tagged unions (标签联合)</strong> |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">tag_field</code> | `"type"` | The field name used to store the tag value                   |

------

### 2) `frozen=True` — Immutable Struct

```python
class ImmutablePoint(Struct, frozen=True):
    x: float
    y: float

p = ImmutablePoint(x=1.0, y=2.0)
p.x = 99.0   # ❌ TypeError: immutable type
```

**Scenario:** Configuration objects, cache keys, value objects (值对象) that should never change.

------

### 3) `order=True` — Sortable Structs

```python
class Version(Struct, order=True):
    major: int
    minor: int
    patch: int

versions = [Version(1, 2, 0), Version(1, 0, 5), Version(2, 0, 0)]
print(sorted(versions))
# [Version(1,0,5), Version(1,2,0), Version(2,0,0)]
```

**Scenario:** Sorting records, priority queues, range comparisons.

------

### 4) `rename="camel"` — Field Name Mapping

```python
class ApiResponse(Struct, rename="camel"):
    user_name: str
    created_at: str

import msgspec
obj = ApiResponse(user_name="Alice", created_at="2025-01-01")
print(msgspec.json.encode(obj))
# b'{"userName":"Alice","createdAt":"2025-01-01"}'
```

| `rename` value                                               | Effect                     |
| ------------------------------------------------------------ | -------------------------- |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">"camel"</code> | `user_name` → `userName`   |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">"pascal"</code> | `user_name` → `UserName`   |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">"lower"</code> | `userName` → `username`    |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">dict</code> | Explicit per-field mapping |

**Scenario:** Interoperating with REST APIs that use camelCase JSON keys.

------

## 4. Serialization & Deserialization (序列化与反序列化)

### 1) JSON Encoding

```python
import msgspec

class Order(Struct):
    id: int
    item: str
    price: float

order = Order(id=1, item="book", price=9.99)

# Encode to JSON bytes
data = msgspec.json.encode(order)
print(data)   # b'{"id":1,"item":"book","price":9.99}'
```

------

### 2) JSON Decoding with Type Validation

```python
# Decode + validate in one step
order2 = msgspec.json.decode(data, type=Order)
print(order2)          # Order(id=1, item='book', price=9.99)
print(order2 == order) # True
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">msgspec.json.decode()</code> performs <strong>schema validation (模式验证)</strong> at decode time. If the JSON does not match the expected type, it raises <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">msgspec.ValidationError</code> with a descriptive message.</div>

------

### 3) MessagePack Encoding (二进制序列化)

```python
# Encode to binary MessagePack
binary = msgspec.msgpack.encode(order)
print(binary)   # b'\x83\xa2id\x01\xa4item\xa4book\xa5price\xcb@#\xeb...'

# Decode from binary
order3 = msgspec.msgpack.decode(binary, type=Order)
```

| Format      | Function                                                     | Output               |
| ----------- | ------------------------------------------------------------ | -------------------- |
| JSON        | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">msgspec.json.encode/decode</code> | Human-readable bytes |
| MessagePack | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">msgspec.msgpack.encode/decode</code> | Compact binary       |

------

### 4) `array_like=True` — Array Serialization

```python
class Point(Struct, array_like=True):
    x: float
    y: float

p = Point(1.0, 2.0)
print(msgspec.json.encode(p))   # b'[1.0,2.0]'
```

**Scenario:** Compact serialization for large volumes of records (matrices, time series, coordinate data).

------

### 5) Handling Validation Errors

```python
bad_json = b'{"id": "not-a-number", "item": "book", "price": 9.99}'

try:
    msgspec.json.decode(bad_json, type=Order)
except msgspec.ValidationError as e:
    print(e)
    # Expected `int`, got `str` - at `$.id`
```

------

## 5. Type Annotations & Supported Types (类型注解)

### 1) Built-in Types

```python
class Example(Struct):
    a: int
    b: float
    c: str
    d: bool
    e: bytes
    f: None
```

------

### 2) Collections (集合类型)

```python
from typing import Optional

class Collections(Struct):
    items: list[str]
    mapping: dict[str, int]
    pair: tuple[int, str]
    unique: set[int]
    maybe: Optional[str] = None        # str | None
```

------

### 3) `Optional` and `Union` Types

```python
from typing import Union

class Response(Struct):
    data: Union[str, int, None]        # Can be str, int, or None
    error: str | None = None           # Python 3.10+ shorthand
```

------

### 4) `Literal` Types — Constrained Values (约束值)

```python
from typing import Literal

class Status(Struct):
    state: Literal["pending", "running", "done", "failed"]

s = Status(state="running")
msgspec.json.decode(b'{"state":"invalid"}', type=Status)
# ❌ ValidationError: Expected one of 'pending', 'running', 'done', 'failed'
```

**Scenario:** Enforcing valid enum-like values without a full `Enum` class.

------

### 5) `datetime`, `UUID`, `Decimal`

```python
from datetime import datetime
from uuid import UUID
from decimal import Decimal

class Event(Struct):
    id: UUID
    timestamp: datetime
    amount: Decimal
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">datetime</code> is serialized as an ISO 8601 string. <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">UUID</code> as a string. <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Decimal</code> as a JSON number string.</div>

------

## 6. Tagged Unions (标签联合) — Polymorphic Types

### 1) Defining a Tagged Union

Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">tag=True</code> (or a custom tag string) to enable discriminated unions (判别联合):

```python
from typing import Union

class Cat(Struct, tag=True):
    name: str
    indoor: bool

class Dog(Struct, tag=True):
    name: str
    breed: str

Animal = Union[Cat, Dog]
```

When serialized, a `"type"` field is added automatically:

```python
cat = Cat(name="Whiskers", indoor=True)
print(msgspec.json.encode(cat))
# b'{"type":"Cat","name":"Whiskers","indoor":true}'

dog = Dog(name="Rex", breed="Husky")
print(msgspec.json.encode(dog))
# b'{"type":"Dog","name":"Rex","breed":"Husky"}'
```

------

### 2) Decoding a Tagged Union

```python
data = b'{"type":"Dog","name":"Rex","breed":"Husky"}'
animal = msgspec.json.decode(data, type=Animal)
print(type(animal))   # <class 'Dog'>
print(animal.breed)   # Husky
```

**Scenario:** Event systems, polymorphic API responses, command/event patterns where the same endpoint can return different shapes.

------

### 3) Custom Tag Values

```python
class Circle(Struct, tag="circle"):
    radius: float

class Rectangle(Struct, tag="rect"):
    width: float
    height: float

Shape = Union[Circle, Rectangle]

c = Circle(radius=5.0)
print(msgspec.json.encode(c))
# b'{"type":"circle","radius":5.0}'
```

------

## 7. Utility Methods (工具方法)

### 1) `msgspec.structs.asdict()` — Convert to Dict

```python
from msgspec import structs

p = Point(x=1.0, y=2.0)
d = structs.asdict(p)
print(d)   # {'x': 1.0, 'y': 2.0}
```

------

### 2) `msgspec.structs.astuple()` — Convert to Tuple

```python
t = structs.astuple(p)
print(t)   # (1.0, 2.0)
```

------

### 3) `msgspec.structs.replace()` — Copy with Changes

Like `dataclasses.replace()` — creates a new instance with some fields updated:

```python
p2 = structs.replace(p, x=99.0)
print(p2)   # Point(x=99.0, y=2.0)
print(p)    # Point(x=1.0, y=2.0)  ← original unchanged
```

**Scenario:** Immutable update patterns (不可变更新模式) — create a modified copy without mutating the original.

------

### 4) `msgspec.structs.fields()` — Inspect Field Definitions

```python
for f in structs.fields(Point):
    print(f.name, f.type, f.default)
# x  <class 'float'>  NODEFAULT
# y  <class 'float'>  NODEFAULT
```

**Scenario:** Writing generic serializers, validators, or introspection tools.

------

## 8. Inheritance (继承)

```python
class Animal(Struct):
    name: str
    age: int

class Dog(Animal):
    breed: str

d = Dog(name="Rex", age=3, breed="Husky")
print(d)   # Dog(name='Rex', age=3, breed='Husky')
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">A child Struct cannot override a field defined in the parent.</span> Fields defined in the parent always come first in the constructor signature.</div>

------

## 9. Performance Comparison (性能对比)

| Library                                                      | Construct | JSON Encode        | JSON Decode+Validate |
| ------------------------------------------------------------ | --------- | ------------------ | -------------------- |
| <span style="color:#E8600A;font-weight:700">msgspec.Struct</span> | ⚡ Fastest | ⚡ Fastest          | ⚡ Fastest            |
| `dataclasses`                                                | Fast      | Needs `json.dumps` | No validation        |
| `attrs`                                                      | Fast      | Needs extra lib    | No validation        |
| `pydantic v2`                                                | Medium    | Fast               | Fast (Rust core)     |
| `pydantic v1`                                                | Slow      | Slow               | Slow                 |

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> In benchmarks, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">msgspec</code> is typically <strong>5–10× faster than Pydantic v1</strong> and <strong>2–3× faster than Pydantic v2</strong> for both encoding and decoding, while using significantly less memory.</div>

------

## 10. Real-World Scenarios (实战场景)

### 1) FastAPI / HTTP API Request/Response Models

```python
from msgspec import Struct
import msgspec

class CreateUserRequest(Struct):
    username: str
    email: str
    age: int | None = None

class UserResponse(Struct):
    id: int
    username: str
    email: str

# Decoding incoming JSON body
body = b'{"username":"alice","email":"alice@example.com"}'
req = msgspec.json.decode(body, type=CreateUserRequest)

# Encoding outgoing response
resp = UserResponse(id=42, username=req.username, email=req.email)
print(msgspec.json.encode(resp))
# b'{"id":42,"username":"alice","email":"alice@example.com"}'
```

------

### 2) Config File Parsing with Validation

```python
import msgspec
from msgspec import Struct
from typing import Literal

class ServerConfig(Struct):
    host: str = "0.0.0.0"
    port: int = 8080
    mode: Literal["debug", "production"] = "production"
    workers: int = 4

config_json = b'{"host":"127.0.0.1","port":9000,"mode":"debug"}'
config = msgspec.json.decode(config_json, type=ServerConfig)
print(config.mode)   # debug
```

------

### 3) High-throughput MessagePack Messaging (e.g., vLLM, message queues)

```python
class InferenceRequest(Struct):
    request_id: str
    prompt: str
    max_tokens: int = 512
    temperature: float = 1.0

class InferenceResponse(Struct):
    request_id: str
    output: str
    finish_reason: Literal["stop", "length", "error"]

# Fast binary serialization for IPC / queue transport
req = InferenceRequest(request_id="req-001", prompt="Hello!")
binary = msgspec.msgpack.encode(req)

resp_data = msgspec.msgpack.decode(binary, type=InferenceRequest)
```

------

### 4) Event / Command Pattern with Tagged Unions

```python
from typing import Union
from msgspec import Struct
import msgspec

class StartJob(Struct, tag=True):
    job_id: str
    config: dict

class StopJob(Struct, tag=True):
    job_id: str
    reason: str

Command = Union[StartJob, StopJob]

# Dispatcher
def handle(data: bytes):
    cmd = msgspec.json.decode(data, type=Command)
    if isinstance(cmd, StartJob):
        print(f"Starting job {cmd.job_id}")
    elif isinstance(cmd, StopJob):
        print(f"Stopping job {cmd.job_id}: {cmd.reason}")

handle(b'{"type":"StartJob","job_id":"abc","config":{}}')
# Starting job abc
```

------

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">msgspec.Struct</code> gives you the ergonomics of a dataclass, the validation of Pydantic, and the serialization speed of hand-written C — use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">frozen=True</code> for immutable value objects, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">tag=True</code> for polymorphic types, and <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">rename="camel"</code> for seamless REST API integration.</div>
