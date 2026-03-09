---
title: "python dataclass"
published: 2026-03-08
description: "python dataclass"
image: ""
tags: ["python","python dataclass"]
category: python
draft: false
lang: ""
---

# **I. Python `dataclasses` — Complete Learning Handbook**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <strong>Overview:</strong> Python's <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">dataclasses</code> module (introduced in Python 3.7) provides a <strong>decorator (装饰器)</strong> that automatically generates boilerplate methods — <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">__init__</code>, <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">__repr__</code>, <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">__eq__</code> — for classes that primarily store data. It sits between a plain class and a full ORM/validation framework, offering clean syntax with zero runtime overhead beyond standard Python. </div>

------

## 1. Installation & Import

`dataclasses` is part of the Python standard library — no installation required.

```python
from dataclasses import dataclass, field, fields, asdict, astuple, replace, KW_ONLY
```

------

## 2. Defining a Dataclass (定义数据类)

### 1) Basic Definition

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

p = Point(x=1.0, y=2.0)
print(p)          # Point(x=1.0, y=2.0)
print(p.x)        # 1.0
print(p == Point(1.0, 2.0))   # True
```

The `@dataclass` decorator auto-generates:

-   <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">**init**(self, x, y)</code> — constructor
-   <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">**repr**</code> — pretty string representation
-   <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">**eq**</code> — field-by-field equality comparison

------

### 2) Fields with Default Values (默认值)

```python
@dataclass
class User:
    name: str
    age: int = 0
    active: bool = True

u = User(name="Alice")
print(u)   # User(name='Alice', age=0, active=True)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">Fields with defaults must come after fields without defaults</span> — same rule as regular Python function parameters. Violating this raises a <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">TypeError</code>.</div>

------

### 3) `field()` — Advanced Field Configuration

```python
from dataclasses import dataclass, field

@dataclass
class Config:
    tags: list[str] = field(default_factory=list)     # Mutable default
    name: str = field(default="unnamed")
    _secret: str = field(default="", repr=False)       # Hidden from repr
    metadata: dict = field(default_factory=dict, compare=False)  # Excluded from ==
```

### `field()` Parameter Reference

| Parameter                                                    | Default   | Effect                                                    |
| ------------------------------------------------------------ | --------- | --------------------------------------------------------- |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">default</code> | `MISSING` | Scalar default value                                      |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">default_factory</code> | `MISSING` | Callable that produces the default (for mutable types)    |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">repr</code> | `True`    | Include field in `__repr__` output                        |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">compare</code> | `True`    | Include field in `__eq__` and `__lt__` etc.               |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">hash</code> | `None`    | Include field in `__hash__` (None = follow `compare`)     |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">init</code> | `True`    | Include field as a parameter in `__init__`                |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">metadata</code> | `None`    | Arbitrary read-only mapping attached to the field         |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">kw_only</code> | `False`   | Force this field to be keyword-only in `__init__` (3.10+) |

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">Never use a mutable object (list, dict, set) directly as a default value.</span> Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">field(default_factory=list)</code> — otherwise all instances share the same object.</div>

------

### 4) Nested Dataclasses (嵌套数据类)

```python
@dataclass
class Address:
    city: str
    country: str

@dataclass
class Person:
    name: str
    address: Address

p = Person(name="Bob", address=Address(city="NYC", country="US"))
print(p.address.city)   # NYC
print(p)
# Person(name='Bob', address=Address(city='NYC', country='US'))
```

------

## 3. `@dataclass` Decorator Options (装饰器配置)

```python
@dataclass(frozen=True, order=True, eq=True, repr=True, unsafe_hash=False, slots=True)
class Config:
    x: int
    y: int
```

### Option Reference Table

| Option                                                       | Default | Effect                                                       |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">init</code> | `True`  | Generate `__init__`                                          |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">repr</code> | `True`  | Generate `__repr__`                                          |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">eq</code> | `True`  | Generate `__eq__` based on field values                      |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">order</code> | `False` | Generate `__lt__`, `__le__`, `__gt__`, `__ge__`              |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">frozen</code> | `False` | Make instances <strong>immutable (不可变)</strong> — raises `FrozenInstanceError` on assignment |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">unsafe_hash</code> | `False` | Force generate `__hash__` even if `eq=True` (use with care)  |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">slots</code> | `False` | Use `__slots__` for faster attribute access and lower memory (3.10+) |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">kw_only</code> | `False` | All fields must be passed as keyword arguments (3.10+)       |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">match_args</code> | `True`  | Generate `__match_args__` for structural pattern matching (3.10+) |

------

### 1) `frozen=True` — Immutable Dataclass

```python
@dataclass(frozen=True)
class ImmutablePoint:
    x: float
    y: float

p = ImmutablePoint(x=1.0, y=2.0)
p.x = 99.0   # ❌ FrozenInstanceError: cannot assign to field 'x'

# Frozen dataclasses are hashable and can be used as dict keys
d = {p: "origin"}
```

**Scenario:** Configuration objects (配置对象), cache keys, value objects (值对象) that must never be mutated.

------

### 2) `order=True` — Sortable Dataclasses

```python
@dataclass(order=True)
class Version:
    major: int
    minor: int
    patch: int

versions = [Version(1, 2, 0), Version(1, 0, 5), Version(2, 0, 0)]
print(sorted(versions))
# [Version(major=1, minor=0, patch=5), Version(major=1, minor=2, patch=0), Version(major=2, minor=0, patch=0)]
```

**Scenario:** Sorting records, priority queues, range-based comparisons.

------

### 3) `slots=True` — Memory-efficient Dataclass (Python 3.10+)

```python
@dataclass(slots=True)
class FastPoint:
    x: float
    y: float

# __slots__ prevents arbitrary attribute addition and speeds up attribute access
p = FastPoint(1.0, 2.0)
p.z = 3.0   # ❌ AttributeError: 'FastPoint' object has no attribute 'z'
```

**Scenario:** Creating millions of small instances (数百万小对象) — data pipelines, geometry, particle simulations.

------

### 4) `kw_only=True` — Keyword-only Fields (Python 3.10+)

```python
@dataclass(kw_only=True)
class Request:
    url: str
    method: str = "GET"
    timeout: float = 30.0

r = Request(url="https://api.example.com")
# r = Request("https://api.example.com")  ❌ TypeError
```

Use `KW_ONLY` sentinel to make only some fields keyword-only:

```python
from dataclasses import KW_ONLY

@dataclass
class Mixed:
    x: int
    y: int
    _: KW_ONLY          # Everything after this is keyword-only
    label: str = ""
    weight: float = 1.0

m = Mixed(1, 2, label="point")   # x and y are positional, label is kw-only
```

------

## 4. `__post_init__` — Post-initialization Hook (初始化后钩子)

Runs automatically after `__init__` completes. Use it for validation, derived fields, or type coercion.

### 1) Validation

```python
@dataclass
class Temperature:
    celsius: float

    def __post_init__(self):
        if self.celsius < -273.15:
            raise ValueError(f"Temperature {self.celsius}°C is below absolute zero!")

t = Temperature(-300)   # ❌ ValueError
```

------

### 2) Derived Fields with `field(init=False)`

```python
import math

@dataclass
class Circle:
    radius: float
    area: float = field(init=False)        # Not in __init__
    circumference: float = field(init=False)

    def __post_init__(self):
        self.area = math.pi * self.radius ** 2
        self.circumference = 2 * math.pi * self.radius

c = Circle(radius=5.0)
print(c.area)           # 78.539...
print(c.circumference)  # 31.415...
```

------

### 3) Type Coercion

```python
@dataclass
class Coordinate:
    lat: float
    lon: float

    def __post_init__(self):
        # Auto-convert strings to float
        self.lat = float(self.lat)
        self.lon = float(self.lon)

coord = Coordinate(lat="51.5", lon="-0.1")
print(coord.lat, type(coord.lat))   # 51.5 <class 'float'>
```

------

### 4) `InitVar` — Init-only Parameters (仅初始化参数)

Fields that exist in `__init__` but are NOT stored as instance attributes:

```python
from dataclasses import dataclass, field, InitVar

@dataclass
class HashedPassword:
    username: str
    raw_password: InitVar[str]           # Passed to __init__ but not stored
    password_hash: str = field(init=False)

    def __post_init__(self, raw_password: str):
        import hashlib
        self.password_hash = hashlib.sha256(raw_password.encode()).hexdigest()

u = HashedPassword(username="alice", raw_password="secret123")
print(u.password_hash)    # sha256 hash
# print(u.raw_password)   # ❌ AttributeError — not stored
```

------

## 5. Utility Functions (工具函数)

### 1) `asdict()` — Convert to Dictionary

```python
from dataclasses import asdict

@dataclass
class Point:
    x: float
    y: float

p = Point(1.0, 2.0)
print(asdict(p))   # {'x': 1.0, 'y': 2.0}
```

Works recursively on nested dataclasses:

```python
person = Person(name="Bob", address=Address(city="NYC", country="US"))
print(asdict(person))
# {'name': 'Bob', 'address': {'city': 'NYC', 'country': 'US'}}
```

------

### 2) `astuple()` — Convert to Tuple

```python
from dataclasses import astuple

print(astuple(p))   # (1.0, 2.0)
```

------

### 3) `replace()` — Copy with Changes (不可变更新)

Creates a **new instance** with specified fields replaced — the original is unchanged:

```python
from dataclasses import replace

p = Point(x=1.0, y=2.0)
p2 = replace(p, x=99.0)
print(p2)   # Point(x=99.0, y=2.0)
print(p)    # Point(x=1.0, y=2.0)  ← original unchanged
```

**Scenario:** Immutable update patterns — building modified configurations, functional state updates.

------

### 4) `fields()` — Inspect Field Definitions

```python
from dataclasses import fields

for f in fields(Point):
    print(f.name, f.type, f.default)
# x  float  MISSING
# y  float  MISSING
```

**Scenario:** Writing generic serializers, validators, or introspection utilities (内省工具).

------

### 5) `is_dataclass()` — Runtime Type Check

```python
from dataclasses import is_dataclass

print(is_dataclass(Point))      # True  (class)
print(is_dataclass(Point(1,2))) # True  (instance)
print(is_dataclass(int))        # False
```

------

## 6. Inheritance (继承)

```python
@dataclass
class Animal:
    name: str
    age: int

@dataclass
class Dog(Animal):
    breed: str
    trained: bool = False

d = Dog(name="Rex", age=3, breed="Husky")
print(d)   # Dog(name='Rex', age=3, breed='Husky', trained=False)
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">If a parent class has a field with a default value, the child class cannot add fields <em>without</em> a default — this violates the "defaults must come last" rule and raises a <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">TypeError</code>.</span> Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">kw_only=True</code> on the child to avoid this.</div>

```python
@dataclass
class Base:
    x: int = 0     # Has default

@dataclass(kw_only=True)
class Child(Base):
    y: int         # No default — OK because kw_only avoids ordering conflict
```

------

## 7. Hashing & Usage as Dict Keys (哈希与字典键)

```python
# eq=True + frozen=True → hashable
@dataclass(frozen=True)
class Point:
    x: float
    y: float

p = Point(1.0, 2.0)
cache = {p: "result"}   # ✅ Can be used as dict key or in set

# eq=True + frozen=False (default) → NOT hashable
@dataclass
class MutablePoint:
    x: float
    y: float

mp = MutablePoint(1.0, 2.0)
# {mp: "x"}  ❌ TypeError: unhashable type
```

| `eq`    | `frozen`                     | `__hash__`                                                   |
| ------- | ---------------------------- | ------------------------------------------------------------ |
| `False` | `False`                      | Inherited from `object` (id-based)                           |
| `True`  | `False`                      | Set to `None` — <span style="color:#C0392B;font-weight:600">unhashable</span> |
| `True`  | `True`                       | Generated — <span style="color:#E8600A;font-weight:700">hashable</span> ✅ |
| `True`  | `False` + `unsafe_hash=True` | Force-generated — use with caution                           |

------

## 8. Pattern Matching with Dataclasses (Python 3.10+)

```python
@dataclass
class Point:
    x: float
    y: float

def describe(p):
    match p:
        case Point(x=0, y=0):
            return "Origin"
        case Point(x=0, y=y):
            return f"On Y-axis at {y}"
        case Point(x=x, y=0):
            return f"On X-axis at {x}"
        case Point(x=x, y=y):
            return f"Point at ({x}, {y})"

print(describe(Point(0, 5)))     # On Y-axis at 5
print(describe(Point(3, 4)))     # Point at (3, 4)
```

------

## 9. Comparison: `dataclasses` vs Alternatives

| Feature             | `dataclasses`   | `attrs`        | `msgspec.Struct` | `pydantic`  |
| ------------------- | --------------- | -------------- | ---------------- | ----------- |
| **Stdlib**          | ✅ Yes           | ❌              | ❌                | ❌           |
| **Auto `__init__`** | ✅               | ✅              | ✅                | ✅           |
| **Validation**      | ❌ Manual        | ✅ (validators) | ✅ Built-in       | ✅ Built-in  |
| **Serialization**   | ❌ Manual        | ❌ Manual       | ✅ JSON/MsgPack   | ✅ JSON      |
| **Performance**     | ⚡ Fast          | ⚡ Fast         | ⚡⚡ Fastest       | 🐢→⚡ (v1→v2) |
| **Frozen support**  | ✅               | ✅              | ✅                | ✅           |
| **`__slots__`**     | ✅ (3.10+)       | ✅              | ✅ (C-level)      | ❌           |
| **Inheritance**     | ✅               | ✅              | ✅ (limited)      | ✅           |
| **Ecosystem fit**   | Standard Python | Power users    | High-perf I/O    | Web APIs    |

------

## 10. Real-World Scenarios (实战场景)

### 1) Configuration Object

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class AppConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    allowed_origins: tuple[str, ...] = ("*",)

config = AppConfig(port=9000, debug=True)
print(config)
# AppConfig(host='0.0.0.0', port=9000, debug=True, allowed_origins=('*',))
```

------

### 2) Data Pipeline Record

```python
from dataclasses import dataclass, field
from datetime import datetime

@dataclass(slots=True)
class LogRecord:
    timestamp: datetime
    level: str
    message: str
    tags: list[str] = field(default_factory=list)

records = [LogRecord(datetime.now(), "INFO", f"event {i}") for i in range(1_000_000)]
```

------

### 3) API Request / Response Model

```python
import json
from dataclasses import dataclass, asdict, field
from typing import Optional

@dataclass
class CreateUserRequest:
    username: str
    email: str
    age: Optional[int] = None

@dataclass
class UserResponse:
    id: int
    username: str
    email: str

req = CreateUserRequest(username="alice", email="alice@example.com")
resp = UserResponse(id=42, username=req.username, email=req.email)
print(json.dumps(asdict(resp)))
# {"id": 42, "username": "alice", "email": "alice@example.com"}
```

------

### 4) State Machine Node

```python
from dataclasses import dataclass, replace
from typing import Literal

@dataclass(frozen=True)
class JobState:
    job_id: str
    status: Literal["pending", "running", "done", "failed"]
    retries: int = 0

# Immutable state transitions
initial = JobState(job_id="abc", status="pending")
running = replace(initial, status="running")
failed  = replace(running, status="failed", retries=running.retries + 1)
retry   = replace(failed,  status="running", retries=failed.retries)

print(initial)   # JobState(job_id='abc', status='pending', retries=0)
print(retry)     # JobState(job_id='abc', status='running', retries=1)
```

------

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">@dataclass</code> as your default data container, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">frozen=True</code> for immutable value objects and hashable keys, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">__post_init__</code> for validation and derived fields, <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">replace()</code> for functional state updates, and <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">slots=True</code> when creating millions of instances.</div>
