---
title: "Python typing"
published: 2026-03-09
description: "Python typing"
image: ""
tags: ["python","Python typing"]
category: python
draft: false
lang: ""
---

# **I. Python Typing (类型注解)**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
Python's <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">typing</code> module adds <span style="color:#2980B9">type hints (类型提示)</span> to Python. While Python remains dynamically typed, type hints help with <span style="color:#E8600A;font-weight:700">code documentation, IDE autocompletion, and catching bugs</span> before runtime using tools like <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">mypy</code>.
</div>

## 1. Basic Type Annotations

### 1) Variable Annotations

```python
# Basic types
name: str = "Alice"
age: int = 30
height: float = 1.75
is_student: bool = True

# Without initial value
address: str  # Just type annotation, no value yet
address = "123 Main St"  # Later assignment
```

### 2) Function Annotations

```python
def greet(name: str) -> str:
    return f"Hello, {name}!"

def add(a: int, b: int) -> int:
    return a + b

def process_data(data: list) -> None:  # None means returns nothing
    print(f"Processing {len(data)} items")
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span>Type hints are <span style="color:#C0392B;font-weight:600">not enforced at runtime</span>. They're for developers and tools only. Python won't stop you from passing an <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">int</code> to a <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">str</code> parameter!</div>

## 2. Container Types

### 1) List, Tuple, Set, Dict

```python
from typing import List, Tuple, Set, Dict

# List of strings
names: List[str] = ["Alice", "Bob", "Charlie"]

# Tuple of int and str (fixed length, mixed types)
person: Tuple[int, str, bool] = (1, "Alice", True)

# Set of integers
unique_ids: Set[int] = {101, 102, 103}

# Dictionary with string keys and int values
scores: Dict[str, int] = {"Alice": 95, "Bob": 87}
```

### 2) Nested Containers

```python
from typing import List, Dict, Tuple

# List of dictionaries
users: List[Dict[str, str]] = [
    {"name": "Alice", "email": "alice@example.com"},
    {"name": "Bob", "email": "bob@example.com"}
]

# Complex nesting
matrix: List[List[int]] = [[1, 2, 3], [4, 5, 6]]

# Tuple with list inside
data: Tuple[int, List[str], bool] = (1, ["a", "b", "c"], True)
```

## 3. Optional and Union Types

### 1) Optional (value or None)

```python
from typing import Optional

def find_user(user_id: int) -> Optional[str]:
    # Returns name or None if not found
    users = {1: "Alice", 2: "Bob"}
    return users.get(user_id)  # May return None

# Optional[str] means either str or None
result: Optional[str] = find_user(1)  # "Alice"
result2: Optional[str] = find_user(99)  # None
```

### 2) Union (multiple possible types)

```python
from typing import Union

# Function accepts int OR float
def square(value: Union[int, float]) -> Union[int, float]:
    return value * value

# Modern syntax (Python 3.10+)
def square2(value: int | float) -> int | float:
    return value * value

# Multiple types
def process_id(id_value: int | str) -> str:
    return f"ID: {id_value}"
```

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
<span style="color:#E8600A;font-weight:700">Python 3.10+</span> introduced the <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">|</code> syntax for Union types, making code cleaner!
</div>

## 4. Type Aliases

Create readable names for complex types:

```python
from typing import List, Tuple, Dict

# Type alias
UserID = int
UserName = str
UserInfo = Tuple[UserID, UserName, bool]

# Using the alias
def get_user() -> UserInfo:
    return (1, "Alice", True)

# More complex alias
Coordinate = Tuple[float, float]
Polygon = List[Coordinate]

def calculate_area(shape: Polygon) -> float:
    # shape is list of (x, y) tuples
    pass
```

## 5. Callable Types

Type functions that accept functions:

```python
from typing import Callable

# Function that takes an int and returns a str
def apply_twice(func: Callable[[int], str], value: int) -> str:
    return func(func(value))  # First call returns str, second call fails!

# Fixed version - proper typing catches this!
def apply_twice_fixed(func: Callable[[int], int], value: int) -> int:
    return func(func(value))

# More complex callback
def process_items(
    items: List[int],
    callback: Callable[[int], str]
) -> List[str]:
    return [callback(item) for item in items]
```

## 6. Any and TypeVar (Generics)

### 1) Any - escape hatch

```python
from typing import Any

# Use sparingly - defeats type checking!
def debug_print(value: Any) -> None:
    print(f"Value: {value}, Type: {type(value)}")
```

### 2) TypeVar - generic functions

```python
from typing import TypeVar, List

T = TypeVar('T')  # Generic type variable

def first_element(items: List[T]) -> T:
    """Returns first element, type preserved"""
    return items[0]

# Works with any list type
num = first_element([1, 2, 3])        # num is int
text = first_element(["a", "b", "c"]) # text is str

# Multiple type variables
K = TypeVar('K')
V = TypeVar('V')

def get_value(dict: Dict[K, V], key: K, default: V) -> V:
    return dict.get(key, default)
```

## 7. Special Forms

### 1) Literal - exact values

```python
from typing import Literal

def set_status(status: Literal["active", "inactive", "pending"]) -> None:
    print(f"Status set to {status}")

set_status("active")   # OK
set_status("active")    # OK
# set_status("unknown") # Type checker would complain

# Multiple literal values
def move(direction: Literal["north", "south", "east", "west"]) -> None:
    pass
```

### 2) Final - constants

```python
from typing import Final

MAX_SIZE: Final[int] = 100
PI: Final[float] = 3.14159

# Type checker would warn against reassignment
# MAX_SIZE = 200  # Error!
```

## 8. Protocol (Structural Subtyping)

Like interfaces, but duck-typed:

```python
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> None: ...

class Circle:
    def draw(self) -> None:
        print("Drawing circle")

class Square:
    def draw(self) -> None:
        print("Drawing square")
    def area(self) -> float:
        return 16.0

def render(obj: Drawable) -> None:
    obj.draw()  # Works with anything that has draw()

render(Circle())  # OK
render(Square())  # OK - Square has draw()
```

## 9. TypedDict - Dictionary with fixed keys

```python
from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int
    email: str

# Works like a dict, but with type checking
alice: Person = {
    "name": "Alice",
    "age": 30,
    "email": "alice@example.com"
}

# Error if missing keys or wrong types
# bob: Person = {"name": "Bob"}  # Missing age, email
```

## 10. Practical Examples

### 1) API Response Handler

```python
from typing import Dict, List, Optional, Union, TypedDict
import json

class User(TypedDict):
    id: int
    name: str
    email: str
    is_active: bool

class APIResponse(TypedDict):
    status: Literal["success", "error"]
    data: Optional[List[User]]
    message: Optional[str]

def parse_api_response(json_str: str) -> APIResponse:
    data = json.loads(json_str)
    return {
        "status": data["status"],
        "data": data.get("users"),
        "message": data.get("message")
    }
```

### 2) Data Processing Pipeline

```python
from typing import List, Callable, TypeVar

T = TypeVar('T')
U = TypeVar('U')

def pipeline(
    data: List[T],
    *transforms: Callable[[T], U]
) -> List[U]:
    """Apply multiple transforms to data"""
    result: List[U] = []
    for item in data:
        current = item
        for transform in transforms:
            current = transform(current)  # type: ignore
        result.append(current)  # type: ignore
    return result

# Usage
numbers = [1, 2, 3, 4]
result = pipeline(
    numbers,
    lambda x: x * 2,
    lambda x: str(x)
)  # result is List[str]
```

### 3) Configuration Manager

```python
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class DatabaseConfig:
    host: str
    port: int = 5432
    username: str
    password: str
    database: str

@dataclass
class AppConfig:
    debug: bool = False
    database: DatabaseConfig
    allowed_hosts: List[str]
    secret_key: Optional[str] = None

def load_config(config_dict: Dict[str, Any]) -> AppConfig:
    db_config = DatabaseConfig(**config_dict["database"])
    return AppConfig(
        debug=config_dict.get("debug", False),
        database=db_config,
        allowed_hosts=config_dict["allowed_hosts"],
        secret_key=config_dict.get("secret_key")
    )
```

## 11. Type Checking Tools

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">🔧 Tools to check types: </span>

1. **mypy** - Most popular
   ```bash
   pip install mypy
   mypy your_script.py
   ```

2. **pydantic** - Runtime type checking + data validation
   ```python
   from pydantic import BaseModel
   
   class User(BaseModel):
       name: str
       age: int
   
   user = User(name="Alice", age=30)  # Validates at runtime!
   ```

3. **VS Code/PyCharm** - Built-in type checking with Pylance/Pyright
</div>

## 12. Best Practices

### ✅ DO:

```python
# 1. Use type hints for public APIs
def calculate_total(prices: List[float]) -> float:
    return sum(prices)

# 2. Use Optional for values that can be None
def find_by_id(id: int) -> Optional[User]:
    pass

# 3. Use type aliases for complex types
JSON = Dict[str, Any]

# 4. Use Literal for limited options
def set_mode(mode: Literal["read", "write", "append"]) -> None:
    pass
```

### ❌ DON'T:

```python
# 1. Don't overuse Any (defeats purpose)
def process(data: Any) -> Any:  # Better to be specific

# 2. Don't ignore type errors without reason
result = func()  # type: ignore  # Add comment explaining why

# 3. Don't use type hints for everything in simple scripts
# For small scripts, they can be overkill
```

## 13. Comparison Table

| Feature        | Python 3.8-3.9          | Python 3.10+            | Python 3.11+     |
| -------------- | ----------------------- | ----------------------- | ---------------- |
| Union          | `Union[int, str]`       | `int \| str`            | `int \| str`     |
| Optional       | `Optional[str]`         | `str \| None`           | `str \| None`    |
| List type      | `List[int]`             | `list[int]`             | `list[int]`      |
| Dict type      | `Dict[str, int]`        | `dict[str, int]`        | `dict[str, int]` |
| Self reference | Forward ref `'MyClass'` | Forward ref `'MyClass'` | `Self` type      |

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>Python's <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">typing</code> adds optional <span style="color:#2980B9">type hints (类型提示)</span> that won't affect runtime but make code <span style="color:#E8600A;font-weight:700">self-documenting, IDE-friendly, and catch bugs early</span> with tools like <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">mypy</code>.</div>
