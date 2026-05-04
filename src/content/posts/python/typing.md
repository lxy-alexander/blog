---
title: "typing"
published: 2026-05-02
description: "typing"
image: ""
tags: ["python","typing"]
category: python
draft: false
lang: ""
createdAt: "2026-05-02T15:06:51.466.368658107Z"
---

# Python Typing

These imports are common tools from the `typing` module (类型模块), and you should explain them by focusing on intent, readability, and static type checking (静态类型检查).

## 1. TYPE_CHECKING

`TYPE_CHECKING` (类型检查标志) is flag that is`False` at runtime but treated as `True` by type checkers (类型检查器), so it is used to avoid runtime imports.

```
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import math

print(TYPE_CHECKING)
# Output: False
```

<br>

## 2. Annotated

`Annotated` (带注解类型) attaches extra metadata (元数据) to a type without changing the original runtime type.

```
from typing import Annotated, get_args, get_origin

Age = Annotated[int, "must be >= 0"]

print(get_origin(Age))
# Output: <class 'typing.Annotated'>

print(get_args(Age))
# Output: (<class 'int'>, 'must be >= 0')
```

<br>

## 3. Any

`Any` (任意类型, arbitrary type) disables strict type checking (严格类型检查) for a value, so it should be used carefully.

```
from typing import Any

def show(value: Any) -> None:
    print(value)

show(123)
# Output: 123

show("hello")
# Output: hello
```

<br>

## 4. Literal

`Literal` (字面量类型) restricts a value to specific allowed constants (常量).

```
from typing import Literal

def set_mode(mode: Literal["train", "test"]) -> None:
    print(f"Mode: {mode}")

set_mode("train")
# Output: Mode: train
```

<br>

## 5. TypeAlias

`TypeAlias` (类型别名) explicitly declares that a name is an alias (别名) for another type.

```
from typing import TypeAlias

UserId: TypeAlias = int

def get_user(user_id: UserId) -> None:
    print(f"User ID: {user_id}")

get_user(101)
# Output: User ID: 101
```

<br>

## 6. TypeVar

`TypeVar` (类型变量) represents a generic type (泛型类型) that can be reused consistently across a function or class.

```
from typing import TypeVar

T = TypeVar("T")

def first(items: list[T]) -> T:
    return items[0]

print(first([1, 2, 3]))
# Output: 1

print(first(["a", "b", "c"]))
# Output: a
```

<br>

## 7. Union

`Union` (联合类型) means a value can be one of multiple possible types (可能类型).

```
from typing import Union

def stringify(value: Union[int, str]) -> str:
    return str(value)

print(stringify(100))
# Output: 100

print(stringify("abc"))
# Output: abc
```

<br>

## 8. cast

`cast` (类型转换提示) tells the type checker (类型检查器) to treat a value as a specific type, but it does not change the value at runtime.

```
from typing import Any, cast

value: Any = "hello"

text = cast(str, value)

print(text.upper())
# Output: HELLO

print(type(text))
# Output: <class 'str'>
```

<br>

## 9. get_args

`get_args` (获取类型参数) returns the inner type arguments (类型参数) of a generic type (泛型类型).

```
from typing import Union, get_args

MyType = Union[int, str]

print(get_args(MyType))
# Output: (<class 'int'>, <class 'str'>)
```

<br>

## 10. get_origin

`get_origin` (获取原始类型) returns the base generic type (基础泛型类型) behind a typing object.

```
from typing import Union, get_origin

MyType = Union[int, str]

print(get_origin(MyType))
# Output: typing.Union
```

<br>

## 11. @overload

The `@overload` (重载) is It is "Function Overloading declaration". It is used for precise static type checking. It allows you to define multiple type signatures for the same function, while keeping only one real implementation.

```
from typing import overload, Union

@overload
def process(response: int) -> int: ...

@overload
def process(response: str) -> str: ...

def process(response: Union[int, str]) -> Union[int, str]:
    if isinstance(response, int):
        return response + 1
    return response.upper()
```

<br><br>
