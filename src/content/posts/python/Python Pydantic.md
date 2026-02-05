---
title: "Python Pydantic"
published: 2026-02-02
description: "Python Pydantic"
image: ""
tags: ["python","Python Pydantic"]
category: python
draft: false
lang: ""
---

>   **Pydantic is a Python data validation and settings management library.**（Pydantic 是一个用于 Python 数据校验和配置管理的库）

## 1) `BaseModel` (Define models)

It defines a typed data structure (schema). Pydantic will create an object and validate the input based on the type hints.

```python
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str

u = User(id=1, name="Tom")
print(u)
```

------



## 2) Validation + Type Coercion

It validates input types and will automatically convert compatible values (e.g. `"123"` → `123`). If it can’t validate/convert, it raises an error.

```python
from pydantic import BaseModel

class User(BaseModel):
    id: int

u = User(id="123")  # "123" -> 123
print(u.id, type(u.id))
```

------





## 3) `Field` (Constraints / Metadata)

`Field()` lets you add constraints (like min/max) and extra metadata (like description). Pydantic will enforce the constraints during validation.

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    age: int = Field(ge=0, le=150, description="Age must be between 0 and 150")

u = User(age=18)
print(u)
```

------

## 4) Nested Models

A model can contain another model as a field. Pydantic will validate nested dictionaries and convert them into nested model objects automatically.

```python
from pydantic import BaseModel

class Address(BaseModel):
    city: str

class User(BaseModel):
    name: str
    address: Address

u = User(name="Tom", address={"city": "Shanghai"})
print(u.address.city)
```

------

## 5) Optional Fields + Defaults

Optional fields (`T | None`) can be missing or set to `None`. Default values make the field optional during initialization.

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    email: str | None = None

u1 = User(name="Tom")
u2 = User(name="Jerry", email="jerry@example.com")

print(u1)
print(u2)
```

------

## 6) `@field_validator` (Field-level validation)

It lets you write custom validation logic for a specific field (e.g., trimming spaces, format checks, rejecting certain values).

```python
from pydantic import BaseModel, field_validator

class User(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str):
        if not v.strip():
            raise ValueError("name must not be empty")
        return v.strip()

u = User(name="  Tom  ")
print(u)
```

------

## 7) `@model_validator` (Model-level validation)

It validates the model as a whole, which is useful for rules involving multiple fields (cross-field validation).

```python
from pydantic import BaseModel, model_validator

class Login(BaseModel):
    username: str
    password: str

    @model_validator(mode="after")
    def check_password(self):
        if len(self.password) < 6:
            raise ValueError("password too short (min 6)")
        return self

ok = Login(username="tom", password="123456")
print(ok)
```

------

## 8) `model_dump()` (Export to dict)

It converts a validated model instance into a standard Python dictionary, often used for business logic or API responses.

```python
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str

u = User(id=1, name="Tom")
print(u.model_dump())
print(type(u.model_dump()))
```

------

## 9) `model_dump_json()` (Export to JSON)

It converts the model into a JSON string, convenient for HTTP responses, logs, caches, or storing data.

```python
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str

u = User(id=1, name="Tom")
print(u.model_dump_json())
print(type(u.model_dump_json()))
```

------

## 10) `BaseSettings` (Settings management)

It reads configuration from environment variables (and other sources) into a typed model, validating types just like `BaseModel`.

```bash
pip install pydantic-settings
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "demo"
    debug: bool = False

s = Settings()
print(s.app_name, s.debug)
```

(Optional) run with environment variables:

```bash
export APP_NAME=myapp
export DEBUG=true
python your_file.py
```
