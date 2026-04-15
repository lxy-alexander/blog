---
title: "dataclass"
published: 2026-03-19
description: "dataclass"
image: ""
tags: ["python","oop","dataclass"]
category: python / oop
draft: false
lang: ""
---

# I. Dataclass

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9">
Python dataclass is a <span style="color:#E8600A;font-weight:700">decorator (装饰器)</span> that automatically generates special methods like <code>__init__</code> and <code>__repr__</code> for classes primarily used to <span style="color:#2980B9;font-weight:700">store data</span>. It reduces boilerplate code by letting you <span style="color:#2980B9;font-weight:700">declare fields as class variables</span> with type annotations. The dataclass makes your code more <span style="color:#E8600A;font-weight:700">readable and maintainable (可读性和可维护性)</span> by eliminating repetitive method definitions.
</div>

## 1. Basic Dataclass Definition

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85">
The <span style="color:#E8600A;font-weight:700">@dataclass decorator (装饰器)</span> automatically adds <span style="color:#2980B9;font-weight:700">__init__</span>, <span style="color:#2980B9;font-weight:700">__repr__</span>, and <span style="color:#2980B9;font-weight:700">__eq__</span> methods based on the class variables you define with <span style="color:#E8600A;font-weight:700">type hints (类型提示)</span>. Use this when you need a simple container for data without writing repetitive constructor code.
</div>
The `@dataclass` decorator auto-generates:

-   <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">**init**(self, x, y)</code> — constructor
-   <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">**repr**</code> — pretty string representation
-   <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">**eq**</code> — equality comparison


### 1) Basic Implementation

```python
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int
    email: str = "unknown@email.com"  # Default value

# Usage example
person1 = Person("Alice", 25, "alice@email.com")
person2 = Person("Bob", 30)  # Uses default email

print(person1)  # Automatically generated __repr__
print(person1 == person2)  # Automatically generated __eq__
```

<div style="background:#F5F5F5;border-left:4px solid #C0392B;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85;font-size:0.80em"><span style="color:#C0392B;font-weight:700">Note: </span> Fields without default values <span style="color:#C0392B;font-weight:700">must come before</span> fields with default values, otherwise Python raises a <span style="color:#C0392B;font-weight:700">SyntaxError (语法错误)</span>.
</div>

## 2. Field Customization

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85">
The <span style="color:#E8600A;font-weight:700">field() function (字段函数)</span> provides <span style="color:#2980B9;font-weight:700">fine-grained control</span> over individual dataclass fields, allowing you to set <span style="color:#E8600A;font-weight:700">default factories (默认工厂)</span>, exclude fields from comparisons, or mark fields as <span style="color:#2980B9;font-weight:700">private (私有)</span>.
</div>

### 1) Using field() with Parameters

```python
from dataclasses import dataclass, field
import random
from typing import List

@dataclass
class Student:
    name: str
    student_id: int = field(init=False)  # Not in __init__
    grades: List[int] = field(default_factory=list)  # Mutable default
    _internal_id: int = field(default=0, repr=False)  # Hidden in __repr__
    
    def __post_init__(self):
        # Initialize after dataclass generation
        self.student_id = random.randint(1000, 9999)
        self._internal_id = hash(self.name)

# Usage example
student = Student("Alice")
student.grades.append(95)  # Works with mutable default
print(student)  # Shows name and grades, but not _internal_id
```

<div style="background:#F5F5F5;border-left:4px solid #C0392B;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85;font-size:0.80em"><span style="color:#C0392B;font-weight:700">Note: </span> Always use <span style="color:#E8600A;font-weight:700">default_factory (默认工厂)</span> for mutable types like lists or dictionaries. Using <code>grades: List[int] = []</code> would cause all instances to <span style="color:#C0392B;font-weight:700">share the same list</span>.
</div>

## 3. Dataclass Parameters

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85">
The <span style="color:#E8600A;font-weight:700">@dataclass decorator</span> accepts parameters that <span style="color:#2980B9;font-weight:700">control which methods are generated</span>. Use <span style="color:#2980B9;font-weight:700">frozen=True</span> for immutable objects, <span style="color:#2980B9;font-weight:700">order=True</span> for sorting capabilities, and <span style="color:#2980B9;font-weight:700">kw_only=True</span> to enforce keyword arguments.
</div>

### 1) Configuration Options

```python
from dataclasses import dataclass

@dataclass(frozen=True, order=True)
class Point:
    x: int
    y: int

@dataclass(kw_only=True)  # Python 3.10+
class Configuration:
    host: str
    port: int = 8080

# Usage examples
p1 = Point(1, 2)
p2 = Point(1, 3)
# p1.x = 5  # This would raise FrozenInstanceError
print(p1 < p2)  # Works because order=True

# Must use keyword arguments
config = Configuration(host="localhost", port=3000)
# config = Configuration("localhost", 3000)  # This would fail
```

<div style="background:#F5F5F5;border-left:4px solid #C0392B;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85;font-size:0.80em"><span style="color:#C0392B;font-weight:700">Note: </span> When using <span style="color:#E8600A;font-weight:700">frozen=True</span>, the dataclass becomes <span style="color:#E8600A;font-weight:700">immutable (不可变的)</span> — you cannot modify attributes after creation. This is ideal for <span style="color:#2980B9;font-weight:700">configuration objects</span> or <span style="color:#2980B9;font-weight:700">value objects</span>.
</div>

## 4. Inheritance with Dataclasses

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85">
Dataclasses <span style="color:#2980B9;font-weight:700">support inheritance (继承)</span>, with fields from parent classes being combined with child class fields. Use this when you need to <span style="color:#2980B9;font-weight:700">extend data containers</span> while maintaining the automatic method generation.
</div>

### 1) Extending Dataclasses

```python
from dataclasses import dataclass

@dataclass
class Vehicle:
    brand: str
    model: str
    year: int

@dataclass
class Car(Vehicle):
    doors: int
    electric: bool = False

# Usage example
my_car = Car("Tesla", "Model 3", 2023, doors=4, electric=True)
print(my_car)  # Includes all fields from both classes
```

<div style="background:#F5F5F5;border-left:4px solid #C0392B;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85;font-size:0.80em"><span style="color:#C0392B;font-weight:700">Note: </span> When inheriting, the <span style="color:#E8600A;font-weight:700">field order matters</span> — child class fields are appended after parent fields. All fields without defaults in the parent must come before child fields with defaults.
</div>

## 5. Comparison Table: Regular Class vs Dataclass

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85">
This table compares the <span style="color:#E8600A;font-weight:700">boilerplate code (样板代码)</span> required for a simple data container using a regular class versus a dataclass.
</div>

| Feature | Regular Class | Dataclass |
|---------|--------------|-----------|
| <span style="color:#2980B9;font-weight:700">Lines of Code</span> | ~10-15 lines | ~3-5 lines |
| <span style="color:#E8600A;font-weight:700">__init__ method</span> | Manual implementation | Auto-generated |
| <span style="color:#E8600A;font-weight:700">__repr__ method</span> | Manual implementation | Auto-generated |
| <span style="color:#E8600A;font-weight:700">__eq__ method</span> | Manual implementation | Auto-generated |
| <span style="color:#2980B9;font-weight:700">Type hints</span> | Optional in body | Required for fields |
| <span style="color:#2980B9;font-weight:700">Default values</span> | In __init__ method | Direct field assignment |
| <span style="color:#C0392B;font-weight:700">Mutable defaults</span> | Safe with proper code | Must use default_factory |

### 1) Code Comparison Example

```python
# Regular class - 15 lines
class RegularPerson:
    def __init__(self, name: str, age: int, email: str = "unknown"):
        self.name = name
        self.age = age
        self.email = email
    
    def __repr__(self):
        return f"RegularPerson(name='{self.name}', age={self.age}, email='{self.email}')"
    
    def __eq__(self, other):
        if not isinstance(other, RegularPerson):
            return False
        return (self.name, self.age, self.email) == (other.name, other.age, other.email)

# Dataclass - 4 lines
@dataclass
class DataclassPerson:
    name: str
    age: int
    email: str = "unknown"
```

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br>Python dataclasses automatically generate <span style="color:#E8600A;font-weight:700">__init__, __repr__, and __eq__</span> from type-annotated fields, eliminating boilerplate code for simple data containers.
</div>