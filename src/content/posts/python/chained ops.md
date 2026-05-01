---
title: "chained ops"
published: 2026-04-30
description: "chained ops"
image: ""
tags: ["python","chained ops"]
category: python
draft: false
lang: ""
createdAt: "2026-05-01T01:01:56.564.589986454Z"
---

# Python Chain

Python Chain refers to **method chaining (方法链)** and **chained comparison (链式比较)**, two common patterns where multiple operations are linked together in a single expression to produce concise and readable code.

## 1. Chained Comparison (链式比较)

Python allows multiple comparison operators (比较运算符) to be chained in one expression, which is evaluated as the logical AND (逻辑与) of each adjacent pair.

```python
x = 15

# Chained comparison (链式比较)
result = 10 < x < 20
print(result)  # Output: True

# Equivalent to (等价于)
result2 = (10 < x) and (x < 20)
print(result2)  # Output: True

# Multiple chains (多重链)
a, b, c = 1, 2, 3
print(a < b < c)        # Output: True
print(a < b > 0)        # Output: True
print(1 == 1.0 == True) # Output: True
```

<br>

## 2. Method Chaining (方法链)

Method chaining means calling multiple methods on the same object in sequence, where each method returns an object (usually `self` or a new instance) that supports the next call.

```python
# String method chaining (字符串方法链)
text = "  Hello World  "
result = text.strip().lower().replace("world", "python")
print(result)  # Output: hello python

# List comprehension + chaining (列表推导式 + 链式调用)
nums = [3, 1, 4, 1, 5, 9, 2, 6]
result = sorted(set(nums))[:3]
print(result)  # Output: [1, 2, 3]
```

<br>

## 3. Custom Chainable Class (自定义链式类)

To support method chaining in a custom class, each method must `return self` so that subsequent calls operate on the same instance (实例).

```python
class Pipeline:
    def __init__(self, value):
        self.value = value
    
    def add(self, n):
        self.value += n
        return self  # Return self for chaining (返回自身以支持链式调用)
    
    def multiply(self, n):
        self.value *= n
        return self
    
    def result(self):
        return self.value

# Chained calls (链式调用)
output = Pipeline(5).add(3).multiply(2).add(1).result()
print(output)  # Output: 17
```

<br>

## 4. Pandas Chain (Pandas 链式操作)

In data analysis (数据分析), Pandas DataFrame methods are commonly chained to build a clean transformation pipeline (转换管道).

```python
import pandas as pd

df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie", "David"],
    "age":  [25, 30, 35, 40],
    "score":[88, 72, 95, 60]
})

# Method chaining on DataFrame (DataFrame 方法链)
result = (df
          .query("age > 28")
          .sort_values("score", ascending=False)
          .reset_index(drop=True)
          .head(2))

print(result)
# Output:
#       name  age  score
# 0  Charlie   35     95
# 1      Bob   30     72
```

<br>

## 5. functools Chain (functools 链)

`itertools.chain` and `functools.reduce` are functional-style (函数式) tools that chain iterables (可迭代对象) or operations together.

```python
from itertools import chain
from functools import reduce

# itertools.chain: merge iterables (合并可迭代对象)
combined = list(chain([1, 2], [3, 4], [5, 6]))
print(combined)  # Output: [1, 2, 3, 4, 5, 6]

# functools.reduce: chain operations (链式归约操作)
total = reduce(lambda x, y: x + y, [1, 2, 3, 4, 5])
print(total)  # Output: 15
```

<br>

## 6. Key Points (关键要点)

The core idea of Python chaining is **returning a usable object** (返回可用对象) from each step so the next call can continue the pipeline (流水线), improving readability (可读性) and reducing intermediate variables (中间变量).

| Type (类型)        | Mechanism (机制)                | Example                    |
| ------------------ | ------------------------------- | -------------------------- |
| Chained Comparison | Implicit `and` (隐式逻辑与)     | `1 < x < 10`               |
| Method Chaining    | Each method returns object      | `s.strip().lower()`        |
| Custom Chain       | `return self`                   | `obj.a().b().c()`          |
| Pandas Chain       | DataFrame returns new DataFrame | `df.query().sort_values()` |
| Functional Chain   | `chain` / `reduce`              | `chain(a, b, c)`           |

<br> <br>
