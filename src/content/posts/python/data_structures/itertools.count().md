---
title: "itertools.count()"
published: 2026-03-20
description: "itertools.count()"
image: ""
tags: ["python","data_structures","itertools.count()"]
category: python / data_structures
draft: false
lang: ""
---

# I. `itertools.count()`

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <span style="color:#E8600A;font-weight:700">`itertools.count()` (计数迭代器)</span> creates an <span style="color:#E8600A;font-weight:700">infinite iterator (无限迭代器)</span> that generates evenly spaced numbers（等间距数字） starting from a specified value. </div>

## 1. Basic Usage

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">`itertools.count()` (计数迭代器)</span> generates an infinite arithmetic progression. <span style="color:#2980B9;font-weight:700">Use it when you need an endless sequence of numbers</span> for generating IDs, indices, or combining with other iterators. <span style="color:#C0392B;font-weight:700">Warning: Always provide a termination condition when iterating over count()</span> to avoid infinite loops.</div>

### 1) Function Parameters

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">`itertools.count(start=0, step=1)` (起始值, 步长)</span> accepts two numeric parameters. <span style="color:#2980B9;font-weight:700">`start` (起始值)</span> defines the first value in the sequence, while <span style="color:#2980B9;font-weight:700">`step` (步长)</span> determines the increment between consecutive values. <span style="color:#E8600A;font-weight:700">Both parameters can be integers, floats, or any numeric type</span> that supports addition.</div>

```python
import itertools

# Default: start=0, step=1
counter = itertools.count()
print(next(counter))  # 0
print(next(counter))  # 1

# Custom start and step
counter = itertools.count(start=5, step=3)
print(next(counter))  # 5
print(next(counter))  # 8
print(next(counter))  # 11

# Using float step
counter = itertools.count(start=1.0, step=0.5)
print([next(counter) for _ in range(3)])  # [1.0, 1.5, 2.0]
```

<div style="background:#F5F5F5;border-left:4px solid #C0392B;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85;font-size:0.80em"><span style="color:#C0392B;font-weight:700">Note: </span> Using floating-point steps may lead to <span style="color:#C0392B;font-weight:700">precision accumulation errors</span> over many iterations. Consider using integers and dividing when precise decimal values are needed.</div>

## 2. Practical Applications

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">`itertools.count()` (计数迭代器)</span> shines in scenarios requiring automatic indexing or sequence generation. <span style="color:#2980B9;font-weight:700">Common use cases include adding line numbers to data, generating unique IDs, and creating paginated sequences.</span> Its infinite nature makes it particularly useful when the iteration length is determined by another iterable.</div>

### 1) Adding Indices to Data

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> Combine <span style="color:#E8600A;font-weight:700">`count()`</span> with <span style="color:#2980B9;font-weight:700">`zip()`</span> to <span style="color:#E8600A;font-weight:700">automatically number items (自动编号项目)</span> in any iterable. <span style="color:#2980B9;font-weight:700">This pattern is memory-efficient because it generates indices on-the-fly</span> rather than storing them in a list.</div>

```python
import itertools

# Adding line numbers to text lines
lines = ["First line", "Second line", "Third line"]
numbered_lines = zip(itertools.count(1), lines)

for num, line in numbered_lines:
    print(f"{num}: {line}")
# Output:
# 1: First line
# 2: Second line
# 3: Third line

# Creating dictionary with auto-generated keys
names = ["Alice", "Bob", "Charlie"]
user_dict = dict(zip(itertools.count(100), names))
print(user_dict)  # {100: 'Alice', 101: 'Bob', 102: 'Charlie'}
```

### 2) Generating Infinite Sequences

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> Use <span style="color:#E8600A;font-weight:700">`count()`</span> with <span style="color:#2980B9;font-weight:700">`itertools.islice()`</span> to <span style="color:#E8600A;font-weight:700">generate finite slices of arithmetic sequences (生成有限段的算术序列)</span>. <span style="color:#2980B9;font-weight:700">This approach is ideal for generating test data, mathematical sequences, or pagination</span> where you need predictable, spaced values.</div>

```python
import itertools

# First 5 multiples of 10
multiples_of_10 = itertools.islice(itertools.count(10, 10), 5)
print(list(multiples_of_10))  # [10, 20, 30, 40, 50]

# Skip first 3, then take 4 numbers
skip_take = itertools.islice(itertools.count(100, -1), 3, 7)
print(list(skip_take))  # [97, 96, 95, 94]

# Generating powers of 2 using indices
powers_of_2 = (2 ** i for i in itertools.islice(itertools.count(), 6))
print(list(powers_of_2))  # [1, 2, 4, 8, 16, 32]
```

| <span style="color:#2980B9;font-weight:700">Pattern (模式)</span> | <span style="color:#2980B9;font-weight:700">Code Example (代码示例)</span> | <span style="color:#2980B9;font-weight:700">Use Case (使用场景)</span> |
| :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| <span style="color:#2980B9;font-weight:700">1-based indexing</span> | `zip(itertools.count(1), data)`                              | Displaying numbered lists, generating SQL IDs                |
| <span style="color:#E8600A;font-weight:700">Staggered steps</span> | `itertools.islice(itertools.count(0, 5), 10)`                | Creating evenly spaced time intervals, pagination offsets    |
| <span style="color:#E8600A;font-weight:700">Descending sequences</span> | `itertools.islice(itertools.count(100, -1), 5)`              | Generating countdowns, reverse numbering                     |

## 3. Performance Comparison

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">`itertools.count()` (计数迭代器)</span> is implemented in C, making it <span style="color:#E8600A;font-weight:700">significantly faster than manual Python counter loops (比手动Python计数循环快得多)</span>. <span style="color:#2980B9;font-weight:700">Choose count() when you need infinite sequences or functional composition</span>, and use `range()` for simple finite sequences.</div>

```python
import itertools
import time

# Manual counter (Python loop)
def manual_counter(n):
    result = []
    i = 0
    while i < n:
        result.append(i)
        i += 1
    return result

# Count with islice (C implementation)
def count_islice(n):
    return list(itertools.islice(itertools.count(), n))

# Range (most optimized for finite sequences)
def range_approach(n):
    return list(range(n))

# n = 10,000,000
# manual_counter: ~0.85s
# count_islice: ~0.42s
# range_approach: ~0.28s
```

| <span style="color:#2980B9;font-weight:700">Method (方法)</span> | <span style="color:#2980B9;font-weight:700">Implementation (实现)</span> | <span style="color:#2980B9;font-weight:700">Best For (最佳场景)</span> | <span style="color:#C0392B;font-weight:700">Limitation (限制)</span> |
| :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| <span style="color:#2980B9;font-weight:700">Manual counter</span> | Python loop                                                  | Simple educational examples                                  | Slow for large iterations                                    |
| <span style="color:#E8600A;font-weight:700">`itertools.count()`</span> | C implementation                                             | Infinite sequences, functional pipelines                     | Requires `islice` for finite use                             |
| <span style="color:#E8600A;font-weight:700">`enumerate()`</span> | Built-in function                                            | Indexing existing iterables                                  | Fixed start=0, no custom step                                |
| <span style="color:#E8600A;font-weight:700">`range()`</span> | C implementation                                             | Simple finite sequences                                      | Cannot be infinite                                           |

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> <span style="color:#E8600A;font-weight:700">`itertools.count()`</span> is your <span style="color:#2980B9;font-weight:700">go-to C-level tool for infinite arithmetic sequences</span> — pair it with <span style="color:#2980B9;font-weight:700">`zip()` for auto-indexing</span> or <span style="color:#2980B9;font-weight:700">`islice()` for finite slices</span>, <span style="color:#C0392B;font-weight:700">but never iterate directly without a termination condition</span>.</div>
