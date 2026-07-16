---
title: "List"
published: 2026-06-09
description: "List"
image: ""
tags: ["python","List"]
category: python
draft: false
lang: ""
createdAt: "2026-06-10T00:08:23.826.468115170Z"
---

# Python List

A Python list is an ordered and changeable container (容器) that can store multiple values. list is built in type, List is typing hint. 

An ordered and changeable collection. It allows duplicate elements.
列表是一种有顺序、可修改、允许重复元素的集合。

| Operation          | API                   | Example               | Time Complexity  |
| ------------------ | --------------------- | --------------------- | ---------------- |
| Create             | `[]`, `list()`        | `nums = [1, 2, 3]`    | `O(n)`           |
| Access by index    | `list[index]`         | `nums[0]`             | `O(1)`           |
| Modify by index    | `list[index] = value` | `nums[0] = 100`       | `O(1)`           |
| Add to end         | `append(x)`           | `nums.append(4)`      | `O(1)` amortized |
| Add multiple items | `extend(iterable)`    | `nums.extend([5, 6])` | `O(k)`           |
| Insert at index    | `insert(index, x)`    | `nums.insert(0, 99)`  | `O(n)`           |
| Remove by value    | `remove(x)`           | `nums.remove(2)`      | `O(n)`           |
| Remove by index    | `pop(index)`          | `nums.pop(0)`         | `O(n)`           |
| Remove last item   | `pop()`               | `nums.pop()`          | `O(1)`           |
| Check existence    | `x in list`           | `3 in nums`           | `O(n)`           |
| Get length         | `len(list)`           | `len(nums)`           | `O(1)`           |
| Sort               | `sort()`              | `nums.sort()`         | `O(n log n)`     |
| Reverse            | `reverse()`           | `nums.reverse()`      | `O(n)`           |

```python
nums = [1, 2, 3]

nums.append(4)
print(nums)
# Output: [1, 2, 3, 4]

nums[0] = 100
print(nums)
# Output: [100, 2, 3, 4]

nums.remove(2)
print(nums)
# Output: [100, 3, 4]

nums.pop()
print(nums)
# Output: [100, 3]
```



## 1. Create a List

Use square brackets to create a list.

```python
from typing import List

a: List[int] = [1, 2, 3]

arr = [1, 2, 3, 4]

print(arr)  # Output: [1, 2, 3, 4]
```

## 2. Access Elements

Use index (索引) to access elements, and the index starts from `0`.

```python
arr = [10, 20, 30]

print(arr[0])   # Output: 10
print(arr[1])   # Output: 20
print(arr[-1])  # Output: 30
```

## 3. Modify Elements

Lists are mutable (可变的), so you can change their elements.

```python
arr = [10, 20, 30]

arr[1] = 99

print(arr)  # Output: [10, 99, 30]
```

## 4. Add Elements

Use `append()` to add one element to the end of a list.

```python
arr = [1, 2, 3]

arr.append(4)

print(arr)  # Output: [1, 2, 3, 4]
```

## 5. Remove Elements

Use `pop()` to remove and return the last element by default.

```python
arr = [1, 2, 3]

x = arr.pop()

print(x)    # Output: 3
print(arr)  # Output: [1, 2]
```

## 6. List Length

Use `len()` to get the number of elements.

```python
arr = [5, 6, 7, 8]

print(len(arr))  # Output: 4
```

## 7. All-Zero List

Use multiplication to create a list filled with zeros.

```python
n = 5
arr = [0] * n

print(arr)  # Output: [0, 0, 0, 0, 0]
```

## 8. Loop Through a List

Use a `for` loop to visit each element.

```python
arr = [10, 20, 30]

for x in arr:
    print(x)

# Output:
# 10
# 20
# 30
```

## 9. List as Stack

A list can be used as a stack (栈), which follows LIFO (后进先出).

```python
stack = []

stack.append(1)
stack.append(2)
stack.append(3)

print(stack.pop())  # Output: 3
print(stack.pop())  # Output: 2
print(stack)        # Output: [1]
```

## 10. 2D List

Use list comprehension (列表推导式) to create a 2D list safely.

```python
rows = 2
cols = 3

matrix = [[0] * cols for _ in range(rows)]

print(matrix)  # Output: [[0, 0, 0], [0, 0, 0]]
```

## 11. Common Operations

Lists support search, slice (切片), and membership check (成员检查).

```python
arr = [10, 20, 30, 40]

print(20 in arr)     # Output: True
print(arr[1:3])      # Output: [20, 30]
print(arr.index(30)) # Output: 2
```

## 12. Key Summary

A list is the most common Python data structure (数据结构) for storing and changing ordered data.

```python
arr = [1, 2, 3]

arr.append(4)
arr.pop()

print(arr)  # Output: [1, 2, 3]
```
