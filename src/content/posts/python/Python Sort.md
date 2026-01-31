---
title: "Python Sort"
published: 2026-01-31
description: "Python Sort"
image: ""
tags: ["python","Python Sort"]
category: python
draft: false
lang: ""

---

## 1) Sort a List (In-Place)

```python
a = [3, 1, 2]
a.sort()
print(a)  # [1, 2, 3]
```



## 2) Return a New Sorted List (Do Not Modify the Original)

```python
a = [3, 1, 2]
b = sorted(a)
print(b)  # [1, 2, 3]
print(a)  # [3, 1, 2]
```



## 3) Sort in Descending Order

```python
a.sort(reverse=True)
# or
b = sorted(a, reverse=True)
```



## 4) Sort by Key (Common for Tuples / Objects)

```python
arr = [(1, 5), (2, 3), (3, 4)]
arr.sort(key=lambda x: x[1])
print(arr)  # sort by the second element

```



------

## 5) Sort by multiple key and value 

### 1.Ascending, Ascending

```python
arr.sort(key=lambda x: (x[0], x[1]))
```

Example:

```python
arr = [(2, 0), (1, 4), (2, 1), (1, 3)]
arr.sort(key=lambda x: (x[0], x[1]))
print(arr)
# [(1, 3), (1, 4), (2, 0), (2, 1)]
```

Meaning:

-   sort by `x[0]` first
-   if `x[0]` is equal, sort by `x[1]`

------

### 2.Ascending + Descending

```python
arr.sort(key=lambda x: (x[0], -x[1]))
```

Example:

```python
# Ascending + Descending (most common in contests)
arr = [(2, 0), (1, 4), (2, 1), (1, 3)]
arr.sort(key=lambda x: (x[0], -x[1]))
print(arr)
# [(1, 4), (1, 3), (2, 1), (2, 0)]
```

------

### 3.All descending (two ways)

#### Method A: reverse=True (global reverse) ==reverse=True==

```python
arr.sort(key=lambda x: (x[0], x[1]), reverse=True)
```

⚠️ This reverses the whole result, not “first asc, second desc”.

#### Method B: negate each key (more flexible)

```python
arr.sort(key=lambda x: (-x[0], -x[1]))
```

------



### 4.key&value sorting for strings

Sort by length descending, then lexicographically ascending

```python
words.sort(key=lambda s: (-len(s), s))
```

Example:

```python
words = ["apple", "bat", "banana", "app"]
words.sort(key=lambda s: (-len(s), s))
print(words)
# ['banana', 'apple', 'app', 'bat']
```

------



### 5.Sort a dictionary / Counter by value, then key

Example: sort by frequency descending, then number ascending

```python
from collections import Counter

# Sort a dictionary / Counter by value, then key
nums = [1,1,1,2,2,3,3,4]
cnt = Counter(nums)
print(cnt)
res = sorted(cnt.items(), key=lambda x: (-x[1], x[0]))
print(res)
# Counter({1: 3, 2: 2, 3: 2, 4: 1})
# [(1, 3), (2, 2), (3, 2), (4, 1)]

nums = {4:1, 1:3, 2:2, 3:2}
print(nums)
res = sorted(nums.items(), key=lambda x: (-x[1], x[0]))
print(res)
# [(1, 3), (2, 2), (3, 2), (4, 1)]
```
