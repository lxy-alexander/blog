---
title: "bisect left"
published: 2026-01-26
description: "bisect left"
image: ""
tags: ["python","bisect left"]
category: python
draft: false
lang: ""

---

## 速记总结

### 找 “第一个 >= x”

```
i = bisect_left(a, x)
if i < len(a):
    ans = a[i]   # 第一个 >= x
```

------



### 找 “最后一个 < x”

```
i = bisect_left(a, x)
if i > 0:
    ans = a[i-1]  # 最后一个 < x
```

（在 Heaters 里就是左边最近的 heater）

------



### 判断 x 是否存在于数组中

```
i = bisect_left(a, x)
exists = (i < len(a) and a[i] == x)
```

------



### 找 “x 应该插入的位置”

```
pos = bisect_left(a, x)
```

插入到 `pos` 可以保持有序（并且插到同值最左边）

------



### 找 “小于 x 的元素个数”

```
count = bisect_left(a, x)
```

因为 `i` 左边全是 `< x` 的。

------



### 找 “>= x 的元素个数”

```
count = len(a) - bisect_left(a, x)
```

------

## 

## Example

r = len(nums) - 1 # 记得减1

```python
import bisect as b

nums = [1, 2, 2, 3]

# 第一个 >= x 的位置
a = b.bisect_left(nums, 2)
print(a) # 1

def find_bisect_left(nums, k):
    l = 0
    r = len(nums) - 1 # 记得减1
    while l <= r:
        m = (l + r) // 2
        if nums[m] >= k:
            r = m - 1
        else:
            l = m + 1
    return l

print(find_bisect_left(nums, 2))


# 第一个 > x 的位置
a = b.bisect(nums, 2)
print(a) # 3
a = b.bisect_right(nums, 2)
print(a) # 4

def find_bisect_right(nums, k):
    l = 0
    r = len(nums) - 1
    while l <= r:
        m = (l + r) // 2
        if nums[m] > k:
            r = m - 1
        else:
            l = m + 1
    return l

print(find_bisect_right(nums, 2))
```

