---
title: "Python & C++"
published: 2026-10-06
description: "Python & C++"
image: ""
tags: ["llm_inference","Python & C++"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-10-06T13:32:58.496.803128587Z"
---

# Python

## sort & sorted

1）`sorted()` 是通用排序函数

```
nums = [3, 1, 2]
result = sorted(nums) 
result = [1, 2, 3]   # nums 不变
```

2）`.sort()` 是 `list` 自己的方法

```
nums = [3, 1, 2]
nums.sort()
nums = [1, 2, 3]  # 直接修改原 list
```

3）`sorted()` 可以用于很多类型

```
sorted([3, 1, 2])       # list
sorted((3, 1, 2))       # tuple
sorted({3, 1, 2})       # set
sorted("cab")           # string
```

返回结果都是 `list`。

4）哪些常见类型有 `.sort()`

```
list      有 .sort()
tuple     没有
set       没有
dict      没有
Counter   没有
```

所以除了 `list`，一般用 `sorted()`。

5）默认是升序

```
nums = [3, 1, 2]sorted(nums)# [1, 2, 3]
```

从小到大。

6）降序用 `reverse=True`

```
nums = [3, 1, 2]sorted(nums, reverse=True)# [3, 2, 1]
```

`.sort()` 也一样支持。

7）`key` 用来指定“按什么排序”

```
arr = [(3, 2), (4, 1)]sorted(arr, key=lambda x: x[1])# 按第二个元素排序
```

这里 `x[1]` 是 `2、1`。

8）多条件排序

```
sorted(    counts.items(),    key=lambda x: (-x[1], x[0]))
```

先按 `x[1]` 降序，再按 `x[0]` 升序。

9）`dict` / `Counter` 直接 sorted

```
sorted(counts)
```

排序的是 key。

```
sorted(counts.items())
```

排序的是 `(key, value)`。

10）最重要的区别

```
list.sort()
→ 修改原 list
→ 返回 None

sorted(...)
→ 不修改原数据
→ 返回新 list
```



## Collections

# Python Collections 常用 API

## 1）list

### 初始化

```python
arr = []                    # 空列表

arr = [1, 2, 3]             # 指定元素

arr = [0] * n               # 创建 n 个 0

arr = list(iterable)        # 从 iterable 创建，O(n)
```

### 添加

```python
arr.append(x)               # 末尾添加，均摊 O(1)

arr.extend(nums)            # 添加多个元素，O(k)

arr.insert(i, x)            # 在位置 i 插入，O(n)
```

### 删除

```python
arr.pop()                   # 删除并返回最后一个元素，O(1)

arr.pop(i)                  # 删除并返回位置 i，O(n)

arr.remove(x)               # 删除第一个 x，O(n)

arr.clear()                 # 清空，O(n)
```

### 访问 / 查找

```python
arr[i]                      # 下标访问，O(1)

arr[i] = x                  # 修改位置 i，O(1)

x in arr                    # 判断是否存在，O(n)

arr.index(x)                # 查找 x 的位置，O(n)

arr.count(x)                # 统计 x 次数，O(n)

len(arr)                    # 长度，O(1)
```

### 排序

```python
arr.sort()                  # 原地升序，O(n log n)

arr.sort(reverse=True)      # 原地降序，O(n log n)

arr.sort(key=lambda x: ...) # 自定义排序，O(n log n)

arr.reverse()               # 原地反转，O(n)
```

---

## 2）set

### 初始化

```python
s = set()                   # 空集合

s = {1, 2, 3}               # 指定元素

s = set(iterable)           # 从 iterable 创建，平均 O(n)
```

### 添加

```python
s.add(x)                    # 添加一个元素，平均 O(1)

s.update(nums)              # 添加多个元素，平均 O(k)
```

### 删除

```python
s.remove(x)                 # 删除，不存在会报错，平均 O(1)

s.discard(x)                # 删除，不存在不报错，平均 O(1)

s.pop()                     # 删除任意元素，平均 O(1)

s.clear()                   # 清空，O(n)
```

### 查找

```python
x in s                      # 判断存在，平均 O(1)

x not in s                  # 判断不存在，平均 O(1)

len(s)                      # 元素数量，O(1)
```

### 集合运算

```python
a | b                       # 并集

a & b                       # 交集

a - b                       # 差集

a ^ b                       # 对称差集

a <= b                      # a 是否为 b 的子集
```

`set` 不支持下标访问：

```python
s[0]                        # 错误
```

---

## 3）dict

### 初始化

```python
d = {}                      # 空字典

d = dict()                  # 空字典

d = {"a": 1, "b": 2}        # 指定 key-value
```

### 添加 / 修改

```python
d[key] = value              # 添加或修改，平均 O(1)

d.update(other)             # 批量添加或修改，平均 O(k)

d.setdefault(key, default)  # key 不存在时设置默认值
```

### 删除

```python
d.pop(key)                  # 删除并返回 value，平均 O(1)

d.pop(key, default)         # 不存在时返回 default

del d[key]                  # 删除 key，平均 O(1)

d.clear()                   # 清空，O(n)
```

### 访问 / 查找

```python
d[key]                      # 获取 value，平均 O(1)

d.get(key)                  # 安全获取，平均 O(1)

d.get(key, default)         # 不存在返回 default

key in d                    # 判断 key 是否存在，平均 O(1)

len(d)                      # key 数量，O(1)
```

### 遍历

```python
d.keys()                    # 所有 key

d.values()                  # 所有 value

d.items()                   # 所有 (key, value)
```

```python
for key in d:
    pass
```

```python
for key, value in d.items():
    pass
```

---

## 4）tuple

### 初始化

```python
t = ()                      # 空 tuple

t = (1, 2, 3)               # 多个元素

t = (1,)                    # 单元素 tuple

t = tuple(iterable)         # 从 iterable 创建
```

### 访问 / 查找

```python
t[i]                        # 下标访问，O(1)

t[a:b]                      # 切片，O(k)

x in t                      # 查找，O(n)

t.index(x)                  # 查找位置，O(n)

t.count(x)                  # 统计次数，O(n)

len(t)                      # 长度，O(1)
```

`tuple` 不可修改：

```python
t[0] = 10                   # 错误
```

---

## 5）deque

需要导入：

```python
from collections import deque
```

### 初始化

```python
q = deque()                 # 空 deque

q = deque([1, 2, 3])        # 指定元素
```

### 添加

```python
q.append(x)                 # 右边添加，O(1)

q.appendleft(x)             # 左边添加，O(1)

q.extend(nums)              # 右边添加多个，O(k)

q.extendleft(nums)          # 左边添加多个，O(k)
```

### 删除

```python
q.pop()                     # 右边删除并返回，O(1)

q.popleft()                 # 左边删除并返回，O(1)

q.clear()                   # 清空，O(n)
```

### 访问

```python
q[0]                        # 第一个元素，O(1)

q[-1]                       # 最后一个元素，O(1)

q[i]                        # 中间随机访问，O(n)

len(q)                      # 长度，O(1)
```

### BFS

```python
q = deque([start])

while q:
    cur = q.popleft()
```

---

## 6）Counter

需要导入：

```python
from collections import Counter
```

### 初始化

```python
c = Counter()               # 空 Counter

c = Counter([1, 1, 2])      # 自动统计

c = Counter("hello")        # 统计字符
```

### 计数

```python
c[x] += 1                   # 计数 +1，平均 O(1)

c[x] -= 1                   # 计数 -1，平均 O(1)

c[x]                        # 获取次数，平均 O(1)

x in c                      # 判断 key，平均 O(1)
```

### 删除 / 遍历

```python
del c[x]                    # 删除 key，平均 O(1)

c.pop(x)                    # 删除并返回次数，平均 O(1)

c.items()                   # (元素, 次数)

c.keys()                    # 所有元素

c.values()                  # 所有次数
```

### 高频

```python
c.most_common()             # 按次数从高到低

c.most_common(k)            # 次数最多的前 k 个
```

---

## 7）defaultdict

需要导入：

```python
from collections import defaultdict
```

### 计数

```python
counts = defaultdict(int)

counts[x] += 1
```

### list

```python
graph = defaultdict(list)

graph[a].append(b)
```

### set

```python
graph = defaultdict(set)

graph[a].add(b)
```

### 常用默认类型

```python
defaultdict(int)            # 默认 0

defaultdict(list)           # 默认 []

defaultdict(set)            # 默认 set()
```

---

## 8）heapq

需要导入：

```python
import heapq
```

### 初始化

```python
heap = []
```

### 添加 / 删除

```python
heapq.heappush(heap, x)     # 插入，O(log n)

heapq.heappop(heap)         # 删除并返回最小值，O(log n)

heap[0]                     # 查看最小值，O(1)
```

### 其他

```python
heapq.heapify(arr)          # 原地建堆，O(n)

heapq.heappushpop(heap, x)  # push 后 pop

heapq.heapreplace(heap, x)  # pop 后 push

heapq.nsmallest(k, arr)     # 最小 k 个

heapq.nlargest(k, arr)      # 最大 k 个
```

Python `heapq` 默认是 Min Heap。

---

## 9）sorted

`sorted()` 是 Python 内置函数，不需要 import。

```python
sorted(arr)                         # 升序，O(n log n)

sorted(arr, reverse=True)           # 降序，O(n log n)

sorted(arr, key=lambda x: x[1])     # 自定义排序

sorted(d)                           # 按 dict 的 key 排序

sorted(d.items())                   # 排序 (key, value)

sorted(s)                           # set 排序，返回 list
```

### 多条件排序

```python
sorted(
    arr,
    key=lambda x: (-x[1], x[0])
)
```

表示：

```text
先按照 x[1] 降序
再按照 x[0] 升序
```

### `sort()` vs `sorted()`

```python
arr.sort()                  # 修改原 list，返回 None

result = sorted(arr)        # 原数据不变，返回新 list
```

只有 `list` 有 `.sort()`。

---

## 10）str

### 初始化

```python
s = ""

s = "hello"
```

### 访问 / 查找

```python
s[i]                        # 字符访问，O(1)

len(s)                      # 长度，O(1)

sub in s                    # 判断子串

s.find(sub)                 # 查找子串位置

s.count(sub)                # 统计次数
```

### 转换

```python
s.lower()                   # 小写

s.upper()                   # 大写

s.strip()                   # 去掉两端空白

s.replace(a, b)             # 替换

s.split(",")                # 分割，返回 list

",".join(arr)               # 拼接字符串
```

### 切片

```python
s[a:b]                      # 子字符串，O(k)

s[::-1]                     # 反转，O(n)
```

字符串不可修改：

```python
s[0] = "a"                  # 错误
```

---

## 11）常用 Collection 对应关系

| 需求           | Python        |
| -------------- | ------------- |
| 动态数组       | `list`        |
| 哈希集合       | `set`         |
| 哈希表         | `dict`        |
| 不可变序列     | `tuple`       |
| 双端队列 / BFS | `deque`       |
| 计数器         | `Counter`     |
| 默认值字典     | `defaultdict` |
| 最小堆         | `heapq`       |
| 通用排序       | `sorted()`    |

---

## 12）核心复杂度

| Collection | 操作               | 复杂度       |
| ---------- | ------------------ | ------------ |
| `list`     | `arr[i]`           | `O(1)`       |
| `list`     | `append()`         | 均摊 `O(1)`  |
| `list`     | `pop()`            | `O(1)`       |
| `list`     | `x in arr`         | `O(n)`       |
| `set`      | add / remove / in  | 平均 `O(1)`  |
| `dict`     | get / set / delete | 平均 `O(1)`  |
| `deque`    | 两端添加 / 删除    | `O(1)`       |
| `heapq`    | push / pop         | `O(log n)`   |
| `heapq`    | 查看堆顶           | `O(1)`       |
| `sorted`   | 排序               | `O(n log n)` |

