---
title: "Counter()"
published: 2026-03-01
description: "Counter()"
image: ""
tags: ["python","Counter()"]
category: python
draft: false
lang: ""
---



`Counter` 的 **输入是一个“可迭代对象（iterable）”**。

也就是说，只要能被 `for` 遍历的东西都可以。Counter 的输入 = 任何可以被 for 遍历的东西（iterable）。





Counter：不存在返回 0，但不创建键

defaultdict：不存在返回默认值，并创建键





------

## 常见输入类型

###  1️⃣ list

```python
Counter([1,2,2,3])
```

------

###  2️⃣ 字符串

```python
Counter("aabcc")
# 统计每个字符
```

------

###  3️⃣ 生成器 generator

A generator is an object that generates data one by one as needed, rather than storing all the data in memory at once. 生成器（generator） 是一种按需一个一个地产生数据的对象，而不是一次性把所有数据都存到内存里。

```
(x for x in range(10**9))
```

这是 **生成器（generator）**，它是**惰性计算（lazy evaluation）**。

为什么不会存 10 亿个数？

它只做了两件事：

1.  记住规则：`x for x in range(10**9)`
2.  记住当前进行到哪一步

它 **不会提前把所有数字算出来存起来**。

```python
Counter(x + y for x in nums1 for y in nums2)
```

------

###  4️⃣ 元组

```python
Counter((1,2,2,3))
```

------

###  5️⃣ 直接传字典

```python
Counter({1:3, 2:5})
```

表示：

-   1 出现 3 次
-   2 出现 5 次

