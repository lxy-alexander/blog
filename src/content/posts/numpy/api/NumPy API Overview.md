---
title: "NumPy API Overview"
published: 2026-04-27
description: "NumPy API Overview"
image: ""
tags: ["numpy","api","NumPy API Overview"]
category: numpy / api
draft: false
lang: ""
createdAt: "2026-04-27T17:25:49.150.427409317Z"
---

# NumPy API Overview

NumPy (数值计算库) is the core Python library (核心库) for ndarray (多维数组) computing, and most interview questions reduce to creation (创建), indexing (索引), vectorization (向量化), and linear algebra (线性代数).

## 1. API Map

The NumPy API (接口) is easier to remember when you group it by data lifecycle (数据生命周期) instead of memorizing isolated functions (函数).

| Category (类别) | Core APIs (核心接口) | What to Remember (记忆点) |
| --- | --- | --- |
| Array Creation (数组创建) | `np.array`, `np.zeros`, `np.ones`, `np.arange`, `np.linspace` | Build the right shape (形状) first. |
| Array Operations (数组操作) | `reshape`, `transpose`, `concatenate`, `split`, `flatten` | Most transforms (变换) change layout, not meaning. |
| Math Operations (数学运算) | `sum`, `mean`, `max`, `min`, `sqrt`, `exp` | Vectorized code (向量化代码) is faster and cleaner than Python loops. |
| Logical Operations (逻辑运算) | `>`, `<`, `==`, `np.any`, `np.all`, `np.where` | Boolean masks (布尔掩码) are the key filtering tool. |
| Statistical Analysis (统计分析) | `std`, `var`, `percentile`, `argmax`, `argmin` | Statistics (统计量) summarize arrays quickly. |
| Random Sampling (随机采样) | `np.random.rand`, `randn`, `randint`, `choice`, `seed` | Reproducibility (可复现性) matters in experiments. |
| Linear Algebra (线性代数) | `@`, `np.dot`, `np.linalg.inv`, `solve`, `eig`, `svd` | Use matrix APIs (矩阵接口) instead of manual loops. |
| Set Operations (集合运算) | `unique`, `intersect1d`, `union1d`, `setdiff1d` | Useful for deduplication (去重) and comparison (比较). |
| Input and Output (输入输出) | `save`, `load`, `savetxt`, `loadtxt` | Persist arrays (持久化数组) with the right file format (文件格式). |

<br>

## 2. Array Creation and Shape

Array creation (数组创建) is the first interview checkpoint because shape (形状) mistakes usually break everything downstream (下游流程).

```python
import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.zeros((2, 3))
c = np.arange(0, 6).reshape(2, 3)

print(a.shape)  # output: (2, 2)
print(b.shape)  # output: (2, 3)
print(c)        # output: [[0 1 2]
                #         [3 4 5]]
```

<br>

## 3. Indexing and Array Operations

Indexing (索引), slicing (切片), and reshaping (重塑) are the fastest way to prove you understand how ndarray (多维数组) data is organized.

```python
import numpy as np

a = np.arange(1, 7).reshape(2, 3)

print(a[0, 1])       # output: 2
print(a[:, 1])       # output: [2 5]
print(a.T)           # output: [[1 4]
                     #         [2 5]
                     #         [3 6]]
print(a.flatten())   # output: [1 2 3 4 5 6]
```

<br>

## 4. Vectorized Math

Vectorization (向量化) means NumPy (数值计算库) runs element-wise operations (逐元素运算) in optimized native code (原生代码), which is why it beats Python loops.

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

print(a + b)          # output: [11 22 33]
print(a * 2)          # output: [2 4 6]
print(np.sqrt(b))     # output: [3.16227766 4.47213595 5.47722558]
print(np.sum(a))      # output: 6
```

<br>

## 5. Logical Operations and Filtering

Boolean masking (布尔掩码) is the NumPy pattern (模式) for filtering data without writing explicit control flow (显式控制流).

```python
import numpy as np

a = np.array([3, 8, 1, 9, 5])
mask = a > 4

print(mask)                # output: [False  True False  True  True]
print(a[mask])             # output: [8 9 5]
print(np.where(a > 4, 1, 0))  # output: [0 1 0 1 1]
print(np.all(a > 0))       # output: True
```

<br>

## 6. Statistics and Random Sampling

Statistics (统计分析) and random sampling (随机采样) are common together because experiments (实验) need both measurement (度量) and reproducibility (可复现性).

```python
import numpy as np

np.random.seed(7)
a = np.array([2, 4, 6, 8])
samples = np.random.randint(0, 10, size=5)

print(np.mean(a))      # output: 5.0
print(np.std(a))       # output: 2.23606797749979
print(np.argmax(a))    # output: 3
print(samples)         # output: [4 9 6 3 3]
```

<br>

## 7. Linear Algebra

Linear algebra (线性代数) APIs matter because matrix multiplication (矩阵乘法) and equation solving (方程求解) appear often in ML (机器学习) and interviews.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
b = np.array([5, 11])

print(A @ A)                     # output: [[ 7 10]
                                 #         [15 22]]
print(np.linalg.det(A))          # output: -2.0000000000000004
print(np.linalg.solve(A, b))     # output: [1. 2.]
```

<br>

## 8. Set Operations and I/O

Set operations (集合运算) clean data (数据清洗), while I/O (输入输出) APIs move arrays between memory (内存) and storage (存储).

```python
import numpy as np
import tempfile
import os

a = np.array([1, 2, 2, 3])
b = np.array([2, 3, 4])

print(np.unique(a))            # output: [1 2 3]
print(np.intersect1d(a, b))    # output: [2 3]

tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
tmp.close()
np.save(tmp.name, a)
loaded = np.load(tmp.name)
print(loaded)                  # output: [1 2 2 3]
os.unlink(tmp.name)
```

<br>

## 9. Interview Summary

If you remember shape (形状), axis (轴), broadcasting (广播), masking (掩码), and `linalg` (线性代数模块), you already cover most practical NumPy (数值计算库) questions.

1. Create arrays correctly with `array`, `zeros`, `ones`, and `arange`.
2. Manipulate structure with `reshape`, `transpose`, and `concatenate`.
3. Prefer vectorized math over Python loops for performance (性能).
4. Filter data with boolean masks and `where`.
5. Use `mean`, `std`, and `argmax` for quick analysis (分析).
6. Use `@` and `np.linalg.solve()` for matrix problems (矩阵问题).

<br>

<br>
