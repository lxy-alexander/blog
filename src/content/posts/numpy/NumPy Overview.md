---
title: "NumPy Overview"
published: 2026-04-15
description: "NumPy Overview"
image: ""
tags: ["numpy","NumPy Overview"]
category: numpy
draft: false
lang: ""
---

# I. NumPy Overview(数值计算库)

==NumPy is a fundamental library for numerical computing (数值计算) in Python==. It provides efficient data structures and operations for handling large-scale numerical data.

------

## 1. What is NumPy (是什么)

NumPy is a library that provides the `ndarray (多维数组)` object for storing and manipulating numerical data efficiently.

### 1) Core Data Structure

-   `ndarray (多维数组)`
    A homogeneous (同质的) multi-dimensional array.

```python
import numpy as np

x = np.array([[1, 2], [3, 4]])
print(x)
print(type(x))
print(x.shape)
```

------

## 2. Why Learn NumPy (为什么需要学习)

### 1) High Performance (高性能)

NumPy operations are implemented in C (底层C实现), making them much faster than Python loops.

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b)  # element-wise addition (逐元素加法)
```

------

### 2) Vectorization (向量化)

Vectorization avoids explicit loops (避免显式循环), improving speed and readability.

```python
import numpy as np

a = np.array([1, 2, 3])
print(a * 2)  # vectorized operation (向量化操作)
```

------

### 3) Foundation of AI Libraries (AI基础)

Many libraries depend on NumPy:

-   PyTorch (深度学习框架)
-   TensorFlow (深度学习框架)
-   SciPy (科学计算库)

------

## 3. Comparison 

### 1) NumPy (数值计算库)

-   Focus: numerical computing (数值计算)
-   Core object: `ndarray (多维数组)`
-   Mainly used for CPU computation (CPU计算)

```
import numpy as np

x = np.array([1, 2, 3])
print(x * 2)
```

### 2) PyTorch (深度学习框架)

-   Focus: deep learning (深度学习)
-   Core object: `Tensor (张量)`
-   Supports GPU acceleration (GPU加速)
-   Supports automatic differentiation (自动求导)

```
import torch

x = torch.tensor([1, 2, 3], dtype=torch.float32)
print(x * 2)
```

------

## 4. Basic Operations (基础操作)

### 1) Create Arrays (创建数组)

```python
import numpy as np

a = np.zeros((2, 3))   # all zeros (全0)
b = np.ones((2, 3))    # all ones (全1)
c = np.arange(0, 10)   # range (范围)
```

------

### 2) Shape and Reshape (形状操作)

```python
import numpy as np

x = np.array([1, 2, 3, 4])
y = x.reshape(2, 2)

print(y)
```

------

### 3) Indexing (索引)

```python
import numpy as np

x = np.array([[1, 2], [3, 4]])

print(x[0, 1])  # access element (访问元素)
```



# II. NumPy Interview Questions 

------

## 1. What is NumPy (是什么)

NumPy is a numerical computing library (数值计算库) in Python that provides the `ndarray (多维数组)` for efficient computation.

------

## 2. What is ndarray (多维数组)

`ndarray` is a homogeneous array (同质数组) that stores elements of the same data type in contiguous memory (连续内存).

------

## 3. Difference between list and ndarray (区别)

`list` allows mixed types (可混合类型), while `ndarray` requires a single type (统一类型) and supports vectorized operations (向量化运算).

------

## 4. What is vectorization (向量化)

Vectorization is performing operations on entire arrays without explicit loops (无需显式循环).

------

## 5. What is broadcasting (广播机制)

Broadcasting allows arrays of different shapes (不同形状) to be operated together automatically.

------

## 6. Why is NumPy faster than Python (为什么更快)

Because it uses C implementation (底层C实现), contiguous memory (连续内存), and vectorization (向量化).

------

## 7. How to create an array (创建数组)

Use functions like `np.array()`, `np.zeros()`, `np.ones()`, `np.arange()`.

------

## 8. What is shape (形状)

`shape` describes the dimensions (维度) of an array.

------

## 9. What is reshape (重塑)

`reshape` changes the shape (改变形状) of an array without changing its data.

------

## 10. Difference between copy and view (拷贝 vs 视图)

`copy` creates new memory (新内存), while `view` shares memory (共享内存).

------

## 11. What is slicing (切片)

Slicing extracts a subset (子数组) of an array.

------

## 12. What is dtype (数据类型)

`dtype` defines the type (数据类型) of elements in an array.

------

## 13. What is axis (轴)

`axis` specifies the direction (方向) along which operations are performed.

------

## 14. What is flatten (展平)

`flatten` converts a multi-dimensional array (多维数组) into one dimension.

------

## 15. Difference between arange and linspace (区别)

`arange` uses step size (步长), while `linspace` uses number of points (点的数量).

------

## 16. What is indexing (索引)

Indexing accesses specific elements (访问元素) using positions.

------

## 17. What is boolean indexing (布尔索引)

Boolean indexing selects elements based on conditions (条件筛选).

------

## 18. What is aggregation (聚合)

Aggregation performs operations like `sum`, `mean` on arrays.

------

## 19. What is matrix multiplication (矩阵乘法)

Matrix multiplication follows:

$$
C = A \cdot B
$$

using `np.dot()` or `@`.

------

## 20. What is the main purpose of NumPy (核心作用)

NumPy is used for efficient numerical computation (高效数值计算) and is the foundation of scientific computing (科学计算基础).
