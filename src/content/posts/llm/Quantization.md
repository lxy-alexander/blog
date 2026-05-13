---
title: "Quantization"
published: 2026-05-10
description: "Quantization"
image: ""
tags: ["llm","Quantization"]
category: llm
draft: false
lang: ""
createdAt: "2026-05-11T01:28:11.673.491777550Z"
---

# Quantization

Quantization (量化) is a Model Compression (模型压缩) technique that converts high-precision values into low-precision values to reduce memory usage and improve inference speed.

## 1. Core Idea

Quantization (量化) converts high-precision values (高精度数值), such as FP32 (32位浮点数), into lower-precision values (低精度数值), such as INT8 (8位整数).
$$
q=\text{round}\left(\frac{x}{s}+z\right)
$$

## 2. Scale and Zero Point

Scale (缩放因子) and Zero Point (零点) map floating-point values (浮点数值) to integer values (整数值) and map them back during computation.
$$
x \approx s(q-z)
$$

```
import numpy as np

x = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32)

scale = 0.1
zero_point = 0

q = np.round(x / scale + zero_point).astype(np.int8)
x_dequantized = scale * (q.astype(np.float32) - zero_point)

print(q)
print(x_dequantized)

# Output:
# [ 0  5 10 15]
# [0.  0.5 1.  1.5]
```

<br>

## 3. Dequantization

Dequantization (反量化) converts quantized integers (量化整数) back to approximate floating-point values (近似浮点数值).

```
import numpy as np

q = np.array([0, 5, 10, 15], dtype=np.int8)

scale = 0.1
zero_point = 0

x = scale * (q.astype(np.float32) - zero_point)

print(x)

# Output:
# [0.  0.5 1.  1.5]
```

<br>

## 4. Quantization Error

Quantization Error (量化误差) is the small difference between the original value (原始值) and the dequantized value (反量化值).

```
import numpy as np

x = np.array([0.03, 0.27, 0.61, 0.99], dtype=np.float32)

scale = 0.1
zero_point = 0

q = np.round(x / scale + zero_point).astype(np.int8)
x_dequantized = scale * (q.astype(np.float32) - zero_point)
error = x - x_dequantized

print(q)
print(x_dequantized)
print(error)

# Output:
# [ 0  3  6 10]
# [0.  0.3 0.6 1. ]
# [ 0.03       -0.03000000  0.00999999 -0.00999999]
```

<br>
