---
title: "Data Types"
published: 2026-05-20
description: "Data Types"
image: ""
tags: ["llm","Data Types"]
category: llm
draft: false
lang: ""
createdAt: "2026-05-20T14:58:43.303.489028611Z"
---

# Numeric Data Types: FP32, FP16, BF16, FP8, FP4, INT64, INT32, INT8, UINT8, INT4, BOOL

## 1. Overview

Numeric data types fall into three families: **floating-point (浮点数)** for real numbers, **integer (整数)** for whole numbers, and **boolean (布尔)** for true/false values.

Key insight about floating-point:

>   **The total significant bits (总有效位数) are fixed, but the decimal point can "float."** When the integer part takes more bits, fewer bits are left for the fractional part. That's why FP32 represents 0.1 with high precision, but at 100 million, it can't even store a fractional part as small as 0.1.

Bit-width comparison:

| Type  | Bits       | Family           | Typical Use                                |
| ----- | ---------- | ---------------- | ------------------------------------------ |
| FP32  | 32         | Float            | Default training (训练) precision          |
| FP16  | 16         | Float            | Mixed-precision training, inference (推理) |
| BF16  | 16         | Float            | LLM training, wide range                   |
| FP8   | 8          | Float            | H100+ inference & training                 |
| FP4   | 4          | Float            | Blackwell-era extreme quantization (量化)  |
| INT64 | 64         | Integer          | Indexing (索引), large counters            |
| INT32 | 32         | Integer          | General integer math                       |
| INT8  | 8          | Integer          | Quantized inference                        |
| UINT8 | 8          | Unsigned integer | Image pixels (0–255)                       |
| INT4  | 4          | Integer          | Weight-only quantization (权重量化)        |
| BOOL  | 8 (stored) | Boolean          | Masks (掩码), conditions                   |

<br>

## 2. FP32

FP32 is the default high-precision floating-point type — slow but safe, used as the ground-truth (基准) reference in mixed precision.

Structure:

```
符号位(1) | 指数位(8) | 尾数位(23)
   S      | EEEEEEEE  | MMMMMMMMMMMMMMMMMMMMMMM
  1 bit   |  8 bits   |        23 bits
import numpy as np
import torch
import triton
import triton.language as tl

# NumPy
a = np.array([1.0, 2.5], dtype=np.float32)
print(a.dtype)                              # float32

# PyTorch
t = torch.tensor([1.0, 2.5], dtype=torch.float32)
t = torch.zeros(3, dtype=torch.float32)
print(t.dtype)                              # torch.float32

# Triton
@triton.jit
def kernel(x_ptr):
    x = tl.load(x_ptr).to(tl.float32)       # cast to fp32
```

<br>

## 3. FP16

FP16 halves memory and doubles throughput on Tensor Cores (张量核心), but the narrow range (±65504) requires loss scaling (损失放大) during training.

Structure:

```
符号位(1) | 指数位(5) | 尾数位(10)
   S      |  EEEEE    | MMMMMMMMMM
  1 bit   |  5 bits   |   10 bits
import numpy as np
import torch
import triton
import triton.language as tl

# NumPy
a = np.array([1.0, 2.5], dtype=np.float16)
print(a.dtype)                              # float16

# PyTorch
t = torch.tensor([1.0, 2.5], dtype=torch.float16)
t = torch.zeros(3, dtype=torch.half)        # 'half' is an alias
t_gpu = t.cuda().half()                     # move + cast

# Automatic mixed precision (自动混合精度)
with torch.autocast(device_type='cuda', dtype=torch.float16):
    out = model(x)

# Triton
@triton.jit
def kernel(x_ptr):
    x = tl.load(x_ptr).to(tl.float16)
```

<br>

## 4. BF16

BF16 (Brain Float 16) keeps FP32's 8-bit exponent for wide range but trims mantissa to 7 bits — preferred for large model training because it rarely overflows (溢出).

Structure:

```
符号位(1) | 指数位(8) | 尾数位(7)
   S      | EEEEEEEE  | MMMMMMM
  1 bit   |  8 bits   |  7 bits
import numpy as np
import torch
import triton
import triton.language as tl

# NumPy: no native bf16 → use ml_dtypes
from ml_dtypes import bfloat16
a = np.array([1.0, 2.5], dtype=bfloat16)

# PyTorch (native support)
t = torch.tensor([1.0, 2.5], dtype=torch.bfloat16)
t = torch.zeros(3, dtype=torch.bfloat16)

# Mixed precision training with bf16 (recommended for LLMs)
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    out = model(x)                          # no GradScaler needed

# Triton
@triton.jit
def kernel(x_ptr):
    x = tl.load(x_ptr).to(tl.bfloat16)
```

<br>

## 5. FP8

FP8 has two variants — **E4M3** (more precision, less range) for forward/weights and **E5M2** (more range, less precision) for gradients (梯度) — supported on Hopper (H100) and newer GPUs.

Structure:

```
E4M3:
符号位(1) | 指数位(4) | 尾数位(3)
   S      |   EEEE    |   MMM
  1 bit   |  4 bits   |  3 bits

E5M2:
符号位(1) | 指数位(5) | 尾数位(2)
   S      |   EEEEE   |    MM
  1 bit   |  5 bits   |  2 bits
import numpy as np
import torch
import triton
import triton.language as tl

# PyTorch (native fp8 dtypes, requires CUDA 12+)
t1 = torch.tensor([1.0, 2.5], dtype=torch.float8_e4m3fn)
t2 = torch.tensor([1.0, 2.5], dtype=torch.float8_e5m2)

# Cast from fp32
x = torch.randn(4, dtype=torch.float32, device='cuda')
x_fp8 = x.to(torch.float8_e4m3fn)

# NumPy: no native fp8 → use ml_dtypes
from ml_dtypes import float8_e4m3fn, float8_e5m2
a = np.array([1.0, 2.5], dtype=float8_e4m3fn)

# Triton
@triton.jit
def kernel(x_ptr):
    x = tl.load(x_ptr).to(tl.float8e4nv)    # e4m3 variant
    y = tl.load(x_ptr).to(tl.float8e5)      # e5m2 variant
```

<br>

## 6. FP4

FP4 (typically E2M1) is a 4-bit floating-point introduced with Blackwell (B100/B200) GPUs — used almost exclusively for inference with calibrated scaling (校准缩放).

Structure:

```
E2M1:
符号位(1) | 指数位(2) | 尾数位(1)
   S      |    EE     |    M
  1 bit   |  2 bits   |  1 bit
import torch
import triton
import triton.language as tl

# PyTorch (preview support in nightly builds, dtype API still evolving)
# Typically accessed through libraries:
#   - TransformerEngine (NVIDIA)
#   - bitsandbytes
#   - torch.float4_e2m1fn_x2  (packs 2 values per byte)

# Example via bitsandbytes (FP4 quantization)
from bitsandbytes.nn import Linear4bit
layer = Linear4bit(1024, 1024, quant_type='fp4')

# Triton (experimental, version-dependent)
@triton.jit
def kernel(x_ptr):
    x = tl.load(x_ptr).to(tl.float4)        # availability depends on version
```

<br>

## 7. INT64

INT64 is the default integer type in PyTorch — required for tensor indexing (张量索引) and large value ranges (±9.2 × 10¹⁸).

Structure:

```
符号位(1) | 数值位(63)
   S      | NNNNNNNN...NNNNNNNN
  1 bit   |       63 bits
import numpy as np
import torch
import triton
import triton.language as tl

# NumPy
a = np.array([1, 2, 3], dtype=np.int64)
a = np.arange(5, dtype=np.int64)

# PyTorch (int64 = long)
t = torch.tensor([1, 2, 3], dtype=torch.int64)
t = torch.tensor([1, 2, 3], dtype=torch.long)     # alias

# Indexing always uses int64
data = torch.randn(10)
indices = torch.tensor([0, 2], dtype=torch.long)
result = data[indices]

# Triton
@triton.jit
def kernel(x_ptr):
    x = tl.load(x_ptr).to(tl.int64)
```

<br>

## 8. INT32

INT32 is the standard integer type in NumPy and a memory-efficient alternative to INT64 when values fit in ±2.1 × 10⁹.

Structure:

```
符号位(1) | 数值位(31)
   S      | NNNNNNNN...NNNNNNN
  1 bit   |       31 bits
import numpy as np
import torch
import triton
import triton.language as tl

# NumPy
a = np.array([1, 2, 3], dtype=np.int32)
a = np.zeros(5, dtype=np.int32)

# PyTorch
t = torch.tensor([1, 2, 3], dtype=torch.int32)
t = torch.tensor([1, 2, 3], dtype=torch.int)      # alias

# Triton (most common integer type in kernels)
@triton.jit
def kernel(x_ptr, n):
    offsets = tl.arange(0, 128).to(tl.int32)
    mask = offsets < n
```

<br>

## 9. INT8

INT8 is the workhorse of quantized inference (量化推理) — 4× memory reduction vs FP32 with Tensor Core acceleration for matrix multiply (矩阵乘法).

Structure:

```
符号位(1) | 数值位(7)
   S      | NNNNNNN
  1 bit   |  7 bits
```

Range: −128 to 127.

```python
import numpy as np
import torch
import triton
import triton.language as tl

# NumPy
a = np.array([-1, 0, 127], dtype=np.int8)

# PyTorch
t = torch.tensor([-1, 0, 127], dtype=torch.int8)

# Quantization workflow
x_fp32 = torch.randn(4)
scale = x_fp32.abs().max() / 127
x_int8 = (x_fp32 / scale).round().to(torch.int8)

# Triton
@triton.jit
def kernel(x_ptr):
    x = tl.load(x_ptr).to(tl.int8)
```

<br>

## 10. UINT8

UINT8 (unsigned 8-bit integer, 无符号 8 位整数) covers 0–255 — the native format for image pixels (图像像素) and raw byte data.

Structure:

```
数值位(8)
NNNNNNNN
 8 bits     (no sign bit, all bits for value)
import numpy as np
import torch
import triton
import triton.language as tl

# NumPy (default image dtype)
img = np.array([[0, 128, 255]], dtype=np.uint8)
img = np.zeros((224, 224, 3), dtype=np.uint8)        # RGB image

# PyTorch
t = torch.tensor([0, 128, 255], dtype=torch.uint8)

# Image preprocessing
from torchvision import transforms
img_uint8 = torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8)
img_float = img_uint8.float() / 255.0                # normalize to [0, 1]

# Triton
@triton.jit
def kernel(x_ptr):
    x = tl.load(x_ptr).to(tl.uint8)
```

<br>

## 11. INT4

INT4 stores 16 values (−8 to 7 signed, or 0 to 15 unsigned), used in weight-only quantization (权重量化) where two values are packed into one byte.

Structure:

```
符号位(1) | 数值位(3)
   S      |   NNN
  1 bit   |  3 bits
```

Range: −8 to 7 (signed), or 0 to 15 (unsigned).

```python
import torch
import triton
import triton.language as tl

# PyTorch: no native int4, use bitsandbytes or packed uint8
from bitsandbytes.nn import Linear4bit
layer = Linear4bit(1024, 1024, quant_type='nf4')     # 4-bit linear layer

# Manual packing: two int4 values per uint8 byte
def pack_int4(x_int4):
    # x_int4: tensor with values in [-8, 7], shape (..., even_size)
    low = x_int4[..., 0::2] & 0x0F
    high = (x_int4[..., 1::2] & 0x0F) << 4
    return (low | high).to(torch.uint8)

# Triton (experimental, requires recent versions)
@triton.jit
def kernel(x_ptr):
    x = tl.load(x_ptr)                                # load as uint8, unpack manually
```

<br>

## 12. BOOL

BOOL holds `True` or `False` — logically 1 bit, but stored as 1 byte (8 bits) due to memory alignment (内存对齐).

Structure:

```
数值位(1 logical, 8 stored)
0 0 0 0 0 0 0 B
 padding      └─ actual bool value
 1 byte total
import numpy as np
import torch
import triton
import triton.language as tl

# NumPy
a = np.array([True, False, True], dtype=np.bool_)
mask = np.array([1, 0, 1], dtype=bool)

# PyTorch
t = torch.tensor([True, False, True], dtype=torch.bool)

# Boolean masking (布尔掩码)
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
mask = x > 2.0                                        # dtype=torch.bool
selected = x[mask]                                    # tensor([3., 4.])

# Triton
@triton.jit
def kernel(x_ptr, n):
    offsets = tl.arange(0, 128)
    mask = offsets < n                                # boolean mask
    x = tl.load(x_ptr + offsets, mask=mask)
```

<br> <br>
