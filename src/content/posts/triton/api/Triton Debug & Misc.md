---
title: "Triton Debug & Misc"
published: 2026-04-27
description: "Triton Debug & Misc"
image: ""
tags: ["triton","api","Triton Debug & Misc"]
category: triton / api
draft: false
lang: ""
createdAt: "2026-04-27T17:21:44.875.413683444Z"
---

```
export CUDNN_HOME=$HOME/software/cudnn
export CUDNN_FRONTEND_HOME=$HOME/software/cudnn-frontend
export CUDA_HOME=$(dirname $(dirname $(readlink -f $(which nvcc))))

export LD_LIBRARY_PATH=$CUDNN_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

nvcc -O3 -std=c++17 \
  -gencode arch=compute_90a,code=sm_90a \
  --expt-relaxed-constexpr \
  --expt-extended-lambda \
  -I"$CUDNN_HOME/include" \
  -I"$CUDNN_FRONTEND_HOME/include" \
  gelu_4096x16384x4096.cu \
  -L"$CUDNN_HOME/lib" \
  -L"$CUDA_HOME/lib64" \
  -lcublas \
  -lcudnn \
  -lnvrtc \
  -lcuda \
  -o gelu_4096x16384x4096
  
  
  
export CUDNN_HOME=$HOME/software/cudnn
export CUDNN_FRONTEND_HOME=$HOME/software/cudnn-frontend
export CUDA_HOME=$(dirname $(dirname $(readlink -f $(which nvcc))))

export LD_LIBRARY_PATH=$CUDNN_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

nvcc -O3 -std=c++17 \
  -gencode arch=compute_90a,code=sm_90a \
  --expt-relaxed-constexpr \
  --expt-extended-lambda \
  -I"$CUDNN_HOME/include" \
  -I"$CUDNN_FRONTEND_HOME/include" \
  vibegemm_M4096_N4096_K4096_FP16FP16FP32FP16_H100_submission_gelu.cu \
  -L"$CUDNN_HOME/lib" \
  -L"$CUDA_HOME/lib64" \
  -lcublas -lcudnn -lnvrtc -lcuda \
  -o vibegemm_M4096_N4096_K4096_FP16FP16FP32FP16_H100_gelu
```







```
单gemm
(base) [xli49@ghpc009 gemm]$ ./vibegemm_M4096_N4096_K4096_FP16FP16FP32FP16_H1001
Device: NVIDIA H100 NVL
Problem: M=4096, N=4096, K=4096
Layout : A row-major, B column-major, C column-major; C[4096,4096] = A[4096,4096] * B[4096,4096]
Iters  : warmup=10, benchmark=30

Correctness compared with cuBLAS:
  max absolute error = 0.000000e+00 at index 0
  max relative error = 0.000000e+00 at index 0
  mismatches         = 0
  result             = PASS

Performance:
  Backend           Latency(ms)       TFLOPS
  cuBLAS                 0.2820      487.446
  Custom                 0.2863      480.135

Custom/cuBLAS TFLOPS ratio = 0.9850
Custom speedup vs cuBLAS   = 0.9850x


(base) [xli49@ghpc009 gemm]$ ./vibegemm_M8192_N8192_K8192_FP16FP16FP32FP16_H100
Device: NVIDIA H100 NVL
Problem: M=8192, N=8192, K=8192
Layout : A row-major, B column-major, C column-major; C[8192,8192] = A[8192,8192] * B[8192,8192]
Iters  : warmup=10, benchmark=30

Correctness compared with cuBLAS:
  max absolute error = 0.000000e+00 at index 0
  max relative error = 0.000000e+00 at index 0
  mismatches         = 0
  result             = PASS

Performance:
  Backend           Latency(ms)       TFLOPS
  cuBLAS                 2.3800      461.985
  Custom                 2.4208      454.194

Custom/cuBLAS TFLOPS ratio = 0.9831
Custom speedup vs cuBLAS   = 0.9831x



(base) [xli49@ghpc009 gemm]$ ./vibegemm_M16384_N16384_K16384_FP16FP16FP32FP16_H100
Device: NVIDIA H100 NVL
Problem: M=16384, N=16384, K=16384
Layout : A row-major, B column-major, C column-major; C[16384,16384] = A[16384,16384] * B[16384,16384]
Iters  : warmup=10, benchmark=30

Correctness compared with cuBLAS:
  max absolute error = 0.000000e+00 at index 0
  max relative error = 0.000000e+00 at index 0
  mismatches         = 0
  result             = PASS

Performance:
  Backend           Latency(ms)       TFLOPS
  cuBLAS                19.2952      455.869
  Custom                21.5451      408.264

Custom/cuBLAS TFLOPS ratio = 0.8956
Custom speedup vs cuBLAS   = 0.8956x

```





```
(base) [xli49@ghpc009 gemm]$ ./vibegemm_M4096_N4096_K4096_FP16FP16FP32FP16_H100_cudnn_gelu
Device: NVIDIA H100 NVL
Problem: M=4096, N=4096, K=4096
Layout : A row-major, B column-major, C column-major; C[4096,4096] = A[4096,4096] * B[4096,4096]
Iters  : warmup=10, benchmark=30

Correctness compared with cuBLAS + cuDNN GELU:
  max absolute error = 3.906250e-03 at index 12092
  max relative error = 5.960464e-01 at index 3492
  mismatches         = 0
  result             = PASS

Performance:
  Backend           Latency(ms)       TFLOPS
  cuBLAS+cuDNN           0.3050      450.571
  Custom                 0.2963      463.846

Custom/(cuBLAS+cuDNN GELU) TFLOPS ratio = 1.0295
Custom speedup vs cuBLAS+cuDNN GELU   = 1.0295x


(base) [xli49@ghpc009 gemm]$ ./vibegemm_M8192_N8192_K8192_FP16FP16FP32FP16_H100_cudnn_gelu
Device: NVIDIA H100 NVL
Problem: M=8192, N=8192, K=8192
Layout : A row-major, B column-major, C column-major; C[8192,8192] = A[8192,8192] * B[8192,8192]
Iters  : warmup=10, benchmark=30

Correctness compared with cuBLAS + cuDNN GELU:
  max absolute error = 3.906250e-03 at index 31435
  max relative error = 5.960464e-01 at index 4451
  mismatches         = 0
  result             = PASS

Performance:
  Backend           Latency(ms)       TFLOPS
  cuBLAS+cuDNN           2.3945      459.192
  Custom                 2.5188      436.516

Custom/(cuBLAS+cuDNN GELU) TFLOPS ratio = 0.9506
Custom speedup vs cuBLAS+cuDNN GELU   = 0.9506x


(base) [xli49@ghpc009 gemm]$ ./vibegemm_M16384_N16384_K16384_FP16FP16FP32FP16_H100_cudnn_gelu
Device: NVIDIA H100 NVL
Problem: M=16384, N=16384, K=16384
Layout : A row-major, B column-major, C column-major; C[16384,16384] = A[16384,16384] * B[16384,16384]
Iters  : warmup=10, benchmark=30

Correctness compared with cuBLAS + cuDNN GELU:
  max absolute error = 3.906250e-03 at index 11836
  max relative error = 5.960464e-01 at index 5535
  mismatches         = 0
  result             = PASS

Performance:
  Backend           Latency(ms)       TFLOPS
  cuBLAS+cuDNN          19.5810      449.215
  Custom                22.2365      395.571

Custom/(cuBLAS+cuDNN GELU) TFLOPS ratio = 0.8806
Custom speedup vs cuBLAS+cuDNN GELU   = 0.8806x

```



```


(base) [xli49@ghpc009 gemm]$ ./vibegemm_M4096_N4096_K4096_FP16FP16FP32FP16_H100_rmsnorm
Device: NVIDIA H100 NVL
Problem: M=4096, N=4096, K=4096
Layout : A row-major, B column-major, C column-major; C[4096,4096] = A[4096,4096] * B[4096,4096]
Iters  : warmup=10, benchmark=30

Correctness compared with cuBLAS + cuDNN RMSNorm:
  max absolute error = 1.953125e-03 at index 238130
  max relative error = 9.302326e-04 at index 12416053
  mismatches         = 0
  result             = PASS

Performance:
  Backend           Latency(ms)       TFLOPS
  cuBLAS+cuDNN           0.3080      446.227
  Custom                 0.3746      366.917

Custom/(cuBLAS+cuDNN) TFLOPS ratio = 0.8223
Custom speedup vs cuBLAS+cuDNN   = 0.8223x
(base) [xli49@ghpc009 gemm]$ 


(base) [xli49@ghpc009 gemm]$ ./vibegemm_M8192_N8192_K8192_FP16FP16FP32FP16_H100_cudnn_rmsnorm
Device: NVIDIA H100 NVL
Problem: M=8192, N=8192, K=8192
Layout : A row-major, B column-major, C column-major; C[8192,8192] = A[8192,8192] * B[8192,8192]
Iters  : warmup=10, benchmark=30

Correctness compared with cuBLAS + cuDNN RMSNorm:
  max absolute error = 3.906250e-03 at index 65848658
  max relative error = 9.746589e-04 at index 2646752
  mismatches         = 0
  result             = PASS

Performance:
  Backend           Latency(ms)       TFLOPS
  cuBLAS+cuDNN           2.4332      451.882
  Custom                 2.6537      414.330

Custom/(cuBLAS+cuDNN) TFLOPS ratio = 0.9169
Custom speedup vs cuBLAS+cuDNN   = 0.9169x


(base) [xli49@ghpc009 gemm]$ ./vibegemm_M16384_N16384_K16384_FP16FP16FP32FP16_H100
Device: NVIDIA H100 NVL
Problem: M=16384, N=16384, K=16384
Layout : A row-major, B column-major, C column-major; C[16384,16384] = A[16384,16384] * B[16384,16384]
Iters  : warmup=10, benchmark=30

Correctness compared with cuBLAS + cuDNN RMSNorm:
  max absolute error = 1.953125e-03 at index 3947274
  max relative error = 9.718173e-04 at index 13502064
  mismatches         = 0
  result             = PASS

Performance:
  Backend           Latency(ms)       TFLOPS
  cuBLAS+cuDNN          19.6570      447.478
  Custom                22.7098      387.326

Custom/(cuBLAS+cuDNN) TFLOPS ratio = 0.8656
Custom speedup vs cuBLAS+cuDNN   = 0.8656x
```































# Triton Debug & Misc (调试与杂项)

These APIs help with kernel development (内核开发), letting you print values (打印值), enforce invariants (强制不变量), and inspect compile-time (编译期) information.

## 1. `tl.device_print`

`tl.device_print(prefix, x)` prints a block tensor (块张量) value from inside the GPU kernel (GPU 内核), useful for runtime debugging (运行时调试).

```python
import torch
import triton
import triton.language as tl

@triton.jit
def device_print_kernel(x_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.device_print("x =", x)                          # prints at runtime

BLOCK = 4
x = torch.tensor([10., 20., 30., 40.], device='cuda')

device_print_kernel[(1,)](x, BLOCK=BLOCK)
torch.cuda.synchronize()
# Example stdout (one line per element, may interleave):
# pid (0, 0, 0) idx (0) x = 10.000000
# pid (0, 0, 0) idx (1) x = 20.000000
# pid (0, 0, 0) idx (2) x = 30.000000
# pid (0, 0, 0) idx (3) x = 40.000000
print("done")
# done
```

<br>

## 2. `tl.static_print`

`tl.static_print(...)` prints at compile time (编译期), used to debug `constexpr` values (`constexpr` 值) and template specialization (模板特化).

```python
import torch
import triton
import triton.language as tl

@triton.jit
def static_print_kernel(out_ptr, BLOCK: tl.constexpr, SCALE: tl.constexpr):
    tl.static_print("Compiling with BLOCK =", BLOCK, "SCALE =", SCALE)
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, offs.to(tl.float32) * SCALE)

BLOCK = 4
out = torch.empty(BLOCK, device='cuda')

static_print_kernel[(1,)](out, BLOCK=BLOCK, SCALE=10.0)
# Compile-time stdout (printed once during JIT compilation):
# Compiling with BLOCK = 4 SCALE = 10.0
print(out)
# tensor([ 0., 10., 20., 30.], device='cuda:0')
```

<br>

## 3. `tl.static_assert`

`tl.static_assert(cond, msg)` raises a compile-time error (编译期错误) if the condition (条件) is false, used to validate `constexpr` arguments before the kernel runs.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def static_assert_kernel(out_ptr, BLOCK: tl.constexpr):
    tl.static_assert(BLOCK >= 4, "BLOCK must be >= 4")
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, offs.to(tl.float32))

BLOCK = 8
out = torch.empty(BLOCK, device='cuda')

static_assert_kernel[(1,)](out, BLOCK=BLOCK)           # passes: 8 >= 4
print(out)
# tensor([0., 1., 2., 3., 4., 5., 6., 7.], device='cuda:0')

# If we called with BLOCK=2 it would raise at compile time:
# CompilationError: BLOCK must be >= 4
```

<br>

## 4. `tl.debug_barrier`

`tl.debug_barrier()` inserts a synchronization barrier (同步屏障) inside the kernel, ensuring all threads (所有线程) reach the same point before continuing — useful for debugging shared memory (共享内存) issues.

```python
import torch
import triton
import triton.language as tl

@triton.jit
def barrier_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    tl.debug_barrier()                                 # sync all threads here
    tl.store(out_ptr + offs, x + 1.0)

BLOCK = 4
x = torch.tensor([1., 2., 3., 4.], device='cuda')
out = torch.empty(BLOCK, device='cuda')

barrier_kernel[(1,)](x, out, BLOCK=BLOCK)
print(out)
# tensor([2., 3., 4., 5.], device='cuda:0')
```

<br>

## 5. `TRITON_INTERPRET` (Environment Variable)

Setting `TRITON_INTERPRET=1` runs kernels in pure-Python interpret mode (解释模式), enabling standard `print` and Python debuggers (Python 调试器) like `pdb` for line-by-line debugging (逐行调试).

```python
import os
os.environ["TRITON_INTERPRET"] = "1"                   # enable BEFORE importing triton

import torch
import triton
import triton.language as tl

@triton.jit
def interpret_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    print(f"[interpret] pid={pid} x={x}")              # plain Python print works
    tl.store(out_ptr + offs, x * 10.0)

BLOCK = 4
x = torch.tensor([1., 2., 3., 4.], device='cuda')
out = torch.empty(BLOCK, device='cuda')

interpret_kernel[(1,)](x, out, BLOCK=BLOCK)
# Stdout (in interpret mode):
# [interpret] pid=0 x=tensor([1., 2., 3., 4.])
print(out)
# tensor([10., 20., 30., 40.], device='cuda:0')
```

<br> <br>
