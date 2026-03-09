---
title: "Coalesced memory access"
published: 2026-03-09
description: "Coalesced memory access"
image: ""
tags: ["cuda","Coalesced memory access"]
category: cuda
draft: false
lang: ""
---



# **I. Coalesced Memory Access (合并访存) — Complete Learning Note**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <strong>Overview:</strong> <strong>Coalesced Memory Access (合并访存)</strong> is the single most impactful GPU memory optimization. When threads in a <strong>Warp (线程束, 32 threads)</strong> access consecutive, aligned memory addresses simultaneously, the GPU hardware merges them into a single wide memory transaction — maximizing <strong>Global Memory Bandwidth (全局内存带宽)</strong>. Uncoalesced access wastes bandwidth by issuing multiple narrow transactions instead of one wide one, and is the most common cause of memory-bound performance problems. </div>

------

## 1. Why Coalesced Access Matters (为什么需要合并访存)

### 1) The Hardware Reality: Warp Execution Model

A GPU executes threads in groups of <span style="color:#E8600A;font-weight:700">32 threads called a Warp (线程束)</span>. All 32 threads in a warp execute the **same instruction at the same time** (SIMT 单指令多线程).

When those 32 threads all execute a memory load instruction, the memory controller checks:

>   <span style="color:#2980B9">Can all these 32 accesses be served by a single (or minimal number of) 128-byte cache line transaction?</span>

-   If **YES** → <span style="color:#E8600A;font-weight:700">Coalesced (合并)</span>: **1 transaction**, full bandwidth utilized
-   If **NO** → <span style="color:#C0392B;font-weight:600">Uncoalesced (未合并)</span>: up to **32 separate transactions**, massive bandwidth waste

------

### 2) Memory Transaction Width (内存事务宽度)

Modern NVIDIA GPUs (Ampere, Hopper) use a **128-byte L2 cache line (缓存行)**.

One warp of 32 threads × 4 bytes (float) = **128 bytes** — exactly one cache line.

This means: if 32 float threads access 32 **consecutive** floats, the hardware perfectly merges them into **1 transaction**.

```
Ideal coalesced access pattern:
────────────────────────────────────────────────────────
Thread:   T0   T1   T2   T3  ...  T30  T31
Address:  [0]  [1]  [2]  [3] ... [30] [31]   ← consecutive float addresses
           └──────────────────────────────┘
                  1 × 128-byte transaction ✅
────────────────────────────────────────────────────────

Strided (non-coalesced) access pattern:
────────────────────────────────────────────────────────
Thread:   T0   T1   T2   T3  ...  T30  T31
Address:  [0]  [2]  [4]  [6] ... [60] [62]   ← stride-2
           |    |    |    |         |    |
           🔴   🔴   🔴   🔴  ...  🔴   🔴
              up to 32 separate transactions ❌
────────────────────────────────────────────────────────
```

------

### 3) Bandwidth Impact (带宽影响)

On an A100 GPU with ~2 TB/s memory bandwidth:

| Access Pattern                                               | Effective Bandwidth | Utilization |
| ------------------------------------------------------------ | ------------------- | ----------- |
| <span style="color:#E8600A;font-weight:700">Coalesced (stride-1)</span> | ~2000 GB/s          | ~100%       |
| Stride-2                                                     | ~1000 GB/s          | ~50%        |
| Stride-4                                                     | ~500 GB/s           | ~25%        |
| <span style="color:#C0392B;font-weight:600">Random (stride-32+)</span> | ~62 GB/s            | ~3%         |

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> A kernel bottlenecked by uncoalesced memory can be <strong>10–32× slower</strong> than the coalesced version — without changing a single line of arithmetic. This is why CUDA profilers (如 Nsight Compute) report <strong>Global Load/Store Efficiency</strong> as a key metric. Anything below 80% warrants investigation.</div>

------

## 2. Visual Diagrams (图解)

### 1) Coalesced vs Uncoalesced — Memory Layout

```
═══════════════════════════════════════════════════════════════════
  GLOBAL MEMORY (全局内存)
═══════════════════════════════════════════════════════════════════

  Byte offset: 0    4    8    12   16   20   24  ...  124  128
               ┌────┬────┬────┬────┬────┬────┬────┬────┬────┐
               │ f0 │ f1 │ f2 │ f3 │ f4 │ f5 │ f6 │... │f31 │
               └────┴────┴────┴────┴────┴────┴────┴────┴────┘
               ◄─────────────── 128 bytes (one cache line) ───────────────►

  ✅ COALESCED: Thread Ti loads float[i]
  ─────────────────────────────────────────────────────────────────
  T0→f0  T1→f1  T2→f2  T3→f3  ...  T31→f31
  All 32 accesses land on the SAME 128-byte cache line
  → Hardware issues: 1 memory transaction

  ❌ UNCOALESCED (Column-major on row-major data, stride=N):
  ─────────────────────────────────────────────────────────────────
  T0→f[0]   T1→f[N]   T2→f[2N]  ...  T31→f[31N]
  Each thread hits a DIFFERENT cache line (N floats apart)
  → Hardware issues: up to 32 memory transactions


═══════════════════════════════════════════════════════════════════
  TRANSACTION COUNT COMPARISON (for 1 warp, 32 float loads)
═══════════════════════════════════════════════════════════════════

  Pattern           Transactions   Bandwidth Used
  ──────────────────────────────────────────────
  Coalesced         ██ 1           ████████████ 100%
  Stride-2          ████ 2         ██████ 50%
  Stride-4          ████████ 4     ███ 25%
  Stride-32+        ████████...32  █ ~3%
  ──────────────────────────────────────────────
```

------

### 2) Matrix Transpose — The Classic Problem

```
  ROW-MAJOR MATRIX in memory (行主序矩阵):
  ┌─────────────────────────────────────┐
  │ [0,0][0,1][0,2]...[0,N-1]           │  ← Row 0 is contiguous
  │ [1,0][1,1][1,2]...[1,N-1]           │  ← Row 1 is contiguous
  │  ...                                │
  │ [M-1,0]...[M-1,N-1]                 │
  └─────────────────────────────────────┘

  READING a row   (Thread Ti reads matrix[row][Ti]):
  → Addresses: row*N+0, row*N+1, ..., row*N+31
  → STRIDE 1 → COALESCED ✅

  READING a column (Thread Ti reads matrix[Ti][col]):
  → Addresses: 0*N+col, 1*N+col, ..., 31*N+col
  → STRIDE N → UNCOALESCED ❌ (each in a different cache line)
```

------

## 3. Code Examples (代码示例)

### 1) ❌ Naive Vector Copy — Uncoalesced (Intentionally Bad)

```cpp
// BAD: threads access memory with stride = blockDim.x * gridDim.x
// Each thread iterates over every Nth element (column-style access)
__global__ void copy_strided_bad(float* dst, const float* src, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // Thread 0 touches: 0, stride, 2*stride ...
    // Thread 1 touches: 1, stride+1, 2*stride+1 ...
    // Within one iteration, consecutive threads access consecutive elements → OK
    // But if we rewrite as:
    for (int i = tid; i < N; i += stride) {
        dst[i] = src[i];   // This is actually FINE — stride-1 within a warp
    }
}
// The above is actually coalesced. See the TRULY bad example below:

// Truly uncoalesced: threads access with a large stride
__global__ void copy_column_bad(float* dst, const float* src, int rows, int cols) {
    // Threads assigned to COLUMNS — each thread reads an entire column
    int col = threadIdx.x + blockIdx.x * blockDim.x;   // Thread → column

    if (col < cols) {
        for (int row = 0; row < rows; row++) {
            // Warp threads read: src[0*cols + col], src[0*cols + col+1], ...
            // WAIT — this is column access: src[row*cols + col]
            // Thread 0 reads row 0 col 0, Thread 1 reads row 0 col 1 → coalesced!
            // But: all threads read the SAME row across different cols → depends on layout
            dst[row * cols + col] = src[row * cols + col];
        }
    }
}
```

### 2) ✅ Coalesced Vector Copy — Correct Pattern

```cpp
// GOOD: thread index maps directly to array index
// T0→[0], T1→[1], T2→[2], ... T31→[31]  — perfect stride-1 access
__global__ void copy_coalesced(float* dst, const float* src, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N) {
        dst[idx] = src[idx];   // Thread Ti → element [i]: COALESCED ✅
    }
}

// Launch
int N = 1 << 24;   // 16M floats
int blockSize = 256;
int gridSize  = (N + blockSize - 1) / blockSize;
copy_coalesced<<<gridSize, blockSize>>>(d_dst, d_src, N);
```

------

### 3) Matrix Transpose — Uncoalesced vs Coalesced via Shared Memory

```cpp
#define TILE_DIM  32
#define BLOCK_ROWS 8

// ❌ NAIVE TRANSPOSE: reads are coalesced, but WRITES are not
__global__ void transpose_naive(float* out, const float* in,
                                 int rows, int cols) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (x < cols && y < rows) {
        // READ  in[y*cols + x]      → stride-1 across warp: COALESCED ✅
        // WRITE out[x*rows + y]     → stride = rows across warp: UNCOALESCED ❌
        out[x * rows + y] = in[y * cols + x];
    }
}


// ✅ TILED TRANSPOSE: uses shared memory to fix uncoalesced writes
__global__ void transpose_shared(float* out, const float* in,
                                  int rows, int cols) {
    // Shared memory tile — extra column to avoid bank conflicts (银行冲突)
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Phase 1: COALESCED READ from global memory into shared memory
    // Threads load a TILE_DIM×TILE_DIM block, row by row
    if (x < cols) {
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
            if (y + j < rows) {
                // in[(y+j)*cols + x]: consecutive x → stride-1 → COALESCED ✅
                tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * cols + x];
            }
        }
    }

    __syncthreads();   // Ensure all threads have written to shared memory

    // Transpose the block indices
    int tx = blockIdx.y * TILE_DIM + threadIdx.x;
    int ty = blockIdx.x * TILE_DIM + threadIdx.y;

    // Phase 2: COALESCED WRITE to global memory from shared memory
    if (tx < rows) {
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
            if (ty + j < cols) {
                // tile[threadIdx.x][threadIdx.y+j]: reading transposed from smem
                // Write to out[(ty+j)*rows + tx]: consecutive tx → COALESCED ✅
                out[(ty + j) * rows + tx] = tile[threadIdx.x][threadIdx.y + j];
            }
        }
    }
}

// Benchmark launch
dim3 dimGrid (cols / TILE_DIM, rows / TILE_DIM);
dim3 dimBlock(TILE_DIM, BLOCK_ROWS);
transpose_naive <<<dimGrid, dimBlock>>>(d_out, d_in, rows, cols);
transpose_shared<<<dimGrid, dimBlock>>>(d_out, d_in, rows, cols);
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> The <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">+1</code> padding in <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">float tile[TILE_DIM][TILE_DIM + 1]</code> avoids <strong>Shared Memory Bank Conflicts (共享内存访问冲突)</strong>. Without it, all 32 threads in a warp would hit the same bank when reading the transposed tile column, serializing accesses and losing all shared memory bandwidth benefit.</div>

------

### 4) Structure of Arrays (SoA) vs Array of Structures (AoS)

A very common real-world source of uncoalesced access is choosing **AoS over SoA** for structured data.

```cpp
// ❌ AoS (Array of Structures / 结构体数组) — UNCOALESCED
struct ParticleAoS {
    float x, y, z;   // 12 bytes per particle
    float vx, vy, vz;
};

__global__ void update_positions_aos(ParticleAoS* particles, int N, float dt) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        // Thread Ti reads particles[i].x, Thread Ti+1 reads particles[i+1].x
        // Memory: [x0,y0,z0,vx0,vy0,vz0,  x1,y1,z1,vx1,vy1,vz1, ...]
        //          ↑                        ↑
        //          T0                       T1
        // Stride = 6 floats = 24 bytes → NOT contiguous → UNCOALESCED ❌
        particles[i].x += particles[i].vx * dt;
        particles[i].y += particles[i].vy * dt;
        particles[i].z += particles[i].vz * dt;
    }
}


// ✅ SoA (Structure of Arrays / 数组结构体) — COALESCED
struct ParticlesSoA {
    float* x;    // All x-coordinates contiguous
    float* y;    // All y-coordinates contiguous
    float* z;
    float* vx;
    float* vy;
    float* vz;
};

__global__ void update_positions_soa(ParticlesSoA p, int N, float dt) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        // Thread Ti reads p.x[i], Thread Ti+1 reads p.x[i+1]
        // Memory: [x0, x1, x2, x3, ... x31, x32, ...]
        //          T0  T1  T2  T3       T31
        // Stride = 1 float = 4 bytes → CONTIGUOUS → COALESCED ✅
        p.x[i] += p.vx[i] * dt;
        p.y[i] += p.vy[i] * dt;
        p.z[i] += p.vz[i] * dt;
    }
}
Memory layout comparison for 4 particles:

AoS:  [x0 y0 z0 vx0 vy0 vz0 | x1 y1 z1 vx1 vy1 vz1 | x2 ... | x3 ...]
       T0's x-field           T1's x-field
       ◄──────── 24 bytes ───────►◄────────24 bytes────►
       STRIDE 6 — T0 and T1's x-fields are 6 floats apart ❌

SoA:  [x0 x1 x2 x3 ... xN | y0 y1 y2 ... | z0 z1 ... | vx0 vx1 ...]
       T0 T1 T2 T3          
       STRIDE 1 — all x-fields are consecutive ✅
```

------

### 5) Benchmark: Measuring the Difference

```cpp
#include <cuda_runtime.h>
#include <stdio.h>

#define N (1 << 25)   // 32M floats = 128 MB

__global__ void stride1(float* dst, const float* src, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) dst[i] = src[i];                         // Coalesced
}

__global__ void stride32(float* dst, const float* src, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = (i % 32) * (n / 32) + (i / 32);            // Shuffle: stride ~1M
    if (j < n) dst[j] = src[j];                         // Uncoalesced
}

float benchmark(void (*launch)(float*, float*, int), float* d_dst, float* d_src) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int block = 256, grid = (N + block - 1) / block;

    // Warmup
    launch<<<grid, block>>>(d_dst, d_src, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
        launch<<<grid, block>>>(d_dst, d_src, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / 100.0f;
}

int main() {
    float *d_src, *d_dst;
    cudaMalloc(&d_src, N * sizeof(float));
    cudaMalloc(&d_dst, N * sizeof(float));

    float t1  = benchmark(stride1,  d_dst, d_src);
    float t32 = benchmark(stride32, d_dst, d_src);

    float bw1  = 2.0f * N * sizeof(float) / (t1  * 1e-3f) / 1e9f;
    float bw32 = 2.0f * N * sizeof(float) / (t32 * 1e-3f) / 1e9f;

    printf("Coalesced   (stride-1):  %.2f ms  →  %.1f GB/s\n", t1,  bw1);
    printf("Uncoalesced (stride-32): %.2f ms  →  %.1f GB/s\n", t32, bw32);
    printf("Speedup: %.1fx\n", t32 / t1);

    cudaFree(d_src);
    cudaFree(d_dst);
}

/* Typical output on A100:
   Coalesced   (stride-1):  0.43 ms  →  1580.3 GB/s
   Uncoalesced (stride-32): 13.1 ms  →   51.8 GB/s
   Speedup: 30.5x
*/
```

------

## 4. Rules for Achieving Coalesced Access (实现合并访存的规则)

### 1) The Golden Rule

>   <span style="color:#E8600A;font-weight:700">Thread index and array index should increase together.</span> Thread `threadIdx.x + blockIdx.x * blockDim.x` should map to the **innermost (fastest-varying) dimension** of your data.

```cpp
// ✅ CORRECT: thread index = array index (innermost dim)
int i = threadIdx.x + blockIdx.x * blockDim.x;
float val = array[i];

// ❌ WRONG: threads map to outer dimension (stride = cols)
int row = threadIdx.x + blockIdx.x * blockDim.x;
float val = matrix[row * cols + fixed_col];   // All threads hit different rows
```

------

### 2) 2D Grid — Assign Threads to Columns (Not Rows)

```cpp
// For a (rows × cols) matrix stored row-major:

// ✅ threadIdx.x → column direction (fast-varying)
int col = threadIdx.x + blockIdx.x * blockDim.x;
int row = threadIdx.y + blockIdx.y * blockDim.y;
float val = matrix[row * cols + col];   // Consecutive threads → consecutive cols ✅

// ❌ threadIdx.x → row direction (slow-varying)
int row = threadIdx.x + blockIdx.x * blockDim.x;
int col = threadIdx.y + blockIdx.y * blockDim.y;
float val = matrix[row * cols + col];   // Consecutive threads → stride-cols ❌
```

------

### 3) Use Shared Memory as a Coalescing Buffer (共享内存作为合并访问缓冲)

When the algorithm requires non-sequential access (e.g., matrix transpose, stencil), use the pattern:

1.  <span style="color:#2980B9">Load</span> from global memory **coalesced** into shared memory tile
2.  <span style="color:#2980B9">Process</span> data in shared memory (any access pattern is fine — shared memory has no bandwidth penalty for stride)
3.  <span style="color:#2980B9">Write</span> from shared memory to global memory **coalesced**

```cpp
__shared__ float smem[TILE][TILE + 1];

// Step 1: Coalesced load — threadIdx.x maps to column
smem[threadIdx.y][threadIdx.x] = global_in[row * N + col];
__syncthreads();

// Step 2: Arbitrary computation in smem (e.g., transpose)
float val = smem[threadIdx.x][threadIdx.y];   // Transposed read — OK in smem

// Step 3: Coalesced write back
global_out[out_row * N + out_col] = val;
```

------

## 5. Checklist for Coalesced Access (合并访存检查清单)

| Check                         | Question                                                 | Fix                                                          |
| ----------------------------- | -------------------------------------------------------- | ------------------------------------------------------------ |
| ✅ **Thread-to-index mapping** | Does `threadIdx.x` map to the innermost array dimension? | Swap loop/thread assignment order                            |
| ✅ **Data layout**             | Is your struct layout AoS or SoA?                        | Convert to SoA                                               |
| ✅ **2D grid orientation**     | Does `threadIdx.x` vary along columns (not rows)?        | Transpose block/grid assignment                              |
| ✅ **Shared memory usage**     | Is non-sequential access buffered via shared memory?     | Add shared memory tiling                                     |
| ✅ **Alignment**               | Is the base address aligned to 128 bytes?                | Use `cudaMalloc` (always 256-byte aligned) or `__align__`    |
| ✅ **Profiler check**          | Is Global Memory Load/Store Efficiency > 80%?            | Run Nsight Compute → `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum` |

------

## 6. Profiling Coalescing with Nsight Compute

```bash
# Profile global memory efficiency
ncu --metrics \
  l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
  ./my_kernel

# Key metric: sectors per request
# Ideal = 1 sector/request (coalesced)
# Bad   = 32 sectors/request (fully uncoalesced)

# Also check:
ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct ./my_kernel
# 100% = perfectly coalesced
# 12.5% = fully uncoalesced (1/8 of 128 bytes used per sector)
```

------

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> Coalesced access means <strong>32 warp threads reading 32 consecutive addresses → 1 memory transaction</strong>; always map <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">threadIdx.x</code> to the <strong>innermost (fastest-varying) array dimension</strong>, prefer <strong>SoA over AoS</strong> for structured data, and use a <strong>shared memory tile</strong> to convert unavoidably non-sequential access (e.g., transpose) into coalesced global loads and stores.</div>
