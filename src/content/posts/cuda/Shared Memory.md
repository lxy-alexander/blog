---
title: "Shared Memory"
published: 2026-03-09
description: "Shared Memory"
image: ""
tags: ["cuda","Shared Memory"]
category: cuda
draft: false
lang: ""
---

# **I. CUDA Shared Memory (共享内存) — Complete Learning Note**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <strong>Overview:</strong> <strong>Shared Memory (共享内存)</strong> is a fast, on-chip memory (片上内存) that is shared among all threads within a <strong>thread block (线程块)</strong>. It sits physically on the SM (Streaming Multiprocessor) chip — its latency is ~100× lower than global memory and its bandwidth is ~10× higher. The core programming pattern is: <span style="color:#E8600A;font-weight:700">load from global memory → process in shared memory → write back to global memory</span>, replacing expensive repeated global accesses with cheap shared memory accesses. </div>

------

## 1. Memory Hierarchy — Where Shared Memory Lives (内存层次结构)

```
═══════════════════════════════════════════════════════════════════════
  GPU MEMORY HIERARCHY (GPU 内存层次)
═══════════════════════════════════════════════════════════════════════

  ┌───────────────────────────────────────────────────────────────┐
  │                    Global Memory (全局内存)                    │
  │              HBM2 / GDDR6 — off-chip DRAM                     │
  │   Size: 16–80 GB    Latency: ~400–800 cycles    BW: ~2 TB/s   │
  └────────────────────────────┬──────────────────────────────────┘
                               │
  ┌────────────────────────────▼──────────────────────────────────┐
  │               L2 Cache (二级缓存)  — on-chip                   │
  │         Size: 40–50 MB    Latency: ~200 cycles                 │
  └────────────────────────────┬──────────────────────────────────┘
                               │
       ┌───────────────────────▼────────────────────────────┐
       │              SM (Streaming Multiprocessor)          │
       │  ┌──────────────────────────────────────────────┐  │
       │  │  L1 Cache + Shared Memory  ← same SRAM chip  │  │
       │  │  Size: 32–228 KB  Latency: ~20–30 cycles     │  │
       │  │  BW: ~19 TB/s (A100)                         │  │
       │  │                                              │  │
       │  │  ┌────────────┐   ┌──────────────────────┐  │  │
       │  │  │ L1 Cache   │   │  Shared Memory (SMEM) │  │  │
       │  │  │(自动管理)  │   │  (手动管理, per block)│  │  │
       │  │  └────────────┘   └──────────────────────┘  │  │
       │  └──────────────────────────────────────────────┘  │
       │                                                     │
       │  ┌─────────────────────────────────────────────┐   │
       │  │  Registers (寄存器)                          │   │
       │  │  Size: 256 KB   Latency: 0 cycles (fastest) │   │
       │  └─────────────────────────────────────────────┘   │
       └─────────────────────────────────────────────────────┘

  Key point: Shared Memory is ON the SM chip → no PCIe/HBM bus → ultra-fast
═══════════════════════════════════════════════════════════════════════
```

### Memory Comparison Table (内存对比)

| Memory                                                       | Location         | Scope       | Latency        | Size           | Managed by |
| ------------------------------------------------------------ | ---------------- | ----------- | -------------- | -------------- | ---------- |
| <span style="color:#E8600A;font-weight:700">Registers (寄存器)</span> | On-chip          | Per thread  | 0 cycles       | 256 KB / SM    | Compiler   |
| <span style="color:#E8600A;font-weight:700">Shared Memory (共享内存)</span> | On-chip          | Per block   | ~20–30 cy      | 32–228 KB / SM | Programmer |
| L1 Cache                                                     | On-chip          | Per SM      | ~30 cy         | Configurable   | Hardware   |
| L2 Cache                                                     | On-chip          | All SMs     | ~200 cy        | 40–50 MB       | Hardware   |
| <span style="color:#C0392B;font-weight:600">Global Memory (全局内存)</span> | Off-chip         | All threads | ~400–800 cy    | 16–80 GB       | Programmer |
| Constant Memory (常量内存)                                   | Off-chip + cache | All threads | ~4 cy (cached) | 64 KB          | Programmer |
| Texture Memory (纹理内存)                                    | Off-chip + cache | All threads | ~4 cy (cached) | N/A            | Programmer |

------

## 2. Shared Memory Architecture — Banks (共享内存架构：Bank)

### 1) What Are Banks? (什么是 Bank)

Shared memory is divided into <span style="color:#E8600A;font-weight:700">32 banks (存储体)</span> — one per thread in a warp. Each bank is 4 bytes wide (32-bit). Consecutive 4-byte words are mapped to consecutive banks:

```
═══════════════════════════════════════════════════════════════
  SHARED MEMORY BANK LAYOUT (32 banks × 32-bit words)
═══════════════════════════════════════════════════════════════

  Address:  0     4     8    12    16  ...  124   128   132
            │     │     │     │     │         │     │     │
  Bank:     0     1     2     3     4  ...   31     0     1
                                              ↑  wraps around
  ┌──────┬──────┬──────┬──────┬──────┬─────┬──────┐
  │Bank 0│Bank 1│Bank 2│Bank 3│Bank 4│ ... │Bank31│  ← Row 0
  ├──────┼──────┼──────┼──────┼──────┼─────┼──────┤
  │Bank 0│Bank 1│Bank 2│Bank 3│Bank 4│ ... │Bank31│  ← Row 1
  └──────┴──────┴──────┴──────┴──────┴─────┴──────┘

  word_index = byte_address / 4
  bank_id    = word_index % 32
═══════════════════════════════════════════════════════════════
```

### 2) Bank Conflicts (Bank 冲突)

A <span style="color:#C0392B;font-weight:600">Bank Conflict (存储体冲突)</span> occurs when two or more threads in the same warp access **different addresses in the same bank**. The hardware serializes these accesses, reducing throughput.

```
═══════════════════════════════════════════════════════════════
  BANK CONFLICT SCENARIOS
═══════════════════════════════════════════════════════════════

  ✅ NO CONFLICT — each thread hits a different bank
  ─────────────────────────────────────────────────
  Thread:  T0   T1   T2   T3  ...  T31
  Bank:     0    1    2    3  ...   31
  → 1 cycle, full throughput

  ✅ NO CONFLICT — broadcast (all threads same address)
  ─────────────────────────────────────────────────
  Thread:  T0   T1   T2   T3  ...  T31
  Address: [4]  [4]  [4]  [4] ...  [4]   ← all same
  Bank:     1    1    1    1  ...    1
  → Hardware broadcasts: still 1 cycle ✅

  ❌ 2-WAY CONFLICT — 2 threads share a bank
  ─────────────────────────────────────────────────
  Thread:  T0   T1   T2  ...  T16  T17
  Address: [0] [128] [8] ... [64] [192]
  Bank:     0    0    2  ...   16   16
  T0 & T16 share bank 0 → serialized → 2 cycles

  ❌ 32-WAY CONFLICT (worst case) — all threads hit bank 0
  ─────────────────────────────────────────────────
  Thread:  T0    T1    T2   ...  T31
  Address: [0]  [128] [256] ...  [3968]   ← stride 32
  Bank:     0     0     0   ...    0
  → All serialized → 32 cycles (32× slowdown!) ❌
═══════════════════════════════════════════════════════════════
```

### 3) The +1 Padding Trick (填充消除冲突)

The classic fix for the 32-way bank conflict in 2D tiles:

```cpp
// ❌ WITHOUT padding: columns of a 32×32 tile all map to same bank
__shared__ float tile[32][32];
// tile[row][col]: bank = col % 32
// Reading column 0: tile[0][0], tile[1][0], ... tile[31][0]
// All have col=0 → bank 0 → 32-way conflict ❌

// ✅ WITH padding: shift each row by 1 word to spread banks
__shared__ float tile[32][32 + 1];
// tile[row][col]: bank = (col + row * 33) % 32  — now all different ✅
```

------

## 3. Declaring Shared Memory (声明共享内存)

### 1) Static Shared Memory (静态共享内存)

Size known at compile time:

```cpp
__global__ void kernel_static(float* data, int N) {
    __shared__ float smem[256];   // 256 floats = 1024 bytes, fixed size

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Load into shared memory
    smem[tid] = (idx < N) ? data[idx] : 0.0f;
    __syncthreads();

    // Process...
    smem[tid] *= 2.0f;
    __syncthreads();

    // Write back
    if (idx < N) data[idx] = smem[tid];
}
```

------

### 2) Dynamic Shared Memory (动态共享内存)

Size specified at launch time via the third kernel argument:

```cpp
// Declare with extern and empty brackets — size determined at launch
__global__ void kernel_dynamic(float* data, int N) {
    extern __shared__ float smem[];   // Size unknown at compile time

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    smem[tid] = (idx < N) ? data[idx] : 0.0f;
    __syncthreads();

    smem[tid] *= 2.0f;
    __syncthreads();

    if (idx < N) data[idx] = smem[tid];
}

// Launch: 3rd parameter = shared memory bytes
int blockSize = 256;
int gridSize  = (N + blockSize - 1) / blockSize;
size_t smemBytes = blockSize * sizeof(float);

kernel_dynamic<<<gridSize, blockSize, smemBytes>>>(d_data, N);
```

------

### 3) Multiple Dynamic Arrays (多个动态数组)

Use pointer arithmetic to carve up the single `extern __shared__` allocation:

```cpp
__global__ void multi_array_kernel(float* a, int* b, int N) {
    extern __shared__ char raw_smem[];

    float* smem_float = (float*)raw_smem;
    int*   smem_int   = (int*)(raw_smem + blockDim.x * sizeof(float));

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N) {
        smem_float[tid] = a[idx];
        smem_int[tid]   = b[idx];
    }
    __syncthreads();

    // Use both arrays...
}

// Launch with combined size
size_t smemBytes = blockSize * (sizeof(float) + sizeof(int));
multi_array_kernel<<<gridSize, blockSize, smemBytes>>>(d_a, d_b, N);
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> When carving multiple arrays from dynamic shared memory, <span style="color:#C0392B;font-weight:600">align each sub-array to its type's alignment requirements.</span> Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">__align__(8)</code> or manually pad offsets to 8-byte boundaries for <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">double</code> / <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">int2</code> types to avoid misaligned accesses.</div>

------

## 4. `__syncthreads()` — The Barrier (同步屏障)

`__syncthreads()` is a <span style="color:#E8600A;font-weight:700">block-level barrier (块级屏障)</span>: no thread in the block proceeds past it until **all** threads have reached it.

```
Thread 0:  ─── write smem[0] ──────────────── __syncthreads() ─── read smem[1] ──►
Thread 1:  ─── write smem[1] ──── __syncthreads() ─────────────── read smem[0] ──►
Thread 2:  ─── write smem[2] ────────── __syncthreads() ────────── read smem[3] ──►
                                                ↑
                              All must arrive here before any proceeds
                              → Guarantees smem[0..31] all written before reading
__global__ void sync_example(float* data) {
    __shared__ float smem[256];
    int tid = threadIdx.x;

    // Phase 1: write
    smem[tid] = data[tid];
    __syncthreads();   // ← MUST sync before reading another thread's value

    // Phase 2: read neighbor — safe because all writes completed
    float neighbor = smem[(tid + 1) % blockDim.x];
    __syncthreads();   // ← sync again before next write phase

    // Phase 3: write result
    data[tid] = neighbor;
}
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">Never place <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">__syncthreads()</code> inside a conditional branch that not all threads will take.</span> If some threads skip the barrier while others do not, the behavior is <strong>undefined (未定义行为)</strong> — typically a silent deadlock or data corruption. Use <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">__syncwarp()</code> for warp-level synchronization when only intra-warp coordination is needed.</div>

------

## 5. Classic Use Cases with Full Code (经典使用场景)

### 1) Tiled Matrix Multiplication (分块矩阵乘法)

The most canonical shared memory optimization. Without tiling, every element of A and B is loaded from global memory O(N) times. With tiling, each element is loaded once into shared memory per tile.

```
  C[i][j] = Σ A[i][k] * B[k][j]   for k = 0..N-1

  NAIVE (no tiling):
  ─────────────────────────────────────────────────────────
  For a 1024×1024 matrix:
  Each C element needs 1024 reads from A and 1024 from B
  → Total global reads: 1024³ × 2 ≈ 2 billion reads ❌

  TILED (shared memory):
  ─────────────────────────────────────────────────────────
  Load TILE×TILE blocks of A and B into smem
  Each element in the tile is loaded once and reused TILE times
  → Global reads reduced by factor of TILE (e.g., 32×) ✅

  Visual (TILE_SIZE = 4, showing one tile step):

        B columns →
        ┌───────────────┐
     A  │ B_tile loaded │
  rows  │ into smem     │
   ↓    └───────────────┘
  ┌───┐  ┌─────────────┐
  │A  │  │ C_tile      │
  │til│× │ accumulates │
  │e  │  │ per step    │
  └───┘  └─────────────┘
  Each step: TILE² smem reads instead of TILE² global reads
#define TILE_SIZE 32

__global__ void matmul_tiled(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    // Shared memory tiles for A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    // Sweep tiles along the K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // ── Load tile of A into shared memory ──
        int aCol = t * TILE_SIZE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;

        // ── Load tile of B into shared memory ──
        int bRow = t * TILE_SIZE + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        // ── Sync: wait for all threads to finish loading ──
        __syncthreads();

        // ── Compute partial dot product using shared memory ──
        // TILE_SIZE multiply-adds from fast smem, not global memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // ── Sync: wait before loading next tile ──
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// Launch
dim3 block(TILE_SIZE, TILE_SIZE);
dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
          (M + TILE_SIZE - 1) / TILE_SIZE);
matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
```

------

### 2) Parallel Reduction (并行规约)

Reduce an array to a single sum using shared memory accumulation within each block.

```
  REDUCTION TREE (block of 8 threads, values [3,1,7,2,5,4,8,6]):

  Initial:   3   1   7   2   5   4   8   6     (smem loaded)
  Step 1:    4   .   9   .  13   .  14   .     (+stride 1: T0+=T4, etc.)
  Step 2:   13   .   .   .  27   .   .   .     (+stride 2: T0+=T2, etc.)
  Step 3:   40   .   .   .   .   .   .   .     (+stride 4: T0+=T4 → final sum)
             ↑
             Block partial sum written to global output[blockIdx.x]
  After all blocks: CPU or second kernel sums output[]
__global__ void reduce_sum(const float* input, float* output, int N) {
    extern __shared__ float smem[];   // blockDim.x floats

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Load: each thread loads one element (or 0 if out of bounds)
    smem[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // Reduction tree — halve active threads each step
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 holds the block sum — write to global memory
    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}

// Launch
int blockSize = 256;
int gridSize  = (N + blockSize - 1) / blockSize;
size_t smemBytes = blockSize * sizeof(float);

reduce_sum<<<gridSize, blockSize, smemBytes>>>(d_input, d_partial, N);
// Then reduce d_partial on CPU or call reduce_sum again
```

------

### 3) 1D Stencil / Halo Exchange (一维模板计算 / 光晕交换)

Stencil operations (模板计算) require each thread to read its neighbors. Without shared memory, every element is loaded multiple times from global memory.

```
  7-point stencil: out[i] = in[i-3] + in[i-2] + in[i-1] + in[i]
                           + in[i+1] + in[i+2] + in[i+3]

  TILE of 8 threads, radius = 3:

  Global memory:   ... [i-3][i-2][i-1] | [i][i+1][i+2][i+3][i+4][i+5][i+6][i+7] | [i+8]...
                         ◄──halos──►   |◄───────── tile data ─────────────────────►|◄─halos─►

  Shared memory (tile + halos):
  [ h0  h1  h2 | d0  d1  d2  d3  d4  d5  d6  d7 | h3  h4  h5 ]
    ◄─radius=3─►◄─────── blockDim.x = 8 ──────────►◄─radius=3─►
#define RADIUS 3

__global__ void stencil_1d(const float* in, float* out, int N) {
    // Shared memory: tile data + halo on both sides
    extern __shared__ float smem[];   // (blockDim.x + 2*RADIUS) floats

    int tid  = threadIdx.x;
    int gid  = threadIdx.x + blockIdx.x * blockDim.x;
    int smid = tid + RADIUS;   // Offset into smem to account for left halo

    // Load main tile
    smem[smid] = (gid < N) ? in[gid] : 0.0f;

    // Load left halo (first RADIUS threads load extra elements to the left)
    if (tid < RADIUS) {
        int left = gid - RADIUS;
        smem[smid - RADIUS] = (left >= 0) ? in[left] : 0.0f;
    }

    // Load right halo (last RADIUS threads load extra elements to the right)
    if (tid >= blockDim.x - RADIUS) {
        int right = gid + blockDim.x;
        smem[smid + blockDim.x - tid] =
            (right < N) ? in[right] : 0.0f;
    }

    __syncthreads();

    // Apply stencil — all reads from fast shared memory
    if (gid < N) {
        float result = 0.0f;
        for (int r = -RADIUS; r <= RADIUS; r++) {
            result += smem[smid + r];
        }
        out[gid] = result;
    }
}

// Launch with dynamic smem = (blockSize + 2*RADIUS) floats
int blockSize = 256;
int gridSize  = (N + blockSize - 1) / blockSize;
size_t smemBytes = (blockSize + 2 * RADIUS) * sizeof(float);
stencil_1d<<<gridSize, blockSize, smemBytes>>>(d_in, d_out, N);
```

------

### 4) Histogram (直方图)

Using shared memory to build per-block histograms before atomic-merging into global memory — drastically reducing contention on global atomic operations.

```cpp
#define NUM_BINS 256

__global__ void histogram(const unsigned char* data,
                           unsigned int* global_hist, int N) {
    // Per-block histogram in shared memory
    __shared__ unsigned int local_hist[NUM_BINS];

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Step 1: Zero-initialize local histogram
    if (tid < NUM_BINS) local_hist[tid] = 0;
    __syncthreads();

    // Step 2: Accumulate into shared memory using atomics
    // Much cheaper than global atomics — smem atomics ~4 cycles vs ~600 cycles
    if (idx < N) {
        atomicAdd(&local_hist[data[idx]], 1);
    }
    __syncthreads();

    // Step 3: Merge local histogram into global histogram
    if (tid < NUM_BINS) {
        atomicAdd(&global_hist[tid], local_hist[tid]);
    }
}

// Launch
int blockSize = 256;
int gridSize  = (N + blockSize - 1) / blockSize;
histogram<<<gridSize, blockSize>>>(d_data, d_hist, N);
```

------

## 6. Shared Memory Configuration (共享内存配置)

### 1) L1 vs Shared Memory Split

On Ampere (A100) and newer GPUs, L1 cache and shared memory share the same 228 KB SRAM pool per SM. Configure the split with:

```cpp
// Set preferred shared memory size for a specific kernel
cudaFuncSetAttribute(
    my_kernel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    75   // Percentage: 0–100 (or use cudaSharedmemCarveoutMaxShared = 100)
);

// Or set globally for all kernels
cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);   // Maximize shared mem
cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);       // Maximize L1 cache
cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);    // 50/50 split
```

------

### 2) Maximum Shared Memory per Block

On Ampere, the default limit per block is 48 KB. To use more (up to 228 KB), opt in explicitly:

```cpp
// Request up to 96 KB of shared memory for this kernel
cudaFuncSetAttribute(
    my_kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    96 * 1024   // 96 KB
);

// Then launch with the requested size
kernel<<<grid, block, 96 * 1024>>>(args...);
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Increasing shared memory per block <span style="color:#C0392B;font-weight:600">reduces occupancy (占用率)</span> because fewer blocks can simultaneously reside on an SM. Query the actual limit with <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaDeviceGetAttribute(&val, cudaDevAttrMaxSharedMemoryPerBlockOptin, device)</code>.</div>

------

## 7. Occupancy & Shared Memory Trade-off (占用率与共享内存权衡)

```
  SM resources on A100:
  ─────────────────────────────────
  Max threads per SM:        2048
  Max blocks per SM:           32
  Shared memory per SM:   228 KB
  ─────────────────────────────────

  Example: kernel uses 48 KB smem/block, blockSize = 256 threads

  Blocks limited by smem:  228 KB / 48 KB = 4 blocks
  Threads from 4 blocks:   4 × 256 = 1024 threads
  Occupancy:               1024 / 2048 = 50%

  If reduce to 16 KB smem/block:
  Blocks limited by smem:  228 KB / 16 KB = 14 blocks  (capped at 32)
  Threads from 14 blocks:  14 × 256 = 3584 → capped at 2048
  Occupancy:               2048 / 2048 = 100% ✅

  Rule of thumb: more shared memory per block → fewer concurrent blocks
                 → lower occupancy → less latency hiding
```

Use `cudaOccupancyMaxPotentialBlockSize` to find the sweet spot:

```cpp
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(
    &minGridSize, &blockSize,
    my_kernel,
    smemBytesPerBlock,   // Dynamic smem per block
    0                    // Max block size (0 = no limit)
);
printf("Optimal blockSize: %d, minGridSize: %d\n", blockSize, minGridSize);
```

------

## 8. Nsight Compute — Profiling Shared Memory

```bash
# Check bank conflicts
ncu --metrics \
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
  ./my_kernel

# Ideal: 0 bank conflicts
# High value → add +1 padding or restructure access pattern

# Check shared memory throughput
ncu --metrics \
  l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed \
  ./my_kernel
# > 80% → shared memory is well-utilized
```

------

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> Shared memory is a <strong>programmer-managed L1 cache (程序员管理的一级缓存)</strong> — use it to <strong>tile data reused within a block</strong>, eliminate <strong>bank conflicts with +1 padding</strong>, always guard phase transitions with <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">__syncthreads()</code>, and balance <strong>smem size vs occupancy</strong> to keep the SM's warp scheduler busy hiding latency.</div>
