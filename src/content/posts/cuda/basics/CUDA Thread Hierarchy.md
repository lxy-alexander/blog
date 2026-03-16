---
title: "CUDA Thread Hierarchy"
published: 2026-03-14
description: "CUDA Thread Hierarchy"
image: ""
tags: ["cuda","basics","CUDA Thread Hierarchy"]
category: cuda / basics
draft: false
lang: ""
---

# I. CUDA Thread Hierarchy

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> CUDA organizes threads in a <span style="color:#E8600A;font-weight:700">three-level hierarchy</span> (三级层次结构): <span style="color:#2980B9;font-weight:700">Grid → Block → Thread</span>. This hierarchy enables <span style="color:#2980B9;font-weight:700">scalable parallelism</span> across different GPU architectures and allows threads to be <span style="color:#2980B9;font-weight:700">organized in 1D, 2D, or 3D</span> to match data shapes. </div>

## 1. Grid-Block-Thread Structure

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> The <span style="color:#E8600A;font-weight:700">thread hierarchy</span> (线程层次结构) consists of <span style="color:#2980B9;font-weight:700">threads grouped into blocks</span>, and <span style="color:#2980B9;font-weight:700">blocks grouped into grids</span>. This two-level organization enables both <span style="color:#2980B9;font-weight:700">intra-block cooperation</span> and <span style="color:#2980B9;font-weight:700">scalability across GPU cores</span>. </div>

### 1) Grid, Block, and Thread Relationship

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> A <span style="color:#E8600A;font-weight:700">grid</span> (网格) contains multiple <span style="color:#E8600A;font-weight:700">blocks</span> (线程块), and each block contains multiple <span style="color:#E8600A;font-weight:700">threads</span> (线程). Threads in the same block can <span style="color:#2980B9;font-weight:700">cooperate via shared memory and synchronize</span>. <span style="color:#2980B9;font-weight:700">Use when</span> designing parallel algorithms that require both independent and cooperative execution. </div>

```cpp
// Visual representation of thread hierarchy
__global__ void demonstrateHierarchy() {
    // Each thread has unique identifiers
    int threadId = threadIdx.x;                    // Position within block
    int blockId = blockIdx.x;                       // Block index in grid
    int globalId = blockId * blockDim.x + threadId; // Global thread index
    
    printf("Grid %p | Block %d (%d threads) | Thread %d | Global ID: %d\n",
           gridDim, blockId, blockDim.x, threadId, globalId);
}

// Launch configuration: 3 blocks, each with 5 threads
// Total threads = 3 * 5 = 15 threads
demonstrateHierarchy<<<3, 5>>>();
cudaDeviceSynchronize();

/* Sample output (order may vary):
Grid 0x3 | Block 0 (5 threads) | Thread 0 | Global ID: 0
Grid 0x3 | Block 0 (5 threads) | Thread 1 | Global ID: 1
...
Grid 0x3 | Block 2 (5 threads) | Thread 4 | Global ID: 14
*/
```

<div style="background:#F5F5F5;border-left:4px solid #C0392B;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85;font-size:0.80em"><span style="color:#C0392B;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:700">Threads in different blocks cannot synchronize or directly communicate</span>. They must communicate through global memory, which is much slower than shared memory within a block. </div>

## 2. Multi-dimensional Thread Organization

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> Threads and blocks can be organized in <span style="color:#E8600A;font-weight:700">1D, 2D, or 3D</span> (一维/二维/三维) using `dim3` types. This <span style="color:#2980B9;font-weight:700">matches natural data structures</span> like vectors (1D), matrices (2D), or volumes (3D), simplifying index calculations. </div>

### 1) 1D Organization

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#2980B9;font-weight:700">1D organization</span> uses a single dimension for both grid and blocks. <span style="color:#E8600A;font-weight:700">Use when</span> processing linear data like arrays, lists, or vectors where each element is independent. </div>

```cpp
// 1D grid of 1D blocks - for array processing
__global__ void vectorProcess(float* data, int N) {
    // Simple 1D global index calculation
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < N) {
        data[idx] = data[idx] * 2.0f;
    }
}

// Launch 1D configuration
int N = 1000000;
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;  // Ceiling division

// Result: blocksPerGrid = 3907 blocks (3907 * 256 ≈ 1,000,000)
vectorProcess<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
```

### 2) 2D Organization

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#2980B9;font-weight:700">2D organization</span> uses two dimensions for blocks and/or grids. <span style="color:#E8600A;font-weight:700">Use when</span> processing matrix data, images, or any 2D grid structure where row and column indices are natural. </div>

```cpp
// 2D grid of 2D blocks - for image processing
__global__ void imageProcess(float* image, int width, int height) {
    // Calculate 2D coordinates
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        // Convert 2D to 1D index (row-major)
        int idx = row * width + col;
        image[idx] = image[idx] * 2.0f;
    }
}

// Launch 2D configuration for 1920x1080 image
dim3 threadsPerBlock(16, 16);  // 256 threads per block
dim3 blocksPerGrid(
    (1920 + threadsPerBlock.x - 1) / threadsPerBlock.x,  // 120 blocks in x
    (1080 + threadsPerBlock.y - 1) / threadsPerBlock.y   // 68 blocks in y
);  // Total blocks = 120 * 68 = 8160 blocks

imageProcess<<<blocksPerGrid, threadsPerBlock>>>(d_image, 1920, 1080);
```

### 3) 3D Organization

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#2980B9;font-weight:700">3D organization</span> uses three dimensions for thread organization. <span style="color:#E8600A;font-weight:700">Use when</span> processing volumetric data like CT scans, 3D simulations, or voxel grids. </div>

```cpp
// 3D grid of 3D blocks - for volume processing
__global__ void volumeProcess(float* volume, int dimX, int dimY, int dimZ) {
    // Calculate 3D coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < dimX && y < dimY && z < dimZ) {
        // Convert 3D to 1D index (x fastest, then y, then z)
        int idx = (z * dimY + y) * dimX + x;
        volume[idx] = volume[idx] * 2.0f;
    }
}

// Launch 3D configuration for 256x256x256 volume
dim3 threadsPerBlock(8, 8, 4);  // 256 threads per block (8*8*4)
dim3 blocksPerGrid(
    (256 + threadsPerBlock.x - 1) / threadsPerBlock.x,  // 32 blocks
    (256 + threadsPerBlock.y - 1) / threadsPerBlock.y,  // 32 blocks
    (256 + threadsPerBlock.z - 1) / threadsPerBlock.z   // 64 blocks
);  // Total blocks = 32 * 32 * 64 = 65,536 blocks

volumeProcess<<<blocksPerGrid, threadsPerBlock>>>(d_volume, 256, 256, 256);
```

## 3. Warp

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> A <span style="color:#E8600A;font-weight:700">warp</span> (线程束) is the <span style="color:#2980B9;font-weight:700">fundamental execution unit</span> in CUDA - a group of 32 threads that execute together in lockstep on a Streaming Multiprocessor. Understanding warps is crucial for <span style="color:#2980B9;font-weight:700">performance optimization</span>. </div>

### 1) Warp Execution Model

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> Threads within a warp <span style="color:#2980B9;font-weight:700">execute the same instruction simultaneously</span> (SIMD - Single Instruction Multiple Data). When threads take different paths (divergence), they execute serially. <span style="color:#2980B9;font-weight:700">Use when</span> optimizing code to maintain uniform execution paths within warps. </div>

```cpp
__global__ void demonstrateWarpBehavior(float* data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int laneId = threadIdx.x % 32;  // Position within warp (0-31)
    int warpId = threadIdx.x / 32;   // Which warp this thread belongs to
    
    if (idx < N) {
        // Good: All threads in warp take same path (no divergence)
        data[idx] = data[idx] * 2.0f;
        
        // Bad: Warp divergence - threads with laneId < 16 take different path
        // than threads with laneId >= 16, causing serialization
        if (laneId < 16) {
            data[idx] += 1.0f;  // First half of warp executes
        } else {
            data[idx] -= 1.0f;  // Second half executes separately
        }
        
        // Better: Use predicated execution or restructure to avoid divergence
        // For example, process data in warp-uniform way:
        float factor = (laneId < 16) ? 1.0f : -1.0f;
        data[idx] += factor;  // Same instruction, different data
    }
}
```

<div style="background:#F5F5F5;border-left:4px solid #C0392B;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85;font-size:0.80em"><span style="color:#C0392B;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:700">Warp divergence kills performance</span>. When threads in a warp take different branches, each branch path executes serially, potentially wasting up to 32× execution time. <span style="color:#C0392B;font-weight:700">Always design algorithms to be warp-uniform</span> when possible. </div>

### 2) Warp-Level Optimizations

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#2980B9;font-weight:700">Warp-level programming</span> leverages the fact that 32 threads execute together. CUDA provides <span style="color:#E8600A;font-weight:700">warp shuffle instructions</span> for efficient communication. <span style="color:#2980B9;font-weight:700">Use when</span> implementing reductions, scans, or other cooperative algorithms. </div>

```cpp
__global__ void warpReduce(float* data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float val = data[idx];
    
    // Warp-level reduction using shuffle (faster than shared memory)
    // Each warp reduces its 32 values to 1
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    
    // Thread 0 of each warp now has the sum
    if (threadIdx.x % 32 == 0) {
        // Store warp result in shared memory for block-level reduction
        extern __shared__ float shared[];
        shared[threadIdx.x / 32] = val;
    }
    
    __syncthreads();
    
    // Final block-level reduction...
}
```

## 4. Streaming Multiprocessor (SM)

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> A <span style="color:#E8600A;font-weight:700">Streaming Multiprocessor</span> (流多处理器) is the <span style="color:#2980B9;font-weight:700">physical hardware unit</span> that executes warps. Each SM has its own shared memory, registers, and schedulers. Understanding SM capabilities helps <span style="color:#2980B9;font-weight:700">optimize resource usage</span>. </div>

### 1) SM Architecture and Occupancy

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">SMs</span> execute multiple blocks concurrently, with resources divided among them. <span style="color:#E8600A;font-weight:700">Occupancy</span> (占用率) measures how many warps are active on an SM. <span style="color:#2980B9;font-weight:700">Use when</span> tuning kernel launch parameters for maximum hardware utilization. </div>

```cpp
// Helper function to calculate theoretical occupancy
void calculateOccupancy(int threadsPerBlock, int sharedMemPerBlock) {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    
    printf("Device: %s\n", props.name);
    printf("Max threads per block: %d\n", props.maxThreadsPerBlock);
    printf("Max threads per SM: %d\n", props.maxThreadsPerMultiProcessor);
    printf("Max blocks per SM: %d\n", props.maxBlocksPerMultiProcessor);
    printf("Shared memory per SM: %zu KB\n", props.sharedMemPerMultiprocessor / 1024);
    
    // Calculate maximum warps per SM
    int warpsPerSM = props.maxThreadsPerMultiProcessor / 32;
    int warpsPerBlock = (threadsPerBlock + 31) / 32;
    int blocksPerSM = min(props.maxBlocksPerMultiProcessor, 
                          warpsPerSM / warpsPerBlock);
    int activeWarps = blocksPerSM * warpsPerBlock;
    float occupancy = (float)activeWarps / warpsPerSM * 100.0f;
    
    printf("With %d threads/block: %d warps/block\n", 
           threadsPerBlock, warpsPerBlock);
    printf("Estimated blocks per SM: %d\n", blocksPerSM);
    printf("Active warps per SM: %d/%d (%.1f%% occupancy)\n", 
           activeWarps, warpsPerSM, occupancy);
}

// Usage: calculateOccupancy(256, 0);
```

### 2) Resource Limits and Block Scheduling

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> SMs have <span style="color:#2980B9;font-weight:700">limited resources</span>: registers, shared memory, and thread slots. Blocks are <span style="color:#2980B9;font-weight:700">scheduled onto SMs</span> until resources are exhausted. <span style="color:#2980B9;font-weight:700">Use when</span> balancing block size and resource usage for maximum parallelism. </div>

```cpp
// Kernel that demonstrates resource usage
__global__ void resourceDemo(float* data) {
    // Each thread uses registers
    float local1 = data[threadIdx.x];
    float local2 = local1 * 2.0f;
    
    // Shared memory usage (per block)
    __shared__ float shared[256];
    
    // Thread cooperation
    shared[threadIdx.x] = local2;
    __syncthreads();
    
    // Warp-level operations
    float sum = 0.0f;
    for (int i = 0; i < 32; i++) {
        sum += shared[threadIdx.x / 32 * 32 + i];
    }
    
    data[threadIdx.x] = sum;
}

// Resource considerations:
// - Block size: 256 threads
// - Shared memory: 256 * 4 = 1024 bytes per block
// - Registers: depends on compiler (check with --ptxas-options=-v)
// - These determine how many blocks can run simultaneously on each SM
```

## Thread Hierarchy Comparison

<div style="margin:20px 0">

| Level                                     | <span style="color:#2980B9">Logical Unit</span>   | <span style="color:#E8600A">Physical Unit</span> | <span style="color:#2980B9">Cooperation Scope</span> | <span style="color:#E8600A">Size Limit</span> |
| ----------------------------------------- | ------------------------------------------------- | ------------------------------------------------ | ---------------------------------------------------- | --------------------------------------------- |
| <span style="color:#2980B9">Grid</span>   | <span style="color:#E8600A">All blocks</span>     | Multiple SMs                                     | Global memory only                                   | Up to 2³¹-1 blocks                            |
| <span style="color:#E8600A">Block</span>  | <span style="color:#2980B9">Thread group</span>   | Single SM                                        | Shared memory + sync                                 | 1024 threads                                  |
| <span style="color:#2980B9">Warp</span>   | <span style="color:#E8600A">32 threads</span>     | SIMD unit                                        | Shuffle instructions                                 | 32 threads                                    |
| <span style="color:#E8600A">Thread</span> | <span style="color:#2980B9">Execution unit</span> | CUDA core                                        | Local variables                                      | N/A                                           |

</div>

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> <span style="color:#E8600A;font-weight:700">CUDA thread hierarchy</span> organizes work in <span style="color:#2980B9;font-weight:700">Grids→Blocks→Threads</span> with 1D/2D/3D flexibility, executes in <span style="color:#2980B9;font-weight:700">32-thread warps</span> on <span style="color:#2980B9;font-weight:700">Streaming Multiprocessors</span>, requiring careful resource management for optimal occupancy. </div>
