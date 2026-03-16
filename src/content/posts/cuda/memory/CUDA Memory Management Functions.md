---
title: "CUDA Memory Management Functions"
published: 2026-03-15
description: "CUDA Memory Management Functions"
image: ""
tags: ["cuda","memory","CUDA Memory Management Functions"]
category: cuda / memory
draft: false
lang: ""
---

# I. CUDA Memory Management Functions

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <span style="color:#E8600A;font-weight:700">CUDA memory management</span> involves explicit allocation and deallocation of device memory using <span style="color:#2980B9;font-weight:700">cudaMalloc() and cudaFree()</span>. Data transfer between host and device uses <span style="color:#2980B9;font-weight:700">cudaMemcpy()</span>, while <span style="color:#E8600A;font-weight:700">asynchronous operations with streams</span> enable overlapping computation and data movement for maximum parallelism. </div>

## 1. Memory Allocation and Deallocation

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">cudaMalloc() (设备内存分配)</span> allocates memory on the GPU device. <span style="color:#E8600A;font-weight:700">cudaFree() (设备内存释放)</span> deallocates it. Unlike CPU malloc/free, <span style="color:#C0392B;font-weight:700">failure to free device memory causes memory leaks that persist until program exit</span>. </div>

### 1) Basic Allocation and Freeing

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">cudaMalloc()</span> takes a pointer address and size in bytes. <span style="color:#2980B9;font-weight:700">Always check return values for errors</span>. <span style="color:#2980B9;font-weight:700">Use cudaFree() in reverse order of allocation</span> to avoid fragmentation. </div>

```cpp
// Basic memory allocation example
int main() {
    int n = 1000000;
    size_t bytes = n * sizeof(float);
    
    // Host memory
    float *h_data = (float*)malloc(bytes);
    
    // Device memory - pointer must be to device memory
    float *d_data;
    
    // Allocate device memory
    cudaError_t err = cudaMalloc(&d_data, bytes);
    if (err != cudaSuccess) {
        printf("Allocation failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Use the memory...
    
    // Free device memory - MUST do this!
    cudaFree(d_data);
    
    // Free host memory
    free(h_data);
    
    return 0;
}

// Multiple allocations with error checking
void allocateMultiple() {
    float *d_a, *d_b, *d_c;
    size_t bytes = 1024 * sizeof(float);
    
    // Allocate with error checking
    if (cudaMalloc(&d_a, bytes) != cudaSuccess) {
        printf("Failed to allocate d_a\n");
        return;
    }
    
    if (cudaMalloc(&d_b, bytes) != cudaSuccess) {
        printf("Failed to allocate d_b\n");
        cudaFree(d_a);  // Clean up already allocated
        return;
    }
    
    if (cudaMalloc(&d_c, bytes) != cudaSuccess) {
        printf("Failed to allocate d_c\n");
        cudaFree(d_a);
        cudaFree(d_b);
        return;
    }
    
    // Use memory...
    
    // Free in reverse order
    cudaFree(d_c);
    cudaFree(d_b);
    cudaFree(d_a);
}

// RAII-style wrapper for C++ (simplified)
class CUDABuffer {
    float* d_ptr = nullptr;
    size_t size;

public:
    CUDABuffer(size_t bytes) : size(bytes) {
        if (cudaMalloc(&d_ptr, bytes) != cudaSuccess)
            throw std::runtime_error("CUDA malloc failed");
    }

    ~CUDABuffer() {
        if (d_ptr) cudaFree(d_ptr);
    }

    // 禁用拷贝，防止双重 free
    CUDABuffer(const CUDABuffer&) = delete;
    CUDABuffer& operator=(const CUDABuffer&) = delete;

    // 允许移动
    CUDABuffer(CUDABuffer&& other) noexcept : d_ptr(other.d_ptr), size(other.size) {
        other.d_ptr = nullptr;
    }

    float* get() const { return d_ptr; }
};
```

<div style="background:#F5F5F5;border-left:4px solid #C0392B;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85;font-size:0.80em"><span style="color:#C0392B;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:700">cudaMalloc() aligns memory to 256 bytes</span> by default, which is optimal for coalesced access. Never use host pointers with device memory - they point to different address spaces. </div>

## 2. Memory Copy Operations

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">cudaMemcpy() (内存拷贝)</span> transfers data between host and device. The direction is specified by <span style="color:#2980B9;font-weight:700">cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, or cudaMemcpyDeviceToDevice</span>. It's a <span style="color:#C0392B;font-weight:700">synchronous operation</span> that blocks the CPU until complete. </div>

### 1) Synchronous Memory Copy

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Synchronous memcpy</span> ensures data is transferred before continuing. <span style="color:#2980B9;font-weight:700">Use for simple programs and when data dependencies require completion</span>. </div>

```cpp
// Complete example with host-device-host transfer
int main() {
    int n = 1000000;
    size_t bytes = n * sizeof(float);
    
    // Host arrays
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    
    // Initialize host data
    for (int i = 0; i < n; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }
    
    // Device arrays
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // Copy data from host to device (SYNCHRONOUS)
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
    
    // Copy result back to host (SYNCHRONOUS)
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    // Verify
    for (int i = 0; i < 10; i++) {
        printf("h_c[%d] = %f\n", i, h_c[i]);
    }
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}

// Device-to-device copy
void deviceToDeviceCopy(float* d_src, float* d_dst, size_t bytes) {
    // Copy data entirely on device (no host involvement)
    cudaMemcpy(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice);
}
```

### 2) Asynchronous Memory Copy

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">cudaMemcpyAsync() (异步内存拷贝)</span> returns immediately without waiting for transfer completion. <span style="color:#2980B9;font-weight:700">Requires pinned host memory</span> (allocated with cudaHostAlloc). <span style="color:#2980B9;font-weight:700">Use with streams to overlap operations</span>. </div>

```cpp
// Asynchronous copy requires pinned (page-locked) memory
int main() {
    int n = 1000000;
    size_t bytes = n * sizeof(float);
    
    // Allocate pinned host memory (required for async)
    float *h_a, *h_b, *h_c;
    cudaHostAlloc(&h_a, bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_b, bytes, cudaHostAllocDefault);
    cudaHostAlloc(&h_c, bytes, cudaHostAllocDefault);
    
    // Initialize
    for (int i = 0; i < n; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }
    
    // Device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // Create stream for async operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Asynchronous copies (non-blocking)
    cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, h_b, bytes, cudaMemcpyHostToDevice, stream);
    
    // Launch kernel in same stream (executes after copies complete)
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize, 0, stream>>>(d_a, d_b, d_c, n);
    
    // Asynchronous copy back
    cudaMemcpyAsync(h_c, d_c, bytes, cudaMemcpyDeviceToHost, stream);
    
    // Do other CPU work while GPU is busy...
    cpuWork();
    
    // Wait for stream to complete
    cudaStreamSynchronize(stream);
    
    // Now h_c is ready
    for (int i = 0; i < 10; i++) {
        printf("h_c[%d] = %f\n", i, h_c[i]);
    }
    
    // Cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    
    return 0;
}
```

## 3. Streams for Concurrent Operations

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">CUDA streams (流)</span> are sequences of operations that execute in order on the GPU. <span style="color:#2980B9;font-weight:700">Different streams can execute concurrently</span>, enabling overlap of computation, data transfer, and CPU work. </div>

### 1) Multiple Streams for Parallelism

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Multiple streams</span> allow concurrent kernel execution and overlap with data transfers. <span style="color:#2980B9;font-weight:700">Use when processing independent data chunks</span> to maximize GPU utilization. </div>

```cpp
// Processing data in multiple streams for parallelism
#define N_STREAMS 4
#define N 10000000  // 10M elements
#define CHUNK_SIZE (N / N_STREAMS)

int main() {
    size_t bytes = N * sizeof(float);
    size_t chunk_bytes = CHUNK_SIZE * sizeof(float);
    
    // Pinned host memory
    float *h_data;
    cudaHostAlloc(&h_data, bytes, cudaHostAllocDefault);
    initialize(h_data, N);
    
    // Device memory
    float *d_data;
    cudaMalloc(&d_data, bytes);
    
    // Create streams
    cudaStream_t streams[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Process each chunk in its own stream
    for (int i = 0; i < N_STREAMS; i++) {
        int offset = i * CHUNK_SIZE;
        
        // Asynchronous copy of this chunk
        cudaMemcpyAsync(d_data + offset, h_data + offset, 
                       chunk_bytes, cudaMemcpyHostToDevice, 
                       streams[i]);
        
        // Kernel launch in same stream
        int blockSize = 256;
        int gridSize = (CHUNK_SIZE + blockSize - 1) / blockSize;
        processKernel<<<gridSize, blockSize, 0, streams[i]>>>(
            d_data + offset, CHUNK_SIZE);
        
        // Asynchronous copy back
        cudaMemcpyAsync(h_data + offset, d_data + offset,
                       chunk_bytes, cudaMemcpyDeviceToHost,
                       streams[i]);
    }
    
    // All streams are now running concurrently
    // Do CPU work while GPU processes
    
    // Wait for all streams
    for (int i = 0; i < N_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    // Cleanup
    for (int i = 0; i < N_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(d_data);
    cudaFreeHost(h_data);
    
    return 0;
}
```

### 2) Stream Priorities and Events

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;line-height:1.85"> <span style="color:#E8600A;font-weight:700">Stream priorities</span> allow high-priority streams to execute first. <span style="color:#E8600A;font-weight:700">Events</span> enable synchronization between streams. <span style="color:#2980B9;font-weight:700">Use events to coordinate dependencies</span> across streams. </div>

```cpp
// Stream priorities and events
int main() {
    cudaStream_t high_prio, low_prio;
    
    // Get priority range
    int lowest, highest;
    cudaDeviceGetStreamPriorityRange(&lowest, &highest);
    
    // Create streams with different priorities
    cudaStreamCreateWithPriority(&high_prio, cudaStreamNonBlocking, highest);
    cudaStreamCreateWithPriority(&low_prio, cudaStreamNonBlocking, lowest);
    
    // Create events for synchronization
    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
    
    // Record events in streams
    kernel1<<<grid, block, 0, high_prio>>>(d_data);
    cudaEventRecord(event1, high_prio);  // Mark point in high_prio stream
    
    // Make low_prio stream wait for event
    cudaStreamWaitEvent(low_prio, event1, 0);
    
    // This kernel in low_prio waits for kernel1 to complete
    kernel2<<<grid, block, 0, low_prio>>>(d_data);
    
    // Measure time between events
    cudaEventRecord(event2, low_prio);
    cudaEventSynchronize(event2);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, event1, event2);
    printf("Time between events: %f ms\n", milliseconds);
    
    // Cleanup
    cudaEventDestroy(event1);
    cudaEventDestroy(event2);
    cudaStreamDestroy(high_prio);
    cudaStreamDestroy(low_prio);
}
```

## Memory Management Comparison

<table style="width:100%; border-collapse: collapse; margin: 20px 0;">
  <tr style="background: #f2f2f2;">
    <th style="padding: 10px; border: 1px solid #ddd;">Function</th>
    <th style="padding: 10px; border: 1px solid #ddd;">Type</th>
    <th style="padding: 10px; border: 1px solid #ddd;">Memory Type</th>
    <th style="padding: 10px; border: 1px solid #ddd;">Behavior</th>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><span style="color:#2980B9; font-weight:700">cudaMalloc</span></td>
    <td style="padding: 10px; border: 1px solid #ddd;">Allocation</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Device</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Synchronous</td>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><span style="color:#E8600A; font-weight:700">cudaHostAlloc</span></td>
    <td style="padding: 10px; border: 1px solid #ddd;">Allocation</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Pinned Host</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Synchronous</td>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><span style="color:#2980B9; font-weight:700">cudaMemcpy</span></td>
    <td style="padding: 10px; border: 1px solid #ddd;">Copy</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Host/Device</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Synchronous</td>
  </tr>
  <tr>
    <td style="padding: 10px; border: 1px solid #ddd;"><span style="color:#E8600A; font-weight:700">cudaMemcpyAsync</span></td>
    <td style="padding: 10px; border: 1px solid #ddd;">Copy</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Pinned Host/Device</td>
    <td style="padding: 10px; border: 1px solid #ddd;">Asynchronous</td>
  </tr>
</table>

## Common Patterns and Best Practices

```cpp
// Comprehensive example showing all concepts
#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("Error at %s:%d - %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Kernel that processes data
__global__ void processData(float* data, int n, float factor) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        data[idx] = data[idx] * factor;
    }
}

int main() {
    const int N = 1 << 20;  // 1M elements
    const int N_STREAMS = 4;
    const int CHUNK_SIZE = N / N_STREAMS;
    size_t bytes = N * sizeof(float);
    size_t chunk_bytes = CHUNK_SIZE * sizeof(float);
    
    // Allocate pinned host memory
    float *h_data;
    CHECK_CUDA(cudaHostAlloc(&h_data, bytes, cudaHostAllocDefault));
    
    // Initialize
    for (int i = 0; i < N; i++) {
        h_data[i] = i * 1.0f;
    }
    
    // Allocate device memory
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, bytes));
    
    // Create streams
    cudaStream_t streams[N_STREAMS];
    for (int i = 0; i < N_STREAMS; i++) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }
    
    // Launch processing in multiple streams
    for (int i = 0; i < N_STREAMS; i++) {
        int offset = i * CHUNK_SIZE;
        
        // Async copy to device
        CHECK_CUDA(cudaMemcpyAsync(d_data + offset, h_data + offset,
                                   chunk_bytes, cudaMemcpyHostToDevice,
                                   streams[i]));
        
        // Kernel launch
        int blockSize = 256;
        int gridSize = (CHUNK_SIZE + blockSize - 1) / blockSize;
        processData<<<gridSize, blockSize, 0, streams[i]>>>(
            d_data + offset, CHUNK_SIZE, 2.0f);
        
        // Async copy back
        CHECK_CUDA(cudaMemcpyAsync(h_data + offset, d_data + offset,
                                   chunk_bytes, cudaMemcpyDeviceToHost,
                                   streams[i]));
    }
    
    // CPU can do other work here
    
    // Wait for all streams
    for (int i = 0; i < N_STREAMS; i++) {
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
    
    // Verify
    for (int i = 0; i < 10; i++) {
        printf("h_data[%d] = %f\n", i, h_data[i]);
    }
    
    // Cleanup
    for (int i = 0; i < N_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(d_data);
    cudaFreeHost(h_data);
    
    return 0;
}
```

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> <span style="color:#E8600A;font-weight:700">Use cudaMalloc/cudaFree for device memory, cudaMemcpy for simple transfers</span>, but <span style="color:#2980B9;font-weight:700">switch to pinned memory and cudaMemcpyAsync with streams</span> to overlap data movement with computation for maximum throughput. </div>
