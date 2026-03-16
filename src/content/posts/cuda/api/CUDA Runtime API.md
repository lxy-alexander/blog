---
title: "CUDA Runtime API"
published: 2026-03-08
description: "CUDA Runtime API"
image: ""
tags: ["cuda","CUDA Runtime API"]
category: cuda
draft: false
lang: ""
---

# **I. CUDA Runtime API — Top 30 Most-Used Interfaces**

<div style="background:#EBF0FF;border-left:4px solid #3B5BDB;border-radius:0 6px 6px 0;padding:14px 18px;margin:16px 0;line-height:1.9"> <strong>Overview:</strong> The <strong>CUDA Runtime API (运行时接口)</strong> contains 200+ functions organized into categories: device management, memory management, streams, events, unified memory, and CUDA graphs. All functions return a <code style="background:#E8F4FD;color:#1a3a5c;border-radius:4px;padding:1px 6px">cudaError_t</code> status code — always check it in production code. This reference covers the <strong>30 most commonly used APIs</strong>, grouped by category, each with signature, explanation, and usage example. </div>

------

## 1. Device Management (设备管理)

### 1) `cudaGetDeviceCount`

```cpp
cudaError_t cudaGetDeviceCount(int* count);
```

<span style="color:#2980B9">Returns</span> the number of CUDA-capable devices (支持 CUDA 的设备数量) available on the host.

```cpp
int deviceCount = 0;
cudaGetDeviceCount(&deviceCount);
printf("Found %d CUDA device(s)\n", deviceCount);
```

------

### 2) `cudaSetDevice`

```cpp
cudaError_t cudaSetDevice(int device);
```

<span style="color:#2980B9">Sets</span> the active GPU (当前活动 GPU) for the calling host thread. All subsequent CUDA calls will target this device.

```cpp
cudaSetDevice(0);   // Use GPU 0
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> In multi-GPU systems, call <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaSetDevice</code> before any memory allocation or kernel launch. <span style="color:#C0392B;font-weight:600">Memory allocated on one device cannot be directly accessed from another</span> (unless Peer Access is enabled).</div>

------

### 3) `cudaGetDevice`

```cpp
cudaError_t cudaGetDevice(int* device);
```

<span style="color:#2980B9">Returns</span> the index of the currently active device (当前活动设备索引) for the calling host thread.

```cpp
int currentDevice;
cudaGetDevice(&currentDevice);
printf("Currently on GPU %d\n", currentDevice);
```

------

### 4) `cudaGetDeviceProperties`

```cpp
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device);
```

<span style="color:#2980B9">Fills</span> a `cudaDeviceProp` struct with hardware properties (硬件属性) of the specified device — SM count, memory size, warp size, compute capability, etc.

```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("Device: %s\n", prop.name);
printf("Compute capability: %d.%d\n", prop.major, prop.minor);
printf("Total global memory: %.2f GB\n",
       prop.totalGlobalMem / 1e9);
printf("SM count: %d\n", prop.multiProcessorCount);
printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
```

------

### 5) `cudaDeviceSynchronize`

```cpp
cudaError_t cudaDeviceSynchronize(void);
```

<span style="color:#2980B9">Blocks the host thread</span> until all previously issued CUDA work on the current device has completed (等待设备上所有任务完成).

```cpp
kernel<<<grid, block>>>(d_data);
cudaDeviceSynchronize();   // Wait for kernel to finish
// Safe to read results now
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> <span style="color:#C0392B;font-weight:600">Avoid in performance-critical code</span> — it serializes the CPU and GPU. Prefer stream-based synchronization (<code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaStreamSynchronize</code>) or event-based synchronization for fine-grained control.</div>

------

## 2. Memory Management (内存管理)

### 6) `cudaMalloc`

```cpp
cudaError_t cudaMalloc(void** devPtr, size_t size);
```

<span style="color:#2980B9">Allocates</span> `size` bytes of linear memory (线性显存) on the current device. The pointer stored in `devPtr` points to device memory.

```cpp
float* d_array;
size_t N = 1024;
cudaMalloc(&d_array, N * sizeof(float));

// ... use d_array in kernels ...

cudaFree(d_array);   // Always free when done
```

------

### 7) `cudaFree`

```cpp
cudaError_t cudaFree(void* devPtr);
```

<span style="color:#2980B9">Frees</span> device memory (释放显存) previously allocated by `cudaMalloc` or `cudaMallocPitch`.

```cpp
cudaFree(d_array);
d_array = nullptr;   // Good practice to null the pointer
```

------

### 8) `cudaMemcpy`

```cpp
cudaError_t cudaMemcpy(void* dst, const void* src,
                        size_t count, cudaMemcpyKind kind);
```

<span style="color:#2980B9">Copies</span> `count` bytes between host and device memory (主机与设备间的内存拷贝). <span style="color:#E8600A;font-weight:700">Synchronous</span> — blocks the host until the copy completes.

| `cudaMemcpyKind`                                             | Direction                          |
| ------------------------------------------------------------ | ---------------------------------- |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaMemcpyHostToDevice</code> | CPU → GPU                          |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaMemcpyDeviceToHost</code> | GPU → CPU                          |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaMemcpyDeviceToDevice</code> | GPU → GPU (same or peer)           |
| <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaMemcpyDefault</code> | Infer direction from pointer types |

```cpp
float h_data[1024] = {1.0f, 2.0f, /*...*/};
float* d_data;
cudaMalloc(&d_data, 1024 * sizeof(float));

// Host → Device
cudaMemcpy(d_data, h_data, 1024 * sizeof(float), cudaMemcpyHostToDevice);

// Device → Host
cudaMemcpy(h_data, d_data, 1024 * sizeof(float), cudaMemcpyDeviceToHost);
```

------

### 9) `cudaMemcpyAsync`

```cpp
cudaError_t cudaMemcpyAsync(void* dst, const void* src,
                             size_t count, cudaMemcpyKind kind,
                             cudaStream_t stream = 0);
```

<span style="color:#E8600A;font-weight:700">Asynchronous</span> version of `cudaMemcpy` — returns immediately and performs the copy in the specified stream (异步内存拷贝). Requires the host buffer to be <span style="color:#E8600A;font-weight:700">pinned memory (锁页内存)</span>.

```cpp
float* h_pinned;
cudaMallocHost(&h_pinned, N * sizeof(float));   // Pinned allocation

cudaStream_t stream;
cudaStreamCreate(&stream);

// Copy overlaps with other GPU work on the same stream
cudaMemcpyAsync(d_data, h_pinned, N * sizeof(float),
                cudaMemcpyHostToDevice, stream);
kernel<<<grid, block, 0, stream>>>(d_data);   // Queued after the copy
```

------

### 10) `cudaMemset`

```cpp
cudaError_t cudaMemset(void* devPtr, int value, size_t count);
```

<span style="color:#2980B9">Sets</span> `count` bytes of device memory to `value` (将显存初始化为指定字节值). Commonly used to zero-initialize arrays.

```cpp
float* d_output;
cudaMalloc(&d_output, N * sizeof(float));
cudaMemset(d_output, 0, N * sizeof(float));   // Zero-initialize
```

------

### 11) `cudaMallocHost` (Pinned Memory 锁页内存)

```cpp
cudaError_t cudaMallocHost(void** ptr, size_t size);
```

<span style="color:#2980B9">Allocates</span> page-locked (pinned) host memory (锁页主机内存). The OS guarantees this memory is never paged out, enabling <span style="color:#E8600A;font-weight:700">DMA transfers (直接内存访问)</span> and higher bandwidth for `cudaMemcpyAsync`.

```cpp
float* h_pinned;
cudaMallocHost(&h_pinned, N * sizeof(float));

// ... fill h_pinned, use with cudaMemcpyAsync ...

cudaFreeHost(h_pinned);   // Free with cudaFreeHost, NOT free()
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Pinned memory improves transfer bandwidth by 2–3× but <span style="color:#C0392B;font-weight:600">reduces available system RAM for other processes.</span> Only pin memory used for frequent DMA transfers — do not pin all host allocations.</div>

------

### 12) `cudaFreeHost`

```cpp
cudaError_t cudaFreeHost(void* ptr);
```

<span style="color:#2980B9">Frees</span> pinned host memory (释放锁页主机内存) allocated by `cudaMallocHost`. <span style="color:#C0392B;font-weight:600">Never use `free()` for pinned allocations.</span>

```cpp
cudaFreeHost(h_pinned);
```

------

### 13) `cudaMemcpy2D`

```cpp
cudaError_t cudaMemcpy2D(
    void* dst, size_t dpitch,
    const void* src, size_t spitch,
    size_t width, size_t height,
    cudaMemcpyKind kind);
```

<span style="color:#2980B9">Copies</span> a 2D region of memory (二维内存拷贝), respecting row pitch (行间距) — useful for pitched (padded) 2D arrays allocated with `cudaMallocPitch`.

```cpp
// Copy a 64×64 float matrix from host to device
size_t pitch;
float* d_matrix;
cudaMallocPitch(&d_matrix, &pitch, 64 * sizeof(float), 64);

cudaMemcpy2D(d_matrix, pitch,
             h_matrix, 64 * sizeof(float),
             64 * sizeof(float), 64,
             cudaMemcpyHostToDevice);
```

------

## 3. Streams (流)

### 14) `cudaStreamCreate`

```cpp
cudaError_t cudaStreamCreate(cudaStream_t* pStream);
```

<span style="color:#2980B9">Creates</span> an asynchronous stream (异步流) — a queue of GPU operations that execute in order. Operations in different streams can run concurrently.

```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// These two kernels may overlap on the GPU
kernel_A<<<grid, block, 0, stream1>>>(d_a);
kernel_B<<<grid, block, 0, stream2>>>(d_b);

cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);

cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

------

### 15) `cudaStreamDestroy`

```cpp
cudaError_t cudaStreamDestroy(cudaStream_t stream);
```

<span style="color:#2980B9">Destroys</span> a stream and frees its resources (销毁流并释放资源). Waits for all queued operations to complete before destroying.

```cpp
cudaStreamDestroy(stream);
```

------

### 16) `cudaStreamSynchronize`

```cpp
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
```

<span style="color:#2980B9">Blocks the host</span> until all operations queued in the specified stream have completed (等待指定流中所有操作完成). Finer-grained than `cudaDeviceSynchronize`.

```cpp
kernel<<<grid, block, 0, stream>>>(d_data);
cudaMemcpyAsync(h_result, d_data, size, cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);   // Wait only for this stream
// h_result is now ready
```

------

### 17) `cudaStreamWaitEvent`

```cpp
cudaError_t cudaStreamWaitEvent(cudaStream_t stream,
                                 cudaEvent_t event,
                                 unsigned int flags = 0);
```

<span style="color:#2980B9">Makes</span> all future work submitted to `stream` wait until `event` is recorded (流间依赖：让一个流等待另一个流的事件). Enables cross-stream synchronization without blocking the host.

```cpp
// stream2 waits for stream1's kernel before starting
kernel_A<<<grid, block, 0, stream1>>>(d_a);
cudaEventRecord(eventA, stream1);

cudaStreamWaitEvent(stream2, eventA);   // stream2 waits here
kernel_B<<<grid, block, 0, stream2>>>(d_b);   // Uses output of kernel_A
```

------

### 18) `cudaStreamCreateWithPriority`

```cpp
cudaError_t cudaStreamCreateWithPriority(
    cudaStream_t* pStream,
    unsigned int flags,
    int priority);
```

<span style="color:#2980B9">Creates</span> a stream with a specified priority (带优先级的流). Lower numerical value = higher priority. High-priority streams preempt low-priority streams on SMs.

```cpp
int leastPriority, greatestPriority;
cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);

cudaStream_t highPriorityStream;
cudaStreamCreateWithPriority(&highPriorityStream,
                              cudaStreamNonBlocking,
                              greatestPriority);
```

------

## 4. Events (事件)

### 19) `cudaEventCreate`

```cpp
cudaError_t cudaEventCreate(cudaEvent_t* event);
```

<span style="color:#2980B9">Creates</span> a CUDA event (CUDA 事件) — a timestamp marker that can be inserted into a stream's operation queue for timing or synchronization.

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
```

------

### 20) `cudaEventRecord`

```cpp
cudaError_t cudaEventRecord(cudaEvent_t event,
                             cudaStream_t stream = 0);
```

<span style="color:#2980B9">Records</span> an event in a stream (在流中记录事件时间戳). The event captures a timestamp when the GPU reaches that point in the stream.

```cpp
cudaEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>(d_data);
cudaEventRecord(stop, stream);

cudaEventSynchronize(stop);   // Wait for the stop event

float ms;
cudaEventElapsedTime(&ms, start, stop);
printf("Kernel took %.3f ms\n", ms);
```

------

### 21) `cudaEventSynchronize`

```cpp
cudaError_t cudaEventSynchronize(cudaEvent_t event);
```

<span style="color:#2980B9">Blocks the host</span> until the event has been recorded (等待事件完成并阻塞主机). All GPU work up to the `cudaEventRecord` call is guaranteed complete.

```cpp
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);   // Blocks until GPU reaches this event
```

------

### 22) `cudaEventElapsedTime`

```cpp
cudaError_t cudaEventElapsedTime(float* ms,
                                  cudaEvent_t start,
                                  cudaEvent_t end);
```

<span style="color:#2980B9">Computes</span> the elapsed time in milliseconds (计算两个事件之间的耗时，单位毫秒) between two recorded events. Standard method for GPU kernel profiling (GPU 性能分析).

```cpp
float elapsed_ms;
cudaEventElapsedTime(&elapsed_ms, start, stop);
printf("Elapsed: %.3f ms\n", elapsed_ms);
```

------

### 23) `cudaEventDestroy`

```cpp
cudaError_t cudaEventDestroy(cudaEvent_t event);
```

<span style="color:#2980B9">Destroys</span> a CUDA event and frees its resources (销毁事件并释放资源).

```cpp
cudaEventDestroy(start);
cudaEventDestroy(stop);
```

------

## 5. Error Handling (错误处理)

### 24) `cudaGetLastError`

```cpp
cudaError_t cudaGetLastError(void);
```

<span style="color:#2980B9">Returns and clears</span> the last CUDA error code (获取并清除最近一次错误码). Commonly called after kernel launches (which return no error directly).

```cpp
kernel<<<grid, block>>>(d_data);
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    fprintf(stderr, "Kernel launch failed: %s\n",
            cudaGetErrorString(err));
}
```

------

### 25) `cudaGetErrorString`

```cpp
const char* cudaGetErrorString(cudaError_t error);
```

<span style="color:#2980B9">Returns</span> a human-readable string (可读的错误描述字符串) for a `cudaError_t` error code.

```cpp
#define CUDA_CHECK(call)                                        \
    do {                                                        \
        cudaError_t err = (call);                               \
        if (err != cudaSuccess) {                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",      \
                    __FILE__, __LINE__,                          \
                    cudaGetErrorString(err));                    \
            exit(EXIT_FAILURE);                                 \
        }                                                       \
    } while (0)

// Usage:
CUDA_CHECK(cudaMalloc(&d_data, size));
CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> Always define a <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">CUDA_CHECK</code> macro and wrap every CUDA API call in production code. Silent failures (默默失败) are the most common source of mysterious bugs — a failed <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaMalloc</code> leaves the pointer null, corrupting all subsequent kernel behavior.</div>

------

## 6. Unified Memory (统一内存)

### 26) `cudaMallocManaged`

```cpp
cudaError_t cudaMallocManaged(void** devPtr,
                               size_t size,
                               unsigned int flags = cudaMemAttachGlobal);
```

<span style="color:#2980B9">Allocates</span> Unified Memory (统一内存) — a single pointer accessible from both host and device. The CUDA driver <span style="color:#E8600A;font-weight:700">automatically migrates pages (自动迁移内存页)</span> between CPU and GPU as needed.

```cpp
float* unified;
cudaMallocManaged(&unified, N * sizeof(float));

// Initialize on CPU
for (int i = 0; i < N; i++) unified[i] = (float)i;

// Use on GPU — no explicit cudaMemcpy needed!
kernel<<<grid, block>>>(unified, N);
cudaDeviceSynchronize();

// Read back on CPU — driver has migrated data automatically
printf("unified[0] = %f\n", unified[0]);

cudaFree(unified);
```

------

### 27) `cudaMemPrefetchAsync`

```cpp
cudaError_t cudaMemPrefetchAsync(const void* devPtr,
                                  size_t count,
                                  int dstDevice,
                                  cudaStream_t stream = 0);
```

<span style="color:#2980B9">Asynchronously prefetches</span> Unified Memory pages to a target device (异步预取统一内存页), avoiding on-demand page faults (按需缺页中断) during kernel execution.

```cpp
// Prefetch to GPU before kernel (avoids page-fault overhead)
cudaMemPrefetchAsync(unified, N * sizeof(float), 0, stream);   // GPU 0
kernel<<<grid, block, 0, stream>>>(unified, N);

// Prefetch back to CPU before reading
cudaMemPrefetchAsync(unified, N * sizeof(float), cudaCpuDeviceId, stream);
cudaStreamSynchronize(stream);
printf("Result: %f\n", unified[0]);
```

------

## 7. CUDA Graph (CUDA 图)

### 28) `cudaGraphCreate`

```cpp
cudaError_t cudaGraphCreate(cudaGraph_t* pGraph, unsigned int flags);
```

<span style="color:#2980B9">Creates</span> an empty CUDA Graph (CUDA 计算图). A graph captures a sequence of GPU operations (kernels, memcpies, etc.) as a DAG (有向无环图) that can be launched repeatedly with minimal CPU overhead.

```cpp
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);
```

------

### 29) `cudaStreamBeginCapture` / `cudaStreamEndCapture`

```cpp
cudaError_t cudaStreamBeginCapture(cudaStream_t stream,
                                    cudaStreamCaptureMode mode);
cudaError_t cudaStreamEndCapture(cudaStream_t stream,
                                  cudaGraph_t* pGraph);
```

<span style="color:#2980B9">Captures</span> all operations submitted to a stream into a CUDA Graph (将流中的操作捕获为计算图). The most practical way to build a graph.

```cpp
cudaStream_t captureStream;
cudaStreamCreate(&captureStream);

// Begin capture
cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeGlobal);

// Queue operations — these are RECORDED, not executed yet
cudaMemcpyAsync(d_in, h_in, size, cudaMemcpyHostToDevice, captureStream);
kernel<<<grid, block, 0, captureStream>>>(d_in, d_out, N);
cudaMemcpyAsync(h_out, d_out, size, cudaMemcpyDeviceToHost, captureStream);

// End capture → produces a graph
cudaGraph_t graph;
cudaStreamEndCapture(captureStream, &graph);
```

------

### 30) `cudaGraphInstantiate` / `cudaGraphLaunch`

```cpp
cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pGraphExec,
                                  cudaGraph_t graph,
                                  unsigned long long flags = 0);
cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec,
                             cudaStream_t stream);
```

<span style="color:#2980B9">Instantiates</span> a graph into an executable form (将图编译为可执行对象) and <span style="color:#2980B9">launches</span> it on a stream with minimal CPU overhead (以极低 CPU 开销启动).

```cpp
// Instantiate once
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, 0);

// Launch many times — CPU overhead is near zero
for (int iter = 0; iter < 1000; iter++) {
    cudaGraphLaunch(graphExec, stream);
}
cudaStreamSynchronize(stream);

// Cleanup
cudaGraphExecDestroy(graphExec);
cudaGraphDestroy(graph);
```

<div style="background:#F5F5F5;border-left:4px solid #E8600A;border-radius:0 6px 6px 0;padding:12px 16px;margin:14px 0;font-size:14px;line-height:1.85"><span style="color:#E8600A;font-weight:700">Note: </span> CUDA Graphs eliminate per-launch CPU overhead (每次启动的 CPU 开销) — critical for workloads with many small kernels (e.g., LLM token generation, physics simulations). In vLLM and TensorRT, CUDA Graphs are used to capture the decode step for maximum throughput.</div>

------

## 8. Quick Reference Table (速查表)

| #    | API                                                          | Category | Key Trait                     |
| ---- | ------------------------------------------------------------ | -------- | ----------------------------- |
| 1    | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaGetDeviceCount</code> | Device   | Query available GPUs          |
| 2    | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaSetDevice</code> | Device   | Select active GPU             |
| 3    | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaGetDevice</code> | Device   | Query active GPU              |
| 4    | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaGetDeviceProperties</code> | Device   | Hardware specs                |
| 5    | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaDeviceSynchronize</code> | Device   | Barrier — wait all GPU work   |
| 6    | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaMalloc</code> | Memory   | Allocate device memory        |
| 7    | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaFree</code> | Memory   | Free device memory            |
| 8    | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaMemcpy</code> | Memory   | Synchronous H↔D copy          |
| 9    | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaMemcpyAsync</code> | Memory   | Async H↔D copy (needs pinned) |
| 10   | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaMemset</code> | Memory   | Zero-fill device memory       |
| 11   | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaMallocHost</code> | Memory   | Pinned host allocation        |
| 12   | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaFreeHost</code> | Memory   | Free pinned host memory       |
| 13   | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaMemcpy2D</code> | Memory   | 2D pitched copy               |
| 14   | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaStreamCreate</code> | Stream   | Create async work queue       |
| 15   | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaStreamDestroy</code> | Stream   | Release stream                |
| 16   | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaStreamSynchronize</code> | Stream   | Wait for one stream           |
| 17   | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaStreamWaitEvent</code> | Stream   | Cross-stream dependency       |
| 18   | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaStreamCreateWithPriority</code> | Stream   | Prioritized stream            |
| 19   | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaEventCreate</code> | Event    | Create timestamp marker       |
| 20   | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaEventRecord</code> | Event    | Insert marker in stream       |
| 21   | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaEventSynchronize</code> | Event    | Wait for event on host        |
| 22   | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaEventElapsedTime</code> | Event    | Measure kernel time (ms)      |
| 23   | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaEventDestroy</code> | Event    | Release event                 |
| 24   | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaGetLastError</code> | Error    | Fetch & clear last error      |
| 25   | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaGetErrorString</code> | Error    | Error code → string           |
| 26   | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaMallocManaged</code> | UM       | Unified Memory alloc          |
| 27   | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaMemPrefetchAsync</code> | UM       | Prefetch pages to device      |
| 28   | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaGraphCreate</code> | Graph    | Create empty graph            |
| 29   | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaStreamBeginCapture</code> / <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">EndCapture</code> | Graph    | Capture stream → graph        |
| 30   | <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaGraphInstantiate</code> / <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">Launch</code> | Graph    | Compile & replay graph        |

------

<div style="background:linear-gradient(135deg,#EBF0FF 0%,#FFF3E0 100%);border:1.5px solid #c5d3ff;border-radius:8px;padding:14px 20px;margin-top:24px"><span style="color:#3B5BDB;font-weight:700">💡 One-line Takeaway</span><br> Master the core trio first — <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaMalloc</code> / <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaMemcpy</code> / <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">cudaFree</code>; wrap everything with a <code style="background:#FFF3E0;color:#7a2e00;border-radius:4px;padding:1px 6px">CUDA_CHECK</code> macro; use <strong>streams + async copies + pinned memory</strong> for overlap; use <strong>events</strong> for timing and cross-stream sync; and use <strong>CUDA Graphs</strong> to eliminate CPU launch overhead for repeated workloads.</div>
