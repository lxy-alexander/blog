cuda/
├── 01-基础概念/ Fundamentals
│   ├── GPU 架构基础.md           (SM、Warp、CUDA Core 是什么)
│   ├── CUDA 编程模型.md           (Grid/Block/Thread 三层结构)
│   ├── 内存层级.md                (Global/Shared/Register/Constant)
│   └── 第一个 CUDA 程序.md        (vector add, hello world)
│
├── 02-核心 API/ Core APIs
│   ├── kernel 启动配置.md         (<<<grid, block>>> 怎么算)
│   ├── 内存管理.md                (cudaMalloc/cudaMemcpy/Unified Memory)
│   ├── 流与事件.md                (Stream、Event、异步执行)
│   └── 同步机制.md                (__syncthreads、原子操作)
│
├── 03-性能优化/ Performance
│   ├── 内存合并访问.md            (Coalesced Access)
│   ├── Shared Memory 优化.md      (Bank Conflict、Tile)
│   ├── Warp 级原语.md             (shuffle、vote、ballot)
│   ├── Occupancy 分析.md          (寄存器、Block 大小权衡)
│   └── Roofline 模型.md           (算力 vs 带宽瓶颈分析)
│
├── 04-经典 Kernel/ Classic Kernels
│   ├── Reduction.md               (规约,经典优化案例)
│   ├── GEMM.md                    (矩阵乘,从 naive 到 Tensor Core)
│   ├── Softmax.md                 (online softmax、safe softmax)
│   ├── Conv.md                    (im2col、Winograd、隐式 GEMM)
│   ├── Scan-Prefix Sum.md         (前缀和)
│   └── Transpose.md               (矩阵转置)
│
├── 05-高级特性/ Advanced
│   ├── Tensor Core.md             (WMMA、MMA PTX)
│   ├── Cooperative Groups.md
│   ├── CUDA Graph.md
│   ├── PTX 内联汇编.md
│   └── 多 GPU 与 NCCL.md
│
├── 06-工具链/ Toolchain
│   ├── nvcc 编译选项.md
│   ├── Nsight Compute.md          (kernel profiling)
│   ├── Nsight Systems.md          (timeline、整体性能)
│   └── cuda-gdb 调试.md
│
├── 07-生态库/ Libraries
│   ├── cuBLAS.md
│   ├── cuDNN.md
│   ├── CUTLASS.md                 (和你已有的 CuTe 目录联动)
│   ├── Thrust.md
│   └── CCCL-CUB.md
│
└── 08-面试与项目/ Interview & Projects
    ├── 高频面试题.md
    ├── 手写 GEMM 优化路径.md
    └── 学习路线总结.md