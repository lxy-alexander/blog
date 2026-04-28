调整后可以这样写：

1）**Tesla / Early CUDA**

1）Kernel Launch
2）Grid / Block / Thread
3）Streams
4）Events
5）Pinned Memory
6）Early Atomic Operations

------

2）**Fermi**

1）L1 Cache
2）L2 Cache
3）Shared Memory
4）Global Memory Atomics
5）`atomicAdd`

------

3）**Kepler**

1）Warp-level primitives
　　1）`__shfl`
2）Dynamic Parallelism
3）Hyper-Q

------

4）**Maxwell**

1）Native Shared Memory Atomics
　　1）32-bit Integer Shared Memory Atomics
2）Histogram-friendly Atomic Operations

------

5）**Pascal**

1）Unified Memory
　　1）Managed Memory
　　2）Page Migration
　　3）Fine-grained Memory Migration

------

6）**Volta**

1）Tensor Cores
　　1）Mixed Precision
2）Independent Thread Scheduling
3）Synchronized Warp Primitives
　　1）`__shfl_sync`
4）Cooperative Groups
5）Warp-level Synchronization

------

7）**Turing**

1）Second-generation Tensor Cores
　　1）INT8
　　2）INT4
　　3）Mixed Precision
2）CUDA Graphs

------

8）**Ampere**

1）Third-generation Tensor Cores
　　1）TF32
　　2）BF16
　　3）Mixed Precision
2）Asynchronous Copy
　　1）`cp.async`
　　2）Global Memory to Shared Memory Asynchronous Copy
　　3）CUDA Pipeline API
3）CUDA Graphs

------

9）**Hopper**

1）Fourth-generation Tensor Cores
　　1）FP8
　　2）Mixed Precision
2）Transformer Engine
3）Tensor Memory Accelerator
4）Thread Block Cluster
5）Distributed Shared Memory
6）Cluster-level Cooperative Groups

------

10）**Ada Lovelace**

1）Fourth-generation Tensor Cores
　　1）FP8
　　2）INT8
　　3）Mixed Precision
2）DLSS 3 / Optical Flow Accelerator
3）CUDA Graphs

------

11）**Blackwell**

1）Fifth-generation Tensor Cores
　　1）FP4
　　2）Mixed Precision
2）Second-generation Transformer Engine
3）CUDA Graphs Conditional Nodes
4）LLM Training / Inference Acceleration
5）CUDA Toolkit 12.8 Blackwell Architecture Support