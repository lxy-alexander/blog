---
title: "Mooncake BatchTransfer"
published: 2026-08-16
description: "Mooncake BatchTransfer"
image: ""
tags: ["llm_inference","Mooncake BatchTransfer"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-08-16T21:24:16.974.902555693Z"
---



## 为什么要 BatchTransfer

因为 KV cache / tensor 通常不是一个小块，而是很多块：

```
多个 layer
多个 head
多个 block
多个非连续地址
多个 replica
```

如果每块都单独提交，会有很多额外开销。BatchTransfer 可以：

```
减少 API 调用次数
批量分配 BatchID
批量提交给 transport
让 RDMA/TCP/NVLink 后端统一调度
聚合多 NIC 带宽
统一查询完成状态
```

**带入数值**

假设要写一个 256KB 对象，切成 4 个 64KB：

```
本地:
0x60000000 - 0x6003ffff

远端:
0x7f0000000000 - 0x7f000003ffff
```

构造 4 个 request：

```
req0:
  opcode = WRITE
  source = 0x60000000
  target_offset = 0x7f0000000000
  length = 65536

req1:
  source = 0x60010000
  target_offset = 0x7f0000010000
  length = 65536

req2:
  source = 0x60020000
  target_offset = 0x7f0000020000
  length = 65536

req3:
  source = 0x60030000
  target_offset = 0x7f0000030000
  length = 65536
```

提交：

```
BatchID batch = engine.allocateBatchID(4);
engine.submitTransfer(batch, {req0, req1, req2, req3});
```

然后查状态：

```
TransferStatus status;
engine.getBatchTransferStatus(batch, status);
```

结果可能是：

```
WAITING     还没全部完成
COMPLETED   全部完成
FAILED      至少一个失败
```

**在代码里对应什么**

`BatchID` 实际上指向一个 `BatchDesc`：

```
auto batch_desc = new BatchDesc();
batch_desc->id = BatchID(batch_desc);
batch_desc->batch_size = batch_size;
batch_desc->task_list.reserve(batch_size);
```

也就是说：

```
BatchTransfer = BatchDesc + 多个 TransferTask + 每个 task 下的 slices
```

结构大概是：

```
BatchDesc
  ├── TransferTask 0
  │     ├── Slice 0
  │     └── Slice 1
  ├── TransferTask 1
  │     ├── Slice 0
  │     └── Slice 1
  └── TransferTask 2
        └── Slice 0
```

`TransferRequest` 是用户提交的逻辑请求。
`Slice` 是 transport 内部进一步拆出来的物理传输片段。

**一句话**

`BatchTransfer` 就是：**把多个 READ/WRITE 请求打包成一批异步提交，底层再切片、选 transport、选 NIC 并行传输，最后用 BatchID 统一查询状态。**
