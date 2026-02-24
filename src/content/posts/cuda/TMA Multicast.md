---
title: "TMA Multicast"
published: 2026-02-20
description: "TMA Multicast"
image: ""
tags: ["cuda","TMA Multicast"]
category: cuda
draft: false
lang: ""
---



>Hopper 允许多个 thread block 组成 cluster，跨 SM 协作。配合 TMA multicast，可以让多个 SM 共享同一块 global memory 读取，大幅提升带宽利用率和性能。

![image-20260220095929340](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260220095929340)



## TMA Multicast

正常情况：

```
SM0 需要 tile A → 从 L2 读
SM1 也需要 tile A → 再读一次
```

-   消耗两次 L2 带宽
-   增加延迟

------

### Hopper 的解决方案：

TMA Multicast 只加载一次 tile：

```
global → L2 → 同时送到多个 SM 的 shared memory
```

只读一次，广播到 cluster 内多个 SM
