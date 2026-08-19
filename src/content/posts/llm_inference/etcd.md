---
title: "etcd"
published: 2026-08-18
description: "etcd"
image: ""
tags: ["llm_inference","etcd"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-08-18T23:26:25.255.878084668Z"
---

# etcd 学习笔记

## 1. 基本概念

`etcd` 是一个**分布式键值数据库**，基于 Raft 一致性算法实现，具备强一致性、高可用性。常见定位是：分布式系统的"配置中心 + 协调服务"，Kubernetes 的核心存储组件就是 etcd。

核心特性：

-   **强一致性**：所有写操作通过 Raft 协议在多数节点达成共识后才生效
-   **Watch 机制**：客户端可以监听某个 key（或前缀）的变化，实时收到通知
-   **租约（Lease）机制**：key 可以绑定 TTL，常用于实现"存活检测"（心跳续约，超时自动删除）
-   **多版本并发控制（MVCC）**：保留 key 的历史版本，支持基于版本号的查询

------

## 2. 在分布式系统里常见的四大用途

| 用途                        | 说明                                                         |
| --------------------------- | ------------------------------------------------------------ |
| **选主（Leader Election）** | 利用租约 + 事务，多个候选者竞争同一个 key，抢到的成为 Leader |
| **服务发现**                | 客户端不直接连固定地址，而是查 etcd 拿到当前可用节点/Leader 地址 |
| **配置/状态存储**           | 存放需要多节点共享、且要求强一致的元数据、日志               |
| **故障协调**                | Leader 因租约到期或主动退出后，etcd 通知其他节点重新选主     |

------

## 3. 在 Mooncake HA 测试环境中的具体角色

在这套 Mooncake HA（Master 高可用）测试场景里，etcd 承担四件事：

1.  **Master 选主**：记录哪个 Master 实例是当前 Leader
2.  **服务发现**：客户端通过 `etcd://...` 这种地址格式找到当前 Leader 是谁
3.  **保存 HA OpLog**：Standby Master 读取这些日志，用来恢复 KV 元数据状态
4.  **协助故障切换**：Leader（Master-0）挂掉后，etcd 协助 Standby（Master-1）提升为新 Leader

### 复现链路（这是 etcd 在这个 Bug 场景里的作用路径）

```
Master-0 是 Leader
    ↓
Master-0 被杀（模拟故障）
    ↓
etcd 协助 Master-1 成为新 Leader
    ↓
P 节点（Transfer Engine 客户端）发现 Leader 改变
    ↓
P 节点向 Master-1 执行 ReMountSegment
    ↓
触发全量扫描 + 锁阻塞（真正的 Bug 所在）
```

------

## 4. 关键澄清：etcd 和 Bug 的关系

-   etcd **不是 Bug 本身的组成部分**，它只是本地自动化复现脚本用来**触发** HA 切主场景的工具

-   原始 issue 环境用的是 `--ha_backend_type=redis`

-   仓库自带脚本用的是 `--ha_backend_type=etcd`

-   但无论后端是 Redis 还是 etcd，切主后最终都会调用同一个函数：

    ```
    MasterService::ReMountSegment()
    ```

-   真正的 Bug 出在 `ReMountSegment()` 内部：

    -   持有**独占锁**（`snapshot_mutex_` / `client_mutex_`）
    -   对 **1024 个 metadata shard 做全量扫描**
    -   导致 `PutStart` 被阻塞
    -   最终触发 `RPC_TIMEOUT`

**结论**：etcd 只是"引信"（触发切主的手段），不是"炸药"（锁竞争问题）本身。如果已经有可用的 Redis 集群 + 正在运行的 Transfer Engine metadata 服务，完全可以复现同样的 Bug，不需要额外搭建 etcd 环境。



>   etcd 在这里的作用是**制造"切主"这个触发条件**，让 `ReMountSegment()` 被调用；真正导致超时的锁竞争问题，和用什么 HA 后端（etcd 还是 Redis）无关，是 `MasterService` 内部实现的问题。
