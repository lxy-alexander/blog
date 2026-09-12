---
title: "llmd Contributor"
published: 2026-08-10
description: "llmd Contributor"
image: ""
tags: ["llm_inference","llmd Contributor"]
category: llm_inference
draft: false
lang: ""
createdAt: "2026-08-11T00:23:03.744.899495957Z"
---

![image-20260810202308278](https://pub-c69d652d2a0747fab9aad1fab48ff742.r2.dev/images/image-20260810202308278)

==llm-d 用 Kubernetes 部署多个 LLM model server，通过 Proxy 接请求，通过 EPP 根据缓存、负载、队列和策略选择最佳 pod，并用 autoscaler 动态调整不同 variant 的副本数。==

>   "llm-d deploys multiple LLM model servers using Kubernetes, receives requests through a proxy, selects the optimal pod using EPP based on cache, load, queue, and policies, and dynamically adjusts the replica count for different variants via an autoscaler."

GPU 容器透传，就是让**容器里的程序能够直接使用宿主机上的 NVIDIA GPU**。

1）什么是 Pod？ / What is a Pod?

Pod 是 Kubernetes 中最小的部署和调度单元。一个 Pod 可以包含一个或多个 Container，它们共享网络、IP、Volume 和生命周期。

A Pod is the smallest deployable and schedulable unit in Kubernetes. A Pod can contain one or more containers that share networking, IP address, volumes, and lifecycle.

2）Pod 和 Container 有什么区别？ / What is the difference between a Pod and a Container?

Container 是真正运行应用程序的环境，而 Pod 是 Kubernetes 用来管理一个或多个 Container 的封装单位。Kubernetes 调度的是 Pod，而不是单独调度 Container。

A container is the runtime environment where the application actually runs, while a Pod is the Kubernetes abstraction used to manage one or more containers. Kubernetes schedules Pods rather than individual containers.

3）什么是 InferencePool？ / What is an InferencePool?

InferencePool 是 Kubernetes Gateway API Inference Extension 中的推理后端池资源。它可以表示一组 vLLM Pod，并通过 selector 动态发现这些后端。

InferencePool is an inference backend pool resource in the Kubernetes Gateway API Inference Extension. It can represent a group of vLLM Pods and dynamically discover them through selectors.

4）InferencePool 和 Kubernetes Service 有什么区别？ / What is the difference between InferencePool and Kubernetes Service?

Service 主要用于普通服务发现和负载均衡，而 InferencePool 面向 AI 推理场景，可以和 EPP 配合，根据 KV Cache、队列长度和负载等信息进行更智能的路由。

A Kubernetes Service is mainly used for standard service discovery and load balancing, while InferencePool is designed for AI inference workloads and can work with EPP to make smarter routing decisions based on KV Cache, queue depth, and workload status.

5）什么是 EPP？ / What is EPP?

EPP 是 Endpoint Picker，负责从多个推理后端中选择一个最合适的 Endpoint。它可以根据负载、KV Cache locality、队列长度等信息进行调度。

EPP stands for Endpoint Picker. It selects the most appropriate endpoint from multiple inference backends based on information such as load, KV Cache locality, and queue depth.

6）EPP 是 Kubernetes 专属的吗？ / Is EPP Kubernetes-specific?

不是。EPP 可以运行在 Kubernetes 环境，也可以运行在非 Kubernetes 环境。主要区别是后端发现方式不同。

No. EPP can run in both Kubernetes and non-Kubernetes environments. The main difference is how backend endpoints are discovered.

7）没有 Kubernetes，还能做负载均衡吗？ / Can we do load balancing without Kubernetes?

可以。只要 EPP 能获得多个 vLLM Endpoint，就可以做负载均衡。没有 Kubernetes 时，可以通过 `endpoints.yaml` 提供后端地址。

Yes. As long as EPP has access to multiple vLLM endpoints, it can perform load balancing. Without Kubernetes, backend addresses can be provided through `endpoints.yaml`.

8）InferencePool 和 endpoints.yaml 有什么区别？ / What is the difference between InferencePool and endpoints.yaml?

InferencePool 依赖 Kubernetes，可以动态发现和更新 Pod；`endpoints.yaml` 通常是静态配置，需要手工维护 vLLM 的地址。它们都可以为 EPP 提供候选后端。

InferencePool depends on Kubernetes and can dynamically discover and update Pods, while `endpoints.yaml` is usually a static configuration that requires manual maintenance of vLLM addresses. Both can provide candidate backends to EPP.

9）为什么 LLM 推理需要智能负载均衡？ / Why does LLM inference need intelligent load balancing?

因为不同 vLLM 实例的状态可能差别很大，例如队列长度、GPU 负载、KV Cache 命中情况都不同。简单的 Round Robin 不一定能获得最低延迟。

Because different vLLM instances may have very different runtime states, such as queue depth, GPU load, and KV Cache hit rate. Simple round-robin load balancing may not provide the lowest latency.

10）在 llm-d 中，Gateway、EPP、InferencePool 和 vLLM 的关系是什么？ / What is the relationship between Gateway, EPP, InferencePool, and vLLM in llm-d?

InferencePool 定义哪些 vLLM Pod 属于后端池，EPP 负责从这些后端中选择最合适的一个，Gateway 负责把请求转发过去，vLLM 负责真正执行模型推理。

InferencePool defines which vLLM Pods belong to the backend pool, EPP selects the most appropriate backend, Gateway forwards the request to that backend, and vLLM performs the actual model inference.



## 22 个公开仓库

截至 2026-09-10，[llm-d GitHub 组织](https://github.com/orgs/llm-d/repositories)有 22 个公开仓库，其中 19 个活跃、3 个归档，但很多是工具、模板或网站，不属于运行时依赖。

**核心仓库**

| 仓库                                                         | 作用                               | 是否必需       |
| ------------------------------------------------------------ | ---------------------------------- | -------------- |
| [`llm-d`](https://github.com/llm-d/llm-d)                    | 集成、Guide、部署配置、文档        | 是             |
| [`llm-d-router`](https://github.com/llm-d/llm-d-router)      | EPP、请求调度、过滤和评分          | 是             |
| [`llm-d-latency-predictor`](https://github.com/llm-d/llm-d-latency-predictor) | XGBoost 训练与预测                 | 仅预测延迟模式 |
| [`llm-d-kv-cache`](https://github.com/llm-d/llm-d-kv-cache)  | 精确 KV Cache 索引和 offloading    | 高级缓存功能   |
| [`llm-d-pd-utils`](https://github.com/llm-d/llm-d-pd-utils)  | Prefill/Decode 分离辅助能力        | P/D 场景       |
| [`llm-d-autoscaling`](https://github.com/llm-d/llm-d-autoscaling) | 推理副本和 workload variant 扩缩容 | 自动扩缩容场景 |

**扩展服务**

| 仓库                                                         | 作用                        |
| ------------------------------------------------------------ | --------------------------- |
| [`llm-d-async`](https://github.com/llm-d/llm-d-async)        | 异步请求调度                |
| [`llm-d-batch-gateway`](https://github.com/llm-d/llm-d-batch-gateway) | OpenAI-compatible Batch API |
| [`llm-d-inference-payload-processor`](https://github.com/llm-d/llm-d-inference-payload-processor) | 根据请求内容做多 Pool 路由  |
| [`llm-d-api-extensions`](https://github.com/llm-d/llm-d-api-extensions) | llm-d API 扩展定义          |

**开发和性能工具**

| 仓库                                                         | 作用                 |
| ------------------------------------------------------------ | -------------------- |
| [`llm-d-benchmark`](https://github.com/llm-d/llm-d-benchmark) | 性能测试             |
| [`llm-d-inference-sim`](https://github.com/llm-d/llm-d-inference-sim) | 不使用 GPU 模拟 vLLM |
| [`llm-d-prism`](https://github.com/llm-d/llm-d-prism)        | 分布式推理性能分析   |

其他如 `llm-d-infra`、网站、Go/Python 模板和组织元数据，不是推理运行时组件。`llm-d-deployer`、`llm-d-model-service`、`llm-d-routing-sidecar` 已归档，不建议作为当前入口。









## 下载更新

你的系统没有安装 GitHub CLI。可以直接使用 `curl + python3 + git`，不需要 `gh` 或 `jq`。

在当前目录执行：

```
mkdir -p lxy/llm-d-all
cd lxy/llm-d-all

curl -fsSL \
  'https://api.github.com/orgs/llm-d/repos?per_page=100&type=all' |
python3 -c '
import json
import sys

for repo in json.load(sys.stdin):
    print(repo["clone_url"])
' |
while IFS= read -r url; do
  name="${url##*/}"
  name="${name%.git}"

  if [ -d "$name/.git" ]; then
    echo "SKIP: $name already exists"
  else
    echo "CLONE: $name"
    git clone --recurse-submodules "$url"
  fi
done
```

下载完成后检查：

```
cd /data/home/xli49/lxy/llm-d-all
find . -maxdepth 2 -type d -name .git | sort
```

查看仓库数量：

```
find . -maxdepth 2 -type d -name .git | wc -l
```

以后批量更新：

```
cd ~/lxy/llm-d-all

for dir in */; do
  [ -d "$dir/.git" ] || continue

  echo "===== $dir ====="

  if [ -n "$(git -C "$dir" status --porcelain)" ]; then
    echo "SKIP: 存在未提交修改"
    continue
  fi

  git -C "$dir" fetch --all --tags --prune
  git -C "$dir" pull --ff-only
  git -C "$dir" submodule update --init --recursive
done
```

如果 `curl` 也不存在，可检查：

```
command -v curl
command -v python3
command -v git
```

当前这套下载方式只需要这三个命令。
