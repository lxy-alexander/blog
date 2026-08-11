---
title: "llm-d"
published: 2026-08-10
description: "llm-d"
image: ""
tags: ["vllm","llm-d"]
category: vllm
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



















## K3s 和 K8s

K3s 和 K8s 的关系可以简单理解成：**K3s 是轻量版的 Kubernetes。**

1）**K8s**
就是标准 Kubernetes，功能完整，组件多，适合生产集群、大规模部署。

2）**K3s**
是基于 Kubernetes 做的轻量发行版，安装更简单、占资源更少，很多组件被打包到一个二进制里。

3）主要区别

| 对比         | K8s              | K3s                              |
| ------------ | ---------------- | -------------------------------- |
| 安装         | 相对复杂         | 一条命令即可                     |
| 资源占用     | 较高             | 较低                             |
| 组件         | 多个独立组件     | 很多被整合                       |
| 数据存储     | 常用 etcd        | 默认可用 SQLite，也支持 etcd     |
| 使用场景     | 企业生产、大集群 | 学习、边缘设备、小集群、开发环境 |
| kubectl/Helm | 支持             | 同样支持                         |

比如标准 K8s 通常是：

```text
kube-apiserver
kube-scheduler
kube-controller-manager
etcd
kubelet
container runtime
...
```

K3s 会把很多东西整合起来：

```text
k3s server
  ├─ API Server
  ├─ Scheduler
  ├─ Controller
  └─ 其他组件
```

所以如果你现在只是**自己服务器搭 Kubernetes、学习 Docker/K8s、跑几个服务**，K3s 往往更省事。

一句话：

**K8s = 标准完整版 Kubernetes**

**K3s = 更轻、更容易安装的 Kubernetes，基本的 K8s 用法没有变。**



##  `kubectl` 管理 K3s 集群

这段是在让你当前用户可以直接用 `kubectl` 管理 K3s 集群。

1）创建 kubectl 配置目录

```bash
mkdir -p ~/.kube
```

`~` 是当前用户的家目录，比如：

```text
/home/ubuntu/.kube
```

2）复制 K3s 的集群配置

```bash
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
```

K3s 默认把 kubeconfig 放在：

```text
/etc/rancher/k3s/k3s.yaml
```

复制后，`kubectl` 就可以使用标准位置：

```text
~/.kube/config
```

3）把文件所有者改成当前用户

```bash
sudo chown $USER:$USER ~/.kube/config
```

否则因为刚才用了 `sudo cp`，文件可能属于 `root`，普通用户操作不方便。

4）指定 kubectl 使用哪个配置

```bash
export KUBECONFIG=~/.kube/config
```

不过这里其实**通常可以省略**，因为 `kubectl` 默认就会寻找：

```text
~/.kube/config
```

所以前面已经复制到这个位置的话，可以直接：

```bash
kubectl get nodes
```

5）检查节点

```bash
kubectl get nodes
```

正常可能看到：

```text
NAME       STATUS   ROLES                  AGE   VERSION
server01   Ready    control-plane,master   5m    v1.x.x+k3s
```

`Ready` 就说明节点正常。

6）检查所有 Pod

```bash
kubectl get pods -A
```

`-A` = `--all-namespaces`，查看整个集群所有 namespace 的 Pod。

你可能会看到：

```text
NAMESPACE     NAME                       STATUS
kube-system   coredns-xxx                Running
kube-system   metrics-server-xxx         Running
kube-system   local-path-provisioner-xxx Running
```

这些基本都是 K3s 自带的系统组件。

整体流程就是：

```text
K3s 已经运行
       ↓
/etc/rancher/k3s/k3s.yaml
       ↓
复制到 ~/.kube/config
       ↓
kubectl 读取配置
       ↓
kubectl → K3s/Kubernetes API
       ↓
管理 Node、Pod、Service 等
```

另外，你之前安装时已经用了：

```bash
--write-kubeconfig-mode 644
```

所以严格来说，**你甚至不一定需要复制配置文件**，可以直接：

```bash
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
kubectl get nodes
```

但复制到 `~/.kube/config` 更符合标准 Kubernetes 使用习惯，后面学习 Helm 也更顺手。



## Helm 给 K3s/K8s 集群安装 NVIDIA GPU Device Plugin

这几条命令是在用 **Helm 给 K3s/K8s 集群安装 NVIDIA GPU Device Plugin**。这个插件的作用是让 Kubernetes 能识别并调度 NVIDIA GPU。NVIDIA 官方仓库就是这么安装的。([GitHub](https://github.com/nvidia/k8s-device-plugin?utm_source=chatgpt.com))

1）添加 Helm 仓库

```bash
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
```

`nvdp` 只是你给这个仓库起的本地名字。

以后：

```bash
nvdp/nvidia-device-plugin
```

就表示这个 NVIDIA 仓库里的 `nvidia-device-plugin` Chart。

2）更新 Helm 仓库索引

```bash
helm repo update
```

相当于告诉 Helm：

```text
去各个已添加的仓库
→ 获取最新的软件包列表
```

3）安装 NVIDIA Device Plugin

```bash
helm upgrade -i nvdp nvdp/nvidia-device-plugin \
  --namespace nvidia-device-plugin \
  --create-namespace
```

这里：

```text
helm upgrade -i
```

意思是：

```text
已经安装了 → 升级
没安装 → 安装
```

其中第一个：

```text
nvdp
```

是这个 Helm Release 的名字。

第二个：

```text
nvdp/nvidia-device-plugin
```

才是要安装的 Helm Chart。

4）指定 Namespace

```bash
--namespace nvidia-device-plugin
```

表示把它安装到：

```text
nvidia-device-plugin
```

这个 namespace 里面。

5）如果 Namespace 不存在就创建

```bash
--create-namespace
```

相当于省掉了手动执行：

```bash
kubectl create namespace nvidia-device-plugin
```

安装完成后，可以检查：

```bash
kubectl get pods -n nvidia-device-plugin
```

然后看节点是否识别到了 GPU：

```bash
kubectl describe node
```

正常情况下会出现类似：

```text
Capacity:
  nvidia.com/gpu: 1

Allocatable:
  nvidia.com/gpu: 1
```

这样以后 Pod 就可以申请 GPU：

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```

整体关系可以理解成：

```text
物理 NVIDIA GPU
      ↓
NVIDIA 驱动
      ↓
NVIDIA Container Toolkit
      ↓
NVIDIA Device Plugin
      ↓
K3s / Kubernetes
      ↓
Pod 可以申请 nvidia.com/gpu
```

要注意：**这个 Helm Chart 本身不是 NVIDIA 显卡驱动。** NVIDIA Device Plugin 官方也要求节点先具备 NVIDIA 驱动以及相应的容器运行时支持，否则插件可能启动了但识别不到 GPU。([GitHub](https://github.com/NVIDIA/k8s-device-plugin/issues/478?utm_source=chatgpt.com))

所以如果你是在搭 **K3s + NVIDIA GPU**，这一步的作用就是：**把宿主机上的 GPU 暴露给 Kubernetes，让 Pod 能使用 GPU。**





## CRD

````python
CRD 是 Kubernetes 里的 **Custom Resource Definition**，中文叫“自定义资源定义”。

1）Kubernetes 原生认识这些资源：

```text
Pod
Service
Deployment
Node
DaemonSet
```

所以你可以直接：

```bash
kubectl get pods
kubectl get nodes
```

2）但有些软件会自己创造新的资源类型。

比如 NFD 用到了：

```text
NodeFeature
```

Kubernetes 默认并不知道 `NodeFeature` 是什么。

所以要先安装一个 CRD，告诉 Kubernetes：

> 以后有一种新的资源，名字叫 NodeFeature，它有哪些字段、什么结构。

3）关系可以这样理解：

```text
CRD
= 定义“资源类型”

CR
= 按这个类型创建出来的具体对象
```

比如：

```text
CRD：定义“Car”这种资源
↓
CR：my-car、your-car
```

在你前面的场景里：

```text
CRD
nodefeatures.nfd.k8s-sigs.io
↓
定义 NodeFeature

NodeFeature
crimson-cipher-fin-01
↓
保存这个节点发现到的硬件特征
```

4）你之前为什么会报错：

```text
failed to create NodeFeature object
the server could not find the requested resource
```

就是因为：

```text
NFD Worker
↓
想创建 NodeFeature
↓
Kubernetes 还没有对应 CRD
↓
“不认识 NodeFeature”
↓
创建失败
```

安装 CRD 后，Kubernetes 才认识这种资源。

一句话记：

**CRD 就是给 Kubernetes“扩展新资源类型”的机制。**

````

