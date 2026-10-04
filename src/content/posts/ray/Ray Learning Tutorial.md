---
title: "Ray Learning Tutorial"
published: 2026-10-04
description: "Ray Learning Tutorial"
image: ""
tags: ["ray","distributed computing","python"]
category: ray
draft: false
lang: ""
createdAt: "2026-10-04T00:00:00.000Z"
---

# Ray Learning Tutorial

Ray 是一个 Python-first distributed computing framework，它让你把普通 Python 函数、类和机器学习工作流扩展到多进程、多 CPU/GPU，甚至多机器集群。学习 Ray 的最好顺序不是先看所有库，而是先掌握 Ray Core 的几个 primitive：Task、ObjectRef、Object Store、Actor、Resource，然后再进入 Ray Data、Ray Train、Ray Tune、Ray Serve。

这篇教程按从小到大的方式推进：先在单机上理解 Ray 的执行模型，再写并行任务，再维护状态，最后把它放到数据处理、训练、调参和服务部署里。

<br>

## 1. Installation

如果只是学习 Ray Core，可以安装最小版本：

```bash
pip install -U "ray"
```

如果想要 Dashboard、Cluster Launcher 等常用工具，可以安装：

```bash
pip install -U "ray[default]"
```

如果目标是机器学习工作流，通常一次安装 Data、Train、Tune、Serve：

```bash
pip install -U "ray[data,train,tune,serve]"
```

快速检查：

```python
import ray

ray.init()
print(ray.cluster_resources())
ray.shutdown()
```

`ray.init()` 会启动或连接 Ray runtime。Runtime 可以理解为 Ray 的执行环境，the execution environment that owns workers, object store, scheduling, and runtime metadata。

<br>

## 2. Mental Model

Ray 的核心不是“把 Python 变快”，而是把可并行的工作拆成许多 independent units，然后由 Ray 调度到不同 worker process 上执行。

### 1) Driver

Driver 是你运行的主 Python 进程，the main process that submits work to Ray。它负责编排任务、创建 Actor、收集结果，但真正耗时的计算通常发生在 worker 上。

### 2) Worker

Worker 是 Ray 启动的执行进程，a process that runs tasks or actor methods。一个普通 Python 函数加上 `@ray.remote` 后，就可以被 worker 异步执行。

### 3) Task

Task 是无状态远程函数，a stateless remote function invocation。每次调用 `.remote()` 都提交一次异步工作，返回的不是实际值，而是 `ObjectRef`。

### 4) ObjectRef

ObjectRef 是远程对象引用，a future-like reference to a value that may already exist or may still be computing。你可以把它传给其它任务，也可以用 `ray.get()` 取回真实值。

### 5) Object Store

Object Store 是 Ray 的分布式共享内存存储，a distributed shared-memory store for immutable remote objects。大对象可以放进去，然后多个任务通过引用共享，避免反复序列化和复制。

### 6) Actor

Actor 是有状态远程对象，a stateful worker created from a Python class。它适合维护计数器、模型副本、连接池、缓存等长期状态。

### 7) Resource

Resource 是 Ray 调度时使用的逻辑资源，logical CPU/GPU/custom resource labels used for scheduling。`num_cpus=2` 不会自动让代码开两个线程，而是告诉调度器这个任务需要占用 2 个逻辑 CPU slot。

<br>

## 3. First Task

先从最小例子开始。普通函数是同步执行，Ray Task 是异步提交。

```python
import time
import ray

ray.init()


@ray.remote
def slow_square(x: int) -> int:
    time.sleep(1)
    return x * x


refs = [slow_square.remote(i) for i in range(8)]
print(refs)

results = ray.get(refs)
print(results)

ray.shutdown()
```

`slow_square.remote(i)` 立即返回 `ObjectRef`，the driver does not wait for the function body to finish。`ray.get(refs)` 会阻塞，直到所有结果都准备好。

如果电脑有足够 CPU，这段代码大约 1 秒多完成，而不是 8 秒，因为 8 个 task 可以并行执行。

<br>

## 4. Return Early with ray.wait

`ray.get()` 适合一次性拿所有结果。`ray.wait()` 适合流式处理结果，process results as soon as each task finishes。

```python
import random
import time
import ray

ray.init()


@ray.remote
def work(i: int) -> str:
    time.sleep(random.uniform(0.2, 2.0))
    return f"task {i} finished"


remaining = [work.remote(i) for i in range(10)]

while remaining:
    ready, remaining = ray.wait(remaining, num_returns=1)
    result = ray.get(ready[0])
    print(result)

ray.shutdown()
```

`ray.wait()` 不取值，只返回已经 ready 的 `ObjectRef` 和还没 ready 的 `ObjectRef`。这样 driver 可以边等边消费，especially useful for long-running jobs, streaming pipelines, or progress reporting。

<br>

## 5. Object Store and ray.put

当很多 task 都要读同一个大对象时，不要把对象重复作为普通参数传进去。可以用 `ray.put()` 先放进 Object Store。

```python
import numpy as np
import ray

ray.init()


@ray.remote
def compute_mean(arr: np.ndarray) -> float:
    return float(np.mean(arr))


big_array = np.arange(10_000_000, dtype=np.float32)
big_array_ref = ray.put(big_array)

refs = [compute_mean.remote(big_array_ref) for _ in range(4)]
print(ray.get(refs))

ray.shutdown()
```

`ray.put()` 返回 ObjectRef。Ray 的 remote objects 是 immutable，创建之后不应该被原地修改。这个设计让对象可以安全地被多个 worker 共享或复制，without coordinating mutable shared state。

常见经验：

- 大的只读数据，put once and pass ObjectRef。
- 小参数直接传，simple values are fine as normal arguments。
- 不要把 `ray.get()` 放在提交 task 的循环内部，否则会把并行执行变成串行等待。

<br>

## 6. Actor for Stateful Work

Task 适合无状态计算。需要维护状态时，用 Actor。

```python
import ray

ray.init()


@ray.remote
class Counter:
    def __init__(self):
        self.value = 0

    def add(self, n: int = 1) -> int:
        self.value += n
        return self.value

    def get(self) -> int:
        return self.value


counter = Counter.remote()

refs = [counter.add.remote(1) for _ in range(5)]
print(ray.get(refs))
print(ray.get(counter.get.remote()))

ray.shutdown()
```

`Counter.remote()` 创建一个远程实例，a stateful actor process。`counter.add.remote()` 调用 actor method，方法可以读写 `self.value`。默认情况下，同一个 Actor 的方法会按顺序执行，所以它适合维护一致的内部状态。

如果你想要并发状态，可以创建多个 Actor，把数据分片：

```python
import ray

ray.init()


@ray.remote
class Shard:
    def __init__(self, shard_id: int):
        self.shard_id = shard_id
        self.items = []

    def add(self, item: str) -> int:
        self.items.append(item)
        return len(self.items)

    def size(self) -> int:
        return len(self.items)


shards = [Shard.remote(i) for i in range(4)]

for i in range(20):
    shard = shards[i % len(shards)]
    shard.add.remote(f"item-{i}")

sizes = ray.get([shard.size.remote() for shard in shards])
print(sizes)

ray.shutdown()
```

这种模式叫 sharded actors，用多个 actor 分担状态和吞吐量，similar to partitioning state across workers。

<br>

## 7. Resource Requirements

Ray 调度 task 和 actor 时会检查资源。默认 task 使用 1 个 logical CPU。对于 Actor，最好显式设置 `num_cpus`，避免默认行为带来的误解。

```python
import os
import ray

ray.init()


@ray.remote(num_cpus=2)
def cpu_heavy_task(x: int) -> tuple[int, int]:
    return (os.getpid(), x * x)


@ray.remote(num_cpus=1)
class ModelWorker:
    def predict(self, x: float) -> float:
        return x * 2.0


task_refs = [cpu_heavy_task.remote(i) for i in range(4)]
model = ModelWorker.remote()
pred_ref = model.predict.remote(3.0)

print(ray.get(task_refs))
print(ray.get(pred_ref))

ray.shutdown()
```

`num_cpus` 是调度约束，a scheduling constraint, not a Python threading instruction。如果 task 是 IO-bound，可以设置小于 1 的 fractional CPU；如果 task 需要 GPU，可以设置 `num_gpus=1`。

```python
@ray.remote(num_gpus=1)
def gpu_task(batch):
    # Use PyTorch, TensorFlow, JAX, or other GPU code here.
    return len(batch)
```

Ray 会通过 `CUDA_VISIBLE_DEVICES` 等机制限制 worker 看到的 GPU。业务代码仍然需要自己正确使用 GPU library。

<br>

## 8. Nested Tasks

一个 remote task 里也可以继续提交 remote task，这叫 nested tasks。它适合递归拆分问题，但要注意 task 粒度，too many tiny tasks can make scheduling overhead dominate。

```python
import ray

ray.init()


@ray.remote
def square(x: int) -> int:
    return x * x


@ray.remote
def sum_of_squares(start: int, end: int) -> int:
    refs = [square.remote(i) for i in range(start, end)]
    return sum(ray.get(refs))


refs = [
    sum_of_squares.remote(0, 10),
    sum_of_squares.remote(10, 20),
    sum_of_squares.remote(20, 30),
]

partial_sums = ray.get(refs)
print(partial_sums)
print(sum(partial_sums))

ray.shutdown()
```

原则：让 task 有足够工作量。A good Ray task usually does tens or hundreds of milliseconds of work at minimum, often more。

<br>

## 9. Error Handling

Remote task 里的异常不会在 `.remote()` 时抛出，而是在 `ray.get()` 时回到 driver。

```python
import ray

ray.init()


@ray.remote
def divide(a: float, b: float) -> float:
    return a / b


refs = [divide.remote(10, 2), divide.remote(10, 0)]

for ref in refs:
    try:
        print(ray.get(ref))
    except Exception as exc:
        print(type(exc).__name__, exc)

ray.shutdown()
```

这让你可以先批量提交任务，然后集中处理成功和失败。In production, combine this with retries, logging, and idempotent task design。

<br>

## 10. Ray Data

Ray Data 用来做分布式数据读取、预处理和批处理，a distributed data processing library that works well with files, pandas-like batches, and ML pipelines。

```python
import ray

ray.init()

ds = ray.data.from_items(
    [
        {"text": "Ray Core"},
        {"text": "Ray Data"},
        {"text": "Ray Train"},
        {"text": "Ray Serve"},
    ]
)


def add_length(batch):
    batch["length"] = batch["text"].str.len()
    return batch


result = ds.map_batches(add_length, batch_format="pandas")
result.show()

ray.shutdown()
```

`Dataset.map_batches()` 以 batch 为单位处理数据，batch can be pandas DataFrame, pyarrow Table, numpy dict, or another supported batch format。和逐行 map 相比，batch 处理通常更快，也更适合向量化计算。

读取文件时通常这样写：

```python
import ray

ray.init()

ds = ray.data.read_parquet("s3://bucket/path/to/data/")
ds = ds.map_batches(preprocess_fn, batch_format="pandas")

for batch in ds.iter_batches(batch_size=1024, batch_format="pandas"):
    print(batch.head())

ray.shutdown()
```

Ray Data 的定位不是替代所有 SQL engine，而是把 data loading, preprocessing, and batch inference 放进 Ray 的统一执行模型里。

<br>

## 11. Ray Train

Ray Train 用来做分布式训练，especially data-parallel training for PyTorch, TensorFlow, XGBoost, LightGBM, Hugging Face, and related frameworks。

下面是一个很小的 PyTorch 线性回归例子。真实项目里你会换成自己的 dataset、model、loss 和 checkpoint。

```python
import torch
from torch import nn

import ray
import ray.train
import ray.train.torch
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, x):
        return self.layer(x)


def train_loop_per_worker(config):
    dataset = ray.train.get_dataset_shard("train")

    model = LinearModel()
    model = ray.train.torch.prepare_model(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
    loss_fn = nn.MSELoss()

    for epoch in range(config["epochs"]):
        last_loss = None

        for batch in dataset.iter_torch_batches(
            batch_size=config["batch_size"],
            dtypes=torch.float32,
        ):
            x = batch["x"].unsqueeze(1)
            y = batch["y"].unsqueeze(1)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            last_loss = loss.item()

        ray.train.report({"loss": last_loss, "epoch": epoch})


if __name__ == "__main__":
    ray.init()

    train_ds = ray.data.from_items(
        [{"x": float(i), "y": 2.0 * float(i) + 1.0} for i in range(1024)]
    )

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={"lr": 0.000001, "epochs": 5, "batch_size": 64},
        scaling_config=ScalingConfig(num_workers=2, use_gpu=False),
        datasets={"train": train_ds},
    )

    result = trainer.fit()
    print(result.metrics)

    ray.shutdown()
```

`TorchTrainer` 会启动多个 training worker。`prepare_model()` 会把模型准备成分布式训练形式，usually wrapping it with DistributedDataParallel when needed。`get_dataset_shard("train")` 让每个 worker 读取自己的数据分片，so workers do not all train on the exact same full dataset unless you choose that behavior。

<br>

## 12. Ray Tune

Ray Tune 用来做超参数搜索，a scalable hyperparameter tuning library。你定义训练函数和搜索空间，Tune 并行运行多个 trial。

```python
import math
import time

from ray import tune


def trainable(config):
    x = config["x"]
    y = config["y"]

    score = (x - 3) ** 2 + math.sin(y)
    time.sleep(0.2)

    tune.report({"score": score})


tuner = tune.Tuner(
    trainable=trainable,
    param_space={
        "x": tune.uniform(-10, 10),
        "y": tune.uniform(0, 6.28),
    },
    tune_config=tune.TuneConfig(
        num_samples=20,
        metric="score",
        mode="min",
    ),
)

results = tuner.fit()
best = results.get_best_result(metric="score", mode="min")
print(best.config)
print(best.metrics)
```

Trial 是一次具体超参数配置下的运行，one training run with one sampled configuration。`num_samples=20` 表示采样 20 组配置。实际训练时，你会在 `trainable(config)` 里训练模型，并定期 `tune.report()` 指标。

<br>

## 13. Ray Serve

Ray Serve 用来部署在线推理或 Python 服务，a scalable serving library for online inference APIs and general Python business logic。

```python
from ray import serve


@serve.deployment(num_replicas=2)
class Adder:
    def __init__(self, bias: int):
        self.bias = bias

    def __call__(self, request):
        x = int(request.query_params.get("x", "0"))
        return {"result": x + self.bias}


app = Adder.bind(bias=10)
serve.run(app, route_prefix="/add", blocking=True)
```

运行后可以在另一个 terminal 请求：

```bash
curl "http://127.0.0.1:8000/add?x=5"
```

`Deployment` 是一个可扩缩容的服务单元，a class or function wrapped by `@serve.deployment` and run as one or more replicas。Replica 本质上运行在 Ray Actor 上，所以 Serve 继承了 Ray 的调度、资源隔离和多节点扩展能力。

<br>

## 14. Local Cluster and Dashboard

本地学习时，`ray.init()` 足够了。需要手动启动 head node 时：

```bash
ray start --head --dashboard-host=0.0.0.0
```

然后在 Python 里连接：

```python
import ray

ray.init(address="auto")
print(ray.cluster_resources())
```

停止：

```bash
ray stop
```

Dashboard 默认在 `http://127.0.0.1:8265`。Dashboard 可以查看 task、actor、worker、object store、logs、resource usage，是调试 Ray 程序时最应该打开的页面。

<br>

## 15. A Progressive Mini Project

目标：批量处理文本，统计长度，并保留一个全局进度计数器。

### Step 1: Start with normal Python

```python
texts = ["ray", "distributed", "actor", "object store"]
lengths = [len(text) for text in texts]
print(lengths)
```

### Step 2: Turn pure function into Ray Task

```python
import ray

ray.init()


@ray.remote
def text_length(text: str) -> int:
    return len(text)


texts = ["ray", "distributed", "actor", "object store"]
refs = [text_length.remote(text) for text in texts]
print(ray.get(refs))

ray.shutdown()
```

### Step 3: Add Actor state

```python
import ray

ray.init()


@ray.remote
class Progress:
    def __init__(self):
        self.done = 0

    def mark_done(self) -> int:
        self.done += 1
        return self.done

    def get(self) -> int:
        return self.done


@ray.remote
def text_length(text: str, progress) -> tuple[str, int]:
    n = len(text)
    ray.get(progress.mark_done.remote())
    return (text, n)


progress = Progress.remote()
texts = ["ray", "distributed", "actor", "object store"]
refs = [text_length.remote(text, progress) for text in texts]

print(ray.get(refs))
print(ray.get(progress.get.remote()))

ray.shutdown()
```

### Step 4: Use ray.wait for streaming output

```python
import random
import time
import ray

ray.init()


@ray.remote
def text_length(text: str) -> tuple[str, int]:
    time.sleep(random.uniform(0.1, 1.0))
    return (text, len(text))


remaining = [
    text_length.remote(text)
    for text in ["ray", "distributed", "actor", "object store", "serve"]
]

while remaining:
    ready, remaining = ray.wait(remaining, num_returns=1)
    text, length = ray.get(ready[0])
    print(f"{text}: {length}")

ray.shutdown()
```

这个小项目覆盖了 Ray Core 最重要的使用路径：remote task for parallel work, actor for state, ObjectRef for async execution, ray.wait for streaming completion。

<br>

## 16. Common Pitfalls

### 1) Calling ray.get too early

下面写法会串行化：

```python
results = []
for x in range(10):
    ref = f.remote(x)
    results.append(ray.get(ref))
```

更好的写法：

```python
refs = [f.remote(x) for x in range(10)]
results = ray.get(refs)
```

### 2) Making tasks too small

如果每个 task 只做几微秒的工作，调度开销会比计算本身更大。Batch small work into larger tasks，让每个 task 至少有足够的实际计算量。

### 3) Using Actor when Task is enough

Actor 有状态，但也更难扩展。If the operation is stateless, prefer tasks。只有需要维护模型、缓存、连接或可变状态时，再用 Actor。

### 4) Forgetting resources

显式写资源需求可以减少调度意外：

```python
@ray.remote(num_cpus=4)
def heavy_cpu_job():
    ...


@ray.remote(num_gpus=1)
class GPUModel:
    ...
```

### 5) Passing huge objects repeatedly

大对象多次复用时，用 `ray.put()`。This avoids repeatedly serializing the same large read-only object from the driver。

<br>

## 17. Learning Path

1. 先熟练 Ray Core：`ray.init()`、`@ray.remote`、`.remote()`、`ray.get()`、`ray.wait()`、`ray.put()`、Actor。
2. 再学资源控制：`num_cpus`、`num_gpus`、custom resources、placement and scheduling basics。
3. 数据处理用 Ray Data：read files, transform batches, feed training or inference。
4. 分布式训练用 Ray Train：start from single-node multi-worker CPU, then move to GPUs。
5. 超参搜索用 Ray Tune：wrap your training function, define search space, report metrics。
6. 在线服务用 Ray Serve：deploy model or business logic, then tune replicas and resources。
7. 最后再学集群：head node, worker nodes, Dashboard, logs, runtime environments, Kubernetes if needed。

<br>

## 18. Reference Checklist

常用 API 记忆表：

| API | Use |
| --- | --- |
| `ray.init()` | 启动或连接 Ray runtime |
| `@ray.remote` | 把函数变成 Task，把类变成 Actor |
| `f.remote()` | 异步提交 task |
| `Actor.remote()` | 创建 actor instance |
| `actor.method.remote()` | 异步调用 actor method |
| `ray.get(ref)` | 从 ObjectRef 获取真实值 |
| `ray.wait(refs)` | 等一部分 ObjectRef 完成 |
| `ray.put(obj)` | 把对象放入 Object Store |
| `.options(num_cpus=..., num_gpus=...)` | 覆盖 task 或 actor 资源 |
| `ray.data.Dataset.map_batches()` | 批处理数据 |
| `TorchTrainer` | 启动分布式 PyTorch 训练 |
| `tune.Tuner` | 启动超参数搜索 |
| `@serve.deployment` | 定义可部署服务单元 |

<br>

## 19. Official References

- [Installing Ray](https://docs.ray.io/en/latest/ray-overview/installation.html)
- [Getting Started with Ray](https://docs.ray.io/en/latest/ray-overview/getting-started.html)
- [Ray Core Objects](https://docs.ray.io/en/latest/ray-core/objects.html)
- [Ray Core Actors](https://docs.ray.io/en/latest/ray-core/actors.html)
- [Ray Core Resources](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html)
- [Ray Data map_batches](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html)
- [Ray Train](https://docs.ray.io/en/latest/train/train.html)
- [TorchTrainer](https://docs.ray.io/en/latest/train/api/doc/ray.train.torch.TorchTrainer.html)
- [Ray Tune Tuner](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.Tuner.html)
- [Ray Serve](https://docs.ray.io/en/latest/serve/index.html)
