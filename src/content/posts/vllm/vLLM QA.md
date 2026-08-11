---
title: "vLLM QA"
published: 2026-08-05
description: "vLLM QA"
image: ""
tags: ["vllm","vLLM QA"]
category: vllm
draft: false
lang: ""
createdAt: "2026-08-05T13:54:42.821.801932051Z"
---

# request 怎么进 scheduler

# How does a request enter the scheduler?

请求先在 frontend 被处理成 `EngineCoreRequest`，再由 `EngineCore` 转成 scheduler 内部的 `Request`，最后调用 `scheduler.add_request()` 进入 waiting 队列。  
A request is first processed into an `EngineCoreRequest` in the frontend, then converted by `EngineCore` into the scheduler’s internal `Request`, and finally enters the waiting queue via `scheduler.add_request()`.



## ZMQ怎么启动的

````python
ZMQ 这段在 V1 multiprocess engine 里是这样走的：

```text
AsyncLLM._add_request()
  -> AsyncMPClient.add_request_async(request)
  -> _send_input(ADD, request)
  -> ROUTER socket send_multipart(...)
  -> EngineCoreProc DEALER socket recv_multipart(...)
  -> decode EngineCoreRequest
  -> preprocess_add_request()
  -> input_queue.put_nowait((ADD, (Request, wave)))
  -> EngineCore busy loop _process_input_queue()
  -> _handle_client_request(ADD, ...)
  -> EngineCore.add_request()
  -> scheduler.add_request()
```

核心点：

1. frontend 侧是 `ROUTER` socket  
   在 `vllm/v1/engine/core_client.py` 里，`MPClient` 创建 `self.input_socket`，类型是 `zmq.ROUTER`，并 bind 到 input address。

2. EngineCoreProc 侧是 `DEALER` socket  
   在 `vllm/v1/engine/core.py` 的 `process_input_sockets()` 里，engine process 用 `zmq.DEALER` 连接 frontend 的 ROUTER，并设置自己的 `identity`，通常是 data-parallel rank 的 bytes。

3. engine 先发 ready  
   DEALER 必须先发一条 ready payload 给 ROUTER。frontend 在 `MPClient.__init__()` 里等待每个 engine 的 ready message：

```text
ROUTER recv:
  identity, payload
```

这样 ROUTER 知道后面消息该 route 到哪个 engine identity。

4. 发送 request 时，frontend 手动把 engine identity 放第一帧  
   `AsyncMPClient._send_input()` 会组 multipart message：

```text
(engine_identity, request_type, *encoded_request_frames)
```

对于 add request：

```text
(engine_identity, b"\x00", *msgpack(EngineCoreRequest))
```

其中 `b"\x00"` 是 `EngineCoreRequestType.ADD`。

5. EngineCoreProc 收到时，identity 已被 ROUTER/DEALER 路由层消费掉  
   所以 `process_input_sockets()` 里收到的是：

```text
type_frame, *data_frames
```

也就是：

```text
b"\x00", *msgpack_frames
```

然后它判断 `request_type == ADD`，用 `MsgpackDecoder(EngineCoreRequest)` 解码。

6. ADD 消息不会直接在 socket IO thread 里进 scheduler  
   它先在 IO thread 里做：

```text
EngineCoreRequest -> preprocess_add_request() -> Request
```

然后放入 `self.input_queue`。真正调用 scheduler 的是 EngineCore busy loop：

```text
input_queue -> _handle_client_request(ADD) -> add_request() -> scheduler.add_request()
```

一句话：ZMQ 只负责把 frontend 里的 `EngineCoreRequest` 作为 multipart msgpack 消息送到 EngineCoreProc；真正进入 scheduler 是 ZMQ 收包后，经 `preprocess_add_request()` 转成 `Request`，再由 EngineCore 主循环调用 `self.scheduler.add_request(request)`。
````



## AsyncLLM和LLM

````python
对，`LLM` 的非多进程路径不走 ZMQ。它用的是 `InprocClient`，也就是 frontend `LLMEngine` 和 `EngineCore` 在同一个 Python 进程里，直接函数调用。

主链路是：

```text
examples/basic/offline_inference/basic.py
  -> LLM.generate(...)
  -> OfflineInferenceMixin._run_completion(...)
  -> _add_completion_requests(...)
  -> _render_and_add_requests(...)
  -> _add_request(...)
  -> self.llm_engine.add_request(...)
  -> LLMEngine.add_request(...)
  -> self.engine_core.add_request(...)
  -> InprocClient.add_request(...)
  -> EngineCore.preprocess_add_request(...)
  -> EngineCore.add_request(...)
  -> Scheduler.add_request(...)
```

关键差异在这里：

```text
AsyncLLM / multiprocess:
  AsyncMPClient.add_request_async()
  -> ZMQ ROUTER/DEALER
  -> EngineCoreProc.process_input_sockets()
  -> input_queue
  -> EngineCore.add_request()

LLM / non-multiprocess:
  InprocClient.add_request()
  -> EngineCore.preprocess_add_request()
  -> EngineCore.add_request()
```

也就是说，`LLMEngine` 初始化时会通过 `EngineCoreClient.make_client(...)` 创建 client。非多进程时拿到的是 `InprocClient`，它的 `add_request()` 很直接：

```text
EngineCoreRequest
  -> preprocess_add_request()
  -> Request
  -> engine_core.add_request()
  -> scheduler.add_request()
```

后面执行也不是后台 busy loop，而是 `LLM` 自己同步驱动：

```text
OfflineInferenceMixin._run_engine()
  while llm_engine.has_unfinished_requests():
      llm_engine.step()
```

`LLMEngine.step()` 里再调用：

```text
InprocClient.get_output()
  -> EngineCore.step_fn()
  -> scheduler.schedule()
  -> model_executor.execute_model(...)
  -> scheduler.update_from_output(...)
```

所以一句话：`LLM` 非多进程路径里，request 进入 scheduler 是同步、同进程、直接函数调用；没有 ZMQ、没有 socket IO thread、没有 `input_queue` 转发。
````





## LLM 和 AsyncLLM 的 busy_loop

````python
是的，`AsyncLLM` 有 busy loop，但准确说：

```text
AsyncLLM 本身所在的 frontend 进程没有 EngineCore busy loop；
EngineCore 后台进程里有 run_busy_loop()。
```

`AsyncLLM.__init__()` 里直接创建的是：

```text
EngineCoreClient.make_async_mp_client(...)
```

也就是 `AsyncMPClient`。它会启动后台 `EngineCoreProc`，而 `EngineCoreProc.run_engine_core()` 里最终调用：

```text
engine_core.run_busy_loop()
```

所以 `AsyncLLM` 的结构是：

```text
frontend asyncio process
  AsyncLLM
  output_handler asyncio task
  AsyncMPClient
  ZMQ ROUTER/PULL

background EngineCore process
  EngineCoreProc
  ZMQ DEALER/PUSH
  input_queue
  run_busy_loop()
  Scheduler
  ModelExecutor
```

## busy loop 是 serve 专属吗？

不是。busy loop 不是 `vllm serve` 专属功能，它属于 **multiprocess EngineCore**。

只是 `vllm serve` 用的是 `AsyncLLM`，而 `AsyncLLM` 当前必须用 async multiprocess engine。代码里也写了：

```text
asyncio_mode and not multiprocess_mode -> NotImplementedError
```

所以：

```text
vllm serve
  -> AsyncLLM
  -> AsyncMPClient
  -> EngineCoreProc
  -> run_busy_loop()
```

看起来就像 busy loop 是 serve 的功能，但本质上它是 **后台 EngineCore 进程的调度执行循环**。

## 和 LLM 的区别

### `LLM` 非多进程

```text
LLM.generate()
  -> 当前线程 while has_unfinished_requests()
  -> llm_engine.step()
  -> InprocClient.get_output()
  -> EngineCore.step_fn()
```

没有后台 busy loop，没有 ZMQ。

### `LLM` 多进程

如果 `LLMEngine` 开了 multiprocess mode，也会走：

```text
SyncMPClient
  -> ZMQ
  -> EngineCoreProc
  -> run_busy_loop()
```

### `AsyncLLM`

基本就是：

```text
AsyncMPClient
  -> ZMQ
  -> EngineCoreProc
  -> run_busy_loop()
```

所以一句话：`AsyncLLM` 有 EngineCore busy loop，但它跑在后台 EngineCore 进程里；`vllm serve` 只是通过 `AsyncLLM` 使用了这个机制，不是 serve 独有。
````



## busyloop是把请求循环拿出来吗？然后交给谁去处理

````python
可以这么理解：

`busy_loop` 做两件事：

1. 从 `input_queue` 里循环取请求  
   例如 `ADD`、`ABORT`、`UTILITY`。

2. 推进一次 engine 执行  
   调 `scheduler.schedule()` 选 batch，再交给 `model_executor.execute_model()` 跑模型，最后 `scheduler.update_from_output()` 更新 request 状态。

所以 `ADD` 请求的路径是：

```text
input_queue
  -> _handle_client_request(ADD)
  -> EngineCore.add_request()
  -> Scheduler.add_request()
  -> waiting queue
```

然后真正模型执行路径是：

```text
busy_loop
  -> _process_engine_step()
  -> scheduler.schedule()
  -> model_executor.execute_model()
  -> scheduler.update_from_output()
```

一句话：`busy_loop` 先把新请求交给 scheduler 入队，然后每轮让 scheduler 选一批请求交给 model executor 执行。


主要看这几个地方：

`vllm/v1/engine/async_llm.py`

`AsyncLLM._add_request()`：frontend 把请求交给 EngineCore client。

```400:412:vllm/v1/engine/async_llm.py
    async def _add_request(
        self,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None,
        index: int,
        queue: RequestOutputCollector,
    ):
        # Add the request to OutputProcessor (this process).
        self.output_processor.add_request(request, prompt, parent_req, index, queue)

        # Add the EngineCoreRequest to EngineCore (separate process).
        await self.engine_core.add_request_async(request)
```

`vllm/v1/engine/core_client.py`

`AsyncMPClient.add_request_async()`：通过 ZMQ 发 `ADD`。

```1121:1124:vllm/v1/engine/core_client.py
    async def add_request_async(self, request: EngineCoreRequest) -> None:
        request.client_index = self.client_index
        await self._send_input(EngineCoreRequestType.ADD, request)
        self._ensure_output_queue_task()
```

`vllm/v1/engine/core.py`

`run_busy_loop()`：后台 EngineCore 进程的循环。

```1259:1265:vllm/v1/engine/core.py
    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""
        while self._handle_shutdown():
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # 2) Step the engine core and return the outputs.
            self._process_engine_step()
```

`_handle_client_request()`：从 `input_queue` 拿到 `ADD` 后交给 `EngineCore.add_request()`。

```1372:1383:vllm/v1/engine/core.py
    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        """Dispatch request from client."""

        if request_type == EngineCoreRequestType.WAKEUP:
            return
        elif request_type == EngineCoreRequestType.ADD:
            req, request_wave = request
            if self._reject_add_in_shutdown(req):
                return
            self.add_request(req, request_wave)
```

`EngineCore.add_request()`：真正调用 scheduler。

```372:403:vllm/v1/engine/core.py
    def add_request(self, request: Request, request_wave: int = 0):
        """Add request to the scheduler.
...
        self.scheduler.add_request(request)
```

`vllm/v1/core/sched/scheduler.py`

`Scheduler.add_request()`：请求进入 scheduler 的 waiting 队列。

```2018:2036:vllm/v1/core/sched/scheduler.py
    def add_request(self, request: Request) -> None:
        existing = self.requests.get(request.request_id)
...
        else:
            if request.resumable:
                request.streaming_queue = deque()
            self._enqueue_waiting_request(request)
            self.requests[request.request_id] = request
```

核心顺序就是：

```text
AsyncLLM._add_request()
 -> AsyncMPClient.add_request_async()
 -> ZMQ ADD
 -> EngineCoreProc input_queue
 -> run_busy_loop()
 -> _handle_client_request(ADD)
 -> EngineCore.add_request()
 -> Scheduler.add_request()
```
````



## Scheduler.add_request后，vllm是如何取request做处理的？做了哪些？

````python
`Scheduler.add_request()` 之后，scheduler 不会立刻跑模型。它只是把 request 放进队列。真正“取出来处理”发生在下一次 engine step：

```text
EngineCore busy_loop
  -> _process_engine_step()
  -> scheduler.schedule()
  -> model_executor.execute_model()
  -> scheduler.update_from_output()
```

## 1. 入队后在哪

`Scheduler.add_request()` 里：

```text
self.requests[request_id] = request
self._enqueue_waiting_request(request)
```

普通 request 进：

```text
self.waiting
```

暂时不能调度的 request 进：

```text
self.skipped_waiting
```

比如 structured output grammar 还没准备好、远端 KV 还没 ready、streaming input 等。

## 2. 谁把它取出来

在 `vllm/v1/core/sched/scheduler.py` 的 `Scheduler.schedule()` 里取。

核心逻辑是：

```text
while waiting/skipped_waiting 还有请求，并且 token_budget > 0:
    request_queue = _select_waiting_queue_for_scheduling()
    request = request_queue.peek_request()
    ...
    request = request_queue.pop_request()
    self.running.append(request)
```

也就是先 `peek_request()` 看队头能不能调度，确认可以后才 `pop_request()`，然后放进 `self.running`。

## 3. 取出来后做了哪些事

对一个新 request，scheduler 主要做这些：

1. 检查并发限制  
   看 `len(self.running)` 是否超过 `max_num_running_reqs`。

2. 检查 request 是否暂时 blocked  
   例如 grammar 未完成、远端 KV 未完成、streaming input 暂停。

3. 检查 LoRA 约束  
   如果当前 batch 里的 LoRA 数超过限制，就先跳过。

4. 查 prefix cache  
   如果开启 prefix caching，会检查 prompt 前缀是否已有 KV cache，可复用的 token 会记为 already computed。

5. 处理 KV transfer / P-D disaggregation  
   如果有远端 KV，可能先进入 `WAITING_FOR_REMOTE_KVS`，不马上跑模型。

6. 决定本轮算多少 token  
   计算 `num_new_tokens`，受这些限制影响：

```text
token_budget
max_model_len
chunked prefill
long_prefill_token_threshold
spec decode
encoder/multimodal budget
```

7. 分配 KV cache slots  
   调 `kv_cache_manager.allocate_slots(...)` 给这个 request 分配 KV blocks。

8. 从 waiting 变 running  
   成功后：

```text
request_queue.pop_request()
self.running.append(request)
request.status = RUNNING
num_scheduled_tokens[request_id] = num_new_tokens
```

9. 生成 `SchedulerOutput`  
   里面告诉 worker/model runner：

```text
哪些是新 request
哪些是已有 running request
每个 request 本轮算几个 token
KV block ids 是什么
encoder inputs 是什么
spec decode tokens 是什么
```

## 4. 然后交给谁处理

`EngineCore._process_engine_step()` 里：

```text
scheduler_output = scheduler.schedule()
model_output = model_executor.execute_model(scheduler_output)
scheduler.update_from_output(scheduler_output, model_output)
```

所以 scheduler 只负责“选谁、算多少、分配资源”。真正跑模型的是：

```text
model_executor.execute_model()
```

模型跑完后，`scheduler.update_from_output()` 再更新 request：

```text
把新 token append 到 request
更新 num_computed_tokens / num_in_flight_tokens
检查 stop / max_tokens / eos
如果没结束，留在 running
如果结束，释放 KV cache，移出 requests
生成 EngineCoreOutput
```

一句话：`Scheduler.add_request()` 只是入队；下一轮 `Scheduler.schedule()` 从 `waiting` 里挑 request，分配 KV cache 和 token budget，放进 `running`，生成 `SchedulerOutput` 交给 `model_executor` 跑模型，最后 `update_from_output()` 更新 request 状态。






主要看这 4 段代码。

`vllm/v1/engine/core.py`

`EngineCore._process_engine_step()`：每轮先让 scheduler 选 batch，再交给 model executor，最后回写 scheduler。

```486:508:vllm/v1/engine/core.py
        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            return {}, False
        scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                model_output = self.model_executor.sample_tokens(grammar_output)

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
```

`vllm/v1/core/sched/scheduler.py`

`Scheduler.schedule()`：从 `waiting/skipped_waiting` 里看下一个 request。

```636:650:vllm/v1/core/sched/scheduler.py
# 接下来，调度处于等待状态（WAITING）的请求。
if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
    # 创建一个临时队列，用于存放本轮因条件不满足（如 Token 预算不够或 KV Cache 空间不足）而被跳过的等待请求
    step_skipped_waiting = create_request_queue(self.policy)

    # 只要存在待调度的请求（主等待队列或之前跳过的队列），且当前步骤的 Token 预算还未耗尽，就持续循环
    while (self.waiting or self.skipped_waiting) and token_budget > 0:
        # 暂停的流式会话（WAITING_FOR_STREAMING_REQ）虽然不在 `running` 列表中，
        # 但它们仍然占用着模型运行器（Model Runner）的请求槽位。
        # 计算当前已占用的总运行槽位数：当前运行中的请求数 + 正在等待流式输入的请求数
        num_running = len(self.running) + self.num_waiting_for_streaming_input
        
        # 如果已运行请求数达到或超过了系统允许的最大并发数（max_num_running_reqs），停止继续调度新请求
        if num_running >= self.max_num_running_reqs:
            break

        # 根据当前的调度策略（如 FCFS 先来先服务、优先级等），选择下一个要从中获取请求的等待队列
        request_queue = self._select_waiting_queue_for_scheduling()
        assert request_queue is not None

        # 查看（偷窥）所选队列队首的请求，但不立即将其出队
        # （后续代码会检查该请求所需的 Token 预算和 KV Cache 显存是否满足条件）
        request = request_queue.peek_request()
```

确认能调度后，把 request 从 waiting 取出，放进 running。

```948:991:vllm/v1/core/sched/scheduler.py
                request = request_queue.pop_request()
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    step_skipped_waiting.prepend_request(request)
...
                self.running.append(request)
...
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
...
                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
```

然后构造 `SchedulerOutput`，给 `model_executor.execute_model()` 用。

```1095:1100:vllm/v1/core/sched/scheduler.py
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
```

模型跑完后，`update_from_output()` 更新 request 状态。

```1503:1508:vllm/v1/core/sched/scheduler.py
    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
```

最核心路径就是：

```text
Scheduler.add_request()
  -> waiting queue

EngineCore._process_engine_step()
  -> Scheduler.schedule()
      -> request_queue.peek_request()
      -> request_queue.pop_request()
      -> self.running.append(request)
      -> request.status = RUNNING
      -> SchedulerOutput
  -> model_executor.execute_model(SchedulerOutput)
  -> Scheduler.update_from_output()
```
````

## self._should_throttle_prefills() 做了什么

````python
self._should_throttle_prefills() 返回一个布尔值，表示是否限制预填充（prefill）请求：
True：限制新请求的 prefill，优先保证已有请求的 decode（防止 decode 饥饿）
False：正常调度，允许新请求进行 prefill

默认情况下，vLLM 不会限制预填充（prefill）请求，总是返回 False，即允许所有新请求正常进行 prefill
注释明确指出这是一个 DP（Data Parallel，数据并行）专用的钩子方法：
只在 数据并行引擎核心（DP engine core） 中被重写（override）
非 DP 模式下，始终返回 False，不做任何限制


在 数据并行 场景下，多个 GPU 各自处理不同的请求批次。为了平衡各 GPU 的负载，可能需要：
根据当前各 GPU 的繁忙程度，动态决定是否让某个 GPU 接受新的 prefill 请求
避免某些 GPU 过载而其他 GPU 空闲


在 vLLM 中，`_should_throttle_prefills()` 的重写发生在**数据并行（Data Parallel）**相关的引擎实现中。让我为你详细说明：

## 重写位置

### 1. **DataParallelEngineCore** 类
这是主要的重写位置，位于 `vllm/v1/engine/core.py` 或相关文件中：

```python
class DataParallelEngineCore(EngineCore):
    def _should_throttle_prefills(self) -> bool:
        """DP 特有的限流逻辑"""
        # 检查各个 DP worker 的负载情况
        if self.dp_engine_cores:
            # 获取所有 worker 的待处理请求数
            pending_counts = [
                core.get_num_running_requests() 
                for core in self.dp_engine_cores
            ]
            # 如果某个 worker 负载过高，则限流
            max_pending = max(pending_counts)
            avg_pending = sum(pending_counts) / len(pending_counts)
            
            # 当最大负载超过平均负载的一定阈值时返回 True
            return max_pending > avg_pending * 1.2
        return False
```

### 2. **具体的重写策略**

vLLM 可能实现以下几种限流策略：

#### **基于负载均衡**
```python
def _should_throttle_prefills(self) -> bool:
    # 检查各个 DP rank 的队列长度
    queue_lengths = [core.get_queue_length() for core in self.dp_cores]
    max_len = max(queue_lengths)
    min_len = min(queue_lengths)
    
    # 如果最忙和最闲的 worker 差距过大，限制新请求
    return (max_len - min_len) > self.throttle_threshold
```

#### **基于 GPU 利用率**
```python
def _should_throttle_prefills(self) -> bool:
    # 检查所有 DP worker 的 GPU 利用率
    utilizations = [core.get_gpu_utilization() for core in self.dp_cores]
    # 如果任何 worker 利用率超过 90%，开始限流
    return any(util > 0.9 for util in utilizations)
```

## 调用链

```
EngineCore.schedule()
    └── _should_throttle_prefills()  # 多态调用
            ├── 基类: return False (默认)
            └── DataParallelEngineCore: 返回动态决策
```

## 实际使用场景

在数据并行推理中：
- **Worker 0**：正在处理大量请求，GPU 满载
- **Worker 1**：相对空闲
- 调度器调用 `_should_throttle_prefills()` → 返回 `True`
- 新请求的 prefill 被延迟，优先让 Worker 1 处理已有请求，避免 Worker 0 过载

这样设计的目的是**在数据并行场景下实现动态负载均衡**，防止个别 GPU 成为性能瓶颈。
````



## self.policy 策略有哪些，默认是什么？

````python
`self.policy` 只有两种：

可以通过参数配置：

```text
--scheduling-policy fcfs
--scheduling-policy priority
```


```13:18:vllm/vllm/v1/core/sched/request_queue.py
class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""

    FCFS = "fcfs"
    PRIORITY = "priority"
```

默认是 `fcfs`：# First-Come-First-Serve

```109:115:vllm/vllm/config/scheduler.py
    policy: SchedulerPolicy = "fcfs"
    """The scheduling policy to use:

    - "fcfs" means first come first served, i.e. requests are handled in order 
      of arrival.
    - "priority" means requests are handled based on given priority (lower
      value means earlier handling) and time of arrival deciding any ties)."""
```

`Scheduler.__init__()` 里把配置转成 enum，并创建对应队列：

```173:183:vllm/vllm/v1/core/sched/scheduler.py
        # Scheduling policy
        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)
        except ValueError as e:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            ) from e
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)
        # requests skipped in waiting flow due async deps or constraints.
        self.skipped_waiting = create_request_queue(self.policy)
```

队列类型在这里选：

```201:208:vllm/vllm/v1/core/sched/request_queue.py
def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
```

区别：

- `fcfs`：默认策略，先来先服务，底层是 `deque`，`append()` 入队，`popleft()` 出队。
- `priority`：优先级策略，底层是 heap，`priority` 数值越小越先处理；如果 priority 相同，就按 `arrival_time` 早的先处理。 the smaller the value, the higher the priority


一句话：默认是 `fcfs`；如果设成 `priority`，scheduler 的 waiting 队列会变成优先队列，priority 越小越优先。
````



## 请求怎么设置priority？ fcfs的请求需要设置什么吗？

````python
我没有找到额外的 scheduler 内部校验逻辑；核心行为是：`priority` 字段一直传到 `Request.priority`，只有 `priority` policy 的队列会用它排序，`fcfs` 队列不会用。
请求的 `priority` 是入口传进来的，默认是 `0`。

## 怎么设置

### 离线 `LLM.generate()`

可以传一个 `priority` 列表，长度要和 prompts 一样：

```python
llm.generate(
    prompts=["a", "b", "c"],
    sampling_params=sampling_params,
    priority=[0, -1, 10],
)
```

这里 **数值越小越优先**，所以 `-1` 比 `0` 先，`0` 比 `10` 先。

代码在 `vllm/entrypoints/llm.py`：

```422:429:vllm/entrypoints/llm.py
    def generate(
        self,
        prompts: PromptType | Sequence[PromptType],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
        *,
        use_tqdm: bool | Callable[..., tqdm] = True,
        lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
        priority: list[int] | None = None,
```

如果不传，默认全是 `0`：

```274:288:vllm/entrypoints/offline_utils.py
    def _priority_to_seq(
        self,
        priority: list[int] | None,
        num_requests: int,
    ) -> Sequence[int]:
        if priority is not None:
            if len(priority) != num_requests:
                raise ValueError(
                    f"The lengths of prompts ({num_requests}) "
                    f"and priority ({len(priority)}) must be the same."
                )

            return priority

        return [0] * num_requests
```

### OpenAI server 请求

`/v1/completions`、`/v1/chat/completions` 请求体里也有 `priority` 字段，默认 `0`。

```351:359:vllm/entrypoints/openai/chat_completion/protocol.py
    priority: int = Field(
        default=0,
        ge=_INT64_MIN,
        le=_INT64_MAX,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."
        ),
    )
```

然后 serving 会传给 engine：

```357:364:vllm/entrypoints/openai/chat_completion/serving.py
                generator = self.engine_client.generate(
                    engine_input,
                    sampling_params,
                    sub_request_id,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=request.priority,
                    data_parallel_rank=data_parallel_rank,
```

## priority 怎么传到 scheduler

路径是：

```text
LLM/OpenAI request priority
  -> LLMEngine.add_request / AsyncLLM.add_request
  -> InputProcessor.process_inputs(priority=...)
  -> EngineCoreRequest.priority
  -> Request.priority
  -> PriorityRequestQueue
```

这里把它写进 `EngineCoreRequest`：

```370:382:vllm/v1/engine/input_processor.py
        return EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            prompt_embeds=prompt_embeds,
            prompt_is_token_ids=prompt_is_token_ids,
            mm_features=mm_features,
            sampling_params=sampling_params,
            pooling_params=pooling_params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            cache_salt=decoder_inputs.get("cache_salt"),
            priority=priority,
            data_parallel_rank=data_parallel_rank,
```

`priority` 排序规则在 `Request.__lt__()`：

```319:329:vllm/v1/request.py
    def __lt__(self, other: "Request") -> bool:
        """
        Compare two requests based on priority, arrival time, and request ID.
        Used in priority scheduling.
        """
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        if self.request_id != other.request_id:
```

## FCFS 需要设置什么吗？

不需要。

`fcfs` 是默认策略，普通请求不用设置 `priority`，默认 `0` 就行。

在 `fcfs` 下，队列是 `deque`：

```75:84:vllm/v1/core/sched/request_queue.py
class FCFSRequestQueue(deque[Request], RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()
```

所以 `fcfs` 看的是入队顺序，不看 `request.priority`。

一句话：想用优先级，启动时设 `--scheduling-policy priority`，请求里传 `priority`，数值越小越先；默认 `fcfs` 不需要设置 priority，默认 `0` 即可。
````





## 推理调度器（Scheduler）核心方法schedule做了什么？

````python
这是一个复杂的**推理调度器（Scheduler）**核心方法，负责在每一步决定哪些请求可以执行、分配多少token、管理KV缓存等。我将整个方法分为**6大模块**来讲解：

---

## 📊 整体架构概览

```
schedule()
    ├── 1. 初始化阶段（设置状态、变量、预算）
    ├── 2. 调度 RUNNING 请求（正在执行的请求）
    │   ├── 检查可调度性
    │   ├── 计算token分配
    │   ├── 处理编码器输入
    │   ├── 分配KV缓存（可能触发抢占）
    │   └── 记录调度结果
    ├── 3. 调度 WAITING 请求（等待中的请求）
    │   ├── 检查并发限制
    │   ├── 检查缓存命中
    │   ├── 处理远程KV加载
    │   └── 分配资源并移入RUNNING
    ├── 4. 后处理阶段
    │   ├── 断言约束条件
    │   ├── 计算公共前缀
    │   └── 构建输出数据
    ├── 5. 构建调度输出（SchedulerOutput）
    └── 6. 连接器处理和状态更新
```

---

## 1️⃣ **初始化阶段** (第1-56行)

```python
self.current_step += 1  # 步数计数

# 三种调度队列
scheduled_new_reqs: list[Request] = []       # 新请求
scheduled_resumed_reqs: list[Request] = []   # 恢复的请求
scheduled_running_reqs: list[Request] = []   # 继续运行的请求
preempted_reqs: list[Request] = []           # 被抢占的请求

# 资源预算
token_budget = self.max_num_scheduled_tokens  # token预算
encoder_compute_budget = self.max_num_encoder_input_tokens  # 编码器预算

# DP预填充平衡策略
defer_prefills = throttle_prefills and not self.prefill_capacity_bound
```

**作用**：初始化所有调度需要的状态变量和资源预算。

---

## 2️⃣ **调度 RUNNING 请求** (第58-248行)

这是最复杂的部分，处理已经在执行的请求。

### 核心流程：

```python
while req_index < len(self.running) and token_budget > 0:
    request = self.running[req_index]
    
    # 2.1 检查是否可调度
    if request.num_computed_tokens + 2 - request.num_output_placeholders >= ...:
        continue  # 已完成的请求跳过
    
    if self.current_step < request.next_decode_eligible_step:
        continue  # PP约束
    
    # 2.2 计算本次分配的token数
    num_new_tokens = request.num_tokens_with_spec - request.num_computed_tokens
    num_new_tokens = min(num_new_tokens, token_budget)
    num_new_tokens = min(num_new_tokens, self.max_model_len - ...)
    
    # 2.3 处理编码器输入（多模态模型）
    if request.has_encoder_inputs:
        encoder_inputs_to_schedule, num_new_tokens = self._try_schedule_encoder_inputs(...)
    
    # 2.4 分配KV缓存（关键！）
    while True:
        new_blocks = self.kv_cache_manager.allocate_slots(...)
        if new_blocks is not None:
            break  # 分配成功
        
        # 分配失败 → 触发抢占
        if self.policy == SchedulingPolicy.PRIORITY:
            preempted_req = max(self.running, key=lambda r: r.priority)
            self.running.remove(preempted_req)
        else:
            preempted_req = self.running.pop()
        
        self._preempt_request(preempted_req, ...)
        preempted_reqs.append(preempted_req)
    
    # 2.5 记录调度结果
    scheduled_running_reqs.append(request)
    req_to_new_blocks[request_id] = new_blocks
    num_scheduled_tokens[request_id] = num_new_tokens
    token_budget -= num_new_tokens
```

**关键点**：
- **KV缓存分配**：如果缓存不足，会触发**抢占**（Preemption），移除低优先级请求
- **抢占策略**：`PRIORITY`策略抢占优先级最低的，否则抢占队尾（FCFS）
- **Token预算**：每个请求分配的token数受全局预算限制

---

## 3️⃣ **调度 WAITING 请求** (第250-535行)

处理等待队列中的新请求或恢复的请求。

### 核心流程：

```python
if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
    step_skipped_waiting = create_request_queue(self.policy)
    
    while (self.waiting or self.skipped_waiting) and token_budget > 0:
        # 3.1 检查并发限制
        num_running = len(self.running) + self.num_waiting_for_streaming_input
        if num_running >= self.max_num_running_reqs:
            break
        
        # 3.2 选择优先级最高的队列
        request_queue = self._select_waiting_queue_for_scheduling()
        request = request_queue.peek_request()
        
        # 3.3 检查缓存命中（Prefix Caching）
        if request.num_computed_tokens == 0:
            # 本地缓存查找
            new_computed_blocks, num_new_local_computed_tokens = 
                self.kv_cache_manager.get_computed_blocks(request)
            
            # 远程KV查找（分布式场景）
            if self.connector is not None:
                ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(...)
                num_external_computed_tokens = ext_tokens
            
            num_computed_tokens = num_new_local_computed_tokens + num_external_computed_tokens
        
        # 3.4 分配KV缓存
        new_blocks = self.kv_cache_manager.allocate_slots(
            request,
            num_new_tokens,
            new_computed_blocks=new_computed_blocks,
            delay_cache_blocks=load_kv_async,  # 远程加载时延迟缓存
            ...
        )
        
        if new_blocks is None:
            break  # 无法分配
        
        # 3.5 移入RUNNING队列
        request = request_queue.pop_request()
        self.running.append(request)
        request.status = RequestStatus.RUNNING
        scheduled_new_reqs.append(request)
```

**关键点**：
- **Prefix Caching**：利用缓存命中减少计算
- **异步KV加载**：`load_kv_async`表示从远程加载KV，此时不分配本地缓存
- **跳过机制**：无法调度的请求放入`step_skipped_waiting`，后续重试

---

## 4️⃣ **后处理阶段** (第537-560行)

```python
# 4.1 断言约束
assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
assert len(self.running) <= self.max_num_running_reqs

# 4.2 计算公共前缀（用于Cascade Attention优化）
if self.running:
    num_common_prefix_blocks = self.kv_cache_manager.get_num_common_prefix_blocks(...)

# 4.3 构建请求数据
new_reqs_data = [NewRequestData.from_request(req, block_ids) for req in scheduled_new_reqs]
cached_reqs_data = self._make_cached_request_data(scheduled_running_reqs, ...)
```

---

## 5️⃣ **构建调度输出** (第562-610行)

```python
scheduler_output = SchedulerOutput(
    scheduled_new_reqs=new_reqs_data,        # 新请求
    scheduled_cached_reqs=cached_reqs_data,  # 继续运行的请求
    num_scheduled_tokens=num_scheduled_tokens,  # 每个请求分配的token数
    scheduled_spec_decode_tokens=...,        # 推测解码token
    scheduled_encoder_inputs=...,            # 编码器输入
    num_common_prefix_blocks=...,            # 公共前缀
    preempted_req_ids=...,                   # 被抢占的请求
    finished_req_ids=...,                    # 已完成的请求
    new_block_ids_to_zero=...,               # 需要清零的KV块
    num_spec_tokens_to_schedule=...,         # 推测解码数
)
```

---

## 6️⃣ **连接器处理和状态更新** (第612-628行)

```python
# KV连接器（分布式KV传输）
if self.connector is not None:
    meta = self._build_kv_connector_meta(self.connector, scheduler_output)
    scheduler_output.kv_connector_metadata = meta

# 编码器缓存连接器
if self.ec_connector is not None:
    ec_meta = self.ec_connector.build_connector_meta(scheduler_output)
    scheduler_output.ec_connector_metadata = ec_meta

# 更新调度器状态
self._update_after_schedule(scheduler_output)
return scheduler_output
```

---

## 🎯 **核心设计思想**

### 1. **Token级调度**
不是按请求调度，而是按token数量调度，支持chunked prefill

### 2. **KV缓存管理**
- 分配失败 → 抢占
- 支持prefix caching和远程KV传输

### 3. **多种调度策略**
- FCFS（先来先服务）
- PRIORITY（优先级）

### 4. **推测解码支持**
- 动态调整spec token数量
- 支持padding保持cudagraph一致性

### 5. **分布式支持**
- KVConnector：远程KV加载/存储
- ECConnector：编码器缓存管理

---

## 📈 **调度流程示例**

```
Step N:
1. 预算: token_budget=1024, 当前running=5个请求
2. 遍历running:
   - Req A: 分配256 tokens, 更新缓存
   - Req B: 分配128 tokens
   - Req C: 缓存不足 → 抢占 Req E (低优先级)
   - Req D: 分配512 tokens, 预算用完
3. 遍历waiting:
   - 尝试调度新请求, 但token_budget=0, 跳过
4. 输出: A,B,D继续执行, C被抢占, 新请求等待
```

这个调度器设计得非常灵活，支持了vLLM中多种高级特性（speculative decoding、chunked prefill、prefix caching、分布式推理等）。
````





## 带入数值，然后这个分为几步解释Scheduler.schedule

### next_decode_eligible_step 怎么更新

```python
next_decode_eligible_step 初始是 0：

request.next_decode_eligible_step = 0
所以刚开始一定可以被调度，因为：

current_step >= 0
它只在 AsyncScheduler._update_after_schedule() 里更新：

if self.use_v2_model_runner:
    request.next_decode_eligible_step = self.current_step + self.pp_size
其中：

self.pp_size = pipeline_parallel_size
所以公式是：

next_decode_eligible_step = 当前调度轮次 + pipeline_parallel_size
例子
假设：

pipeline_parallel_size = 4
某个 request 在第 10 轮被调度 decode：

current_step = 10
调度结束后更新：

next_decode_eligible_step = 10 + 4 = 14
之后 scheduler 每轮都会检查：

if self.current_step < request.next_decode_eligible_step:
    continue
所以：

第 11 轮：11 < 14，跳过
第 12 轮：12 < 14，跳过
第 13 轮：13 < 14，跳过
第 14 轮：14 < 14 为 False，可以调度
总结
current_step 是 scheduler 的调度轮数；next_decode_eligible_step 是某个 request 下一次允许 decode 的轮数。每次这个 request 被 decode 调度后，vLLM 会把它更新成：

当前轮数 + pipeline_parallel_size
用来控制同一个 request 不要在 pipeline parallel + async scheduling 下被连续 decode 太快。



为什么需要间隔
因为 pipeline parallel 里，一次 decode 不是一个 stage 立刻全部完成，而是要经过多个 pipeline stage。

如果有 4 个 pipeline stage：

PP stage 0 -> PP stage 1 -> PP stage 2 -> PP stage 3
同一个 request 在 step 10 发出去后，它的结果要沿着 pipeline 往后走。还没走完时，如果 step 11 又把同一个 request 的下一次 decode 塞进去，就可能和 worker 侧的 sampled token 传递节奏对不上。

所以 vLLM 让它等：

pp_size 个 scheduler step
等 pipeline 节奏走完，再调度这个 request 的下一次 decode。


```



### run_busy_loop()

````
**EngineCore 的 busy loop 是一个常驻循环；它根据 scheduler 里有没有未完成请求来决定是否继续 step。没请求时阻塞等 input queue，有请求时不等新请求，直接进入 `_process_engine_step()`，调用 `scheduler.schedule()` 选 batch，再交给 `model_executor` 执行。**

具体是：

```text
run_busy_loop()
  -> _process_input_queue()
  -> _process_engine_step()
```

## 1. 什么时候 step

判断依据是：

```python
self.has_work()
```

里面主要看：

```text
self.scheduler.has_requests()
self.batch_queue
self.engines_running
```

也就是：

```text
只要 scheduler 里还有 waiting/running/未清理的 request，
engine 就会继续 step。
```

如果没有 work：

```text
_process_input_queue()
  -> 阻塞在 input_queue.get()
  -> 等 ADD / ABORT / WAKEUP
```

如果有 work：

```text
_process_input_queue()
  -> 不继续阻塞等新请求
  -> drain 已有 input
  -> 进入 _process_engine_step()
```

_process_input_queue() 会阻塞
当 engine 里没有 work 的时候，它会阻塞等新请求。

代码逻辑是：

while not self.has_work() and self.is_running():
    req = self.input_queue.get(block=block)
    self._handle_client_request(*req)
也就是说：

如果 scheduler 里没有 request
并且 batch_queue 也空
那 EngineCore 会卡在 input_queue.get()
等待 ADD / ABORT / WAKEUP 等消息
这就是 idle 状态。

所以空闲时：

_process_input_queue() 是阻塞的


## 2. `_process_engine_step()` 做什么

它会执行一轮 engine：

```text
scheduler.schedule()
  -> 选这一轮要跑的 requests
  -> 分配 token budget / KV cache
  -> 生成 SchedulerOutput

model_executor.execute_model(SchedulerOutput)
  -> 跑模型 forward / sample

scheduler.update_from_output()
  -> 更新 request 状态
  -> append 新 token
  -> 检查是否 finished
  -> 释放 KV cache
```

## 3. 阻塞关系

合起来看：

```text
没有请求时：
  busy loop 阻塞在 _process_input_queue()
  等新请求进来

有请求时：
  _process_input_queue() 不阻塞
  _process_engine_step() 会推进一轮模型执行
  通常会等本轮 model output 返回
```

所以它不是一直空转烧 CPU，而是：

```text
idle 时阻塞等输入
busy 时循环调度和执行
```

一句话总结：**busy loop 的节奏由 `has_work()` 控制；没活就阻塞等 input queue，有活就一轮轮调用 `scheduler.schedule()` 和 `model_executor.execute_model()`，直到 scheduler 里没有未完成请求。**
````





### 只要 scheduler 里还有没完成的 request，EngineCore 就会反复执行一轮调度 + 一轮模型计算。

````python
主要在 `vllm/v1/engine/core.py`。

`run_busy_loop()` 是外层循环：

```1259:1265:vllm/v1/engine/core.py
    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""
        while self._handle_shutdown():
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # 2) Step the engine core and return the outputs.
            self._process_engine_step()
```

“有没有活”看 `has_work()`：

```1247:1253:vllm/v1/engine/core.py
    def has_work(self) -> bool:
        """Returns true if the engine should be stepped."""
        return (
            self.engines_running
            or self.scheduler.has_requests()
            or bool(self.batch_queue)
        )
```

`_process_input_queue()` 在没活时等输入，有活时退出，进入下一步：

```1269:1286:vllm/v1/engine/core.py
    def _process_input_queue(self):
        """Exits when an engine step needs to be performed."""

        waited = False
        while not self.has_work() and self.is_running():
            # Notify callbacks waiting for engine to become idle.
            self._notify_idle_state_callbacks()
            if self.input_queue.empty():
                # Drain aborts queue; all aborts are also processed via input_queue.
                with self.aborts_queue.mutex:
                    self.aborts_queue.queue.clear()
                if logger.isEnabledFor(DEBUG):
                    logger.debug("EngineCore waiting for work.")
                    waited = True
            block = self.process_input_queue_block
            try:
                req = self.input_queue.get(block=block)
                self._handle_client_request(*req)
```

真正“一轮调度 + 一轮模型计算”在 `_process_engine_step()`：

```1300:1309:vllm/v1/engine/core.py
    def _process_engine_step(self) -> bool:
        """Called only when there are unfinished local requests."""

        # Step the engine core.
        outputs, model_executed = self.step_fn()
        # Put EngineCoreOutputs into the output queue.
        for output in outputs.items() if outputs else ():
            self.output_queue.put_nowait(output)
        # Post-step hook.
        self.post_step(model_executed)
```

`self.step_fn()` 默认就是 `EngineCore.step()`，里面调用 scheduler 和 model executor：

```486:508:vllm/v1/engine/core.py
        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            return {}, False
        scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                model_output = self.model_executor.sample_tokens(grammar_output)

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
```

所以对应关系是：

```text
while self._handle_shutdown():
  -> _process_input_queue()
     没活就等 input_queue
     有活就返回

  -> _process_engine_step()
     -> self.step_fn()
        -> scheduler.schedule()
        -> model_executor.execute_model()
        -> scheduler.update_from_output()
```

这就是“只要 scheduler 里还有没完成 request，就一轮轮调度和执行”的代码位置。
````



### def _handle_shutdown(self) -> bool: 和 while not self.has_work() and self.is_running()

````python
这两个条件不是重复的：`_handle_shutdown()` 管整个 busy loop 的生命周期；`while not self.has_work() and self.is_running()` 管空闲时是否阻塞等输入。
这两个地方控制的是不同层级。

## 1. `while self._handle_shutdown():`

这是 **外层 busy loop 是否继续跑** 的判断。

```python
def run_busy_loop(self):
    while self._handle_shutdown():
        self._process_input_queue()
        self._process_engine_step()

    raise SystemExit
```

`_handle_shutdown()` 的意思是：

```text
EngineCore 是否还应该继续运行？
```

如果当前状态是：

```text
RUNNING
```

它直接返回 `True`，继续 busy loop。

```1324:1327:vllm/v1/engine/core.py
    def _handle_shutdown(self) -> bool:
        # Check if shutdown was requested and handle it
        if self.shutdown_state == EngineShutdownState.RUNNING:
            return True
```

如果收到 shutdown 请求：

```text
REQUESTED
```

它会进入 shutdown 流程：

```text
如果 shutdown_timeout == 0:
  abort 所有未完成请求

否则:
  drain，等已有请求跑完
```

然后状态变成：

```text
SHUTTING_DOWN
```

```1329:1360:vllm/v1/engine/core.py
        if self.shutdown_state == EngineShutdownState.REQUESTED:
            shutdown_timeout = self.vllm_config.shutdown_timeout
            mode = "abort" if shutdown_timeout == 0 else "drain"
...
            if shutdown_timeout == 0:
                num_requests = self.scheduler.get_num_unfinished_requests()
...
                aborted_reqs = self.scheduler.finish_requests(
                    None, RequestStatus.FINISHED_ABORTED
                )
                self._send_abort_outputs(aborted_reqs)
...
            self.shutdown_state = EngineShutdownState.SHUTTING_DOWN
```

最后，如果已经没有 work 了，就返回 `False`，外层 loop 退出：

```1362:1368:vllm/v1/engine/core.py
        # Exit when no work remaining
        if not self.has_work():
            logger.info(
                "[shutdown] EngineCore: request processing complete; "
                "starting resource teardown"
            )
            return False
```

所以：

```text
_handle_shutdown()
```

控制的是：

```text
整个 EngineCore 要不要继续活着
```

## 2. `while not self.has_work() and self.is_running():`

这是 `_process_input_queue()` 里面的 **空闲等待逻辑**。

```1272:1276:vllm/v1/engine/core.py
        waited = False
        while not self.has_work() and self.is_running():
            # Notify callbacks waiting for engine to become idle.
            self._notify_idle_state_callbacks()
            if self.input_queue.empty():
```

意思是：

```text
只要现在没有 work，并且 EngineCore 还处于 RUNNING，
就阻塞等 input_queue 里的新消息。
```

也就是等：

```text
ADD
ABORT
UTILITY
WAKEUP
```

核心是这里：

```1283:1286:vllm/v1/engine/core.py
            block = self.process_input_queue_block
            try:
                req = self.input_queue.get(block=block)
                self._handle_client_request(*req)
```

如果来了 `ADD`，`_handle_client_request()` 会把 request 加进 scheduler。

一旦：

```text
self.has_work() == True
```

比如 scheduler 里有 request 了，这个 inner while 就退出，然后进入 `_process_engine_step()`。

## 两个条件的区别

### 外层

```python
while self._handle_shutdown():
```

回答的是：

```text
EngineCore 还要不要继续运行？
```

### 内层

```python
while not self.has_work() and self.is_running():
```

回答的是：

```text
现在没活干，要不要阻塞等新输入？
```

## 整体流程

```text
run_busy_loop()
  while _handle_shutdown():
    _process_input_queue()
      while not has_work() and is_running():
        input_queue.get()  # 阻塞等新请求

    _process_engine_step()
      scheduler.schedule()
      model_executor.execute_model()
      scheduler.update_from_output()
```

一句话：`_handle_shutdown()` 管 EngineCore 生命周期；`while not has_work() and is_running()` 管空闲时等待请求。前者决定 loop 是否退出，后者决定当前是否阻塞在 input queue。
````



### 具体数值

````python
我用一个具体场景把数值代入，然后按代码执行顺序逐段拆解讲解。

## 假设的输入场景

- `self.current_step = 5`（执行后变成 6）
- `token_budget = self.max_num_scheduled_tokens = 512`
- `self._pause_state = UNPAUSED`（未暂停）
- `self.running = [A, B]`
- `self.waiting` 中只有一个新请求 `C`
- 没有 encoder 输入、没有 LoRA、没有 KV connector、没有投机解码（`num_spec_tokens=0`）、`long_prefill_token_threshold=0`（禁用）、`max_model_len=4096`

请求状态：

| 请求 | 状态 | num_computed_tokens | num_tokens_with_spec / num_tokens | is_prefill_chunk |
|---|---|---|---|---|
| A | running（纯解码） | 100 | 101 | False |
| B | running（分块 prefill 进行中） | 200 | 400 | True |
| C | waiting（新请求，prompt=150） | 0 | 150 | — |

---

## 第 1 步：初始化

```python
self.current_step += 1          # 5 -> 6
token_budget = self.max_num_scheduled_tokens   # 512
```
`_pause_state` 不是 `PAUSED_ALL`，所以 `token_budget` 保持 512（如果暂停，这里会被强制置 0，直接跳过所有调度）。

`defer_prefills` 的计算：
```python
defer_prefills = (throttle_prefills and not self.prefill_capacity_bound) and any(...)
```
假设 `throttle_prefills=False`，所以 `defer_prefills = False`，后面所有依赖它的分支都会被跳过。

---

## 第 2 步：调度 RUNNING 队列 —— 处理请求 A（纯解码）

```python
if request.num_output_placeholders > 0 and ...:
    req_index += 1
    continue
```
A 的 `num_output_placeholders=0`，条件为假，不跳过。

```python
if self.current_step < request.next_decode_eligible_step:
```
假设 A 的 `next_decode_eligible_step=0`，`6 < 0` 为假，不跳过（PP 场景下才会触发这个限制）。

`defer_prefills=False`，`is_prefill_chunk=False`，跳过 defer 分支。

```python
num_new_tokens = request.num_tokens_with_spec + num_output_placeholders - num_computed_tokens
             = 101 + 0 - 100 = 1
```
这里的“1”就是标准的单步解码：只需要计算 1 个新 token。

`long_prefill_token_threshold=0`，跳过截断；`min(1, token_budget=512) = 1`。

```python
num_new_tokens = min(num_new_tokens, max_model_len - num_computed_tokens - num_sampled_tokens_per_step)
             = min(1, 4096 - 100 - 1) = min(1, 3995) = 1
```
无 encoder 输入、无 mamba 对齐，`num_new_tokens=1 != 0`，进入 `allocate_slots`。假设有足够空闲块，第一次调用就成功，`new_blocks is not None`，跳出 while True。

```python
scheduled_running_reqs.append(request)      # [A]
prefill_scheduled |= request.is_prefill_chunk  # False，仍是 False
req_to_new_blocks["A"] = new_blocks
num_scheduled_tokens["A"] = 1
token_budget -= 1     # 512 -> 511
req_index += 1        # 0 -> 1
```
`request.spec_token_ids` 为空，跳过投机解码分支；`has_encoder_inputs=False`，跳过 encoder 分配。

---

## 第 3 步：调度 RUNNING 队列 —— 处理请求 B（分块 prefill）

`req_index=1 < len(running)=2`，`token_budget=511>0`，继续循环。

前两个 `continue` 条件同样都为假（`num_output_placeholders=0`，PP 限制不触发）。

`defer_prefills=False`，所以即使 `B.is_prefill_chunk=True` 也不会被推迟（如果 `defer_prefills=True`，这里就会 `req_index+=1; continue`，把这个 chunk 推到下一个"节奏对齐"的 step）。

```python
num_new_tokens = 400 + 0 - 200 = 200
```
这是 B 这个 prefill 请求本次 chunk 还需要计算的 token 数（剩余 400-200=200 个 prompt token）。

`threshold=0` 不截断；`min(200, token_budget=511) = 200`；`max_model_len` 限制：`min(200, 4096-200-1=3895)=200`，不受影响。

无 encoder、无 mamba split，`num_new_tokens=200≠0`，进入 `allocate_slots` 并假设成功：

```python
scheduled_running_reqs.append(request)        # [A, B]
prefill_scheduled |= True                     # 现在变成 True
req_to_new_blocks["B"] = new_blocks
num_scheduled_tokens["B"] = 200
token_budget -= 200      # 511 -> 311
req_index += 1           # 1 -> 2
```
循环结束（`req_index=2 == len(running)=2`）。

**注意**：如果 `allocate_slots` 在这一步返回 `None`（KV block 不够），就会进入内部的 `while True` 抢占逻辑：按 `self.policy` 找到优先级最低的请求（FCFS 下就是 `self.running.pop()` 弹出最后一个），调用 `self._preempt_request` 并记录到 `preempted_reqs`，然后重试分配，直到成功或把自己也抢占掉（`if preempted_req == request: break`）。在我们的场景里没有触发这段。

---

## 第 4 步：记录 LoRA

```python
if self.lora_config:
    scheduled_loras = {...}
```
`self.lora_config=None`，跳过，`scheduled_loras=set()`。

---

## 第 5 步：调度 WAITING 队列 —— 处理请求 C（新 prefill）

进入条件：`not preempted_reqs`（True，因为没抢占任何请求）且 `_pause_state==UNPAUSED`。

```python
while (self.waiting or self.skipped_waiting) and token_budget > 0:
```
`waiting=[C]`，`token_budget=311>0`，进入。

```python
num_running = len(self.running) + self.num_waiting_for_streaming_input
            = 2 + 0 = 2
if num_running >= self.max_num_running_reqs:   # 2 >= 4? False
    break
```
不触发上限，继续。

取出 `request=C`，不是 blocked 状态，跳过 promote 逻辑；无 LoRA 限制。

**计算已缓存 token 数**（`num_computed_tokens==0` 分支）：
```python
new_computed_blocks, num_new_local_computed_tokens = self.kv_cache_manager.get_computed_blocks(request)
```
假设前缀缓存完全没命中：`num_new_local_computed_tokens = 0`，`new_computed_blocks` 为空。

`self.connector=None`，跳过 external token 查询，`num_external_computed_tokens` 保持 0。

```python
num_computed_tokens = 0 + 0 = 0
```

`self.ec_connector=None`，跳过 mm 预取检查；`request.prefill_stats` 记录 `num_prompt_tokens=150, local_cached=0, external_cached=0`。

**计算本步要调度的 token 数**：
```python
load_kv_async = False   # 没有 connector
```
`defer_prefills=False`，走 else 分支：
```python
num_new_tokens = request.num_tokens - num_computed_tokens = 150 - 0 = 150
```
投机解码 padding 条件里 `self.num_spec_tokens=0`，条件为假，跳过 padding。
`threshold=0`，不截断。
`enable_chunked_prefill` 假设为 True，跳过“未启用分块预填充直接 break”的分支。
```python
num_new_tokens = min(150, token_budget=311) = 150
```
无 encoder 输入，跳过 `_try_schedule_encoder_inputs`。

无 mamba split。

```python
limit_lookahead_tokens = load_kv_async and num_lookahead_tokens>0   # False
effective_lookahead_tokens = self.num_lookahead_tokens              # 假设 0
```
非 encoder-decoder 模型，`num_encoder_tokens=0`；`load_kv_async=False`，所以 `reserved_blocks=0`。

调用：
```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request, 150,
    num_new_computed_tokens=0,
    new_computed_blocks=<empty>,
    num_lookahead_tokens=0,
    num_external_computed_tokens=0,
    delay_cache_blocks=False,
    num_encoder_tokens=0,
    ...
)
```
假设成功分配。

```python
request = request_queue.pop_request()   # C 从 waiting 队列移除
```
`load_kv_async=False`，跳过异步 KV 加载分支，走正常路径：
```python
self.running.append(request)     # running = [A, B, C]
```
`request.status == WAITING` → `scheduled_new_reqs = [C]`（而不是 `scheduled_resumed_reqs`，那是给 `PREEMPTED` 状态复用的）。

```python
req_to_new_blocks["C"] = self.kv_cache_manager.get_blocks("C")
num_scheduled_tokens["C"] = 150
token_budget -= 150       # 311 -> 161
request.status = RequestStatus.RUNNING
request.num_computed_tokens = 0
```
`pad_spec_decode=False`，跳过。

```python
if num_computed_tokens + num_new_tokens < request.num_tokens:   # 0+150 < 150? False
    self._inflight_prefills.add(request)
```
因为 150 不小于 150，这次调度**一次性吃完了 C 的整个 prompt**，所以它不会被标记为"还在 prefill 中"的 inflight 请求。

循环再次检查 `while` 条件：`self.waiting` 现在为空、`self.skipped_waiting` 也是空 → 退出循环。

`step_skipped_waiting` 为空，跳过 `prepend_requests`。
`defer_prefills=False` → `self.prefill_capacity_bound = bool(self.waiting) = bool([]) = False`（表示这一步没有因为容量不足而卡住等待队列，为下一次 DP 均衡决策提供依据）。

---

## 第 6 步：调度后的一致性检查

```python
total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
                            = 1 + 200 + 150 = 351
assert 351 <= 512            # ✓ (max_num_scheduled_tokens)
assert token_budget >= 0     # 161 >= 0 ✓
assert len(self.running) <= self.max_num_running_reqs   # 3 <= 4 ✓
assert len(scheduled_new_reqs)+len(scheduled_resumed_reqs)+len(scheduled_running_reqs) <= len(self.running)
     # 1 + 0 + 2 = 3 <= 3 ✓
```
全部通过，说明我们编的这组数字是自洽的（`512 - 351 = 161`，正好对上最终的 `token_budget`）。

---

## 第 7 步：计算公共前缀块 & 构造 SchedulerOutput

```python
if self.running:   # [A, B, C]，非空
    num_common_prefix_blocks = self.kv_cache_manager.get_num_common_prefix_blocks(A.request_id)
```
这是为 cascade attention 服务的，取所有 running 请求共享的最长公共前缀块数（具体值依赖 KV cache 管理器内部状态，这里不展开数值，仅说明是拿 `self.running[0]` 作为代表去查询）。

`use_v2_model_runner` 假设为 False，走：
```python
new_reqs_data = [NewRequestData.from_request(C, blocks_C)]
```
（`scheduled_new_reqs` 里只有 C）。

```python
cached_reqs_data = self._make_cached_request_data(
    scheduled_running_reqs=[A, B],
    scheduled_resumed_reqs=[],
    num_scheduled_tokens={"A":1, "B":200, "C":150},
    scheduled_spec_decode_tokens={},
    req_to_new_blocks=...,
)
```
这里打包的是"已经在 running 里、继续跑"的请求（A、B）的增量信息——本步新算的 token 数、新分配的 KV block 等，供 worker 侧做 attention 计算。

```python
num_spec_tokens_to_schedule = self.num_spec_tokens = 0
```
因为 `dynamic_sd_lookup=None`，不走动态投机解码查表逻辑。

最终构造：
```python
scheduler_output = SchedulerOutput(
    scheduled_new_reqs=[C的NewRequestData],
    scheduled_cached_reqs=[A, B 的增量数据],
    num_scheduled_tokens={"A":1, "B":200, "C":150},
    total_num_scheduled_tokens=351,
    scheduled_spec_decode_tokens={},
    scheduled_encoder_inputs={},
    num_common_prefix_blocks=[...],
    preempted_req_ids=self.reset_preempted_req_ids,   # 本例为空集
    finished_req_ids=self.finished_req_ids,           # 本例为空集
    free_encoder_mm_hashes=[],
    new_block_ids_to_zero=None,   # 假设 needs_kv_cache_zeroing=False
    num_spec_tokens_to_schedule=0,
)
```

## 第 8 步：Connector 元数据与收尾

`self.connector=None`、`self.ec_connector=None`，两段 connector 元数据构建都跳过。

```python
if self.defer_block_free and total_num_scheduled_tokens > 0:
    self.sched_step_seq += 1
```
假设 `defer_block_free=False`，跳过。

最后：
```python
self._update_after_schedule(scheduler_output)
return scheduler_output
```
调用完成后，把这个包含 A（1 token 解码）、B（200 token prefill chunk）、C（150 token 完整 prefill）三个请求信息的 `SchedulerOutput` 返回给上层（通常是 executor/worker，用来真正跑一次前向计算）。

---

### 小结（这一步调度的整体效果）
- **A**：纯解码，多算 1 个 token。
- **B**：分块 prefill 继续推进 200 个 token（还没吃完 400 个 prompt token，会在 `_inflight_prefills` 之外 —— 因为它是走 running 循环里的分支，并没有显式判断是否加入 `_inflight_prefills`，这个字段的加入逻辑只在 waiting 循环里出现；B 本身是从上一步就已经在 running 里，说明它此前已经被加入过）。
- **C**：新请求一次性吃完 150 个 prompt token，直接完成 prefill 进入纯解码状态，下一步就会走"A 那种"的 1-token 解码路径。
- 本步共用了 351/512 的 token 预算，剩余 161 没有用满，因为 waiting 队列已经空了，没有更多请求可调度。

如果你想我换一组数字（比如加入投机解码、KV connector 异步加载、或者触发一次抢占）来讲对应的分支，告诉我具体想覆盖哪部分，我可以针对性地再走一遍。
````





## Request是怎么ADD的

````python
`ADD` 是 frontend 发给 EngineCore 的一种消息类型：

```python
EngineCoreRequestType.ADD
```

定义在 `vllm/v1/engine/__init__.py`：

```251:258:vllm/v1/engine/__init__.py
class EngineCoreRequestType(enum.Enum):
    """
    Request types defined as hex byte strings, so it can be sent over sockets
    without separate encoding step.
    """

    ADD = b"\x00"
    ABORT = b"\x01"
```

## 1. AsyncLLM 里发 ADD

在 `AsyncLLM._add_request()` 里：

```400:412:vllm/v1/engine/async_llm.py
    async def _add_request(
        self,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None,
        index: int,
        queue: RequestOutputCollector,
    ):
        # Add the request to OutputProcessor (this process).
        self.output_processor.add_request(request, prompt, parent_req, index, queue)

        # Add the EngineCoreRequest to EngineCore (separate process).
        await self.engine_core.add_request_async(request)
```

这里的：

```python
await self.engine_core.add_request_async(request)
```

会进入 `AsyncMPClient.add_request_async()`。

## 2. AsyncMPClient 把 request 包成 ADD 消息

在 `vllm/v1/engine/core_client.py`：

```1121:1124:vllm/v1/engine/core_client.py
    async def add_request_async(self, request: EngineCoreRequest) -> None:
        request.client_index = self.client_index
        await self._send_input(EngineCoreRequestType.ADD, request)
        self._ensure_output_queue_task()
```

关键是：

```python
self._send_input(EngineCoreRequestType.ADD, request)
```

## 3. _send_input 编码消息

```1064:1074:vllm/v1/engine/core_client.py
    def _send_input(
        self,
        request_type: EngineCoreRequestType,
        request: Any,
        engine: EngineIdentity | None = None,
    ) -> Awaitable[Any]:
        if engine is None:
            engine = self.core_engine

        message = (request_type.value, *self.encoder.encode(request))
        return self._send_input_message(message, engine, request)
```

这里构造出来的是：

```text
(request_type.value, *encoded_request)
```

对于 ADD，就是：

```text
(b"\x00", *msgpack(EngineCoreRequest))
```

然后 `_send_input_message()` 会把 engine identity 加到第一帧，通过 ZMQ 发出去：

```1086:1089:vllm/v1/engine/core_client.py
        msg = (engine,) + message
        if not objects or len(msg) <= 3:
            # No auxiliary buffers => no tensor backing buffers in request.
            return self.input_socket.send_multipart(msg, copy=False)
```

所以 ZMQ multipart 大概是：

```text
(engine_identity, ADD, encoded EngineCoreRequest...)
```

## 4. EngineCoreProc 收到 ADD

在 `vllm/v1/engine/core.py` 的 `process_input_sockets()`：

```1557:1575:vllm/v1/engine/core.py
                for input_socket, _ in poller.poll():
                    # (RequestType, RequestData)
                    type_frame, *data_frames = input_socket.recv_multipart(copy=False)
...
                    request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                    # Deserialize the request data.
                    request: Any
                    if request_type == EngineCoreRequestType.ADD:
                        req: EngineCoreRequest = add_request_decoder.decode(data_frames)
                        try:
                            request = self.preprocess_add_request(req)
                        except Exception:
                            self._handle_request_preproc_error(req)
                            continue
```

这里：

```text
type_frame = b"\x00"
```

所以：

```python
request_type == EngineCoreRequestType.ADD
```

然后解码 `EngineCoreRequest`，再做：

```python
request = self.preprocess_add_request(req)
```

这个会把：

```text
EngineCoreRequest
```

转成：

```text
(Request, request_wave)
```

## 5. 放进 input_queue

同一个方法最后：

```1586:1587:vllm/v1/engine/core.py
                    # Push to input queue for core busy loop.
                    self.input_queue.put_nowait((request_type, request))
```

所以 input queue 里放的是：

```text
(EngineCoreRequestType.ADD, (Request, request_wave))
```

## 6. busy loop 取出 ADD

`_process_input_queue()` 里：

```1283:1286:vllm/v1/engine/core.py
            block = self.process_input_queue_block
            try:
                req = self.input_queue.get(block=block)
                self._handle_client_request(*req)
```

也就是取出：

```text
request_type = ADD
request = (Request, request_wave)
```

然后交给 `_handle_client_request()`。

## 7. _handle_client_request 处理 ADD

```1377:1383:vllm/v1/engine/core.py
        if request_type == EngineCoreRequestType.WAKEUP:
            return
        elif request_type == EngineCoreRequestType.ADD:
            req, request_wave = request
            if self._reject_add_in_shutdown(req):
                return
            self.add_request(req, request_wave)
```

最后：

```text
self.add_request(req, request_wave)
  -> self.scheduler.add_request(request)
```

## 总结路径

```text
AsyncLLM._add_request()
  -> AsyncMPClient.add_request_async()
  -> _send_input(EngineCoreRequestType.ADD, request)
  -> ZMQ send_multipart(engine, ADD, encoded_request)
  -> EngineCoreProc.process_input_sockets()
  -> decode EngineCoreRequest
  -> preprocess_add_request()
  -> input_queue.put_nowait((ADD, (Request, wave)))
  -> _process_input_queue()
  -> _handle_client_request(ADD)
  -> EngineCore.add_request()
  -> Scheduler.add_request()
```
````



## request_wave是？block = self.process_input_queue_block是？

````python
这两个是不同用途。

## 1. `request_wave` 是什么

`request_wave` 主要给 **Data Parallel / DP coordinator** 用，普通单机单 engine 情况基本可以理解成默认 `0`。

它来自 `EngineCoreRequest.current_wave`：

```115:118:vllm/v1/engine/__init__.py
    # Used in DP case to indicate which wave of requests this is expected to
    # belong to, to cover a race condition where the request is sent before
    # a wave finished notification is received.
    current_wave: int = 0
```

在 `preprocess_add_request()` 里返回：

```855:877:vllm/v1/engine/core.py
    def preprocess_add_request(self, request: EngineCoreRequest) -> tuple[Request, int]:
...
        req = Request.from_engine_core_request(request, self.request_block_hasher)
...
        return req, request.current_wave
```

所以 ZMQ 收到 ADD 后，放进 `input_queue` 的其实是：

```text
(ADD, (Request, request_wave))
```

然后 `_handle_client_request()` 里取出来：

```1379:1383:vllm/v1/engine/core.py
        elif request_type == EngineCoreRequestType.ADD:
            req, request_wave = request
            if self._reject_add_in_shutdown(req):
                return
            self.add_request(req, request_wave)
```

### wave 是干嘛的

在 DP 多 engine 情况下，所有 DP rank 会一起一波一波运行。`wave` 可以理解成：

```text
当前这一波 DP 调度/运行的编号
```

它用来处理一种竞态：

```text
frontend 发了新 request
但同时 engine 可能刚刚认为上一波请求结束，准备 pause
```

所以 request 带着 `current_wave`，EngineCore 收到后知道这个 request 属于哪一波。

DP 里有特殊处理：

```1836:1849:vllm/v1/engine/core.py
    def add_request(self, request: Request, request_wave: int = 0):
        super().add_request(request, request_wave)
        if self.has_coordinator and request_wave != self.current_wave:
            if request_wave > self.current_wave:
                self.current_wave = request_wave
            elif (
                not self.engines_running
                and self.scheduler.pause_state == PauseState.UNPAUSED
            ):
                # Request received for an already-completed wave, notify
                # front-end that we need to start the next one.
                self.engines_running = True
                self.output_queue.put_nowait(
                    (-1, EngineCoreOutputs(start_wave=self.current_wave))
                )
```

一句话：`request_wave` 是 DP 场景下的“请求属于第几波运行”的标记，用来同步/唤醒 DP engines；非 DP 下基本就是 `0`。

为什么要唤醒？怎么唤醒的？是什么场景使用的？




## 2. `block = self.process_input_queue_block` 是什么

这个控制：

```python
self.input_queue.get(block=block)
```

到底是 **阻塞等消息**，还是 **不阻塞地试一下**。

代码：

```1283:1286:vllm/v1/engine/core.py
            block = self.process_input_queue_block
            try:
                req = self.input_queue.get(block=block)
                self._handle_client_request(*req)
```

如果：

```text
block = True
```

那么：

```text
input_queue.get(block=True)
```

会阻塞等待新消息。

如果：

```text
block = False
```

那么：

```text
input_queue.get(block=False)
```

如果 queue 为空，会立刻抛 `queue.Empty`，然后 break，不会卡住。

默认是 `True`：

```957:958:vllm/v1/engine/core.py
            self.addresses = addresses
            self.process_input_queue_block = True
```

## 为什么需要这个开关

正常情况：

```text
process_input_queue_block = True
```

没请求时 EngineCore 就阻塞等 input queue，避免空转。

但某些特殊场景，比如 Elastic EP scaling/reconfiguration，需要 EngineCore 不要卡死在 `input_queue.get()`，而是继续循环推进状态机，所以会设成 `False`：

```2038:2047:vllm/v1/engine/core.py
        self.eep_scaling_state = ElasticEPScalingState(
            model_executor=self.model_executor,
            engine_core=self,
            vllm_config=self.vllm_config,
            new_parallel_config=new_parallel_config,
            worker_type="removing" if is_shutdown else "existing",
            scale_type="scale_down" if is_scale_down else "scale_up",
            reconfig_request=reconfig_request,
        )
        self.process_input_queue_block = False
```

等 scaling 完成后再设回 `True`：

```1935:1941:vllm/v1/engine/core.py
            if self.eep_scaling_state is not None:
                _ = self.eep_scaling_state.progress()
                if self.eep_scaling_state.is_complete():
                    if self.eep_scaling_state.worker_type == "removing":
                        raise SystemExit
                    self.process_input_queue_block = True
                    self.eep_scaling_state = None
```

## 总结

```text
request_wave:
  DP 多 engine 用的 wave 编号，标记 request 属于哪一波运行；
  普通非 DP 基本是 0。

process_input_queue_block:
  控制 input_queue.get() 是否阻塞；
  True = 没消息就等；
  False = 没消息就立刻返回，让 busy loop 继续推进别的状态。
```
````



## 为什么要唤醒？怎么唤醒的？是什么场景使用的?

````python
这里的“唤醒”有两层意思，要分开看。

## 1. 普通唤醒：让阻塞的 EngineCore 从 `input_queue.get()` 返回

EngineCore 没有 work 时，会阻塞在这里：

```python
req = self.input_queue.get(block=block)
```

如果此时有新请求 ADD 进来，IO thread 会做：

```1586:1587:vllm/v1/engine/core.py
                    # Push to input queue for core busy loop.
                    self.input_queue.put_nowait((request_type, request))
```

`put_nowait()` 放入 queue 后，阻塞在 `get()` 的 busy loop 就会醒过来。

然后走：

```text
_process_input_queue()
  -> _handle_client_request(ADD)
  -> add_request()
  -> scheduler.add_request()
```

所以普通场景下：

```text
新 ADD 消息本身就能唤醒 EngineCore
```

## 2. DP 场景唤醒：唤醒所有 DP engines 进入同一波运行

这个是 `request_wave` 相关的“唤醒”。

在 data parallel 场景下，可能有多个 EngineCore：

```text
EngineCore rank 0
EngineCore rank 1
EngineCore rank 2
...
```

如果只有某一个 rank 收到了新 request，其他 rank 可能还是 idle/paused。

但 DP/MoE 场景里，多个 rank 往往需要一起进入 loop，做：

```text
dummy batch
all-reduce
global unfinished request 判断
EP/DP 同步
```

所以不能只唤醒一个 rank，需要通过 coordinator 启动一波新的 wave。

## 3. 怎么唤醒 DP engines

frontend 发送第一个请求时，如果发现 engines 当前没在跑：

```1362:1370:vllm/v1/engine/core_client.py
        request.current_wave = self.current_wave
        request.client_index = self.client_index

        chosen_engine = self.get_core_engine_for_request(request)
        to_await = self._send_input(EngineCoreRequestType.ADD, request, chosen_engine)
        if not self.engines_running:
            # Notify coordinator that we're sending a request
            req_msg = msgspec.msgpack.encode(("FIRST_REQ", chosen_engine))
            await self.first_req_send_socket.send(req_msg)
```

这里做了两件事：

```text
1. 把 ADD 发给 chosen_engine
2. 如果 engines_running == False，额外发送 FIRST_REQ 通知
```

`FIRST_REQ` 会让 coordinator 广播：

```text
START_DP_WAVE
```

给所有 engines。

EngineCore 收到 `START_DP_WAVE` 后：

```1883:1900:vllm/v1/engine/core.py
    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        if request_type == EngineCoreRequestType.START_DP_WAVE:
            if self.ignore_start_dp_wave:
                return
            new_wave, exclude_eng_index = request
            if exclude_eng_index != self.engine_index and (
                new_wave >= self.current_wave
            ):
                self.current_wave = new_wave
                if not self.engines_running:
                    logger.debug(
                        "EngineCore starting idle loop for wave %d.",
                        new_wave,
                    )
                    self.engines_running = True
```

关键是：

```python
self.engines_running = True
```

一旦这个变成 True，`has_work()` 就会变 True：

```1247:1253:vllm/v1/engine/core.py
    def has_work(self) -> bool:
        """Returns true if the engine should be stepped."""
        return (
            self.engines_running
            or self.scheduler.has_requests()
            or bool(self.batch_queue)
        )
```

于是 idle loop 退出，EngineCore 开始 `_process_engine_step()`。

## 4. 为什么需要唤醒

因为 DP engines 可能已经完成上一波请求并 pause 了。

这时新请求来了：

```text
只有一个 engine 收到 ADD
其他 engine 没有 request
```

但 DP/MoE 需要多个 engine 一起参与某些同步逻辑。如果其他 engine 不醒：

```text
有的 rank 在跑
有的 rank 睡着
collective / all-reduce / dummy batch 可能对不上
```

所以要通过 wave 机制唤醒所有相关 engines：

```text
FIRST_REQ
  -> coordinator
  -> START_DP_WAVE
  -> all DP engines
  -> engines_running = True
```

## 5. 什么场景使用

主要是：

```text
data_parallel_size > 1
DP coordinator
MoE / EP / internal load balancing
engine 已经 idle，新的第一条请求到来
```

普通单 engine / 非 DP 场景不需要这个 wave 唤醒。

普通场景是：

```text
ADD 放进 input_queue
-> input_queue.get() 返回
-> EngineCore 被唤醒
```

DP 场景额外需要：

```text
唤醒其他没有收到 ADD 的 DP engines
```

一句话：普通唤醒是用 `input_queue.put()` 唤醒阻塞的 EngineCore；DP 场景还要用 `FIRST_REQ -> START_DP_WAVE` 唤醒其他 DP engines，让所有 rank 进入同一波运行。
````





## 前端先通过 input socket 发到 EngineCore 的 `input_queue`。

````python
你记得的是对的：**前端确实先通过 input socket 发到 EngineCore 的 `input_queue`。**

但这里有两个不同的 queue：

```text
1. EngineCore.input_queue
   socket IO -> busy loop 的消息队列

2. Scheduler.waiting / skipped_waiting / running
   scheduler 真正调度 request 的队列
```

它们不是同一个东西。

## 路径是两段

### 第一段：ZMQ input socket -> EngineCore.input_queue

frontend 通过 ZMQ 发 ADD。

EngineCore 的 socket IO thread 收到后：

```1557:1587:vllm/v1/engine/core.py
                for input_socket, _ in poller.poll():
                    # (RequestType, RequestData)
                    type_frame, *data_frames = input_socket.recv_multipart(copy=False)
...
                    if request_type == EngineCoreRequestType.ADD:
                        req: EngineCoreRequest = add_request_decoder.decode(data_frames)
                        try:
                            request = self.preprocess_add_request(req)
                        except Exception:
                            self._handle_request_preproc_error(req)
                            continue
...
                    # Push to input queue for core busy loop.
                    self.input_queue.put_nowait((request_type, request))
```

这里放进的是：

```text
EngineCore.input_queue
```

内容是：

```text
(ADD, (Request, request_wave))
```

这个 queue 只是 **EngineCore 内部的消息转发队列**。

## 第二段：EngineCore.input_queue -> Scheduler queue

busy loop 再从 `input_queue` 取出来：

```1283:1286:vllm/v1/engine/core.py
            block = self.process_input_queue_block
            try:
                req = self.input_queue.get(block=block)
                self._handle_client_request(*req)
```

然后 `_handle_client_request(ADD)`：

```1379:1383:vllm/v1/engine/core.py
        elif request_type == EngineCoreRequestType.ADD:
            req, request_wave = request
            if self._reject_add_in_shutdown(req):
                return
            self.add_request(req, request_wave)
```

再到：

```403:403:vllm/v1/engine/core.py
        self.scheduler.add_request(request)
```

这一步才是真正进入 scheduler：

```text
scheduler.waiting
scheduler.skipped_waiting
scheduler.requests
```

## 为什么不能只放 input_queue 就结束？

因为 `input_queue` 只是 mailbox，不参与调度。

它里面可能有各种控制消息：

```text
ADD
ABORT
UTILITY
WAKEUP
START_DP_WAVE
EXECUTOR_FAILED
```

scheduler 只认识调度用的 `Request`，并且需要把 request 放进自己的状态结构：

```text
self.requests[request_id] = request
self.waiting.add_request(request)
```

所以必须有：

```text
input_queue
  -> _handle_client_request()
  -> scheduler.add_request()
```

## 为什么不在 socket thread 里直接 scheduler.add_request？

主要是为了让 scheduler 状态由 EngineCore 主 loop 统一修改。

socket IO thread 只负责：

```text
收 ZMQ
decode
preprocess
放 input_queue
```

EngineCore busy loop 负责：

```text
处理 ADD/ABORT
修改 scheduler 状态
调用 schedule()
调用 model_executor
更新 request
```

这样可以避免 socket thread 和 scheduler/model 执行线程同时改 scheduler 状态。

一句话：ZMQ input socket 发到的是 EngineCore 的 `input_queue`，这是消息队列；`scheduler.add_request()` 是把这个消息真正转成 scheduler 内部的 waiting request，之后才能被 `scheduler.schedule()` 选中执行。
````



##  scheduler如何取request

````python
scheduler 是在 `Scheduler.schedule()` 里取 request 的，主要从两个地方取：

```text
self.running
self.waiting / self.skipped_waiting
```

## 1. 先处理 running requests

代码在 `vllm/v1/core/sched/scheduler.py`：

```440:443:vllm/v1/core/sched/scheduler.py
        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]
```

这里不是 pop，而是按 index 遍历：

```python
request = self.running[req_index]
```

因为 running 里的请求已经被接纳过了，只是每轮继续 decode / prefill chunk。

## 2. 再处理 waiting requests

running 处理完后，scheduler 再看 waiting 队列：

```636:650:vllm/v1/core/sched/scheduler.py
        # Next, schedule the WAITING requests.
        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
            step_skipped_waiting = create_request_queue(self.policy)

            while (self.waiting or self.skipped_waiting) and token_budget > 0:
                # Paused streaming sessions (WAITING_FOR_STREAMING_REQ) are not
                # in `running` but still hold a model-runner request slot.
                num_running = len(self.running) + self.num_waiting_for_streaming_input
                if num_running >= self.max_num_running_reqs:
                    break

                request_queue = self._select_waiting_queue_for_scheduling()
                assert request_queue is not None

                request = request_queue.peek_request()
```

这里先用：

```python
request = request_queue.peek_request()
```

只是看队头 request，先不拿走。

因为 scheduler 还要检查：

```text
KV cache 够不够
token budget 够不够
LoRA 限制
prefix cache
remote KV
encoder/multimodal budget
pause state
```

确认能调度后，才真正 pop：

```948:971:vllm/v1/core/sched/scheduler.py
                request = request_queue.pop_request()
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    step_skipped_waiting.prepend_request(request)
...
                self.running.append(request)
```

也就是：

```text
peek_request()
  -> 检查能不能调度
  -> pop_request()
  -> append 到 running
```

## 3. waiting 队列怎么选

`waiting` 和 `skipped_waiting` 通过这个方法选：

```1873:1883:vllm/v1/core/sched/scheduler.py
    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        if self.policy == SchedulingPolicy.FCFS:
            return self.skipped_waiting or self.waiting or None

        # PRIORITY mode: compare queue heads when both queues are non-empty.
        if self.waiting and self.skipped_waiting:
            waiting_req = self.waiting.peek_request()
            skipped_req = self.skipped_waiting.peek_request()
            return self.waiting if waiting_req < skipped_req else self.skipped_waiting

        return self.waiting or self.skipped_waiting or None
```

默认 `fcfs` 下：

```text
优先 skipped_waiting
否则 waiting
```

priority 下：

```text
比较两个队头 request 的 priority/arrival_time
选更优先的那个
```

## 总结

scheduler 取 request 的顺序是：

```text
Scheduler.schedule()

1. 先遍历 self.running
   request = self.running[req_index]

2. 再从 waiting/skipped_waiting 选队列
   request_queue = _select_waiting_queue_for_scheduling()

3. 先 peek
   request = request_queue.peek_request()

4. 检查资源和约束

5. 能调度才 pop
   request = request_queue.pop_request()

6. 放入 running
   self.running.append(request)
```


frontend 发 ADD
  -> EngineCore.input_queue
  -> _process_input_queue()
  -> _handle_client_request(ADD)
  -> EngineCore.add_request()
  -> Scheduler.add_request()
  -> request 进入 waiting queue

然后下一步：
  -> _process_engine_step()
  -> Scheduler.schedule()
  -> 从 waiting/running 里取 request 执行
    
对应 busy loop：

while self._handle_shutdown():
    self._process_input_queue()
    self._process_engine_step()
````





## waiting -> running & running -> model_executor在哪

````python
`waiting -> running` 和 `running -> model_executor` 不是同一行代码完成的，分两步。

## 1. `waiting -> running` 在 `Scheduler.schedule()`

位置：`vllm/v1/core/sched/scheduler.py`

scheduler 先从 waiting queue 真正 pop 出 request：

```948:948:vllm/v1/core/sched/scheduler.py
                request = request_queue.pop_request()
```

如果不是 async KV load，就放进 `self.running`：

```971:990:vllm/v1/core/sched/scheduler.py
                self.running.append(request)
                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
                    request_id
                )
                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
```

所以这里就是：

```text
waiting queue
  -> pop_request()
  -> self.running.append(request)
  -> request.status = RUNNING
```

## 2. `running -> model_executor` 不是直接传 `self.running`

注意：`model_executor` 不直接拿 `self.running` list。

scheduler 会把本轮选中的 running/new requests 打包成：

```text
SchedulerOutput
```

代码：

```1095:1100:vllm/v1/core/sched/scheduler.py
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
```

这里面包含：

```text
scheduled_new_reqs        新进入 running 的 request
scheduled_cached_reqs     已经在 running 里的 request
num_scheduled_tokens      每个 request 本轮算几个 token
KV block ids              每个 request 用哪些 KV blocks
```

然后 `EngineCore.step()` 把这个 `scheduler_output` 交给 model executor：

```490:491:vllm/v1/engine/core.py
        scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
```

所以真正的路径是：

```text
self.running
  -> Scheduler.schedule()
  -> SchedulerOutput
  -> model_executor.execute_model(SchedulerOutput)
```

## 3. model runner 怎么用 SchedulerOutput

在 V2 GPU model runner 里：

```1122:1128:vllm/v1/worker/gpu/model_runner.py
        if not dummy_run:
            # Update the request states.
            self.update_pp_decode_requests()
            self.finish_requests(scheduler_output)
            self.free_states(scheduler_output)
            self.add_requests(scheduler_output)
            self.update_requests(scheduler_output)
```

其中新 request 会通过 `add_requests()` 加到 model runner 自己的 batch state：

```767:786:vllm/v1/worker/gpu/model_runner.py
    def add_requests(self, scheduler_output: SchedulerOutput) -> None:
        for new_req_data in scheduler_output.scheduled_new_reqs:
            assert new_req_data.prompt_token_ids is not None
            assert new_req_data.prefill_token_ids is not None
            req_id = new_req_data.req_id

            # Streaming input update: request already exists from a prior
            # chunk. Remove old state so it can be cleanly re-added below
            # with the updated prompt_token_ids and mm_features.
            self._remove_request(req_id)

            prompt_len = len(new_req_data.prompt_token_ids)
            sampling_params = new_req_data.sampling_params
            self.req_states.add_request(
                req_id=req_id,
                prompt_len=prompt_len,
                all_token_ids=new_req_data.prefill_token_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
```

已有 running request 会通过 `update_requests()` 更新：

```814:824:vllm/v1/worker/gpu/model_runner.py
    def update_requests(self, scheduler_output: SchedulerOutput) -> None:
        # Add new blocks and update num_computed_tokens for the existing requests.
        reqs = scheduler_output.scheduled_cached_reqs
        num_computed_tokens_np = self.req_states.num_computed_tokens_np
        for req_id, num_computed_tokens, req_new_block_ids in zip(
            reqs.req_ids, reqs.num_computed_tokens, reqs.new_block_ids
        ):
            req_index = self.req_states.req_id_to_index[req_id]
            num_computed_tokens_np[req_index] = num_computed_tokens
            if req_new_block_ids is not None:
```

## 总结

```text
waiting -> running:
  scheduler.py
  request_queue.pop_request()
  self.running.append(request)
  request.status = RUNNING

running -> model_executor:
  scheduler.py
  SchedulerOutput(...)

  core.py
  model_executor.execute_model(scheduler_output)

  gpu/model_runner.py
  add_requests(scheduler_output)
  update_requests(scheduler_output)
```
````







## 如何KV load的?

````python
拉 KV 分两半：

```text
scheduler 侧：决定“要不要拉、拉多少、拉到哪些 KV blocks”
worker 侧：真正执行“从 LMCache/Mooncake/NIXL 拉 KV 到本地 GPU KV cache”
```

## 1. scheduler 先判断远端有没有 KV

在 `Scheduler.schedule()` 里：

```736:742:vllm/v1/core/sched/scheduler.py
                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens
                            )
                        )
```

这里 connector 返回：

```text
ext_tokens        远端能命中的 token 数
load_kv_async     是否需要异步拉 KV
```

接口定义在 `base.py`：

```454:477:vllm/distributed/kv_transfer/kv_connector/v1/base.py
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.
...
                - `True` if external KV cache tokens will be loaded
                  asynchronously (between scheduler steps). Must be
                  'False' if the first element is 0.
```

## 2. scheduler 分配本地 KV blocks

如果要拉 KV，scheduler 先给这个 request 分配本地 KV cache 位置：

```905:913:vllm/v1/core/sched/scheduler.py
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    num_external_computed_tokens=num_external_computed_tokens,
                    delay_cache_blocks=load_kv_async,
```

这里的意思是：

```text
给远端 KV 准备好本地目标 block
```

## 3. scheduler 通知 connector：这些 block 要用来接 KV

```932:937:vllm/v1/core/sched/scheduler.py
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        self.kv_cache_manager.get_blocks(request_id),
                        num_external_computed_tokens,
                    )
```

接口说明：

```489:510:vllm/distributed/kv_transfer/kv_connector/v1/base.py
    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.
...
            num_external_tokens (int): the number of tokens to load from the
            external KV cache. 0 means nothing should be loaded.
```

## 4. scheduler 把 connector metadata 塞进 SchedulerOutput

后面 scheduler 构造 `SchedulerOutput`，并加上 connector metadata：

```1118:1120:vllm/v1/core/sched/scheduler.py
        if self.connector is not None:
            meta = self._build_kv_connector_meta(self.connector, scheduler_output)
            scheduler_output.kv_connector_metadata = meta
```

这个 metadata 会随着：

```text
scheduler_output
```

一起传给 worker。

## 5. EngineCore 把 SchedulerOutput 交给 model_executor

```490:491:vllm/v1/engine/core.py
        scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
```

从这里开始进入 worker/model runner 侧。

## 6. worker 侧真正开始拉 KV

V2 worker 的 KV connector 入口在：

```61:75:vllm/v1/worker/gpu/kv_connector.py
    def pre_forward(self, scheduler_output: "SchedulerOutput") -> None:
        if self._disabled:
            return

        kv_connector_metadata = scheduler_output.kv_connector_metadata
        assert kv_connector_metadata is not None
        self.kv_connector.handle_preemptions(kv_connector_metadata)
        self.kv_connector.bind_connector_metadata(kv_connector_metadata)

        # TODO: sort out KV Connectors' use of forward_context
        if is_forward_context_available():
            self.kv_connector.start_load_kv(get_forward_context())
        else:
            with set_forward_context(None, self.vllm_config):
                self.kv_connector.start_load_kv(get_forward_context())
```

关键就是：

```python
self.kv_connector.start_load_kv(...)
```

这一步才是真正发起拉 KV。

## 7. 不同 backend 的实现不同

### LMCache

```136:151:vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.
...
        """
        self._lmcache_engine.start_load_kv(forward_context, **kwargs)
```

也就是交给 LMCache engine 拉。

### Mooncake

```2028:2032:vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py
    def start_load_kv(self, metadata: MooncakeConnectorMetadata):
        if not self.is_kv_producer and metadata.reqs_to_recv:
            asyncio.run_coroutine_threadsafe(
                self._start_load_kv(metadata.reqs_to_recv), self.receiver_loop
            )
```

Mooncake 是把接收任务丢到 receiver loop 里异步执行。

### NIXL

```343:347:vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self.connector_worker, NixlPullConnectorWorker)
        assert isinstance(self._connector_metadata, NixlConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)
```

NIXL 交给 NIXL worker 去拉。

## 8. attention 层使用 KV 前会等对应 layer load 完成

有些 connector 是 layer-wise 的，attention layer 入口会等：

```50:57:vllm/model_executor/layers/attention/kv_transfer_utils.py
        # Wait for KV layer on entry
        connector.wait_for_layer_load(layer_name)

        # Execute the function
        result = func(*args, **kwargs)

        # Save KV cache layer on exit
        connector.save_kv_layer(layer_name, kv_cache, attn_metadata)
```

所以流程是：

```text
start_load_kv() 先异步开始拉
attention layer 真要用某层 KV 时
  -> wait_for_layer_load(layer_name)
  -> 确保这一层 KV 已经到本地
```

## 9. 拉完后怎么通知 scheduler

worker forward 后会调用 `post_forward()`：

```83:95:vllm/v1/worker/gpu/kv_connector.py
        output = KVConnectorOutput()
        if wait_for_save:
            self.kv_connector.wait_for_save()
        output.finished_sending, output.finished_recving = (
            self.kv_connector.get_finished(finished_req_ids)
        )
        output.invalid_block_ids = self.kv_connector.get_block_ids_with_load_errors()
        output.kv_connector_stats = self.kv_connector.get_kv_connector_stats()
        output.kv_cache_events = self.kv_connector.get_kv_connector_kv_cache_events()
        output.kv_connector_worker_meta = (
            self.kv_connector.build_connector_worker_meta()
        )
```

`finished_recving` 会回到 scheduler。

scheduler 收到后：

```2497:2502:vllm/v1/core/sched/scheduler.py
        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            assert req_id in self.requests
            req = self.requests[req_id]
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)
```

下一轮 schedule 就能把这个 request 从：

```text
WAITING_FOR_REMOTE_KVS
```

恢复成：

```text
WAITING / PREEMPTED
```

再进入 `running`。

## 一句话

拉 KV 的流程是：

```text
Scheduler:
  connector.get_num_new_matched_tokens()
  allocate_slots()
  connector.update_state_after_alloc()
  scheduler_output.kv_connector_metadata = meta

Worker:
  bind_connector_metadata()
  start_load_kv()
  wait_for_layer_load()
  post_forward()
  get_finished() -> finished_recving

Scheduler:
  收到 finished_recving
  request 从 WAITING_FOR_REMOTE_KVS 恢复可调度
```
````





## 解释生成一份 KV 传输 metadata，并挂到 scheduler_output 上，传给 worker

````python
可以把它理解成：

**scheduler 不是自己去拉 KV。scheduler 只决定“这一轮哪些 request 要跑、每个 request 分到了哪些 KV block、哪些 KV 要从远端来”。然后它把这些信息整理成 metadata，塞进 `scheduler_output`，交给 worker 去真正执行。**

流程是这样：

```text
Scheduler.schedule()
  1. 选 request
  2. 分配本地 KV block
  3. 判断哪些 token 已经在远端 KV cache 里
  4. 生成 connector metadata
  5. scheduler_output.kv_connector_metadata = meta
    
   

ModelExecutor.execute_model(scheduler_output)
  6. worker 收到 scheduler_output
  7. worker 从 scheduler_output.kv_connector_metadata 取出 metadata
  8. worker 根据 metadata 拉远端 KV / 保存 KV
  9. 然后才执行 forward
```

“挂到 `scheduler_output` 上”就是这个赋值：

```python
scheduler_output.kv_connector_metadata = meta
```

`SchedulerOutput` 是 scheduler 给 model runner / worker 的本轮执行说明，里面本来就有：

```text
scheduled_new_reqs        本轮新进来的请求
scheduled_cached_reqs     已在 batch 里的请求
num_scheduled_tokens      每个请求本轮算多少 token
block_ids                 分到哪些 KV cache block
finished_req_ids          哪些请求结束了
kv_connector_metadata     KV connector 额外需要的信息
```

所以 `kv_connector_metadata` 是附加字段，专门给 KV connector 用。

举个远端 prefill 的例子：

```text
请求 req-1 来了
scheduler 发现：
  本地 prefix cache 命中 128 tokens
  远端 KV cache 命中 512 tokens
  还需要把远端 384 tokens 的 KV 拉到本地 block 里

scheduler 分配本地 block：
  local_block_ids = [10, 11, 12, 13]

connector metadata 里就会写：
  req_id = req-1
  local_block_ids = [10, 11, 12, 13]
  remote_engine_id = prefill-engine-0
  remote_block_ids = [80, 81, 82, 83]
  remote_host = ...
  remote_port = ...
```

然后 worker 拿到它：

```text
worker:
  看到 req-1 需要 recv KV
  从 remote_engine/block 拉 KV
  写入本地 local_block_ids
  KV 拉完后，模型 forward 可以直接接着 decode
```

一句话：**metadata 是 scheduler 给 worker 的 KV 传输任务单；scheduler 负责规划，worker 负责执行。**
````

````python
```text
1. 选 request
2. 查本地 prefix cache
3. 查远端 KV cache 命中多少 token
4. 分配本地 KV block
5. connector 记录本轮要拉/存哪些 KV
6. build_connector_meta()
7. scheduler_output.kv_connector_metadata = meta
```

### 1. 选 request

scheduler 从 `waiting` 队列里拿请求，判断它这一轮能不能被调度。

主要看这些限制：

```text
token_budget 是否够
KV cache block 是否够
encoder / multimodal 资源是否够
是否有结构化输出等待
是否正在 WAITING_FOR_REMOTE_KVS
```

如果可以调度，才继续往下走。

### 2. 查本地 prefix cache

scheduler 先问本地 KV cache manager：

```python
new_computed_blocks, num_new_local_computed_tokens = (
    self.kv_cache_manager.get_computed_blocks(request)
)
```

意思是：

```text
这个 request 的前缀，有多少 token 的 KV 已经在本地 GPU KV cache 里？
```

例如：

```text
prompt 1000 tokens
本地 prefix cache 命中 256 tokens
num_new_local_computed_tokens = 256
```

这些 token 不需要重新 prefill。

### 3. 查远端 KV cache

如果启用了 connector：

```python
ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(
    request, num_new_local_computed_tokens
)
```

意思是：

```text
除了本地已经命中的 token，远端 KV cache 里还能命中多少 token？
是否需要异步拉 KV？
```

例如：

```text
prompt 1000 tokens
本地命中 256
远端 LMCache / Mooncake / NIXL 命中到 768

num_external_computed_tokens = 768
```

注意这里的 `768` 通常表示总的 external matched token 数，scheduler 后面会结合本地命中来算本轮要不要拉、拉多少。

### 4. 分配本地 KV block

scheduler 知道远端有 KV 以后，还要先在本地 GPU KV cache 里分配位置：

```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_new_computed_tokens=num_new_local_computed_tokens,
    new_computed_blocks=new_computed_blocks,
    num_external_computed_tokens=num_external_computed_tokens,
    delay_cache_blocks=load_kv_async,
)
```

意思是：

```text
远端 KV 拉回来以后，要写到本地哪些 block？
```

比如：

```text
local_block_ids = [10, 11, 12, 13]
```

这些 block 是 worker 后面接收 KV 的目标地址。

如果是 async KV load：

```text
这轮不做 forward
先把 request 放到 WAITING_FOR_REMOTE_KVS
等 KV 拉完，再重新进入 WAITING
```

### 5. connector 记录本轮任务

分配完 block 后，scheduler 调：

```python
self.connector.update_state_after_alloc(
    request,
    self.kv_cache_manager.get_blocks(request_id),
    num_external_computed_tokens,
)
```

这一步不是生成最终 metadata，而是让 connector 暂存信息：

```text
req_id 是谁
本地分到了哪些 block
远端要拉多少 token
kv_transfer_params 里远端 engine / host / port / block ids 是什么
```

例如 Mooncake 会把请求放进 `_reqs_need_recv` 或 `_reqs_need_send`。

### 6. 生成 connector metadata

等这一轮所有 request 都调度完，scheduler 创建 `SchedulerOutput`，然后：

```python
meta = self._build_kv_connector_meta(self.connector, scheduler_output)
```

内部就是：

```python
connector.build_connector_meta(scheduler_output)
```

connector 会把刚才暂存的 `_reqs_need_recv`、`_reqs_need_send` 等整理成一个对象。

例如 NIXL 可能长这样：

```python
NixlConnectorMetadata(
    reqs_to_recv={
        "req-1": ReqMeta(
            local_block_ids=...,
            remote=RemoteMeta(
                block_ids=...,
                engine_id="prefill-engine",
                host="...",
                port=...,
                request_id="remote-req-1",
            ),
        )
    }
)
```

### 7. 挂到 scheduler_output

最后：

```python
scheduler_output.kv_connector_metadata = meta
```

意思是：

```text
把这份 KV 传输任务单，附加到本轮 scheduler 输出里。
```

后面 `model_executor.execute_model(scheduler_output)` 会把它传给 worker。worker 侧在 forward 前读取它，然后真正执行：

```text
拉远端 KV
写入 local_block_ids
等待必要 layer 的 KV load 完成
继续 forward / decode
```

一句话总结：

**scheduler 先决定“哪些 token 不用算、KV 应该放到哪些本地 block”，connector 再把这些信息包装成 metadata，worker 根据 metadata 真正去拉 KV。**
````





## request 的 prompt 前缀，有多少 KV 已经在本地 vLLM prefix cache 里，可以直接复用

````python
这行算的是：**这个 request 的 prompt 前缀，有多少 KV 已经在本地 vLLM prefix cache 里，可以直接复用。**

```python
new_computed_blocks, num_new_local_computed_tokens = (
    self.kv_cache_manager.get_computed_blocks(request)
)
```

返回两个东西：

`new_computed_blocks`  
本地命中的 KV cache blocks，也就是已经算好 KV 的 block 对象。

`num_new_local_computed_tokens`  
这些命中的 block 覆盖了多少 token。

核心逻辑在 `vllm/v1/core/kv_cache_manager.py`：

```207:237:vllm/v1/core/kv_cache_manager.py
    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        ...
        if not self.enable_caching or request.skip_reading_prefix_cache:
            return self.empty_kv_cache_blocks, 0

        max_cache_hit_length = request.num_tokens - 1
        computed_blocks, num_new_computed_tokens = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )
```

### 第一步：如果不能读 prefix cache，直接返回 0

```python
if not self.enable_caching or request.skip_reading_prefix_cache:
    return self.empty_kv_cache_blocks, 0
```

也就是说：

```text
没有开启 prefix caching
或者这个 request 不允许读 prefix cache
```

那就没有本地命中：

```text
new_computed_blocks = empty
num_new_local_computed_tokens = 0
```

### 第二步：拿 request.block_hashes 去查

`request.block_hashes` 是这个 request 按 block 切分后生成的 hash。

比如 block size = 16：

```text
tokens: [0 ... 15]    -> block_hash_0
tokens: [0 ... 31]    -> block_hash_1
tokens: [0 ... 47]    -> block_hash_2
```

注意它是链式 hash，后一个 block hash 依赖前面的 prefix，所以能代表“到这个 block 为止的完整前缀”。

hash 在 `Request` 创建和追加 token 时更新：

```247:250:vllm/v1/request.py
    def update_block_hashes(self) -> None:
        """Compute block hashes for any new full blocks and append them."""
        if self._block_hasher is not None:
            self.block_hashes.extend(self._block_hasher(self))
```

### 第三步：为什么是 `request.num_tokens - 1`

```python
max_cache_hit_length = request.num_tokens - 1
```

原因是：**即使整个 prompt 都命中 cache，也要重新算最后一个 token 来拿 logits。**

比如 prompt 有 100 tokens，如果 100 个 token 全部 cache 命中，也不能直接采样，因为需要最后位置的 logits。所以最多认为：

```text
max_cache_hit_length = 99
```

### 第四步：找最长连续命中的前缀

对普通 full attention，逻辑很直观：

```592:602:vllm/v1/core/single_type_kv_cache_manager.py
        for block_hash in itertools.islice(block_hashes, max_num_blocks):
            # block_hashes is a chain of block hashes. If a block hash is not
            # in the cached_block_hash_to_id, the following block hashes are
            # not computed yet for sure.
            if cached_block := block_pool.get_cached_block(
                block_hash, kv_cache_group_ids
            ):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
```

意思是从第一个 block 开始查：

```text
block_hash_0 命中 -> 继续
block_hash_1 命中 -> 继续
block_hash_2 miss -> 停
```

它只认**连续前缀命中**，不会说：

```text
block 0 命中
block 1 miss
block 2 命中
```

然后跳过 block 1 去复用 block 2。因为 attention 需要连续上下文 KV。

### 第五步：命中 token 数怎么算

普通单一 KV group 的情况：

```486:497:vllm/v1/core/kv_cache_coordinator.py
        hit_blocks = self.single_type_managers[0].find_longest_cache_hit(
            ...
        )
        return hit_blocks, len(hit_blocks[0]) * self.block_size
```

也就是：

```text
num_new_local_computed_tokens = 命中 block 数 * block_size
```

举例：

```text
block_size = 16
request 有 80 tokens
本地 prefix cache 命中前 3 个 block

new_computed_blocks = [blockA, blockB, blockC]
num_new_local_computed_tokens = 3 * 16 = 48
```

scheduler 后面就知道：

```text
前 48 个 token 的 KV 不用重新算
从 token 48 之后继续 prefill / decode
```

### 第六步：block 是从哪里查出来的

实际查的是 `BlockPool` 里的 hash map：

```198:223:vllm/v1/core/block_pool.py
    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        ...
        for group_id in kv_cache_group_ids:
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, group_id
            )
            block = self.cached_block_hash_to_block.get_one_block(
                block_hash_with_group_id
            )
            if not block:
                return None
            cached_blocks.append(block)
        return cached_blocks
```

也就是说 key 大概是：

```text
(block_hash, kv_cache_group_id)
```

value 是：

```text
KVCacheBlock
```

如果有多个 KV group，必须每个 group 都能找到对应 block，否则这一段不能算完整命中。

一句话总结：

**`get_computed_blocks(request)` 用 request 的 block hash 从本地 prefix cache 的 hash map 里，从头开始查最长连续命中的完整 block，然后返回这些 block 和它们覆盖的 token 数。**
````





## 为什么是coordinator调用（self.coordinator.find_longest_cache_hit），真正的代码实现

````python
因为 `KVCacheManager` 本身不直接知道“这个模型的 KV cache 结构有几组、每组是什么 attention 类型”。它把这件事交给 `coordinator`。

这行：

```python
self.coordinator.find_longest_cache_hit(
    request.block_hashes, max_cache_hit_length
)
```

意思是：

```text
KVCacheManager: 我要查本地 prefix cache 命中
Coordinator: 我知道当前模型的 KV cache 结构，我来决定怎么查
```

### 为什么需要 coordinator

现在 vLLM 里 KV cache 不一定只有一种。

可能是简单模型：

```text
只有一组 full attention KV cache
```

也可能是 hybrid 模型：

```text
一部分 layer 是 full attention
一部分 layer 是 sliding window attention
一部分可能是 Mamba / MLA / local attention
```

不同 KV group 的 block size、cache hit 规则、可复用长度可能不同。scheduler 最终需要的是一个统一结果：

```text
这个 request 最长可以复用多少 token？
每个 KV group 对应哪些 cached blocks？
```

所以中间有 `KVCacheCoordinator` 来协调。

### coordinator 是在哪里创建的

在 `get_kv_cache_coordinator()`：

```808:849:vllm/v1/core/kv_cache_coordinator.py
def get_kv_cache_coordinator(
    kv_cache_config: KVCacheConfig,
    ...
) -> KVCacheCoordinator:
    if not enable_caching:
        return KVCacheCoordinatorNoPrefixCache(
            ...
        )
    if len(kv_cache_config.kv_cache_groups) == 1:
        return UnitaryKVCacheCoordinator(
            ...
        )
    return HybridKVCacheCoordinator(
```

也就是三种情况：

```text
prefix caching 关闭:
  KVCacheCoordinatorNoPrefixCache

只有一个 KV cache group:
  UnitaryKVCacheCoordinator

多个 KV cache group:
  HybridKVCacheCoordinator
```

### 1. 如果没开 prefix cache

真正实现是：

```417:425:vllm/v1/core/kv_cache_coordinator.py
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(self.num_single_type_manager)
        )
        return blocks, 0
```

直接返回：

```text
命中 blocks = 空
命中 token 数 = 0
```

### 2. 普通单 KV group 模型

真正实现是 `UnitaryKVCacheCoordinator.find_longest_cache_hit()`：

```481:497:vllm/v1/core/kv_cache_coordinator.py
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        hit_blocks = self.single_type_managers[0].find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=max_cache_hit_length,
            kv_cache_group_ids=[0],
            block_pool=self.block_pool,
            kv_cache_spec=self.kv_cache_spec,
            drop_eagle_block=0 in self.eagle_group_ids,
            alignment_tokens=self.block_size,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=self.pcp_world_size,
        )
        return hit_blocks, len(hit_blocks[0]) * self.block_size
```

它又继续委托给具体的 `SingleTypeKVCacheManager`。

最常见 full attention 的真正查找代码在 `FullAttentionManager`：

```592:602:vllm/v1/core/single_type_kv_cache_manager.py
        for block_hash in itertools.islice(block_hashes, max_num_blocks):
            # block_hashes is a chain of block hashes. If a block hash is not
            # in the cached_block_hash_to_id, the following block hashes are
            # not computed yet for sure.
            if cached_block := block_pool.get_cached_block(
                block_hash, kv_cache_group_ids
            ):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
```

这才是真正查 hash map 的地方：

```text
按 block_hash 从头往后查
命中就 append cached block
遇到第一个 miss 就 break
```

然后返回：

```text
hit_blocks
len(hit_blocks[0]) * block_size
```

### 3. Hybrid 模型

如果模型有多个 KV cache group，就走 `HybridKVCacheCoordinator.find_longest_cache_hit()`。

核心逻辑：

```647:667:vllm/v1/core/kv_cache_coordinator.py
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """
        Find the longest cache hit using an iterative fixed-point algorithm.

        Each attention type either accepts the current candidate length or
        reduces it. If any type reduces the length, restart checks over all
        types. This converges because length monotonically decreases and is
        bounded below by 0.
```

简单说：

```text
先假设可以命中 max_cache_hit_length
问 full attention group 能不能接受这个长度
问 sliding window / mamba / 其他 group 能不能接受这个长度
如果某个 group 只能命中更短，就把整体 hit_length 降低
再重新检查
最后得到所有 KV group 都能共同接受的最长 prefix hit
```

所以这里必须用 coordinator，否则单个 group 查出来的 hit length 可能不适用于其他 group。

### 总结

这行：

```python
self.coordinator.find_longest_cache_hit(...)
```

不是最终查 cache 的地方，而是入口。

真正实现路径通常是：

```text
KVCacheManager.get_computed_blocks()
  -> KVCacheCoordinator.find_longest_cache_hit()

如果普通模型:
  -> UnitaryKVCacheCoordinator.find_longest_cache_hit()
  -> FullAttentionManager.find_longest_cache_hit()
  -> block_pool.get_cached_block()

如果 hybrid 模型:
  -> HybridKVCacheCoordinator.find_longest_cache_hit()
  -> 各个 SingleTypeKVCacheManager.find_longest_cache_hit()
  -> 取所有 group 都兼容的最长 hit_length
```

一句话：**coordinator 是为了协调不同 KV cache group；真正查 hash map 的代码在具体的 `SingleTypeKVCacheManager`，最常见的是 `FullAttentionManager.find_longest_cache_hit()`。**
````



## 一次 engine step把整条链路展开

````python
下面按**一次 engine step**把整条链路展开，并带数值。

---

## 总览：从哪里进，到哪里用

```text
EngineCore.step()
  │
  ├─① scheduler.schedule()
  │     输出 SchedulerOutput：
  │       - 每个 request 本步跑多少 token
  │       - 新/旧请求的 block_ids
  │
  └─② model_executor.execute_model(scheduler_output)
        └─ GPUModelRunner.execute_model()
              ├─③ _update_states()          # 把 block_ids 写进 InputBatch
              ├─④ _prepare_inputs()         # 算 positions，启动 Triton 写 slot_mapping.gpu
              ├─⑤ _get_slot_mappings()      # 取出 slot_mapping.gpu 的引用
              ├─⑥ set_forward_context(slot_mapping=...)
              └─⑦ model forward
                    └─ Attention.forward()
                         └─ unified_kv_cache_update()
                              └─ do_kv_cache_update(slot_mapping)
                                   └─ reshape_and_cache(..., slot_mapping)
```

入口就是 `EngineCore.step()` 里的这两行：

```490:491:vllm/v1/engine/core.py
        scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
```

---

## 数值设定

```text
block_size = 4
本步 batch：
  req A: prompt 已算完 0 个 token，本步 schedule 6 个 prompt token
  req B: 已算完 10 个 token，本步 schedule 1 个 decode token

scheduler 分配的 block：
  req A: block_table = [7, 9]
  req B: block_table = [3, 12, 15]   # 10 个 token 占 3 个 block（第 3 个只写了部分）
```

`SchedulerOutput` 里大致有：

```text
num_scheduled_tokens = {"A": 6, "B": 1}
req A.block_ids = ([7, 9],)
req B 本步可能追加 new_block_ids（若需要新页）
```

---

## ① `scheduler.schedule()`

CPU 上做：

- 决定本步每个 request 算几个 token；
- 查 prefix cache / 分配 KV block；
- 产出 `SchedulerOutput`。

**这里还不生成 `slot_mapping`。** 它只告诉 worker：用哪些 `block_id`、算哪些 token。

---

## ②③ `execute_model` → `_update_states`

`GPUModelRunner.execute_model()` 开头：

```4121:4121:vllm/v1/worker/gpu_model_runner.py
            deferred_state_corrections_fn = self._update_states(scheduler_output)
```

`_update_states` 会：

1. 把新请求放进 `self.requests` / `InputBatch`；
2. 把 `block_ids` 写进 `input_batch.block_table`（CPU 侧 numpy，再 `commit_block_table` 拷到 GPU）；
3. 更新 `num_computed_tokens`。

对数值例子：

```text
InputBatch 行序假设：
  row 0 = req A, block_table.np[0] = [7, 9]
  row 1 = req B, block_table.np[1] = [3, 12, 15]

num_computed_tokens_cpu:
  A = 0
  B = 10
num_scheduled_tokens:
  A = 6
  B = 1
```

---

## ④ `_prepare_inputs`：算 position，再算 slot

### 4.1 先把 block_table 拷到 GPU

```1940:1942:vllm/v1/worker/gpu_model_runner.py
        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit_block_table(num_reqs)
```

### 4.2 展开每个 request 的 token

```1944:1952:vllm/v1/worker/gpu_model_runner.py
        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # self.query_pos.np[:10]: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens = self._get_cumsum_and_arange(
            num_scheduled_tokens, self.query_pos.np
        )
```

对我们的例子：

```text
num_scheduled_tokens = [6, 1] # 表示这一轮：req A 调度 6 个 token  req B 调度 1 个 token
req_indices          = [0,0,0,0,0,0, 1]
query_pos            = [0,1,2,3,4,5, 0]
query_start_loc      = [0, 6, 7]
```

含义：

```text
flatten batch 里：
  index 0..5 属于 req A
  index 6     属于 req B
```

### 4.3 算每个 token 的 sequence position

```2151:2154:vllm/v1/worker/gpu_model_runner.py
        self.positions[:total_num_scheduled_tokens] = (
            self.num_computed_tokens[req_indices_gpu].to(torch.int64)
            + self.query_pos.gpu[:total_num_scheduled_tokens]
        )
```

```text
req A: num_computed=0
  positions = 0+0, 0+1, ..., 0+5 = [0,1,2,3,4,5]

req B: num_computed=10
  positions = 10+0 = [10]
```

所以 GPU 上：

```text
positions = [0, 1, 2, 3, 4, 5, 10]
```

### 4.4 Triton 写 `slot_mapping.gpu`

```2160:2164:vllm/v1/worker/gpu_model_runner.py
        self.input_batch.block_table.compute_slot_mapping(
            num_reqs,
            self.query_start_loc.gpu[: num_reqs + 1],
            self.positions[:total_num_scheduled_tokens],
        )
```

Triton 对每个 token：

```text
block_idx = position // block_size
offset    = position %  block_size
block_id  = block_table[req_row, block_idx]
slot_id   = block_id * block_size + offset
```

带入数值：

```text
req A:
  pos 0 -> block_table[0]=7, offset 0 -> slot 28
  pos 1 -> block 7, offset 1 -> slot 29
  pos 2 -> block 7, offset 2 -> slot 30
  pos 3 -> block 7, offset 3 -> slot 31
  pos 4 -> block_table[1]=9, offset 0 -> slot 36
  pos 5 -> block 9, offset 1 -> slot 37

req B:
  pos 10 -> block_table[2]=15, offset 2 -> slot 15*4+2 = 62
```

结果直接写在：

```text
block_table.slot_mapping.gpu = [28, 29, 30, 31, 36, 37, 62]
```

**注意：这里没有拷回 CPU。** 结果就在 GPU buffer 里。

---

## ⑤ `_get_slot_mappings`：取出 GPU tensor 引用

```4038:4039:vllm/v1/worker/gpu_model_runner.py
                blk_table = self.input_batch.block_table[kv_cache_gid]
                slot_mapping = blk_table.slot_mapping.gpu[:num_tokens_padded]
```

然后按 layer 名字挂一份同一个 GPU tensor：

```text
slot_mappings_by_layer = {
  "model.layers.0.self_attn.attn": tensor([28,29,30,31,36,37,62], device=cuda),
  "model.layers.1.self_attn.attn": 同上（同一个 GPU 内存引用）,
  ...
}
```

“直接用”的意思就是：**Python 只传 pointer/tensor 引用，不会把 slot_mapping 搬到 CPU。**

---

## ⑥ `set_forward_context`：挂到全局 forward 上下文

```4345:4354:vllm/v1/worker/gpu_model_runner.py
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                ...
                slot_mapping=slot_mappings,
            ),
```

`ForwardContext` 里有字段：

```136:136:vllm/forward_context.py
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]
```

模型 forward 期间，任意 attention 层都能通过 `get_forward_context()` 拿到这份 GPU tensor。

为什么不直接当函数参数传？因为 attention 是自定义 op / torch.compile opaque 区域，很多元数据走 forward context，避免把复杂 Python 对象硬塞进编译图。

---

## ⑦ attention 真正写 KV cache

每层 `Attention.forward()`：

1. 投影出当前 batch 的 `key/value`（形状约 `[7, num_kv_heads, head_dim]`）；
2. 调 `unified_kv_cache_update(key, value, layer_name)`；
3. 内部：

```761:790:vllm/model_executor/layers/attention/attention.py
    slot_mapping = forward_context.slot_mapping
    ...
    layer_slot_mapping = slot_mapping.get(layer_name)
    ...
        attn_layer.impl.do_kv_cache_update(
            attn_layer, key, value, kv_cache, layer_slot_mapping,
        )
```

4. Triton 后端调用：

```804:809:vllm/v1/attention/backends/triton_attn.py
        triton_reshape_and_cache_flash(
            key, value, key_cache, value_cache, slot_mapping, ...
        )
```

`reshape_and_cache` 对每个 token：

```text
token 0 的 K/V -> 写到 KV cache slot 28
token 1 的 K/V -> slot 29
...
token 6 的 K/V -> slot 62
```

也就是：

```text
key_cache[block=7, offset=0] = K of A@pos0
...
key_cache[block=15, offset=2] = K of B@pos10
```

---

## 一句话串起来

| 阶段 | 在哪 | 算什么 |
|---|---|---|
| `schedule()` | CPU | 选哪些 token、分配哪些 `block_id` |
| `_update_states` | CPU→GPU | 把 `block_table` 同步到 GPU |
| `_prepare_inputs` | GPU | 由 `position + block_table` 算出 `slot_mapping.gpu` |
| `set_forward_context` | Python 引用 | 把 GPU tensor 挂到全局上下文 |
| `reshape_and_cache` | GPU | 按 `slot_mapping` 把 K/V 写入 GPU KV cache |

所以：**不是 schedule 直接用 `slot_mapping`**；schedule 只产出 block 分配。`slot_mapping` 是 worker 在 execute 前用 Triton 算好的写入地址，然后通过 `ForwardContext` 传给 attention，由 `reshape_and_cache` 完成真正的 KV 存储。
````







## 创建 / 启动 WorkerProc 子进程

````python
创建 / 启动发生在 `MultiprocExecutor._init_executor()` 里，核心是 `WorkerProc.make_worker_process()` + `proc.start()`。

## 总流程

```text
EngineCore 创建 MultiprocExecutor
  -> MultiprocExecutor._init_executor()
     1. 创建 rpc_broadcast_mq（给 worker 传命令）
     2. for local_rank in range(local_world_size):
          WorkerProc.make_worker_process(...)
            -> multiprocessing.Process(target=WorkerProc.worker_main)
            -> proc.start()
     3. WorkerProc.wait_for_ready(...)  # 等所有 worker 发 READY
     4. 可选 start_worker_monitor()
```

## 1. Executor 初始化时批量创建

```176:205:vllm/v1/executor/multiproc_executor.py
            for local_rank in range(self.local_world_size):
                global_rank = global_start_rank + local_rank
                is_driver_worker = self._is_driver_worker(global_rank)
                ...
                    unready_worker_handle = WorkerProc.make_worker_process(
                        vllm_config=self.vllm_config,
                        local_rank=local_rank,
                        rank=global_rank,
                        distributed_init_method=distributed_init_method,
                        input_shm_handle=scheduler_output_handle,
                        shared_worker_lock=shared_worker_lock,
                        is_driver_worker=is_driver_worker,
                        inherited_fds=inherited_fds,
                    )
                unready_workers.append(unready_worker_handle)
            ...
            self.workers = WorkerProc.wait_for_ready(unready_workers)

            if self.monitor_workers:
                self.start_worker_monitor()
```

`local_world_size` 一般等于本机 GPU 数。比如 TP=2，就会起 2 个 `VllmWorker-*` 进程。

## 2. `make_worker_process`：真正 fork/spawn

```672:712:vllm/v1/executor/multiproc_executor.py
        context = get_mp_context()
        ready_reader, ready_writer = context.Pipe(duplex=False)
        death_reader, death_writer = context.Pipe(duplex=False)
        ...
        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "ready_pipe": ready_writer,
            "death_pipe": death_reader,
            ...
        }
        proc = context.Process(
            target=WorkerProc.worker_main,
            kwargs=process_kwargs,
            name=f"VllmWorker-{rank}",
            daemon=True,
        )
        ...
            proc.start()
```

这里做了几件事：

1. 建两条 pipe  
   - `ready_pipe`：子进程告诉父进程“我准备好了”  
   - `death_pipe`：父进程退出时，子进程能感知到并退出  
2. 组装 `process_kwargs`，传给子进程  
3. `Process(target=WorkerProc.worker_main, ...)`  
4. `proc.start()` 真正启动子进程  

## 3. 子进程入口 `worker_main`

子进程起来后跑：

```868:892:vllm/v1/executor/multiproc_executor.py
            worker = WorkerProc(*args, **kwargs)
            ...
            ready_writer.send(
                {
                    "status": WorkerProc.READY_STR,
                    "handle": worker.worker_response_mq.export_handle(),
                    "peer_response_handles": worker.peer_response_handles,
                }
            )
            ...
            worker.rpc_broadcast_mq.wait_until_ready()
            worker.worker_response_mq.wait_until_ready()
            ...
            worker.worker_busy_loop()
```

`WorkerProc.__init__` 里会：

- 创建 `WorkerWrapperBase`
- `init_device()` / `load_model()`
- `_init_message_queues(...)` 接上广播 MQ 和 response MQ

然后发 `READY`，再进入 `worker_busy_loop()` 等 RPC。

## 4. 父进程等待 READY

```text
父进程: wait_for_ready(unready_workers)
  从每个 ready_pipe.recv()
  收到 {"status": "READY", "handle": ...}
  用 handle 建 response_mq
  得到正式的 WorkerProcHandle 列表
```

全部 READY 后，`MultiprocExecutor` 才能开始 `collective_rpc("execute_model", ...)`。

## 一句话

```text
MultiprocExecutor._init_executor
  -> 对每个 local GPU rank
     WorkerProc.make_worker_process
       -> multiprocessing.Process(target=worker_main).start()
  -> wait_for_ready
  -> 子进程内 WorkerProc(...) 初始化后进入 worker_busy_loop
```

所以 `WorkerProc` 不是在每次 `execute_model` 时新建，而是 **executor 初始化时一次性创建并常驻**。
````























continuous batching 怎么工作
How does continuous batching work?

PagedAttention 为什么存在
Why does PagedAttention exist?

KV block 怎么分配和回收
How are KV blocks allocated and reclaimed?

prefill 和 decode 有什么不同
What is the difference between prefill and decode?

model runner 怎么执行
How does the model runner execute the model?

TP 怎么接进去
How is Tensor Parallelism integrated into vLLM?

distributed executor 怎么组织 worker
How does the distributed executor organize workers?

prefix cache 怎么复用
How does prefix caching reuse cached KV states?

speculative decoding 怎么接
How is speculative decoding integrated into vLLM?

scheduler 的吞吐和延迟怎么权衡
How does the scheduler balance throughput and latency?

一次 OOM/性能下降怎么定位
How do you diagnose an OOM or a performance regression?

1）vLLM 的 request 生命周期是什么？从 API Server 到最终返回 token，中间经过哪些核心对象？
1）What is the lifecycle of a request in vLLM, from the API server to the final token output, and which core components does it pass through?

2）Scheduler 每轮调度具体维护哪些状态？waiting、running、finished request 如何迁移？
2）What states does the scheduler maintain in each scheduling iteration, and how do requests transition among waiting, running, and finished states?

3）Scheduler 如何决定一次 iteration 里哪些 request 做 prefill、哪些做 decode？
3）How does the scheduler decide which requests perform prefill and which perform decode in each iteration?

4）Chunked Prefill 是怎么工作的？为什么它能改善长 prompt 对 decode 请求的阻塞？
4）How does Chunked Prefill work, and why can it reduce the blocking caused by long prompts on decode requests?

5）max_num_batched_tokens、max_num_seqs、block size 分别会怎样影响吞吐、延迟和显存？
5）How do max_num_batched_tokens, max_num_seqs, and block size affect throughput, latency, and GPU memory usage?

6）KV Cache 的实际显存占用怎么算？给定模型结构、dtype、TP 和 GPU 显存，能容纳多少 token？
6）How do you calculate the actual GPU memory usage of the KV Cache, and given the model architecture, dtype, TP size, and available GPU memory, how many tokens can it hold?

7）PagedAttention 的 block table 是什么？逻辑 token block 如何映射到物理 KV block？
7）What is the block table in PagedAttention, and how are logical token blocks mapped to physical KV blocks?

8）一次 attention forward 中，PagedAttention kernel 怎么根据 block table 找到对应的 K/V？
8）During an attention forward pass, how does the PagedAttention kernel locate the corresponding K/V data using the block table?

9）KV Cache fragmentation 会不会发生？vLLM 的 block-based 管理解决了什么碎片问题，又没有解决什么？
9）Can KV Cache fragmentation still occur, and which fragmentation problems does vLLM's block-based memory management solve or fail to solve?

10）当 KV Cache 不够时，Scheduler 怎么处理？哪些 request 会被抢占，preemption 的代价是什么？
10）What does the scheduler do when the KV Cache runs out of space, which requests are preempted, and what is the cost of preemption?

11）Prefix Caching 的 hash key 怎么生成？两个 request 到什么程度才能真正共享 KV block？
11）How is the hash key for Prefix Caching generated, and under what conditions can two requests actually share KV blocks?

12）Prefix Cache、LoRA、MultiModal input、不同 sampling 参数之间有哪些兼容性和复用限制？
12）What compatibility and reuse limitations exist among Prefix Cache, LoRA, multimodal inputs, and different sampling parameters?

13）CUDA Graph 在 vLLM 里解决什么问题？为什么 decode 特别适合 CUDA Graph，而动态 shape 会让它变复杂？
13）What problem does CUDA Graph solve in vLLM, why is decode especially suitable for CUDA Graph, and why do dynamic shapes make it more complicated?

14）vLLM 怎么选择 eager execution 和 CUDA Graph execution？哪些情况会触发 graph miss / fallback？
14）How does vLLM choose between eager execution and CUDA Graph execution, and what situations trigger a graph miss or fallback?

15）一次 model forward 里有哪些主要 kernel？RMSNorm、QKV projection、RoPE、Attention、MLP、Sampling 分别可能成为怎样的瓶颈？
15）What are the major kernels in a model forward pass, and how can RMSNorm, QKV projection, RoPE, Attention, MLP, and Sampling become performance bottlenecks?

16）Tensor Parallel 下一个 Transformer Layer 怎么执行？ColumnParallel、RowParallel 分别在哪里产生通信？
16）How is a Transformer layer executed under Tensor Parallelism, and where do ColumnParallel and RowParallel introduce communication?

17）TP 为什么经常出现 GPU utilization 很高但 tokens/s 上不去？怎么判断瓶颈是 compute、memory bandwidth 还是 NCCL？
17）Why can Tensor Parallelism show high GPU utilization while tokens per second remains low, and how do you determine whether the bottleneck is compute, memory bandwidth, or NCCL communication?

18）多 GPU 推理时 TP、PP、DP、EP 分别解决什么问题？什么模型/流量应该选择哪种并行策略？
18）In multi-GPU inference, what problems do TP, PP, DP, and EP each solve, and how should you choose among them for different models and traffic patterns?

19）MoE 模型在 vLLM 里怎么执行？Expert Parallel、token routing、all-to-all 为什么可能成为新的性能瓶颈？
19）How are MoE models executed in vLLM, and why can Expert Parallelism, token routing, and all-to-all communication become new performance bottlenecks?

20）如果线上出现 TTFT 正常但 TPOT 突然恶化，怎么从 scheduler、batch size、KV cache、CUDA kernel、NCCL、GPU utilization 一层层定位？
20）If TTFT remains normal but TPOT suddenly degrades in production, how do you diagnose the issue step by step across the scheduler, batch size, KV Cache, CUDA kernels, NCCL, and GPU utilization?
