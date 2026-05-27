---
title: "TensorRT-LLM Workflow"
published: 2026-05-26
description: "TensorRT-LLM Workflow"
image: ""
tags: ["trt","TensorRT-LLM Workflow"]
category: trt
draft: false
lang: ""
createdAt: "2026-05-26T05:23:13.982.985026969Z"
---

# TensorRT-LLM PyTorch Backend

`examples/llm-api/llm_inference.py`

**图 1：脚本入口到 LLM 初始化**

```mermaid
sequenceDiagram
    participant LLM
    participant _TorchLLM
    participant BaseLLM
    participant TorchLlmArgs
    participant MpiPoolSession

    LLM->>_TorchLLM: __init__(model, sampling/runtime kwargs)
    _TorchLLM->>BaseLLM: __init__(backend="pytorch")
    BaseLLM->>TorchLlmArgs: parse + validate args
    Note over TorchLlmArgs: 配置入口：parallel_config, kv_cache_config,<br/>enable_chunked_prefill, speculative_config,<br/>lora_config, guided_decoding_backend,<br/>checkpoint_loader, checkpoint_format, attn_backend
    alt world_size > 1
        BaseLLM->>MpiPoolSession: start worker processes
        Note over MpiPoolSession: Multi-GPU/Multi-Node：TP/PP/CP/EP/DP 依赖这里的 worker/rank 编排
    end
    BaseLLM->>_TorchLLM: _build_model()
```

Base-> MpiPoolSession: 这段表示当模型需要多 GPU 推理时，`BaseLLM` 会启动一组 MPI worker 进程。TP/PP/CP 这些并行方式已经体现在 `parallel_config.world_size` 里，这里做的是按这个数量创建 worker。

1. `world_size > 1` 表示需要多进程/多 GPU  
   代码里对应的是：

   ```python
   if self.args.parallel_config.is_multi_gpu:
   ```

   也就是只要 `parallel_config` 判断当前不是单 GPU，就进入多 GPU 初始化逻辑。

2. `BaseLLM->>MpiPoolSession: start worker processes` 对应创建 MPI session  
   代码位置在 `BaseLLM.__init__`：

   ```python
   self.mpi_session = MpiPoolSession(
       n_workers=self.args.parallel_config.world_size
   )
   ```

   这里的 `n_workers` 就是总 rank 数。

3. `TP/PP/CP 等并行 rank 编排` 来自 `parallel_config`  
   `parallel_config` 会根据这些参数组成：

   ```python
   tp_size=self.tensor_parallel_size
   pp_size=self.pipeline_parallel_size
   cp_size=self.context_parallel_size
   ```

   然后得到整体 `world_size`。

4. `MpiPoolSession` 具体做的是启动 `MPIPoolExecutor`  
   它内部会执行：

   ```python
   MPIPoolExecutor(max_workers=self.n_workers)
   ```

   所以如果 `world_size = 4`，就会启动 4 个 MPI worker/rank 来跑模型的不同并行分片。



mpi4py 是 MPI 的 Python 绑定，让 Python 代码也能调用 MPI 的进程管理和通信能力。TensorRT-LLM 里用它来创建 MPI worker、获取 rank/world size、做进程间通信。



world_size=1 只表示**模型并行 rank 数是 1**，所以 BaseLLM 不会为了多 GPU 创建 MPI session。但 PyTorch backend 的 GenerationExecutor 还有一套**单 GPU worker 进程机制**，它也可能用 mpi4py 创建 MpiPoolSession，所以你还是会看到 MPI 创建的是单 GPU worker 的 MpiPoolSession，不是多 GPU 并行的 MpiPoolSession。



**图 2：下载/解析 checkpoint，到创建 GenerationExecutor**

```mermaid
sequenceDiagram
    participant _TorchLLM
    participant CachedModelLoader
    participant ModelLoader
    participant DefaultInputProcessor
    participant GenerationExecutor

    _TorchLLM->>CachedModelLoader: resolve model path / HF cache
    CachedModelLoader-->>_TorchLLM: hf_model_dir
    _TorchLLM->>ModelLoader: load tokenizer / HF config / generation config
    _TorchLLM->>DefaultInputProcessor: create_input_processor(hf_model_dir, tokenizer)
    DefaultInputProcessor-->>_TorchLLM: tokenizer + prompt preprocessor
    _TorchLLM->>GenerationExecutor: create(hf_model_dir, tokenizer, llm_args)
    Note over GenerationExecutor: 下一图开始进入 PyTorch executor 创建流程
```

**图 3：模型类分派，决定具体调用哪个模型**
```mermaid
sequenceDiagram
    participant GenerationExecutor
    participant BaseCheckpointLoader
    participant ModelConfig
    participant AutoModelForCausalLM
    participant LlamaForCausalLM
    participant Qwen3ForCausalLM
    participant DeepseekV3ForCausalLM
    participant Qwen3MoeForCausalLM
    participant PyTorchModelEngine

    GenerationExecutor->>BaseCheckpointLoader: load_config(hf_model_dir)
    BaseCheckpointLoader-->>ModelConfig: pretrained_config.architectures
    ModelConfig->>AutoModelForCausalLM: _resolve_class(config)
    Note over AutoModelForCausalLM: 查 MODEL_CLASS_MAPPING[architectures[0]]；<br/>映射来自 @register_auto_model("...")

    alt TinyLlama / Llama architecture
        AutoModelForCausalLM-->>LlamaForCausalLM: selected class
    else Qwen3 architecture
        AutoModelForCausalLM-->>Qwen3ForCausalLM: selected class
    else DeepSeek V3 architecture
        AutoModelForCausalLM-->>DeepseekV3ForCausalLM: selected class
    else Qwen3 MoE architecture
        AutoModelForCausalLM-->>Qwen3MoeForCausalLM: selected class
    end

    GenerationExecutor->>PyTorchModelEngine: construct concrete model module
    GenerationExecutor->>BaseCheckpointLoader: load_weights()
    BaseCheckpointLoader-->>PyTorchModelEngine: mapped tensors loaded into selected model
    Note over BaseCheckpointLoader: Checkpoint Loading 自定义点：checkpoint_loader 或 register_checkpoint_loader(format)
```

**图 4：Executor 运行时资源构建**
```mermaid
sequenceDiagram
    participant GenerationExecutor
    participant PyTorchModelEngine
    participant Distributed
    participant AttentionRuntimeFeatures
    participant KVCacheManagerV2
    participant RequestScheduler
    participant GuidedDecoder
    participant Drafter
    participant Sampler
    participant KvCacheConnectorManager
    participant PyExecutor

    GenerationExecutor->>Distributed: Distributed.get(Mapping)
    Note over Distributed: Multi-GPU/Multi-Node：tensor/pipeline/context/expert parallel
    GenerationExecutor->>AttentionRuntimeFeatures: build(chunked_prefill, cache_reuse, spec_draft_tokens)
    GenerationExecutor->>KVCacheManagerV2: allocate paged KV cache
    Note over KVCacheManagerV2: KV Cache Management：paged blocks, block reuse,<br/>partial reuse, memory estimation, KV events
    GenerationExecutor->>RequestScheduler: create(ctx_chunk_config, kv_cache_manager)
    Note over RequestScheduler: In-Flight Batching + Chunked Prefill：<br/>context/generation 同 iteration 调度；长 prompt 拆 chunk
    alt guided_decoding_backend enabled
        GenerationExecutor->>GuidedDecoder: create xgrammar / llguidance decoder
    end
    alt speculative_config enabled
        GenerationExecutor->>Drafter: create EAGLE/MTP/NGram/PARD/DFlash drafter
    end
    GenerationExecutor->>Sampler: instantiate sampler
    alt kv_connector_config enabled
        GenerationExecutor->>KvCacheConnectorManager: dynamic import connector worker/scheduler
        Note over KvCacheConnectorManager: 自定义 plugin 点：connector_module,<br/>connector_worker_class, connector_scheduler_class,<br/>register_forward_pass_callable()
    end
    GenerationExecutor->>PyExecutor: create(model_engine, scheduler, resources, sampler, drafter)
    PyExecutor-->>GenerationExecutor: start_worker()
```

**图 5：generate 请求进入调度循环**
```mermaid
sequenceDiagram
    participant LLM
    participant DefaultInputProcessor
    participant GenerationExecutor
    participant PyExecutor
    participant RequestScheduler
    participant KVCacheManagerV2
    participant GuidedDecoder
    participant Drafter
    participant PyTorchModelEngine

    LLM->>DefaultInputProcessor: tokenize prompts
    DefaultInputProcessor-->>LLM: prompt_token_ids
    LLM->>GenerationExecutor: generate_async(prompt_token_ids, SamplingParams)
    GenerationExecutor->>PyExecutor: submit(GenerationRequest)
    PyExecutor->>RequestScheduler: schedule_request(active_requests, inflight_request_ids)
    RequestScheduler->>KVCacheManagerV2: prepare_context / resize_context / resize_generation
    Note over RequestScheduler,KVCacheManagerV2: In-Flight Batching & Paged Attention：<br/>context 与 generation 混排，KV blocks 按 page 管理和复用
    RequestScheduler-->>PyExecutor: ScheduledRequests(context_requests, generation_requests)
    alt guided decoding enabled
        PyExecutor->>GuidedDecoder: add_batch / execute constraints
    end
    alt speculative decoding enabled
        PyExecutor->>Drafter: prepare draft tokens
    end
    PyExecutor->>PyTorchModelEngine: forward(ScheduledRequests, ResourceManager)
```

**图 6：PyTorchModelEngine 到具体模型层**
```mermaid
sequenceDiagram
    participant PyTorchModelEngine
    participant LlamaForCausalLM
    participant LlamaModel
    participant LlamaDecoderLayer
    participant Qwen3MoeForCausalLM
    participant Qwen3MoEModel
    participant Qwen3MoEDecoderLayer
    participant DeepseekV3ForCausalLM
    participant DeepseekV3Model
    participant DeepseekV3DecoderLayer

    alt selected model is LlamaForCausalLM
        PyTorchModelEngine->>LlamaForCausalLM: forward(model_inputs)
        LlamaForCausalLM->>LlamaModel: forward(...)
        LlamaModel->>LlamaDecoderLayer: layer loop
    else selected model is Qwen3MoeForCausalLM
        PyTorchModelEngine->>Qwen3MoeForCausalLM: forward(model_inputs)
        Qwen3MoeForCausalLM->>Qwen3MoEModel: forward(...)
        Qwen3MoEModel->>Qwen3MoEDecoderLayer: layer loop
    else selected model is DeepseekV3ForCausalLM
        PyTorchModelEngine->>DeepseekV3ForCausalLM: forward(model_inputs)
        DeepseekV3ForCausalLM->>DeepseekV3Model: forward(...)
        DeepseekV3Model->>DeepseekV3DecoderLayer: layer loop
    end
```

**图 7：DecoderLayer 到 Attention / MLP / MoE / LoRA**
```mermaid
sequenceDiagram
    participant LlamaDecoderLayer
    participant Qwen3MoEDecoderLayer
    participant DeepseekV3DecoderLayer
    participant RMSNorm
    participant Attention
    participant GatedMLP
    participant MoE
    participant LoraLayer

    alt dense decoder layer
        LlamaDecoderLayer->>RMSNorm: input_layernorm(hidden_states)
        LlamaDecoderLayer->>Attention: self_attn(position_ids, hidden_states, attn_metadata, lora_params)
        Attention->>LoraLayer: optional qkv/o_proj adapter delta
        Note over LoraLayer: LoRA Support：多 adapter 由 lora_config + PEFT cache 管理
        LlamaDecoderLayer->>RMSNorm: post_attention_layernorm(hidden_states)
        LlamaDecoderLayer->>GatedMLP: mlp(hidden_states, lora_params)
    else MoE decoder layer
        Qwen3MoEDecoderLayer->>RMSNorm: input_layernorm(hidden_states)
        Qwen3MoEDecoderLayer->>Attention: self_attn(...)
        Qwen3MoEDecoderLayer->>MoE: route tokens to experts
        Note over MoE: Expert Parallelism + MoE kernels；FP8/FP4 MoE 路径在这里
    else MLA / DeepSeek style layer
        DeepseekV3DecoderLayer->>RMSNorm: input_layernorm(hidden_states)
        DeepseekV3DecoderLayer->>Attention: MLA attention(...)
        DeepseekV3DecoderLayer->>MoE: routed/shared experts
    end
```

**图 8：Attention 到 CUDA / Triton / Quant kernels**
```mermaid
sequenceDiagram
    participant Attention
    participant TrtllmAttention
    participant FlashInferAttention
    participant VanillaAttention
    participant TorchCustomOp
    participant TritonKernel
    participant CudaKernel

    Attention->>TrtllmAttention: default attn_backend="TRTLLM"
    alt attn_backend == TRTLLM
        TrtllmAttention->>TorchCustomOp: torch.ops.trtllm.attention(...)
        TorchCustomOp->>CudaKernel: C++/CUDA FMHA, paged KV, MLA ops
        Note over CudaKernel: Paged Attention, FP8 KV, FP4/NVFP4 output,<br/>mla_rope_append_paged_kv_assign_q, load_paged_kv_cache_for_mla
    else attn_backend == FLASHINFER
        Attention->>FlashInferAttention: forward(...)
        FlashInferAttention->>TorchCustomOp: flashinfer paged cache ops
    else attn_backend == VANILLA
        Attention->>VanillaAttention: torch native attention path
    end

    alt sparse / DSA attention enabled
        TrtllmAttention->>TritonKernel: @triton.jit sparse kernels
        TritonKernel-->>TrtllmAttention: sparse attention result
    end

    alt quantization enabled
        Attention->>TorchCustomOp: fp8/fp4 quantized GEMM / BMM / MoE ops
        TorchCustomOp->>CudaKernel: cpp/tensorrt_llm/thop + kernels
    end
```

**图 9：logits 到采样，再回到用户输出**
```mermaid
sequenceDiagram
    participant PyTorchModelEngine
    participant LMHead
    participant LogitsProcessor
    participant PyExecutor
    participant Sampler
    participant GuidedDecoder
    participant GenerationExecutor
    participant LLM

    PyTorchModelEngine->>LMHead: project hidden_states to vocab logits
    PyTorchModelEngine->>LogitsProcessor: gather context/generation logits
    LogitsProcessor-->>PyExecutor: logits
    PyExecutor->>Sampler: sample_async(logits, SamplingParams)
    alt guided decoding enabled
        Sampler->>GuidedDecoder: apply grammar/stop/bad-word constraints
    end
    Sampler-->>PyExecutor: next token ids
    PyExecutor-->>GenerationExecutor: GenerationResult
    GenerationExecutor-->>LLM: RequestOutput
    LLM-->>LLM: detokenize text for caller
```

**图 10：Disaggregated Serving 旁路，接在图 5/6 之间**
```mermaid
sequenceDiagram
    participant PyExecutor
    participant RequestScheduler
    participant KVCacheTransceiver
    participant KVCacheManagerV2
    participant PyTorchModelEngine

    PyExecutor->>RequestScheduler: schedule context worker requests
    PyExecutor->>PyTorchModelEngine: prefill forward()
    PyExecutor->>KVCacheTransceiver: respond_and_send_async(context KV)
    KVCacheTransceiver->>KVCacheManagerV2: commit_blocks_for_reuse()

    PyExecutor->>KVCacheTransceiver: request_and_receive_async(generation KV)
    KVCacheTransceiver-->>PyExecutor: KV transfer complete
    PyExecutor->>KVCacheManagerV2: prepare generation blocks
    PyExecutor->>PyTorchModelEngine: decode forward()
    Note over KVCacheTransceiver: Disaggregated Serving：prefill/context 与 decode/generation<br/>可分布到不同 GPU/节点，KV cache 通过 transceiver/connector 交换
```

这套图的连续数据流就是：

`LLM.__init__` -> `_TorchLLM._build_model` -> `GenerationExecutor.create` -> `BaseCheckpointLoader + AutoModelForCausalLM` -> `PyTorchModelEngine` -> `PyExecutor` -> `RequestScheduler + KVCacheManagerV2` -> `具体 *ForCausalLM` -> `*Model` -> `*DecoderLayer` -> `Attention / MLP / MoE` -> `torch.ops.trtllm / Triton / CUDA` -> `LMHead` -> `Sampler` -> `RequestOutput`。





# TensorRT-LLM C++ / TensorRT Backend

`from tensorrt_llm._tensorrt_engine import LLM. `

**图 1：入口到 TensorRT Backend 初始化**

```mermaid
sequenceDiagram
    participant LLM
    participant _TrtLLM
    participant BaseLLM
    participant TrtLlmArgs
    participant MpiPoolSession

    LLM->>_TrtLLM: __init__(model, backend="tensorrt" or _tensorrt_engine.LLM)
    _TrtLLM->>BaseLLM: __init__(...)
    BaseLLM->>TrtLlmArgs: parse + validate TensorRT backend args
    Note over TrtLlmArgs: TensorRT backend 配置入口：build_config,<br/>scheduler_config, batching_type, kv_cache_config,<br/>peft_cache_config, decoding_config,<br/>cache_transceiver_config, parallel_config
    alt world_size > 1
        BaseLLM->>MpiPoolSession: start worker processes
        Note over MpiPoolSession: Multi-GPU/Multi-Node：TP/PP/CP 等并行 rank 编排
    end
    BaseLLM->>_TrtLLM: _build_model()
```

1.   BaseLLM 按 backend 的值选择对应后端的参数类，"pytorch" 对应 TorchLlmArgs，"_autodeploy" 对应 AutoDeployLlmArgs，其余值对应 TrtLlmArgs。

     `backend == "pytorch"`：使用 `TorchLlmArgs`，走 `PyExecutor -> PyTorchModelEngine -> 具体 PyTorch 模型类 -> torch.ops/Triton/CUDA`。  
     `backend == "_autodeploy"`：使用 `AutoDeployLlmArgs`，走 AutoDeploy shim，对 PyTorch 模型做 graph transform / torch.export 后执行。  
     `else`：使用 `TrtLlmArgs`，走 TensorRT engine build/load -> C++ Executor -> BatchManager -> TensorRT Plugin/CUDA。

```
if backend == "pytorch":
    logger.info("Using LLM with PyTorch backend")
    llm_args_cls = TorchLlmArgs
    if self._orchestrator_type == "ray" or mpi_disabled():
        self._orchestrator_type = "ray"
        os.environ["TLLM_DISABLE_MPI"] = "1"
        # Propagate to args construction
        kwargs["orchestrator_type"] = "ray"

elif backend == '_autodeploy':
    logger.info("Using LLM with AutoDeploy backend")
    from .._torch.auto_deploy.llm_args import \
        LlmArgs as AutoDeployLlmArgs
    llm_args_cls = AutoDeployLlmArgs
else:
    logger.info("Using LLM with TensorRT backend")
    llm_args_cls = TrtLlmArgs
```



2.当模型需要多 GPU 推理时，`BaseLLM` 会启动一组 MPI worker 进程。TP/PP/CP 这些并行方式已经体现在 `parallel_config.world_size` 里，这里做的是按这个数量创建 worker。

1. `world_size > 1` 表示需要多进程/多 GPU  
   代码里对应的是：

   ```python
   if self.args.parallel_config.is_multi_gpu:
   ```

   也就是只要 `parallel_config` 判断当前不是单 GPU，就进入多 GPU 初始化逻辑。

2. `BaseLLM->>MpiPoolSession: start worker processes` 对应创建 MPI session  
   代码位置在 `BaseLLM.__init__`：

   ```python
   self.mpi_session = MpiPoolSession(
       n_workers=self.args.parallel_config.world_size
   )
   ```

   这里的 `n_workers` 就是总 rank 数。

3. `TP/PP/CP 等并行 rank 编排` 来自 `parallel_config`  
   `parallel_config` 会根据这些参数组成：

   ```python
   tp_size=self.tensor_parallel_size
   pp_size=self.pipeline_parallel_size
   cp_size=self.context_parallel_size
   ```

   然后得到整体 `world_size`。

4. `MpiPoolSession` 具体做的是启动 `MPIPoolExecutor`  
   它内部会执行：

   ```python
   MPIPoolExecutor(max_workers=self.n_workers)
   ```

   所以如果 `world_size = 4`，就会启动 4 个 MPI worker/rank 来跑模型的不同并行分片。



**图 2：构建/加载 TensorRT Engine**

```mermaid
sequenceDiagram
    participant _TrtLLM
    participant CachedModelLoader
    participant ModelLoader
    participant EngineConfig
    participant DefaultInputProcessor
    participant ExecutorConfig

    _TrtLLM->>CachedModelLoader: resolve/build TensorRT engine
    Note over CachedModelLoader: Legacy engine-build workflow：<br/>HF/NeMo/custom checkpoint -> TRT-LLM checkpoint -> TensorRT engine
    CachedModelLoader-->>_TrtLLM: engine_dir + hf_model_dir

    _TrtLLM->>ModelLoader: load tokenizer / HF config / generation config
    _TrtLLM->>DefaultInputProcessor: create_input_processor(hf_model_dir, tokenizer)
    DefaultInputProcessor-->>_TrtLLM: tokenizer + prompt preprocessor

    _TrtLLM->>EngineConfig: from_json_file(engine_dir/config.json)
    EngineConfig-->>_TrtLLM: build_config + pretrained_config
    _TrtLLM->>ExecutorConfig: create max_batch_size/max_num_tokens/max_seq_len
    Note over ExecutorConfig: 这里把 in-flight batching, chunked context,<br/>KV cache, LoRA/PEFT, guided decoding,<br/>disagg transceiver, parallel_config 传给 C++ runtime
```

**图 3：创建 C++ Executor**
```mermaid
sequenceDiagram
    participant _TrtLLM
    participant GenerationExecutor
    participant ExecutorConfig
    participant Executor
    participant PostprocWorkerConfig

    _TrtLLM->>GenerationExecutor: create(engine_dir, ExecutorConfig)
    GenerationExecutor->>PostprocWorkerConfig: configure detokenization workers
    GenerationExecutor->>Executor: construct C++ Executor(engine_dir, ExecutorConfig)
    Note over Executor: 后续请求进入 C++ Executor / BatchManager / TensorRT Engine
    Executor-->>GenerationExecutor: ready
    GenerationExecutor-->>_TrtLLM: executor ready
```

**图 4：generate 请求提交到 C++ Executor**
```mermaid
sequenceDiagram
    participant LLM
    participant DefaultInputProcessor
    participant GenerationExecutor
    participant Executor
    participant Request

    LLM->>DefaultInputProcessor: tokenize prompts
    DefaultInputProcessor-->>LLM: prompt_token_ids
    LLM->>GenerationExecutor: generate_async(prompt_token_ids, SamplingParams)
    GenerationExecutor->>Request: create GenerationRequest / executor request
    GenerationExecutor->>Executor: enqueueRequest(Request)
    Note over Executor: Python 层到这里基本完成提交；<br/>核心调度和执行进入 C++ runtime
```

**图 5：C++ In-Flight Batching 调度**
```mermaid
sequenceDiagram
    participant Executor
    participant BatchManager
    participant CapacityScheduler
    participant KVCacheManager
    participant LlmRequest
    participant ScheduledRequests

    Executor->>BatchManager: fetch queued requests
    BatchManager->>LlmRequest: update request states
    BatchManager->>CapacityScheduler: select fitting requests
    CapacityScheduler->>KVCacheManager: check/allocate KV blocks
    Note over CapacityScheduler,KVCacheManager: In-Flight Batching & Paged KV：<br/>context/generation 混合调度；KV blocks 按 page 管理；<br/>可暂停、复用、淘汰或延后请求
    KVCacheManager-->>CapacityScheduler: capacity result
    CapacityScheduler-->>BatchManager: fitting / paused requests
    BatchManager-->>ScheduledRequests: build scheduled batch
```

**图 6：Chunked Prefill / LoRA / Guided / Spec Decode 准备**
```mermaid
sequenceDiagram
    participant BatchManager
    participant ScheduledRequests
    participant KVCacheManager
    participant PeftCacheManager
    participant GuidedDecoder
    participant DecodingInput

    BatchManager->>KVCacheManager: prepare context/generation KV cache
    alt enable_chunked_context
        BatchManager->>ScheduledRequests: split long context into chunks
        Note over ScheduledRequests: Chunked Prefill：长 prompt 被拆成多个 context chunk
    end

    alt LoRA / PEFT enabled
        BatchManager->>PeftCacheManager: load/select adapter pages
        Note over PeftCacheManager: LoRA Support：多 adapter 权重缓存和请求级选择
    end

    alt guided decoding enabled
        BatchManager->>GuidedDecoder: build constraints for batch
        Note over GuidedDecoder: Guided Decoding：stop words, bad words, grammar/schema constraints
    end

    BatchManager->>DecodingInput: prepare runtime inputs
```

**图 7：TensorRT Engine Forward**
```mermaid
sequenceDiagram
    participant BatchManager
    participant ScheduledRequests
    participant Executor
    participant TensorRTEngine
    participant TensorRTPlugin
    participant CudaKernel
    participant KVCacheManager

    BatchManager->>Executor: execute scheduled batch
    Executor->>TensorRTEngine: enqueue forward bindings
    TensorRTEngine->>TensorRTPlugin: attention / GEMM / LoRA / MoE plugins
    TensorRTPlugin->>KVCacheManager: read/write paged KV cache metadata
    TensorRTPlugin->>CudaKernel: launch CUDA kernels
    Note over TensorRTPlugin,CudaKernel: Advanced Quantization：FP8 / FP4 / INT4 等 plugin/kernel<br/>Paged Attention / FMHA / GEMM / MoE 在这里执行
    CudaKernel-->>TensorRTEngine: logits / hidden states
    TensorRTEngine-->>Executor: forward outputs
```

**图 8：C++ Decoder / Sampling**
```mermaid
sequenceDiagram
    participant Executor
    participant Decoder
    participant Sampling
    participant GuidedDecoder
    participant KVCacheManager
    participant Response

    Executor->>Decoder: decode step with logits
    Decoder->>Sampling: apply SamplingConfig
    alt guided decoding enabled
        Sampling->>GuidedDecoder: apply constraints / bitmask
    end
    Sampling-->>Decoder: next token ids
    Decoder->>KVCacheManager: update sequence state / KV ownership
    Decoder-->>Response: partial or final response
    Response-->>Executor: ready to return
```

**图 9：返回结果到 Python LLM**
```mermaid
sequenceDiagram
    participant Executor
    participant GenerationExecutor
    participant RequestOutput
    participant LLM
    participant DefaultInputProcessor

    Executor-->>GenerationExecutor: awaitResponses()
    GenerationExecutor->>RequestOutput: convert executor response
    GenerationExecutor->>DefaultInputProcessor: detokenize token ids
    DefaultInputProcessor-->>RequestOutput: generated text
    RequestOutput-->>LLM: output
    LLM-->>LLM: yield / return result to caller
```

**图 10：Disaggregated Serving 旁路**
```mermaid
sequenceDiagram
    participant BatchManager
    participant KVCacheTransferManager
    participant CacheTransceiver
    participant KVCacheManager
    participant Executor

    alt context / prefill worker
        BatchManager->>Executor: run context forward
        Executor->>KVCacheTransferManager: collect KV blocks
        KVCacheTransferManager->>CacheTransceiver: send KV cache
        CacheTransceiver-->>KVCacheTransferManager: transfer status
        KVCacheManager->>KVCacheManager: mark reusable / releasable blocks
    else generation / decode worker
        BatchManager->>CacheTransceiver: request KV cache
        CacheTransceiver-->>KVCacheTransferManager: receive KV cache
        KVCacheTransferManager->>KVCacheManager: register received blocks
        BatchManager->>Executor: run decode forward
    end
    Note over CacheTransceiver: Disaggregated Serving：context 与 generation<br/>可以分布在不同 GPU/节点，KV cache 通过传输层交换
```

**C++ Backend 连续主链**
```text
LLM
 -> _TrtLLM
 -> BaseLLM
 -> TrtLlmArgs
 -> CachedModelLoader
 -> EngineConfig / ExecutorConfig
 -> GenerationExecutor
 -> C++ Executor
 -> BatchManager
 -> CapacityScheduler
 -> C++ KVCacheManager
 -> ScheduledRequests
 -> TensorRT Engine
 -> TensorRT Plugin
 -> CUDA kernels
 -> Decoder / Sampling
 -> Response
 -> RequestOutput
```

和 PyTorch backend 最大的图上区别是：

```text
PyTorch backend:
会出现 AutoModelForCausalLM、LlamaForCausalLM、Qwen3ForCausalLM、PyTorchModelEngine。

C++ / TensorRT backend:
不会在运行时逐层调用 LlamaForCausalLM.forward；
模型已经固化进 TensorRT Engine，运行时主要看到 Executor、BatchManager、KVCacheManager、TensorRTPlugin、CudaKernel。
```
