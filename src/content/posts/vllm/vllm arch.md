---
title: "vllm Arch"
published: 2026-04-27
description: "vllm Arch"
image: ""
tags: ["vllm","vllm Arch"]
category: vllm
draft: false
lang: ""
createdAt: "2026-04-27T17:07:09.230.909869266Z"
---

```
vllm
├── assets 
│   ├── __init__.py 
│   ├── audio.py
│   ├── base.py
│   ├── image.py
│   └── video.py
```









```
vllm
├── benchmarks
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── create_txt_slices_dataset.py
│   │   ├── datasets.py
│   │   └── utils.py
│   ├── lib
│   │   ├── __init__.py
│   │   ├── endpoint_request_func.py
│   │   ├── ready_checker.py
│   │   └── utils.py
│   ├── sweep
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   ├── param_sweep.py
│   │   ├── plot.py
│   │   ├── plot_pareto.py
│   │   ├── serve.py
│   │   ├── serve_workload.py
│   │   ├── server.py
│   │   ├── startup.py
│   │   └── utils.py
│   ├── __init__.py
│   ├── latency.py
│   ├── mm_processor.py
│   ├── plot.py
│   ├── serve.py
│   ├── startup.py
│   └── throughput.py
├── compilation
│   ├── passes
│   │   ├── fusion
│   │   │   ├── __init__.py
│   │   │   ├── act_quant_fusion.py
│   │   │   ├── allreduce_rms_fusion.py
│   │   │   ├── attn_quant_fusion.py
│   │   │   ├── collective_fusion.py
│   │   │   ├── matcher_utils.py
│   │   │   ├── minimax_qk_norm_fusion.py
│   │   │   ├── mla_attn_quant_fusion.py
│   │   │   ├── qk_norm_rope_fusion.py
│   │   │   ├── rms_quant_fusion.py
│   │   │   ├── rocm_aiter_fusion.py
│   │   │   ├── rope_kvcache_fusion.py
│   │   │   └── sequence_parallelism.py
│   │   ├── ir
│   │   │   ├── __init__.py
│   │   │   └── lowering_pass.py
│   │   ├── utility
│   │   │   ├── __init__.py
│   │   │   ├── fix_functionalization.py
│   │   │   ├── noop_elimination.py
│   │   │   ├── post_cleanup.py
│   │   │   ├── scatter_split_replace.py
│   │   │   └── split_coalescing.py
│   │   ├── __init__.py
│   │   ├── fx_utils.py
│   │   ├── inductor_pass.py
│   │   ├── pass_manager.py
│   │   └── vllm_inductor_pass.py
│   ├── __init__.py
│   ├── backends.py
│   ├── base_static_graph.py
│   ├── caching.py
│   ├── codegen.py
│   ├── compiler_interface.py
│   ├── counter.py
│   ├── cuda_graph.py
│   ├── decorators.py
│   ├── monitor.py
│   ├── partition_rules.py
│   ├── piecewise_backend.py
│   └── wrapper.py
├── config
│   ├── __init__.py
│   ├── attention.py
│   ├── cache.py
│   ├── compilation.py
│   ├── device.py
│   ├── ec_transfer.py
│   ├── kernel.py
│   ├── kv_events.py
│   ├── kv_transfer.py
│   ├── load.py
│   ├── lora.py
│   ├── mamba.py
│   ├── model.py
│   ├── model_arch.py
│   ├── multimodal.py
│   ├── observability.py
│   ├── offload.py
│   ├── parallel.py
│   ├── pooler.py
│   ├── profiler.py
│   ├── quantization.py
│   ├── reasoning.py
│   ├── scheduler.py
│   ├── speculative.py
│   ├── speech_to_text.py
│   ├── structured_outputs.py
│   ├── utils.py
│   ├── vllm.py
│   └── weight_transfer.py
├── device_allocator
│   ├── __init__.py
│   └── cumem.py
├── distributed
│   ├── device_communicators
│   │   ├── __init__.py
│   │   ├── all2all.py
│   │   ├── all_reduce_utils.py
│   │   ├── base_device_communicator.py
│   │   ├── cpu_communicator.py
│   │   ├── cuda_communicator.py
│   │   ├── cuda_wrapper.py
│   │   ├── custom_all_reduce.py
│   │   ├── flashinfer_all_reduce.py
│   │   ├── mnnvl_compat.py
│   │   ├── pynccl.py
│   │   ├── pynccl_allocator.py
│   │   ├── pynccl_wrapper.py
│   │   ├── quick_all_reduce.py
│   │   ├── ray_communicator.py
│   │   ├── shm_broadcast.py
│   │   ├── shm_object_storage.py
│   │   ├── symm_mem.py
│   │   └── xpu_communicator.py
│   ├── ec_transfer
│   │   ├── ec_connector
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── example_connector.py
│   │   │   └── factory.py
│   │   ├── __init__.py
│   │   └── ec_transfer_state.py
│   ├── elastic_ep
│   │   ├── __init__.py
│   │   ├── elastic_execute.py
│   │   ├── elastic_state.py
│   │   └── standby_state.py
│   ├── eplb
│   │   ├── policy
│   │   │   ├── __init__.py
│   │   │   ├── abstract.py
│   │   │   └── default.py
│   │   ├── __init__.py
│   │   ├── async_worker.py
│   │   ├── eplb_communicator.py
│   │   ├── eplb_state.py
│   │   ├── eplb_utils.py
│   │   └── rebalance_execute.py
│   ├── kv_transfer
│   │   ├── kv_connector
│   │   │   ├── v1
│   │   │   │   ├── hf3fs
│   │   │   │   │   ├── utils
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   ├── common.py
│   │   │   │   │   │   ├── gather_scatter_helper.py
│   │   │   │   │   │   ├── hf3fs_mock_client.py
│   │   │   │   │   │   └── hf3fs_utils.cpp
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── hf3fs_client.py
│   │   │   │   │   ├── hf3fs_connector.py
│   │   │   │   │   └── hf3fs_metadata_server.py
│   │   │   │   ├── lmcache_integration
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── multi_process_adapter.py
│   │   │   │   │   ├── utils.py
│   │   │   │   │   └── vllm_v1_adapter.py
│   │   │   │   ├── mooncake
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── mooncake_connector.py
│   │   │   │   │   └── mooncake_utils.py
│   │   │   │   ├── moriio
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── moriio_common.py
│   │   │   │   │   ├── moriio_connector.py
│   │   │   │   │   └── moriio_engine.py
│   │   │   │   ├── nixl
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── connector.py
│   │   │   │   │   ├── metadata.py
│   │   │   │   │   ├── scheduler.py
│   │   │   │   │   ├── stats.py
│   │   │   │   │   ├── utils.py
│   │   │   │   │   └── worker.py
│   │   │   │   ├── offloading
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── common.py
│   │   │   │   │   ├── metrics.py
│   │   │   │   │   ├── scheduler.py
│   │   │   │   │   └── worker.py
│   │   │   │   ├── p2p
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── p2p_nccl_connector.py
│   │   │   │   │   ├── p2p_nccl_engine.py
│   │   │   │   │   └── tensor_memory_pool.py
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py
│   │   │   │   ├── decode_bench_connector.py
│   │   │   │   ├── example_connector.py
│   │   │   │   ├── example_hidden_states_connector.py
│   │   │   │   ├── flexkv_connector.py
│   │   │   │   ├── lmcache_connector.py
│   │   │   │   ├── lmcache_mp_connector.py
│   │   │   │   ├── metrics.py
│   │   │   │   ├── multi_connector.py
│   │   │   │   ├── offloading_connector.py
│   │   │   │   ├── simple_cpu_offload_connector.py
│   │   │   │   └── ssm_conv_transfer_utils.py
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── factory.py
│   │   │   └── utils.py
│   │   ├── __init__.py
│   │   ├── disagg_prefill_workflow.jpg
│   │   ├── kv_transfer_state.py
│   │   └── README.md
│   ├── weight_transfer
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── factory.py
│   │   ├── ipc_engine.py
│   │   ├── nccl_engine.py
│   │   └── packed_tensor.py
│   ├── __init__.py
│   ├── communication_op.py
│   ├── kv_events.py
│   ├── nixl_utils.py
│   ├── parallel_state.py
│   ├── stateless_coordinator.py
│   └── utils.py
├── engine
│   ├── __init__.py
│   ├── arg_utils.py
│   ├── async_llm_engine.py
│   ├── llm_engine.py
│   └── protocol.py
├── entrypoints
│   ├── anthropic
│   │   ├── __init__.py
│   │   ├── api_router.py
│   │   ├── protocol.py
│   │   └── serving.py
│   ├── cli
│   │   ├── benchmark
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── latency.py
│   │   │   ├── main.py
│   │   │   ├── mm_processor.py
│   │   │   ├── serve.py
│   │   │   ├── startup.py
│   │   │   ├── sweep.py
│   │   │   └── throughput.py
│   │   ├── __init__.py
│   │   ├── collect_env.py
│   │   ├── launch.py
│   │   ├── main.py
│   │   ├── openai.py
│   │   ├── run_batch.py
│   │   ├── serve.py
│   │   └── types.py
│   ├── mcp
│   │   ├── __init__.py
│   │   ├── tool.py
│   │   └── tool_server.py
│   ├── openai
│   │   ├── chat_completion
│   │   │   ├── __init__.py
│   │   │   ├── api_router.py
│   │   │   ├── batch_serving.py
│   │   │   ├── protocol.py
│   │   │   ├── serving.py
│   │   │   └── stream_harmony.py
│   │   ├── completion
│   │   │   ├── __init__.py
│   │   │   ├── api_router.py
│   │   │   ├── protocol.py
│   │   │   └── serving.py
│   │   ├── engine
│   │   │   ├── __init__.py
│   │   │   ├── protocol.py
│   │   │   └── serving.py
│   │   ├── generate
│   │   │   ├── __init__.py
│   │   │   ├── api_router.py
│   │   │   └── factories.py
│   │   ├── generative_scoring
│   │   │   ├── __init__.py
│   │   │   ├── api_router.py
│   │   │   └── serving.py
│   │   ├── models
│   │   │   ├── __init__.py
│   │   │   ├── api_router.py
│   │   │   ├── protocol.py
│   │   │   └── serving.py
│   │   ├── parser
│   │   │   ├── __init__.py
│   │   │   ├── harmony_utils.py
│   │   │   └── responses_parser.py
│   │   ├── realtime
│   │   │   ├── __init__.py
│   │   │   ├── api_router.py
│   │   │   ├── connection.py
│   │   │   ├── metrics.py
│   │   │   ├── protocol.py
│   │   │   └── serving.py
│   │   ├── responses
│   │   │   ├── __init__.py
│   │   │   ├── api_router.py
│   │   │   ├── context.py
│   │   │   ├── harmony.py
│   │   │   ├── protocol.py
│   │   │   ├── serving.py
│   │   │   ├── streaming_events.py
│   │   │   └── utils.py
│   │   ├── speech_to_text
│   │   │   ├── __init__.py
│   │   │   ├── api_router.py
│   │   │   ├── protocol.py
│   │   │   ├── serving.py
│   │   │   └── speech_to_text.py
│   │   ├── __init__.py
│   │   ├── api_server.py
│   │   ├── cli_args.py
│   │   ├── orca_metrics.py
│   │   ├── run_batch.py
│   │   ├── server_utils.py
│   │   └── utils.py
│   ├── pooling
│   │   ├── base
│   │   │   ├── __init__.py
│   │   │   ├── io_processor.py
│   │   │   ├── protocol.py
│   │   │   └── serving.py
│   │   ├── classify
│   │   │   ├── __init__.py
│   │   │   ├── api_router.py
│   │   │   ├── io_processor.py
│   │   │   ├── protocol.py
│   │   │   └── serving.py
│   │   ├── embed
│   │   │   ├── __init__.py
│   │   │   ├── api_router.py
│   │   │   ├── io_processor.py
│   │   │   ├── protocol.py
│   │   │   └── serving.py
│   │   ├── pooling
│   │   │   ├── __init__.py
│   │   │   ├── api_router.py
│   │   │   ├── io_processor.py
│   │   │   ├── protocol.py
│   │   │   └── serving.py
│   │   ├── scoring
│   │   │   ├── __init__.py
│   │   │   ├── api_router.py
│   │   │   ├── io_processor.py
│   │   │   ├── protocol.py
│   │   │   ├── serving.py
│   │   │   ├── typing.py
│   │   │   └── utils.py
│   │   ├── __init__.py
│   │   ├── factories.py
│   │   ├── typing.py
│   │   └── utils.py
│   ├── sagemaker
│   │   ├── __init__.py
│   │   └── api_router.py
│   ├── serve
│   │   ├── cache
│   │   │   ├── __init__.py
│   │   │   └── api_router.py
│   │   ├── disagg
│   │   │   ├── __init__.py
│   │   │   ├── api_router.py
│   │   │   ├── mm_serde.py
│   │   │   ├── protocol.py
│   │   │   └── serving.py
│   │   ├── elastic_ep
│   │   │   ├── __init__.py
│   │   │   ├── api_router.py
│   │   │   └── middleware.py
│   │   ├── instrumentator
│   │   │   ├── static
│   │   │   │   ├── swagger-ui-bundle.js
│   │   │   │   └── swagger-ui.css
│   │   │   ├── __init__.py
│   │   │   ├── basic.py
│   │   │   ├── health.py
│   │   │   ├── metrics.py
│   │   │   ├── offline_docs.py
│   │   │   └── server_info.py
│   │   ├── lora
│   │   │   ├── __init__.py
│   │   │   ├── api_router.py
│   │   │   └── protocol.py
│   │   ├── profile
│   │   │   ├── __init__.py
│   │   │   └── api_router.py
│   │   ├── render
│   │   │   ├── __init__.py
│   │   │   ├── api_router.py
│   │   │   └── serving.py
│   │   ├── rlhf
│   │   │   ├── __init__.py
│   │   │   └── api_router.py
│   │   ├── rpc
│   │   │   ├── __init__.py
│   │   │   └── api_router.py
│   │   ├── sleep
│   │   │   ├── __init__.py
│   │   │   └── api_router.py
│   │   ├── tokenize
│   │   │   ├── __init__.py
│   │   │   ├── api_router.py
│   │   │   ├── protocol.py
│   │   │   └── serving.py
│   │   └── __init__.py
│   ├── __init__.py
│   ├── api_server.py
│   ├── chat_utils.py
│   ├── constants.py
│   ├── grpc_server.py
│   ├── launcher.py
│   ├── llm.py
│   ├── logger.py
│   ├── ssl.py
│   └── utils.py
├── inputs
│   ├── __init__.py
│   ├── engine.py
│   ├── llm.py
│   └── preprocess.py
├── ir
│   ├── ops
│   │   ├── __init__.py
│   │   └── layernorm.py
│   ├── __init__.py
│   ├── op.py
│   ├── tolerances.py
│   └── util.py
├── kernels
│   ├── helion
│   │   ├── configs
│   │   │   └── silu_mul_fp8
│   │   │       ├── nvidia_h100.json
│   │   │       └── nvidia_h200.json
│   │   ├── ops
│   │   │   ├── __init__.py
│   │   │   └── silu_mul_fp8.py
│   │   ├── __init__.py
│   │   ├── config_manager.py
│   │   ├── register.py
│   │   └── utils.py
│   ├── __init__.py
│   ├── aiter_ops.py
│   ├── oink_ops.py
│   ├── vllm_c.py
│   └── xpu_ops.py
├── logging_utils
│   ├── __init__.py
│   ├── access_log_filter.py
│   ├── dump_input.py
│   ├── formatter.py
│   ├── lazy.py
│   ├── log_time.py
│   └── torch_tensor.py
├── lora
│   ├── layers
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── base_linear.py
│   │   ├── column_parallel_linear.py
│   │   ├── fused_moe.py
│   │   ├── logits_processor.py
│   │   ├── replicated_linear.py
│   │   ├── row_parallel_linear.py
│   │   ├── utils.py
│   │   └── vocal_parallel_embedding.py
│   ├── ops
│   │   ├── torch_ops
│   │   │   ├── __init__.py
│   │   │   └── lora_ops.py
│   │   ├── triton_ops
│   │   │   ├── __init__.py
│   │   │   ├── fp8_kernel_utils.py
│   │   │   ├── fused_moe_lora_fp8_op.py
│   │   │   ├── fused_moe_lora_op.py
│   │   │   ├── kernel_utils.py
│   │   │   ├── lora_expand_fp8_op.py
│   │   │   ├── lora_expand_op.py
│   │   │   ├── lora_kernel_metadata.py
│   │   │   ├── lora_shrink_fp8_op.py
│   │   │   ├── lora_shrink_op.py
│   │   │   ├── README_TUNING.md
│   │   │   └── utils.py
│   │   ├── xpu_ops
│   │   │   ├── __init__.py
│   │   │   └── lora_ops.py
│   │   └── __init__.py
│   ├── punica_wrapper
│   │   ├── __init__.py
│   │   ├── punica_base.py
│   │   ├── punica_cpu.py
│   │   ├── punica_gpu.py
│   │   ├── punica_selector.py
│   │   ├── punica_xpu.py
│   │   └── utils.py
│   ├── __init__.py
│   ├── lora_model.py
│   ├── lora_weights.py
│   ├── model_manager.py
│   ├── peft_helper.py
│   ├── request.py
│   ├── resolver.py
│   ├── utils.py
│   └── worker_manager.py
├── model_executor
│   ├── kernels
│   │   ├── linear
│   │   │   ├── mixed_precision
│   │   │   │   ├── __init__.py
│   │   │   │   ├── allspark.py
│   │   │   │   ├── conch.py
│   │   │   │   ├── cpu.py
│   │   │   │   ├── cutlass.py
│   │   │   │   ├── dynamic_4bit.py
│   │   │   │   ├── exllama.py
│   │   │   │   ├── machete.py
│   │   │   │   ├── marlin.py
│   │   │   │   ├── MPLinearKernel.py
│   │   │   │   ├── triton_w4a16.py
│   │   │   │   └── xpu.py
│   │   │   ├── mxfp8
│   │   │   │   ├── __init__.py
│   │   │   │   ├── emulation.py
│   │   │   │   ├── flashinfer.py
│   │   │   │   ├── marlin.py
│   │   │   │   ├── Mxfp8LinearKernel.py
│   │   │   │   └── xpu.py
│   │   │   ├── nvfp4
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py
│   │   │   │   ├── cutlass.py
│   │   │   │   ├── emulation.py
│   │   │   │   ├── fbgemm.py
│   │   │   │   ├── flashinfer.py
│   │   │   │   └── marlin.py
│   │   │   ├── scaled_mm
│   │   │   │   ├── __init__.py
│   │   │   │   ├── aiter.py
│   │   │   │   ├── BlockScaledMMLinearKernel.py
│   │   │   │   ├── cpu.py
│   │   │   │   ├── cutlass.py
│   │   │   │   ├── deep_gemm.py
│   │   │   │   ├── flashinfer.py
│   │   │   │   ├── marlin.py
│   │   │   │   ├── pytorch.py
│   │   │   │   ├── rocm.py
│   │   │   │   ├── ScaledMMLinearKernel.py
│   │   │   │   ├── triton.py
│   │   │   │   └── xpu.py
│   │   │   ├── __init__.py
│   │   │   └── base.py
│   │   └── __init__.py
│   ├── layers
│   │   ├── attention
│   │   │   ├── __init__.py
│   │   │   ├── attention.py
│   │   │   ├── chunked_local_attention.py
│   │   │   ├── cross_attention.py
│   │   │   ├── encoder_only_attention.py
│   │   │   ├── kv_transfer_utils.py
│   │   │   ├── mla_attention.py
│   │   │   ├── mm_encoder_attention.py
│   │   │   └── static_sink_attention.py
│   │   ├── fla
│   │   │   ├── ops
│   │   │   │   ├── __init__.py
│   │   │   │   ├── chunk.py
│   │   │   │   ├── chunk_delta_h.py
│   │   │   │   ├── chunk_o.py
│   │   │   │   ├── chunk_scaled_dot_kkt.py
│   │   │   │   ├── cumsum.py
│   │   │   │   ├── fused_gdn_prefill_post_conv.py
│   │   │   │   ├── fused_recurrent.py
│   │   │   │   ├── fused_sigmoid_gating.py
│   │   │   │   ├── index.py
│   │   │   │   ├── kda.py
│   │   │   │   ├── l2norm.py
│   │   │   │   ├── layernorm_guard.py
│   │   │   │   ├── op.py
│   │   │   │   ├── solve_tril.py
│   │   │   │   ├── utils.py
│   │   │   │   └── wy_fast.py
│   │   │   └── __init__.py
│   │   ├── fused_moe
│   │   │   ├── configs
│   │   │   │   ├── E=1,N=14336,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json
│   │   │   │   ├── E=1,N=14336,device_name=NVIDIA_A100-SXM4-80GB.json
│   │   │   │   ├── E=1,N=1792,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json
│   │   │   │   ├── E=1,N=1792,device_name=NVIDIA_A100-SXM4-80GB.json
│   │   │   │   ├── E=1,N=1792,device_name=NVIDIA_H100_80GB_HBM3,dtype=int8_w8a16.json
│   │   │   │   ├── E=1,N=3072,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json
│   │   │   │   ├── E=1,N=3072,device_name=NVIDIA_H100_80GB_HBM3,dtype=int8_w8a16.json
│   │   │   │   ├── E=1,N=3072,device_name=NVIDIA_H100_80GB_HBM3.json
│   │   │   │   ├── E=1,N=3072,device_name=NVIDIA_H200,dtype=int8_w8a16.json
│   │   │   │   ├── E=1,N=3584,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json
│   │   │   │   ├── E=1,N=3584,device_name=NVIDIA_A100-SXM4-80GB.json
│   │   │   │   ├── E=1,N=3584,device_name=NVIDIA_H100_80GB_HBM3,dtype=int8_w8a16.json
│   │   │   │   ├── E=1,N=7168,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json
│   │   │   │   ├── E=1,N=7168,device_name=NVIDIA_A100-SXM4-80GB.json
│   │   │   │   ├── E=1,N=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=int8_w8a16.json
│   │   │   │   ├── E=128,N=1024,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8.json
│   │   │   │   ├── E=128,N=1024,device_name=AMD_Instinct_MI300X.json
│   │   │   │   ├── E=128,N=1024,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8.json
│   │   │   │   ├── E=128,N=1024,device_name=NVIDIA_H100,dtype=fp8_w8a8.json
│   │   │   │   ├── E=128,N=1024,device_name=NVIDIA_H200,dtype=fp8_w8a8.json
│   │   │   │   ├── E=128,N=1024,device_name=NVIDIA_H200.json
│   │   │   │   ├── E=128,N=1856,device_name=NVIDIA_B200.json
│   │   │   │   ├── E=128,N=1856,device_name=NVIDIA_H100_80GB_HBM3.json
│   │   │   │   ├── E=128,N=1856,device_name=NVIDIA_H200,dtype=fp8_w8a8.json
│   │   │   │   ├── E=128,N=1856,device_name=NVIDIA_L40S.json
│   │   │   │   ├── E=128,N=192,device_name=NVIDIA_A100-SXM4-80GB.json
│   │   │   │   ├── E=128,N=192,device_name=NVIDIA_H100_80GB_HBM3.json
│   │   │   │   ├── E=128,N=192,device_name=NVIDIA_H20-3e.json
│   │   │   │   ├── E=128,N=192,device_name=NVIDIA_H20.json
│   │   │   │   ├── E=128,N=192,device_name=NVIDIA_H200.json
│   │   │   │   ├── E=128,N=192,device_name=NVIDIA_H800,dtype=fp8_w8a8.json
│   │   │   │   ├── E=128,N=352,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json
│   │   │   │   ├── E=128,N=384,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=128,N=384,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=128,N=384,device_name=NVIDIA_GB200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=128,N=384,device_name=NVIDIA_H100_80GB_HBM3.json
│   │   │   │   ├── E=128,N=384,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=128,N=384,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=128,N=384,device_name=NVIDIA_H20-3e.json
│   │   │   │   ├── E=128,N=384,device_name=NVIDIA_H20.json
│   │   │   │   ├── E=128,N=384,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=128,N=384,device_name=NVIDIA_H200.json
│   │   │   │   ├── E=128,N=512,device_name=AMD_Radeon_R9700,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=128,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8.json
│   │   │   │   ├── E=128,N=512,device_name=NVIDIA_B200.json
│   │   │   │   ├── E=128,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json
│   │   │   │   ├── E=128,N=512,device_name=NVIDIA_H100_80GB_HBM3.json
│   │   │   │   ├── E=128,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=128,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8.json
│   │   │   │   ├── E=128,N=512,device_name=NVIDIA_H200.json
│   │   │   │   ├── E=128,N=704,device_name=NVIDIA_B200,dtype=fp8_w8a8.json
│   │   │   │   ├── E=128,N=704,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json
│   │   │   │   ├── E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json
│   │   │   │   ├── E=128,N=768,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=128,N=768,device_name=AMD_Instinct_MI308X.json
│   │   │   │   ├── E=128,N=768,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=128,N=768,device_name=NVIDIA_B200.json
│   │   │   │   ├── E=128,N=768,device_name=NVIDIA_GB200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=128,N=768,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=128,N=768,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=128,N=768,device_name=NVIDIA_H20.json
│   │   │   │   ├── E=128,N=768,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=128,N=768,device_name=NVIDIA_H200.json
│   │   │   │   ├── E=128,N=768,device_name=Radeon_8060S_Graphics,dtype=int4_w4a16.json
│   │   │   │   ├── E=128,N=8960,device_name=NVIDIA_H100_80GB_HBM3,dtype=bf16.json
│   │   │   │   ├── E=128,N=8960,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json
│   │   │   │   ├── E=128,N=928,device_name=NVIDIA_H100_80GB_HBM3.json
│   │   │   │   ├── E=128,N=928,device_name=NVIDIA_L40S.json
│   │   │   │   ├── E=128,N=96,device_name=NVIDIA_H20.json
│   │   │   │   ├── E=128,N=96,device_name=NVIDIA_H200,dtype=fp8_w8a8.json
│   │   │   │   ├── E=128,N=96,device_name=NVIDIA_H200.json
│   │   │   │   ├── E=129,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json
│   │   │   │   ├── E=16,N=1024,device_name=AMD_Instinct_MI300X.json
│   │   │   │   ├── E=16,N=1024,device_name=NVIDIA_B200,dtype=fp8_w8a8.json
│   │   │   │   ├── E=16,N=1024,device_name=NVIDIA_B200.json
│   │   │   │   ├── E=16,N=1024,device_name=NVIDIA_H100.json
│   │   │   │   ├── E=16,N=1024,device_name=NVIDIA_H200,dtype=fp8_w8a8.json
│   │   │   │   ├── E=16,N=1024,device_name=NVIDIA_H200.json
│   │   │   │   ├── E=16,N=1344,device_name=NVIDIA_A100-SXM4-40GB.json
│   │   │   │   ├── E=16,N=1344,device_name=NVIDIA_A100-SXM4-80GB.json
│   │   │   │   ├── E=16,N=1344,device_name=NVIDIA_H100_80GB_HBM3.json
│   │   │   │   ├── E=16,N=14336,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json
│   │   │   │   ├── E=16,N=14336,device_name=NVIDIA_A100-SXM4-80GB.json
│   │   │   │   ├── E=16,N=14336,device_name=NVIDIA_H100_80GB_HBM3,dtype=int8_w8a16.json
│   │   │   │   ├── E=16,N=1792,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json
│   │   │   │   ├── E=16,N=1792,device_name=NVIDIA_A100-SXM4-80GB.json
│   │   │   │   ├── E=16,N=1792,device_name=NVIDIA_H100_80GB_HBM3,dtype=int8_w8a16.json
│   │   │   │   ├── E=16,N=1792,device_name=NVIDIA_H100_80GB_HBM3.json
│   │   │   │   ├── E=16,N=2048,device_name=NVIDIA_H200,dtype=fp8_w8a8.json
│   │   │   │   ├── E=16,N=2048,device_name=NVIDIA_H200.json
│   │   │   │   ├── E=16,N=2688,device_name=NVIDIA_A100-SXM4-80GB.json
│   │   │   │   ├── E=16,N=2688,device_name=NVIDIA_H100_80GB_HBM3.json
│   │   │   │   ├── E=16,N=3072,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json
│   │   │   │   ├── E=16,N=3072,device_name=NVIDIA_H100_80GB_HBM3,dtype=float8.json
│   │   │   │   ├── E=16,N=3072,device_name=NVIDIA_H100_80GB_HBM3,dtype=int8_w8a16.json
│   │   │   │   ├── E=16,N=3072,device_name=NVIDIA_H200,dtype=int8_w8a16.json
│   │   │   │   ├── E=16,N=3200,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json
│   │   │   │   ├── E=16,N=3584,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json
│   │   │   │   ├── E=16,N=3584,device_name=NVIDIA_A100-SXM4-80GB.json
│   │   │   │   ├── E=16,N=3584,device_name=NVIDIA_H100_80GB_HBM3,dtype=int8_w8a16.json
│   │   │   │   ├── E=16,N=4096,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=16,N=4096,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=16,N=6400,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json
│   │   │   │   ├── E=16,N=7168,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a16.json
│   │   │   │   ├── E=16,N=7168,device_name=NVIDIA_A100-SXM4-80GB.json
│   │   │   │   ├── E=16,N=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=float8.json
│   │   │   │   ├── E=16,N=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=int8_w8a16.json
│   │   │   │   ├── E=16,N=800,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json
│   │   │   │   ├── E=160,N=192,device_name=AMD_Instinct_MI300X.json
│   │   │   │   ├── E=160,N=192,device_name=AMD_Instinct_MI350_OAM,dtype=fp8_w8a8.json
│   │   │   │   ├── E=160,N=192,device_name=NVIDIA_A800-SXM4-80GB.json
│   │   │   │   ├── E=160,N=192,device_name=NVIDIA_B300_SXM6_AC,dtype=fp8_w8a8.json
│   │   │   │   ├── E=160,N=192,device_name=NVIDIA_H20-3e.json
│   │   │   │   ├── E=160,N=192,device_name=NVIDIA_H200,dtype=fp8_w8a8.json
│   │   │   │   ├── E=160,N=320,device_name=NVIDIA_H20-3e.json
│   │   │   │   ├── E=160,N=384,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8.json
│   │   │   │   ├── E=160,N=384,device_name=AMD_Instinct_MI350_OAM,dtype=fp8_w8a8.json
│   │   │   │   ├── E=160,N=384,device_name=AMD_Instinct_MI355_OAM,dtype=fp8_w8a8.json
│   │   │   │   ├── E=160,N=384,device_name=NVIDIA_B200,dtype=fp8_w8a8.json
│   │   │   │   ├── E=160,N=384,device_name=NVIDIA_B300_SXM6_AC,dtype=fp8_w8a8.json
│   │   │   │   ├── E=160,N=640,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=160,N=640,device_name=NVIDIA_GB200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=160,N=640,device_name=NVIDIA_H100,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=160,N=768,device_name=NVIDIA_B300_SXM6_AC,dtype=fp8_w8a8.json
│   │   │   │   ├── E=20,N=1536,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition,dtype=fp8_w8a8.json
│   │   │   │   ├── E=20,N=2560,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=20,N=2560,device_name=NVIDIA_GB200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=20,N=2560,device_name=NVIDIA_H100,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=20,N=2560,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=256,N=1024,device_name=AMD_Instinct_MI325_OAM,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=256,N=1024,device_name=AMD_Instinct_MI325X,block_shape=[128,128].json
│   │   │   │   ├── E=256,N=128,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=256,N=128,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a8.json
│   │   │   │   ├── E=256,N=128,device_name=NVIDIA_A800-SXM4-80GB,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=256,N=128,device_name=NVIDIA_A800-SXM4-80GB,dtype=int8_w8a8.json
│   │   │   │   ├── E=256,N=128,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=256,N=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=256,N=128,device_name=NVIDIA_L20Y,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=256,N=256,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=256,N=256,device_name=AMD_Instinct_MI325_OAM,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=256,N=256,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=256,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=256,N=256,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=256,N=256,device_name=NVIDIA_H20,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=256,N=256,device_name=NVIDIA_H20-3e,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=256,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=256,N=256,device_name=NVIDIA_L20,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=256,N=384,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=256,N=384,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=256,N=512,device_name=AMD_Instinct_MI325_OAM,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=256,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=256,N=512,device_name=NVIDIA_H100_80GB_HBM3.json
│   │   │   │   ├── E=256,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=256,N=512,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=256,N=64,device_name=NVIDIA_A800-SXM4-80GB.json
│   │   │   │   ├── E=32,N=1408,device_name=NVIDIA_B200.json
│   │   │   │   ├── E=32,N=1792,device_name=NVIDIA_H100_80GB_HBM3.json
│   │   │   │   ├── E=32,N=2048,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=32,N=2048,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=384,N=128,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=384,N=128,device_name=NVIDIA_GB200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=384,N=128,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=384,N=256,device_name=AMD_Instinct_MI350_OAM,dtype=int4_w4a16.json
│   │   │   │   ├── E=384,N=256,device_name=AMD_Instinct_MI350X,dtype=int4_w4a16.json
│   │   │   │   ├── E=384,N=256,device_name=AMD_Instinct_MI355_OAM,dtype=int4_w4a16.json
│   │   │   │   ├── E=384,N=256,device_name=AMD_Instinct_MI355X,dtype=int4_w4a16.json
│   │   │   │   ├── E=384,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=384,N=256,device_name=NVIDIA_GB200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=40,N=1536,device_name=NVIDIA_B200,dtype=fp8_w8a8.json
│   │   │   │   ├── E=40,N=2560,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=40,N=2560,device_name=NVIDIA_GB200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=40,N=2560,device_name=NVIDIA_H100,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=512,N=128,device_name=NVIDIA_A100-SXM4-80GB.json
│   │   │   │   ├── E=512,N=128,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=512,N=128,device_name=NVIDIA_B200.json
│   │   │   │   ├── E=512,N=128,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json
│   │   │   │   ├── E=512,N=128,device_name=NVIDIA_H100_80GB_HBM3.json
│   │   │   │   ├── E=512,N=128,device_name=NVIDIA_H20-3e.json
│   │   │   │   ├── E=512,N=128,device_name=NVIDIA_H200.json
│   │   │   │   ├── E=512,N=1344,device_name=NVIDIA_B200.json
│   │   │   │   ├── E=512,N=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=512,N=256,device_name=NVIDIA_B200.json
│   │   │   │   ├── E=512,N=256,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json
│   │   │   │   ├── E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=512,N=256,device_name=NVIDIA_H100_80GB_HBM3.json
│   │   │   │   ├── E=512,N=256,device_name=NVIDIA_H20-3e.json
│   │   │   │   ├── E=512,N=256,device_name=NVIDIA_H200.json
│   │   │   │   ├── E=512,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=512,N=512,device_name=NVIDIA_B200.json
│   │   │   │   ├── E=512,N=512,device_name=NVIDIA_GB200,dtype=fp8_w8a8.json
│   │   │   │   ├── E=512,N=512,device_name=NVIDIA_H100_80GB_HBM3.json
│   │   │   │   ├── E=512,N=512,device_name=NVIDIA_H20-3e.json
│   │   │   │   ├── E=512,N=512,device_name=NVIDIA_H200.json
│   │   │   │   ├── E=512,N=64,device_name=NVIDIA_A100-SXM4-80GB.json
│   │   │   │   ├── E=512,N=64,device_name=NVIDIA_B200.json
│   │   │   │   ├── E=512,N=64,device_name=NVIDIA_H20-3e.json
│   │   │   │   ├── E=512,N=64,device_name=NVIDIA_H200.json
│   │   │   │   ├── E=512,N=672,device_name=NVIDIA_B200.json
│   │   │   │   ├── E=60,N=1408,device_name=AMD_Instinct_MI300X.json
│   │   │   │   ├── E=60,N=176,device_name=AMD_Instinct_MI300X.json
│   │   │   │   ├── E=60,N=352,device_name=AMD_Instinct_MI300X.json
│   │   │   │   ├── E=60,N=704,device_name=AMD_Instinct_MI300X.json
│   │   │   │   ├── E=62,N=128,device_name=AMD_Instinct_MI300X.json
│   │   │   │   ├── E=62,N=256,device_name=AMD_Instinct_MI300X.json
│   │   │   │   ├── E=62,N=256,device_name=NVIDIA_H100_80GB_HBM3.json
│   │   │   │   ├── E=62,N=512,device_name=AMD_Instinct_MI300X.json
│   │   │   │   ├── E=62,N=512,device_name=NVIDIA_H100_80GB_HBM3.json
│   │   │   │   ├── E=64,N=1280,device_name=NVIDIA_A100-SXM4-80GB.json
│   │   │   │   ├── E=64,N=1280,device_name=NVIDIA_A800-SXM4-80GB.json
│   │   │   │   ├── E=64,N=1280,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json
│   │   │   │   ├── E=64,N=1280,device_name=NVIDIA_H100_80GB_HBM3.json
│   │   │   │   ├── E=64,N=1280,device_name=NVIDIA_H200,dtype=fp8_w8a8.json
│   │   │   │   ├── E=64,N=1280,device_name=NVIDIA_H200.json
│   │   │   │   ├── E=64,N=1408,device_name=NVIDIA_B200.json
│   │   │   │   ├── E=64,N=1536,device_name=NVIDIA_H100_80GB_HBM3.json
│   │   │   │   ├── E=64,N=1536,device_name=NVIDIA_H20,dtype=fp8_w8a8.json
│   │   │   │   ├── E=64,N=1536,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=64,N=2560,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json
│   │   │   │   ├── E=64,N=2560,device_name=NVIDIA_H200,dtype=fp8_w8a8.json
│   │   │   │   ├── E=64,N=2560,device_name=NVIDIA_H200.json
│   │   │   │   ├── E=64,N=3072,device_name=NVIDIA_H20,dtype=fp8_w8a8.json
│   │   │   │   ├── E=64,N=3072,device_name=NVIDIA_H20.json
│   │   │   │   ├── E=64,N=320,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json
│   │   │   │   ├── E=64,N=320,device_name=NVIDIA_H100_80GB_HBM3.json
│   │   │   │   ├── E=64,N=320,device_name=NVIDIA_H200,dtype=fp8_w8a8.json
│   │   │   │   ├── E=64,N=320,device_name=NVIDIA_H200.json
│   │   │   │   ├── E=64,N=384,device_name=NVIDIA_H20,dtype=fp8_w8a8.json
│   │   │   │   ├── E=64,N=384,device_name=NVIDIA_H20.json
│   │   │   │   ├── E=64,N=512,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=64,N=640,device_name=NVIDIA_A100-SXM4-80GB.json
│   │   │   │   ├── E=64,N=640,device_name=NVIDIA_A800-SXM4-80GB.json
│   │   │   │   ├── E=64,N=640,device_name=NVIDIA_GeForce_RTX_4090,dtype=fp8_w8a8.json
│   │   │   │   ├── E=64,N=640,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json
│   │   │   │   ├── E=64,N=640,device_name=NVIDIA_H100_80GB_HBM3.json
│   │   │   │   ├── E=64,N=640,device_name=NVIDIA_H200,dtype=fp8_w8a8.json
│   │   │   │   ├── E=64,N=640,device_name=NVIDIA_H200.json
│   │   │   │   ├── E=64,N=768,device_name=AMD_Radeon_R9700,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=64,N=768,device_name=NVIDIA_H100_PCIe,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=64,N=768,device_name=NVIDIA_H20,dtype=fp8_w8a8.json
│   │   │   │   ├── E=64,N=768,device_name=NVIDIA_H20.json
│   │   │   │   ├── E=64,N=896,device_name=NVIDIA_H20.json
│   │   │   │   ├── E=64,N=8960,device_name=NVIDIA_H100_80GB_HBM3,dtype=bf16.json
│   │   │   │   ├── E=64,N=8960,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json
│   │   │   │   ├── E=72,N=192,device_name=AMD_Instinct_MI300X.json
│   │   │   │   ├── E=72,N=384,device_name=AMD_Instinct_MI300X.json
│   │   │   │   ├── E=72,N=384,device_name=NVIDIA_H100_80GB_HBM3.json
│   │   │   │   ├── E=72,N=768,device_name=AMD_Instinct_MI300X.json
│   │   │   │   ├── E=72,N=768,device_name=NVIDIA_H100_80GB_HBM3.json
│   │   │   │   ├── E=8,N=14336,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8.json
│   │   │   │   ├── E=8,N=14336,device_name=AMD_Instinct_MI300X.json
│   │   │   │   ├── E=8,N=14336,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8.json
│   │   │   │   ├── E=8,N=14336,device_name=AMD_Instinct_MI325X.json
│   │   │   │   ├── E=8,N=14336,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json
│   │   │   │   ├── E=8,N=14336,device_name=NVIDIA_H200,dtype=fp8_w8a8.json
│   │   │   │   ├── E=8,N=14336,device_name=NVIDIA_H200.json
│   │   │   │   ├── E=8,N=16384,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8.json
│   │   │   │   ├── E=8,N=16384,device_name=AMD_Instinct_MI300X.json
│   │   │   │   ├── E=8,N=16384,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8.json
│   │   │   │   ├── E=8,N=16384,device_name=AMD_Instinct_MI325X.json
│   │   │   │   ├── E=8,N=1792,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8.json
│   │   │   │   ├── E=8,N=1792,device_name=AMD_Instinct_MI300X.json
│   │   │   │   ├── E=8,N=1792,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8.json
│   │   │   │   ├── E=8,N=1792,device_name=AMD_Instinct_MI325X.json
│   │   │   │   ├── E=8,N=1792,device_name=NVIDIA_A100-SXM4-40GB.json
│   │   │   │   ├── E=8,N=1792,device_name=NVIDIA_A100-SXM4-80GB.json
│   │   │   │   ├── E=8,N=1792,device_name=NVIDIA_H100_80GB_HBM3.json
│   │   │   │   ├── E=8,N=1792,device_name=NVIDIA_H200,dtype=fp8_w8a8.json
│   │   │   │   ├── E=8,N=1792,device_name=NVIDIA_H200.json
│   │   │   │   ├── E=8,N=2048,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8.json
│   │   │   │   ├── E=8,N=2048,device_name=AMD_Instinct_MI300X.json
│   │   │   │   ├── E=8,N=2048,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8.json
│   │   │   │   ├── E=8,N=2048,device_name=AMD_Instinct_MI325X.json
│   │   │   │   ├── E=8,N=2048,device_name=NVIDIA_A100-SXM4-80GB.json
│   │   │   │   ├── E=8,N=2048,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json
│   │   │   │   ├── E=8,N=2048,device_name=NVIDIA_H100_80GB_HBM3.json
│   │   │   │   ├── E=8,N=2048,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   ├── E=8,N=2048,device_name=NVIDIA_H200,dtype=fp8_w8a8.json
│   │   │   │   ├── E=8,N=2048,device_name=NVIDIA_H200.json
│   │   │   │   ├── E=8,N=3584,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8.json
│   │   │   │   ├── E=8,N=3584,device_name=AMD_Instinct_MI300X.json
│   │   │   │   ├── E=8,N=3584,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8.json
│   │   │   │   ├── E=8,N=3584,device_name=AMD_Instinct_MI325X.json
│   │   │   │   ├── E=8,N=3584,device_name=NVIDIA_A100-SXM4-40GB.json
│   │   │   │   ├── E=8,N=3584,device_name=NVIDIA_A100-SXM4-80GB.json
│   │   │   │   ├── E=8,N=3584,device_name=NVIDIA_GeForce_RTX_4090,dtype=fp8_w8a8.json
│   │   │   │   ├── E=8,N=3584,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json
│   │   │   │   ├── E=8,N=3584,device_name=NVIDIA_H100_80GB_HBM3.json
│   │   │   │   ├── E=8,N=3584,device_name=NVIDIA_H200,dtype=fp8_w8a8.json
│   │   │   │   ├── E=8,N=3584,device_name=NVIDIA_H200.json
│   │   │   │   ├── E=8,N=3584,device_name=NVIDIA_L40S.json
│   │   │   │   ├── E=8,N=4096,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8.json
│   │   │   │   ├── E=8,N=4096,device_name=AMD_Instinct_MI300X.json
│   │   │   │   ├── E=8,N=4096,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8.json
│   │   │   │   ├── E=8,N=4096,device_name=AMD_Instinct_MI325X.json
│   │   │   │   ├── E=8,N=4096,device_name=NVIDIA_A100-SXM4-80GB.json
│   │   │   │   ├── E=8,N=4096,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json
│   │   │   │   ├── E=8,N=4096,device_name=NVIDIA_H100_80GB_HBM3.json
│   │   │   │   ├── E=8,N=4096,device_name=NVIDIA_H200,dtype=fp8_w8a8.json
│   │   │   │   ├── E=8,N=4096,device_name=NVIDIA_H200.json
│   │   │   │   ├── E=8,N=7168,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8.json
│   │   │   │   ├── E=8,N=7168,device_name=AMD_Instinct_MI300X.json
│   │   │   │   ├── E=8,N=7168,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8.json
│   │   │   │   ├── E=8,N=7168,device_name=AMD_Instinct_MI325X.json
│   │   │   │   ├── E=8,N=7168,device_name=NVIDIA_A100-SXM4-80GB.json
│   │   │   │   ├── E=8,N=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json
│   │   │   │   ├── E=8,N=7168,device_name=NVIDIA_H100_80GB_HBM3.json
│   │   │   │   ├── E=8,N=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8.json
│   │   │   │   ├── E=8,N=7168,device_name=NVIDIA_H200.json
│   │   │   │   ├── E=8,N=8192,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8.json
│   │   │   │   ├── E=8,N=8192,device_name=AMD_Instinct_MI300X.json
│   │   │   │   ├── E=8,N=8192,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8.json
│   │   │   │   ├── E=8,N=8192,device_name=AMD_Instinct_MI325X.json
│   │   │   │   ├── E=8,N=8192,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json
│   │   │   │   ├── E=8,N=8192,device_name=NVIDIA_H200,dtype=fp8_w8a8.json
│   │   │   │   └── README
│   │   │   ├── experts
│   │   │   │   ├── __init__.py
│   │   │   │   ├── batched_deep_gemm_moe.py
│   │   │   │   ├── cutlass_moe.py
│   │   │   │   ├── deep_gemm_moe.py
│   │   │   │   ├── flashinfer_cutedsl_batched_moe.py
│   │   │   │   ├── flashinfer_cutedsl_moe.py
│   │   │   │   ├── gpt_oss_triton_kernels_moe.py
│   │   │   │   ├── nvfp4_emulation_moe.py
│   │   │   │   ├── ocp_mx_emulation_moe.py
│   │   │   │   ├── trtllm_bf16_moe.py
│   │   │   │   ├── trtllm_fp8_moe.py
│   │   │   │   ├── trtllm_mxfp4_moe.py
│   │   │   │   ├── trtllm_nvfp4_moe.py
│   │   │   │   └── xpu_moe.py
│   │   │   ├── oracle
│   │   │   │   ├── __init__.py
│   │   │   │   ├── fp8.py
│   │   │   │   ├── int8.py
│   │   │   │   ├── int_wna16.py
│   │   │   │   ├── mxfp4.py
│   │   │   │   ├── mxfp8.py
│   │   │   │   ├── nvfp4.py
│   │   │   │   └── unquantized.py
│   │   │   ├── prepare_finalize
│   │   │   │   ├── __init__.py
│   │   │   │   ├── batched.py
│   │   │   │   ├── deepep_ht.py
│   │   │   │   ├── deepep_ll.py
│   │   │   │   ├── flashinfer_nvlink_one_sided.py
│   │   │   │   ├── flashinfer_nvlink_two_sided.py
│   │   │   │   ├── mori.py
│   │   │   │   ├── naive_dp_ep.py
│   │   │   │   ├── nixl_ep.py
│   │   │   │   └── no_dp_ep.py
│   │   │   ├── router
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base_router.py
│   │   │   │   ├── custom_routing_router.py
│   │   │   │   ├── fused_moe_router.py
│   │   │   │   ├── fused_topk_bias_router.py
│   │   │   │   ├── fused_topk_router.py
│   │   │   │   ├── gate_linear.py
│   │   │   │   ├── grouped_topk_router.py
│   │   │   │   ├── router_factory.py
│   │   │   │   ├── routing_simulator_router.py
│   │   │   │   └── zero_expert_router.py
│   │   │   ├── runner
│   │   │   │   ├── __init__.py
│   │   │   │   ├── moe_runner.py
│   │   │   │   ├── moe_runner_interface.py
│   │   │   │   └── shared_experts.py
│   │   │   ├── __init__.py
│   │   │   ├── activation.py
│   │   │   ├── all2all_utils.py
│   │   │   ├── config.py
│   │   │   ├── cpu_fused_moe.py
│   │   │   ├── deep_gemm_utils.py
│   │   │   ├── fallback.py
│   │   │   ├── flashinfer_cutlass_moe.py
│   │   │   ├── fused_batched_moe.py
│   │   │   ├── fused_humming_moe.py
│   │   │   ├── fused_marlin_moe.py
│   │   │   ├── fused_moe.py
│   │   │   ├── fused_moe_method_base.py
│   │   │   ├── fused_moe_modular_method.py
│   │   │   ├── layer.py
│   │   │   ├── lora_context.py
│   │   │   ├── lora_experts_mixin.py
│   │   │   ├── modular_kernel.py
│   │   │   ├── moe_align_block_size.py
│   │   │   ├── moe_fused_mul_sum.py
│   │   │   ├── moe_permute_unpermute.py
│   │   │   ├── rocm_aiter_fused_moe.py
│   │   │   ├── routed_experts_capturer.py
│   │   │   ├── topk_weight_and_reduce.py
│   │   │   ├── triton_cutlass_moe.py
│   │   │   ├── triton_deep_gemm_moe.py
│   │   │   ├── unquantized_fused_moe_method.py
│   │   │   └── utils.py
│   │   ├── mamba
│   │   │   ├── ops
│   │   │   │   ├── __init__.py
│   │   │   │   ├── causal_conv1d.py
│   │   │   │   ├── layernorm_gated.py
│   │   │   │   ├── mamba_ssm.py
│   │   │   │   ├── ssd_bmm.py
│   │   │   │   ├── ssd_chunk_scan.py
│   │   │   │   ├── ssd_chunk_state.py
│   │   │   │   ├── ssd_combined.py
│   │   │   │   ├── ssd_state_passing.py
│   │   │   │   ├── ssu_dispatch.py
│   │   │   │   └── triton_helpers.py
│   │   │   ├── __init__.py
│   │   │   ├── abstract.py
│   │   │   ├── gdn_linear_attn.py
│   │   │   ├── lamport_workspace.py
│   │   │   ├── linear_attn.py
│   │   │   ├── mamba_mixer.py
│   │   │   ├── mamba_mixer2.py
│   │   │   ├── mamba_utils.py
│   │   │   └── short_conv.py
│   │   ├── pooler
│   │   │   ├── seqwise
│   │   │   │   ├── __init__.py
│   │   │   │   ├── heads.py
│   │   │   │   ├── methods.py
│   │   │   │   └── poolers.py
│   │   │   ├── tokwise
│   │   │   │   ├── __init__.py
│   │   │   │   ├── heads.py
│   │   │   │   ├── methods.py
│   │   │   │   └── poolers.py
│   │   │   ├── __init__.py
│   │   │   ├── abstract.py
│   │   │   ├── activations.py
│   │   │   ├── common.py
│   │   │   └── special.py
│   │   ├── quantization
│   │   │   ├── compressed_tensors
│   │   │   │   ├── compressed_tensors_moe
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── compressed_tensors_moe.py
│   │   │   │   │   ├── compressed_tensors_moe_w4a4_mxfp4.py
│   │   │   │   │   ├── compressed_tensors_moe_w4a4_nvfp4.py
│   │   │   │   │   ├── compressed_tensors_moe_w4a8_fp8.py
│   │   │   │   │   ├── compressed_tensors_moe_w4a8_int8.py
│   │   │   │   │   ├── compressed_tensors_moe_w8a8_fp8.py
│   │   │   │   │   ├── compressed_tensors_moe_w8a8_int8.py
│   │   │   │   │   ├── compressed_tensors_moe_w8a8_mxfp8.py
│   │   │   │   │   ├── compressed_tensors_moe_wna16.py
│   │   │   │   │   └── compressed_tensors_moe_wna16_marlin.py
│   │   │   │   ├── schemes
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── compressed_tensors_24.py
│   │   │   │   │   ├── compressed_tensors_scheme.py
│   │   │   │   │   ├── compressed_tensors_w4a16_mxfp4.py
│   │   │   │   │   ├── compressed_tensors_w4a16_nvfp4.py
│   │   │   │   │   ├── compressed_tensors_w4a4_nvfp4.py
│   │   │   │   │   ├── compressed_tensors_w4a8_fp8.py
│   │   │   │   │   ├── compressed_tensors_w4a8_int.py
│   │   │   │   │   ├── compressed_tensors_w8a16_fp8.py
│   │   │   │   │   ├── compressed_tensors_w8a8_fp8.py
│   │   │   │   │   ├── compressed_tensors_w8a8_int8.py
│   │   │   │   │   ├── compressed_tensors_w8a8_mxfp8.py
│   │   │   │   │   └── compressed_tensors_wNa16.py
│   │   │   │   ├── transform
│   │   │   │   │   ├── schemes
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── linear_qutlass_nvfp4.py
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── linear.py
│   │   │   │   │   ├── module.py
│   │   │   │   │   └── utils.py
│   │   │   │   ├── __init__.py
│   │   │   │   ├── compressed_tensors.py
│   │   │   │   ├── triton_scaled_mm.py
│   │   │   │   └── utils.py
│   │   │   ├── online
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py
│   │   │   │   ├── fp8.py
│   │   │   │   ├── int8.py
│   │   │   │   ├── moe_base.py
│   │   │   │   └── mxfp8.py
│   │   │   ├── quark
│   │   │   │   ├── schemes
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── quark_ocp_mx.py
│   │   │   │   │   ├── quark_scheme.py
│   │   │   │   │   ├── quark_w4a8_mxfp4_fp8.py
│   │   │   │   │   ├── quark_w8a8_fp8.py
│   │   │   │   │   └── quark_w8a8_int8.py
│   │   │   │   ├── __init__.py
│   │   │   │   ├── quark.py
│   │   │   │   ├── quark_moe.py
│   │   │   │   └── utils.py
│   │   │   ├── turboquant
│   │   │   │   ├── __init__.py
│   │   │   │   ├── centroids.py
│   │   │   │   ├── config.py
│   │   │   │   └── quantizer.py
│   │   │   ├── utils
│   │   │   │   ├── configs
│   │   │   │   │   ├── N=1024,K=2048,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=10240,K=5120,device_name=NVIDIA_L40S,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=12288,K=2048,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=12288,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=12288,K=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=1536,K=1536,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=1536,K=1536,device_name=AMD_Instinct_MI325_OAM,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=1536,K=1536,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=1536,K=1536,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=1536,K=1536,device_name=NVIDIA_A800-SXM4-80GB,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=1536,K=1536,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=1536,K=1536,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=1536,K=1536,device_name=NVIDIA_L20Y,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=1536,K=7168,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=1536,K=7168,device_name=AMD_Instinct_MI325_OAM,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=1536,K=7168,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=1536,K=7168,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=1536,K=7168,device_name=NVIDIA_A800-SXM4-80GB,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=1536,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=1536,K=7168,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=1536,K=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=1536,K=7168,device_name=NVIDIA_L20,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=1536,K=7168,device_name=NVIDIA_L20Y,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=2048,K=4096,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=2048,K=512,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=2048,K=512,device_name=AMD_Instinct_MI325_OAM,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=2048,K=512,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=2048,K=512,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=2048,K=512,device_name=NVIDIA_A800-SXM4-80GB,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=2048,K=512,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=2048,K=512,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=2048,K=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=2048,K=512,device_name=NVIDIA_L20Y,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=2112,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=2112,K=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=2304,K=7168,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=2304,K=7168,device_name=AMD_Instinct_MI325_OAM,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=2304,K=7168,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=2304,K=7168,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=2304,K=7168,device_name=NVIDIA_A800-SXM4-80GB,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=2304,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=2304,K=7168,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=2304,K=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=2304,K=7168,device_name=NVIDIA_L20Y,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=24576,K=1536,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=24576,K=1536,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=24576,K=7168,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=24576,K=7168,device_name=AMD_Instinct_MI325_OAM,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=24576,K=7168,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=24576,K=7168,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=24576,K=7168,device_name=NVIDIA_A800-SXM4-80GB,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=24576,K=7168,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=24576,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=24576,K=7168,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=24576,K=7168,device_name=NVIDIA_H20,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=24576,K=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=24576,K=7168,device_name=NVIDIA_L20,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=24576,K=7168,device_name=NVIDIA_L20Y,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=256,K=7168,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=256,K=7168,device_name=AMD_Instinct_MI325_OAM,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=256,K=7168,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=256,K=7168,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=256,K=7168,device_name=NVIDIA_A800-SXM4-80GB,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=256,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=256,K=7168,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=256,K=7168,device_name=NVIDIA_L20Y,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=3072,K=1536,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=3072,K=1536,device_name=AMD_Instinct_MI325_OAM,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=3072,K=1536,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=3072,K=1536,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=3072,K=1536,device_name=NVIDIA_H20,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=3072,K=1536,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=3072,K=1536,device_name=NVIDIA_L20,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=3072,K=7168,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=3072,K=7168,device_name=AMD_Instinct_MI325_OAM,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=3072,K=7168,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=3072,K=7168,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=3072,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=3072,K=7168,device_name=NVIDIA_H20,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=3072,K=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=3072,K=7168,device_name=NVIDIA_L20,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=32768,K=512,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=32768,K=512,device_name=AMD_Instinct_MI325_OAM,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=32768,K=512,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=32768,K=512,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=32768,K=512,device_name=NVIDIA_A800-SXM4-80GB,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=32768,K=512,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=32768,K=512,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=32768,K=512,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=32768,K=512,device_name=NVIDIA_H20,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=32768,K=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=32768,K=512,device_name=NVIDIA_L20,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=32768,K=512,device_name=NVIDIA_L20Y,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=36864,K=7168,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=36864,K=7168,device_name=AMD_Instinct_MI325_OAM,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=36864,K=7168,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=36864,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=36864,K=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=4096,K=512,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=4096,K=512,device_name=AMD_Instinct_MI325_OAM,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=4096,K=512,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=4096,K=512,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=4096,K=512,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=4096,K=512,device_name=NVIDIA_H20,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=4096,K=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=4096,K=512,device_name=NVIDIA_L20,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=4096,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=4096,K=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=4608,K=7168,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=4608,K=7168,device_name=AMD_Instinct_MI325_OAM,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=4608,K=7168,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=4608,K=7168,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=4608,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=4608,K=7168,device_name=NVIDIA_H20,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=4608,K=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=4608,K=7168,device_name=NVIDIA_L20,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=512,K=7168,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=512,K=7168,device_name=AMD_Instinct_MI325_OAM,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=512,K=7168,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=512,K=7168,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=512,K=7168,device_name=NVIDIA_H20,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=512,K=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=512,K=7168,device_name=NVIDIA_L20,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=5120,K=25600,device_name=NVIDIA_L40S,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=5120,K=8192,device_name=NVIDIA_L40S,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=51200,K=5120,device_name=NVIDIA_L40S,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=576,K=7168,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=576,K=7168,device_name=AMD_Instinct_MI325_OAM,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=576,K=7168,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=576,K=7168,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=576,K=7168,device_name=NVIDIA_A800-SXM4-80GB,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=576,K=7168,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=576,K=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=576,K=7168,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=576,K=7168,device_name=NVIDIA_H20,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=576,K=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=576,K=7168,device_name=NVIDIA_L20,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=576,K=7168,device_name=NVIDIA_L20Y,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=1024,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=1024,device_name=AMD_Instinct_MI325_OAM,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=1024,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=1024,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=1024,device_name=NVIDIA_A800-SXM4-80GB,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=1024,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=1024,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=1024,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=1024,device_name=NVIDIA_L20Y,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=1152,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=1152,device_name=AMD_Instinct_MI325_OAM,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=1152,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=1152,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=1152,device_name=NVIDIA_A800-SXM4-80GB,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=1152,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=1152,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=1152,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=1152,device_name=NVIDIA_L20Y,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=128,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=128,device_name=AMD_Instinct_MI325_OAM,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=128,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=128,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=128,device_name=NVIDIA_A800-SXM4-80GB,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=128,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=128,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=128,device_name=NVIDIA_L20Y,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=16384,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=16384,device_name=AMD_Instinct_MI325_OAM,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=16384,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=16384,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=16384,device_name=NVIDIA_A800-SXM4-80GB,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=16384,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=16384,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=16384,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=16384,device_name=NVIDIA_H20,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=16384,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=16384,device_name=NVIDIA_L20,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=16384,device_name=NVIDIA_L20Y,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=18432,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=18432,device_name=AMD_Instinct_MI325_OAM,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=18432,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=18432,device_name=NVIDIA_A100-SXM4-80GB,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=18432,device_name=NVIDIA_A800-SXM4-80GB,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=18432,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=18432,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=18432,device_name=NVIDIA_H20,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=18432,device_name=NVIDIA_H20,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=18432,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=18432,device_name=NVIDIA_L20,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=18432,device_name=NVIDIA_L20Y,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=2048,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=2048,device_name=AMD_Instinct_MI325_OAM,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=2048,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=2048,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=2048,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=2048,device_name=NVIDIA_H20,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=2048,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=2048,device_name=NVIDIA_L20,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=2304,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=2304,device_name=AMD_Instinct_MI325_OAM,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=2304,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=2304,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=2304,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=2304,device_name=NVIDIA_H20,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=2304,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=2304,device_name=NVIDIA_L20,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=256,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=256,device_name=AMD_Instinct_MI325_OAM,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=256,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=256,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=256,device_name=NVIDIA_H20,dtype=int8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=256,device_name=NVIDIA_L20,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=8192,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=8192,device_name=AMD_Instinct_MI325_OAM,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=7168,K=8192,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=8192,K=1536,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=8192,K=1536,device_name=AMD_Instinct_MI325_OAM,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=8192,K=1536,device_name=AMD_Instinct_MI325X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   ├── N=9216,K=2048,device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8,block_shape=[128,128].json
│   │   │   │   │   └── README.md
│   │   │   │   ├── __init__.py
│   │   │   │   ├── allspark_utils.py
│   │   │   │   ├── flashinfer_fp4_moe.py
│   │   │   │   ├── flashinfer_mxint4_moe.py
│   │   │   │   ├── flashinfer_utils.py
│   │   │   │   ├── fp8_utils.py
│   │   │   │   ├── gptq_utils.py
│   │   │   │   ├── humming_moe_utils.py
│   │   │   │   ├── int8_utils.py
│   │   │   │   ├── layer_utils.py
│   │   │   │   ├── machete_utils.py
│   │   │   │   ├── marlin_utils.py
│   │   │   │   ├── marlin_utils_fp4.py
│   │   │   │   ├── marlin_utils_fp8.py
│   │   │   │   ├── marlin_utils_test.py
│   │   │   │   ├── mxfp4_utils.py
│   │   │   │   ├── mxfp6_utils.py
│   │   │   │   ├── mxfp8_utils.py
│   │   │   │   ├── nvfp4_emulation_utils.py
│   │   │   │   ├── nvfp4_utils.py
│   │   │   │   ├── ocp_mx_utils.py
│   │   │   │   ├── quant_utils.py
│   │   │   │   └── w8a8_utils.py
│   │   │   ├── __init__.py
│   │   │   ├── awq.py
│   │   │   ├── awq_marlin.py
│   │   │   ├── awq_triton.py
│   │   │   ├── base_config.py
│   │   │   ├── bitsandbytes.py
│   │   │   ├── cpu_wna16.py
│   │   │   ├── experts_int8.py
│   │   │   ├── fbgemm_fp8.py
│   │   │   ├── fp8.py
│   │   │   ├── fp_quant.py
│   │   │   ├── gguf.py
│   │   │   ├── gptq.py
│   │   │   ├── gptq_marlin.py
│   │   │   ├── humming.py
│   │   │   ├── inc.py
│   │   │   ├── input_quant_fp8.py
│   │   │   ├── kv_cache.py
│   │   │   ├── modelopt.py
│   │   │   ├── moe_wna16.py
│   │   │   ├── mxfp4.py
│   │   │   ├── qutlass_utils.py
│   │   │   ├── schema.py
│   │   │   └── torchao.py
│   │   ├── rotary_embedding
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── common.py
│   │   │   ├── deepseek_scaling_rope.py
│   │   │   ├── dual_chunk_rope.py
│   │   │   ├── dynamic_ntk_alpha_rope.py
│   │   │   ├── dynamic_ntk_scaling_rope.py
│   │   │   ├── ernie45_vl_rope.py
│   │   │   ├── fope.py
│   │   │   ├── gemma4_rope.py
│   │   │   ├── linear_scaling_rope.py
│   │   │   ├── llama3_rope.py
│   │   │   ├── llama4_vision_rope.py
│   │   │   ├── mrope.py
│   │   │   ├── mrope_interleaved.py
│   │   │   ├── ntk_scaling_rope.py
│   │   │   ├── phi3_long_rope_scaled_rope.py
│   │   │   ├── telechat3_scaling_rope.py
│   │   │   ├── xdrope.py
│   │   │   └── yarn_scaling_rope.py
│   │   ├── __init__.py
│   │   ├── activation.py
│   │   ├── attention_layer_base.py
│   │   ├── batch_invariant.py
│   │   ├── conv.py
│   │   ├── kda.py
│   │   ├── layernorm.py
│   │   ├── lightning_attn.py
│   │   ├── linear.py
│   │   ├── logits_processor.py
│   │   ├── mla.py
│   │   ├── resampler.py
│   │   ├── sparse_attn_indexer.py
│   │   ├── utils.py
│   │   └── vocab_parallel_embedding.py
│   ├── model_loader
│   │   ├── reload
│   │   │   ├── __init__.py
│   │   │   ├── layerwise.py
│   │   │   ├── meta.py
│   │   │   ├── sanitize.py
│   │   │   ├── torchao_decorator.py
│   │   │   ├── types.py
│   │   │   └── utils.py
│   │   ├── __init__.py
│   │   ├── base_loader.py
│   │   ├── bitsandbytes_loader.py
│   │   ├── default_loader.py
│   │   ├── dummy_loader.py
│   │   ├── ep_weight_filter.py
│   │   ├── gguf_loader.py
│   │   ├── runai_streamer_loader.py
│   │   ├── sharded_state_loader.py
│   │   ├── tensorizer.py
│   │   ├── tensorizer_loader.py
│   │   ├── utils.py
│   │   └── weight_utils.py
│   ├── models
│   │   ├── transformers
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── causal.py
│   │   │   ├── legacy.py
│   │   │   ├── moe.py
│   │   │   ├── multimodal.py
│   │   │   ├── pooling.py
│   │   │   └── utils.py
│   │   ├── __init__.py
│   │   ├── adapters.py
│   │   ├── afmoe.py
│   │   ├── aimv2.py
│   │   ├── apertus.py
│   │   ├── arcee.py
│   │   ├── arctic.py
│   │   ├── aria.py
│   │   ├── audioflamingo3.py
│   │   ├── AXK1.py
│   │   ├── aya_vision.py
│   │   ├── bagel.py
│   │   ├── baichuan.py
│   │   ├── bailing_moe.py
│   │   ├── bailing_moe_linear.py
│   │   ├── bamba.py
│   │   ├── bee.py
│   │   ├── bert.py
│   │   ├── bert_with_rope.py
│   │   ├── blip.py
│   │   ├── blip2.py
│   │   ├── bloom.py
│   │   ├── chameleon.py
│   │   ├── chatglm.py
│   │   ├── cheers.py
│   │   ├── clip.py
│   │   ├── cohere2_vision.py
│   │   ├── cohere_asr.py
│   │   ├── colbert.py
│   │   ├── colmodernvbert.py
│   │   ├── colpali.py
│   │   ├── colqwen3.py
│   │   ├── colqwen3_5.py
│   │   ├── commandr.py
│   │   ├── config.py
│   │   ├── conformer_encoder.py
│   │   ├── dbrx.py
│   │   ├── deepencoder.py
│   │   ├── deepencoder2.py
│   │   ├── deepseek_eagle.py
│   │   ├── deepseek_eagle3.py
│   │   ├── deepseek_mtp.py
│   │   ├── deepseek_ocr.py
│   │   ├── deepseek_ocr2.py
│   │   ├── deepseek_v2.py
│   │   ├── deepseek_vl2.py
│   │   ├── dots1.py
│   │   ├── dots_ocr.py
│   │   ├── eagle2_5_vl.py
│   │   ├── ernie.py
│   │   ├── ernie45.py
│   │   ├── ernie45_moe.py
│   │   ├── ernie45_vl.py
│   │   ├── ernie45_vl_moe.py
│   │   ├── ernie_mtp.py
│   │   ├── exaone.py
│   │   ├── exaone4.py
│   │   ├── exaone4_5.py
│   │   ├── exaone4_5_mtp.py
│   │   ├── exaone_moe.py
│   │   ├── exaone_moe_mtp.py
│   │   ├── extract_hidden_states.py
│   │   ├── fairseq2_llama.py
│   │   ├── falcon.py
│   │   ├── falcon_h1.py
│   │   ├── fireredasr2.py
│   │   ├── fireredlid.py
│   │   ├── flex_olmo.py
│   │   ├── funasr.py
│   │   ├── funaudiochat.py
│   │   ├── fuyu.py
│   │   ├── gemma.py
│   │   ├── gemma2.py
│   │   ├── gemma3.py
│   │   ├── gemma3_mm.py
│   │   ├── gemma3n.py
│   │   ├── gemma3n_audio_utils.py
│   │   ├── gemma3n_mm.py
│   │   ├── gemma4.py
│   │   ├── gemma4_mm.py
│   │   ├── glm.py
│   │   ├── glm4.py
│   │   ├── glm4_1v.py
│   │   ├── glm4_moe.py
│   │   ├── glm4_moe_lite.py
│   │   ├── glm4_moe_lite_mtp.py
│   │   ├── glm4_moe_mtp.py
│   │   ├── glm4v.py
│   │   ├── glm_ocr.py
│   │   ├── glm_ocr_mtp.py
│   │   ├── glmasr.py
│   │   ├── glmasr_utils.py
│   │   ├── gpt2.py
│   │   ├── gpt_bigcode.py
│   │   ├── gpt_j.py
│   │   ├── gpt_neox.py
│   │   ├── gpt_oss.py
│   │   ├── granite.py
│   │   ├── granite4_vision.py
│   │   ├── granite_speech.py
│   │   ├── granitemoe.py
│   │   ├── granitemoehybrid.py
│   │   ├── granitemoeshared.py
│   │   ├── gritlm.py
│   │   ├── grok1.py
│   │   ├── h2ovl.py
│   │   ├── hunyuan_v1.py
│   │   ├── hunyuan_vision.py
│   │   ├── hy_v3.py
│   │   ├── hy_v3_mtp.py
│   │   ├── hyperclovax.py
│   │   ├── hyperclovax_vision.py
│   │   ├── hyperclovax_vision_v2.py
│   │   ├── idefics2_vision_model.py
│   │   ├── idefics3.py
│   │   ├── interfaces.py
│   │   ├── interfaces_base.py
│   │   ├── intern_vit.py
│   │   ├── internlm2.py
│   │   ├── internlm2_ve.py
│   │   ├── interns1.py
│   │   ├── interns1_pro.py
│   │   ├── interns1_vit.py
│   │   ├── internvl.py
│   │   ├── iquest_loopcoder.py
│   │   ├── isaac.py
│   │   ├── jais.py
│   │   ├── jais2.py
│   │   ├── jamba.py
│   │   ├── jina.py
│   │   ├── jina_vl.py
│   │   ├── kanana_v.py
│   │   ├── keye.py
│   │   ├── keye_vl1_5.py
│   │   ├── kimi_audio.py
│   │   ├── kimi_k25.py
│   │   ├── kimi_k25_vit.py
│   │   ├── kimi_linear.py
│   │   ├── kimi_vl.py
│   │   ├── lfm2.py
│   │   ├── lfm2_moe.py
│   │   ├── lfm2_siglip2.py
│   │   ├── lfm2_vl.py
│   │   ├── lightonocr.py
│   │   ├── llama.py
│   │   ├── llama4.py
│   │   ├── llama4_eagle.py
│   │   ├── llama_eagle.py
│   │   ├── llama_eagle3.py
│   │   ├── llava.py
│   │   ├── llava_next.py
│   │   ├── llava_next_video.py
│   │   ├── llava_onevision.py
│   │   ├── longcat_flash.py
│   │   ├── longcat_flash_mtp.py
│   │   ├── mamba.py
│   │   ├── mamba2.py
│   │   ├── medusa.py
│   │   ├── midashenglm.py
│   │   ├── mimo.py
│   │   ├── mimo_mtp.py
│   │   ├── mimo_v2_flash.py
│   │   ├── minicpm.py
│   │   ├── minicpm3.py
│   │   ├── minicpm_eagle.py
│   │   ├── minicpmo.py
│   │   ├── minicpmv.py
│   │   ├── minimax_m2.py
│   │   ├── minimax_text_01.py
│   │   ├── minimax_vl_01.py
│   │   ├── mistral.py
│   │   ├── mistral3.py
│   │   ├── mistral_large_3.py
│   │   ├── mistral_large_3_eagle.py
│   │   ├── mixtral.py
│   │   ├── mllama4.py
│   │   ├── mlp_speculator.py
│   │   ├── modernbert.py
│   │   ├── module_mapping.py
│   │   ├── molmo.py
│   │   ├── molmo2.py
│   │   ├── moonvit.py
│   │   ├── mpt.py
│   │   ├── musicflamingo.py
│   │   ├── nano_nemotron_vl.py
│   │   ├── nemotron.py
│   │   ├── nemotron_h.py
│   │   ├── nemotron_h_mtp.py
│   │   ├── nemotron_nas.py
│   │   ├── nemotron_parse.py
│   │   ├── nemotron_vl.py
│   │   ├── nvlm_d.py
│   │   ├── olmo.py
│   │   ├── olmo2.py
│   │   ├── olmo_hybrid.py
│   │   ├── olmoe.py
│   │   ├── opencua.py
│   │   ├── openpangu.py
│   │   ├── openpangu_mtp.py
│   │   ├── openpangu_vl.py
│   │   ├── opt.py
│   │   ├── orion.py
│   │   ├── ouro.py
│   │   ├── ovis.py
│   │   ├── ovis2_5.py
│   │   ├── paddleocr_vl.py
│   │   ├── paligemma.py
│   │   ├── parakeet.py
│   │   ├── param2moe.py
│   │   ├── persimmon.py
│   │   ├── phi.py
│   │   ├── phi3.py
│   │   ├── phi3v.py
│   │   ├── phi4mm.py
│   │   ├── phi4mm_audio.py
│   │   ├── phi4mm_utils.py
│   │   ├── phi4siglip.py
│   │   ├── phimoe.py
│   │   ├── pixtral.py
│   │   ├── plamo2.py
│   │   ├── plamo3.py
│   │   ├── qwen.py
│   │   ├── qwen2.py
│   │   ├── qwen2_5_omni_thinker.py
│   │   ├── qwen2_5_vl.py
│   │   ├── qwen2_audio.py
│   │   ├── qwen2_moe.py
│   │   ├── qwen2_rm.py
│   │   ├── qwen2_vl.py
│   │   ├── qwen3.py
│   │   ├── qwen3_5.py
│   │   ├── qwen3_5_mtp.py
│   │   ├── qwen3_asr.py
│   │   ├── qwen3_asr_forced_aligner.py
│   │   ├── qwen3_asr_realtime.py
│   │   ├── qwen3_dflash.py
│   │   ├── qwen3_moe.py
│   │   ├── qwen3_next.py
│   │   ├── qwen3_next_mtp.py
│   │   ├── qwen3_omni_moe_thinker.py
│   │   ├── qwen3_vl.py
│   │   ├── qwen3_vl_moe.py
│   │   ├── qwen_vl.py
│   │   ├── radio.py
│   │   ├── registry.py
│   │   ├── rnj1.py
│   │   ├── roberta.py
│   │   ├── rvl.py
│   │   ├── sarvam.py
│   │   ├── seed_oss.py
│   │   ├── siglip.py
│   │   ├── siglip2navit.py
│   │   ├── skyworkr1v.py
│   │   ├── smolvlm.py
│   │   ├── solar.py
│   │   ├── stablelm.py
│   │   ├── starcoder2.py
│   │   ├── step1.py
│   │   ├── step3_text.py
│   │   ├── step3_vl.py
│   │   ├── step3p5.py
│   │   ├── step3p5_mtp.py
│   │   ├── step_vl.py
│   │   ├── tarsier.py
│   │   ├── telechat2.py
│   │   ├── teleflm.py
│   │   ├── terratorch.py
│   │   ├── ultravox.py
│   │   ├── utils.py
│   │   ├── vision.py
│   │   ├── voxtral.py
│   │   ├── voxtral_realtime.py
│   │   ├── voyage.py
│   │   ├── whisper.py
│   │   ├── whisper_causal.py
│   │   ├── whisper_utils.py
│   │   └── zamba2.py
│   ├── offloader
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── prefetch.py
│   │   ├── prefetch_ops.py
│   │   └── uva.py
│   ├── warmup
│   │   ├── __init__.py
│   │   ├── deep_gemm_warmup.py
│   │   └── kernel_warmup.py
│   ├── __init__.py
│   ├── custom_op.py
│   ├── parameter.py
│   └── utils.py
├── multimodal
│   ├── media
│   │   ├── __init__.py
│   │   ├── audio.py
│   │   ├── base.py
│   │   ├── connector.py
│   │   ├── image.py
│   │   └── video.py
│   ├── processing
│   │   ├── __init__.py
│   │   ├── context.py
│   │   ├── dummy_inputs.py
│   │   ├── inputs.py
│   │   └── processor.py
│   ├── __init__.py
│   ├── audio.py
│   ├── cache.py
│   ├── encoder_budget.py
│   ├── evs.py
│   ├── hasher.py
│   ├── image.py
│   ├── inputs.py
│   ├── parse.py
│   ├── registry.py
│   ├── utils.py
│   └── video.py
├── parser
│   ├── __init__.py
│   ├── abstract_parser.py
│   ├── minimax_m2_parser.py
│   └── parser_manager.py
├── platforms
│   ├── __init__.py
│   ├── cpu.py
│   ├── cuda.py
│   ├── interface.py
│   ├── rocm.py
│   ├── tpu.py
│   ├── xpu.py
│   └── zen_cpu.py
├── plugins
│   ├── io_processors
│   │   ├── __init__.py
│   │   └── interface.py
│   ├── lora_resolvers
│   │   ├── __init__.py
│   │   ├── filesystem_resolver.py
│   │   └── hf_hub_resolver.py
│   └── __init__.py
├── profiler
│   ├── __init__.py
│   ├── layerwise_profile.py
│   ├── utils.py
│   └── wrapper.py
├── ray
│   ├── __init__.py
│   ├── lazy_utils.py
│   └── ray_env.py
├── reasoning
│   ├── __init__.py
│   ├── abs_reasoning_parsers.py
│   ├── basic_parsers.py
│   ├── deepseek_r1_reasoning_parser.py
│   ├── deepseek_v3_reasoning_parser.py
│   ├── ernie45_reasoning_parser.py
│   ├── gemma4_reasoning_parser.py
│   ├── gemma4_utils.py
│   ├── gptoss_reasoning_parser.py
│   ├── granite_reasoning_parser.py
│   ├── hunyuan_a13b_reasoning_parser.py
│   ├── hy_v3_reasoning_parser.py
│   ├── identity_reasoning_parser.py
│   ├── kimi_k2_reasoning_parser.py
│   ├── minimax_m2_reasoning_parser.py
│   ├── mistral_reasoning_parser.py
│   ├── nemotron_v3_reasoning_parser.py
│   ├── olmo3_reasoning_parser.py
│   ├── qwen3_reasoning_parser.py
│   ├── seedoss_reasoning_parser.py
│   ├── step3_reasoning_parser.py
│   └── step3p5_reasoning_parser.py
├── renderers
│   ├── inputs
│   │   ├── __init__.py
│   │   ├── preprocess.py
│   │   └── tokenize.py
│   ├── __init__.py
│   ├── base.py
│   ├── deepseek_v32.py
│   ├── embed_utils.py
│   ├── grok2.py
│   ├── hf.py
│   ├── mistral.py
│   ├── params.py
│   ├── registry.py
│   └── terratorch.py
├── third_party
│   ├── deep_gemm
│   │   ├── include
│   │   │   ├── cute
│   │   │   │   ├── algorithm
│   │   │   │   │   ├── axpby.hpp
│   │   │   │   │   ├── clear.hpp
│   │   │   │   │   ├── cooperative_copy.hpp
│   │   │   │   │   ├── cooperative_gemm.hpp
│   │   │   │   │   ├── copy.hpp
│   │   │   │   │   ├── fill.hpp
│   │   │   │   │   ├── functional.hpp
│   │   │   │   │   ├── gemm.hpp
│   │   │   │   │   ├── prefer.hpp
│   │   │   │   │   ├── prefetch.hpp
│   │   │   │   │   ├── tensor_algorithms.hpp
│   │   │   │   │   ├── tensor_reduce.hpp
│   │   │   │   │   └── tuple_algorithms.hpp
│   │   │   │   ├── arch
│   │   │   │   │   ├── cluster_sm100.hpp
│   │   │   │   │   ├── cluster_sm90.hpp
│   │   │   │   │   ├── config.hpp
│   │   │   │   │   ├── copy.hpp
│   │   │   │   │   ├── copy_sm100.hpp
│   │   │   │   │   ├── copy_sm100_tma.hpp
│   │   │   │   │   ├── copy_sm50.hpp
│   │   │   │   │   ├── copy_sm75.hpp
│   │   │   │   │   ├── copy_sm80.hpp
│   │   │   │   │   ├── copy_sm90.hpp
│   │   │   │   │   ├── copy_sm90_desc.hpp
│   │   │   │   │   ├── copy_sm90_tma.hpp
│   │   │   │   │   ├── mma.hpp
│   │   │   │   │   ├── mma_sm100.hpp
│   │   │   │   │   ├── mma_sm100_desc.hpp
│   │   │   │   │   ├── mma_sm100_umma.hpp
│   │   │   │   │   ├── mma_sm120.hpp
│   │   │   │   │   ├── mma_sm120_sparse.hpp
│   │   │   │   │   ├── mma_sm61.hpp
│   │   │   │   │   ├── mma_sm70.hpp
│   │   │   │   │   ├── mma_sm75.hpp
│   │   │   │   │   ├── mma_sm80.hpp
│   │   │   │   │   ├── mma_sm89.hpp
│   │   │   │   │   ├── mma_sm90.hpp
│   │   │   │   │   ├── mma_sm90_desc.hpp
│   │   │   │   │   ├── mma_sm90_gmma.hpp
│   │   │   │   │   ├── mma_sm90_gmma_ext.hpp
│   │   │   │   │   ├── mma_sm90_gmma_sparse.hpp
│   │   │   │   │   ├── mma_sm90_gmma_sparse_ext.hpp
│   │   │   │   │   ├── simd_sm100.hpp
│   │   │   │   │   ├── tmem_allocator_sm100.hpp
│   │   │   │   │   └── util.hpp
│   │   │   │   ├── atom
│   │   │   │   │   ├── copy_atom.hpp
│   │   │   │   │   ├── copy_traits.hpp
│   │   │   │   │   ├── copy_traits_sm100.hpp
│   │   │   │   │   ├── copy_traits_sm100_im2col.hpp
│   │   │   │   │   ├── copy_traits_sm100_tma.hpp
│   │   │   │   │   ├── copy_traits_sm50.hpp
│   │   │   │   │   ├── copy_traits_sm75.hpp
│   │   │   │   │   ├── copy_traits_sm80.hpp
│   │   │   │   │   ├── copy_traits_sm90.hpp
│   │   │   │   │   ├── copy_traits_sm90_im2col.hpp
│   │   │   │   │   ├── copy_traits_sm90_tma.hpp
│   │   │   │   │   ├── copy_traits_sm90_tma_swizzle.hpp
│   │   │   │   │   ├── mma_atom.hpp
│   │   │   │   │   ├── mma_traits.hpp
│   │   │   │   │   ├── mma_traits_sm100.hpp
│   │   │   │   │   ├── mma_traits_sm120.hpp
│   │   │   │   │   ├── mma_traits_sm120_sparse.hpp
│   │   │   │   │   ├── mma_traits_sm61.hpp
│   │   │   │   │   ├── mma_traits_sm70.hpp
│   │   │   │   │   ├── mma_traits_sm75.hpp
│   │   │   │   │   ├── mma_traits_sm80.hpp
│   │   │   │   │   ├── mma_traits_sm89.hpp
│   │   │   │   │   ├── mma_traits_sm90.hpp
│   │   │   │   │   ├── mma_traits_sm90_gmma.hpp
│   │   │   │   │   ├── mma_traits_sm90_gmma_ext.hpp
│   │   │   │   │   ├── mma_traits_sm90_gmma_sparse.hpp
│   │   │   │   │   ├── mma_traits_sm90_gmma_sparse_ext.hpp
│   │   │   │   │   └── partitioner.hpp
│   │   │   │   ├── container
│   │   │   │   │   ├── alignment.hpp
│   │   │   │   │   ├── array.hpp
│   │   │   │   │   ├── array_aligned.hpp
│   │   │   │   │   ├── array_subbyte.hpp
│   │   │   │   │   ├── bit_field.hpp
│   │   │   │   │   ├── cuda_types.hpp
│   │   │   │   │   ├── tuple.hpp
│   │   │   │   │   └── type_list.hpp
│   │   │   │   ├── numeric
│   │   │   │   │   ├── arithmetic_tuple.hpp
│   │   │   │   │   ├── complex.hpp
│   │   │   │   │   ├── int.hpp
│   │   │   │   │   ├── integer_sequence.hpp
│   │   │   │   │   ├── integral_constant.hpp
│   │   │   │   │   ├── integral_ratio.hpp
│   │   │   │   │   ├── math.hpp
│   │   │   │   │   ├── numeric_types.hpp
│   │   │   │   │   └── real.hpp
│   │   │   │   ├── util
│   │   │   │   │   ├── debug.hpp
│   │   │   │   │   ├── print.hpp
│   │   │   │   │   ├── print_latex.hpp
│   │   │   │   │   ├── print_svg.hpp
│   │   │   │   │   ├── print_tensor.hpp
│   │   │   │   │   └── type_traits.hpp
│   │   │   │   ├── config.hpp
│   │   │   │   ├── int_tuple.hpp
│   │   │   │   ├── layout.hpp
│   │   │   │   ├── layout_composed.hpp
│   │   │   │   ├── pointer.hpp
│   │   │   │   ├── pointer_base.hpp
│   │   │   │   ├── pointer_flagged.hpp
│   │   │   │   ├── pointer_sparse.hpp
│   │   │   │   ├── pointer_swizzle.hpp
│   │   │   │   ├── stride.hpp
│   │   │   │   ├── swizzle.hpp
│   │   │   │   ├── swizzle_layout.hpp
│   │   │   │   ├── tensor.hpp
│   │   │   │   ├── tensor_impl.hpp
│   │   │   │   ├── tensor_zip.hpp
│   │   │   │   └── underscore.hpp
│   │   │   ├── cutlass
│   │   │   │   ├── arch
│   │   │   │   │   ├── arch.h
│   │   │   │   │   ├── barrier.h
│   │   │   │   │   ├── cache_operation.h
│   │   │   │   │   ├── config.h
│   │   │   │   │   ├── grid_dependency_control.h
│   │   │   │   │   ├── memory.h
│   │   │   │   │   ├── memory_sm75.h
│   │   │   │   │   ├── memory_sm80.h
│   │   │   │   │   ├── mma.h
│   │   │   │   │   ├── mma_sm100.h
│   │   │   │   │   ├── mma_sm50.h
│   │   │   │   │   ├── mma_sm60.h
│   │   │   │   │   ├── mma_sm61.h
│   │   │   │   │   ├── mma_sm70.h
│   │   │   │   │   ├── mma_sm75.h
│   │   │   │   │   ├── mma_sm80.h
│   │   │   │   │   ├── mma_sm89.h
│   │   │   │   │   ├── mma_sm90.h
│   │   │   │   │   ├── mma_sparse_sm80.h
│   │   │   │   │   ├── mma_sparse_sm89.h
│   │   │   │   │   ├── reg_reconfig.h
│   │   │   │   │   ├── simd.h
│   │   │   │   │   ├── simd_sm60.h
│   │   │   │   │   ├── simd_sm61.h
│   │   │   │   │   ├── synclog.hpp
│   │   │   │   │   ├── wmma.h
│   │   │   │   │   ├── wmma_sm70.h
│   │   │   │   │   ├── wmma_sm72.h
│   │   │   │   │   └── wmma_sm75.h
│   │   │   │   ├── conv
│   │   │   │   │   ├── collective
│   │   │   │   │   │   ├── builders
│   │   │   │   │   │   │   ├── sm100_common.inl
│   │   │   │   │   │   │   ├── sm100_umma_builder.inl
│   │   │   │   │   │   │   ├── sm90_common.inl
│   │   │   │   │   │   │   └── sm90_gmma_builder.inl
│   │   │   │   │   │   ├── collective_builder.hpp
│   │   │   │   │   │   ├── collective_conv.hpp
│   │   │   │   │   │   ├── detail.hpp
│   │   │   │   │   │   ├── sm100_implicit_gemm_umma_warpspecialized.hpp
│   │   │   │   │   │   └── sm90_implicit_gemm_gmma_ss_warpspecialized.hpp
│   │   │   │   │   ├── device
│   │   │   │   │   │   ├── conv_universal_adapter.hpp
│   │   │   │   │   │   ├── direct_convolution.h
│   │   │   │   │   │   ├── implicit_gemm_convolution.h
│   │   │   │   │   │   └── implicit_gemm_convolution_fusion.h
│   │   │   │   │   ├── kernel
│   │   │   │   │   │   ├── conv_universal.hpp
│   │   │   │   │   │   ├── default_conv2d.h
│   │   │   │   │   │   ├── default_conv2d_dgrad.h
│   │   │   │   │   │   ├── default_conv2d_fprop.h
│   │   │   │   │   │   ├── default_conv2d_fprop_fusion.h
│   │   │   │   │   │   ├── default_conv2d_fprop_with_absmax.h
│   │   │   │   │   │   ├── default_conv2d_fprop_with_broadcast.h
│   │   │   │   │   │   ├── default_conv2d_fprop_with_reduction.h
│   │   │   │   │   │   ├── default_conv2d_group_fprop.h
│   │   │   │   │   │   ├── default_conv2d_wgrad.h
│   │   │   │   │   │   ├── default_conv2d_wgrad_fusion.h
│   │   │   │   │   │   ├── default_conv3d_dgrad.h
│   │   │   │   │   │   ├── default_conv3d_fprop.h
│   │   │   │   │   │   ├── default_conv3d_fprop_fusion.h
│   │   │   │   │   │   ├── default_conv3d_fprop_with_broadcast.h
│   │   │   │   │   │   ├── default_conv3d_wgrad.h
│   │   │   │   │   │   ├── default_deconv2d.h
│   │   │   │   │   │   ├── default_deconv2d_with_broadcast.h
│   │   │   │   │   │   ├── default_deconv3d.h
│   │   │   │   │   │   ├── default_deconv3d_with_broadcast.h
│   │   │   │   │   │   ├── default_depthwise_fprop.h
│   │   │   │   │   │   ├── direct_convolution.h
│   │   │   │   │   │   ├── implicit_gemm_convolution.h
│   │   │   │   │   │   ├── implicit_gemm_convolution_fusion.h
│   │   │   │   │   │   ├── implicit_gemm_convolution_strided_dgrad.h
│   │   │   │   │   │   ├── implicit_gemm_convolution_with_absmax.h
│   │   │   │   │   │   ├── implicit_gemm_convolution_with_fused_epilogue.h
│   │   │   │   │   │   ├── sm100_implicit_gemm_tma_warpspecialized.hpp
│   │   │   │   │   │   └── sm90_implicit_gemm_tma_warpspecialized.hpp
│   │   │   │   │   ├── thread
│   │   │   │   │   │   └── depthwise_mma.h
│   │   │   │   │   ├── threadblock
│   │   │   │   │   │   ├── conv2d_dgrad_filter_tile_access_iterator_analytic.h
│   │   │   │   │   │   ├── conv2d_dgrad_filter_tile_access_iterator_optimized.h
│   │   │   │   │   │   ├── conv2d_dgrad_output_gradient_tile_access_iterator_analytic.h
│   │   │   │   │   │   ├── conv2d_dgrad_output_gradient_tile_access_iterator_optimized.h
│   │   │   │   │   │   ├── conv2d_fprop_activation_tile_access_iterator_analytic.h
│   │   │   │   │   │   ├── conv2d_fprop_activation_tile_access_iterator_few_channels.h
│   │   │   │   │   │   ├── conv2d_fprop_activation_tile_access_iterator_fixed_channels.h
│   │   │   │   │   │   ├── conv2d_fprop_activation_tile_access_iterator_optimized.h
│   │   │   │   │   │   ├── conv2d_fprop_filter_tile_access_iterator_analytic.h
│   │   │   │   │   │   ├── conv2d_fprop_filter_tile_access_iterator_few_channels.h
│   │   │   │   │   │   ├── conv2d_fprop_filter_tile_access_iterator_fixed_channels.h
│   │   │   │   │   │   ├── conv2d_fprop_filter_tile_access_iterator_optimized.h
│   │   │   │   │   │   ├── conv2d_params.h
│   │   │   │   │   │   ├── conv2d_tile_iterator.h
│   │   │   │   │   │   ├── conv2d_wgrad_activation_tile_access_iterator_analytic.h
│   │   │   │   │   │   ├── conv2d_wgrad_activation_tile_access_iterator_optimized.h
│   │   │   │   │   │   ├── conv2d_wgrad_output_gradient_tile_access_iterator_analytic.h
│   │   │   │   │   │   ├── conv2d_wgrad_output_gradient_tile_access_iterator_optimized.h
│   │   │   │   │   │   ├── conv3d_dgrad_filter_tile_access_iterator_analytic.h
│   │   │   │   │   │   ├── conv3d_dgrad_filter_tile_access_iterator_optimized.h
│   │   │   │   │   │   ├── conv3d_dgrad_output_gradient_tile_access_iterator_analytic.h
│   │   │   │   │   │   ├── conv3d_dgrad_output_gradient_tile_access_iterator_optimized.h
│   │   │   │   │   │   ├── conv3d_fprop_activation_tile_access_iterator_analytic.h
│   │   │   │   │   │   ├── conv3d_fprop_activation_tile_access_iterator_optimized.h
│   │   │   │   │   │   ├── conv3d_fprop_filter_tile_access_iterator_analytic.h
│   │   │   │   │   │   ├── conv3d_fprop_filter_tile_access_iterator_optimized.h
│   │   │   │   │   │   ├── conv3d_params.h
│   │   │   │   │   │   ├── conv3d_wgrad_activation_tile_access_iterator_analytic.h
│   │   │   │   │   │   ├── conv3d_wgrad_activation_tile_access_iterator_optimized.h
│   │   │   │   │   │   ├── conv3d_wgrad_output_gradient_tile_access_iterator_analytic.h
│   │   │   │   │   │   ├── conv3d_wgrad_output_gradient_tile_access_iterator_optimized.h
│   │   │   │   │   │   ├── depthwise_direct_conv_params.h
│   │   │   │   │   │   ├── depthwise_fprop_activation_tile_access_iterator_direct_conv_fixed_stride_dilation.h
│   │   │   │   │   │   ├── depthwise_fprop_activation_tile_access_iterator_direct_conv_optimized.h
│   │   │   │   │   │   ├── depthwise_fprop_direct_conv_multistage.h
│   │   │   │   │   │   ├── depthwise_fprop_filter_tile_access_iterator_direct_conv_optimized.h
│   │   │   │   │   │   ├── depthwise_fprop_pipelined.h
│   │   │   │   │   │   ├── depthwise_mma_base.h
│   │   │   │   │   │   ├── depthwise_mma_core_with_lane_access_size.h
│   │   │   │   │   │   ├── implicit_gemm_fprop_fusion_multistage.h
│   │   │   │   │   │   ├── implicit_gemm_multistage.h
│   │   │   │   │   │   ├── implicit_gemm_pipelined.h
│   │   │   │   │   │   ├── implicit_gemm_wgrad_fusion_multistage.h
│   │   │   │   │   │   ├── predicated_scale_bias_vector_access_iterator.h
│   │   │   │   │   │   ├── predicated_scale_bias_vector_iterator.h
│   │   │   │   │   │   └── threadblock_swizzle.h
│   │   │   │   │   ├── warp
│   │   │   │   │   │   ├── mma_depthwise_simt.h
│   │   │   │   │   │   ├── mma_depthwise_simt_tile_iterator.h
│   │   │   │   │   │   └── scale_bias_relu_transform.h
│   │   │   │   │   ├── conv2d_problem_size.h
│   │   │   │   │   ├── conv3d_problem_size.h
│   │   │   │   │   ├── convnd_problem_shape.hpp
│   │   │   │   │   ├── convolution.h
│   │   │   │   │   ├── detail.hpp
│   │   │   │   │   └── dispatch_policy.hpp
│   │   │   │   ├── detail
│   │   │   │   │   ├── collective
│   │   │   │   │   │   ├── mixed_input_utils.hpp
│   │   │   │   │   │   └── sm103_kernel_type.hpp
│   │   │   │   │   ├── blockwise_scale_layout.hpp
│   │   │   │   │   ├── cluster.hpp
│   │   │   │   │   ├── collective.hpp
│   │   │   │   │   ├── dependent_false.hpp
│   │   │   │   │   ├── helper_macros.hpp
│   │   │   │   │   ├── layout.hpp
│   │   │   │   │   ├── mainloop_fusion_helper_scale_factor.hpp
│   │   │   │   │   ├── mma.hpp
│   │   │   │   │   ├── sm100_blockscaled_layout.hpp
│   │   │   │   │   ├── sm100_mixed_dtype_blockwise_layout.hpp
│   │   │   │   │   ├── sm100_tmem_helper.hpp
│   │   │   │   │   └── sm103_blockscaled_layout.hpp
│   │   │   │   ├── epilogue
│   │   │   │   │   ├── collective
│   │   │   │   │   │   ├── builders
│   │   │   │   │   │   │   ├── sm100_builder.inl
│   │   │   │   │   │   │   ├── sm103_builder.inl
│   │   │   │   │   │   │   ├── sm120_builder.inl
│   │   │   │   │   │   │   ├── sm120_common.inl
│   │   │   │   │   │   │   ├── sm90_builder.inl
│   │   │   │   │   │   │   └── sm90_common.inl
│   │   │   │   │   │   ├── collective_builder.hpp
│   │   │   │   │   │   ├── collective_epilogue.hpp
│   │   │   │   │   │   ├── default_epilogue.hpp
│   │   │   │   │   │   ├── default_epilogue_array.hpp
│   │   │   │   │   │   ├── detail.hpp
│   │   │   │   │   │   ├── epilogue_tensor_broadcast.hpp
│   │   │   │   │   │   ├── sm100_epilogue_array_nosmem.hpp
│   │   │   │   │   │   ├── sm100_epilogue_array_tma_warpspecialized.hpp
│   │   │   │   │   │   ├── sm100_epilogue_nosmem.hpp
│   │   │   │   │   │   ├── sm100_epilogue_tma_warpspecialized.hpp
│   │   │   │   │   │   ├── sm70_epilogue_vectorized.hpp
│   │   │   │   │   │   ├── sm70_epilogue_vectorized_array.hpp
│   │   │   │   │   │   ├── sm90_epilogue_array_tma_warpspecialized.hpp
│   │   │   │   │   │   ├── sm90_epilogue_tma_warpspecialized.hpp
│   │   │   │   │   │   └── sm90_epilogue_tma_warpspecialized_bias_elementwise.hpp
│   │   │   │   │   ├── fusion
│   │   │   │   │   │   ├── callbacks.hpp
│   │   │   │   │   │   ├── operations.hpp
│   │   │   │   │   │   ├── sm100_callbacks_tma_warpspecialized.hpp
│   │   │   │   │   │   ├── sm100_visitor_compute_tma_warpspecialized.hpp
│   │   │   │   │   │   ├── sm100_visitor_store_tma_warpspecialized.hpp
│   │   │   │   │   │   ├── sm120_callbacks_tma_warpspecialized.hpp
│   │   │   │   │   │   ├── sm120_visitor_store_tma_warpspecialized.hpp
│   │   │   │   │   │   ├── sm90_callbacks_tma_warpspecialized.hpp
│   │   │   │   │   │   ├── sm90_visitor_compute_tma_warpspecialized.hpp
│   │   │   │   │   │   ├── sm90_visitor_load_tma_warpspecialized.hpp
│   │   │   │   │   │   ├── sm90_visitor_store_tma_warpspecialized.hpp
│   │   │   │   │   │   ├── sm90_visitor_tma_warpspecialized.hpp
│   │   │   │   │   │   └── sm90_visitor_topk_softmax.hpp
│   │   │   │   │   ├── thread
│   │   │   │   │   │   ├── activation.h
│   │   │   │   │   │   ├── conversion_op.h
│   │   │   │   │   │   ├── detail.hpp
│   │   │   │   │   │   ├── linear_combination.h
│   │   │   │   │   │   ├── linear_combination_bias_elementwise.h
│   │   │   │   │   │   ├── linear_combination_bias_relu.h
│   │   │   │   │   │   ├── linear_combination_clamp.h
│   │   │   │   │   │   ├── linear_combination_dgelu.h
│   │   │   │   │   │   ├── linear_combination_drelu.h
│   │   │   │   │   │   ├── linear_combination_gelu.h
│   │   │   │   │   │   ├── linear_combination_generic.h
│   │   │   │   │   │   ├── linear_combination_generic_with_scaling.h
│   │   │   │   │   │   ├── linear_combination_hardswish.h
│   │   │   │   │   │   ├── linear_combination_leaky_relu.h
│   │   │   │   │   │   ├── linear_combination_params.h
│   │   │   │   │   │   ├── linear_combination_planar_complex.h
│   │   │   │   │   │   ├── linear_combination_relu.h
│   │   │   │   │   │   ├── linear_combination_relu0.h
│   │   │   │   │   │   ├── linear_combination_residual_block.h
│   │   │   │   │   │   ├── linear_combination_sigmoid.h
│   │   │   │   │   │   ├── linear_combination_silu.h
│   │   │   │   │   │   ├── linear_combination_tensor_broadcast.hpp
│   │   │   │   │   │   ├── linear_combination_with_elementwise.h
│   │   │   │   │   │   ├── reduction_op.h
│   │   │   │   │   │   └── scale_type.h
│   │   │   │   │   ├── threadblock
│   │   │   │   │   │   ├── fusion
│   │   │   │   │   │   │   ├── visitor_2x.hpp
│   │   │   │   │   │   │   ├── visitor_compute.hpp
│   │   │   │   │   │   │   ├── visitor_load.hpp
│   │   │   │   │   │   │   ├── visitor_store.hpp
│   │   │   │   │   │   │   └── visitors.hpp
│   │   │   │   │   │   ├── default_epilogue_complex_tensor_op.h
│   │   │   │   │   │   ├── default_epilogue_complex_tensor_op_blas3.h
│   │   │   │   │   │   ├── default_epilogue_direct_store.h
│   │   │   │   │   │   ├── default_epilogue_planar_complex.h
│   │   │   │   │   │   ├── default_epilogue_simt.h
│   │   │   │   │   │   ├── default_epilogue_tensor_op.h
│   │   │   │   │   │   ├── default_epilogue_tensor_op_blas3.h
│   │   │   │   │   │   ├── default_epilogue_volta_tensor_op.h
│   │   │   │   │   │   ├── default_epilogue_with_absmax.h
│   │   │   │   │   │   ├── default_epilogue_with_broadcast.h
│   │   │   │   │   │   ├── default_epilogue_with_reduction.h
│   │   │   │   │   │   ├── default_epilogue_wmma_tensor_op.h
│   │   │   │   │   │   ├── default_thread_map_simt.h
│   │   │   │   │   │   ├── default_thread_map_tensor_op.h
│   │   │   │   │   │   ├── default_thread_map_volta_tensor_op.h
│   │   │   │   │   │   ├── default_thread_map_wmma_tensor_op.h
│   │   │   │   │   │   ├── direct_store_epilogue_iterator.h
│   │   │   │   │   │   ├── epilogue.h
│   │   │   │   │   │   ├── epilogue_base.h
│   │   │   │   │   │   ├── epilogue_base_streamk.h
│   │   │   │   │   │   ├── epilogue_depthwise.h
│   │   │   │   │   │   ├── epilogue_direct_store.h
│   │   │   │   │   │   ├── epilogue_gemm_k_reduction.h
│   │   │   │   │   │   ├── epilogue_planar_complex.h
│   │   │   │   │   │   ├── epilogue_smem_accumulator.h
│   │   │   │   │   │   ├── epilogue_streamk_with_broadcast.h
│   │   │   │   │   │   ├── epilogue_visitor_with_softmax.h
│   │   │   │   │   │   ├── epilogue_with_absmax.h
│   │   │   │   │   │   ├── epilogue_with_broadcast.h
│   │   │   │   │   │   ├── epilogue_with_reduction.h
│   │   │   │   │   │   ├── epilogue_with_scaling_factor.h
│   │   │   │   │   │   ├── epilogue_with_visitor.h
│   │   │   │   │   │   ├── epilogue_with_visitor_callbacks.h
│   │   │   │   │   │   ├── epilogue_workspace.h
│   │   │   │   │   │   ├── interleaved_epilogue.h
│   │   │   │   │   │   ├── output_iterator_parameter.h
│   │   │   │   │   │   ├── output_tile_thread_map.h
│   │   │   │   │   │   ├── predicated_tile_iterator.h
│   │   │   │   │   │   ├── predicated_tile_iterator_affine.h
│   │   │   │   │   │   ├── predicated_tile_iterator_affine_layout_params.h
│   │   │   │   │   │   ├── predicated_tile_iterator_blas3.h
│   │   │   │   │   │   ├── predicated_tile_iterator_conv.h
│   │   │   │   │   │   ├── predicated_tile_iterator_direct_conv.h
│   │   │   │   │   │   ├── predicated_tile_iterator_params.h
│   │   │   │   │   │   ├── predicated_tile_iterator_predicates.h
│   │   │   │   │   │   ├── predicated_tile_iterator_strided_dgrad.h
│   │   │   │   │   │   ├── shared_load_iterator.h
│   │   │   │   │   │   ├── shared_load_iterator_mixed.h
│   │   │   │   │   │   └── shared_load_iterator_pitch_linear.h
│   │   │   │   │   ├── warp
│   │   │   │   │   │   ├── fragment_iterator_complex_tensor_op.h
│   │   │   │   │   │   ├── fragment_iterator_gaussian_complex_tensor_op.h
│   │   │   │   │   │   ├── fragment_iterator_simt.h
│   │   │   │   │   │   ├── fragment_iterator_tensor_op.h
│   │   │   │   │   │   ├── fragment_iterator_volta_tensor_op.h
│   │   │   │   │   │   ├── fragment_iterator_wmma_tensor_op.h
│   │   │   │   │   │   ├── simt_policy.h
│   │   │   │   │   │   ├── tensor_op_policy.h
│   │   │   │   │   │   ├── tile_iterator_simt.h
│   │   │   │   │   │   ├── tile_iterator_tensor_op.h
│   │   │   │   │   │   ├── tile_iterator_tensor_op_mixed.h
│   │   │   │   │   │   ├── tile_iterator_volta_tensor_op.h
│   │   │   │   │   │   ├── tile_iterator_wmma_tensor_op.h
│   │   │   │   │   │   ├── volta_tensor_op_policy.h
│   │   │   │   │   │   └── wmma_tensor_op_policy.h
│   │   │   │   │   └── dispatch_policy.hpp
│   │   │   │   ├── experimental
│   │   │   │   │   └── distributed
│   │   │   │   │       ├── device
│   │   │   │   │       │   ├── detail.hpp
│   │   │   │   │       │   ├── dist_gemm_universal_wrapper.hpp
│   │   │   │   │       │   └── full_barrier.hpp
│   │   │   │   │       ├── kernel
│   │   │   │   │       │   ├── detail.hpp
│   │   │   │   │       │   ├── dist_gemm_kernel_wrapper.hpp
│   │   │   │   │       │   └── full_barrier.hpp
│   │   │   │   │       └── schedules
│   │   │   │   │           ├── dist_gemm_1d_schedules.hpp
│   │   │   │   │           └── dist_gemm_base_schedule.hpp
│   │   │   │   ├── gemm
│   │   │   │   │   ├── collective
│   │   │   │   │   │   ├── builders
│   │   │   │   │   │   │   ├── sm100_9xBF16_umma_builder.inl
│   │   │   │   │   │   │   ├── sm100_blockscaled_mixed_tma_cpasync_umma_builder.inl
│   │   │   │   │   │   │   ├── sm100_blockscaled_sparse_umma_builder.inl
│   │   │   │   │   │   │   ├── sm100_blockscaled_umma_builder.inl
│   │   │   │   │   │   │   ├── sm100_blockwise_umma_builder.inl
│   │   │   │   │   │   │   ├── sm100_common.inl
│   │   │   │   │   │   │   ├── sm100_cpasync_umma_builder.inl
│   │   │   │   │   │   │   ├── sm100_mixed_input_umma_builder.inl
│   │   │   │   │   │   │   ├── sm100_mixed_tma_cpasync_umma_builder.inl
│   │   │   │   │   │   │   ├── sm100_pipeline_carveout.inl
│   │   │   │   │   │   │   ├── sm100_simt_builder.inl
│   │   │   │   │   │   │   ├── sm100_sparse_umma_builder.inl
│   │   │   │   │   │   │   ├── sm100_umma_builder.inl
│   │   │   │   │   │   │   ├── sm103_blockscaled_umma_builder.inl
│   │   │   │   │   │   │   ├── sm120_blockscaled_mma_builder.inl
│   │   │   │   │   │   │   ├── sm120_blockscaled_sparse_mma_builder.inl
│   │   │   │   │   │   │   ├── sm120_blockwise_mma_builder.inl
│   │   │   │   │   │   │   ├── sm120_common.inl
│   │   │   │   │   │   │   ├── sm120_mma_builder.inl
│   │   │   │   │   │   │   ├── sm120_sparse_mma_builder.inl
│   │   │   │   │   │   │   ├── sm1xx_common.inl
│   │   │   │   │   │   │   ├── sm1xx_sparse_config.inl
│   │   │   │   │   │   │   ├── sm90_common.inl
│   │   │   │   │   │   │   ├── sm90_gmma_builder.inl
│   │   │   │   │   │   │   ├── sm90_sparse_config.inl
│   │   │   │   │   │   │   └── sm90_sparse_gmma_builder.inl
│   │   │   │   │   │   ├── collective_builder.hpp
│   │   │   │   │   │   ├── collective_builder_decl.hpp
│   │   │   │   │   │   ├── collective_mma.hpp
│   │   │   │   │   │   ├── collective_mma_decl.hpp
│   │   │   │   │   │   ├── fp8_accumulation.hpp
│   │   │   │   │   │   ├── sm100_blockscaled_mma_array_warpspecialized.hpp
│   │   │   │   │   │   ├── sm100_blockscaled_mma_mixed_tma_cpasync_warpspecialized.hpp
│   │   │   │   │   │   ├── sm100_blockscaled_mma_warpspecialized.hpp
│   │   │   │   │   │   ├── sm100_blockscaled_sparse_mma_warpspecialized.hpp
│   │   │   │   │   │   ├── sm100_mma_array_warpspecialized.hpp
│   │   │   │   │   │   ├── sm100_mma_array_warpspecialized_blockwise_scaling.hpp
│   │   │   │   │   │   ├── sm100_mma_array_warpspecialized_emulated.hpp
│   │   │   │   │   │   ├── sm100_mma_cpasync_warpspecialized.hpp
│   │   │   │   │   │   ├── sm100_mma_mixed_tma_cpasync_warpspecialized.hpp
│   │   │   │   │   │   ├── sm100_mma_warpspecialized.hpp
│   │   │   │   │   │   ├── sm100_mma_warpspecialized_blockwise_scaling.hpp
│   │   │   │   │   │   ├── sm100_mma_warpspecialized_emulated.hpp
│   │   │   │   │   │   ├── sm100_mma_warpspecialized_mixed_input.hpp
│   │   │   │   │   │   ├── sm100_sparse_mma_warpspecialized.hpp
│   │   │   │   │   │   ├── sm103_blockscaled_mma_array_warpspecialized.hpp
│   │   │   │   │   │   ├── sm103_blockscaled_mma_warpspecialized.hpp
│   │   │   │   │   │   ├── sm120_blockscaled_mma_array_tma.hpp
│   │   │   │   │   │   ├── sm120_blockscaled_mma_tma.hpp
│   │   │   │   │   │   ├── sm120_blockscaled_sparse_mma_tma.hpp
│   │   │   │   │   │   ├── sm120_mma_array_tma_blockwise_scaling.hpp
│   │   │   │   │   │   ├── sm120_mma_tma.hpp
│   │   │   │   │   │   ├── sm120_mma_tma_blockwise_scaling.hpp
│   │   │   │   │   │   ├── sm120_sparse_mma_tma.hpp
│   │   │   │   │   │   ├── sm70_mma_twostage.hpp
│   │   │   │   │   │   ├── sm80_mma_array_multistage.hpp
│   │   │   │   │   │   ├── sm80_mma_multistage.hpp
│   │   │   │   │   │   ├── sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input.hpp
│   │   │   │   │   │   ├── sm90_mma_array_tma_gmma_ss_warpspecialized.hpp
│   │   │   │   │   │   ├── sm90_mma_array_tma_gmma_ss_warpspecialized_fp8.hpp
│   │   │   │   │   │   ├── sm90_mma_array_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp
│   │   │   │   │   │   ├── sm90_mma_multistage_gmma_rs_warpspecialized.hpp
│   │   │   │   │   │   ├── sm90_mma_multistage_gmma_ss_warpspecialized.hpp
│   │   │   │   │   │   ├── sm90_mma_tma_gmma_rs_warpspecialized.hpp
│   │   │   │   │   │   ├── sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp
│   │   │   │   │   │   ├── sm90_mma_tma_gmma_ss.hpp
│   │   │   │   │   │   ├── sm90_mma_tma_gmma_ss_warpspecialized.hpp
│   │   │   │   │   │   ├── sm90_mma_tma_gmma_ss_warpspecialized_fp8.hpp
│   │   │   │   │   │   ├── sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp
│   │   │   │   │   │   ├── sm90_sparse_mma_tma_gmma_ss_warpspecialized.hpp
│   │   │   │   │   │   └── sm90_sparse_mma_tma_gmma_ss_warpspecialized_fp8.hpp
│   │   │   │   │   ├── device
│   │   │   │   │   │   ├── base_grouped.h
│   │   │   │   │   │   ├── default_gemm_configuration.h
│   │   │   │   │   │   ├── ell_gemm.h
│   │   │   │   │   │   ├── gemm.h
│   │   │   │   │   │   ├── gemm_array.h
│   │   │   │   │   │   ├── gemm_batched.h
│   │   │   │   │   │   ├── gemm_complex.h
│   │   │   │   │   │   ├── gemm_grouped.h
│   │   │   │   │   │   ├── gemm_layernorm_mainloop_fusion.h
│   │   │   │   │   │   ├── gemm_sparse.h
│   │   │   │   │   │   ├── gemm_sparse_universal.h
│   │   │   │   │   │   ├── gemm_sparse_universal_with_absmax.h
│   │   │   │   │   │   ├── gemm_sparse_with_absmax.h
│   │   │   │   │   │   ├── gemm_sparse_with_visitor.h
│   │   │   │   │   │   ├── gemm_splitk_parallel.h
│   │   │   │   │   │   ├── gemm_universal.h
│   │   │   │   │   │   ├── gemm_universal_adapter.h
│   │   │   │   │   │   ├── gemm_universal_base.h
│   │   │   │   │   │   ├── gemm_universal_streamk_with_broadcast.h
│   │   │   │   │   │   ├── gemm_universal_with_absmax.h
│   │   │   │   │   │   ├── gemm_universal_with_broadcast.h
│   │   │   │   │   │   ├── gemm_with_k_reduction.h
│   │   │   │   │   │   ├── gemv.h
│   │   │   │   │   │   ├── gemv_blockscaled.h
│   │   │   │   │   │   ├── rank_2k.h
│   │   │   │   │   │   ├── rank_2k_grouped.h
│   │   │   │   │   │   ├── rank_k.h
│   │   │   │   │   │   ├── symm.h
│   │   │   │   │   │   └── trmm.h
│   │   │   │   │   ├── kernel
│   │   │   │   │   │   ├── default_ell_gemm.h
│   │   │   │   │   │   ├── default_gemm.h
│   │   │   │   │   │   ├── default_gemm_complex.h
│   │   │   │   │   │   ├── default_gemm_grouped.h
│   │   │   │   │   │   ├── default_gemm_grouped_per_group_scale.h
│   │   │   │   │   │   ├── default_gemm_grouped_softmax_mainloop_fusion.h
│   │   │   │   │   │   ├── default_gemm_layernorm_mainloop_fusion.h
│   │   │   │   │   │   ├── default_gemm_planar_complex_universal.h
│   │   │   │   │   │   ├── default_gemm_sparse.h
│   │   │   │   │   │   ├── default_gemm_sparse_universal.h
│   │   │   │   │   │   ├── default_gemm_sparse_universal_with_absmax.h
│   │   │   │   │   │   ├── default_gemm_sparse_with_absmax.h
│   │   │   │   │   │   ├── default_gemm_sparse_with_visitor.h
│   │   │   │   │   │   ├── default_gemm_splitk_parallel.h
│   │   │   │   │   │   ├── default_gemm_streamk_with_broadcast.h
│   │   │   │   │   │   ├── default_gemm_universal.h
│   │   │   │   │   │   ├── default_gemm_universal_with_visitor.h
│   │   │   │   │   │   ├── default_gemm_with_absmax.h
│   │   │   │   │   │   ├── default_gemm_with_broadcast.h
│   │   │   │   │   │   ├── default_gemm_with_k_reduction.h
│   │   │   │   │   │   ├── default_gemm_with_reduction.h
│   │   │   │   │   │   ├── default_gemv.h
│   │   │   │   │   │   ├── default_rank_2k.h
│   │   │   │   │   │   ├── default_rank_2k_complex.h
│   │   │   │   │   │   ├── default_rank_2k_grouped.h
│   │   │   │   │   │   ├── default_rank_2k_universal.h
│   │   │   │   │   │   ├── default_rank_k.h
│   │   │   │   │   │   ├── default_rank_k_complex.h
│   │   │   │   │   │   ├── default_rank_k_universal.h
│   │   │   │   │   │   ├── default_symm.h
│   │   │   │   │   │   ├── default_symm_complex.h
│   │   │   │   │   │   ├── default_symm_universal.h
│   │   │   │   │   │   ├── default_trmm.h
│   │   │   │   │   │   ├── default_trmm_complex.h
│   │   │   │   │   │   ├── default_trmm_universal.h
│   │   │   │   │   │   ├── ell_gemm.h
│   │   │   │   │   │   ├── gemm.h
│   │   │   │   │   │   ├── gemm_array.h
│   │   │   │   │   │   ├── gemm_batched.h
│   │   │   │   │   │   ├── gemm_grouped.h
│   │   │   │   │   │   ├── gemm_grouped_per_group_scale.h
│   │   │   │   │   │   ├── gemm_grouped_problem_visitor.h
│   │   │   │   │   │   ├── gemm_grouped_softmax_mainloop_fusion.h
│   │   │   │   │   │   ├── gemm_layernorm_mainloop_fusion.h
│   │   │   │   │   │   ├── gemm_params.h
│   │   │   │   │   │   ├── gemm_pipelined.h
│   │   │   │   │   │   ├── gemm_planar_complex.h
│   │   │   │   │   │   ├── gemm_planar_complex_array.h
│   │   │   │   │   │   ├── gemm_sparse_universal.h
│   │   │   │   │   │   ├── gemm_sparse_universal_with_absmax.h
│   │   │   │   │   │   ├── gemm_splitk_parallel.h
│   │   │   │   │   │   ├── gemm_streamk_with_fused_epilogue.h
│   │   │   │   │   │   ├── gemm_transpose_operands.h
│   │   │   │   │   │   ├── gemm_universal.h
│   │   │   │   │   │   ├── gemm_universal.hpp
│   │   │   │   │   │   ├── gemm_universal_decl.h
│   │   │   │   │   │   ├── gemm_universal_streamk.h
│   │   │   │   │   │   ├── gemm_universal_with_visitor.h
│   │   │   │   │   │   ├── gemm_universal_with_visitor_streamk.h
│   │   │   │   │   │   ├── gemm_with_absmax.h
│   │   │   │   │   │   ├── gemm_with_fused_epilogue.h
│   │   │   │   │   │   ├── gemm_with_k_reduction.h
│   │   │   │   │   │   ├── gemv.h
│   │   │   │   │   │   ├── gemv_batched_strided.h
│   │   │   │   │   │   ├── gemv_blockscaled.h
│   │   │   │   │   │   ├── grouped_problem_visitor.h
│   │   │   │   │   │   ├── params_sparse_base.h
│   │   │   │   │   │   ├── params_universal_base.h
│   │   │   │   │   │   ├── rank_2k_grouped.h
│   │   │   │   │   │   ├── rank_2k_grouped_problem_visitor.h
│   │   │   │   │   │   ├── rank_2k_transpose_operands.h
│   │   │   │   │   │   ├── rank_2k_universal.h
│   │   │   │   │   │   ├── rank_k_universal.h
│   │   │   │   │   │   ├── sm100_gemm_array_tma_warpspecialized.hpp
│   │   │   │   │   │   ├── sm100_gemm_array_tma_warpspecialized_input_transform.hpp
│   │   │   │   │   │   ├── sm100_gemm_array_tma_warpspecialized_mma_transform.hpp
│   │   │   │   │   │   ├── sm100_gemm_cpasync_warpspecialized.hpp
│   │   │   │   │   │   ├── sm100_gemm_mixed_tma_cpasync_warpspecialized.hpp
│   │   │   │   │   │   ├── sm100_gemm_tma_warpspecialized.hpp
│   │   │   │   │   │   ├── sm100_gemm_tma_warpspecialized_input_transform.hpp
│   │   │   │   │   │   ├── sm100_gemm_tma_warpspecialized_mixed_input_transform.hpp
│   │   │   │   │   │   ├── sm100_gemm_tma_warpspecialized_mma_transform.hpp
│   │   │   │   │   │   ├── sm100_sparse_gemm_tma_warpspecialized.hpp
│   │   │   │   │   │   ├── sm100_static_tile_scheduler.hpp
│   │   │   │   │   │   ├── sm100_tile_scheduler.hpp
│   │   │   │   │   │   ├── sm100_tile_scheduler_group.hpp
│   │   │   │   │   │   ├── sm100_tile_scheduler_stream_k.hpp
│   │   │   │   │   │   ├── sm103_blockscaled_gemm_array_tma_warpspecialized.hpp
│   │   │   │   │   │   ├── sm103_blockscaled_gemm_tma_warpspecialized.hpp
│   │   │   │   │   │   ├── sm120_gemm_tma_warpspecialized_cooperative_asymmetric_dma.hpp
│   │   │   │   │   │   ├── sm70_gemm.hpp
│   │   │   │   │   │   ├── sm70_gemm_array.hpp
│   │   │   │   │   │   ├── sm90_gemm_array_tma_warpspecialized_cooperative.hpp
│   │   │   │   │   │   ├── sm90_gemm_array_tma_warpspecialized_pingpong.hpp
│   │   │   │   │   │   ├── sm90_gemm_tma.hpp
│   │   │   │   │   │   ├── sm90_gemm_tma_warpspecialized.hpp
│   │   │   │   │   │   ├── sm90_gemm_tma_warpspecialized_cooperative.hpp
│   │   │   │   │   │   ├── sm90_gemm_tma_warpspecialized_pingpong.hpp
│   │   │   │   │   │   ├── sm90_gemm_warpspecialized.hpp
│   │   │   │   │   │   ├── sm90_gemm_warpspecialized_cooperative.hpp
│   │   │   │   │   │   ├── sm90_gemm_warpspecialized_pingpong.hpp
│   │   │   │   │   │   ├── sm90_tile_scheduler.hpp
│   │   │   │   │   │   ├── sm90_tile_scheduler_group.hpp
│   │   │   │   │   │   ├── sm90_tile_scheduler_stream_k.hpp
│   │   │   │   │   │   ├── sparse_gemm.h
│   │   │   │   │   │   ├── sparse_gemm_with_absmax.h
│   │   │   │   │   │   ├── sparse_gemm_with_visitor.h
│   │   │   │   │   │   ├── static_tile_scheduler.hpp
│   │   │   │   │   │   ├── symm_universal.h
│   │   │   │   │   │   ├── tile_scheduler.hpp
│   │   │   │   │   │   ├── tile_scheduler_detail.hpp
│   │   │   │   │   │   ├── tile_scheduler_params.h
│   │   │   │   │   │   └── trmm_universal.h
│   │   │   │   │   ├── thread
│   │   │   │   │   │   ├── mma.h
│   │   │   │   │   │   ├── mma_sm50.h
│   │   │   │   │   │   ├── mma_sm60.h
│   │   │   │   │   │   └── mma_sm61.h
│   │   │   │   │   ├── threadblock
│   │   │   │   │   │   ├── default_ell_mma.h
│   │   │   │   │   │   ├── default_gemv_core.h
│   │   │   │   │   │   ├── default_mma.h
│   │   │   │   │   │   ├── default_mma_core.h
│   │   │   │   │   │   ├── default_mma_core_simt.h
│   │   │   │   │   │   ├── default_mma_core_sm70.h
│   │   │   │   │   │   ├── default_mma_core_sm75.h
│   │   │   │   │   │   ├── default_mma_core_sm80.h
│   │   │   │   │   │   ├── default_mma_core_sparse_sm80.h
│   │   │   │   │   │   ├── default_mma_core_with_access_size.h
│   │   │   │   │   │   ├── default_mma_core_with_reduction.h
│   │   │   │   │   │   ├── default_mma_core_wmma.h
│   │   │   │   │   │   ├── default_mma_layernorm_mainloop_fusion.h
│   │   │   │   │   │   ├── default_mma_planar_complex_multistage.h
│   │   │   │   │   │   ├── default_mma_planar_complex_pipelined.h
│   │   │   │   │   │   ├── default_mma_softmax_mainloop_fusion.h
│   │   │   │   │   │   ├── default_mma_with_reduction.h
│   │   │   │   │   │   ├── default_multistage_mma_complex.h
│   │   │   │   │   │   ├── default_multistage_mma_complex_core.h
│   │   │   │   │   │   ├── default_multistage_mma_complex_core_sm80.h
│   │   │   │   │   │   ├── default_multistage_trmm_complex.h
│   │   │   │   │   │   ├── default_sparse_mma.h
│   │   │   │   │   │   ├── default_trmm.h
│   │   │   │   │   │   ├── ell_mma_multistage.h
│   │   │   │   │   │   ├── ell_mma_pipelined.h
│   │   │   │   │   │   ├── gemv.h
│   │   │   │   │   │   ├── index_remat.h
│   │   │   │   │   │   ├── mma_base.h
│   │   │   │   │   │   ├── mma_blas3_multistage.h
│   │   │   │   │   │   ├── mma_layernorm_mainloop_fusion_multistage.h
│   │   │   │   │   │   ├── mma_multistage.h
│   │   │   │   │   │   ├── mma_pipelined.h
│   │   │   │   │   │   ├── mma_planar_complex_base.h
│   │   │   │   │   │   ├── mma_planar_complex_multistage.h
│   │   │   │   │   │   ├── mma_planar_complex_pipelined.h
│   │   │   │   │   │   ├── mma_singlestage.h
│   │   │   │   │   │   ├── mma_softmax_mainloop_fusion_multistage.h
│   │   │   │   │   │   ├── mma_sparse_base.h
│   │   │   │   │   │   ├── mma_sparse_multistage.h
│   │   │   │   │   │   ├── mma_with_reduction_multistage.h
│   │   │   │   │   │   ├── threadblock_swizzle.h
│   │   │   │   │   │   └── threadblock_swizzle_streamk.h
│   │   │   │   │   ├── warp
│   │   │   │   │   │   ├── default_mma_complex_tensor_op.h
│   │   │   │   │   │   ├── default_mma_sparse_tensor_op.h
│   │   │   │   │   │   ├── default_mma_tensor_op.h
│   │   │   │   │   │   ├── default_mma_tensor_op_sm80.h
│   │   │   │   │   │   ├── default_mma_with_reduction_tensor_op.h
│   │   │   │   │   │   ├── default_mma_wmma_tensor_op.h
│   │   │   │   │   │   ├── layernorm_scale_bias_transform.h
│   │   │   │   │   │   ├── mma.h
│   │   │   │   │   │   ├── mma_complex_tensor_op.h
│   │   │   │   │   │   ├── mma_complex_tensor_op_fast_f32.h
│   │   │   │   │   │   ├── mma_complex_tensor_op_tile_iterator_sm80.h
│   │   │   │   │   │   ├── mma_gaussian_complex_tensor_op.h
│   │   │   │   │   │   ├── mma_gaussian_complex_tensor_op_tile_iterator_sm80.h
│   │   │   │   │   │   ├── mma_mixed_input_tensor_op.h
│   │   │   │   │   │   ├── mma_planar_complex.h
│   │   │   │   │   │   ├── mma_simt.h
│   │   │   │   │   │   ├── mma_simt_policy.h
│   │   │   │   │   │   ├── mma_simt_tile_iterator.h
│   │   │   │   │   │   ├── mma_sparse_tensor_op.h
│   │   │   │   │   │   ├── mma_tensor_op.h
│   │   │   │   │   │   ├── mma_tensor_op_fast_f32.h
│   │   │   │   │   │   ├── mma_tensor_op_fragment_iterator.h
│   │   │   │   │   │   ├── mma_tensor_op_policy.h
│   │   │   │   │   │   ├── mma_tensor_op_sm70.h
│   │   │   │   │   │   ├── mma_tensor_op_tile_access_iterator.h
│   │   │   │   │   │   ├── mma_tensor_op_tile_iterator.h
│   │   │   │   │   │   ├── mma_tensor_op_tile_iterator_sm70.h
│   │   │   │   │   │   ├── mma_tensor_op_tile_iterator_sm80.h
│   │   │   │   │   │   ├── mma_tensor_op_tile_iterator_sparse.h
│   │   │   │   │   │   ├── mma_tensor_op_tile_iterator_wmma.h
│   │   │   │   │   │   ├── mma_tensor_op_wmma.h
│   │   │   │   │   │   ├── mma_with_reduction_tensor_op.h
│   │   │   │   │   │   ├── scale_bias_tile_iterator.h
│   │   │   │   │   │   ├── softmax_scale_bias_transform.h
│   │   │   │   │   │   └── tile_iterator_planar_complex.h
│   │   │   │   │   ├── dispatch_policy.hpp
│   │   │   │   │   ├── gemm.h
│   │   │   │   │   ├── gemm_enumerated_types.h
│   │   │   │   │   └── group_array_problem_shape.hpp
│   │   │   │   ├── layout
│   │   │   │   │   ├── layout.h
│   │   │   │   │   ├── matrix.h
│   │   │   │   │   ├── permute.h
│   │   │   │   │   ├── pitch_linear.h
│   │   │   │   │   ├── tensor.h
│   │   │   │   │   ├── tensor_op_multiplicand_sm70.h
│   │   │   │   │   ├── tensor_op_multiplicand_sm75.h
│   │   │   │   │   ├── tensor_op_multiplicand_sm80.h
│   │   │   │   │   └── vector.h
│   │   │   │   ├── pipeline
│   │   │   │   │   ├── pipeline.hpp
│   │   │   │   │   ├── sm100_pipeline.hpp
│   │   │   │   │   └── sm90_pipeline.hpp
│   │   │   │   ├── platform
│   │   │   │   │   └── platform.h
│   │   │   │   ├── reduction
│   │   │   │   │   ├── device
│   │   │   │   │   │   ├── reduce_split_k.h
│   │   │   │   │   │   ├── tensor_reduce.h
│   │   │   │   │   │   ├── tensor_reduce_affine_contiguous.h
│   │   │   │   │   │   └── tensor_reduce_affine_strided.h
│   │   │   │   │   ├── kernel
│   │   │   │   │   │   ├── reduce_softmax_final.h
│   │   │   │   │   │   ├── reduce_split_k.h
│   │   │   │   │   │   ├── tensor_reduce_affine_contiguous.h
│   │   │   │   │   │   └── tensor_reduce_affine_strided.h
│   │   │   │   │   ├── thread
│   │   │   │   │   │   ├── reduce.h
│   │   │   │   │   │   └── reduction_operators.h
│   │   │   │   │   └── threadblock_swizzle.h
│   │   │   │   ├── thread
│   │   │   │   │   └── matrix.h
│   │   │   │   ├── transform
│   │   │   │   │   ├── collective
│   │   │   │   │   │   └── sm90_wgmma_transpose.hpp
│   │   │   │   │   ├── device
│   │   │   │   │   │   └── transform_universal_adapter.hpp
│   │   │   │   │   ├── kernel
│   │   │   │   │   │   ├── filter_format_transformer.hpp
│   │   │   │   │   │   ├── sm90_sparse_gemm_compressor.hpp
│   │   │   │   │   │   └── sparse_gemm_compressor.hpp
│   │   │   │   │   ├── thread
│   │   │   │   │   │   ├── transpose.h
│   │   │   │   │   │   └── unary_op.h
│   │   │   │   │   ├── threadblock
│   │   │   │   │   │   ├── ell_iterator.h
│   │   │   │   │   │   ├── ell_predicated_tile_access_iterator.h
│   │   │   │   │   │   ├── ell_predicated_tile_iterator.h
│   │   │   │   │   │   ├── predicated_scale_bias_vector_access_iterator.h
│   │   │   │   │   │   ├── predicated_scale_bias_vector_iterator.h
│   │   │   │   │   │   ├── predicated_tile_access_iterator.h
│   │   │   │   │   │   ├── predicated_tile_access_iterator_2dthreadtile.h
│   │   │   │   │   │   ├── predicated_tile_access_iterator_params.h
│   │   │   │   │   │   ├── predicated_tile_access_iterator_triangular_matrix.h
│   │   │   │   │   │   ├── predicated_tile_iterator.h
│   │   │   │   │   │   ├── predicated_tile_iterator_2dthreadtile.h
│   │   │   │   │   │   ├── predicated_tile_iterator_triangular_matrix.h
│   │   │   │   │   │   ├── predicated_vector_access_iterator.h
│   │   │   │   │   │   ├── regular_scale_bias_vector_access_iterator.h
│   │   │   │   │   │   ├── regular_tile_access_iterator.h
│   │   │   │   │   │   ├── regular_tile_access_iterator_pitch_linear.h
│   │   │   │   │   │   ├── regular_tile_access_iterator_pitch_linear_direct_conv.h
│   │   │   │   │   │   ├── regular_tile_access_iterator_tensor_op.h
│   │   │   │   │   │   ├── regular_tile_access_iterator_tensor_op_sm80.h
│   │   │   │   │   │   ├── regular_tile_iterator.h
│   │   │   │   │   │   ├── regular_tile_iterator_pitch_linear.h
│   │   │   │   │   │   ├── regular_tile_iterator_pitch_linear_2dthreadtile.h
│   │   │   │   │   │   ├── regular_tile_iterator_tensor_op.h
│   │   │   │   │   │   ├── regular_tile_iterator_tensor_op_sm70.h
│   │   │   │   │   │   └── vector_iterator.h
│   │   │   │   │   ├── warp
│   │   │   │   │   │   └── vector_fragment_iterator.h
│   │   │   │   │   └── pitch_linear_thread_map.h
│   │   │   │   ├── aligned_buffer.h
│   │   │   │   ├── array.h
│   │   │   │   ├── array_planar_complex.h
│   │   │   │   ├── array_subbyte.h
│   │   │   │   ├── barrier.h
│   │   │   │   ├── bfloat16.h
│   │   │   │   ├── blas3.h
│   │   │   │   ├── blas3_types.h
│   │   │   │   ├── block_striped.h
│   │   │   │   ├── cluster_launch.hpp
│   │   │   │   ├── complex.h
│   │   │   │   ├── constants.h
│   │   │   │   ├── coord.h
│   │   │   │   ├── core_io.h
│   │   │   │   ├── cuda_host_adapter.hpp
│   │   │   │   ├── cutlass.h
│   │   │   │   ├── device_kernel.h
│   │   │   │   ├── exmy_base.h
│   │   │   │   ├── fast_math.h
│   │   │   │   ├── float8.h
│   │   │   │   ├── float_subbyte.h
│   │   │   │   ├── floating_point_nvrtc.h
│   │   │   │   ├── functional.h
│   │   │   │   ├── gemm_coord.h
│   │   │   │   ├── gemm_coord.hpp
│   │   │   │   ├── half.h
│   │   │   │   ├── integer_subbyte.h
│   │   │   │   ├── kernel_hardware_info.h
│   │   │   │   ├── kernel_hardware_info.hpp
│   │   │   │   ├── kernel_launch.h
│   │   │   │   ├── matrix.h
│   │   │   │   ├── matrix_coord.h
│   │   │   │   ├── matrix_shape.h
│   │   │   │   ├── numeric_conversion.h
│   │   │   │   ├── numeric_size.h
│   │   │   │   ├── numeric_types.h
│   │   │   │   ├── pitch_linear_coord.h
│   │   │   │   ├── predicate_vector.h
│   │   │   │   ├── quaternion.h
│   │   │   │   ├── real.h
│   │   │   │   ├── relatively_equal.h
│   │   │   │   ├── semaphore.h
│   │   │   │   ├── subbyte_reference.h
│   │   │   │   ├── tensor_coord.h
│   │   │   │   ├── tensor_ref.h
│   │   │   │   ├── tensor_ref_planar_complex.h
│   │   │   │   ├── tensor_view.h
│   │   │   │   ├── tensor_view_planar_complex.h
│   │   │   │   ├── tfloat32.h
│   │   │   │   ├── trace.h
│   │   │   │   ├── uint128.h
│   │   │   │   ├── uint256.h
│   │   │   │   ├── version.h
│   │   │   │   ├── wmma_array.h
│   │   │   │   └── workspace.h
│   │   │   └── deep_gemm
│   │   │       ├── common
│   │   │       │   ├── cute_tie.cuh
│   │   │       │   ├── epilogue_utils.cuh
│   │   │       │   ├── reduction.cuh
│   │   │       │   ├── scheduler.cuh
│   │   │       │   ├── sm100_utils.cuh
│   │   │       │   ├── sm90_utils.cuh
│   │   │       │   ├── tma_utils.cuh
│   │   │       │   ├── types.hpp
│   │   │       │   └── utils.cuh
│   │   │       └── impls
│   │   │           ├── sm100_bf16_gemm.cuh
│   │   │           ├── sm100_bmk_bnk_mn.cuh
│   │   │           ├── sm100_fp8_gemm_1d1d.cuh
│   │   │           ├── sm100_fp8_mqa_logits.cuh
│   │   │           ├── sm100_fp8_paged_mqa_logits.cuh
│   │   │           ├── sm100_tf32_hc_prenorm_gemm.cuh
│   │   │           ├── sm90_bf16_gemm.cuh
│   │   │           ├── sm90_bmk_bnk_mn.cuh
│   │   │           ├── sm90_fp8_gemm_1d1d.cuh
│   │   │           ├── sm90_fp8_gemm_1d2d.cuh
│   │   │           ├── sm90_fp8_mqa_logits.cuh
│   │   │           ├── sm90_fp8_paged_mqa_logits.cuh
│   │   │           ├── sm90_tf32_hc_prenorm_gemm.cuh
│   │   │           ├── smxx_clean_logits.cuh
│   │   │           └── smxx_layout.cuh
│   │   ├── legacy
│   │   │   ├── __init__.py
│   │   │   ├── a_fused_k_grouped_gemm.py
│   │   │   ├── a_fused_m_grouped_gemm.py
│   │   │   ├── b_fused_k_grouped_gemm.py
│   │   │   ├── m_grouped_gemm.py
│   │   │   └── tune_options.py
│   │   ├── testing
│   │   │   ├── __init__.py
│   │   │   ├── bench.py
│   │   │   ├── numeric.py
│   │   │   └── utils.py
│   │   ├── utils
│   │   │   ├── __init__.py
│   │   │   ├── layout.py
│   │   │   └── math.py
│   │   ├── __init__.py
│   │   ├── _C.cpython-312-x86_64-linux-gnu.so
│   │   └── envs.py
│   ├── flashmla
│   │   ├── __init__.py
│   │   └── flash_mla_interface.py
│   ├── triton_kernels
│   │   ├── compaction_details
│   │   │   └── _masked_compaction.py
│   │   ├── matmul_ogs_details
│   │   │   ├── opt_flags_details
│   │   │   │   ├── opt_flags_amd.py
│   │   │   │   └── opt_flags_nvidia.py
│   │   │   ├── _common.py
│   │   │   ├── _matmul_ogs.py
│   │   │   ├── _p_matmul_ogs.py
│   │   │   └── opt_flags.py
│   │   ├── numerics_details
│   │   │   ├── mxfp_details
│   │   │   │   ├── _downcast_to_mxfp.py
│   │   │   │   └── _upcast_from_mxfp.py
│   │   │   ├── __init__.py
│   │   │   ├── flexpoint.py
│   │   │   └── mxfp.py
│   │   ├── swiglu_details
│   │   │   └── _swiglu.py
│   │   ├── tensor_details
│   │   │   ├── bitmatrix_details
│   │   │   │   └── sum_bitmatrix_rows.py
│   │   │   ├── layout_details
│   │   │   │   ├── base.py
│   │   │   │   ├── blackwell_scale.py
│   │   │   │   ├── blackwell_value.py
│   │   │   │   ├── cdna4_scale.py
│   │   │   │   ├── hopper_scale.py
│   │   │   │   ├── hopper_value.py
│   │   │   │   └── strided.py
│   │   │   ├── bitmatrix.py
│   │   │   ├── layout.py
│   │   │   └── ragged_tensor.py
│   │   ├── topk_details
│   │   │   ├── __init__.py
│   │   │   ├── _topk_backward.py
│   │   │   └── _topk_forward.py
│   │   ├── __init__.py
│   │   ├── compaction.py
│   │   ├── distributed.py
│   │   ├── matmul_ogs.py
│   │   ├── numerics.py
│   │   ├── proton_opts.py
│   │   ├── reduce.py
│   │   ├── roofline.py
│   │   ├── specialize.py
│   │   ├── swiglu.py
│   │   ├── target_info.py
│   │   ├── tensor.py
│   │   ├── testing.py
│   │   └── topk.py
│   ├── __init__.py
│   └── pynvml.py
├── tokenizers
│   ├── __init__.py
│   ├── deepseek_v32.py
│   ├── deepseek_v32_encoding.py
│   ├── detokenizer_utils.py
│   ├── grok2.py
│   ├── hf.py
│   ├── kimi_audio.py
│   ├── mistral.py
│   ├── protocol.py
│   ├── qwen_vl.py
│   └── registry.py
├── tool_parsers
│   ├── __init__.py
│   ├── abstract_tool_parser.py
│   ├── deepseekv31_tool_parser.py
│   ├── deepseekv32_tool_parser.py
│   ├── deepseekv3_tool_parser.py
│   ├── ernie45_tool_parser.py
│   ├── functiongemma_tool_parser.py
│   ├── gemma4_tool_parser.py
│   ├── gemma4_utils.py
│   ├── gigachat3_tool_parser.py
│   ├── glm47_moe_tool_parser.py
│   ├── glm4_moe_tool_parser.py
│   ├── granite4_tool_parser.py
│   ├── granite_20b_fc_tool_parser.py
│   ├── granite_tool_parser.py
│   ├── hermes_tool_parser.py
│   ├── hunyuan_a13b_tool_parser.py
│   ├── hy_v3_tool_parser.py
│   ├── internlm2_tool_parser.py
│   ├── jamba_tool_parser.py
│   ├── kimi_k2_tool_parser.py
│   ├── llama4_pythonic_tool_parser.py
│   ├── llama_tool_parser.py
│   ├── longcat_tool_parser.py
│   ├── minimax_m2_tool_parser.py
│   ├── minimax_tool_parser.py
│   ├── mistral_tool_parser.py
│   ├── olmo3_tool_parser.py
│   ├── openai_tool_parser.py
│   ├── phi4mini_tool_parser.py
│   ├── pythonic_tool_parser.py
│   ├── qwen3coder_tool_parser.py
│   ├── qwen3xml_tool_parser.py
│   ├── seed_oss_tool_parser.py
│   ├── step3_tool_parser.py
│   ├── step3p5_tool_parser.py
│   ├── utils.py
│   └── xlam_tool_parser.py
├── tracing
│   ├── __init__.py
│   ├── otel.py
│   └── utils.py
├── transformers_utils
│   ├── chat_templates
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   ├── template_basic.jinja
│   │   ├── template_blip2.jinja
│   │   ├── template_chatml.jinja
│   │   ├── template_deepseek_ocr.jinja
│   │   ├── template_deepseek_vl2.jinja
│   │   ├── template_fuyu.jinja
│   │   ├── template_kimi_audio.jinja
│   │   └── template_minicpmv45.jinja
│   ├── configs
│   │   ├── speculators
│   │   │   ├── __init__.py
│   │   │   ├── algos.py
│   │   │   └── base.py
│   │   ├── __init__.py
│   │   ├── afmoe.py
│   │   ├── arctic.py
│   │   ├── AXK1.py
│   │   ├── bagel.py
│   │   ├── chatglm.py
│   │   ├── cheers.py
│   │   ├── colmodernvbert.py
│   │   ├── colpali.py
│   │   ├── colqwen3.py
│   │   ├── deepseek_vl2.py
│   │   ├── dotsocr.py
│   │   ├── eagle.py
│   │   ├── extract_hidden_states.py
│   │   ├── falcon.py
│   │   ├── fireredlid.py
│   │   ├── flex_olmo.py
│   │   ├── funaudiochat.py
│   │   ├── granite4_vision.py
│   │   ├── hunyuan_vl.py
│   │   ├── hy_v3.py
│   │   ├── hyperclovax.py
│   │   ├── isaac.py
│   │   ├── jais.py
│   │   ├── kimi_k25.py
│   │   ├── kimi_linear.py
│   │   ├── kimi_vl.py
│   │   ├── lfm2_moe.py
│   │   ├── medusa.py
│   │   ├── midashenglm.py
│   │   ├── mistral.py
│   │   ├── mlp_speculator.py
│   │   ├── moonvit.py
│   │   ├── nemotron.py
│   │   ├── nemotron_h.py
│   │   ├── olmo_hybrid.py
│   │   ├── ovis.py
│   │   ├── parakeet.py
│   │   ├── qwen3_5.py
│   │   ├── qwen3_5_moe.py
│   │   ├── qwen3_asr.py
│   │   ├── qwen3_next.py
│   │   ├── radio.py
│   │   ├── step3_vl.py
│   │   ├── step3p5.py
│   │   ├── tarsier2.py
│   │   └── ultravox.py
│   ├── processors
│   │   ├── __init__.py
│   │   ├── bagel.py
│   │   ├── cheers.py
│   │   ├── cohere_asr.py
│   │   ├── deepseek_ocr.py
│   │   ├── deepseek_vl2.py
│   │   ├── fireredasr2.py
│   │   ├── fireredlid.py
│   │   ├── funasr.py
│   │   ├── glm4v.py
│   │   ├── granite4_vision.py
│   │   ├── h2ovl.py
│   │   ├── hunyuan_vl.py
│   │   ├── hunyuan_vl_image.py
│   │   ├── internvl.py
│   │   ├── isaac.py
│   │   ├── kimi_audio.py
│   │   ├── kimi_k25.py
│   │   ├── nano_nemotron_vl.py
│   │   ├── nemotron_vl.py
│   │   ├── nvlm_d.py
│   │   ├── ovis.py
│   │   ├── ovis2_5.py
│   │   ├── pixtral.py
│   │   ├── qwen3_asr.py
│   │   ├── qwen_vl.py
│   │   ├── step3_vl.py
│   │   └── voxtral.py
│   ├── __init__.py
│   ├── config.py
│   ├── config_parser_base.py
│   ├── dynamic_module.py
│   ├── gguf_utils.py
│   ├── model_arch_config_convertor.py
│   ├── processor.py
│   ├── repo_utils.py
│   ├── runai_utils.py
│   ├── s3_utils.py
│   ├── tokenizer.py
│   └── utils.py
├── triton_utils
│   ├── __init__.py
│   ├── allocation.py
│   └── importing.py
├── usage
│   ├── __init__.py
│   └── usage_lib.py
├── utils
│   ├── __init__.py
│   ├── argparse_utils.py
│   ├── async_utils.py
│   ├── cache.py
│   ├── collection_utils.py
│   ├── counter.py
│   ├── cpu_resource_utils.py
│   ├── cpu_triton_utils.py
│   ├── deep_gemm.py
│   ├── flashinfer.py
│   ├── func_utils.py
│   ├── gc_utils.py
│   ├── hashing.py
│   ├── import_utils.py
│   ├── jsontree.py
│   ├── math_utils.py
│   ├── mem_constants.py
│   ├── mem_utils.py
│   ├── mistral.py
│   ├── multi_stream_utils.py
│   ├── nccl.py
│   ├── network_utils.py
│   ├── numa_utils.py
│   ├── numa_wrapper.sh
│   ├── nvtx_pytorch_hooks.py
│   ├── ompmultiprocessing.py
│   ├── platform_utils.py
│   ├── print_utils.py
│   ├── profiling.py
│   ├── registry.py
│   ├── serial_utils.py
│   ├── system_utils.py
│   ├── tensor_schema.py
│   ├── torch_utils.py
│   └── tqdm_utils.py
├── v1
│   ├── attention
│   │   ├── backends
│   │   │   ├── mla
│   │   │   │   ├── __init__.py
│   │   │   │   ├── aiter_triton_mla.py
│   │   │   │   ├── cutlass_mla.py
│   │   │   │   ├── flashattn_mla.py
│   │   │   │   ├── flashinfer_mla.py
│   │   │   │   ├── flashinfer_mla_sparse.py
│   │   │   │   ├── flashmla.py
│   │   │   │   ├── flashmla_sparse.py
│   │   │   │   ├── indexer.py
│   │   │   │   ├── rocm_aiter_mla.py
│   │   │   │   ├── rocm_aiter_mla_sparse.py
│   │   │   │   ├── sparse_utils.py
│   │   │   │   ├── triton_mla.py
│   │   │   │   └── xpu_mla_sparse.py
│   │   │   ├── __init__.py
│   │   │   ├── cpu_attn.py
│   │   │   ├── fa_utils.py
│   │   │   ├── flash_attn.py
│   │   │   ├── flash_attn_diffkv.py
│   │   │   ├── flashinfer.py
│   │   │   ├── flex_attention.py
│   │   │   ├── gdn_attn.py
│   │   │   ├── linear_attn.py
│   │   │   ├── mamba1_attn.py
│   │   │   ├── mamba2_attn.py
│   │   │   ├── mamba_attn.py
│   │   │   ├── registry.py
│   │   │   ├── rocm_aiter_fa.py
│   │   │   ├── rocm_aiter_unified_attn.py
│   │   │   ├── rocm_attn.py
│   │   │   ├── short_conv_attn.py
│   │   │   ├── tree_attn.py
│   │   │   ├── triton_attn.py
│   │   │   ├── turboquant_attn.py
│   │   │   └── utils.py
│   │   ├── ops
│   │   │   ├── __init__.py
│   │   │   ├── chunked_prefill_paged_decode.py
│   │   │   ├── common.py
│   │   │   ├── dcp_alltoall.py
│   │   │   ├── flashmla.py
│   │   │   ├── merge_attn_states.py
│   │   │   ├── paged_attn.py
│   │   │   ├── prefix_prefill.py
│   │   │   ├── rocm_aiter_mla_sparse.py
│   │   │   ├── triton_attention_helpers.py
│   │   │   ├── triton_decode_attention.py
│   │   │   ├── triton_merge_attn_states.py
│   │   │   ├── triton_prefill_attention.py
│   │   │   ├── triton_reshape_and_cache_flash.py
│   │   │   ├── triton_turboquant_decode.py
│   │   │   ├── triton_turboquant_store.py
│   │   │   ├── triton_unified_attention.py
│   │   │   ├── vit_attn_wrappers.py
│   │   │   └── xpu_mla_sparse.py
│   │   ├── __init__.py
│   │   ├── backend.py
│   │   └── selector.py
│   ├── core
│   │   ├── sched
│   │   │   ├── __init__.py
│   │   │   ├── async_scheduler.py
│   │   │   ├── interface.py
│   │   │   ├── output.py
│   │   │   ├── request_queue.py
│   │   │   ├── scheduler.py
│   │   │   └── utils.py
│   │   ├── __init__.py
│   │   ├── block_pool.py
│   │   ├── encoder_cache_manager.py
│   │   ├── kv_cache_coordinator.py
│   │   ├── kv_cache_manager.py
│   │   ├── kv_cache_metrics.py
│   │   ├── kv_cache_utils.py
│   │   └── single_type_kv_cache_manager.py
│   ├── engine
│   │   ├── __init__.py
│   │   ├── async_llm.py
│   │   ├── coordinator.py
│   │   ├── core.py
│   │   ├── core_client.py
│   │   ├── detokenizer.py
│   │   ├── exceptions.py
│   │   ├── input_processor.py
│   │   ├── llm_engine.py
│   │   ├── logprobs.py
│   │   ├── output_processor.py
│   │   ├── parallel_sampling.py
│   │   ├── tensor_ipc.py
│   │   └── utils.py
│   ├── executor
│   │   ├── __init__.py
│   │   ├── abstract.py
│   │   ├── multiproc_executor.py
│   │   ├── ray_env_utils.py
│   │   ├── ray_executor.py
│   │   ├── ray_executor_v2.py
│   │   ├── ray_utils.py
│   │   └── uniproc_executor.py
│   ├── kv_offload
│   │   ├── cpu
│   │   │   ├── policies
│   │   │   │   ├── __init__.py
│   │   │   │   ├── abstract.py
│   │   │   │   ├── arc.py
│   │   │   │   └── lru.py
│   │   │   ├── __init__.py
│   │   │   ├── manager.py
│   │   │   ├── shared_offload_region.py
│   │   │   └── spec.py
│   │   ├── worker
│   │   │   ├── __init__.py
│   │   │   ├── cpu_gpu.py
│   │   │   └── worker.py
│   │   ├── __init__.py
│   │   ├── abstract.py
│   │   ├── factory.py
│   │   ├── mediums.py
│   │   ├── reuse_manager.py
│   │   └── spec.py
│   ├── metrics
│   │   ├── __init__.py
│   │   ├── loggers.py
│   │   ├── perf.py
│   │   ├── prometheus.py
│   │   ├── ray_wrappers.py
│   │   ├── reader.py
│   │   ├── stats.py
│   │   └── utils.py
│   ├── pool
│   │   ├── __init__.py
│   │   ├── late_interaction.py
│   │   └── metadata.py
│   ├── sample
│   │   ├── logits_processor
│   │   │   ├── __init__.py
│   │   │   ├── builtin.py
│   │   │   ├── interface.py
│   │   │   └── state.py
│   │   ├── ops
│   │   │   ├── __init__.py
│   │   │   ├── bad_words.py
│   │   │   ├── logprobs.py
│   │   │   ├── penalties.py
│   │   │   ├── topk_topp_sampler.py
│   │   │   └── topk_topp_triton.py
│   │   ├── __init__.py
│   │   ├── metadata.py
│   │   ├── rejection_sampler.py
│   │   └── sampler.py
│   ├── simple_kv_offload
│   │   ├── __init__.py
│   │   ├── copy_backend.py
│   │   ├── cuda_mem_ops.py
│   │   ├── manager.py
│   │   ├── metadata.py
│   │   └── worker.py
│   ├── spec_decode
│   │   ├── __init__.py
│   │   ├── dflash.py
│   │   ├── draft_model.py
│   │   ├── eagle.py
│   │   ├── extract_hidden_states.py
│   │   ├── llm_base_proposer.py
│   │   ├── medusa.py
│   │   ├── metadata.py
│   │   ├── metrics.py
│   │   ├── ngram_proposer.py
│   │   ├── ngram_proposer_gpu.py
│   │   ├── suffix_decoding.py
│   │   └── utils.py
│   ├── structured_output
│   │   ├── __init__.py
│   │   ├── backend_guidance.py
│   │   ├── backend_lm_format_enforcer.py
│   │   ├── backend_outlines.py
│   │   ├── backend_types.py
│   │   ├── backend_xgrammar.py
│   │   ├── request.py
│   │   └── utils.py
│   ├── worker
│   │   ├── gpu
│   │   │   ├── metrics
│   │   │   │   ├── __init__.py
│   │   │   │   └── logits.py
│   │   │   ├── mm
│   │   │   │   ├── __init__.py
│   │   │   │   ├── encoder_cache.py
│   │   │   │   ├── encoder_runner.py
│   │   │   │   └── rope.py
│   │   │   ├── model_states
│   │   │   │   ├── __init__.py
│   │   │   │   ├── default.py
│   │   │   │   ├── interface.py
│   │   │   │   └── whisper.py
│   │   │   ├── pool
│   │   │   │   ├── __init__.py
│   │   │   │   ├── late_interaction_runner.py
│   │   │   │   └── pooling_runner.py
│   │   │   ├── sample
│   │   │   │   ├── __init__.py
│   │   │   │   ├── bad_words.py
│   │   │   │   ├── gumbel.py
│   │   │   │   ├── logit_bias.py
│   │   │   │   ├── logprob.py
│   │   │   │   ├── min_p.py
│   │   │   │   ├── output.py
│   │   │   │   ├── penalties.py
│   │   │   │   ├── prompt_logprob.py
│   │   │   │   ├── sampler.py
│   │   │   │   └── states.py
│   │   │   ├── spec_decode
│   │   │   │   ├── eagle
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── cudagraph.py
│   │   │   │   │   ├── eagle3_utils.py
│   │   │   │   │   ├── speculator.py
│   │   │   │   │   └── utils.py
│   │   │   │   ├── __init__.py
│   │   │   │   ├── probabilistic_rejection_sampler_utils.py
│   │   │   │   ├── rejection_sampler.py
│   │   │   │   ├── synthetic_rejection_sampler_utils.py
│   │   │   │   └── utils.py
│   │   │   ├── __init__.py
│   │   │   ├── async_utils.py
│   │   │   ├── attn_utils.py
│   │   │   ├── block_table.py
│   │   │   ├── buffer_utils.py
│   │   │   ├── cp_utils.py
│   │   │   ├── cudagraph_utils.py
│   │   │   ├── dp_utils.py
│   │   │   ├── eplb_utils.py
│   │   │   ├── input_batch.py
│   │   │   ├── kv_connector.py
│   │   │   ├── lora_utils.py
│   │   │   ├── model_runner.py
│   │   │   ├── pp_utils.py
│   │   │   ├── README.md
│   │   │   ├── states.py
│   │   │   ├── structured_outputs.py
│   │   │   └── warmup.py
│   │   ├── __init__.py
│   │   ├── block_table.py
│   │   ├── cp_utils.py
│   │   ├── cpu_model_runner.py
│   │   ├── cpu_worker.py
│   │   ├── dp_utils.py
│   │   ├── ec_connector_model_runner_mixin.py
│   │   ├── encoder_cudagraph.py
│   │   ├── encoder_cudagraph_defs.py
│   │   ├── gpu_input_batch.py
│   │   ├── gpu_model_runner.py
│   │   ├── gpu_ubatch_wrapper.py
│   │   ├── gpu_worker.py
│   │   ├── kv_connector_model_runner_mixin.py
│   │   ├── lora_model_runner_mixin.py
│   │   ├── mamba_utils.py
│   │   ├── tpu_input_batch.py
│   │   ├── ubatch_utils.py
│   │   ├── ubatching.py
│   │   ├── utils.py
│   │   ├── worker_base.py
│   │   ├── workspace.py
│   │   ├── xpu_model_runner.py
│   │   └── xpu_worker.py
│   ├── __init__.py
│   ├── cudagraph_dispatcher.py
│   ├── kv_cache_interface.py
│   ├── outputs.py
│   ├── request.py
│   ├── serial_utils.py
│   └── utils.py
├── vllm_flash_attn
│   ├── cute
│   │   ├── __init__.py
│   │   ├── ampere_helpers.py
│   │   ├── barrier.py
│   │   ├── bench_utils.py
│   │   ├── benchmark.py
│   │   ├── blackwell_helpers.py
│   │   ├── block_info.py
│   │   ├── block_sparse_utils.py
│   │   ├── block_sparsity.py
│   │   ├── cache_utils.py
│   │   ├── compute_block_sparsity.py
│   │   ├── copy_utils.py
│   │   ├── cute_dsl_ptxas.py
│   │   ├── cute_dsl_utils.py
│   │   ├── fa_logging.py
│   │   ├── fast_math.py
│   │   ├── flash_bwd.py
│   │   ├── flash_bwd_postprocess.py
│   │   ├── flash_bwd_preprocess.py
│   │   ├── flash_bwd_sm100.py
│   │   ├── flash_bwd_sm120.py
│   │   ├── flash_bwd_sm90.py
│   │   ├── flash_fwd.py
│   │   ├── flash_fwd_combine.py
│   │   ├── flash_fwd_sm100.py
│   │   ├── flash_fwd_sm120.py
│   │   ├── flash_fwd_sm90.py
│   │   ├── hopper_helpers.py
│   │   ├── interface.py
│   │   ├── mask.py
│   │   ├── mma_sm100_desc.py
│   │   ├── named_barrier.py
│   │   ├── pack_gqa.py
│   │   ├── paged_kv.py
│   │   ├── pipeline.py
│   │   ├── seqlen_info.py
│   │   ├── sm90_config_search.py
│   │   ├── softmax.py
│   │   ├── testing.py
│   │   ├── tile_scheduler.py
│   │   └── utils.py
│   ├── layers
│   │   └── rotary.py
│   ├── ops
│   │   └── triton
│   │       └── rotary.py
│   ├── .gitkeep
│   ├── __init__.py
│   ├── _vllm_fa2_C.abi3.so
│   ├── _vllm_fa3_C.abi3.so
│   └── flash_attn_interface.py
├── __init__.py
├── _aiter_ops.py
├── _C.abi3.so
├── _C_stable_libtorch.abi3.so
├── _custom_ops.py
├── _flashmla_C.abi3.so
├── _flashmla_extension_C.abi3.so
├── _moe_C.abi3.so
├── _oink_ops.py
├── _version.py
├── _xpu_ops.py
├── beam_search.py
├── collect_env.py
├── connections.py
├── cumem_allocator.abi3.so
├── env_override.py
├── envs.py
├── exceptions.py
├── forward_context.py
├── logger.py
├── logits_process.py
├── logprobs.py
├── model_inspection.py
├── outputs.py
├── pooling_params.py
├── py.typed
├── sampling_params.py
├── scalar_type.py
├── scripts.py
├── sequence.py
├── tasks.py
└── version.py

```



