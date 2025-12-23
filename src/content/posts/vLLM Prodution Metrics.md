---
title: vLLM Production Metrics
published: 2025-12-19
description: "vLLM Production Metrics"
image: "./cover.jpeg"
tags: ["Blogging", "vLLM", "vLLM Production Metrics"]
category: Guides
draft: false
---

# vLLM Production Metrics

## I. System Overview and Goals

- vLLM exposes runtime metrics for monitoring inference systems.
- vLLM 暴露运行时指标用于监控推理系统。
- These metrics reflect performance, stability, and resource usage.
- 这些指标反映性能、稳定性和资源使用情况。
- Metrics are served through an HTTP endpoint.
- 指标通过 HTTP 端点提供。
- The endpoint follows the Prometheus format.
- 该端点遵循 Prometheus 格式。
- The endpoint path is `/metrics`.
- 端点路径是 `/metrics`。

## II. Metrics Exposure and Collection

### Metrics Endpoint

- vLLM starts a server with metrics enabled by default.
- vLLM 默认启动时启用指标服务。
- Metrics are exposed on the OpenAI-compatible API server.
- 指标通过兼容 OpenAI 的 API 服务器暴露。
- Prometheus scrapes the metrics periodically.
- Prometheus 定期拉取这些指标。

```bash
vllm serve unsloth/Llama-3.2-1B-Instruct
```

- This command starts the vLLM server.
- 该命令启动 vLLM 服务器。

```bash
curl http://localhost:8000/metrics
```

- This command queries the metrics endpoint.
- 该命令查询指标端点。

### Metric Types

- Counters track cumulative events.
- 计数器跟踪累计事件。
- Gauges track current states.
- 仪表盘指标跟踪当前状态。
- Histograms track distributions.
- 直方图跟踪分布情况。

## III. Core Metric Categories

### Request and Token Metrics

- `vllm:request_success` counts successful requests.
- `vllm:request_success` 统计成功请求数。
- `vllm:prompt_tokens` counts prefill tokens.
- `vllm:prompt_tokens` 统计预填充 token 数。
- `vllm:generation_tokens` counts generated tokens.
- `vllm:generation_tokens` 统计生成的 token 数。
- These metrics measure workload size.
- 这些指标衡量工作负载规模。

### Latency Metrics

- `vllm:e2e_request_latency_seconds` measures total request time.
- `vllm:e2e_request_latency_seconds` 测量请求总耗时。
- `vllm:time_to_first_token_seconds` measures first token delay.
- `vllm:time_to_first_token_seconds` 测量首 token 延迟。
- `vllm:inter_token_latency_seconds` measures token spacing.
- `vllm:inter_token_latency_seconds` 测量 token 间隔时间。
- These metrics reflect user experience.
- 这些指标反映用户体验。

### Scheduler and Queue Metrics

- `vllm:num_requests_running` shows active requests.
- `vllm:num_requests_running` 显示正在运行的请求数。
- `vllm:num_requests_waiting` shows queued requests.
- `vllm:num_requests_waiting` 显示等待中的请求数。
- `vllm:request_queue_time_seconds` measures wait time.
- `vllm:request_queue_time_seconds` 测量排队时间。
- These metrics show system pressure.
- 这些指标反映系统压力。

### KV Cache Metrics

- `vllm:kv_cache_usage_perc` shows cache usage ratio.
- `vllm:kv_cache_usage_perc` 显示 KV 缓存使用率。
- `vllm:prefix_cache_hits` counts cached tokens reused.
- `vllm:prefix_cache_hits` 统计复用的缓存 token 数。
- `vllm:prefix_cache_queries` counts cache lookups.
- `vllm:prefix_cache_queries` 统计缓存查询次数。
- These metrics measure memory efficiency.
- 这些指标衡量内存效率。

### Speculative Decoding Metrics

- `vllm:spec_decode_num_drafts` counts draft steps.
- `vllm:spec_decode_num_drafts` 统计草稿步骤数。
- `vllm:spec_decode_num_accepted_tokens` counts accepted tokens.
- `vllm:spec_decode_num_accepted_tokens` 统计被接受的 token 数。
- These metrics show speculation effectiveness.
- 这些指标反映推测解码效果。

### NIXL KV Connector Metrics

- These metrics track cross-instance KV transfer.
- 这些指标跟踪跨实例 KV 传输。
- `vllm:nixl_num_failed_transfers` counts failed transfers.
- `vllm:nixl_num_failed_transfers` 统计失败的传输次数。
- `vllm:nixl_xfer_time_seconds` measures transfer duration.
- `vllm:nixl_xfer_time_seconds` 测量传输耗时。
- These metrics detect distributed failures.
- 这些指标用于发现分布式故障。

## IV. Usage and Operational Practices

### Monitoring Integration

- Prometheus scrapes metrics from vLLM.
- Prometheus 从 vLLM 拉取指标。
- Grafana visualizes time series data.
- Grafana 可视化时间序列数据。
- Alerts trigger on threshold breaches.
- 当超过阈值时触发告警。

### Performance Diagnosis

- High queue time indicates overload.
- 排队时间高表示系统过载。
- High inter-token latency indicates slow decoding.
- token 间延迟高表示解码变慢。
- High KV usage indicates memory pressure.
- KV 使用率高表示内存压力大。

### Versioning and Deprecation

- Metrics follow a deprecation policy.
- 指标遵循弃用策略。
- Deprecated metrics are hidden in later versions.
- 被弃用的指标在后续版本中隐藏。
- Hidden metrics can be temporarily re-enabled.
- 隐藏指标可以临时重新启用。
- Deprecated metrics are eventually removed.
- 被弃用的指标最终会被移除。