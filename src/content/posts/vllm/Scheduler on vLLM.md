---
title: "Scheduler on vLLM"
published: 2026-02-01
description: "Scheduler on vLLM"
image: ""
tags: ["vllm","Scheduler on vLLM"]
category: vllm
draft: false
lang: ""
---





```python
# SPDX-License-Identifier: Apache-2.0  # 开源协议声明

import enum
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Iterable, List, Optional
from typing import Sequence as GenericSequence
from typing import Set, Tuple, Union

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupBase, SequenceGroupMetadata,
                           SequenceGroupMetadataDelta, SequenceStage,
                           SequenceStatus)
from vllm.utils import Device, PyObjectCache

logger = init_logger(__name__)  # 初始化日志模块

# Test-only. If configured, decode is preempted with
# ARTIFICIAL_PREEMPTION_PROB% probability.
# 测试专用：如果环境变量开启，则 decode 阶段会以一定概率人为触发抢占(preempt)
ENABLE_ARTIFICIAL_PREEMPT = bool(
    os.getenv("VLLM_TEST_ENABLE_ARTIFICIAL_PREEMPT", False))  # noqa
ARTIFICIAL_PREEMPTION_PROB = 0.5  # 人工抢占概率 50%
ARTIFICIAL_PREEMPTION_MAX_CNT = 500  # 最多人工抢占次数


class PreemptionMode(enum.Enum):
    """Preemption modes.

    1. Swapping: 将被抢占的 seq 的 KV blocks swap 到 CPU，恢复时再 swap 回 GPU
    2. Recomputation: 直接丢弃被抢占的 seq 的 KV blocks，恢复时从头 recompute（当新 prompt）
    """

    SWAP = enum.auto()       # swap 模式
    RECOMPUTE = enum.auto()  # 重算模式


@dataclass
class SchedulingBudget:
    """The available slots for scheduling.

    TODO(sang): budget 当前是 request_id-aware 的：
      - 同一个 request_id 多次 add 不会重复计入 budget
      - 原因是正常调度路径里可能会提前更新 RUNNING 的 num_seqs
    """

    token_budget: int  # 本轮调度允许的 token 总预算（batch token 上限）
    max_num_seqs: int  # 本轮调度允许的 seq 数量上限
    _request_ids_num_batched_tokens: Set[str] = field(default_factory=set)  # 已统计过 token 的 request_id 集合
    _request_ids_num_curr_seqs: Set[str] = field(default_factory=set)       # 已统计过 seq 数的 request_id 集合
    # Number of cached tokens in the batch.
    _num_cached_tokens: int = 0  # batch 内 prefix cache 命中（缓存）token 数
    # Number of actual non-cached tokens in the batch.
    _num_batched_tokens: int = 0  # batch 内真正要算的 token 数（不含 cached）
    _num_curr_seqs: int = 0       # batch 内当前 seq 的数量

    def can_schedule(self, *, num_new_tokens: int, num_new_seqs: int):
        # num_new_tokens 可以为 0（代表全命中 cache）
        assert num_new_tokens >= 0
        assert num_new_seqs != 0
        # 判断新增 token / seq 是否还在预算范围内
        return (self.num_batched_tokens + num_new_tokens <= self.token_budget
                and self.num_curr_seqs + num_new_seqs <= self.max_num_seqs)

    def remaining_token_budget(self):
        # 剩余 token 预算
        return self.token_budget - self.num_batched_tokens

    def add_num_batched_tokens(self,
                               req_id: str,
                               num_batched_tokens: int,
                               num_cached_tokens: int = 0):
        # 同一个 request_id 只统计一次，避免重复记账
        if req_id in self._request_ids_num_batched_tokens:
            return
        assert num_cached_tokens >= 0
        assert num_batched_tokens >= 0

        # 标记该 req_id 已被统计
        self._request_ids_num_batched_tokens.add(req_id)
        # 更新非缓存 token 数
        self._num_batched_tokens += num_batched_tokens
        # 更新缓存 token 数
        self._num_cached_tokens += num_cached_tokens

    def subtract_num_batched_tokens(self, req_id: str,
                                    num_batched_tokens: int):
        # 从预算里移除某个 request 的 token 统计（通常用于 preempt 回退）
        if req_id in self._request_ids_num_batched_tokens:
            self._request_ids_num_batched_tokens.remove(req_id)
            self._num_batched_tokens -= num_batched_tokens

    def add_num_seqs(self, req_id: str, num_curr_seqs: int):
        # 同一个 request_id 只统计一次 seq 数
        if req_id in self._request_ids_num_curr_seqs:
            return

        self._request_ids_num_curr_seqs.add(req_id)
        self._num_curr_seqs += num_curr_seqs

    def subtract_num_seqs(self, req_id: str, num_curr_seqs: int):
        # 从预算里移除某个 request 的 seq 统计（通常用于 preempt 回退）
        if req_id in self._request_ids_num_curr_seqs:
            self._request_ids_num_curr_seqs.remove(req_id)
            self._num_curr_seqs -= num_curr_seqs

    @property
    def num_batched_tokens(self):
        # 当前 batch 真实需要算的 token 数
        return self._num_batched_tokens

    @property
    def num_curr_seqs(self):
        # 当前 batch seq 数
        return self._num_curr_seqs

    @property
    def num_cached_tokens(self):
        # 当前 batch cache 命中的 token 数
        return self._num_cached_tokens


@dataclass
class ScheduledSequenceGroup:
    # 被调度的 seq_group
    seq_group: SequenceGroup
    # 下一轮要处理的 token chunk 大小：
    # - decode 时固定为 1
    # - prefill 时通常等于 prompt tokens，但启用 chunked prefill 会更小
    token_chunk_size: int


@dataclass
class SchedulerOutputs:
    """一个 scheduler 本轮调度的最终决策输出"""

    # 本轮被调度执行的 seq_groups（按顺序）
    scheduled_seq_groups: GenericSequence[ScheduledSequenceGroup]
    # 本轮调度的 prefill 组数量
    num_prefill_groups: int
    # 本轮 batch token 总数（不含 cached 或含 cached，取决于调用处）
    num_batched_tokens: int
    # 需要 swap in 的 blocks: CPU -> GPU
    blocks_to_swap_in: List[Tuple[int, int]]
    # 需要 swap out 的 blocks: GPU -> CPU
    blocks_to_swap_out: List[Tuple[int, int]]
    # 需要 copy 的 blocks（通常 COW / fork 产生）
    blocks_to_copy: List[Tuple[int, int]]
    # 被忽略的 seq_groups（例如 prompt 太长无法分配）
    ignored_seq_groups: List[SequenceGroup]
    # speculative decoding lookahead slots 数量
    num_lookahead_slots: int
    # running queue 当前大小
    running_queue_size: int
    # 本轮抢占数量
    preempted: int

    def __post_init__(self):
        # swap in 和 swap out 不允许同时发生（同一轮不能双向换）
        assert not (self.blocks_to_swap_in and self.blocks_to_swap_out)

        # 当前 batch 里使用了多少个不同 LoRA
        self.num_loras: int = len(self.lora_requests)
        if self.num_loras > 0:
            # 按 LoRA id + request_id 做排序，保证某些执行顺序假设
            self._sort_by_lora_ids()

        # 当前 batch 里使用了多少个 prompt adapters
        self.num_prompt_adapters: int = len(self.prompt_adapter_requests)

    def is_empty(self) -> bool:
        # 注意 ignored_seq_groups 不算“空”
        return (not self.scheduled_seq_groups and not self.blocks_to_swap_in
                and not self.blocks_to_swap_out and not self.blocks_to_copy)

    def _sort_by_lora_ids(self):
        # num_prefill_groups 不能超过 scheduled_seq_groups 总长度
        assert 0 <= self.num_prefill_groups <= len(self.scheduled_seq_groups)

        def key_fn(group: ScheduledSequenceGroup):
            # 排序 key：优先按 lora_int_id，再按 request_id
            key = (group.seq_group.lora_int_id, group.seq_group.request_id)
            if 0 < self.num_prefill_groups < len(self.scheduled_seq_groups):
                # chunked prefill 要求：prefill 必须排在 decode 前面
                return (not group.seq_group.is_prefill(), *key)
            return key

        # 按 key_fn 排序
        self.scheduled_seq_groups = sorted(self.scheduled_seq_groups,
                                           key=key_fn)

    @property
    def lora_requests(self) -> Set[LoRARequest]:
        # 从 scheduled seq_groups 中提取出所有非空 lora_request
        return {
            g.seq_group.lora_request
            for g in self.scheduled_seq_groups
            if g.seq_group.lora_request is not None
        }

    @property
    def prompt_adapter_requests(self) -> Set[PromptAdapterRequest]:
        # 从 scheduled seq_groups 中提取出所有非空 prompt_adapter_request
        return {
            g.seq_group.prompt_adapter_request
            for g in self.scheduled_seq_groups
            if g.seq_group.prompt_adapter_request is not None
        }


@dataclass
class SchedulerRunningOutputs:
    """从 running queue 调度出的结果（可能包含 decode 或 chunked prefill）

    若内存不足，会触发：
    - preempt（recompute 模式）
    - swap out（swap 模式）
    """

    # running queue 中被选中执行的 decode seq_groups
    decode_seq_groups: List[ScheduledSequenceGroup]
    # running queue 中被选中执行的 prefill seq_groups（chunked prefill）
    prefill_seq_groups: List[ScheduledSequenceGroup]
    # 本轮被抢占的 seq_groups（通常进入 waiting，等待 recompute）
    preempted: List[SequenceGroup]
    # 本轮被 swap out 的 seq_groups（进入 swapped 队列）
    swapped_out: List[SequenceGroup]
    # swap out 的 block 映射表 (GPU -> CPU)
    blocks_to_swap_out: List[Tuple[int, int]]
    # copy 的 blocks（COW / fork 使用）
    blocks_to_copy: List[Tuple[int, int]]
    # lookahead slots 数（spec decode 用）
    num_lookahead_slots: int

    # 快速访问用的辅助列表（只存 seq_group）
    decode_seq_groups_list: List[SequenceGroup]
    prefill_seq_groups_list: List[SequenceGroup]

    @classmethod
    def create_empty(cls) -> "SchedulerRunningOutputs":
        # 创建一个空结果
        return SchedulerRunningOutputs(
            decode_seq_groups=[],
            prefill_seq_groups=[],
            preempted=[],
            swapped_out=[],
            blocks_to_swap_out=[],
            blocks_to_copy=[],
            num_lookahead_slots=0,
            decode_seq_groups_list=[],
            prefill_seq_groups_list=[],
        )


@dataclass
class SchedulerSwappedInOutputs:
    """从 swapped queue 调度出的结果（swap in 回 GPU 后继续运行）"""

    # swap in 后要执行 decode 的 seq_groups
    decode_seq_groups: List[ScheduledSequenceGroup]
    # swap in 后要执行 prefill 的 seq_groups（chunked prefill）
    prefill_seq_groups: List[ScheduledSequenceGroup]
    # swap in blocks 映射 (CPU -> GPU)
    blocks_to_swap_in: List[Tuple[int, int]]
    # copy blocks
    blocks_to_copy: List[Tuple[int, int]]
    # lookahead slots 数
    num_lookahead_slots: int
    # 永远无法调度的 seq_groups（KV cache 容量不足等）
    infeasible_seq_groups: List[SequenceGroup]

    @classmethod
    def create_empty(cls) -> "SchedulerSwappedInOutputs":
        return SchedulerSwappedInOutputs(
            decode_seq_groups=[],
            prefill_seq_groups=[],
            blocks_to_swap_in=[],
            blocks_to_copy=[],
            num_lookahead_slots=0,
            infeasible_seq_groups=[],
        )


@dataclass
class SchedulerPrefillOutputs:
    """从 waiting queue 调度的结果（新的 prompt prefill 或 recompute 的请求）"""

    # 本轮被调度的 prefill seq_groups
    seq_groups: List[ScheduledSequenceGroup]
    # 被忽略（例如 prompt 超长）的 seq_groups
    ignored_seq_groups: List[SequenceGroup]
    # lookahead slots 数
    num_lookahead_slots: int

    @classmethod
    def create_empty(cls) -> "SchedulerPrefillOutputs":
        return SchedulerPrefillOutputs(
            seq_groups=[],
            ignored_seq_groups=[],
            num_lookahead_slots=0,
        )


# ---- 下面几个 builder 用于 PyObjectCache 复用对象，减少频繁创建带来的开销 ----

def seq_group_metadata_builder():
    # 构建一个空的 SequenceGroupMetadata 作为缓存模板
    return SequenceGroupMetadata(request_id="",
                                 is_prompt=False,
                                 seq_data={},
                                 sampling_params=None,
                                 block_tables={})


def scheduler_running_outputs_builder():
    # 构建一个空的 SchedulerRunningOutputs 作为缓存模板
    return SchedulerRunningOutputs(decode_seq_groups=[],
                                   prefill_seq_groups=[],
                                   preempted=[],
                                   swapped_out=[],
                                   blocks_to_swap_out=[],
                                   blocks_to_copy=[],
                                   num_lookahead_slots=0,
                                   prefill_seq_groups_list=[],
                                   decode_seq_groups_list=[])


def scheduled_seq_group_builder():
    # 构建一个空的 ScheduledSequenceGroup（用 __new__ 绕开 init）
    return ScheduledSequenceGroup(SequenceGroup.__new__(SequenceGroup),
                                  token_chunk_size=0)
    # return ScheduledSequenceGroup(seq_group=None, token_chunk_size=0)


@dataclass
class PartialPrefillMetadata:
    """chunked prefill 下的一些统计信息（本轮调度用）

    允许同时运行多个 partial prefill，减少大 prompt 阻塞导致的 TTFT 变差
    同时限制长 prompt 的数量，避免 decode 饥饿或队列被大请求霸占
    """

    # 本轮最少应该允许调度的 prefill 数
    schedulable_prefills: int

    # 当前正在运行的长 prefill 请求数
    long_prefills: int

    scheduler_config: SchedulerConfig  # scheduler 配置

    def can_schedule(self, seq_group: SequenceGroup) -> bool:
        """判断某个 seq_group 是否允许被并发调度（针对长 prompt 限制）"""
        return not (seq_group.first_seq.get_num_new_tokens()
                    > self.scheduler_config.long_prefill_token_threshold
                    and self.long_prefills
                    >= self.scheduler_config.max_long_partial_prefills
                    and self.scheduler_config.max_num_partial_prefills > 1)

    def maybe_increment_partial_prefills(self,
                                         seq_group: SequenceGroup) -> None:
        # 当一个新的 prefill 被调度时，如果它是长 prompt，则 long_prefills +1
        if (seq_group.first_seq.get_num_new_tokens()
                > self.scheduler_config.long_prefill_token_threshold):
            self.long_prefills += 1

    @classmethod
    def from_queues(
        cls,
        running: Deque[SequenceGroup],
        waiting: Deque[SequenceGroup],
        scheduler_config: SchedulerConfig,
    ) -> "PartialPrefillMetadata":
        """根据当前 running/waiting 队列统计出 partial prefill 相关信息"""

        prefills = 0          # 预计本轮可能要调度的 prefill 数
        long_prefills = 0     # running 中正在跑的长 prefill 数
        waiting_long_prefills = 0  # waiting 中潜在可调度的长 prefill 数

        # 统计 running 队列中处于 PREFILL 阶段的请求
        for sg in running:
            if sg.first_seq.data.stage == SequenceStage.PREFILL:
                prefills += 1
                if (sg.first_seq.get_num_new_tokens()
                        > scheduler_config.long_prefill_token_threshold):
                    long_prefills += 1

        # 从 waiting 队列中预估还能塞多少 prefill（最多 max_num_partial_prefills）
        for sg in waiting:
            # 如果已经达到最大并发 prefill 数，就不继续看了
            if prefills >= scheduler_config.max_num_partial_prefills:
                break

            # 如果是长请求，需要额外受 max_long_partial_prefills 限制
            if (sg.first_seq.get_num_new_tokens()
                    > scheduler_config.long_prefill_token_threshold):
                if (long_prefills + waiting_long_prefills
                        >= scheduler_config.max_long_partial_prefills):
                    continue
                waiting_long_prefills += 1

            prefills += 1

        # 返回 metadata（注意 waiting_long_prefills 不计入 long_prefills）
        return PartialPrefillMetadata(
            schedulable_prefills=min(
                prefills, scheduler_config.max_num_partial_prefills),
            long_prefills=long_prefills,
            scheduler_config=scheduler_config,
        )


class Scheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
        pipeline_parallel_size: int = 1,
        output_proc_callback: Optional[Callable] = None,
    ) -> None:
        self.scheduler_config = scheduler_config  # 调度器配置
        self.cache_config = cache_config          # KV cache 配置

        # LoRA 调度策略说明：目前很简单，不保证公平，有可能导致某些 LoRA 饥饿
        self.lora_config = lora_config

        # 决定 BlockSpaceManager 的实现版本
        version = "selfattn"
        if (self.scheduler_config.runner_type == "pooling"
                or self.cache_config.is_attention_free):
            # pooling 或 attention-free 模型用 placeholder manager
            version = "placeholder"

        # 根据 version 选择 BlockSpaceManager 实现类
        BlockSpaceManagerImpl = BlockSpaceManager.get_block_space_manager_class(
            version)

        # pipeline parallel 会把总 block 数均分到各 pipeline stage
        num_gpu_blocks = cache_config.num_gpu_blocks
        if num_gpu_blocks:
            num_gpu_blocks //= pipeline_parallel_size

        num_cpu_blocks = cache_config.num_cpu_blocks
        if num_cpu_blocks:
            num_cpu_blocks //= pipeline_parallel_size

        # 创建 block 管理器：负责 KV cache 分配/回收/swap/copy
        self.block_manager = BlockSpaceManagerImpl(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=self.cache_config.enable_prefix_caching,
        )

        # WAITING 队列：新请求 prefill 或 recompute 回来的请求
        self.waiting: Deque[SequenceGroup] = deque()
        # RUNNING 队列：主要放 decode 请求（chunked prefill 时也可能包含 prefill）
        self.running: Deque[SequenceGroup] = deque()
        # SWAPPED 队列：被 swap out 到 CPU 的 decode 请求
        self.swapped: Deque[SequenceGroup] = deque()

        # 上一轮 step 结束后完成的 request_id 列表（用于释放模型状态）
        self._finished_requests_ids: List[str] = list()

        # 上一次 schedule 的时间戳
        self.prev_time = 0.0
        # 上一次是否调度了 prompt（prefill）
        self.prev_prompt = False
        # 上一次 prompt step 的调度延迟
        self.last_prompt_latency = 0.0

        # 用户指定的抢占模式（None / "swap" / "recompute"）
        self.user_specified_preemption_mode = scheduler_config.preemption_mode

        # 测试用：人工抢占开关/计数
        self.enable_artificial_preemption = ENABLE_ARTIFICIAL_PREEMPT
        self.artificial_preempt_cnt = (ARTIFICIAL_PREEMPTION_MAX_CNT
                                       if self.enable_artificial_preemption
                                       else 0)
        self.num_cumulative_preemption: int = 0  # 累计抢占次数（日志用）

        # PyObjectCache：缓存对象减少频繁创建开销
        self._seq_group_metadata_cache: List[PyObjectCache] = []
        self._scheduler_running_outputs_cache: List[PyObjectCache] = []
        self._scheduled_seq_group_cache: List[PyObjectCache] = []

        # async output processing：输出处理比推理慢一拍时，需要 cache 轮转
        self.output_proc_callback = output_proc_callback
        self.use_async_output_proc = self.output_proc_callback is not None
        self.num_cache_iters = 2 if self.use_async_output_proc else 1

        self.cache_id = 0
        for i in range(self.num_cache_iters):
            self._seq_group_metadata_cache.append(
                PyObjectCache(seq_group_metadata_builder))
            self._scheduler_running_outputs_cache.append(
                PyObjectCache(scheduler_running_outputs_builder))
            self._scheduled_seq_group_cache.append(
                PyObjectCache(scheduled_seq_group_builder))

        # async postprocessor 下：达到 max_model_len 的请求会放这里等待释放
        self._async_stopped: List[SequenceGroup] = []

        # chunked prefill：根据并发 prefill 数动态分配 token 预算（查表比除法快）
        self.partial_prefill_budget_lookup_list = [0] * (
            self.scheduler_config.max_num_partial_prefills + 1)
        self.partial_prefill_budget_lookup_list[0] = (
            scheduler_config.max_num_batched_tokens)
        for i in range(1, self.scheduler_config.max_num_partial_prefills + 1):
            self.partial_prefill_budget_lookup_list[i] = (
                scheduler_config.max_num_batched_tokens // i)

    @property
    def next_cache_id(self):
        # cache buffer 轮换（0->1->0...）
        return (self.cache_id + 1) % self.num_cache_iters

    @property
    def lora_enabled(self) -> bool:
        # 是否启用 LoRA
        return bool(self.lora_config)

    @property
    def num_decoding_tokens_per_seq(self) -> int:
        """decode 阶段每个 seq 每步生成的 token 数（默认 1）"""
        return 1

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # 新 seq_group 默认进入 waiting 队列
        self.waiting.append(seq_group)

    def _add_seq_group_to_running(self, seq_group: SequenceGroup) -> None:
        # 仅测试用：强行加入 running 队列
        self.running.append(seq_group)

    def _add_seq_group_to_swapped(self, seq_group: SequenceGroup) -> None:
        # 仅测试用：强行加入 swapped 队列
        self.swapped.append(seq_group)

    def abort_seq_group(
        self,
        request_id: Union[str, Iterable[str]],
        seq_id_to_seq_group: Optional[Dict[str, SequenceGroupBase]] = None,
    ) -> None:
        """中止指定 request_id 的 seq_group

        会遍历 waiting / running / swapped 三个队列：
        - 找到对应 group 后从队列移除
        - 把未完成的 seq 标记 FINISHED_ABORTED 并释放其 KV blocks
        - 同时清理 cross-attention blocks（encoder-decoder 模型）
        """

        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)

        # n>1 并行采样时：子 seq_group request_id 可能带后缀，需要映射回 group_id
        seq_id_to_seq_group = seq_id_to_seq_group or {}

        for state_queue in [self.waiting, self.running, self.swapped]:
            aborted_groups: List[SequenceGroup] = []
            for seq_group in state_queue:
                # 并行采样时 seq_group.request_id 形如 foo_parallel_sample_0
                if seq_group.request_id in seq_id_to_seq_group:
                    real_request_id = seq_id_to_seq_group[
                        seq_group.request_id].group_id
                else:
                    real_request_id = seq_group.request_id

                # 如果 real_request_id 命中需要 abort 的集合，则加入待移除列表
                if real_request_id in request_ids:
                    aborted_groups.append(seq_group)

            for aborted_group in aborted_groups:
                # 从队列中移除该 seq_group
                state_queue.remove(aborted_group)

                # 把 aborted 请求标记为已结束（用于后续清理模型状态，例如 Mamba cache）
                self._finished_requests_ids.append(aborted_group.request_id)

                # 遍历 seq_group 内所有 seq，释放未完成的 seq
                for seq in aborted_group.get_seqs():
                    if seq.is_finished():
                        continue
                    seq.status = SequenceStatus.FINISHED_ABORTED
                    self.free_seq(seq)

                # 清理映射
                if aborted_group.request_id in seq_id_to_seq_group:
                    del seq_id_to_seq_group[aborted_group.request_id]

                # encoder-decoder 模型需要释放 cross-attention blocks
                self._free_seq_group_cross_attn_blocks(aborted_group)

    def _free_seq_group_cross_attn_blocks(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        """
        释放 encoder-decoder 模型的 cross-attention block table
        decoder-only 模型无影响
        """
        if seq_group.is_encoder_decoder():
            self.block_manager.free_cross(seq_group)

    def has_unfinished_seqs(self) -> bool:
        # 判断系统是否还有未完成的请求
        return (len(self.waiting) != 0 or len(self.running) != 0
                or len(self.swapped) != 0)

    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        # 获取 prefix cache 命中率
        return self.block_manager.get_prefix_cache_hit_rate(device)

    def reset_prefix_cache(self, device: Optional[Device] = None) -> bool:
        # 重置 prefix cache
        return self.block_manager.reset_prefix_cache(device)

    def get_num_unfinished_seq_groups(self) -> int:
        # unfinished seq_group 总数 = waiting + running + swapped
        return len(self.waiting) + len(self.running) + len(self.swapped)

    def get_and_reset_finished_requests_ids(self) -> List[str]:
        """取出并清空已完成 request_id 列表"""
        finished_requests_ids = self._finished_requests_ids
        self._finished_requests_ids = list()
        return finished_requests_ids

    def _schedule_running(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
        partial_prefill_metadata: Optional[PartialPrefillMetadata] = None,
    ) -> SchedulerRunningOutputs:
        """调度 running 队列的 seq_groups（decode 或 chunked prefill）

        - 如果 KV cache 不够，会触发抢占 preempt 或 swap out
        - budget 和 curr_loras 会被就地更新（in-place）

        返回 SchedulerRunningOutputs
        """
        # 从缓存池取一个 SchedulerRunningOutputs 对象，避免频繁创建
        ret: SchedulerRunningOutputs = self._scheduler_running_outputs_cache[
            self.cache_id].get_object()

        # 清空上一次缓存对象里的各种列表（复用对象）
        ret.blocks_to_swap_out.clear()
        ret.blocks_to_copy.clear()
        ret.decode_seq_groups.clear()
        ret.prefill_seq_groups.clear()
        ret.preempted.clear()
        ret.swapped_out.clear()

        # 计算 lookahead slots（spec decode 用）
        ret.num_lookahead_slots = self._get_num_lookahead_slots(
            is_prefill=False, enable_chunking=enable_chunking)

        ret.decode_seq_groups_list.clear()
        ret.prefill_seq_groups_list.clear()

        # swap out / copy 列表引用（后面填充）
        blocks_to_swap_out: List[Tuple[int, int]] = ret.blocks_to_swap_out
        blocks_to_copy: List[Tuple[int, int]] = ret.blocks_to_copy

        # 输出的 scheduled lists
        decode_seq_groups: List[ScheduledSequenceGroup] = ret.decode_seq_groups
        prefill_seq_groups: List[
            ScheduledSequenceGroup] = ret.prefill_seq_groups
        preempted: List[SequenceGroup] = ret.preempted
        swapped_out: List[SequenceGroup] = ret.swapped_out

        running_queue = self.running
        assert len(self._async_stopped) == 0  # schedule 前 async_stopped 应该已处理完

        # 从 running_queue 头部开始尝试调度
        while running_queue:
            seq_group = running_queue[0]

            # 获取该 seq_group 本轮要新增计算的 uncached token 数 + cached token 数
            # （running 队列里 cached 信息通常无意义，因为 cached 只用于首次 prefill）
            num_uncached_new_tokens, _ = \
                self._get_num_new_uncached_and_cached_tokens(
                seq_group,
                SequenceStatus.RUNNING,
                enable_chunking,
                budget,
                partial_prefill_metadata,
            )

            num_running_tokens = num_uncached_new_tokens
            if num_running_tokens == 0:
                # token budget 不够了 -> 停止调度
                break

            # 从队列中弹出头部 seq_group（准备调度执行）
            running_queue.popleft()

            # async output proc 模式下：如果 seq 长度超过 max_model_len，避免额外 decode 导致 OOM
            if (self.use_async_output_proc and seq_group.seqs[0].get_len()
                    > self.scheduler_config.max_model_len):
                self._async_stopped.append(seq_group)
                continue

            # 当 KV cache slot 不够时，会触发 preempt / swap out
            while not self._can_append_slots(seq_group, enable_chunking):
                # 预算回退（因为无法运行这个 seq_group）
                budget.subtract_num_batched_tokens(seq_group.request_id,
                                                   num_running_tokens)
                num_running_seqs = seq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(seq_group.request_id,
                                         num_running_seqs)

                # LoRA 计数回退（当前 batch 移除该 LoRA）
                if (curr_loras is not None and seq_group.lora_int_id > 0
                        and seq_group.lora_int_id in curr_loras):
                    curr_loras.remove(seq_group.lora_int_id)

                # 确定 victim（要被抢占的 seq_group）
                cont_loop = True
                if running_queue:
                    # 抢占队列里最低优先级（末尾）seq_group
                    victim_seq_group = running_queue.pop()
                else:
                    # 没有其他可抢占的，只能抢占当前 seq_group
                    victim_seq_group = seq_group
                    cont_loop = False

                # async output proc：抢占前需确保没有挂起的后处理
                do_preempt = True
                if self.use_async_output_proc:
                    assert self.output_proc_callback is not None
                    self.output_proc_callback(
                        request_id=victim_seq_group.request_id)

                    # 如果 victim 因后处理变为 finished，则直接释放
                    if victim_seq_group.is_finished():
                        self._free_finished_seq_group(victim_seq_group)
                        do_preempt = False

                # 执行抢占（recompute 或 swap）
                if do_preempt:
                    preempted_mode = self._preempt(victim_seq_group,
                                                   blocks_to_swap_out)
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        preempted.append(victim_seq_group)
                    else:
                        swapped_out.append(victim_seq_group)

                # 如果抢占的是当前 seq_group，则退出 while（没有更多可抢占）
                if not cont_loop:
                    break
            else:
                # KV cache slot 足够：为 seq_group append slots（可能触发 copy-on-write）
                self._append_slots(seq_group, blocks_to_copy, enable_chunking)
                is_prefill = seq_group.is_prefill()

                # 从缓存池取一个 ScheduledSequenceGroup 对象
                scheduled_seq_group: ScheduledSequenceGroup = (
                    self._scheduled_seq_group_cache[
                        self.cache_id].get_object())
                scheduled_seq_group.seq_group = seq_group

                if is_prefill:
                    # prefill：chunk size = num_running_tokens（可能小于 prompt 长度）
                    scheduled_seq_group.token_chunk_size = num_running_tokens
                    prefill_seq_groups.append(scheduled_seq_group)
                    ret.prefill_seq_groups_list.append(seq_group)
                else:
                    # decode：固定每次 1 token
                    scheduled_seq_group.token_chunk_size = 1
                    decode_seq_groups.append(scheduled_seq_group)
                    ret.decode_seq_groups_list.append(seq_group)

                # 更新 budget token 统计
                budget.add_num_batched_tokens(seq_group.request_id,
                                              num_running_tokens)

                # enable_chunking 时要更新 seq 数（默认路径之前可能已算过）
                if enable_chunking:
                    num_running_seqs = seq_group.get_max_num_running_seqs()
                    budget.add_num_seqs(seq_group.request_id, num_running_seqs)

                # 更新当前 batch LoRA 集合
                if curr_loras is not None and seq_group.lora_int_id > 0:
                    curr_loras.add(seq_group.lora_int_id)

        # 重置下一轮 cache 对象池
        self._scheduler_running_outputs_cache[self.next_cache_id].reset()
        self._scheduled_seq_group_cache[self.next_cache_id].reset()

        return ret

    def _chunk_new_tokens_to_schedule(
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        budget: SchedulingBudget,
        prompt_limit: int,
        num_new_tokens: int,
        partial_prefill_budget_lookup_list: List[int],
        partial_prefill_metadata: Optional[PartialPrefillMetadata] = None,
    ) -> int:
        """
        chunked prefill 模式下，根据 budget 把本轮要计算的 token 数切块

        返回：本轮实际要计算的 token 数
        """
        remaining_token_budget = budget.remaining_token_budget()

        if scheduler_config.is_multi_step:
            # multi-step + chunked prefill：实际上不支持对 prompt 做 chunk
            # 所以如果 prompt 太长，直接返回 0 表示本轮不调度
            if num_new_tokens > prompt_limit:
                # 返回原值让调用者判断是否要 ignore
                return num_new_tokens

            # 如果超出剩余预算，本轮调度为 0
            return 0 if num_new_tokens > \
                remaining_token_budget else num_new_tokens

        # partial prefill 下，每个 prefill slot 平均分配预算
        prefill_slot_budget = (
            remaining_token_budget if partial_prefill_metadata is None else
            partial_prefill_budget_lookup_list[
                partial_prefill_metadata.schedulable_prefills])

        if cache_config.enable_prefix_caching:
            # 如果启用 prefix caching：
            # 部分 prefill 时要保证分配 token 数是 block_size 的整数倍（避免 partial block 匹配失败）
            block_size = cache_config.block_size

            # 预算取 min(total_remaining, slot_budget)，再向下取整到 block_size 的倍数
            remaining_token_budget = (
                min(remaining_token_budget, prefill_slot_budget) //
                block_size) * block_size

            # 注意：如果 num_new_tokens < budget，说明即将完成 prefill，不必填满整块

        # 最终 chunk 大小不能超过：
        # - 本 seq 剩余 token 数 num_new_tokens
        # - 总预算 remaining_token_budget
        # - slot 预算 prefill_slot_budget
        num_new_tokens = min(num_new_tokens, remaining_token_budget,
                             prefill_slot_budget)

        return num_new_tokens

```



