---
title: vLLM_async_llm_streaming
published: 2025-12-19
description: "vLLM_async_llm_streaming"
image: "./cover.jpeg"
tags: ["Blogging","vLLM","vLLM_async_llm_streaming"]
category: Guides
draft: false

---

# vLLM: async_llm_streaming

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Simple example demonstrating streaming offline inference with AsyncLLM (V1 engine).

This script shows the core functionality of vLLM's AsyncLLM engine for streaming
token-by-token output in offline inference scenarios. It demonstrates DELTA mode
streaming where you receive new tokens as they are generated.

Usage:
    python examples/offline_inference/async_llm_streaming.py
"""

import asyncio

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM


async def stream_response(engine: AsyncLLM, prompt: str, request_id: str) -> None:
    """
    Stream response from AsyncLLM and display tokens as they arrive.

    This function demonstrates the core streaming pattern:
    1. Create SamplingParams with DELTA output kind
    2. Call engine.generate() and iterate over the async generator
    3. Print new tokens as they arrive
    4. Handle the finished flag to know when generation is complete
    """
    print(f"\n🚀 Prompt: {prompt!r}")
    print("💬 Response: ", end="", flush=True)

    # Configure sampling parameters for streaming
    sampling_params = SamplingParams(
        max_tokens=100,
        temperature=0.8,
        top_p=0.95,
        seed=42,  # For reproducible results
        output_kind=RequestOutputKind.DELTA,  # Get only new tokens each iteration
    )

    try:
        # Stream tokens from AsyncLLM
        async for output in engine.generate(
            request_id=request_id, prompt=prompt, sampling_params=sampling_params
        ):
            # Process each completion in the output
            for completion in output.outputs:
                # In DELTA mode, we get only new tokens generated since last iteration
                new_text = completion.text
                if new_text:
                    print(new_text, end="", flush=True)

            # Check if generation is finished
            if output.finished:
                print("\n✅ Generation complete!")
                break

    except Exception as e:
        print(f"\n❌ Error during streaming: {e}")
        raise


# main 是一个异步主函数，负责：初始化 vLLM；控制 prompt 的执行顺序；在合适的时候 await / sleep；最终清理资源
async def main():
    print("🔧 Initializing AsyncLLM...")

    # Create AsyncLLM engine with simple configuration
    # 
    engine_args = AsyncEngineArgs(
        # model="meta-llama/Llama-3.2-1B-Instruct",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        enforce_eager=True,  # Faster startup for examples
    )
    engine = AsyncLLM.from_engine_args(engine_args)

    try:
        # Example prompts to demonstrate streaming
        prompts = [
            "The future of artificial intelligence is",
            "In a galaxy far, far away",
            "The key to happiness is",
        ]

        print(f"🎯 Running {len(prompts)} streaming examples...")

        # Process each prompt
        for i, prompt in enumerate(prompts, 1):
            print(f"\n{'=' * 60}")
            print(f"Example {i}/{len(prompts)}")
            print(f"{'=' * 60}")

            request_id = f"stream-example-{i}"
            await stream_response(engine, prompt, request_id)

            # Brief pause between examples
            if i < len(prompts):
                await asyncio.sleep(0.5)

        print("\n🎉 All streaming examples completed!")

    finally:
        # Always clean up the engine
        print("🔧 Shutting down engine...")
        engine.shutdown()


if __name__ == "__main__":
    try:
        # 创建一个事件循环（event loop）把 main() 这个协程丢进去
        asyncio.run(main()) 
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")

```

1. **初始化引擎**：用 `AsyncEngineArgs` 创建配置，并通过 `AsyncLLM.from_engine_args()` 启动异步推理引擎。
2. **设置采样参数**：构造 `SamplingParams`，指定生成长度、随机性，并将 `output_kind` 设为 `DELTA` 以支持流式返回新 token。
3. **发起生成请求**：调用 `engine.generate()`，传入 `prompt`、`request_id` 和采样参数。
4. **流式消费输出**：用 `async for` 迭代生成器，实时读取并打印每次返回的新增文本，直到 `finished` 为真。
5. **清理资源**：所有请求完成后调用 `engine.shutdown()` 关闭引擎并释放资源。