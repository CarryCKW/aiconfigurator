# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declarative registry mapping ops to collector modules for vLLM 0.14.0.

This is a dedicated registry for SM80 GPUs (A800/A100) that only support vllm 0.14.0.
"""

from collector.registry_types import OpEntry, PerfFile

REGISTRY: list[OpEntry] = [
    OpEntry(
        op="gemm",
        module="collector.vllm_014.collect_gemm",
        get_func="get_gemm_test_cases",
        run_func="run_gemm",
        perf_filename=PerfFile.GEMM,
    ),
    OpEntry(
        op="attention_context",
        module="collector.vllm_014.collect_attn",
        get_func="get_context_attention_test_cases",
        run_func="run_attention_torch",
        perf_filename=PerfFile.CONTEXT_ATTENTION,
    ),
    OpEntry(
        op="attention_generation",
        module="collector.vllm_014.collect_attn",
        get_func="get_generation_attention_test_cases",
        run_func="run_attention_torch",
        perf_filename=PerfFile.GENERATION_ATTENTION,
    ),
    OpEntry(
        op="moe",
        module="collector.vllm_014.collect_moe",
        get_func="get_moe_test_cases",
        run_func="run_moe_torch",
        perf_filename=PerfFile.MOE,
    ),
]
