# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep MRCR smoke-test limits after needle and context filtering."""

from dataclasses import replace

from evalscope.api.registry import BENCHMARK_REGISTRY, register_benchmark
from evalscope.benchmarks.openai_mrcr.openai_mrcr_adapter import OpenAIMRCRAdapter


@register_benchmark(replace(BENCHMARK_REGISTRY["openai_mrcr"], name="vllm_mrcr"))
class MRCRAdapter(OpenAIMRCRAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # EvalScope's normal loader limits source rows before record_to_sample;
        # the regrouping path applies limits after its needle/context filters.
        self.reformat_subset = True
