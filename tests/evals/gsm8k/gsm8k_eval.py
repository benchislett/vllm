#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility entrypoint for the packaged GSM8K evaluator."""

from vllm.evals.gsm8k import (
    INVALID,
    _build_gsm8k_prompts,
    _score_gsm8k,
    call_vllm_api,
    call_vllm_chat_api,
    download_and_cache_file,
    evaluate_gsm8k,
    evaluate_gsm8k_offline,
    get_answer_value,
    load_gsm8k_data,
    main,
    read_jsonl,
)

__all__ = [
    "INVALID",
    "_build_gsm8k_prompts",
    "_score_gsm8k",
    "call_vllm_api",
    "call_vllm_chat_api",
    "download_and_cache_file",
    "evaluate_gsm8k",
    "evaluate_gsm8k_offline",
    "get_answer_value",
    "load_gsm8k_data",
    "main",
    "read_jsonl",
]

if __name__ == "__main__":
    main()
