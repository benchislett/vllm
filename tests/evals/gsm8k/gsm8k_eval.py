#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility entrypoint for the packaged GSM8K evaluator."""

from vllm.evals.gsm8k import (
    INVALID as INVALID,
)
from vllm.evals.gsm8k import (
    _build_gsm8k_prompts as _build_gsm8k_prompts,
)
from vllm.evals.gsm8k import (
    _score_gsm8k as _score_gsm8k,
)
from vllm.evals.gsm8k import (
    call_vllm_api as call_vllm_api,
)
from vllm.evals.gsm8k import (
    call_vllm_chat_api as call_vllm_chat_api,
)
from vllm.evals.gsm8k import (
    download_and_cache_file as download_and_cache_file,
)
from vllm.evals.gsm8k import (
    evaluate_gsm8k as evaluate_gsm8k,
)
from vllm.evals.gsm8k import (
    evaluate_gsm8k_offline as evaluate_gsm8k_offline,
)
from vllm.evals.gsm8k import (
    get_answer_value as get_answer_value,
)
from vllm.evals.gsm8k import (
    load_gsm8k_data as load_gsm8k_data,
)
from vllm.evals.gsm8k import (
    main as main,
)
from vllm.evals.gsm8k import (
    read_jsonl as read_jsonl,
)

if __name__ == "__main__":
    main()
