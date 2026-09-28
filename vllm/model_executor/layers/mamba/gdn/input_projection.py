# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from functools import cache

import torch

from vllm.platforms import current_platform
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel

# Measured layouts; tactics are discovered for the installed library versions.
_EXACT_ROWS = {
    4608: (24, 32, 96, 128, 192, 256, 512, 1536),
    9216: (128, 192, 384, 512),
}


@cache
def _resources(device: torch.device, parent_stream: int):
    with torch.accelerator.device_index(device.index):
        stream = torch.cuda.Stream()
        events = (torch.cuda.Event(), torch.cuda.Event())
        workspaces = [
            torch.empty(40 * 1024 * 1024, dtype=torch.uint8, device=device)
            for _ in range(2)
        ]
    return stream, events, workspaces


@torch.library.custom_op("vllm::gdn_input_gemms", mutates_args=(), device_types="cuda")
def gdn_input_gemms(
    x: torch.Tensor,
    qkvz_weight: torch.Tensor,
    ba_weight: torch.Tensor,
    input_scale: torch.Tensor,
    qkvz_scale: torch.Tensor,
    ba_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concurrent scalar-FP8 projections with exact tuning for qualified rows."""
    from flashinfer import autotune

    # A cuBLAS tactic ordinal is relative to the algorithm list for this M.
    # A passive nested context inherits warmup mode and never starts tuning
    # during capture or serving. Cache files retain FlashInfer's version checks.
    exact = (
        current_platform.is_device_capability(107)
        and x.shape[1] == 8192
        and x.dtype == qkvz_weight.dtype == ba_weight.dtype == torch.float8_e4m3fn
        and qkvz_weight.stride() == ba_weight.stride() == (1, 8192)
        and ba_weight.shape[1] * 144 == qkvz_weight.shape[1]
        and x.shape[0] in _EXACT_ROWS.get(qkvz_weight.shape[1], ())
    )
    context = (
        autotune(False, tuning_buckets=(x.shape[0],), round_up=False)
        if exact
        else nullcontext()
    )
    with context:
        return _run_gdn_input_gemms(
            x, qkvz_weight, ba_weight, input_scale, qkvz_scale, ba_scale
        )


def _run_gdn_input_gemms(
    x: torch.Tensor,
    qkvz_weight: torch.Tensor,
    ba_weight: torch.Tensor,
    input_scale: torch.Tensor,
    qkvz_scale: torch.Tensor,
    ba_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concurrent scalar-FP8 projections with separate scratch and BF16 outputs."""
    from flashinfer import bmm_fp8

    parent = torch.cuda.current_stream(x.device)
    side, events, workspaces = _resources(x.device, parent.cuda_stream)
    qkvz = torch.empty(
        (x.shape[0], qkvz_weight.shape[1]), dtype=torch.bfloat16, device=x.device
    )
    ba = torch.empty(
        (x.shape[0], ba_weight.shape[1]), dtype=torch.bfloat16, device=x.device
    )
    maybe_execute_in_parallel(
        lambda: bmm_fp8(
            x.unsqueeze(0),
            qkvz_weight.unsqueeze(0),
            input_scale,
            qkvz_scale,
            torch.bfloat16,
            out=qkvz.unsqueeze(0),
            backend="cublas",
            workspace_buffer=workspaces[0],
        ),
        lambda: bmm_fp8(
            x.unsqueeze(0),
            ba_weight.unsqueeze(0),
            input_scale,
            ba_scale,
            torch.bfloat16,
            out=ba.unsqueeze(0),
            backend="cublas",
            workspace_buffer=workspaces[1],
        ),
        events[0],
        events[1],
        side,
    )
    return qkvz, ba


@gdn_input_gemms.register_fake
def _gdn_input_gemms_fake(x, qkvz_weight, ba_weight, input_scale, qkvz_scale, ba_scale):
    return (
        x.new_empty((x.shape[0], qkvz_weight.shape[1]), dtype=torch.bfloat16),
        x.new_empty((x.shape[0], ba_weight.shape[1]), dtype=torch.bfloat16),
    )


def autotune_gdn_input_projections(model: torch.nn.Module, max_tokens: int) -> None:
    """Warm default and exact projection shapes before graph capture."""
    from flashinfer import autotune
    from flashinfer.autotuner import (
        get_autotune_process_group,
        set_autotune_process_group,
    )

    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.utils.flashinfer import flashinfer_get_hybrid_num_tokens_buckets

    if not current_platform.is_device_capability(107):
        return
    # These independent GEMMs need no cross-rank tactic consensus. Pipeline
    # stages may contain different numbers of GDN layers or projection shapes.
    group = get_autotune_process_group()
    set_autotune_process_group(None)
    try:
        seen = set()
        for layer in model.modules():
            if not isinstance(layer, QwenGatedDeltaNetAttention):
                continue
            if not layer._concurrent_input_gemm:
                continue
            qkvz = layer.in_proj_qkvz
            ba = layer.in_proj_ba
            key = (qkvz.weight.device, tuple(qkvz.weight.shape))
            exact_rows = [
                rows
                for rows in _EXACT_ROWS.get(qkvz.weight.shape[1], ())
                if rows <= max_tokens
            ]
            if key in seen or qkvz.weight.shape[0] != 8192 or not exact_rows:
                continue
            seen.add(key)
            # Exact-M dispatch during the model dummy run can suppress its
            # ordinary buckets. Preserve those before adding exact records.
            x = torch.zeros(
                (max_tokens, 8192), dtype=qkvz.weight.dtype, device=qkvz.weight.device
            )
            with autotune(
                False,
                tuning_buckets=flashinfer_get_hybrid_num_tokens_buckets(max_tokens),
            ):
                _run_gdn_input_gemms(
                    x,
                    qkvz.weight,
                    ba.weight,
                    qkvz.input_scale,
                    qkvz.weight_scale,
                    ba.weight_scale,
                )
            for rows in exact_rows:
                x = torch.zeros(
                    (rows, 8192), dtype=qkvz.weight.dtype, device=qkvz.weight.device
                )
                gdn_input_gemms(
                    x,
                    qkvz.weight,
                    ba.weight,
                    qkvz.input_scale,
                    qkvz.weight_scale,
                    ba.weight_scale,
                )
    finally:
        set_autotune_process_group(group)
