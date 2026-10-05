# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


@torch.library.custom_op(
    "vllm::gdn_custom_fused_input_gemm", mutates_args=(), device_types="cuda"
)
def gdn_custom_fused_input_gemm(
    x: torch.Tensor,
    qkvz_weight: torch.Tensor,
    ba_weight: torch.Tensor,
    input_scale: torch.Tensor,
    qkvz_scale: torch.Tensor,
    ba_scale: torch.Tensor,
    pdl_mode: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tuned TP8 projections; weight loads precede the activation PDL wait."""
    qkvz = x.new_empty((x.shape[0], 4608), dtype=torch.bfloat16)
    ba = x.new_empty((x.shape[0], 32), dtype=torch.bfloat16)
    torch.ops._qwen_gdn_custom.gemms_out(
        x,
        qkvz_weight,
        ba_weight,
        input_scale,
        qkvz_scale,
        ba_scale,
        qkvz,
        ba,
        pdl_mode,
    )
    return qkvz, ba


@gdn_custom_fused_input_gemm.register_fake
def _gdn_custom_fused_input_gemm_fake(
    x, qkvz_weight, ba_weight, input_scale, qkvz_scale, ba_scale, pdl_mode=2
):
    return (
        x.new_empty((x.shape[0], 4608), dtype=torch.bfloat16),
        x.new_empty((x.shape[0], 32), dtype=torch.bfloat16),
    )


@torch.library.custom_op(
    "vllm::gdn_dispatch_input_gemms", mutates_args=(), device_types="cuda"
)
def gdn_dispatch_input_gemms(
    x: torch.Tensor,
    qkvz_weight: torch.Tensor,
    ba_weight: torch.Tensor,
    input_scale: torch.Tensor,
    qkvz_scale: torch.Tensor,
    ba_scale: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select by concrete M during execution/capture, outside Dynamo tracing."""
    # A Python shape branch in the model is frozen by its first large-M trace.
    # vLLM reuses that graph without evaluating Dynamo's shape guards.
    if 0 < x.shape[0] <= 16:
        return gdn_custom_fused_input_gemm(
            x, qkvz_weight, ba_weight, input_scale, qkvz_scale, ba_scale
        )

    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
    from vllm.model_executor.layers.mamba.gdn.input_projection import gdn_input_gemms
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8StaticTensorSym,
    )

    # Use the same layer lookup as the existing GDN attention custom op. Keep
    # normal Linear backend selection when concurrency was not requested.
    layer = get_forward_context().no_compile_layers[layer_name]
    if layer._concurrent_input_gemm:
        return gdn_input_gemms(
            x, qkvz_weight, ba_weight, input_scale, qkvz_scale, ba_scale
        )
    proj_input = QuantizedActivation(
        x, input_scale, torch.bfloat16, x.shape, kFp8StaticTensorSym
    )
    qkvz, _ = layer.in_proj_qkvz(proj_input)
    ba, _ = layer.in_proj_ba(proj_input)
    return qkvz, ba


@gdn_dispatch_input_gemms.register_fake
def _gdn_dispatch_input_gemms_fake(
    x, qkvz_weight, ba_weight, input_scale, qkvz_scale, ba_scale, layer_name
):
    return _gdn_custom_fused_input_gemm_fake(
        x, qkvz_weight, ba_weight, input_scale, qkvz_scale, ba_scale
    )
