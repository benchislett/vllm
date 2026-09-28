# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static FP8 output contract for the wide GDN verification kernel."""

import torch

from vllm import _custom_ops as ops
from vllm.forward_context import get_forward_context
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.fusion.quant_activation import get_input_quant_key
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata


def gdn_output_quant_scale(layer) -> torch.Tensor | None:
    """Return the consumer's static scale, or retain the BF16 output path.

    Scale values follow the loaded projection's positive, finite scale contract.
    Only metadata is inspected here, so selection is safe during tracing.
    """
    projection = layer.out_proj
    # Wrappers such as LoRA need an unquantized activation.
    if (
        type(projection) is not RowParallelLinear
        or not projection.input_is_parallel
        or get_input_quant_key(projection) != kFp8StaticTensorSym
    ):
        return None
    if (
        not layer._is_sm107
        or layer.gdn_decode_kernel != "cuda"
        or layer.num_k_heads // layer.tp_size != 2
        or layer.num_v_heads // layer.tp_size != 16
        or layer.head_k_dim != 128
        or layer.head_v_dim != 128
        or layer.norm.weight.dtype != torch.bfloat16
        or layer.dt_bias.dtype != torch.bfloat16
        or layer.norm.activation != "silu"
        or layer.layer_norm_epsilon != 1e-6
        or not hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp_fp8")
    ):
        return None
    method = getattr(projection, "scheme", projection.quant_method)
    kernel = getattr(method, "fp8_linear", getattr(method, "kernel", None))
    if (
        not isinstance(kernel, FP8ScaledMMLinearKernel)
        or kernel.input_quant_key() != kFp8StaticTensorSym
        or kernel.quant_fp8.num_token_padding is not None
    ):
        return None
    weight, _, scale, _ = kernel._get_layer_params(projection)
    if (
        scale is None
        or scale.dtype != torch.float32
        or scale.numel() != 1
        or not scale.is_contiguous()
        or not scale.is_cuda
        or scale.device != layer.norm.weight.device
        or weight.dtype != torch.float8_e4m3fn
    ):
        return None
    return scale


def forward_gdn_with_output_quant(layer, packed, ba, out, output_scale) -> None:
    """Select fusion with capture-time metadata; quantize the BF16 fallback."""
    all_metadata = get_forward_context().attn_metadata
    metadata = (
        all_metadata.get(layer.prefix) if isinstance(all_metadata, dict) else None
    )
    indices = getattr(metadata, "spec_state_indices_tensor", None)
    if (
        isinstance(metadata, GDNAttentionMetadata)
        and metadata.num_prefills == 0
        and layer._can_use_fused_gdn_mtp_decode(metadata)
        and metadata.num_spec_decodes <= 128
        and indices is not None
        and indices.size(1) in (12, 16)
        and layer.kv_cache[1].dtype == torch.bfloat16
    ):
        qkv, gate = packed.split((2560, 2048), dim=-1)
        b, a = layer.split_ba(ba)
        layer._forward_core_decode_spec_fused_norm(
            mixed_qkv=qkv,
            b=b,
            a=a,
            output_gate=gate.reshape(-1, 16, 128),
            core_attn_out=out,
            attn_metadata=metadata,
            output_scale=output_scale,
        )
        return
    # Prefill, AR, mixed batches and larger MTP batches keep the existing core.
    # Zero initialization also covers dummy/profile runs and unused graph slots.
    bf16 = torch.zeros(out.shape, dtype=torch.bfloat16, device=out.device)
    layer._forward_core_fused_norm_packed(packed, ba, bf16)
    ops.scaled_fp8_quant(bf16.flatten(-2), output_scale, output=out.flatten(-2))
