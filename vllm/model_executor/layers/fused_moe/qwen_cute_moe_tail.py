# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A compiler-owned Qwen MoE producer with CuTe finalization and AllReduce."""

from contextlib import nullcontext

import torch

from vllm.distributed.device_communicators.cute_allreduce import (
    HIDDEN_SIZE,
    MAX_TOKENS,
    TOP_K,
    cute_allreduce_norm,
    get_workspace,
    output_dtype,
)
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
    MoERunner,
    _layer_name_type,
    _resolve_layer_name,
    _unpack,
    get_layer_from_name,
)
from vllm.utils.torch_utils import direct_register_custom_op


def can_defer(
    layer: MoERunner,
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    m: int,
) -> bool:
    cfg = layer.moe_config
    kernel = getattr(layer.routed_experts.quant_method, "moe_kernel", None)
    from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
        TrtLlmNvFp4ExpertsMonolithic,
    )

    return bool(
        0 < m <= MAX_TOKENS
        and x.is_contiguous()
        and x.dtype == weight.dtype == torch.bfloat16
        and x.ndim == 2
        and x.shape[1] == HIDDEN_SIZE
        and weight.shape == (HIDDEN_SIZE,)
        and weight.is_contiguous()
        and (
            scale is None
            or (
                scale.dtype == torch.float32
                and scale.numel() == 1
                and scale.is_contiguous()
                and scale.device == x.device
            )
        )
        and x.device == weight.device
        and cfg.experts_per_token == TOP_K
        and cfg.hidden_dim == cfg.hidden_dim_unpadded == HIDDEN_SIZE
        and cfg.tp_size * cfg.ep_size == 8
        and cfg.dp_size == cfg.pcp_size == 1
        and not cfg.is_sequence_parallel
        and not cfg.moe_parallel_config.use_all2all_kernels
        and not cfg.is_lora_enabled
        and not layer._fused_output_is_reduced
        and not layer.do_naive_dispatch_combine
        and layer.routed_scaling_factor == 1.0
        and layer.shared_experts is not None
        and layer.routed_input_transform is None
        and layer.routed_output_transform is None
        and kernel is not None
        and kernel.supports_deferred_moe_finalize()
        and isinstance(kernel.fused_experts, TrtLlmNvFp4ExpertsMonolithic)
    )


def qwen_cute_moe_tail(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
    residual: torch.Tensor | None,
    rms_gamma: torch.Tensor,
    scale: torch.Tensor | None,
    rms_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer = get_layer_from_name(_resolve_layer_name(layer_name))
    assert isinstance(layer, MoERunner)
    m = hidden_states.shape[0]
    cfg = layer.moe_config
    use_cute = hidden_dim_unpadded in (0, HIDDEN_SIZE) and can_defer(
        layer, hidden_states, rms_gamma, scale, m
    )
    context = (
        cfg.defer_moe_finalize_for_call(MAX_TOKENS, allow_replicated_ep=True)
        if use_cute
        else nullcontext()
    )
    with context:
        use_cute = use_cute and cfg.should_defer_moe_finalize(m)
        shared, routed = _unpack(
            layer._forward_impl(
                hidden_states, router_logits, shared_experts_input, input_ids
            )
        )
    if shared is None:
        raise RuntimeError(
            "Matched Qwen shared-expert producer returned no shared output"
        )
    if use_cute:
        if not isinstance(routed, UnfinalizedMoEOutput):
            raise RuntimeError("Qwen CuTe producer did not defer MoE finalization")
        dtype = output_dtype(scale)
        output = torch.empty_like(shared, dtype=dtype)
        residual_out = torch.empty_like(shared)
        workspace = get_workspace(rms_eps, dtype, residual is not None)
        from flashinfer.comm import allreduce_fusion
        from flashinfer.comm.trtllm_ar import AllReduceFusionPattern

        allreduce_fusion(
            input=routed.gemm2_permuted,
            workspace=workspace,
            pattern=AllReduceFusionPattern.kMoEFinalizeARResidualRMSNorm,
            expanded_idx_to_permuted_idx=routed.expanded_idx_to_permuted_idx,
            expert_scale_factor=routed.expert_weights,
            shared_expert_output=shared,
            residual_in=residual,
            residual_out=residual_out,
            rms_gamma=rms_gamma,
            rms_eps=rms_eps,
            weight_bias=1.0,
            norm_out=output if scale is None else None,
            quant_out=output if scale is not None else None,
            scale_factor=scale,
            launch_with_pdl=True,
        )
        return output, residual_out
    if isinstance(routed, UnfinalizedMoEOutput):
        raise RuntimeError("Fallback producer unexpectedly returned a deferred output")
    # Even when a MoE implementation cannot defer finalization, use CuTe for
    # its eligible AR/norm tail. Never retry a failed collective on a backend.
    return cute_allreduce_norm(shared + routed, residual, rms_gamma, scale, rms_eps)


def qwen_cute_moe_tail_fake(
    hidden_states,
    router_logits,
    shared_experts_input,
    input_ids,
    layer_name,
    hidden_dim_unpadded,
    residual,
    rms_gamma,
    scale,
    rms_eps,
):
    return (
        torch.empty_like(hidden_states, dtype=output_dtype(scale)),
        torch.empty_like(hidden_states),
    )


direct_register_custom_op(
    op_name="qwen_cute_moe_tail",
    op_func=qwen_cute_moe_tail,
    fake_impl=qwen_cute_moe_tail_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)
