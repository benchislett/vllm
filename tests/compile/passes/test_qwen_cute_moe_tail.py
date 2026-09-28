# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from vllm.compilation.passes.fusion.qwen_cute_moe_tail import (
    QwenCuteMoETailFusionPass,
)
from vllm.config import VllmConfig
from vllm.distributed.device_communicators import cute_allreduce
from vllm.utils.torch_utils import _USE_LAYERNAME, LayerName


@pytest.mark.parametrize("extra_user", [None, "shared", "routed", "sum"])
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("view_outputs", [False, True])
def test_qwen_tail_owns_producer_outputs(
    extra_user, quantized, view_outputs, monkeypatch
):
    """A retained intermediate must keep the original tensor-only MoE call."""
    config = VllmConfig()
    config.kernel_config.enable_cute_moe_finalize = True
    # Test graph ownership independently from distributed workspace setup.
    monkeypatch.setattr(cute_allreduce, "enabled_for_config", lambda _: True)
    fusion = QwenCuteMoETailFusionPass(config)
    layer_name = LayerName("test_moe") if _USE_LAYERNAME else "test_moe"

    def forward(x, logits, residual, weight, scale):
        shared, routed = torch.ops.vllm.moe_forward_shared(
            x, logits, None, None, layer_name, 0
        )
        if view_outputs:
            shared = shared.view(-1, 8192)
            routed = routed.view(-1, 8192)
        local = shared + routed
        normalized, updated = torch.ops.vllm.cute_allreduce_norm(
            local, residual, weight, scale if quantized else None, 1e-6
        )
        extras = {"shared": shared, "routed": routed, "sum": local}
        return (
            (normalized, updated)
            if extra_user is None
            else (normalized, updated, extras[extra_user])
        )

    x = torch.empty(16, 8192, dtype=torch.bfloat16)
    graph = make_fx(forward, tracing_mode="fake")(
        x,
        torch.empty(16, 32),
        torch.empty_like(x),
        torch.empty(8192, dtype=torch.bfloat16),
        torch.ones(1),
    )
    fusion(graph.graph)
    graph.graph.lint()
    targets = [node.target for node in graph.graph.nodes]
    if extra_user is None:
        assert fusion.matched_count == 1
        assert targets.count(torch.ops.vllm.qwen_cute_moe_tail.default) == 1
        assert torch.ops.vllm.moe_forward_shared.default not in targets
        assert torch.ops.vllm.cute_allreduce_norm.default not in targets
    else:
        assert fusion.matched_count == 0
        assert targets.count(torch.ops.vllm.moe_forward_shared.default) == 1
        assert targets.count(torch.ops.vllm.cute_allreduce_norm.default) == 1
        assert torch.ops.vllm.qwen_cute_moe_tail.default not in targets
