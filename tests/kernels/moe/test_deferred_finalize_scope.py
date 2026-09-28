# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A call-scoped consumer cannot change other MoE calls or producer limits."""

from dataclasses import replace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)


def make_config(**parallel_overrides):
    parallel = replace(
        FusedMoEParallelConfig.make_no_parallel(),
        **({"tp_size": 8} | parallel_overrides),
    )
    return FusedMoEConfig(
        num_experts=32,
        experts_per_token=10,
        hidden_dim=8192,
        intermediate_size=512,
        num_local_experts=32,
        num_logical_experts=32,
        activation=MoEActivation.SILU,
        device="cpu",
        routing_method=RoutingMethodType.Renormalize,
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
    )


def test_scoped_finalize_restores_nested_requests_and_errors():
    config, other = make_config(), make_config()
    config.limit_deferred_moe_finalize(48)
    assert not config.should_defer_moe_finalize(1)
    with config.defer_moe_finalize_for_call(64):
        assert config.should_defer_moe_finalize(48)
        assert not config.should_defer_moe_finalize(49)
        assert not config.should_defer_moe_finalize(0)
        assert not other.should_defer_moe_finalize(1)
        with (
            pytest.raises(RuntimeError, match="producer failed"),
            config.defer_moe_finalize_for_call(8),
        ):
            assert config.should_defer_moe_finalize(8)
            assert not config.should_defer_moe_finalize(9)
            raise RuntimeError("producer failed")
        assert config.should_defer_moe_finalize(48)
    assert not config.use_deferred_moe_finalize
    assert config.defer_moe_finalize_max_num_tokens == 48
    config.defer_moe_finalize(32)
    with config.defer_moe_finalize_for_call(8):
        assert not config.should_defer_moe_finalize(9)
    assert config.should_defer_moe_finalize(32)
    assert not config.should_defer_moe_finalize(33)


@pytest.mark.parametrize("capacity", [0, -1])
def test_scoped_finalize_requires_capacity(capacity):
    config = make_config()
    with (
        pytest.raises(ValueError, match="positive token capacity"),
        config.defer_moe_finalize_for_call(capacity),
    ):
        pytest.fail("Invalid scope was entered")
    assert not config.use_deferred_moe_finalize


@pytest.mark.parametrize(
    "parallel",
    [
        {},
        {"tp_size": 1, "ep_size": 8, "use_ep": True},
        {"dp_size": 2},
        {"pcp_size": 2},
        {"sp_size": 2},
        {"tp_size": 1},
    ],
)
def test_scoped_finalize_parallel_contract(parallel):
    config = make_config(**parallel)
    is_tp = not parallel
    is_replicated_ep = parallel.get("ep_size") == 8
    with config.defer_moe_finalize_for_call(16):
        assert config.should_defer_moe_finalize(1) == is_tp
    with config.defer_moe_finalize_for_call(16, allow_replicated_ep=True):
        assert config.should_defer_moe_finalize(1) == (is_tp or is_replicated_ep)
    # The existing model-wide API deliberately remains TP-only.
    config.defer_moe_finalize(16)
    assert config.should_defer_moe_finalize(1) == is_tp


def test_scoped_finalize_excludes_padding_and_ep_dispatch():
    config = make_config(tp_size=1, ep_size=8, use_ep=True, dp_size=8)
    with config.defer_moe_finalize_for_call(16, allow_replicated_ep=True):
        assert not config.should_defer_moe_finalize(1)
    config = make_config()
    config.hidden_dim_unpadded = 4096
    with config.defer_moe_finalize_for_call(16):
        assert not config.should_defer_moe_finalize(1)
