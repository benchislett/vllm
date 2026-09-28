# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Balanced expert assignments scoped to FlashInfer's dummy tuning forwards."""

from collections.abc import Iterator
from contextlib import contextmanager

import torch

from vllm.distributed import get_ep_group
from vllm.model_executor.layers.fused_moe import MoERunner
from vllm.model_executor.layers.fused_moe.router.base_router import FusedMoERouter
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter,
)


class BalancedTuningRouter(FusedTopKRouter):
    def __init__(self, router: FusedMoERouter, rank: int):
        if (
            not isinstance(router, FusedTopKRouter)
            or type(router) is not FusedTopKRouter
            or router.eplb_state is not None
            or not router.renormalize
            or router.scoring_func != "softmax"
        ):
            raise ValueError(
                "Balanced tuning requires normalized softmax routing without EPLB"
            )
        super().__init__(router.top_k, router.global_num_experts)
        self.original = router
        self.rank = rank
        self.calls = 0
        self.permutation = torch.randperm(
            router.global_num_experts,
            generator=torch.Generator(device="cpu").manual_seed(7001),
            device="cpu",
        )
        self.cached_ids: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.compiler.is_compiling() or (
            hidden_states.is_cuda and torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError("Balanced tuning routes must not enter serving graphs")
        weights, ids = self.original._compute_routing(
            hidden_states, router_logits, indices_type, input_ids=input_ids
        )
        rows = ids.shape[0]
        key = (rows, ids.device, ids.dtype)
        if key not in self.cached_ids:
            positions = torch.arange(rows * self.top_k, device="cpu")
            # Adjacent rank offsets balance the gathered stream and every prefix.
            positions = (
                positions + self.rank * rows * self.top_k
            ) % self.global_num_experts
            self.cached_ids[key] = self.permutation[positions].reshape_as(ids).to(ids)
        self.calls += 1
        # MoE dispatchers may take ownership of the returned routing tensor.
        return weights, self.cached_ids[key].clone()


@contextmanager
def balanced_moe_routing(model: torch.nn.Module) -> Iterator[None]:
    replacements = []
    rank = get_ep_group().rank_in_group
    for module in model.modules():
        if isinstance(module, MoERunner):
            if module.is_monolithic:
                raise ValueError("Balanced tuning requires a modular MoE backend")
            replacements.append((module, BalancedTuningRouter(module.router, rank)))
    if not replacements:
        raise ValueError("Balanced tuning found no supported MoE layers")
    try:
        for module, router in replacements:
            module.router = router
        yield
        if not any(router.calls for _, router in replacements):
            raise RuntimeError("Balanced tuning did not execute the modular router")
    finally:
        for module, router in replacements:
            module.router = router.original
