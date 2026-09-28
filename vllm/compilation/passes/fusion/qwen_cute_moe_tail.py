# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Absorb an exclusively owned MoE producer into an existing CuTe norm tail."""

import operator

import torch
from torch import fx

from vllm.compilation.passes.vllm_inductor_pass import (
    VllmInductorPass,
    VllmPatternMatcherPass,
)
from vllm.distributed.device_communicators import cute_allreduce
from vllm.model_executor.layers.fused_moe import qwen_cute_moe_tail as runtime

_VIEWS = (
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.alias.default,
)


def _unview(node, chain: set[fx.Node]):
    while (
        isinstance(node, fx.Node)
        and node.op == "call_function"
        and node.target in _VIEWS
    ):
        chain.add(node)
        node = node.args[0]
    return node


def find_moe_producer(node, final_user: fx.Node):
    """Find an unscaled shared+routed sum whose intermediates have no other users."""
    if not isinstance(node, fx.Node) or len(node.users) != 1:
        return None
    value = node.meta.get("val")
    if not (
        isinstance(value, torch.Tensor)
        and value.ndim == 2
        and value.shape[-1] == cute_allreduce.HIDDEN_SIZE
        and value.dtype == torch.bfloat16
    ):
        return None
    chain: set[fx.Node] = set()
    add = _unview(node, chain)
    if not (
        isinstance(add, fx.Node)
        and add.op == "call_function"
        and add.target == torch.ops.aten.add.Tensor
        and add.kwargs.get("alpha", 1) == 1
    ):
        return None
    chain.add(add)
    pieces = [_unview(arg, chain) for arg in add.args[:2]]
    if any(not isinstance(p, fx.Node) or p.target != operator.getitem for p in pieces):
        return None
    if {p.args[1] for p in pieces} != {0, 1} or pieces[0].args[0] is not pieces[1].args[
        0
    ]:
        return None
    moe = pieces[0].args[0]
    if not (
        isinstance(moe, fx.Node)
        and moe.op == "call_function"
        and moe.target == torch.ops.vllm.moe_forward_shared.default
    ):
        return None
    chain.update(pieces)
    chain.add(moe)
    for value in chain:
        if any(
            user not in chain and not (value is node and user is final_user)
            for user in value.users
        ):
            return None
    return moe, chain


class QwenCuteMoETailFusionPass(VllmPatternMatcherPass):
    def __init__(self, config):
        super().__init__(config)
        self.disabled = not (
            config.kernel_config.enable_cute_moe_finalize
            and cute_allreduce.enabled_for_config(config)
        )

    def is_applicable_for_range(self, compile_range):
        return not self.disabled and compile_range.end <= cute_allreduce.MAX_TOKENS

    def uuid(self):
        return self.hash_source(
            type(self),
            find_moe_producer,
            _unview,
            runtime.qwen_cute_moe_tail,
            runtime.can_defer,
            repr(cute_allreduce.build_policy(include_moe_finalize=True)),
            str(self.disabled),
        )

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = 0
        if self.disabled:
            return
        for node in list(graph.nodes):
            if node.target != torch.ops.vllm.cute_allreduce_norm.default:
                continue
            found = find_moe_producer(node.args[0], node)
            if found is None:
                continue
            moe, chain = found
            moe_args = tuple(
                moe.args[i] if i < len(moe.args) else moe.kwargs[arg.name]
                for i, arg in enumerate(moe.target._schema.arguments)
            )
            with graph.inserting_before(node):
                fused = graph.call_function(
                    torch.ops.vllm.qwen_cute_moe_tail.default,
                    args=(*moe_args, *node.args[1:]),
                )
                fused.meta = dict(node.meta)
                node.replace_all_uses_with(fused)
            graph.erase_node(node)
            # Explicit removal also removes the original opaque producer. Relying
            # on generic DCE here could leave two executions of the same MoE.
            for old in reversed(list(graph.nodes)):
                if old in chain:
                    assert not old.users
                    graph.erase_node(old)
            self.matched_count += 1
        self.match_table[self.pass_name] += self.matched_count
        if self.matched_count:
            graph.lint()
