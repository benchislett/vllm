# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Any

import torch

from vllm.v1.attention.backend import CommonAttentionMetadata

RecurrentStateCaches = dict[str, tuple[torch.Tensor, ...]]


class RecurrentDraftStateManager:
    """Roll back recurrent draft state after autoregressive speculation."""

    def __init__(self, get_caches: Callable[[], RecurrentStateCaches]) -> None:
        self.get_caches = get_caches
        self._snapshots: dict[str, tuple[torch.Tensor, tuple[torch.Tensor, ...]]] = {}

    @staticmethod
    def _get_state_indices(metadata: Any) -> torch.Tensor:
        indices = []

        prefill_indices = getattr(metadata, "state_indices_tensor_p", None)
        if prefill_indices is not None and prefill_indices.numel() > 0:
            indices.append(prefill_indices.reshape(-1).to(dtype=torch.long))

        decode_indices = getattr(metadata, "state_indices_tensor_d", None)
        if decode_indices is not None and decode_indices.numel() > 0:
            indices.append(
                decode_indices.reshape(decode_indices.shape[0], -1)[:, 0].to(
                    dtype=torch.long
                )
            )

        if not indices:
            raise ValueError("Recurrent draft metadata has no active state indices")

        return torch.cat(indices)

    def capture(self, per_layer_attn_metadata: dict[str, object]) -> None:
        snapshots = {}
        for layer_name, cache_tensors in self.get_caches().items():
            metadata = per_layer_attn_metadata[layer_name]
            state_index = self._get_state_indices(metadata)
            snapshots[layer_name] = (
                state_index.clone(),
                tuple(
                    state.index_select(0, state_index).clone()
                    for state in cache_tensors
                ),
            )
        self._snapshots = snapshots

    def restore(self) -> None:
        if not self._snapshots:
            return

        caches = self.get_caches()
        for layer_name, (state_index, snapshots) in self._snapshots.items():
            cache_tensors = caches[layer_name]
            if len(cache_tensors) != len(snapshots):
                raise ValueError(
                    f"Recurrent state count changed for {layer_name}: "
                    f"{len(snapshots)} to {len(cache_tensors)}"
                )
            for state, snapshot in zip(cache_tensors, snapshots):
                state.index_copy_(0, state_index, snapshot)


def compact_recurrent_mtp_inputs(
    target_token_ids: torch.Tensor,
    target_positions: torch.Tensor,
    target_hidden_states: torch.Tensor,
    token_indices_to_sample: torch.Tensor,
    common_attn_metadata: CommonAttentionMetadata,
    num_rejected_tokens_gpu: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    CommonAttentionMetadata,
]:
    """Remove each request's rejected suffix from a packed MTP update."""
    if common_attn_metadata.dcp_local_seq_lens is not None:
        raise ValueError("Recurrent MTP does not support decode context parallelism")

    batch_size = common_attn_metadata.batch_size()
    if num_rejected_tokens_gpu.numel() < batch_size:
        raise ValueError("Missing rejected-token counts for recurrent MTP batch")

    num_rejected_cpu = num_rejected_tokens_gpu[:batch_size].to(
        device="cpu", dtype=torch.int32
    )
    if not torch.any(num_rejected_cpu):
        return (
            target_token_ids,
            target_positions,
            target_hidden_states,
            token_indices_to_sample,
            common_attn_metadata,
        )

    query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
    query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    retained_query_lens_cpu = query_lens_cpu - num_rejected_cpu
    if torch.any(retained_query_lens_cpu <= 0):
        raise ValueError("Each recurrent MTP update must retain at least one token")

    retained_token_indices_cpu = torch.cat(
        [
            torch.arange(start, start + length, dtype=torch.long)
            for start, length in zip(
                query_start_loc_cpu[:-1].tolist(),
                retained_query_lens_cpu.tolist(),
            )
        ]
    )
    retained_token_indices = retained_token_indices_cpu.to(
        device=target_token_ids.device
    )
    num_tokens = retained_token_indices.numel()

    new_query_start_loc_cpu = torch.zeros_like(query_start_loc_cpu)
    torch.cumsum(
        retained_query_lens_cpu,
        dim=0,
        out=new_query_start_loc_cpu[1:],
    )
    new_query_start_loc = new_query_start_loc_cpu.to(
        device=common_attn_metadata.query_start_loc.device
    )

    num_rejected_gpu = num_rejected_cpu.to(
        device=common_attn_metadata.seq_lens.device
    )
    seq_lens = common_attn_metadata.seq_lens - num_rejected_gpu

    seq_lens_cpu = common_attn_metadata._seq_lens_cpu
    if seq_lens_cpu is not None:
        seq_lens_cpu = seq_lens_cpu - num_rejected_cpu

    seq_lens_cpu_upper_bound = common_attn_metadata.seq_lens_cpu_upper_bound
    if seq_lens_cpu_upper_bound is not None:
        seq_lens_cpu_upper_bound = (
            seq_lens_cpu_upper_bound - num_rejected_cpu
        )

    positions = common_attn_metadata.positions
    if positions is not None:
        positions = positions[..., retained_token_indices]

    if seq_lens_cpu_upper_bound is not None:
        max_seq_len = int(seq_lens_cpu_upper_bound.max().item())
    elif seq_lens_cpu is not None:
        max_seq_len = int(seq_lens_cpu.max().item())
    else:
        max_seq_len = int(seq_lens.max().item())

    common_attn_metadata = common_attn_metadata.replace(
        query_start_loc=new_query_start_loc,
        query_start_loc_cpu=new_query_start_loc_cpu,
        seq_lens=seq_lens,
        num_actual_tokens=num_tokens,
        max_query_len=int(retained_query_lens_cpu.max().item()),
        max_seq_len=max_seq_len,
        slot_mapping=common_attn_metadata.slot_mapping[retained_token_indices],
        positions=positions,
        seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cache=None,
        _token_to_req_indices_cache=None,
    )
    token_indices_to_sample = (new_query_start_loc[1:] - 1).to(
        device=token_indices_to_sample.device,
        dtype=token_indices_to_sample.dtype,
    )

    return (
        target_token_ids[retained_token_indices],
        target_positions[..., retained_token_indices],
        target_hidden_states[retained_token_indices],
        token_indices_to_sample,
        common_attn_metadata,
    )
