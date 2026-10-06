# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent inputs for graphing the LM head and target sampling."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import product
from typing import cast

import numpy as np
import torch

from vllm.compilation.counter import compilation_counter
from vllm.compilation.cudagraph_pool import capture_pool
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import graph_capture
from vllm.logger import init_logger
from vllm.model_executor.offloader.base import get_offloader
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.sample.ops.topk_topp_triton import get_buffer_cache_tensors
from vllm.v1.worker.gpu.input_batch import InputBatch, get_num_sampled_and_rejected
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import SamplingStates
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
    RejectionSampler,
    get_max_chunk_logits,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

logger = init_logger(__name__)


@triton.jit
def _stage_sampling_inputs(
    hidden,
    logits_indices,
    positions,
    input_ids,
    expanded_idx_mapping,
    expanded_local_pos,
    idx_mapping,
    cu_num_logits,
    seq_lens,
    temperature,
    seeds,
    prefill_len,
    top_k,
    top_p,
    out_hidden,
    out_positions,
    out_draft_tokens,
    out_expanded_idx,
    out_local_pos,
    out_idx,
    out_cu,
    out_seq_lens,
    out_temperature,
    out_seeds,
    out_prefill_len,
    out_top_k,
    out_top_p,
    hidden_stride: tl.constexpr,
    hidden_col_stride: tl.constexpr,
    hidden_size: tl.constexpr,
    num_reqs: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    token_idx = tl.load(logits_indices + row)
    req_idx = tl.load(expanded_idx_mapping + row)
    local_pos = tl.load(expanded_local_pos + row)
    pos = tl.load(positions + token_idx)
    token = tl.load(input_ids + token_idx)
    prefill = tl.load(prefill_len + req_idx)
    # Draft slots before prefill completion are placeholders, as in
    # gather_draft_sampled(). They must reject even when the token matches.
    token = tl.where((local_pos > 0) & (pos - local_pos < prefill), -1, token)
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(
        hidden + token_idx * hidden_stride + offsets * hidden_col_stride,
        offsets < hidden_size,
        other=0,
    )
    tl.store(out_hidden + row * hidden_size + offsets, values, offsets < hidden_size)
    tl.store(out_positions + row, pos)
    tl.store(out_draft_tokens + row, token)
    tl.store(out_expanded_idx + row, req_idx)
    tl.store(out_local_pos + row, local_pos)
    tl.store(out_top_k + row, tl.load(top_k + req_idx))
    tl.store(out_top_p + row, tl.load(top_p + req_idx))
    if row < num_reqs:
        slot = tl.load(idx_mapping + row)
        tl.store(out_idx + row, slot)
        tl.store(out_cu + row, tl.load(cu_num_logits + row))
        tl.store(out_seq_lens + row, tl.load(seq_lens + row))
        tl.store(out_temperature + slot, tl.load(temperature + slot))
        tl.store(out_seeds + slot, tl.load(seeds + slot))
        tl.store(out_prefill_len + slot, tl.load(prefill_len + slot))
        if row == 0:
            tl.store(out_cu + num_reqs, tl.load(cu_num_logits + num_reqs))


@dataclass
class SamplingGraphInputs:
    hidden_states: torch.Tensor
    positions: torch.Tensor
    draft_tokens: torch.Tensor
    expanded_idx_mapping: torch.Tensor
    expanded_local_pos: torch.Tensor
    idx_mapping: torch.Tensor
    cu_num_logits: torch.Tensor
    seq_lens: torch.Tensor
    temperature: torch.Tensor
    seeds: torch.Tensor
    prefill_len: torch.Tensor
    top_k: torch.Tensor
    top_p: torch.Tensor

    @classmethod
    def allocate(
        cls,
        max_num_reqs: int,
        max_num_logits: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "SamplingGraphInputs":
        def zeros(size: int, dtype: torch.dtype) -> torch.Tensor:
            return torch.zeros(size, dtype=dtype, device=device)

        return cls(
            hidden_states=torch.zeros(
                (max_num_logits, hidden_size), dtype=dtype, device=device
            ),
            positions=zeros(max_num_logits, torch.int64),
            draft_tokens=zeros(max_num_logits, torch.int64),
            expanded_idx_mapping=zeros(max_num_logits, torch.int32),
            expanded_local_pos=zeros(max_num_logits, torch.int32),
            idx_mapping=zeros(max_num_reqs, torch.int32),
            cu_num_logits=zeros(max_num_reqs + 1, torch.int32),
            seq_lens=zeros(max_num_reqs, torch.int32),
            temperature=zeros(max_num_reqs, torch.float32),
            seeds=zeros(max_num_reqs, torch.int64),
            prefill_len=zeros(max_num_reqs, torch.int32),
            top_k=zeros(max_num_logits, torch.int32),
            top_p=zeros(max_num_logits, torch.float32),
        )

    def view(self, num_reqs: int, num_logits: int) -> "SamplingGraphInputs":
        # These three state tensors keep persistent request-slot indexing,
        # including when active requests are reordered.
        return replace(
            self,
            hidden_states=self.hidden_states[:num_logits],
            positions=self.positions[:num_logits],
            draft_tokens=self.draft_tokens[:num_logits],
            expanded_idx_mapping=self.expanded_idx_mapping[:num_logits],
            expanded_local_pos=self.expanded_local_pos[:num_logits],
            idx_mapping=self.idx_mapping[:num_reqs],
            cu_num_logits=self.cu_num_logits[: num_reqs + 1],
            seq_lens=self.seq_lens[:num_reqs],
            top_k=self.top_k[:num_logits],
            top_p=self.top_p[:num_logits],
        )

    def stage(
        self,
        hidden_states: torch.Tensor,
        batch: InputBatch,
        states: SamplingStates,
        prefill_len: torch.Tensor,
    ) -> None:
        num_logits, hidden_size = self.hidden_states.shape
        _stage_sampling_inputs[(num_logits,)](
            hidden_states,
            batch.logits_indices,
            batch.positions,
            batch.input_ids,
            batch.expanded_idx_mapping,
            batch.expanded_local_pos,
            batch.idx_mapping,
            batch.cu_num_logits,
            batch.seq_lens,
            states.temperature.gpu,
            states.seeds.gpu,
            prefill_len,
            states.top_k.gpu,
            states.top_p.gpu,
            self.hidden_states,
            self.positions,
            self.draft_tokens,
            self.expanded_idx_mapping,
            self.expanded_local_pos,
            self.idx_mapping,
            self.cu_num_logits,
            self.seq_lens,
            self.temperature,
            self.seeds,
            self.prefill_len,
            self.top_k,
            self.top_p,
            hidden_states.stride(0),
            hidden_states.stride(1),
            hidden_size,
            batch.num_reqs,
            triton.next_power_of_2(hidden_size),
            num_warps=8,
        )


@dataclass(frozen=True)
class SamplingGraphKey:
    num_reqs: int
    speculative: bool
    top_k: bool
    top_p: bool
    flashinfer: bool = False


class SamplingCudaGraphManager:
    """Capture native target sampling at startup; replay never captures or syncs."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        sampler: Sampler,
        rejection_sampler: RejectionSampler | None,
        compute_logits: Callable[[torch.Tensor], torch.Tensor],
        draft_logits: torch.Tensor | None,
        capture_sizes: list[int],
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.vllm_config = vllm_config
        self.sampler = sampler
        self.rejection_sampler = rejection_sampler
        self.compute_logits = compute_logits
        self.draft_logits = draft_logits
        self.device = device
        self.num_speculative_tokens = (
            rejection_sampler.num_speculative_steps if rejection_sampler else 0
        )
        self.capture_sizes = sorted(
            {n for n in capture_sizes if 0 < n <= sampler.sampling_states.max_num_reqs},
            reverse=True,
        )
        self.inputs = SamplingGraphInputs.allocate(
            sampler.sampling_states.max_num_reqs,
            max(self.capture_sizes, default=1) * (self.num_speculative_tokens + 1),
            hidden_size,
            dtype,
            device,
        )
        self.rng_seed = torch.zeros(1, dtype=torch.int64, device=device)
        self.rng_offset = torch.zeros_like(self.rng_seed)
        self.graphs: dict[SamplingGraphKey, torch.cuda.CUDAGraph] = {}
        self.outputs: dict[
            SamplingGraphKey, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        self.input_views: dict[SamplingGraphKey, SamplingGraphInputs] = {}
        self.filter_workspace: tuple[torch.Tensor, ...] = ()

    def _sample(
        self, key: SamplingGraphKey, inputs: SamplingGraphInputs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.compute_logits(inputs.hidden_states)
        top_k = inputs.top_k if key.top_k else None
        top_p = inputs.top_p if key.top_p else None
        if key.top_k or key.top_p:
            logits = logits.to(dtype=torch.float32, copy=True)
        if key.speculative:
            logits = apply_top_k_top_p(logits, top_k, top_p)
            tokens, num_sampled = rejection_sample(
                logits,
                self.draft_logits,
                inputs.draft_tokens,
                inputs.cu_num_logits,
                inputs.positions,
                inputs.idx_mapping,
                inputs.expanded_idx_mapping,
                inputs.expanded_local_pos,
                inputs.temperature,
                inputs.seeds,
                self.num_speculative_tokens,
                use_fp64=self.sampler.use_fp64_gumbel,
            )
        else:
            tokens, _ = self.sampler.sample_from_processed_logits(
                logits,
                inputs.expanded_idx_mapping,
                inputs.positions,
                top_k,
                top_p,
                key.flashinfer,
                temperature=inputs.temperature,
                seeds=inputs.seeds,
                flashinfer_rng=(self.rng_seed, self.rng_offset)
                if key.flashinfer
                else None,
            )
            tokens = tokens.view(-1, 1)
            num_sampled = torch.ones_like(inputs.seq_lens)
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled,
            inputs.seq_lens,
            inputs.cu_num_logits,
            inputs.idx_mapping,
            inputs.prefill_len,
        )
        return tokens, num_sampled, num_rejected

    @torch.inference_mode()
    def capture(self) -> None:
        keys = []
        for num_reqs in self.capture_sizes:
            for speculative in (
                (False, True) if self.num_speculative_tokens else (False,)
            ):
                width = self.num_speculative_tokens + 1 if speculative else 1
                if speculative and num_reqs * width > get_max_chunk_logits(
                    self.sampler.sampling_states.vocab_size
                ):
                    continue
                for top_k, top_p in product((False, True), repeat=2):
                    keys.append(SamplingGraphKey(num_reqs, speculative, top_k, top_p))
                    if (
                        not speculative
                        and (top_k or top_p)
                        and self.sampler.use_flashinfer
                    ):
                        keys.append(
                            SamplingGraphKey(num_reqs, False, top_k, top_p, True)
                        )
        # The shared pool's largest allocations should be captured first.
        keys.sort(
            key=lambda key: key.num_reqs
            * (self.num_speculative_tokens + 1 if key.speculative else 1),
            reverse=True,
        )
        pool = current_platform.get_global_graph_pool()
        with graph_capture(device=self.device) as context:
            for key in keys:
                b = key.num_reqs
                width = self.num_speculative_tokens + 1 if key.speculative else 1
                rows = b * width
                inputs = self.inputs.view(b, rows)
                inputs.idx_mapping.copy_(torch.arange(b, device=self.device))
                inputs.cu_num_logits.copy_(
                    torch.arange(b + 1, device=self.device) * width
                )
                inputs.expanded_idx_mapping.copy_(
                    torch.arange(b, device=self.device).repeat_interleave(width)
                )
                inputs.expanded_local_pos.copy_(
                    torch.arange(width, device=self.device).repeat(b)
                )
                inputs.temperature.fill_(1)
                inputs.top_k.fill_(min(20, self.sampler.sampling_states.vocab_size))
                inputs.top_p.fill_(0.95)
                inputs.seq_lens.fill_(1)
                self._sample(key, inputs)
                graph = torch.cuda.CUDAGraph()
                get_offloader().sync_prev_onload()
                with (
                    capture_pool(pool, self.vllm_config) as active_pool,
                    torch.cuda.graph(graph, pool=active_pool, stream=context.stream),
                ):
                    output = self._sample(key, inputs)
                    get_offloader().join_after_forward()
                self.graphs[key] = graph
                self.outputs[key] = output
                self.input_views[key] = inputs
                # Native eager filtering may grow or reset its global caches.
                self.filter_workspace += get_buffer_cache_tensors()
                compilation_counter.num_cudagraph_captured += 1
        logger.info(
            "Captured %d LM-head and target-sampling CUDA graphs", len(self.graphs)
        )

    def run(
        self,
        hidden_states: torch.Tensor,
        batch: InputBatch,
        draft_logits: torch.Tensor | None,
    ) -> SamplerOutput | None:
        sampler = self.sampler
        ids, states = batch.idx_mapping_np, sampler.sampling_states
        if (
            batch.num_reqs == 0
            or hidden_states.dtype != self.inputs.hidden_states.dtype
            or hidden_states.shape[-1] != self.inputs.hidden_states.shape[-1]
            or sampler.compute_nans
            or sampler.return_sampling_mask
            or sampler.trace_replay_state is not None
            or sampler.get_logprobs_dims(ids) is not None
            or np.any(
                (states.temperature.np[ids] != 0) & (states.temperature.np[ids] != 1)
            )
            or np.any(states.min_p.np[ids] != 0)
            or tuple(type(p) for p in sampler.logits_processors)
            != (LogitBiasState, PenaltiesState, BadWordsState)
        ):
            return None
        bias = cast(LogitBiasState, sampler.logits_processors[0])
        penalties = cast(PenaltiesState, sampler.logits_processors[1])
        bad_words = cast(BadWordsState, sampler.logits_processors[2])
        thinking = sampler.thinking_budget_state
        if (
            np.any(bias.use_logit_bias[ids])
            or np.any(penalties.use_penalty[ids])
            or np.any(bad_words.num_bad_words.np[ids])
            or (thinking.enabled and np.any(thinking.use_thinking_budget[ids]))
        ):
            return None
        speculative = batch.num_draft_tokens > 0
        if speculative and draft_logits is not self.draft_logits:
            return None
        if speculative and (
            self.rejection_sampler is None
            or self.rejection_sampler.enable_adaptive_verification
            or self.rejection_sampler.use_block_verification
            or self.rejection_sampler.synthetic_conditional_rates is not None
            or self.rejection_sampler.watermark_key is not None
        ):
            return None
        width = self.num_speculative_tokens + 1 if speculative else 1
        if batch.logits_indices.numel() != batch.num_reqs * width:
            return None
        top_k = bool(np.any(states.top_k.np[ids] != states.vocab_size))
        top_p = bool(np.any(states.top_p.np[ids] != 1))
        use_flashinfer = (
            not speculative
            and sampler.use_flashinfer
            and (top_k or top_p)
            and not np.any(states.temperature.np[ids] == 0)
            and not np.any(states.seeds_set[ids])
        )
        key = SamplingGraphKey(
            batch.num_reqs, speculative, top_k, top_p, use_flashinfer
        )
        graph = self.graphs.get(key)
        if graph is None:
            return None
        inputs = self.input_views[key]
        inputs.stage(hidden_states, batch, states, sampler.req_states.prefill_len.gpu)
        if use_flashinfer:
            from flashinfer.sampling import get_seed_and_offset

            seed, offset = get_seed_and_offset(32 * batch.num_reqs, device=self.device)
            self.rng_seed.fill_(seed)
            self.rng_offset.fill_(offset)
        graph.replay()
        tokens, sampled, rejected = self.outputs[key]
        # AsyncOutput reads on a different stream, possibly after the next replay.
        return SamplerOutput(
            tokens.clone(), None, None, sampled.clone(), rejected.clone()
        )
