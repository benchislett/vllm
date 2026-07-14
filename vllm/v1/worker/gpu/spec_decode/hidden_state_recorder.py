# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental recorder for speculative-decoding training traces."""

import hashlib
import os
from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import save_file

from vllm.logger import init_logger
from vllm.v1.worker.gpu.input_batch import InputBatch

logger = init_logger(__name__)


@dataclass
class PrefillHiddenStatesChunk:
    request_id: str
    tensors: dict[str, torch.Tensor]


@dataclass
class VerificationHiddenStatesBlock:
    request_id: str
    tensors: dict[str, torch.Tensor]


@dataclass
class _RequestTrace:
    prompt_len: int = 0
    prefill_chunks: list[PrefillHiddenStatesChunk] = field(default_factory=list)
    verification_blocks: list[VerificationHiddenStatesBlock] = field(
        default_factory=list
    )
    prefill_positions: set[int] = field(default_factory=set)


def build_prefill_hidden_states_chunks(
    req_ids: Sequence[str],
    query_start_loc: Sequence[int],
    num_scheduled_tokens: Sequence[int],
    num_computed_tokens: Sequence[int],
    prefill_lens: Sequence[int],
    input_token_ids: torch.Tensor,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
) -> list[PrefillHiddenStatesChunk]:
    """Split newly computed prefill rows into per-request chunks."""
    chunks: list[PrefillHiddenStatesChunk] = []
    for req_idx, request_id in enumerate(req_ids):
        num_prefill_tokens = max(
            0,
            min(
                int(num_scheduled_tokens[req_idx]),
                int(prefill_lens[req_idx]) - int(num_computed_tokens[req_idx]),
            ),
        )
        if num_prefill_tokens == 0:
            continue

        start = int(query_start_loc[req_idx])
        end = start + num_prefill_tokens
        chunks.append(
            PrefillHiddenStatesChunk(
                request_id=request_id,
                tensors={
                    "token_ids": input_token_ids[start:end].clone().contiguous(),
                    "positions": positions[start:end].clone().contiguous(),
                    "hidden_states": hidden_states[start:end].contiguous(),
                },
            )
        )
    return chunks


def build_verification_hidden_states_blocks(
    req_ids: Sequence[str],
    cu_num_logits: Sequence[int],
    verifier_input_token_ids: torch.Tensor,
    verifier_positions: torch.Tensor,
    hidden_states: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    num_sampled: torch.Tensor,
) -> list[VerificationHiddenStatesBlock]:
    """Split a packed verifier batch into per-request trace blocks.

    ``hidden_states`` has shape ``[sum(K + 1), num_layers, hidden_size]``.
    Within each request, row zero is the anchor state and rows ``1:`` are the
    states at the proposed token positions. The same rows shifted left by one
    are the states used to verify each proposal.
    """
    if len(cu_num_logits) != len(req_ids) + 1:
        raise ValueError("cu_num_logits must contain one offset per request plus one")
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [tokens, layers, hidden]")

    blocks: list[VerificationHiddenStatesBlock] = []
    device = hidden_states.device

    for req_idx, request_id in enumerate(req_ids):
        start = int(cu_num_logits[req_idx])
        end = int(cu_num_logits[req_idx + 1])
        num_proposals = end - start - 1
        if num_proposals <= 0:
            continue

        num_sampled_for_req = num_sampled[req_idx : req_idx + 1]
        output_token_ids = sampled_token_ids[req_idx, : num_proposals + 1].clone()
        output_offsets = torch.arange(
            num_proposals + 1, device=device, dtype=num_sampled_for_req.dtype
        )
        output_token_ids = torch.where(
            output_offsets < num_sampled_for_req,
            output_token_ids,
            -1,
        )

        blocks.append(
            VerificationHiddenStatesBlock(
                request_id=request_id,
                tensors={
                    "input_token_ids": verifier_input_token_ids[start:end]
                    .clone()
                    .contiguous(),
                    "positions": verifier_positions[start:end].clone().contiguous(),
                    "hidden_states": hidden_states[start:end].contiguous(),
                    "num_sampled": num_sampled_for_req.contiguous(),
                    "output_token_ids": output_token_ids.contiguous(),
                },
            )
        )

    return blocks


def _offsets(lengths: Sequence[int]) -> torch.Tensor:
    values = [0]
    for length in lengths:
        values.append(values[-1] + length)
    return torch.tensor(values, dtype=torch.int64)


def _cat_or_empty(
    tensors: Sequence[torch.Tensor],
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if tensors:
        return torch.cat(tuple(tensors)).contiguous()
    return torch.empty(shape, dtype=dtype)


def build_request_hidden_states_trace(
    prefill_chunks: Sequence[PrefillHiddenStatesChunk],
    verification_blocks: Sequence[VerificationHiddenStatesBlock],
    prompt_len: int,
) -> dict[str, torch.Tensor]:
    """Combine one request's chunks and blocks into a single tensor mapping."""
    if prefill_chunks:
        example_hidden_states = prefill_chunks[0].tensors["hidden_states"]
    elif verification_blocks:
        example_hidden_states = verification_blocks[0].tensors["hidden_states"]
    else:
        raise ValueError("cannot build an empty hidden-state trace")
    if example_hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [tokens, layers, hidden]")
    num_layers = example_hidden_states.shape[1]
    hidden_size = example_hidden_states.shape[2]

    prefill_token_ids = _cat_or_empty(
        [chunk.tensors["token_ids"] for chunk in prefill_chunks],
        (0,),
        torch.int32,
    )
    prefill_positions = _cat_or_empty(
        [chunk.tensors["positions"] for chunk in prefill_chunks],
        (0,),
        torch.int64,
    )
    prefill_hidden_states = _cat_or_empty(
        [chunk.tensors["hidden_states"] for chunk in prefill_chunks],
        (0, num_layers, hidden_size),
        example_hidden_states.dtype,
    )
    prefill_order = torch.argsort(prefill_positions, stable=True)
    prefill_token_ids = prefill_token_ids[prefill_order]
    prefill_positions = prefill_positions[prefill_order]
    prefill_hidden_states = prefill_hidden_states[prefill_order]

    verification_input_token_ids = _cat_or_empty(
        [block.tensors["input_token_ids"] for block in verification_blocks],
        (0,),
        torch.int32,
    )
    verification_positions = _cat_or_empty(
        [block.tensors["positions"] for block in verification_blocks],
        (0,),
        torch.int64,
    )
    verification_hidden_states = _cat_or_empty(
        [block.tensors["hidden_states"] for block in verification_blocks],
        (0, num_layers, hidden_size),
        example_hidden_states.dtype,
    )
    output_lengths = [
        int(block.tensors["num_sampled"].item()) for block in verification_blocks
    ]
    output_token_ids = _cat_or_empty(
        [
            block.tensors["output_token_ids"][:output_length]
            for block, output_length in zip(verification_blocks, output_lengths)
        ],
        (0,),
        torch.int64,
    )

    return {
        "prompt_len": torch.tensor([prompt_len], dtype=torch.int32),
        "prefill_token_ids": prefill_token_ids,
        "prefill_positions": prefill_positions,
        "prefill_hidden_states": prefill_hidden_states,
        "verification_input_token_ids": verification_input_token_ids,
        "verification_positions": verification_positions,
        "verification_hidden_states": verification_hidden_states,
        "verification_block_offsets": _offsets(
            [len(block.tensors["input_token_ids"]) for block in verification_blocks]
        ),
        "output_token_ids": output_token_ids,
        "output_block_offsets": _offsets(output_lengths),
    }


class SpecDecodeHiddenStatesRecorder:
    """Collect and asynchronously persist per-request verification traces."""

    def __init__(
        self,
        output_dir: str,
        device: torch.device,
        max_pending_writes: int = 8,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.max_pending_writes = max_pending_writes
        self.copy_stream = torch.cuda.Stream(device)
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="vllm-spec-hs"
        )
        self.pending_writes: deque[Future[None]] = deque()
        self.request_traces: dict[str, _RequestTrace] = {}
        self.closed = False

        logger.info(
            "Recording speculative-decoding training hidden states to %s",
            self.output_dir,
        )

    def capture(
        self,
        input_batch: InputBatch,
        aux_hidden_states: list[torch.Tensor],
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        prompt_lens: Sequence[int],
    ) -> None:
        """Enqueue prefill chunks and proposal blocks from one target step."""
        packed_hidden_states = torch.stack(aux_hidden_states, dim=1)
        prefill_chunks = build_prefill_hidden_states_chunks(
            input_batch.req_ids,
            input_batch.query_start_loc_np,
            input_batch.num_scheduled_tokens,
            input_batch.num_computed_tokens_np,
            input_batch.prefill_len_np,
            input_batch.input_ids,
            input_batch.positions,
            packed_hidden_states,
        )
        prefill_chunks = [
            chunk
            for chunk in prefill_chunks
            if self._should_record_request(chunk.request_id)
        ]

        blocks: list[VerificationHiddenStatesBlock] = []
        if input_batch.num_draft_tokens:
            logits_indices = input_batch.logits_indices
            blocks = build_verification_hidden_states_blocks(
                input_batch.req_ids,
                input_batch.cu_num_logits_np.tolist(),
                input_batch.input_ids[logits_indices],
                input_batch.positions[logits_indices],
                packed_hidden_states[logits_indices],
                sampled_token_ids,
                num_sampled,
            )
            blocks = [
                block
                for block in blocks
                if self._should_record_request(block.request_id)
            ]

        if prefill_chunks or blocks:
            prompt_lens_by_request = dict(zip(input_batch.req_ids, prompt_lens))
            self._enqueue_capture(prefill_chunks, blocks, prompt_lens_by_request)

    def finalize_requests(self, request_ids: Sequence[str]) -> None:
        """Finalize one output file for each completed request."""
        if not request_ids:
            return
        self._enqueue_future(
            self.executor.submit(self._finalize_requests, tuple(request_ids))
        )

    @staticmethod
    def _should_record_request(request_id: str) -> bool:
        return not request_id.startswith("_warmup_")

    def _enqueue_capture(
        self,
        prefill_chunks: list[PrefillHiddenStatesChunk],
        blocks: list[VerificationHiddenStatesBlock],
        prompt_lens: dict[str, int],
    ) -> None:
        cpu_prefill_chunks: list[PrefillHiddenStatesChunk] = []
        cpu_blocks: list[VerificationHiddenStatesBlock] = []
        main_stream = torch.cuda.current_stream(self.device)
        self.copy_stream.wait_stream(main_stream)
        with torch.cuda.stream(self.copy_stream):
            for chunk in prefill_chunks:
                cpu_prefill_chunks.append(
                    PrefillHiddenStatesChunk(
                        chunk.request_id,
                        self._copy_tensors_to_cpu(chunk.tensors),
                    )
                )
            for block in blocks:
                cpu_blocks.append(
                    VerificationHiddenStatesBlock(
                        block.request_id,
                        self._copy_tensors_to_cpu(block.tensors),
                    )
                )
            copy_done = torch.Event()
            copy_done.record(self.copy_stream)

        self._enqueue_future(
            self.executor.submit(
                self._store_capture,
                cpu_prefill_chunks,
                cpu_blocks,
                prompt_lens,
                copy_done,
            )
        )

    def _copy_tensors_to_cpu(
        self, tensors: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        cpu_tensors: dict[str, torch.Tensor] = {}
        for name, tensor in tensors.items():
            cpu_tensor = torch.empty_like(tensor, device="cpu", pin_memory=True)
            cpu_tensor.copy_(tensor, non_blocking=True)
            tensor.record_stream(self.copy_stream)
            cpu_tensors[name] = cpu_tensor
        return cpu_tensors

    def _enqueue_future(self, future: Future[None]) -> None:
        self._drain_completed()
        if len(self.pending_writes) >= self.max_pending_writes:
            self.pending_writes.popleft().result()
        self.pending_writes.append(future)

    def _drain_completed(self) -> None:
        while self.pending_writes and self.pending_writes[0].done():
            self.pending_writes.popleft().result()

    def _store_capture(
        self,
        prefill_chunks: list[PrefillHiddenStatesChunk],
        blocks: list[VerificationHiddenStatesBlock],
        prompt_lens: dict[str, int],
        copy_done: torch.Event,
    ) -> None:
        copy_done.synchronize()
        for chunk in prefill_chunks:
            trace = self.request_traces.setdefault(chunk.request_id, _RequestTrace())
            trace.prompt_len = max(trace.prompt_len, int(prompt_lens[chunk.request_id]))
            positions = chunk.tensors["positions"].tolist()
            keep_list = [
                position not in trace.prefill_positions for position in positions
            ]
            keep = torch.tensor(keep_list, dtype=torch.bool)
            if not torch.any(keep):
                continue
            trace.prefill_positions.update(
                position
                for position, should_keep in zip(positions, keep_list)
                if should_keep
            )
            trace.prefill_chunks.append(
                PrefillHiddenStatesChunk(
                    chunk.request_id,
                    {
                        name: tensor[keep].contiguous()
                        for name, tensor in chunk.tensors.items()
                    },
                )
            )
        for block in blocks:
            trace = self.request_traces.setdefault(block.request_id, _RequestTrace())
            trace.prompt_len = max(trace.prompt_len, int(prompt_lens[block.request_id]))
            trace.verification_blocks.append(block)

    def _finalize_requests(self, request_ids: Sequence[str]) -> None:
        for request_id in request_ids:
            trace = self.request_traces.pop(request_id, None)
            if trace is None:
                continue
            tensors = build_request_hidden_states_trace(
                trace.prefill_chunks,
                trace.verification_blocks,
                trace.prompt_len,
            )
            self._write_request_trace(
                self._request_filename(request_id), request_id, tensors
            )

    def _request_filename(self, request_id: str) -> Path:
        digest = hashlib.sha256(request_id.encode()).hexdigest()
        return self.output_dir / f"{digest}.safetensors"

    @staticmethod
    def _write_request_trace(
        filename: Path,
        request_id: str,
        tensors: dict[str, torch.Tensor],
    ) -> None:
        temporary_filename = filename.with_suffix(filename.suffix + ".tmp")
        try:
            save_file(
                tensors,
                str(temporary_filename),
                metadata={
                    "request_id": request_id,
                    "format": "vllm-spec-decode-training-hidden-states-v1",
                },
            )
            os.replace(temporary_filename, filename)
        finally:
            if temporary_filename.exists():
                temporary_filename.unlink()

    def close(self) -> None:
        """Finalize active traces, wait for writes, and surface failures."""
        if self.closed:
            return
        self.closed = True
        self._enqueue_future(self.executor.submit(self._finalize_all_requests))
        try:
            while self.pending_writes:
                self.pending_writes.popleft().result()
        finally:
            self.executor.shutdown(wait=True)

    def _finalize_all_requests(self) -> None:
        self._finalize_requests(tuple(self.request_traces))
