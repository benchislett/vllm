# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for sampler flag tests", allow_module_level=True)

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.logits_processor import (
    LogitsContext,
    LogitsProcessor,
)
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device("cuda")
VOCAB_SIZE = 128


class MockReasoningConfig:
    reasoning_start_token_ids = [90]
    reasoning_end_token_ids = [91]
    natural_reasoning_end_token_ids = [91]


def _make_sampler(custom_logits_processors: Sequence[LogitsProcessor] = ()) -> Sampler:
    req_states = RequestState(
        max_num_reqs=4,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=1,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    return Sampler(
        vllm_config=SimpleNamespace(reasoning_config=MockReasoningConfig()),
        max_num_reqs=4,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
        req_states=req_states,
        custom_logits_processors=custom_logits_processors,
    )


@pytest.mark.parametrize(
    ("sampling_params", "expected"),
    [
        pytest.param(SamplingParams(), False, id="defaults"),
        pytest.param(SamplingParams(temperature=0.0), False, id="greedy"),
        pytest.param(
            SamplingParams(thinking_token_budget=3), True, id="thinking-budget"
        ),
        pytest.param(SamplingParams(logit_bias={1: 1.0}), True, id="logit-bias"),
        pytest.param(SamplingParams(frequency_penalty=0.1), True, id="penalty"),
        pytest.param(SamplingParams(_bad_words_token_ids=[[1]]), True, id="bad-words"),
        pytest.param(SamplingParams(temperature=0.7), True, id="temperature"),
        pytest.param(SamplingParams(min_p=0.1), True, id="min-p"),
        pytest.param(SamplingParams(top_k=10), True, id="top-k"),
        pytest.param(SamplingParams(top_p=0.9), True, id="top-p"),
        pytest.param(
            SamplingParams.for_sampler_warmup(), True, id="all-logits-processors"
        ),
    ],
)
def test_logits_processing_cache_matches_request_features(
    sampling_params: SamplingParams, expected: bool
):
    sampler = _make_sampler()
    sampler.add_request(3, sampling_params=sampling_params)

    assert sampler.needs_logits_processing[3] == expected


def test_logits_processing_cache_is_overwritten_when_slot_is_reused():
    sampler = _make_sampler()
    sampler.add_request(3, SamplingParams.for_sampler_warmup())
    sampler.add_request(3, SamplingParams())

    assert not sampler.needs_logits_processing[3]


class _GateProcessor(LogitsProcessor):
    """Reports a fixed admission decision from add_request()."""

    def __init__(self, admitted: bool):
        self.admitted = admitted

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        return self.admitted

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        raise AssertionError("the pipeline never runs in this test")


@pytest.mark.parametrize("admitted", [True, False])
def test_custom_processor_add_request_gates_pipeline_flag(admitted: bool):
    """A custom processor's add_request() return value is OR-ed into the
    per-request needs_logits_processing flag."""
    sampler = _make_sampler(custom_logits_processors=[_GateProcessor(admitted)])
    sampler.add_request(3, SamplingParams())

    assert sampler.needs_logits_processing[3] == admitted


def test_logits_processing_cache_only_checks_active_requests():
    sampler = _make_sampler()
    sampler.add_request(0, SamplingParams(temperature=0.0))
    sampler.add_request(2, SamplingParams.for_sampler_warmup())

    sampling_only = np.array([0], dtype=np.int32)
    with_processing = np.array([0, 2], dtype=np.int32)

    assert not np.any(sampler.needs_logits_processing[sampling_only])
    assert np.any(sampler.needs_logits_processing[with_processing])


@pytest.mark.parametrize(
    "sampling_params",
    [
        SamplingParams(temperature=0, seed=137),
        SamplingParams(temperature=1, seed=137),
        SamplingParams(temperature=1, seed=137, top_k=20, top_p=0.95),
        SamplingParams(temperature=1, top_k=20, top_p=0.95),
    ],
    ids=["greedy", "seeded", "seeded-filtered", "unseeded-filtered"],
)
@pytest.mark.skipif(torch.version.hip is not None, reason="CUDA sampling graphs")
def test_sampling_graph_preserves_request_reuse_rng_and_outputs(
    sampling_params, default_vllm_config, dist_init
):
    """Replay must refresh request state and leave earlier async outputs intact."""
    from vllm.v1.sample.ops.topk_topp_triton import reset_buffer_cache
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    from vllm.v1.worker.gpu.sample.cudagraph import SamplingCudaGraphManager
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    device = torch.device("cuda:0")
    sampler = _make_sampler()
    hidden_size = 32
    weight = (torch.arange(VOCAB_SIZE * hidden_size, device=device) % 11) - 5
    weight = weight.reshape(VOCAB_SIZE, hidden_size).to(torch.bfloat16)

    def compute_logits(hidden):
        return torch.nn.functional.linear(hidden, weight) / 32

    manager = SamplingCudaGraphManager(
        default_vllm_config,
        sampler,
        None,
        compute_logits,
        None,
        [2],
        hidden_size,
        torch.bfloat16,
        device,
    )
    manager.capture()
    input_buffers = InputBuffers(4, 16, device)
    retained: list[tuple[SamplerOutput, torch.Tensor]] = []
    for iteration, slots in enumerate(([3, 1], [1, 3], [2, 3], [3, 0])):
        batch = InputBatch.make_dummy(2, 4, input_buffers, is_padding=False)
        batch.logits_indices = batch.logits_indices.to(torch.int64)
        batch.idx_mapping_np = np.array(slots, dtype=np.int32)
        batch.idx_mapping = torch.tensor(slots, dtype=torch.int32, device=device)
        batch.expanded_idx_mapping = batch.idx_mapping
        batch.positions.copy_(torch.arange(4, device=device) + iteration * 8)
        for slot in slots:
            params = sampling_params.clone()
            if params.seed is not None:
                params.seed += iteration
            if params.top_k > 0:
                params.top_k = 7 if iteration % 2 else 64
                params.top_p = 0.6 if iteration % 2 else 0.95
            sampler.add_request(slot, params)
            sampler.req_states.prefill_len.np[slot] = 100 if iteration == 2 else 0
        sampler.apply_staged_writes()
        sampler.req_states.prefill_len.copy_to_uva()
        hidden = torch.randint(-4, 5, (4, hidden_size), device=device).to(
            torch.bfloat16
        )
        rng_before = torch.cuda.get_rng_state(device)
        reference = sampler(compute_logits(hidden[batch.logits_indices]), batch)
        rng_after = torch.cuda.get_rng_state(device)
        torch.cuda.set_rng_state(rng_before, device)
        output = manager.run(hidden, batch, None)
        assert output is not None, "The test must exercise graph replay"
        assert torch.equal(torch.cuda.get_rng_state(device), rng_after)
        for name in ("sampled_token_ids", "num_sampled", "num_rejected"):
            assert torch.equal(getattr(output, name), getattr(reference, name))
        for old_output, expected in retained:
            assert torch.equal(old_output.sampled_token_ids, expected)
        retained.append((output, output.sampled_token_ids.clone()))
        if iteration == 1:
            # Captured native filters must own scratch tensors even if eager
            # execution replaces the global cache between replays.
            reset_buffer_cache()

    sampler.add_request(slots[0], SamplingParams(logit_bias={1: 5.0}))
    sampler.apply_staged_writes()
    assert manager.run(hidden, batch, None) is None


@pytest.mark.parametrize("probabilistic", [False, True])
@pytest.mark.parametrize("temperature", [0, 1])
@pytest.mark.skipif(torch.version.hip is not None, reason="CUDA sampling graphs")
def test_sampling_graph_verification_accepts_and_rejects(
    probabilistic, temperature, default_vllm_config, dist_init
):
    """Changing draft tokens must change accepted lengths on actual replay."""
    from vllm.config import SpeculativeConfig
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    from vllm.v1.worker.gpu.sample.cudagraph import SamplingCudaGraphManager
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

    sampler = _make_sampler()
    device = torch.device("cuda:0")
    verifier = RejectionSampler(
        sampler, SpeculativeConfig(method="ngram", num_speculative_tokens=3), device
    )
    weight = torch.full((VOCAB_SIZE, 1), -100, device=device)
    weight[7] = 100

    def compute_logits(hidden):
        return torch.nn.functional.linear(hidden, weight)

    draft_logits = weight.view(1, 1, -1).expand(4, 3, -1).contiguous()
    if not probabilistic:
        draft_logits = None
    manager = SamplingCudaGraphManager(
        default_vllm_config,
        sampler,
        verifier,
        compute_logits,
        draft_logits,
        [2],
        1,
        torch.float32,
        device,
    )
    manager.capture()
    buffers = InputBuffers(4, 8, device)
    hidden = torch.ones(8, 1, device=device)
    for reject in (False, True):
        batch = InputBatch.make_dummy(2, 8, buffers, is_padding=False)
        slots = [3, 1] if reject else [1, 3]
        batch.idx_mapping_np = np.array(slots, dtype=np.int32)
        batch.idx_mapping = torch.tensor(slots, dtype=torch.int32, device=device)
        batch.logits_indices = torch.arange(8, dtype=torch.int64, device=device)
        batch.expanded_idx_mapping = batch.idx_mapping.repeat_interleave(4)
        batch.expanded_local_pos = torch.arange(
            4, dtype=torch.int32, device=device
        ).repeat(2)
        batch.cu_num_logits = batch.query_start_loc
        batch.cu_num_logits_np = batch.query_start_loc_np
        batch.num_draft_tokens = 6
        batch.num_draft_tokens_per_req = np.full(2, 3, dtype=np.int32)
        batch.positions.copy_(torch.arange(8, device=device) + 10)
        batch.input_ids.fill_(7)
        if reject:
            batch.input_ids[2] = 8
            if draft_logits is not None:
                draft_logits[slots[0], 1, 7] = -100
                draft_logits[slots[0], 1, 8] = 100
        for slot in slots:
            sampler.add_request(slot, SamplingParams(temperature=temperature, seed=137))
            sampler.req_states.prefill_len.np[slot] = 0
        sampler.apply_staged_writes()
        sampler.req_states.prefill_len.copy_to_uva()
        reference = verifier(compute_logits(hidden), batch, draft_logits)
        output = manager.run(hidden, batch, draft_logits)
        assert output is not None, "The test must exercise verification replay"
        assert reference.num_sampled.tolist() == ([2, 4] if reject else [4, 4])
        for name in ("num_sampled", "num_rejected"):
            assert torch.equal(getattr(output, name), getattr(reference, name))
        for i, count in enumerate(reference.num_sampled.tolist()):
            assert torch.equal(
                output.sampled_token_ids[i, :count],
                reference.sampled_token_ids[i, :count],
            )
