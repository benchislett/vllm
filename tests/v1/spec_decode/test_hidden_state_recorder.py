# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from vllm.config import SpeculativeConfig
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    get_eagle3_aux_layers_from_config,
)
from vllm.v1.worker.gpu.spec_decode.hidden_state_recorder import (
    PrefillHiddenStatesChunk,
    SpecDecodeHiddenStatesRecorder,
    build_prefill_hidden_states_chunks,
    build_request_hidden_states_trace,
    build_verification_hidden_states_blocks,
)


def _build_blocks():
    verifier_input_token_ids = torch.tensor(
        [10, 11, 12, 13, 20, 21, 22], dtype=torch.int32
    )
    verifier_positions = torch.tensor(
        [100, 101, 102, 103, 200, 201, 202], dtype=torch.int64
    )
    hidden_states = torch.arange(7 * 2 * 3, dtype=torch.float32).reshape(7, 2, 3)
    sampled_token_ids = torch.tensor(
        [
            [11, 99, 777, 888],
            [21, 22, 98, 999],
        ],
        dtype=torch.int64,
    )
    num_sampled = torch.tensor([2, 3], dtype=torch.int32)

    return build_verification_hidden_states_blocks(
        req_ids=["request/0", "request-1"],
        cu_num_logits=[0, 4, 7],
        verifier_input_token_ids=verifier_input_token_ids,
        verifier_positions=verifier_positions,
        hidden_states=hidden_states,
        sampled_token_ids=sampled_token_ids,
        num_sampled=num_sampled,
    )


def _prefill_chunk(
    positions: list[int], hidden_values: list[float]
) -> PrefillHiddenStatesChunk:
    return PrefillHiddenStatesChunk(
        request_id="request/0",
        tensors={
            "token_ids": torch.tensor(positions, dtype=torch.int32) + 10,
            "positions": torch.tensor(positions, dtype=torch.int64),
            "hidden_states": torch.tensor(hidden_values).reshape(-1, 1, 1),
        },
    )


def test_build_prefill_hidden_states_chunks_limits_rows_to_prefill():
    chunks = build_prefill_hidden_states_chunks(
        req_ids=["chunked", "finishing"],
        query_start_loc=[0, 2, 6],
        num_scheduled_tokens=[2, 4],
        num_computed_tokens=[0, 2],
        prefill_lens=[4, 4],
        input_token_ids=torch.tensor([10, 11, 20, 21, 22, 23]),
        positions=torch.tensor([0, 1, 2, 3, 4, 5]),
        hidden_states=torch.arange(12).reshape(6, 1, 2),
    )

    assert len(chunks) == 2
    assert chunks[0].tensors["token_ids"].tolist() == [10, 11]
    assert chunks[0].tensors["positions"].tolist() == [0, 1]
    assert chunks[1].tensors["token_ids"].tolist() == [20, 21]
    assert chunks[1].tensors["positions"].tolist() == [2, 3]


def test_build_verification_hidden_states_blocks_keeps_rejected_suffix():
    first, second = _build_blocks()

    assert first.request_id == "request/0"
    assert first.tensors["input_token_ids"].tolist() == [10, 11, 12, 13]
    assert first.tensors["num_sampled"].tolist() == [2]
    assert first.tensors["output_token_ids"].tolist() == [11, 99, -1, -1]
    assert torch.equal(
        first.tensors["hidden_states"], torch.arange(24).reshape(4, 2, 3)
    )

    assert second.request_id == "request-1"
    assert second.tensors["input_token_ids"].tolist() == [20, 21, 22]
    assert second.tensors["num_sampled"].tolist() == [3]
    assert second.tensors["output_token_ids"].tolist() == [21, 22, 98]
    assert second.tensors["hidden_states"].shape == (3, 2, 3)


def test_build_verification_hidden_states_blocks_skips_non_spec_request():
    blocks = build_verification_hidden_states_blocks(
        req_ids=["prefill", "decode"],
        cu_num_logits=[0, 1, 4],
        verifier_input_token_ids=torch.tensor([1, 2, 3, 4]),
        verifier_positions=torch.arange(4),
        hidden_states=torch.zeros(4, 1, 2),
        sampled_token_ids=torch.tensor([[5, -1, -1], [3, 4, 5]]),
        num_sampled=torch.tensor([1, 3], dtype=torch.int32),
    )

    assert [block.request_id for block in blocks] == ["decode"]


def test_build_request_trace_combines_prefills_and_blocks():
    first_block, _ = _build_blocks()
    chunks = [
        _prefill_chunk([2, 3], [2.0, 3.0]),
        _prefill_chunk([0, 1], [0.0, 1.0]),
    ]
    first_block.tensors["hidden_states"] = first_block.tensors["hidden_states"][
        :, :1, :1
    ]

    trace = build_request_hidden_states_trace(chunks, [first_block], prompt_len=4)

    assert trace["prefill_token_ids"].tolist() == [10, 11, 12, 13]
    assert trace["prefill_positions"].tolist() == [0, 1, 2, 3]
    assert trace["prefill_hidden_states"].flatten().tolist() == [0, 1, 2, 3]
    assert trace["prompt_len"].tolist() == [4]
    assert trace["verification_input_token_ids"].tolist() == [10, 11, 12, 13]
    assert trace["verification_block_offsets"].tolist() == [0, 4]
    assert trace["output_token_ids"].tolist() == [11, 99]
    assert trace["output_block_offsets"].tolist() == [0, 2]


def test_request_trace_preserves_cached_prompt_gap():
    trace = build_request_hidden_states_trace(
        [_prefill_chunk([2, 3], [2.0, 3.0])],
        [],
        prompt_len=4,
    )

    assert trace["prompt_len"].tolist() == [4]
    assert trace["prefill_positions"].tolist() == [2, 3]


def test_write_request_trace_uses_one_file_and_metadata(tmp_path):
    first_block, _ = _build_blocks()
    tensors = build_request_hidden_states_trace([], [first_block], prompt_len=4)
    filename = tmp_path / "request.safetensors"

    SpecDecodeHiddenStatesRecorder._write_request_trace(
        filename, first_block.request_id, tensors
    )
    SpecDecodeHiddenStatesRecorder._write_request_trace(
        filename, first_block.request_id, tensors
    )

    assert list(tmp_path.iterdir()) == [filename]
    saved = load_file(filename)
    assert torch.equal(saved["output_token_ids"], tensors["output_token_ids"])
    assert torch.equal(
        saved["verification_hidden_states"], tensors["verification_hidden_states"]
    )
    with safe_open(filename, framework="pt") as handle:
        assert handle.metadata() == {
            "request_id": "request/0",
            "format": "vllm-spec-decode-training-hidden-states-v1",
        }


def test_store_capture_deduplicates_prefill_positions():
    recorder = SpecDecodeHiddenStatesRecorder.__new__(SpecDecodeHiddenStatesRecorder)
    recorder.request_traces = {}
    copy_done = Mock()
    chunks = [
        _prefill_chunk([0, 1], [0.0, 1.0]),
        _prefill_chunk([1, 2], [10.0, 2.0]),
    ]

    recorder._store_capture(chunks, [], {"request/0": 4}, copy_done)

    copy_done.synchronize.assert_called_once_with()
    trace = recorder.request_traces["request/0"]
    assert [chunk.tensors["positions"].tolist() for chunk in trace.prefill_chunks] == [
        [0, 1],
        [2],
    ]


def test_hidden_state_recorder_skips_warmup_requests():
    assert not SpecDecodeHiddenStatesRecorder._should_record_request("_warmup_12_")
    assert SpecDecodeHiddenStatesRecorder._should_record_request("chatcmpl-user")


def test_hidden_state_recording_rejects_methods_without_aux_hidden_states():
    with pytest.raises(
        ValueError,
        match="methods that use auxiliary target hidden states",
    ):
        SpeculativeConfig(
            method="ngram",
            model="ngram",
            num_speculative_tokens=1,
            verification_hidden_states_output_dir="/tmp/hidden-states",
        )


@pytest.mark.parametrize(
    ("method", "expected"),
    [("eagle3", True), ("dflash", True), ("dspark", True), ("mtp", False)],
)
def test_aux_hidden_state_drafting_capability(method, expected):
    config = object.__new__(SpeculativeConfig)
    config.method = method

    assert config.uses_aux_hidden_states_for_drafting() is expected


@pytest.mark.parametrize(
    ("hf_config", "expected"),
    [
        (SimpleNamespace(eagle_aux_hidden_state_layer_ids=[2, 8]), (2, 8)),
        (SimpleNamespace(dflash_config={"target_layer_ids": [1, 7]}), (2, 8)),
        (SimpleNamespace(dspark_target_layer_ids=[1, 7]), (2, 8)),
    ],
)
def test_aux_hidden_state_layers_from_speculator_config(hf_config, expected):
    spec_config = Mock(draft_model_config=Mock(hf_config=hf_config))

    assert get_eagle3_aux_layers_from_config(spec_config) == expected
