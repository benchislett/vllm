# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    try_get_attention_backend,
)
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.config.load import LoadConfig
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.models.nemotron_h_mtp import (
    NemotronHMTP,
    NemotronHMTPAttentionDecoderLayer,
    NemotronHMTPMambaDecoderLayer,
    NemotronHMTPMoEDecoderLayer,
    _make_mtp_final_layernorm,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.spec_decode.recurrent_draft_state import (
    RecurrentDraftStateManager,
    compact_recurrent_mtp_inputs,
)

mimo_7b_dir = "XiaomiMiMo/MiMo-7B-Base"
DEVICE_TYPE = current_platform.device_type


def test_nemotron_h_mtp_final_norm_matches_checkpoint_architecture():
    config = SimpleNamespace(
        hidden_size=4,
        layer_norm_epsilon=1e-5,
        mlp_bias=False,
    )

    norm = _make_mtp_final_layernorm(config)

    assert isinstance(norm, torch.nn.LayerNorm)
    assert norm.bias is None
    assert norm.eps == config.layer_norm_epsilon


@pytest.mark.parametrize(
    ("layer_types", "requires_recurrent_rollback"),
    [
        (
            (NemotronHMTPAttentionDecoderLayer, NemotronHMTPMoEDecoderLayer),
            False,
        ),
        ((NemotronHMTPMambaDecoderLayer, NemotronHMTPMoEDecoderLayer), True),
        (
            (
                NemotronHMTPAttentionDecoderLayer,
                NemotronHMTPMambaDecoderLayer,
                NemotronHMTPMoEDecoderLayer,
            ),
            True,
        ),
    ],
)
def test_nemotron_h_mtp_recurrent_rollback_tracks_layer_type(
    layer_types,
    requires_recurrent_rollback,
):
    layers = {}
    for index, layer_type in enumerate(layer_types):
        layer = layer_type.__new__(layer_type)
        torch.nn.Module.__init__(layer)
        layers[str(index)] = layer

    mtp = NemotronHMTP.__new__(NemotronHMTP)
    torch.nn.Module.__init__(mtp)
    mtp.model = torch.nn.Module()
    mtp.model.layers = torch.nn.ModuleDict(layers)

    actual = mtp.requires_recurrent_draft_state_rollback
    assert actual is requires_recurrent_rollback


def _create_mtp_proposer(num_speculative_tokens: int) -> EagleProposer:
    """Create an MTP proposer with unified model configuration."""
    model_config = ModelConfig(
        model=mimo_7b_dir, runner="generate", max_model_len=100, trust_remote_code=True
    )

    speculative_config = SpeculativeConfig(
        target_model_config=model_config,
        target_parallel_config=ParallelConfig(),
        model=mimo_7b_dir,
        method="mtp",
        num_speculative_tokens=num_speculative_tokens,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(),
        speculative_config=speculative_config,
        device_config=DeviceConfig(device=DEVICE_TYPE),
        parallel_config=ParallelConfig(),
        load_config=LoadConfig(),
        scheduler_config=SchedulerConfig(
            max_model_len=model_config.max_model_len,
            is_encoder_decoder=model_config.is_encoder_decoder,
        ),
    )

    return EagleProposer(vllm_config=vllm_config, device=DEVICE_TYPE)


@pytest.mark.parametrize(
    ("metadata", "state_index"),
    [
        (
            SimpleNamespace(
                state_indices_tensor_p=torch.tensor([1]),
                state_indices_tensor_d=None,
            ),
            1,
        ),
        (
            SimpleNamespace(
                state_indices_tensor_p=None,
                state_indices_tensor_d=torch.tensor([[2, 3]]),
            ),
            2,
        ),
    ],
)
def test_recurrent_draft_state_restores_only_active_page(metadata, state_index):
    conv_state = torch.arange(12).reshape(4, 3).clone()
    ssm_state = torch.arange(24).reshape(4, 2, 3).clone()
    expected_conv = conv_state[state_index].clone()
    expected_ssm = ssm_state[state_index].clone()
    caches = {"mtp.layers.0.mixer": (conv_state, ssm_state)}
    manager = RecurrentDraftStateManager(lambda: caches)

    manager.capture({"mtp.layers.0.mixer": metadata})
    conv_state[state_index].fill_(-1)
    ssm_state[state_index].fill_(-1)
    conv_state[0].fill_(-2)
    manager.restore()

    assert torch.equal(conv_state[state_index], expected_conv)
    assert torch.equal(ssm_state[state_index], expected_ssm)
    if state_index != 0:
        assert torch.equal(conv_state[0], torch.full_like(conv_state[0], -2))


def test_recurrent_draft_state_restores_all_active_pages():
    conv_state = torch.arange(18).reshape(6, 3).clone()
    ssm_state = torch.arange(36).reshape(6, 2, 3).clone()
    active_indices = torch.tensor([1, 4, 2, 5])
    expected_conv = conv_state[active_indices].clone()
    expected_ssm = ssm_state[active_indices].clone()
    caches = {"mtp.layers.0.mixer": (conv_state, ssm_state)}
    manager = RecurrentDraftStateManager(lambda: caches)
    metadata = SimpleNamespace(
        state_indices_tensor_p=torch.tensor([1, 4]),
        state_indices_tensor_d=torch.tensor([[2, 3], [5, 0]]),
    )

    manager.capture({"mtp.layers.0.mixer": metadata})
    conv_state[active_indices] = -1
    ssm_state[active_indices] = -1
    conv_state[0].fill_(-2)
    manager.restore()

    assert torch.equal(conv_state[active_indices], expected_conv)
    assert torch.equal(ssm_state[active_indices], expected_ssm)
    assert torch.equal(conv_state[0], torch.full_like(conv_state[0], -2))


def test_recurrent_draft_state_discards_speculation_across_cycles():
    conv_state = torch.arange(18).reshape(6, 3).clone()
    ssm_state = torch.arange(36).reshape(6, 2, 3).clone()
    active_indices = torch.tensor([1, 4])
    caches = {"mtp.layers.0.mixer": (conv_state, ssm_state)}
    manager = RecurrentDraftStateManager(lambda: caches)
    metadata = SimpleNamespace(
        state_indices_tensor_p=None,
        state_indices_tensor_d=torch.tensor([[1, 2], [4, 5]]),
    )

    accepted_conv_state = conv_state[active_indices].clone()
    accepted_ssm_state = ssm_state[active_indices].clone()
    manager.capture({"mtp.layers.0.mixer": metadata})

    for _ in range(11):
        conv_state.index_add_(0, active_indices, torch.ones(2, 3, dtype=torch.int64))
        ssm_state.index_add_(
            0, active_indices, torch.ones(2, 2, 3, dtype=torch.int64)
        )
    manager.restore()

    assert torch.equal(conv_state[active_indices], accepted_conv_state)
    assert torch.equal(ssm_state[active_indices], accepted_ssm_state)

    conv_state.index_add_(
        0, active_indices, torch.full((2, 3), 7, dtype=torch.int64)
    )
    ssm_state.index_add_(
        0, active_indices, torch.full((2, 2, 3), 7, dtype=torch.int64)
    )
    next_accepted_conv_state = conv_state[active_indices].clone()
    next_accepted_ssm_state = ssm_state[active_indices].clone()
    manager.capture({"mtp.layers.0.mixer": metadata})

    conv_state[active_indices] = -1
    ssm_state[active_indices] = -1
    manager.restore()

    assert torch.equal(conv_state[active_indices], next_accepted_conv_state)
    assert torch.equal(ssm_state[active_indices], next_accepted_ssm_state)


def test_compact_recurrent_mtp_inputs_removes_rejected_suffix():
    common_attn_metadata = create_common_attn_metadata(
        BatchSpec(seq_lens=[13], query_lens=[4]),
        block_size=16,
        device=torch.device("cpu"),
    )
    common_attn_metadata._seq_lens_cpu = torch.tensor([13], dtype=torch.int32)
    common_attn_metadata._num_computed_tokens_cpu = torch.tensor([9], dtype=torch.int32)
    common_attn_metadata.seq_lens_cpu_upper_bound = torch.tensor(
        [13], dtype=torch.int32
    )
    target_token_ids = torch.arange(4)
    target_positions = torch.arange(4)
    target_hidden_states = torch.arange(8).reshape(4, 2)
    token_indices_to_sample = torch.tensor([1], dtype=torch.int32)

    (
        target_token_ids,
        target_positions,
        target_hidden_states,
        token_indices_to_sample,
        common_attn_metadata,
    ) = compact_recurrent_mtp_inputs(
        target_token_ids,
        target_positions,
        target_hidden_states,
        token_indices_to_sample,
        common_attn_metadata,
        num_rejected_tokens_gpu=torch.tensor([2], dtype=torch.int32),
    )

    assert torch.equal(target_token_ids, torch.tensor([0, 1]))
    assert torch.equal(target_positions, torch.tensor([0, 1]))
    assert torch.equal(target_hidden_states, torch.tensor([[0, 1], [2, 3]]))
    assert torch.equal(token_indices_to_sample, torch.tensor([1], dtype=torch.int32))
    assert torch.equal(
        common_attn_metadata.query_start_loc,
        torch.tensor([0, 2], dtype=torch.int32),
    )
    assert torch.equal(common_attn_metadata.seq_lens, torch.tensor([11]))
    assert common_attn_metadata.num_actual_tokens == 2
    assert common_attn_metadata.max_query_len == 2
    assert common_attn_metadata.slot_mapping.shape == (2,)
    assert torch.equal(
        common_attn_metadata.compute_num_computed_tokens(), torch.tensor([9])
    )


@pytest.mark.parametrize("num_rejected", range(12))
def test_compact_recurrent_mtp_inputs_covers_all_acceptance_lengths(num_rejected):
    query_len = 12
    seq_len = 100
    common_attn_metadata = create_common_attn_metadata(
        BatchSpec(seq_lens=[seq_len], query_lens=[query_len]),
        block_size=16,
        device=torch.device("cpu"),
    )
    target_token_ids = torch.arange(query_len)
    target_positions = torch.arange(query_len)
    target_hidden_states = torch.arange(query_len * 2).reshape(query_len, 2)
    token_indices_to_sample = torch.tensor([query_len - 1], dtype=torch.int32)

    (
        target_token_ids,
        target_positions,
        target_hidden_states,
        token_indices_to_sample,
        common_attn_metadata,
    ) = compact_recurrent_mtp_inputs(
        target_token_ids,
        target_positions,
        target_hidden_states,
        token_indices_to_sample,
        common_attn_metadata,
        num_rejected_tokens_gpu=torch.tensor([num_rejected], dtype=torch.int32),
    )

    retained_query_len = query_len - num_rejected
    assert torch.equal(target_token_ids, torch.arange(retained_query_len))
    assert torch.equal(target_positions, torch.arange(retained_query_len))
    assert target_hidden_states.shape == (retained_query_len, 2)
    assert torch.equal(
        token_indices_to_sample,
        torch.tensor([retained_query_len - 1], dtype=torch.int32),
    )
    assert torch.equal(
        common_attn_metadata.query_start_loc,
        torch.tensor([0, retained_query_len], dtype=torch.int32),
    )
    assert torch.equal(
        common_attn_metadata.seq_lens,
        torch.tensor([seq_len - num_rejected], dtype=torch.int32),
    )
    assert torch.equal(
        common_attn_metadata.compute_num_computed_tokens(),
        torch.tensor([seq_len - query_len], dtype=torch.int32),
    )


def test_compact_recurrent_mtp_inputs_handles_variable_acceptance():
    common_attn_metadata = create_common_attn_metadata(
        BatchSpec(seq_lens=[13, 22, 34], query_lens=[4, 3, 5]),
        block_size=16,
        device=torch.device("cpu"),
        arange_block_indices=True,
    )
    target_token_ids = torch.arange(12)
    target_positions = torch.arange(24).reshape(2, 12)
    target_hidden_states = torch.arange(24).reshape(12, 2)
    token_indices_to_sample = torch.tensor([3, 6, 11], dtype=torch.int32)
    common_attn_metadata.positions = target_positions.clone()

    (
        target_token_ids,
        target_positions,
        target_hidden_states,
        token_indices_to_sample,
        common_attn_metadata,
    ) = compact_recurrent_mtp_inputs(
        target_token_ids,
        target_positions,
        target_hidden_states,
        token_indices_to_sample,
        common_attn_metadata,
        num_rejected_tokens_gpu=torch.tensor([2, 0, 4], dtype=torch.int32),
    )

    retained_indices = torch.tensor([0, 1, 4, 5, 6, 7])
    assert torch.equal(target_token_ids, retained_indices)
    assert torch.equal(
        target_positions, torch.arange(24).reshape(2, 12)[:, retained_indices]
    )
    assert torch.equal(
        target_hidden_states,
        torch.arange(24).reshape(12, 2)[retained_indices],
    )
    assert torch.equal(
        token_indices_to_sample, torch.tensor([1, 4, 5], dtype=torch.int32)
    )
    assert torch.equal(
        common_attn_metadata.query_start_loc,
        torch.tensor([0, 2, 5, 6], dtype=torch.int32),
    )
    assert torch.equal(
        common_attn_metadata.seq_lens,
        torch.tensor([11, 22, 30], dtype=torch.int32),
    )
    assert common_attn_metadata.num_actual_tokens == 6
    assert common_attn_metadata.max_query_len == 3
    assert common_attn_metadata.max_seq_len == 30
    assert torch.equal(common_attn_metadata.slot_mapping, retained_indices)
    assert torch.equal(common_attn_metadata.positions, target_positions)
    assert torch.equal(
        common_attn_metadata.compute_num_computed_tokens(),
        torch.tensor([9, 19, 29], dtype=torch.int32),
    )


def test_recurrent_mtp_pairs_retained_hidden_states_with_successor_tokens():
    common_attn_metadata = create_common_attn_metadata(
        BatchSpec(seq_lens=[13, 22], query_lens=[4, 3]),
        block_size=16,
        device=torch.device("cpu"),
        arange_block_indices=True,
    )
    target_token_ids = torch.tensor([10, 11, 12, 13, 20, 21, 22])
    target_positions = torch.tensor([9, 10, 11, 12, 19, 20, 21])
    target_hidden_states = torch.arange(14).reshape(7, 2)
    token_indices_to_sample = torch.tensor([3, 6], dtype=torch.int32)

    (
        target_token_ids,
        target_positions,
        target_hidden_states,
        token_indices_to_sample,
        common_attn_metadata,
    ) = compact_recurrent_mtp_inputs(
        target_token_ids,
        target_positions,
        target_hidden_states,
        token_indices_to_sample,
        common_attn_metadata,
        num_rejected_tokens_gpu=torch.tensor([2, 0], dtype=torch.int32),
    )

    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.use_heterogeneous_vocab = False
    proposer.needs_extra_input_slots = False
    proposer.uses_mrope = False
    proposer.uses_xdrope_dim = 0
    proposer.draft_uses_xdrope_dim = 0
    proposer.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(uses_mrope=False)
    )
    proposer.input_ids = torch.empty(5, dtype=torch.int64)
    proposer.positions = torch.empty(5, dtype=torch.int64)
    proposer.hidden_states = torch.empty(5, 2, dtype=torch.int64)

    num_tokens, token_indices_to_sample, _ = proposer.set_inputs_first_pass(
        target_token_ids=target_token_ids,
        next_token_ids=torch.tensor([14, 23]),
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        token_indices_to_sample=token_indices_to_sample,
        cad=common_attn_metadata,
        num_rejected_tokens_gpu=None,
    )

    assert num_tokens == 5
    assert torch.equal(proposer.input_ids, torch.tensor([11, 14, 21, 22, 23]))
    assert torch.equal(proposer.positions, torch.tensor([9, 10, 19, 20, 21]))
    assert torch.equal(
        proposer.hidden_states,
        torch.tensor([[0, 1], [2, 3], [8, 9], [10, 11], [12, 13]]),
    )
    assert torch.equal(token_indices_to_sample, torch.tensor([1, 4]))


@mock.patch("vllm.v1.spec_decode.llm_base_proposer.get_pp_group")
@mock.patch("vllm.v1.spec_decode.llm_base_proposer.get_layers_from_vllm_config")
@mock.patch("vllm.v1.spec_decode.llm_base_proposer.get_model")
@pytest.mark.parametrize("requires_recurrent_rollback", [False, True])
def test_mtp_load_model_unified(
    mock_get_model,
    mock_get_layers,
    mock_get_pp_group,
    requires_recurrent_rollback,
):
    """Test MTP-specific model loading with unified model approach."""

    # Setup mocks
    mock_model = mock.MagicMock()
    mock_model.model.embed_tokens.weight.shape = (131072, 4096)
    mock_get_model.return_value = mock_model
    # MTP does not have its own embed_tokens or lm_head
    # so it should share them with the target model
    mock_model.has_own_embed_tokens = False
    mock_model.has_own_lm_head = False
    mock_model.requires_recurrent_draft_state_rollback = (
        requires_recurrent_rollback
    )
    mock_model.get_recurrent_state_caches.return_value = {}

    target_attn_layers = {"target_attn_1": mock.MagicMock()}
    all_attn_layers = {**target_attn_layers, "draft_attn_1": mock.MagicMock()}
    target_indexer_layers: dict = {}
    all_indexer_layers: dict = {}

    mock_get_layers.side_effect = [
        target_attn_layers,
        target_indexer_layers,
        all_attn_layers,
        all_indexer_layers,
    ]

    mock_pp_group = mock.MagicMock()
    mock_pp_group.world_size = 1
    mock_get_pp_group.return_value = mock_pp_group

    # Create target model
    class _TargetModelStub(LlamaForCausalLM):
        model: mock.MagicMock
        lm_head: mock.MagicMock

    target_model = mock.create_autospec(_TargetModelStub, instance=True)
    target_model.model = mock.MagicMock()
    target_model.model.embed_tokens.weight.shape = (131072, 4096)
    target_model.lm_head = mock.MagicMock()

    # Create MTP proposer
    proposer = _create_mtp_proposer(num_speculative_tokens=4)
    proposer.vllm_config.model_config.enforce_eager = requires_recurrent_rollback
    proposer.load_model(target_model)

    # Verify MTP-specific behavior:
    # Model is loaded
    mock_get_model.assert_called_once()
    # MTP shares lm_head with target model
    assert proposer.model.lm_head == target_model.lm_head
    # MTP shares embed_tokens with target model
    assert proposer.model.model.embed_tokens == target_model.model.embed_tokens
    assert (proposer.recurrent_draft_state_manager is not None) is (
        requires_recurrent_rollback
    )


@pytest.mark.parametrize("num_speculative_tokens", [1, 4])
def test_mtp_propose(num_speculative_tokens, monkeypatch):
    """Test that MTP's forward method returns hidden states directly"""

    device = torch.device(DEVICE_TYPE)
    batch_size = 2
    seq_lens = [5, 3]
    total_tokens = sum(seq_lens)
    vocab_size = 100

    proposer = _create_mtp_proposer(num_speculative_tokens)
    hidden_size = proposer.hidden_size

    # Mock the MTP model to verify it returns hidden states directly
    model_mock = mock.MagicMock()

    # MTP returns hidden states directly
    if num_speculative_tokens == 1:
        model_mock.return_value = torch.zeros(total_tokens, hidden_size, device=device)
    else:
        # Multiple forward passes for multi-token speculation
        forward_returns = []
        for i in range(num_speculative_tokens):
            if i == 0:
                h_states = torch.zeros(total_tokens, hidden_size, device=device)
            else:
                h_states = torch.zeros(batch_size, hidden_size, device=device)
            forward_returns.append(h_states)
        model_mock.side_effect = forward_returns

    # Mock compute_logits
    def create_deterministic_logits(batch_size, vocab_size, token_offset):
        logits = torch.full((batch_size, vocab_size), -100.0, device=device)
        logits[:, token_offset] = 100.0
        return logits

    if num_speculative_tokens == 1:
        model_mock.compute_logits.return_value = create_deterministic_logits(
            batch_size, vocab_size, 42
        )
    else:
        logits_returns = [
            create_deterministic_logits(batch_size, vocab_size, 42 + i)
            for i in range(num_speculative_tokens)
        ]
        model_mock.compute_logits.side_effect = logits_returns

    proposer.model = model_mock
    proposer._draft_attn_layer_names = {"layer.0"}

    # Prepare inputs
    batch_spec = BatchSpec(seq_lens=seq_lens, query_lens=seq_lens)
    common_attn_metadata = create_common_attn_metadata(
        batch_spec, block_size=16, device=device
    )

    target_token_ids = torch.randint(0, vocab_size, (total_tokens,), device=device)
    target_positions = torch.cat(
        [
            torch.arange(seq_lens[0], device=device),
            torch.arange(seq_lens[1], device=device),
        ]
    )
    target_hidden_states = torch.randn(total_tokens, hidden_size, device=device)
    next_token_ids = torch.randint(
        0, vocab_size, (batch_size,), dtype=torch.int32, device=device
    )
    sampling_metadata = mock.MagicMock()

    # Setup attention metadata
    attn_metadata_builder_cls, _ = try_get_attention_backend(
        AttentionBackendEnum.FLASH_ATTN
    )

    attn_metadata_builder = attn_metadata_builder_cls(
        kv_cache_spec=create_standard_kv_cache_spec(proposer.vllm_config),
        layer_names=list(proposer._draft_attn_layer_names),
        vllm_config=proposer.vllm_config,
        device=device,
    )
    proposer.block_size = attn_metadata_builder.kv_cache_spec.block_size

    proposer.runner = mock.MagicMock()
    mock_attn_group = mock.MagicMock()
    mock_attn_group.get_metadata_builder.return_value = attn_metadata_builder
    mock_attn_group.layer_names = list(proposer._draft_attn_layer_names)
    mock_attn_group.kv_cache_spec = attn_metadata_builder.kv_cache_spec
    proposer.draft_attn_groups = [mock_attn_group]

    # Run propose
    result = proposer.propose(
        num_speculative_tokens=num_speculative_tokens,
        target_token_ids=target_token_ids,
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        next_token_ids=next_token_ids,
        token_indices_to_sample=None,
        common_attn_metadata=common_attn_metadata,
        sampling_metadata=sampling_metadata,
    )

    # Verify the model was called correctly
    assert model_mock.called
    # Verify output shape
    assert result.shape == (batch_size, num_speculative_tokens)
    expected = torch.arange(
        42,
        42 + num_speculative_tokens,
        device=device,
    ).expand(batch_size, -1)
    assert torch.equal(result, expected)
