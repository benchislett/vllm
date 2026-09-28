# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real NvFP4 MoE producer / CuTe consumer tests on eight SM107 GPUs."""

import os

import pytest
import torch

import vllm.ir
from tests.compile.backend import TestBackend
from tests.compile.passes.distributed.test_fusion_all_reduce import (
    cute_tp_environment,  # noqa: F401
)
from vllm import _custom_ops as ops
from vllm.compilation.passes.fusion.cute_allreduce_fusion import CuteAllReduceFusionPass
from vllm.compilation.passes.fusion.qwen_cute_moe_tail import QwenCuteMoETailFusionPass
from vllm.compilation.passes.utility.fix_functionalization import (
    FixFunctionalizationPass,
)
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import DeviceConfig, ModelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (
    ensure_model_parallel_initialized,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.device_communicators import cute_allreduce
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
    TrtLlmNvFp4ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.qwen_cute_moe_tail import qwen_cute_moe_tail
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeTextConfig


@pytest.mark.skip_global_cleanup
@pytest.mark.usefixtures("cute_tp_environment")
@pytest.mark.parametrize("expert_parallel", [False, True])
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("add_residual", [False, True])
@torch.inference_mode()
def test_qwen_cute_real_moe_graph_replay(
    expert_parallel,
    quantized,
    add_residual,
    tmp_path,
    workspace_init,
    monkeypatch,
):
    import flashinfer.comm
    from flashinfer.comm import AllReduceFusionPattern

    finalize_calls = []
    original_allreduce_fusion = flashinfer.comm.allreduce_fusion

    def record_finalize(**kwargs):
        if kwargs["pattern"] == AllReduceFusionPattern.kMoEFinalizeARResidualRMSNorm:
            finalize_calls.append(kwargs["input"].shape)
        return original_allreduce_fusion(**kwargs)

    monkeypatch.setattr(flashinfer.comm, "allreduce_fusion", record_finalize)
    local_rank = int(os.environ["LOCAL_RANK"])
    config = VllmConfig()
    config.parallel_config.tensor_parallel_size = 8
    config.parallel_config.enable_expert_parallel = expert_parallel
    config.kernel_config.enable_cute_allreduce = True
    config.kernel_config.enable_cute_moe_finalize = True
    config.kernel_config.moe_backend = "flashinfer_trtllm"
    config.compilation_config.custom_ops = ["all", "+rms_norm", "+quant_fp8"]
    config.compilation_config.pass_config.fuse_allreduce_rms = True
    model_path = tmp_path / "shape-config"
    Qwen3_5MoeTextConfig(
        hidden_size=8192,
        num_experts_per_tok=10,
        architectures=["Qwen3_5MoeForCausalLM"],
    ).save_pretrained(model_path)
    config.model_config = ModelConfig(
        model=str(model_path),
        dtype=torch.bfloat16,
        skip_tokenizer_init=True,
    )
    device = torch.device("cuda", local_rank)
    config.device_config = DeviceConfig(device=device)
    with set_current_vllm_config(config, check_compile=False), torch.device(device):
        ensure_model_parallel_initialized(8, 1)
        cute_allreduce.initialize_for_config(config)
        torch.manual_seed(77 + int(os.environ["RANK"]))
        shared = torch.nn.Linear(8192, 8192, bias=False, dtype=torch.bfloat16)
        shared.weight.normal_(std=0.001)
        layer = FusedMoEFactory(
            num_experts=32,
            top_k=10,
            hidden_size=8192,
            intermediate_size=2048,
            params_dtype=torch.bfloat16,
            prefix="model.layers.0.mlp.experts",
            reduce_results=False,
            shared_experts=shared,
            quant_config=ModelOptNvFp4Config(is_checkpoint_nvfp4_serialized=True),
        )
        experts = layer.routed_experts
        for name, parameter in experts.named_parameters():
            if parameter.dtype == torch.uint8:
                parameter.random_(0, 256)
            elif "weight_scale" in name:
                parameter.fill_(
                    0.015625 if parameter.dtype == torch.float8_e4m3fn else 1.0
                )
            elif "input_scale" in name:
                parameter.fill_(1.0)
            else:
                parameter.zero_()
        experts.quant_method.process_weights_after_loading(experts)
        assert isinstance(
            experts.quant_method.moe_kernel.fused_experts,
            TrtLlmNvFp4ExpertsMonolithic,
        )
        ar_fusion = CuteAllReduceFusionPass(config)
        moe_fusion = QwenCuteMoETailFusionPass(config)
        backend = TestBackend(
            ar_fusion,
            moe_fusion,
            FixFunctionalizationPass(config),
            PostCleanupPass(config),
        )

        def reference_norm(x, logits, residual, weight):
            reduced = tensor_model_parallel_all_reduce(layer(x, logits))
            gamma = weight.float() + 1.0
            if add_residual:
                normalized, updated = vllm.ir.ops.fused_add_rms_norm(
                    reduced, residual.clone(), gamma, 1e-6
                )
            else:
                normalized = vllm.ir.ops.rms_norm(reduced, gamma, 1e-6)
                updated = reduced
            return normalized, updated

        def forward(x, logits, residual, weight, scale):
            normalized, updated = reference_norm(x, logits, residual, weight)
            output = (
                ops.scaled_fp8_quant(normalized, scale, group_shape=(-1, -1))[0]
                if quantized
                else normalized
            )
            return output, updated

        compiled = torch.compile(forward, backend=backend, fullgraph=True, dynamic=True)
        # The replicated residual, weights and routing logits agree on every rank.
        generator = torch.Generator(device=device).manual_seed(93)
        weight = torch.randn(8192, generator=generator, dtype=torch.bfloat16) * 0.1
        torch._dynamo.mark_static(weight, 0)
        scale = torch.tensor([0.1], dtype=torch.float32)
        for rows in (24, 25, 48, 49):
            x = torch.randn(rows, 8192, generator=generator, dtype=torch.bfloat16) * 0.1
            residual = torch.randn(
                rows, 8192, generator=generator, dtype=torch.bfloat16
            )
            logits = (
                -torch.arange(32, dtype=torch.float32).expand(rows, -1).contiguous()
            )
            torch._dynamo.mark_static(x, 1)
            torch._dynamo.mark_static(residual, 1)
            torch._dynamo.mark_static(logits, 1)
            with set_forward_context(None, config, num_tokens=rows):
                finalize_calls.clear()
                compiled(x, logits, residual, weight, scale)
                assert finalize_calls, "The compiled path must execute MoE finalization"
                assert ar_fusion.matched_count == moe_fusion.matched_count == 1
                assert backend.op_count(torch.ops.vllm.qwen_cute_moe_tail.default) == 1
                assert backend.op_count(torch.ops.vllm.moe_forward_shared.default) == 0
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = compiled(x, logits, residual, weight, scale)
                for _ in range(2):
                    x.normal_(std=0.1, generator=generator)
                    residual.normal_(generator=generator)
                    # Only ten adjacent experts receive tokens. Some EP ranks
                    # have no routed rows; shifting also detects stale maps.
                    logits.copy_(logits.roll(4, dims=-1))
                    expected = forward(x, logits, residual, weight, scale)
                    graph.replay()
                    torch.accelerator.synchronize()
                    if quantized:
                        norm, _ = qwen_cute_moe_tail(
                            x,
                            logits,
                            x,
                            None,
                            layer.layer_name,
                            0,
                            residual if add_residual else None,
                            weight,
                            None,
                            1e-6,
                        )
                        reference, _ = reference_norm(x, logits, residual, weight)
                        torch.testing.assert_close(
                            norm, reference, atol=0.06, rtol=0.03
                        )
                        quantized_norm = ops.scaled_fp8_quant(norm, scale)[0]
                        expected = (quantized_norm, expected[1])
                    for got, want in zip(actual, expected):
                        if got.dtype == torch.float8_e4m3fn:
                            torch.testing.assert_close(
                                got.float(), want.float(), atol=0, rtol=0
                            )
                        else:
                            torch.testing.assert_close(
                                got.float(), want.float(), atol=0.06, rtol=0.03
                            )
                    assert not layer.moe_config.use_deferred_moe_finalize
        communicator = get_tp_group().device_communicator
        communicator.cute_allreduce.destroy()
        communicator.cute_allreduce = None
