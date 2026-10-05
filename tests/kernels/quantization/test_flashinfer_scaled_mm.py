# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
from vllm.model_executor.layers.mamba.gdn.custom_input_projection import (
    gdn_custom_fused_input_gemm,
)
from vllm.model_executor.layers.mamba.gdn.input_projection import gdn_input_gemms
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.flashinfer import flashinfer_scaled_fp8_mm
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Flashinfer FP8 gemms requires compute capability of 10.0 or above.",
        allow_module_level=True,
    )

DTYPES = [torch.float16, torch.bfloat16]
# m, n, k
SHAPES = [(128, 128, 64), (128, 128, 128), (256, 128, 64), (128, 256, 128)]
PAD_SHAPES = [(150, 128, 64), (128, 128, 96)]
SHAPES.extend(PAD_SHAPES)

SEEDS = [42]
CUDA_DEVICES = ["cuda:0"]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("autotune", [False, True])
@torch.inference_mode()
def test_flashinfer_fp8_gemm(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
    use_bias: bool,
    seed: int,
    device: str,
    autotune: bool,
) -> None:
    set_random_seed(seed)
    m, n, k = shape
    a = torch.randn((m, k), dtype=dtype, device=device)
    b = torch.randn((n, k), dtype=dtype, device=device) / k

    a_fp8, a_scale = ops.scaled_fp8_quant(a)
    b_fp8, b_scale = ops.scaled_fp8_quant(b)

    expected_out = torch.mm(
        a_scale * a_fp8.to(dtype=torch.float32),
        b_scale * b_fp8.to(dtype=torch.float32).t(),
    ).to(dtype=dtype)

    if use_bias:
        bias = torch.randn((n,), dtype=dtype, device=device)
        expected_out = expected_out + bias
    else:
        bias = None

    import flashinfer

    with flashinfer.autotune(autotune):
        out = flashinfer_scaled_fp8_mm(
            a_fp8,
            b_fp8.t(),
            a_scale,
            b_scale,
            dtype,
            bias=bias,
        )

    torch.testing.assert_close(out, expected_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "m, breakable",
    [(m, False) for m in [1, 2, 4, 8, 16, 32, 64, 128, 256]] + [(8, True)],
)
@pytest.mark.parametrize("scalar_shape", [(), (1,)])
@torch.inference_mode()
def test_gdn_concurrent_fp8_pair_graph_replay(m, breakable, scalar_shape):
    """Check scratch reuse and replay with concurrent and breakable captures."""
    device = "cuda:0"
    set_random_seed(42)
    x = torch.randn(m, 512, device=device).to(torch.float8_e4m3fn)
    weights = [
        torch.randn(n, 512, device=device).to(torch.float8_e4m3fn).t()
        for n in (256, 32)
    ]
    scales = [
        torch.tensor(s, device=device).reshape(scalar_shape)
        for s in (0.101, 0.271, 0.183)
    ]
    args = (x, *weights, *scales)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        gdn_input_gemms(*args)
    torch.accelerator.synchronize()

    graph = BreakableCUDAGraphCapture() if breakable else torch.cuda.CUDAGraph()
    capture = graph if breakable else torch.cuda.graph(graph, stream=stream)
    with torch.cuda.stream(stream), capture:
        first = gdn_input_gemms(*args)
        second = gdn_input_gemms(*args)

    for factor in (1.0, -0.75, 2.0):
        x.copy_((torch.randn_like(x, dtype=torch.float32) * factor).to(x.dtype))
        scales[0].fill_(0.101 * abs(factor))
        for output in (*first, *second):
            output.fill_(float("nan"))
        graph.replay()
        for outputs in (first, second):
            for output, weight, weight_scale in zip(outputs, weights, scales[1:]):
                expected = (x.float() @ weight.float()) * scales[0] * weight_scale
                assert output.is_contiguous()
                assert output.dtype == torch.bfloat16
                torch.testing.assert_close(
                    output, expected.to(output.dtype), atol=1e-2, rtol=1e-2
                )

    if m == 8:
        snapshots = [tensor.clone() for tensor in args]
        # SchemaCheckMode's allclose does not support FP8 on CUDA. Check input
        # immutability byte-for-byte and exercise FakeTensor separately.
        torch.library.opcheck(gdn_input_gemms, args, test_utils=("test_faketensor",))
        compiled = torch.compile(gdn_input_gemms, backend="aot_eager", fullgraph=True)
        for actual, expected in zip(compiled(*args), gdn_input_gemms(*args)):
            torch.testing.assert_close(actual, expected)
        for actual, expected in zip(args, snapshots):
            torch.testing.assert_close(
                actual.reshape(-1).view(torch.uint8),
                expected.reshape(-1).view(torch.uint8),
                atol=0,
                rtol=0,
            )


@triton.jit
def delayed_copy(src, dst, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    # Signal before the output exists. A consumer reading before its PDL wait
    # will see the NaNs used to poison dst before each replay.
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    tl.inline_asm_elementwise(
        "{.reg .u64 start, now, elapsed; .reg .pred done; "
        "mov.u64 start, %clock64; L: mov.u64 now, %clock64; "
        "sub.u64 elapsed, now, start; setp.ge.u64 done, elapsed, 100000; "
        "@!done bra L; mov.u32 $0, 0;}",
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    src = src.to(tl.pointer_type(tl.uint8))
    dst = dst.to(tl.pointer_type(tl.uint8))
    tl.store(dst + index, tl.load(src + index, index < COUNT, other=0), index < COUNT)


@triton.jit
def dependent_copy(src, dst, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    tl.extra.cuda.gdc_wait()
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(dst + index, tl.load(src + index, index < COUNT, other=0), index < COUNT)
    tl.extra.cuda.gdc_launch_dependents()


@pytest.fixture(scope="module")
def paired_weights():
    if not current_platform.is_device_capability(107) or not hasattr(
        torch.ops._qwen_gdn_custom, "gemms_out"
    ):
        pytest.skip("The reference paired GEMM requires an SM107 native build")
    set_random_seed(13)
    return [
        torch.randn(n, 8192, device="cuda").to(torch.float8_e4m3fn).t()
        for n in (4608, 32)
    ]


@pytest.mark.parametrize(
    "m,breakable", [(m, False) for m in (1, 2, 3, 4, 7, 8, 9, 15, 16)] + [(8, True)]
)
@pytest.mark.parametrize("scalar_shape", [(), (1,)])
@torch.inference_mode()
def test_gdn_fused_pdl_graph_replay(m, breakable, scalar_shape, paired_weights):
    source = torch.randn(m, 8192, device="cuda").to(torch.float8_e4m3fn)
    x = torch.empty_like(source)
    scales = [
        torch.tensor(s, device="cuda").reshape(scalar_shape)
        for s in (0.101, 0.271, 0.183)
    ]
    args = (x, *paired_weights, *scales)
    observed = [
        torch.empty(m, n, device="cuda", dtype=torch.bfloat16) for n in (4608, 32)
    ]

    def pipeline():
        delayed_copy[(triton.cdiv(x.numel(), 1024),)](
            source, x, x.numel(), 1024, launch_pdl=True
        )
        result = gdn_custom_fused_input_gemm(*args)  # Default must be mode 2.
        for output, dest in zip(result, observed):
            dependent_copy[(triton.cdiv(output.numel(), 1024),)](
                output, dest, output.numel(), 1024, launch_pdl=True
            )
        return result

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        pipeline()
    torch.accelerator.synchronize()
    graph = BreakableCUDAGraphCapture() if breakable else torch.cuda.CUDAGraph()
    capture = graph if breakable else torch.cuda.graph(graph, stream=stream)
    with torch.cuda.stream(stream), capture:
        result = pipeline()

    for factor in (1.0, -0.75, 2.0):
        source.copy_(
            (torch.randn_like(source, dtype=torch.float32) * factor).to(source.dtype)
        )
        x.fill_(float("nan"))
        for output in (*result, *observed):
            output.fill_(float("nan"))
        for scale, value in zip(scales, (0.101, 0.271, 0.183)):
            scale.fill_(value * abs(factor))
        graph.replay()
        for actual, raw, weight, weight_scale in zip(
            observed, result, paired_weights, scales[1:]
        ):
            expected = (
                (source.float() @ weight.float()) * scales[0] * weight_scale
            ).bfloat16()
            assert actual.is_contiguous() and actual.dtype == torch.bfloat16
            assert torch.isfinite(actual).all()
            relative_l2 = (
                actual.float() - expected.float()
            ).norm() / expected.float().norm()
            assert relative_l2 < 0.004, relative_l2
            torch.testing.assert_close(actual, raw, rtol=0, atol=0)

    if m == 8 and not breakable:
        snapshots = [a.clone() for a in args]
        torch.library.opcheck(
            gdn_custom_fused_input_gemm, args, test_utils=("test_faketensor",)
        )
        compiled = torch.compile(
            gdn_custom_fused_input_gemm, backend="aot_eager", fullgraph=True
        )
        for actual, expected in zip(compiled(*args), result):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for a, b in zip(args, snapshots):
            torch.testing.assert_close(
                a.contiguous().reshape(-1).view(torch.uint8),
                b.contiguous().reshape(-1).view(torch.uint8),
                rtol=0,
                atol=0,
            )


def make_layer(paired_weights, custom, concurrent):
    class Quant(torch.nn.Module):
        num_token_padding = None

        def forward(self, x, scale):
            return ops.scaled_fp8_quant(x, scale)

    projections = []
    for weight, ws in zip(paired_weights, (0.271, 0.183)):
        layer = ColumnParallelLinear.__new__(ColumnParallelLinear)
        torch.nn.Module.__init__(layer)
        kernel = CutlassFP8ScaledMMLinearKernel.__new__(CutlassFP8ScaledMMLinearKernel)
        kernel.config = FP8ScaledMMLinearLayerConfig(
            kFp8StaticTensorSym,
            kFp8StaticTensorSym,
            tuple(weight.shape),
            torch.bfloat16,
            torch.bfloat16,
        )
        kernel.quant_fp8 = Quant()
        layer.quant_method = SimpleNamespace(fp8_linear=kernel)
        layer.input_scale = torch.tensor([0.101], device="cuda")
        layer.weight_scale = torch.tensor([ws], device="cuda")
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.bias = None
        layer.gather_output = False
        layer.output_size_per_partition = weight.shape[1]
        # The fallback still receives the shared QuantizedActivation.
        layer.forward = lambda x: (x, None)
        projections.append(layer)
    return SimpleNamespace(
        prefix="test.linear_attn",
        in_proj_qkvz=projections[0],
        in_proj_ba=projections[1],
        hidden_size=8192,
        _allow_shared_input_quant=True,
        _custom_fused_input_gemm_requested=custom,
        _concurrent_input_gemm_requested=concurrent,
    )


@pytest.mark.parametrize(
    "custom,concurrent", [(False, False), (False, True), (True, False), (True, True)]
)
def test_gdn_model_dispatch_and_reload(custom, concurrent, paired_weights):
    layer = make_layer(paired_weights, custom, concurrent)
    gdn.QwenGatedDeltaNetAttention.process_weights_after_loading(layer, torch.bfloat16)
    assert layer._custom_fused_input_gemm == custom
    assert layer._concurrent_input_gemm == concurrent
    with (
        patch.object(
            gdn, "gdn_dispatch_input_gemms", return_value=("dispatch", "dispatch")
        ) as fused,
        patch.object(
            gdn, "gdn_input_gemms", return_value=("library", "library")
        ) as library,
    ):
        for m in (1, 8, 16, 17, 32, 256):
            fused.reset_mock()
            library.reset_mock()
            gdn.QwenGatedDeltaNetAttention._input_projections(
                layer, torch.randn(m, 8192, device="cuda", dtype=torch.bfloat16)
            )
            assert fused.called == custom  # M selection is inside the opaque op.
            assert library.called == (concurrent and not custom)
            if fused.called:
                assert len(fused.call_args.args) == 7
                assert fused.call_args.args[-1] == layer.prefix
    layer.in_proj_ba.input_scale.fill_(0.25)
    gdn.QwenGatedDeltaNetAttention.process_weights_after_loading(layer, torch.bfloat16)
    assert layer._shared_input_quant is None
    assert not layer._custom_fused_input_gemm and not layer._concurrent_input_gemm


def test_gdn_custom_rejects_other_shapes_and_dtypes(paired_weights):
    layer = make_layer(paired_weights, True, False)
    for attribute, value in (("hidden_size", 4096),):
        setattr(layer, attribute, value)
    gdn.QwenGatedDeltaNetAttention.process_weights_after_loading(layer, torch.bfloat16)
    assert not layer._custom_fused_input_gemm
    layer.hidden_size = 8192
    gdn.QwenGatedDeltaNetAttention.process_weights_after_loading(layer, torch.float16)
    assert not layer._custom_fused_input_gemm
    with pytest.raises(RuntimeError, match="contiguous FP8"):
        gdn_custom_fused_input_gemm(
            torch.empty(17, 8192, device="cuda", dtype=torch.float8_e4m3fn),
            *paired_weights,
            *[torch.ones(1, device="cuda") for _ in range(3)],
        )


@pytest.mark.parametrize("concurrent", [False, True])
@pytest.mark.parametrize("breakable", [False, True])
@torch.inference_mode()
def test_gdn_compiled_model_dispatch_cuda_replay(paired_weights, concurrent, breakable):
    """Trace at M256, then capture/replay that same FX graph at small and large M."""
    from vllm.forward_context import override_forward_context
    from vllm.model_executor.layers.mamba.gdn import custom_input_projection as custom
    from vllm.model_executor.layers.mamba.gdn import input_projection as library

    layer = make_layer(paired_weights, True, concurrent)
    gdn.QwenGatedDeltaNetAttention.process_weights_after_loading(layer, torch.bfloat16)
    for linear in (layer.in_proj_qkvz, layer.in_proj_ba):
        kernel = linear.quant_method.fp8_linear
        kernel.layer_param_names = (
            "weight",
            "weight_scale",
            "input_scale",
            "input_scale_ub",
        )
        kernel.fp8_dtype = torch.float8_e4m3fn
        kernel.process_weights_after_loading(linear)
        linear.forward = lambda qa, linear=linear: (
            linear.quant_method.fp8_linear.apply_weights(linear, qa),
            None,
        )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for key, value in vars(layer).items():
                setattr(self, key, value)

        def forward(self, x):
            return gdn.QwenGatedDeltaNetAttention._input_projections(self, x)

    captured = {}

    def backend(graph, examples):
        captured["graph"] = graph

        def run(*args):
            captured["args"] = args
            return graph(*args)

        return run

    model = Model()
    context = SimpleNamespace(no_compile_layers={model.prefix: model})
    large = torch.randn(256, 8192, device="cuda", dtype=torch.bfloat16)
    with override_forward_context(context):
        torch.compile(model, backend=backend, dynamic=True, fullgraph=True)(large)
        graph, args = captured["graph"], captured["args"]
        assert "gdn_dispatch_input_gemms" in graph.code
        assert any(a is large for a in args)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        for m in (1, 8, 16, 17, 32, 256):
            x = torch.randn(m, 8192, device="cuda", dtype=torch.bfloat16)
            resized = [
                x if a is large else m if type(a) is int and a == 256 else a
                for a in args
            ]
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    graph(*resized)
            torch.accelerator.synchronize()
            capture = (
                BreakableCUDAGraphCapture() if breakable else torch.cuda.CUDAGraph()
            )
            scope = capture if breakable else torch.cuda.graph(capture, stream=stream)
            with (
                patch.object(
                    custom,
                    "gdn_custom_fused_input_gemm",
                    wraps=custom.gdn_custom_fused_input_gemm,
                ) as fused,
                patch.object(
                    library, "gdn_input_gemms", wraps=library.gdn_input_gemms
                ) as pair,
            ):
                with torch.cuda.stream(stream), scope:
                    outputs = graph(*resized)
                assert fused.called == (m <= 16)
                assert pair.called == (concurrent and m > 16)
            for factor in (1.0, -0.75):
                x.copy_(torch.randn_like(x) * factor)
                for output in outputs:
                    output.fill_(float("nan"))
                capture.replay()
                data, scale = layer._shared_input_quant(
                    x, layer.in_proj_qkvz.input_scale
                )
                for actual, linear in zip(
                    outputs, (layer.in_proj_qkvz, layer.in_proj_ba)
                ):
                    expected = (
                        (data.float() @ linear.weight.float())
                        * scale
                        * linear.weight_scale
                    )
                    assert torch.isfinite(actual).all()
                    relative_l2 = (actual.float() - expected).norm() / expected.norm()
                    assert relative_l2 < 0.004, relative_l2
    torch._dynamo.reset()
