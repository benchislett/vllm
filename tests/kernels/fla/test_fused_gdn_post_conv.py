# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GDN post-convolution kernels.

The prefill tests cover preparation and the MTP tests cover recurrent state
updates plus output normalization and gating.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.third_party.flash_linear_attention.ops import (
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import (
    fused_post_conv_prep,
)
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (
    rmsnorm_fn,
)


def reference_post_conv(
    conv_output: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    H: int,
    K: int,
    V: int,
    apply_l2norm: bool = True,
    output_g_exp: bool = False,
):
    """Reference implementation using individual ops."""
    L = conv_output.shape[0]
    HV = A_log.shape[0]

    # Split
    q_flat, k_flat, v_flat = torch.split(conv_output, [H * K, H * K, HV * V], dim=-1)

    # Rearrange + contiguous
    q = q_flat.view(L, H, K).contiguous()
    k = k_flat.view(L, H, K).contiguous()
    v = v_flat.view(L, HV, V).contiguous()

    # L2 norm
    if apply_l2norm:
        q = F.normalize(q.float(), p=2, dim=-1, eps=1e-6).to(conv_output.dtype)
        k = F.normalize(k.float(), p=2, dim=-1, eps=1e-6).to(conv_output.dtype)

    # Gating
    x = a.float() + dt_bias.float()
    sp = F.softplus(x, beta=1.0, threshold=20.0)
    g = -torch.exp(A_log.float()) * sp

    if output_g_exp:
        g = torch.exp(g)

    beta_out = torch.sigmoid(b.float())

    return q, k, v, g, beta_out


# Qwen3.5-35B config: H=16, HV=32, K=128, V=128
# Qwen3.5-397B config: H=16, HV=64, K=128, V=128
@pytest.mark.parametrize(
    "H, HV, K, V",
    [
        (16, 32, 128, 128),  # 35B
        (16, 64, 128, 128),  # 397B
        (4, 8, 64, 64),  # small
    ],
)
@pytest.mark.parametrize("L", [1, 16, 128, 512, 2048])
@pytest.mark.parametrize("apply_l2norm", [True, False])
@pytest.mark.parametrize("output_g_exp", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_post_conv_correctness(H, HV, K, V, L, apply_l2norm, output_g_exp, dtype):
    """Test fused kernel matches reference for all configs."""
    torch.manual_seed(42)
    device = "cuda"
    qkv_dim = 2 * H * K + HV * V

    conv_output = torch.randn(L, qkv_dim, dtype=dtype, device=device)
    a = torch.randn(L, HV, dtype=dtype, device=device)
    b = torch.randn(L, HV, dtype=dtype, device=device)
    A_log = torch.randn(HV, dtype=torch.float32, device=device) - 2.0
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device) * 0.1

    # Reference
    ref_q, ref_k, ref_v, ref_g, ref_beta = reference_post_conv(
        conv_output,
        a,
        b,
        A_log,
        dt_bias,
        H,
        K,
        V,
        apply_l2norm,
        output_g_exp,
    )

    # Fused kernel
    fused_q, fused_k, fused_v, fused_g, fused_beta = fused_post_conv_prep(
        conv_output,
        a,
        b,
        A_log,
        dt_bias,
        num_k_heads=H,
        head_k_dim=K,
        head_v_dim=V,
        apply_l2norm=apply_l2norm,
        output_g_exp=output_g_exp,
    )

    # Check shapes
    assert fused_q.shape == (L, H, K), f"q shape: {fused_q.shape}"
    assert fused_k.shape == (L, H, K), f"k shape: {fused_k.shape}"
    assert fused_v.shape == (L, HV, V), f"v shape: {fused_v.shape}"
    assert fused_g.shape == (L, HV), f"g shape: {fused_g.shape}"
    assert fused_beta.shape == (L, HV), f"beta shape: {fused_beta.shape}"

    # Check dtypes
    assert fused_q.dtype == dtype
    assert fused_k.dtype == dtype
    assert fused_v.dtype == dtype
    assert fused_g.dtype == torch.float32
    assert fused_beta.dtype == torch.float32

    # Check contiguity
    assert fused_q.is_contiguous()
    assert fused_k.is_contiguous()
    assert fused_v.is_contiguous()

    # Check values
    atol_qkv = 1e-2 if apply_l2norm else 1e-3
    rtol_qkv = 1e-2 if apply_l2norm else 1e-3

    torch.testing.assert_close(fused_q, ref_q, atol=atol_qkv, rtol=rtol_qkv)
    torch.testing.assert_close(fused_k, ref_k, atol=atol_qkv, rtol=rtol_qkv)
    torch.testing.assert_close(fused_v, ref_v, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(fused_g, ref_g, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(fused_beta, ref_beta, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("L", [1, 64, 256])
def test_fused_post_conv_sanity(L):
    """Sanity checks: no NaN, unit-norm q/k, beta in (0,1)."""
    torch.manual_seed(0)
    device = "cuda"
    H, HV, K, V = 16, 32, 128, 128
    qkv_dim = 2 * H * K + HV * V

    conv_output = torch.randn(L, qkv_dim, dtype=torch.bfloat16, device=device)
    a = torch.randn(L, HV, dtype=torch.bfloat16, device=device)
    b = torch.randn(L, HV, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(HV, dtype=torch.float32, device=device) - 2.0
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device)

    q, k, v, g, beta = fused_post_conv_prep(
        conv_output,
        a,
        b,
        A_log,
        dt_bias,
        num_k_heads=H,
        head_k_dim=K,
        head_v_dim=V,
    )

    # Basic sanity
    assert not torch.isnan(q).any(), "NaN in q"
    assert not torch.isnan(k).any(), "NaN in k"
    assert not torch.isnan(v).any(), "NaN in v"
    assert not torch.isnan(g).any(), "NaN in g"
    assert not torch.isnan(beta).any(), "NaN in beta"

    # L2 norm check: each head vector should have unit norm
    q_norms = torch.norm(q.float(), dim=-1)
    k_norms = torch.norm(k.float(), dim=-1)
    torch.testing.assert_close(q_norms, torch.ones_like(q_norms), atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(k_norms, torch.ones_like(k_norms), atol=1e-3, rtol=1e-3)

    # Beta should be in (0, 1)
    assert (beta >= 0).all() and (beta <= 1).all(), "beta out of range"


def test_fused_post_conv_l0():
    """Test L=0 edge case."""
    device = "cuda"
    H, HV, K, V = 16, 32, 128, 128
    qkv_dim = 2 * H * K + HV * V

    conv_output = torch.empty(0, qkv_dim, dtype=torch.bfloat16, device=device)
    a = torch.empty(0, HV, dtype=torch.bfloat16, device=device)
    b = torch.empty(0, HV, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device)

    q, k, v, g, beta = fused_post_conv_prep(
        conv_output,
        a,
        b,
        A_log,
        dt_bias,
        num_k_heads=H,
        head_k_dim=K,
        head_v_dim=V,
    )
    assert q.shape == (0, H, K)
    assert g.shape == (0, HV)


@pytest.mark.parametrize(
    "head_ratio,tp_size,query_lengths,state_dtype,norm_dtype",
    [
        pytest.param(8, 16, (4, 4), torch.bfloat16, torch.bfloat16, id="tp16-bf16"),
        pytest.param(8, 4, (4, 4), torch.float32, torch.float32, id="tp4-fp32"),
        pytest.param(8, 16, (4, 2, 0), torch.bfloat16, torch.float32, id="tp16-ragged"),
        pytest.param(8, 4, (4, 2, 0), torch.float32, torch.bfloat16, id="tp4-ragged"),
        pytest.param(8, 16, (8,), torch.float32, torch.bfloat16, id="tp16-max"),
        pytest.param(8, 4, (8,), torch.bfloat16, torch.float32, id="tp4-max"),
        pytest.param(
            1,
            1,
            (4, 4),
            torch.float32,
            torch.bfloat16,
            id="ratio1-tp1-fp32",
        ),
        pytest.param(
            2,
            1,
            (4, 2, 0),
            torch.bfloat16,
            torch.bfloat16,
            id="ratio2-tp1-ragged-bf16",
        ),
        pytest.param(
            2,
            4,
            (8,),
            torch.float32,
            torch.float32,
            id="ratio2-tp4-max-fp32",
        ),
        pytest.param(
            3,
            2,
            (4, 2, 0),
            torch.float32,
            torch.float32,
            id="ratio3-tp2-ragged-fp32",
        ),
        pytest.param(
            4,
            4,
            (8,),
            torch.float32,
            torch.bfloat16,
            id="ratio4-tp4-max-fp32",
        ),
        pytest.param(
            8, 4, (12, 9, 0), torch.bfloat16, torch.bfloat16, id="wide-tp4-12"
        ),
        pytest.param(
            8, 4, (16, 13, 0), torch.bfloat16, torch.bfloat16, id="wide-tp4-16"
        ),
        pytest.param(
            8, 8, (12, 9, 0), torch.bfloat16, torch.bfloat16, id="wide-tp8-12"
        ),
        pytest.param(
            8, 8, (16, 13, 0), torch.bfloat16, torch.bfloat16, id="wide-tp8-16"
        ),
    ],
)
@pytest.mark.parametrize("output_gate_activation", ["silu", "sigmoid"])
@torch.inference_mode()
def test_fused_gdn_decode_post_conv_mtp_head_ratios(
    head_ratio: int,
    tp_size: int,
    query_lengths: tuple[int, ...],
    state_dtype: torch.dtype,
    norm_dtype: torch.dtype,
    output_gate_activation: str,
) -> None:
    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("fused GDN decode MTP requires compute capability 8.0+")
    if not hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp"):
        pytest.skip("fused GDN decode MTP op is not built")

    wide = max(query_lengths) > 8
    if wide and (
        torch.cuda.get_device_capability() != (10, 7)
        or output_gate_activation != "silu"
    ):
        pytest.skip("Wide GDN supports SM107 SiLU layouts")
    torch.manual_seed(0)
    device = "cuda"
    H = 16 // tp_size
    HV = head_ratio * H
    K = V = 128
    num_reqs = len(query_lengths)
    state_width = max(query_lengths)
    num_tokens = sum(query_lengths)
    num_slots = num_reqs * state_width + 1
    scale = K**-0.5
    eps = 1e-6

    mixed_qkv = torch.randn(
        num_tokens,
        2 * H * K + HV * V,
        dtype=torch.bfloat16,
        device=device,
    )
    query, key, value = torch.split(
        mixed_qkv,
        [H * K, H * K, HV * V],
        dim=-1,
    )
    query = query.view(1, num_tokens, H, K)
    key = key.view(1, num_tokens, H, K)
    value = value.view(1, num_tokens, HV, V)
    ba = torch.randn(num_tokens, 2 * HV, dtype=torch.bfloat16, device=device)
    b, a = ba.chunk(2, dim=-1)
    assert not a.is_contiguous()
    assert not b.is_contiguous()
    A_log = 0.5 * torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = 0.1 * torch.randn(HV, dtype=torch.float32, device=device)
    output_gate = torch.randn(num_tokens, HV, V, dtype=torch.bfloat16, device=device)
    norm_weight = torch.randn(V, dtype=norm_dtype, device=device)
    state_ref = (
        0.01 * torch.randn(num_slots, HV, V, K, dtype=torch.float32, device=device)
    ).to(state_dtype)
    if wide:
        # Serving packs recurrent state with convolution history in each page.
        pages = torch.full(
            (num_slots, HV * V * K + 128), 37.0, dtype=state_dtype, device=device
        )
        state_actual = pages[:, : HV * V * K].view_as(state_ref)
        state_actual.copy_(state_ref)
        dt_bias = dt_bias.to(torch.bfloat16)
    else:
        state_actual = state_ref.clone()
    output = torch.full(
        (num_tokens + (3 if wide else 0), HV, V),
        17.0,
        dtype=torch.bfloat16,
        device=device,
    )
    state_indices = torch.arange(1, num_slots, dtype=torch.int32, device=device).view(
        num_reqs, state_width
    )
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(query_lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    num_accepted_tokens = torch.ones(num_reqs, dtype=torch.int32, device=device)
    if query_lengths[-1] == 0:
        state_indices[-1].zero_()

    def run():
        return ops.fused_gdn_decode_post_conv_mtp(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            state_indices=state_indices,
            cu_seqlens=cu_seqlens,
            num_accepted_tokens=num_accepted_tokens,
            state=state_actual,
            output_gate=output_gate,
            norm_weight=norm_weight,
            out=output,
            scale=scale,
            norm_eps=eps,
            output_gate_activation=output_gate_activation,
        )

    if wide:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        state_actual.copy_(state_ref)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()

    for step, accepted_tokens in enumerate((1, min(2, state_width), state_width)):
        num_accepted_tokens.fill_(accepted_tokens)
        if query_lengths[-1] == 0:
            num_accepted_tokens[-1] = 1
        raw_ref, _ = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=a,
            b=b,
            dt_bias=dt_bias,
            q=query,
            k=key,
            v=value,
            initial_state=state_ref,
            inplace_final_state=True,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=state_indices,
            num_accepted_tokens=num_accepted_tokens,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
        )
        expected = rmsnorm_fn(
            raw_ref.squeeze(0),
            norm_weight,
            None,
            z=output_gate,
            eps=eps,
            norm_before_gate=True,
            activation=output_gate_activation,
        )
        if wide:
            graph.replay()
            actual = output[:num_tokens]
            assert torch.count_nonzero(output[num_tokens:]) == 0
            assert torch.all(pages[:, -128:] == 37)
        else:
            actual = run()

        output_error = (actual.float() - expected.float()).norm()
        output_relative_l2 = output_error / expected.float().norm().clamp_min(1e-20)
        assert output_relative_l2 < 5e-4, (
            f"MTP output relative L2 mismatch at step {step}: "
            f"{output_relative_l2.item():.6g}"
        )

    torch.testing.assert_close(state_actual, state_ref, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("width", [12, 16])
@pytest.mark.parametrize("batch", [1, 2, 16])
@torch.inference_mode()
def test_wide_gdn_fp8_rounds_through_bf16(width, batch):
    """Fusing quantization preserves BF16 rounding and every speculative checkpoint."""
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Wide GDN requires SM107")
    torch.manual_seed(7)
    tokens = width * batch

    def rand(*shape, dtype=torch.bfloat16):
        return (torch.randn(shape, device="cuda") * 0.1).to(dtype)

    state = rand(tokens + 1, 16, 128, 128)
    initial = state.clone()
    indices = torch.arange(1, tokens + 1, device="cuda", dtype=torch.int32).view(
        batch, width
    )
    args = dict(
        mixed_qkv=rand(tokens, 2560),
        a=rand(tokens, 16),
        b=rand(tokens, 16),
        A_log=rand(16, dtype=torch.float32),
        dt_bias=rand(16),
        state_indices=indices,
        cu_seqlens=torch.arange(batch + 1, device="cuda", dtype=torch.int32) * width,
        num_accepted_tokens=torch.full(
            (batch,), width, device="cuda", dtype=torch.int32
        ),
        state=state,
        output_gate=rand(tokens, 16, 128),
        norm_weight=rand(128),
        norm_eps=1e-6,
    )
    scale = torch.tensor(0.012, device="cuda", dtype=torch.float32)
    fp8 = torch.empty((tokens + 3, 16, 128), device="cuda", dtype=torch.float8_e4m3fn)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        ops.fused_gdn_decode_post_conv_mtp(**args, out=fp8, output_scale=scale)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        ops.fused_gdn_decode_post_conv_mtp(**args, out=fp8, output_scale=scale)
    for null_requests in (False, True):
        if null_requests:
            indices.zero_()
        state.copy_(initial)
        bf16 = torch.empty_like(fp8, dtype=torch.bfloat16)
        ops.fused_gdn_decode_post_conv_mtp(**args, out=bf16)
        expected = ops.scaled_fp8_quant(bf16.flatten(1), scale)[0].view_as(fp8)
        checkpoints = state.clone()
        state.copy_(initial)
        graph.replay()
        torch.testing.assert_close(
            fp8.view(torch.uint8), expected.view(torch.uint8), atol=0, rtol=0
        )
        torch.testing.assert_close(state, checkpoints, atol=0, rtol=0)
