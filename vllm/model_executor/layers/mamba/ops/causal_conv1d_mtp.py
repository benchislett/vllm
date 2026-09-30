# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small-batch, parallel-token convolution for the fused GDN MTP path."""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def _causal_conv1d_mtp_kernel(
    X,
    W,
    S,
    IDX,
    ACC,
    CU,
    C: tl.constexpr,
    T: tl.constexpr,
    XS: tl.constexpr,
    SS: tl.constexpr,
    IS: tl.constexpr,
):
    # Waiting before releasing GDN also establishes readiness of its metadata
    # and recurrent cache, which are independent of this convolution's writes.
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    request = tl.program_id(0)
    # Shared cache pages can put a slot beyond the signed 32-bit offset range.
    slot = tl.load(IDX + request * IS).to(tl.int64)
    bos = tl.load(CU + request)
    n = tl.load(CU + request + 1) - bos
    offset = tl.load(ACC + request) - 1
    if slot == 0 or n <= 0:
        return
    c = tl.program_id(1) * C + tl.arange(0, C)
    t = tl.arange(0, T)
    w0 = tl.load(W + c * 4, c < 2560, 0)
    w1 = tl.load(W + c * 4 + 1, c < 2560, 0)
    w2 = tl.load(W + c * 4 + 2, c < 2560, 0)
    w3 = tl.load(W + c * 4 + 3, c < 2560, 0)
    h0 = tl.load(S + slot * SS + offset * 2560 + c, c < 2560, 0)
    h1 = tl.load(S + slot * SS + (offset + 1) * 2560 + c, c < 2560, 0)
    h2 = tl.load(S + slot * SS + (offset + 2) * 2560 + c, c < 2560, 0)
    mask = (t[:, None] < n) & (c[None, :] < 2560)
    xx = tl.load(X + (bos + t[:, None]) * XS + c[None, :], mask, 0)
    x0 = tl.load(
        X + (bos + t[:, None] - 3) * XS + c[None, :],
        mask & (t[:, None] >= 3),
        0,
    )
    x1 = tl.load(
        X + (bos + t[:, None] - 2) * XS + c[None, :],
        mask & (t[:, None] >= 2),
        0,
    )
    x2 = tl.load(
        X + (bos + t[:, None] - 1) * XS + c[None, :],
        mask & (t[:, None] >= 1),
        0,
    )
    x0 = tl.where(
        t[:, None] == 0,
        h0[None, :],
        tl.where(
            t[:, None] == 1, h1[None, :], tl.where(t[:, None] == 2, h2[None, :], x0)
        ),
    )
    x1 = tl.where(
        t[:, None] == 0, h1[None, :], tl.where(t[:, None] == 1, h2[None, :], x1)
    )
    x2 = tl.where(t[:, None] == 0, h2[None, :], x2)
    # Preserve the stock kernel's BF16 product rounding before ordered FP32
    # accumulation. Accumulating unrounded products changes model outputs.
    y = (x0 * w0[None, :]).to(tl.float32)
    y += (x1 * w1[None, :]).to(tl.float32)
    y += (x2 * w2[None, :]).to(tl.float32)
    y += (xx * w3[None, :]).to(tl.float32)
    y = y / (1 + tl.exp(-y))
    # A CTA owns its channels across all tokens. Read every raw window before
    # overwriting QKV; cache updates also store raw, not convolved, inputs.
    tl.debug_barrier()
    tl.store(S + slot * SS + c, h1, c < 2560)
    tl.store(S + slot * SS + 2560 + c, h2, c < 2560)
    tl.store(S + slot * SS + (t[:, None] + 2) * 2560 + c[None, :], xx, mask)
    tl.store(X + (bos + t[:, None]) * XS + c[None, :], y, mask)


def try_causal_conv1d_update_mtp(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | bool | None,
    indices: torch.Tensor,
    accepted: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_query_len: int,
    num_spec_decode_tokens: int,
) -> bool:
    """Update QKV and history in place if the measured MTP contract is met.

    Return False without mutation for stock fallback. The caller supplies a
    speculative-only slice, unique positive active slots (zero is null), valid
    accepted offsets, and monotone query bounds. Full windows are established
    from host metadata, without reading device values during dispatch.
    """
    requests = indices.numel()
    if (
        not 1 <= requests <= 32
        or max_query_len not in (2, 4, 8)
        or num_spec_decode_tokens != requests * max_query_len
        or bias is not None
        or activation not in ("silu", "swish", True)
        or x.ndim != 2
        or x.shape[1] != 2560
        or x.stride() != (4608, 1)
        or x.shape[0] < num_spec_decode_tokens
        or weight.shape != (2560, 4)
        or weight.stride() != (4, 1)
        or conv_state.ndim != 3
        or conv_state.shape[1] != 2560
        or conv_state.shape[2] < max_query_len + 2
        or conv_state.stride()[1:] != (1, 2560)
        # A page can also contain recurrent state, other layers, and padding.
        or conv_state.stride(0) < 2560 * conv_state.shape[2]
        or indices.ndim != 1
        or indices.dtype != torch.int32
        or accepted.shape != (requests,)
        or accepted.stride() != (1,)
        or accepted.dtype != torch.int32
        or cu_seqlens.shape != (requests + 1,)
        or cu_seqlens.stride() != (1,)
        or cu_seqlens.dtype != torch.int32
    ):
        return False
    if any(t.dtype != torch.bfloat16 for t in (x, conv_state, weight)):
        return False
    tensors = (conv_state, weight, indices, accepted, cu_seqlens)
    if not x.is_cuda or any(t.device != x.device for t in tensors):
        return False
    if not current_platform.is_device_capability(107, device_id=x.device.index):
        return False
    # The clustered BS1 GDN benefits from more convolution CTAs. Retain the
    # broadly qualified configuration for other request/query counts.
    channels, warps = (64, 4) if requests == 1 and max_query_len == 8 else (128, 2)
    _causal_conv1d_mtp_kernel[(requests, triton.cdiv(2560, channels))](
        x,
        weight,
        conv_state,
        indices,
        accepted,
        cu_seqlens,
        C=channels,
        T=max_query_len,
        XS=x.stride(0),
        SS=conv_state.stride(0),
        IS=indices.stride(0),
        num_warps=warps,
        launch_pdl=True,
    )
    return True
