"""Native DFlash2 primitives: block isolation, window boundaries, and FP32 RoPE."""
import ctypes as C

import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_selector_accepts_target_head_candidates(glm, device, dtype):
    torch.manual_seed(73)
    batch, depth, k, rank, vocab = 2, 7, 16, 32, 128
    gates = torch.randn(batch, depth, rank, device=device, dtype=torch.bfloat16)
    predecessor = torch.randn(vocab, rank, device=device, dtype=torch.bfloat16)
    successor = torch.randn_like(predecessor)
    ids = torch.stack([torch.randperm(vocab, device=device)[:k] for _ in range(batch * depth)]).int().view(batch, depth, k)
    anchors = torch.tensor([11, 79], device=device, dtype=torch.int32)
    logits = torch.randn(batch, depth, k, device=device).to(dtype)
    previous = torch.cat((anchors[:, None, None].expand(-1, 1, k), ids[:, :-1]), 1)
    gated = (predecessor[previous.long()] * gates[:, :, None]).float()
    bilinear = torch.einsum("bdpr,bdcr->bdpc", gated, successor[ids.long()].float()).bfloat16()
    expected = bilinear.float() + logits.float()[:, :, None]
    scores = torch.empty_like(expected)
    tokens = torch.empty(batch, depth, dtype=torch.int32, device=device)
    fn = getattr(glm.lib, "glm_dflash2_select_bf16" if dtype == torch.bfloat16 else "glm_dflash2_select")
    fn.argtypes = [C.c_void_p] * 9 + [C.c_int] * 5
    fn.restype = None
    torch.cuda.synchronize(device)
    fn(glm.ctx, scores.data_ptr(), tokens.data_ptr(), gates.data_ptr(), ids.data_ptr(), logits.data_ptr(),
       predecessor.data_ptr(), successor.data_ptr(), anchors.data_ptr(), batch, depth, k, rank, vocab)
    glm.synchronize()
    torch.testing.assert_close(scores, expected, atol=0.125, rtol=0.008)
    prev = torch.zeros(batch, dtype=torch.long, device=device)
    rows = torch.arange(batch, device=device)
    for d in range(depth):
        prev = expected[rows, d, prev].argmax(-1)
        torch.testing.assert_close(tokens[:, d], ids[rows, d, prev], atol=0, rtol=0)


@pytest.mark.parametrize("side", [0, 1])
def test_dynamic_conv_resets_at_each_block(glm, device, side):
    torch.manual_seed(123)
    batch, block, channels, group = 3, 8, 6144, 16
    x = torch.randn(batch, block, channels, device=device, dtype=torch.bfloat16)
    delta = torch.randn(batch, block, 2, 2, channels // group, device=device, dtype=torch.bfloat16)
    base = torch.randn(2, 2, channels, device=device, dtype=torch.bfloat16)
    out = torch.empty_like(x)
    coefficients = delta[:, :, side].float().repeat_interleave(group, -1) + base[side].float()
    previous = torch.zeros_like(x)
    previous[:, 1:] = x[:, :-1]
    expected = (coefficients[:, :, 0] * x.float() + coefficients[:, :, 1] * previous.float()).bfloat16()
    fn = glm.lib.glm_dflash2_conv
    fn.argtypes = [C.c_void_p] * 5 + [C.c_int] * 5
    fn.restype = None
    torch.cuda.synchronize(device)
    fn(glm.ctx, out.data_ptr(), x.data_ptr(), delta.data_ptr(), base.data_ptr(),
       batch * block, channels, block, group, side)
    glm.synchronize()
    torch.testing.assert_close(out, expected, atol=0.03125, rtol=0.008)
    torch.testing.assert_close(out[:, 0], expected[:, 0], atol=0, rtol=0)


def test_noncausal_window_excludes_old_kv_but_includes_future_block(glm, device):
    qlen, context, window, page, nq, nk, hd = 8, 75, 16, 64, 4, 2, 128
    kvlen = context + qlen
    q = torch.zeros(nq, qlen, hd, dtype=torch.bfloat16, device=device)
    k = torch.zeros(2, nk, page, hd, dtype=torch.bfloat16, device=device)
    v = torch.zeros_like(k)
    # An excluded old token, the first query's exact left boundary, and a future token.
    v[0, :, 0] = 1024
    v[0, :, context - window + 1] = 16
    v[1, :, kvlen - page - 1] = 32
    out = torch.empty(qlen, nq, hd, dtype=torch.bfloat16, device=device)
    qo_h = torch.tensor([0, qlen], dtype=torch.int32, pin_memory=True)
    kv_h = torch.tensor([0, 2], dtype=torch.int32, pin_memory=True)
    qo = qo_h.to(device); kv = kv_h.to(device)
    indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
    last = torch.tensor([kvlen % page], dtype=torch.int32, device=device)
    float_ws = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    int_ws = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=device)
    pinned_ws = torch.empty_like(int_ws, device="cpu", pin_memory=True)
    plan = torch.empty(15, dtype=torch.int64, pin_memory=True)
    torch.cuda.synchronize(device)
    glm.batch_prefill_paged_plan(float_ws.data_ptr(), float_ws.numel(), int_ws.data_ptr(), pinned_ws.data_ptr(), int_ws.numel(),
                                plan.data_ptr(), qo_h.data_ptr(), kv_h.data_ptr(), qlen, 1, nq, nk, hd, page, 0)
    fn = glm.lib.glm_batch_prefill_paged_run_window
    fn.argtypes = glm.lib.glm_batch_prefill_paged_run.argtypes + [C.c_int]
    fn.restype = None
    fn(glm.ctx, q.data_ptr(), out.data_ptr(), k.data_ptr(), v.data_ptr(), indices.data_ptr(),
       kv.data_ptr(), last.data_ptr(), float_ws.data_ptr(), int_ws.data_ptr(), qo.data_ptr(),
       plan.data_ptr(), qlen, 1, nq, nk, hd, page, hd, qlen * hd, 0, hd ** -0.5, window - 1)
    glm.synchronize()
    values = v.permute(1, 0, 2, 3).reshape(nk, 2 * page, hd)[:, :kvlen].float()
    mask = torch.arange(kvlen, device=device)[None] >= (context + torch.arange(qlen, device=device)[:, None] - window + 1)
    expected = torch.einsum("qk,hkd->qhd", mask.float() / mask.sum(-1, keepdim=True), values)
    expected = expected.repeat_interleave(nq // nk, 1).bfloat16()
    torch.testing.assert_close(out, expected, atol=0.016, rtol=0.008)
    assert out[0, 0, 0] > out[1, 0, 0]


def test_float_inverse_frequency_preserves_long_position_rope(glm, device):
    hd = 128
    inv = 1.0 / (1000000 ** (torch.arange(0, hd, 2, device=device).float() / hd))
    positions = torch.tensor([0, 103, 2053, 123456, 1048500], dtype=torch.int32, device=device)
    cos = torch.empty(positions.numel(), hd, dtype=torch.bfloat16, device=device)
    sin = torch.empty_like(cos)
    fn = glm.lib.glm_rotary_embedding_f32
    fn.argtypes = [C.c_void_p] * 5 + [C.c_int] * 3
    fn.restype = None
    torch.cuda.synchronize(device)
    fn(glm.ctx, cos.data_ptr(), sin.data_ptr(), inv.data_ptr(), positions.data_ptr(), hd // 2, 1, positions.numel())
    glm.synchronize()
    phase = (positions.float()[:, None] * inv).repeat(1, 2)
    torch.testing.assert_close(cos, phase.cos().bfloat16(), atol=0.004, rtol=0.004)
    torch.testing.assert_close(sin, phase.sin().bfloat16(), atol=0.004, rtol=0.004)
