"""DSV4 SM120 native integration: dual-cache normalization, sinks and masking."""
import ctypes
import math

import pytest
import torch


def packed_cache(device, seed):
    torch.manual_seed(seed)
    values = (torch.randn(4, 64, 512, device=device) * 0.15).bfloat16()
    packed = torch.zeros(4, 64 * 584, dtype=torch.uint8, device=device)
    data = packed[:, :64 * 576].view(4, 64, 576)
    scales = packed[:, 64 * 576:].view(4, 64, 8)
    decoded = values.float().clone()
    for block in range(7):
        x = values[..., block * 64:(block + 1) * 64].float()
        scale = 2 ** torch.ceil(torch.log2(x.abs().amax(-1).clamp_min(1e-4) / 448))
        quant = (x / scale[..., None]).to(torch.float8_e4m3fn)
        data[..., block * 64:(block + 1) * 64] = quant.view(torch.uint8)
        scales[..., block] = ((scale.view(torch.int32) >> 23) & 255).byte()
        decoded[..., block * 64:(block + 1) * 64] = quant.float() * scale[..., None]
    data[..., 448:] = values[..., 448:].contiguous().view(torch.uint8)
    return packed, decoded.reshape(-1, 512)


@pytest.mark.parametrize("tokens,heads,dual,decode", [
    (1, 8, False, True), (3, 8, True, True), (2, 24, True, True),
    (5, 8, False, False), (65, 8, True, False), (5, 64, True, False),
])
def test_dsv4_attention_matches_selected_softmax(glm, device, tokens, heads, dual, decode):
    local, local_values = packed_cache(device, 17)
    extra, extra_values = packed_cache(device, 29)
    q = (torch.randn(tokens, heads, 512, device=device) * 0.2).bfloat16()
    sinks = torch.linspace(-2, 2, heads, device=device)
    indices = torch.stack([torch.randperm(256, device=device)[:128] for _ in range(tokens)]).int()
    extra_indices = torch.stack([torch.randperm(256, device=device)[:64] for _ in range(tokens)]).int()
    lengths = torch.tensor([0 if i == 0 else 67 + i % 60 for i in range(tokens)], device=device, dtype=torch.int32)
    extra_lengths = torch.tensor([0 if i == 0 else 31 + i % 32 for i in range(tokens)], device=device, dtype=torch.int32)
    indices[:, 2] = -1
    extra_indices[:, 4] = -1
    out = torch.empty_like(q)
    lse = torch.empty(tokens, heads, device=device)
    splits = 3 if decode else 0
    mid = torch.empty(tokens, (heads + 15) // 16 * 16, max(splits, 1), 512, dtype=torch.bfloat16, device=device)
    mid_lse = torch.empty(mid.shape[:-1], device=device)
    fn = glm.lib.glm_dsv4_sparse_attention
    fn.restype = ctypes.c_bool
    fn.argtypes = [ctypes.c_void_p] * 13 + [ctypes.c_int] * 6 + [ctypes.c_float]
    torch.cuda.synchronize(device)
    assert fn(glm.ctx, q.data_ptr(), local.data_ptr(), indices.data_ptr(), lengths.data_ptr(),
              extra.data_ptr() if dual else 0, extra_indices.data_ptr() if dual else 0,
              extra_lengths.data_ptr() if dual else 0, sinks.data_ptr(), out.data_ptr(), lse.data_ptr(),
              mid.data_ptr(), mid_lse.data_ptr(), tokens, heads, 128, 64 if dual else 0,
              64 if dual else 0, splits, 512 ** -0.5)
    torch.cuda.synchronize(device)
    for token in range(tokens):
        slots = indices[token, :lengths[token]]
        values = local_values[slots[slots >= 0].long()]
        if dual:
            slots = extra_indices[token, :extra_lengths[token]]
            values = torch.cat([values, extra_values[slots[slots >= 0].long()]])
        scores = q[token].float() @ values.T * (512 ** -0.5)
        scores = torch.cat([scores, sinks[:, None]], dim=1)
        ref_lse = scores.logsumexp(-1)
        ref_out = scores.softmax(-1)[:, :-1] @ values
        torch.testing.assert_close(out[token].float(), ref_out, atol=0.004, rtol=0.04)
        # FlashInfer sparse MLA exports base-2 LSE; sink logits are natural-log units.
        torch.testing.assert_close(lse[token] * math.log(2), ref_lse, atol=0.004, rtol=0.004)
