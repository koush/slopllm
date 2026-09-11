"""Decode chunk hints versus a dense FP32 reference on the same quantized data."""
import math
import pytest
import torch


@pytest.mark.parametrize("queries", [4, 8])
@pytest.mark.parametrize("length", [128, 256, 512, 1024, 2048])
def test_sparse_mla_chunk_hints(glm, device, queries, length):
    torch.manual_seed(631 + length)
    heads, topk, splits, dim = 64, 2048, 32, 512
    # Exact FP8 inputs and unit scales isolate attention errors from quantization.
    q = (torch.randn(queries, heads, dim, device=device) * .3).to(torch.float8_e4m3fn)
    qr = (torch.randn(queries, heads, 64, device=device) * .3).bfloat16()
    qs = torch.ones(queries, heads, 4, device=device)
    k = (torch.randn(topk, dim, device=device) * .3).to(torch.float8_e4m3fn)
    kr = (torch.randn(topk, 64, device=device) * .3).bfloat16()
    packed = torch.empty(topk, 656, dtype=torch.uint8, device=device)
    packed[:, :512] = k.view(torch.uint8)
    packed[:, 512:528] = torch.ones(topk, 4, device=device).view(torch.uint8)
    packed[:, 528:] = kr.view(torch.uint8)
    indices = torch.full((queries, topk), -1, dtype=torch.int32, device=device)
    lengths = [max(1, length - i * 17) for i in range(queries)]
    for i, n in enumerate(lengths):
        indices[i, :n] = torch.randperm(topk, device=device)[:n].int()
    valid = torch.tensor(lengths, dtype=torch.int32, device=device)
    mid = torch.empty(queries, heads, splits, dim, dtype=torch.bfloat16, device=device)
    mid_lse = torch.empty(queries, heads, splits, device=device)
    out = torch.empty(queries, heads, dim, dtype=torch.bfloat16, device=device)
    lse = torch.empty(queries, heads, device=device)
    reference, reference_lse = [], []
    for i, n in enumerate(lengths):
        slots = indices[i, :n].long()
        scores = (q[i].float() @ k.float()[slots].T + qr[i].float() @ kr[slots].float().T) / 16
        reference.append(scores.softmax(-1) @ k.float()[slots])
        # FlashInfer exposes base-2 LSE for the subsequent CP softmax merge.
        reference_lse.append(scores.logsumexp(-1) / math.log(2))
    reference = torch.stack(reference)
    reference_lse = torch.stack(reference_lse)
    torch.cuda.synchronize(device)
    for cpb in [0, 1, 2, 3, 4]:
        glm.sparse_mla_decode_split_q(
            q.data_ptr(), qr.data_ptr(), packed.data_ptr(), indices.data_ptr(),
            mid.data_ptr(), mid_lse.data_ptr(), out.data_ptr(), lse.data_ptr(),
            queries, heads, topk, splits, 1 / 16, 64 * 656,
            chunks_per_block=cpb, topk_length=valid.data_ptr(), q_scales=qs.data_ptr())
        glm.synchronize()
        # FP8 softmax weights and BF16 partial outputs introduce rounding.
        torch.testing.assert_close(out.float(), reference, atol=.003, rtol=.03)
        torch.testing.assert_close(lse, reference_lse, atol=.002, rtol=.001)
