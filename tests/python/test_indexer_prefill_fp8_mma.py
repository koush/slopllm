"""Focused SM120 FP8-MMA prefill scorer checks."""

import pytest
import torch

from helpers import pack_indexer_k, unpack_indexer_k


def _quantize_q(q):
    values = q.float()
    amax = values.abs().amax(dim=-1, keepdim=True)
    raw_scale = torch.maximum(amax, torch.tensor(1e-4, device=q.device)) / 448.0
    q_scale = torch.pow(2.0, torch.ceil(torch.log2(raw_scale)))
    return (values / q_scale).to(torch.float8_e4m3fn).float(), q_scale


def _run_prefill(glm, device, flat, causal):
    torch.manual_seed(9128)
    total_q, length, page_size, topk = 65, 137, 64, 23
    rows, row_scales = pack_indexer_k(torch.randn(length, 128, dtype=torch.bfloat16, device=device))
    page_ids = torch.tensor([3, 1, 5], dtype=torch.int32, device=device)
    paged = torch.zeros(6, page_size, 128, dtype=torch.uint8, device=device)
    paged_scales = torch.zeros(6, page_size, dtype=torch.float32, device=device)
    for page, page_id in enumerate(page_ids.tolist()):
        lo, hi = page * page_size, min((page + 1) * page_size, length)
        paged[page_id, :hi - lo].copy_(rows[lo:hi])
        paged_scales[page_id, :hi - lo].copy_(row_scales[lo:hi])

    q = torch.randn(total_q, 32, 128, dtype=torch.bfloat16, device=device)
    weights = torch.randn(total_q, 32, dtype=torch.bfloat16, device=device)
    out = torch.full((total_q, topk), -2, dtype=torch.int32, device=device)
    out_scores = torch.full((total_q, topk), -torch.inf, dtype=torch.bfloat16, device=device)
    scores = torch.full((total_q, length), 8192, dtype=torch.bfloat16, device=device)
    row_len = torch.zeros(total_q, dtype=torch.int32, device=device)
    coarse = torch.empty(total_q, 1024, dtype=torch.int32, device=device)
    fine = torch.empty(total_q, 64, dtype=torch.int32, device=device)
    meta = torch.empty(total_q, 4, dtype=torch.int32, device=device)
    page_indptr = torch.tensor([0, 3], dtype=torch.int32, device=device)
    last_page_len = torch.tensor([9], dtype=torch.int32, device=device)
    qo_indptr = torch.tensor([0, total_q], dtype=torch.int32, device=device)
    flat_indptr = torch.tensor([0, length], dtype=torch.int32, device=device)

    glm.indexer_score_topk_prefill(
        out, out_scores, q, rows if flat else paged,
        row_scales if flat else paged_scales, weights,
        page_ids, page_indptr, last_page_len, qo_indptr,
        128 ** -0.5, total_q, 32, 128, page_size, topk, causal,
        scores, row_len, length, coarse, fine, meta, 1,
        kv_token_indptr=flat_indptr if flat else None,
    )
    glm.synchronize()
    return q, weights, rows, row_scales, out, out_scores, scores, row_len, topk


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("flat", [False, True])
def test_prefill_fp8_mma_default_scores_and_topk(glm, device, monkeypatch, flat):
    if torch.cuda.get_device_capability(device)[0] < 12:
        pytest.skip("SM120 required")
    monkeypatch.delenv("GLM_INDEXER_DECODE_FP8_MMA", raising=False)
    q, weights, rows, row_scales, out, out_scores, scores, row_len, topk = _run_prefill(
        glm, device, flat, causal=True,
    )

    q8, q_scale = _quantize_q(q)
    k = unpack_indexer_k(rows, row_scales).float()
    ref = (weights.float()[:, :, None] * torch.relu(
        torch.einsum("qhd,kd->qhk", q8 * q_scale, k) * (128 ** -0.5)
    )).sum(1).to(torch.bfloat16)
    expected_lengths = torch.arange(rows.shape[0] - q.shape[0] + 1, rows.shape[0] + 1, device=device)
    torch.testing.assert_close(row_len, expected_lengths.to(torch.int32), rtol=0, atol=0)
    for qi, valid in enumerate(expected_lengths.tolist()):
        torch.testing.assert_close(scores[qi, :valid].float(), ref[qi, :valid].float(), rtol=2e-2, atol=0.5)
        expected = set(torch.topk(ref[qi, :valid].float(), topk).indices.cpu().tolist())
        assert set(out[qi].cpu().tolist()) == expected
        torch.testing.assert_close(out_scores[qi].float(), scores[qi, out[qi].long()].float(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefill_fp8_mma_default_tma_swizzle_matches_retained_kernel(glm, device, monkeypatch):
    if torch.cuda.get_device_capability(device)[0] < 12:
        pytest.skip("SM120 required")

    monkeypatch.setenv("GLM_INDEXER_PREFILL_FP8_CONFIG", "q32_k256_w8_h8")
    baseline = _run_prefill(glm, device, flat=True, causal=True)
    monkeypatch.delenv("GLM_INDEXER_PREFILL_FP8_CONFIG", raising=False)
    actual = _run_prefill(glm, device, flat=True, causal=True)

    torch.testing.assert_close(actual[5], baseline[5], rtol=0, atol=0)
    torch.testing.assert_close(actual[6], baseline[6], rtol=0, atol=0)
    assert torch.equal(actual[7], baseline[7])
    for actual_row, baseline_row in zip(actual[4], baseline[4]):
        assert set(actual_row.cpu().tolist()) == set(baseline_row.cpu().tolist())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefill_fp8_mma_kill_switch_uses_bf16(glm, device, monkeypatch):
    monkeypatch.setenv("GLM_INDEXER_DECODE_FP8_MMA", "0")
    q, weights, rows, row_scales, out, _, scores, row_len, topk = _run_prefill(
        glm, device, flat=True, causal=False,
    )

    k = unpack_indexer_k(rows, row_scales).float()
    ref = (weights.float()[:, :, None] * torch.relu(
        torch.einsum("qhd,kd->qhk", q.float(), k) * (128 ** -0.5)
    )).sum(1).to(torch.bfloat16)
    assert torch.all(row_len == rows.shape[0])
    torch.testing.assert_close(scores.float(), ref.float(), rtol=2e-2, atol=0.5)
    for qi in range(q.shape[0]):
        expected = set(torch.topk(ref[qi].float(), topk).indices.cpu().tolist())
        assert set(out[qi].cpu().tolist()) == expected
