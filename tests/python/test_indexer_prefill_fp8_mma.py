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


def _run_prefill(glm, device, flat, causal, total_q=65, length=137, topk=23):
    torch.manual_seed(9128)
    page_size = 64
    rows, row_scales = pack_indexer_k(torch.randn(length, 128, dtype=torch.bfloat16, device=device))
    num_pages = (length + page_size - 1) // page_size
    page_ids = torch.arange(num_pages, dtype=torch.int32, device=device)
    paged = torch.zeros(num_pages, page_size, 128, dtype=torch.uint8, device=device)
    paged_scales = torch.zeros(num_pages, page_size, dtype=torch.float32, device=device)
    paged.view(-1, 128)[:length].copy_(rows)
    paged_scales.view(-1)[:length].copy_(row_scales)

    q = torch.randn(total_q, 32, 128, dtype=torch.bfloat16, device=device)
    weights = torch.randn(total_q, 32, dtype=torch.bfloat16, device=device)
    out = torch.full((total_q, topk), -2, dtype=torch.int32, device=device)
    out_scores = torch.full((total_q, topk), -torch.inf, dtype=torch.bfloat16, device=device)
    scores = torch.full((total_q, length), 8192, dtype=torch.bfloat16, device=device)
    row_len = torch.zeros(total_q, dtype=torch.int32, device=device)
    coarse = torch.empty(total_q, 1024, dtype=torch.int32, device=device)
    fine = torch.empty(total_q, 64, dtype=torch.int32, device=device)
    meta = torch.empty(total_q, 4, dtype=torch.int32, device=device)
    page_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    last_page_len = torch.tensor([length - (num_pages - 1) * page_size], dtype=torch.int32, device=device)
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


def _assert_topk_scores(out, out_scores, scores, valid, topk):
    selected = out.long()
    assert len(torch.unique(selected)) == topk
    assert torch.all((selected >= 0) & (selected < valid))
    torch.testing.assert_close(out_scores, scores[selected], rtol=0, atol=0)
    expected_scores = torch.topk(scores[:valid].float(), topk).values.sort().values
    actual_scores = out_scores.float().sort().values
    torch.testing.assert_close(actual_scores, expected_scores, rtol=0, atol=0)


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
        _assert_topk_scores(out[qi], out_scores[qi], scores[qi], valid, topk)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefill_fp8_mma_default_tma_swizzle_matches_retained_kernel(glm, device, monkeypatch):
    if torch.cuda.get_device_capability(device)[0] < 12:
        pytest.skip("SM120 required")

    monkeypatch.setenv("GLM_INDEXER_PREFILL_FP8_CONFIG", "q32_k256_w8_h8")
    baseline = _run_prefill(glm, device, flat=True, causal=True)
    monkeypatch.delenv("GLM_INDEXER_PREFILL_FP8_CONFIG", raising=False)
    actual = _run_prefill(glm, device, flat=True, causal=True)

    torch.testing.assert_close(actual[6], baseline[6], rtol=0, atol=0)
    assert torch.equal(actual[7], baseline[7])
    for qi, valid in enumerate(actual[7].cpu().tolist()):
        _assert_topk_scores(actual[4][qi], actual[5][qi], actual[6][qi], valid, actual[8])
        _assert_topk_scores(baseline[4][qi], baseline[5][qi], baseline[6][qi], valid, baseline[8])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefill_deterministic_gather_is_repeatable(glm, device, monkeypatch):
    monkeypatch.setenv("GLM_INDEXER_PREFILL_DETERMINISTIC", "1")
    first = _run_prefill(glm, device, flat=True, causal=False, total_q=2, length=32768)
    second = _run_prefill(glm, device, flat=True, causal=False, total_q=2, length=32768)
    assert torch.equal(first[4], second[4])
    assert torch.equal(first[5], second[5])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefill_atomic_gather_long_context(glm, device, monkeypatch):
    monkeypatch.delenv("GLM_INDEXER_PREFILL_DETERMINISTIC", raising=False)
    result = _run_prefill(glm, device, flat=True, causal=False, total_q=4, length=32768)
    out, out_scores, scores, row_len, topk = result[4:]
    for qi, valid in enumerate(row_len.cpu().tolist()):
        _assert_topk_scores(out[qi], out_scores[qi], scores[qi], valid, topk)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_prefill_fp8_mma_kill_switch_uses_bf16(glm, device, monkeypatch):
    monkeypatch.setenv("GLM_INDEXER_DECODE_FP8_MMA", "0")
    q, weights, rows, row_scales, out, out_scores, scores, row_len, topk = _run_prefill(
        glm, device, flat=True, causal=False,
    )

    k = unpack_indexer_k(rows, row_scales).float()
    ref = (weights.float()[:, :, None] * torch.relu(
        torch.einsum("qhd,kd->qhk", q.float(), k) * (128 ** -0.5)
    )).sum(1).to(torch.bfloat16)
    assert torch.all(row_len == rows.shape[0])
    torch.testing.assert_close(scores.float(), ref.float(), rtol=2e-2, atol=0.5)
    for qi in range(q.shape[0]):
        _assert_topk_scores(out[qi], out_scores[qi], scores[qi], rows.shape[0], topk)
