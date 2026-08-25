"""Focused SM120 FP8-MMA decode scorer checks."""

import numpy as np
import pytest
import torch

from helpers import pack_indexer_k, unpack_indexer_k


TOPK_SCRATCH_I32 = 1056


def _quantize_q(q):
    values = q.float()
    amax = values.abs().amax(dim=-1, keepdim=True)
    raw_scale = torch.maximum(amax, torch.tensor(1e-4, device=q.device)) / 448.0
    q_scale = torch.pow(2.0, torch.ceil(torch.log2(raw_scale)))
    q8 = (values / q_scale).to(torch.float8_e4m3fn).float()
    return q8, q_scale


def _oracle(q, packed_rows, weights, scale, fp8_q):
    k = unpack_indexer_k(packed_rows, 128).float()
    if fp8_q:
        qv, q_scale = _quantize_q(q)
        qv = qv * q_scale
    else:
        qv = q.float()
    dots = torch.einsum("hd,kd->hk", qv[0], k)
    return (weights[0].float()[:, None] * torch.relu(dots * scale)).sum(0).to(torch.bfloat16)


def _run(glm, q, k_data, packed_rows, weights, page_indices, page_indptr,
         last_page_len, qo_indptr, page_size, topk, flat_indptr=None,
         max_kv=None, **kwargs):
    length = packed_rows.shape[0]
    max_kv = max_kv or length
    out = torch.full((1, topk), -2, dtype=torch.int32, device=q.device)
    out_scores = torch.full((1, topk), -torch.inf, dtype=torch.bfloat16, device=q.device)
    scores = torch.full((1, max_kv), -torch.inf, dtype=torch.bfloat16, device=q.device)
    row_len = torch.zeros(1, dtype=torch.int32, device=q.device)
    hist = torch.empty(1, TOPK_SCRATCH_I32, dtype=torch.int32, device=q.device)
    meta = torch.empty(1, 4, dtype=torch.int32, device=q.device)
    glm.indexer_score_topk_v2(
        out, out_scores, q, k_data, weights, page_indices, page_indptr,
        last_page_len, qo_indptr, 128 ** -0.5, 1, 32, 128, page_size,
        topk, False, scores, row_len, hist, meta, max_kv,
        max(1, (max_kv + 255) // 256), kv_token_indptr=flat_indptr,
        **kwargs,
    )
    glm.synchronize()
    return out[0], out_scores[0], scores[0, :length], row_len.item()


@pytest.fixture
def decode_case(device):
    torch.manual_seed(9127)
    length, page_size = 137, 64
    rows = pack_indexer_k(torch.randn(length, 128, dtype=torch.bfloat16, device=device))
    q = torch.randn(1, 32, 128, dtype=torch.bfloat16, device=device)
    weights = torch.randn(1, 32, dtype=torch.bfloat16, device=device)

    page_ids = torch.tensor([3, 1, 5], dtype=torch.int32, device=device)
    paged = torch.zeros(6, page_size, 132, dtype=torch.uint8, device=device)
    for p, page_id in enumerate(page_ids.tolist()):
        lo, hi = p * page_size, min((p + 1) * page_size, length)
        paged[page_id, :hi - lo].copy_(rows[lo:hi])
    flat = rows.clone()
    page_indptr = torch.tensor([0, 3], dtype=torch.int32, device=device)
    last_page_len = torch.tensor([9], dtype=torch.int32, device=device)
    qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    flat_indptr = torch.tensor([0, length], dtype=torch.int32, device=device)
    return q, weights, rows, paged, flat, page_ids, page_indptr, last_page_len, qo_indptr, flat_indptr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("layout", ["paged", "flat"])
def test_decode_fp8_mma_scores_and_topk(glm, device, decode_case, monkeypatch, layout):
    if torch.cuda.get_device_capability(device)[0] < 12:
        pytest.skip("SM120 required")
    monkeypatch.setenv("GLM_INDEXER_DECODE_FP8_MMA", "1")
    q, weights, rows, paged, flat, pi, pip, lpl, qoi, flat_indptr = decode_case
    k_data = paged if layout == "paged" else flat
    out, out_scores, scores, row_len = _run(
        glm, q, k_data, rows, weights, pi, pip, lpl, qoi, 64, 23,
        flat_indptr if layout == "flat" else None,
    )
    ref = _oracle(q, rows, weights, 128 ** -0.5, fp8_q=True)

    assert row_len == rows.shape[0]
    torch.testing.assert_close(scores.float(), ref.float(), rtol=2e-2, atol=0.5)
    expected = set(torch.topk(ref.float(), 23).indices.cpu().tolist())
    assert set(out.cpu().tolist()) == expected
    torch.testing.assert_close(out_scores.float(), scores[out.long()].float(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_decode_fp8_mma_kill_switch_uses_scalar(glm, device, decode_case, monkeypatch):
    monkeypatch.setenv("GLM_INDEXER_DECODE_FP8_MMA", "0")
    q, weights, rows, paged, _, pi, pip, lpl, qoi, _ = decode_case
    out, _, scores, row_len = _run(
        glm, q, paged, rows, weights, pi, pip, lpl, qoi, 64, 23,
    )
    ref = _oracle(q, rows, weights, 128 ** -0.5, fp8_q=False)

    assert row_len == rows.shape[0]
    assert len(np.unique(out.cpu().numpy())) == 23
    torch.testing.assert_close(scores.float(), ref.float(), rtol=2e-2, atol=0.5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_decode_fp8_mma_cp_partial_page_and_mask(glm, device, monkeypatch):
    if torch.cuda.get_device_capability(device)[0] < 12:
        pytest.skip("SM120 required")
    monkeypatch.setenv("GLM_INDEXER_DECODE_FP8_MMA", "1")
    torch.manual_seed(7123)
    world_size, rank, page_size, length = 8, 5, 8, 17
    rows = pack_indexer_k(torch.randn(length, 128, dtype=torch.bfloat16, device=device))
    page_ids = torch.tensor([2, 0, 3], dtype=torch.int32, device=device)
    paged = torch.zeros(4, page_size, 132, dtype=torch.uint8, device=device)
    for page, page_id in enumerate(page_ids.tolist()):
        lo, hi = page * page_size, min((page + 1) * page_size, length)
        paged[page_id, :hi - lo].copy_(rows[lo:hi])
    q = torch.randn(1, 32, 128, dtype=torch.bfloat16, device=device)
    weights = torch.randn(1, 32, dtype=torch.bfloat16, device=device)
    page_indptr = torch.tensor([0, 3], dtype=torch.int32, device=device)
    last_page_len = torch.tensor([1], dtype=torch.int32, device=device)
    qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    global_last_page_len = torch.tensor([8], dtype=torch.int32, device=device)
    custom_mask = torch.tensor([0xff, 0xdf], dtype=torch.uint8, device=device)
    mask_indptr = torch.tensor([0, 2], dtype=torch.int32, device=device)
    mask_kv_len = torch.tensor([16], dtype=torch.int32, device=device)

    out, _, scores, row_len = _run(
        glm, q, paged, rows, weights, page_ids, page_indptr, last_page_len,
        qo_indptr, page_size, 11, cp_world_size=world_size, cp_rank=rank,
        global_last_page_len=global_last_page_len, custom_mask=custom_mask,
        mask_indptr=mask_indptr, mask_kv_len=mask_kv_len,
    )
    ref = _oracle(q, rows, weights, 128 ** -0.5, fp8_q=True)
    ref[-1] = -torch.inf  # Global position 133 is mask-window column 13.
    expected_local = torch.topk(ref.float(), 11).indices
    expected_global = set((expected_local * world_size + rank).flatten().cpu().tolist())

    assert row_len == length
    assert set(out.flatten().cpu().tolist()) == expected_global
    torch.testing.assert_close(scores.float(), ref.float(), rtol=2e-2, atol=0.5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("global_length,rank", [(64013, 5), (127899, 0)])
def test_decode_fp8_mma_production_cp_geometry(
    glm, device, monkeypatch, global_length, rank,
):
    if torch.cuda.get_device_capability(device)[0] < 12:
        pytest.skip("SM120 required")
    monkeypatch.setenv("GLM_INDEXER_DECODE_FP8_MMA", "1")
    torch.manual_seed(8100 + rank)
    world_size, page_size, max_pages = 8, 64, 512
    global_page_size = world_size * page_size
    num_pages = (global_length + global_page_size - 1) // global_page_size
    global_last = global_length - (num_pages - 1) * global_page_size
    local_length = max(0, (global_length - 1 - rank) // world_size + 1)
    local_last = local_length % page_size or page_size

    rows = pack_indexer_k(
        torch.randn(local_length, 128, dtype=torch.bfloat16, device=device)
    )
    page_ids = torch.randperm(max_pages, dtype=torch.int32, device=device)[:num_pages]
    paged = torch.zeros(max_pages, page_size, 132, dtype=torch.uint8, device=device)
    for page, page_id in enumerate(page_ids.cpu().tolist()):
        lo, hi = page * page_size, min((page + 1) * page_size, local_length)
        if hi > lo:
            paged[page_id, :hi - lo].copy_(rows[lo:hi])

    q = torch.randn(1, 32, 128, dtype=torch.bfloat16, device=device)
    weights = torch.randn(1, 32, dtype=torch.bfloat16, device=device)
    out, _, scores, row_len = _run(
        glm, q, paged, rows, weights, page_ids,
        torch.tensor([0, num_pages], dtype=torch.int32, device=device),
        torch.tensor([local_last], dtype=torch.int32, device=device),
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        page_size, 23, max_kv=max_pages * page_size,
        cp_world_size=world_size, cp_rank=rank,
        global_last_page_len=torch.tensor(
            [global_last], dtype=torch.int32, device=device
        ),
    )
    ref = _oracle(q, rows, weights, 128 ** -0.5, fp8_q=True)
    expected = set(
        (torch.topk(ref.float(), 23).indices * world_size + rank).cpu().tolist()
    )

    assert row_len == local_length
    assert set(out.cpu().tolist()) == expected
    torch.testing.assert_close(scores.float(), ref.float(), rtol=2e-2, atol=0.5)
