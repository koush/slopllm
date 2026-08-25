"""Parity tests for the flat packed-FP8 indexer cache representation."""

import pytest
import torch

from helpers import pack_indexer_k


def _make_caches(device, seq_lens, page_size, head_dim, seed=42):
    torch.manual_seed(seed)
    page_counts = [(length + page_size - 1) // page_size for length in seq_lens]
    used_pages = sum(page_counts)
    max_pages = used_pages + 5
    page_ids = list(reversed(range(used_pages)))
    paged_bf16 = torch.randn(max_pages, page_size, head_dim, dtype=torch.bfloat16, device=device)
    paged = pack_indexer_k(paged_bf16)

    page_indptr = [0]
    last_page_len = []
    flat_parts = []
    offset = 0
    for length, page_count in zip(seq_lens, page_counts):
        ids = page_ids[offset:offset + page_count]
        offset += page_count
        page_indptr.append(page_indptr[-1] + page_count)
        last = length - (page_count - 1) * page_size
        last_page_len.append(last)
        parts = [paged[page_id, :last if i == page_count - 1 else page_size]
                 for i, page_id in enumerate(ids)]
        flat_parts.append(torch.cat(parts, dim=0))

    kv_token_indptr = [0]
    for length in seq_lens:
        kv_token_indptr.append(kv_token_indptr[-1] + length)

    return {
        "paged": paged,
        "flat": torch.cat(flat_parts, dim=0).contiguous(),
        "page_indices": torch.tensor(page_ids, dtype=torch.int32, device=device),
        "page_indptr": torch.tensor(page_indptr, dtype=torch.int32, device=device),
        "last_page_len": torch.tensor(last_page_len, dtype=torch.int32, device=device),
        "kv_token_indptr": torch.tensor(kv_token_indptr, dtype=torch.int32, device=device),
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("head_dim", [64, 128])
def test_flat_append_multi_sequence(glm, device, head_dim):
    seq_lens = [7, 5, 9]
    kv_token_indptr = torch.tensor([0, 7, 12, 21], dtype=torch.int32, device=device)
    batch_indices = torch.tensor([0, 1, 2, 0, 2, 1], dtype=torch.int32, device=device)
    positions = torch.tensor([3, 4, 8, 0, 2, 1], dtype=torch.int32, device=device)
    stride = head_dim + 8
    source_storage = torch.arange(len(batch_indices) * stride, dtype=torch.float32, device=device)
    source_storage = source_storage.reshape(len(batch_indices), stride).to(torch.bfloat16)
    destination = torch.full((sum(seq_lens), head_dim + 4), 0xa5, dtype=torch.uint8, device=device)
    expected = destination.clone()

    for i, (batch, position) in enumerate(zip(batch_indices.tolist(), positions.tolist())):
        expected[kv_token_indptr[batch].item() + position] = pack_indexer_k(source_storage[i:i + 1, :head_dim])[0]

    glm.indexer_kv_cache_append_flat(
        destination, source_storage, kv_token_indptr, batch_indices, positions,
        len(batch_indices), head_dim, stride,
    )
    glm.synchronize()

    assert torch.equal(destination, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_paged_append_cp_packs_owned_tokens(glm, device):
    head_dim, page_size, world_size, rank = 128, 8, 2, 1
    append_k = torch.randn(page_size, head_dim, dtype=torch.bfloat16, device=device)
    destination = torch.zeros(1, page_size // world_size, head_dim + 4,
                              dtype=torch.uint8, device=device)
    indices = torch.tensor([0], dtype=torch.int32, device=device)
    indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    last_page_len = torch.tensor([page_size], dtype=torch.int32, device=device)
    batches = torch.zeros(page_size, dtype=torch.int32, device=device)
    positions = torch.arange(page_size, dtype=torch.int32, device=device)

    glm.mla_kv_cache_append(
        destination.data_ptr(), 0,
        indices.data_ptr(), indptr.data_ptr(), last_page_len.data_ptr(),
        append_k.data_ptr(), 0, batches.data_ptr(), positions.data_ptr(),
        page_size, page_size, head_dim, 0, head_dim, 0,
        cp_rank=rank, cp_world_size=world_size,
    )
    glm.synchronize()

    assert torch.equal(destination[0], pack_indexer_k(append_k[rank::world_size]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flat_append_patches_gathered_pages(glm, device):
    seq_lens = [29, 11]
    page_size = 16
    head_dim = 128
    caches = _make_caches(device, seq_lens, page_size, head_dim, seed=99)
    gathered = torch.empty(sum(seq_lens), head_dim + 4, dtype=torch.uint8, device=device)
    glm.gather_pages(
        gathered, caches["paged"], caches["page_indices"], caches["page_indptr"],
        caches["last_page_len"], caches["paged"].shape[0], len(seq_lens),
        page_size, head_dim + 4,
    )

    batch_indices = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
    positions = torch.tensor([28, 4, 10, 0], dtype=torch.int32, device=device)
    append_k = torch.randn(len(batch_indices), head_dim, dtype=torch.bfloat16, device=device)
    expected = caches["flat"].clone()
    for i, (batch, position) in enumerate(zip(batch_indices.tolist(), positions.tolist())):
        expected[caches["kv_token_indptr"][batch].item() + position] = pack_indexer_k(append_k[i:i + 1])[0]

    glm.indexer_kv_cache_append_flat(
        gathered, append_k, caches["kv_token_indptr"], batch_indices, positions,
        len(batch_indices), head_dim, head_dim,
    )
    glm.synchronize()

    assert torch.equal(gathered, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("causal", [False, True])
def test_flat_raw_score_matches_paged(glm, device, causal):
    seq_lens = [37, 19]
    page_size = 16
    n_heads, head_dim = 4, 64
    caches = _make_caches(device, seq_lens, page_size, head_dim)
    total_q = sum(seq_lens)
    max_kv = max(seq_lens)
    q = torch.randn(total_q, n_heads, head_dim, dtype=torch.bfloat16, device=device)
    weights = torch.randn(total_q, n_heads, dtype=torch.bfloat16, device=device)
    qo_indptr = torch.tensor([0, seq_lens[0], total_q], dtype=torch.int32, device=device)
    paged_scores = torch.full((total_q, max_kv), float("-inf"), dtype=torch.bfloat16, device=device)
    flat_scores = paged_scores.clone()
    scale = head_dim ** -0.5

    args = (
        q, caches["paged"], weights, caches["page_indices"], caches["page_indptr"],
        caches["last_page_len"], qo_indptr, scale, total_q, n_heads, head_dim,
        page_size, max_kv, causal,
    )
    glm.indexer_score(paged_scores, *args)
    flat_args = list(args)
    flat_args[1] = caches["flat"]
    glm.indexer_score(flat_scores, *flat_args, kv_token_indptr=caches["kv_token_indptr"])
    glm.synchronize()

    assert torch.equal(flat_scores, paged_scores)


def _run_topk(glm, mode, q, k_data, weights, caches, qo_indptr, topk, max_kv,
              causal, kv_token_indptr=None):
    total_q, n_heads, head_dim = q.shape
    page_size = caches["paged"].shape[1]
    scale = head_dim ** -0.5
    out = torch.full((total_q, topk), -2, dtype=torch.int32, device=q.device)
    out_scores = torch.full((total_q, topk), float("-inf"), dtype=torch.bfloat16, device=q.device)
    scores = torch.empty(total_q, max_kv, dtype=torch.bfloat16, device=q.device)
    row_len = torch.empty(total_q, dtype=torch.int32, device=q.device)
    meta = torch.empty(total_q, 4, dtype=torch.int32, device=q.device)
    num_splits = max(1, (max_kv + 255) // 256)
    common = dict(kv_token_indptr=kv_token_indptr)

    if mode == "v2":
        hist = torch.empty(total_q, 1056, dtype=torch.int32, device=q.device)
        glm.indexer_score_topk_v2(
            out, out_scores, q, k_data, weights,
            caches["page_indices"], caches["page_indptr"], caches["last_page_len"], qo_indptr,
            scale, total_q, n_heads, head_dim, page_size, topk, causal,
            scores, row_len, hist, meta, max_kv, num_splits, **common,
        )
    else:
        coarse_hist = torch.empty(total_q, 1024, dtype=torch.int32, device=q.device)
        fine_hist = torch.empty(total_q, 64, dtype=torch.int32, device=q.device)
        glm.indexer_score_topk_prefill(
            out, out_scores, q, k_data, weights,
            caches["page_indices"], caches["page_indptr"], caches["last_page_len"], qo_indptr,
            scale, total_q, n_heads, head_dim, page_size, topk, causal,
            scores, row_len, max_kv, coarse_hist, fine_hist, meta, num_splits, **common,
        )
    return out, out_scores, row_len


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["v2", "prefill"])
def test_flat_topk_matches_paged(glm, device, mode):
    seq_lens = [72, 35]
    page_size = 32
    n_heads, head_dim = 4, 128
    caches = _make_caches(device, seq_lens, page_size, head_dim, seed=123)
    total_q = sum(seq_lens)
    max_kv = max(seq_lens)
    topk = 16
    torch.manual_seed(456)
    q = torch.randn(total_q, n_heads, head_dim, dtype=torch.bfloat16, device=device)
    weights = torch.randn(total_q, n_heads, dtype=torch.bfloat16, device=device)
    qo_indptr = torch.tensor([0, seq_lens[0], total_q], dtype=torch.int32, device=device)

    paged = _run_topk(glm, mode, q, caches["paged"], weights, caches, qo_indptr,
                      topk, max_kv, True)
    flat = _run_topk(glm, mode, q, caches["flat"], weights, caches, qo_indptr,
                     topk, max_kv, True, caches["kv_token_indptr"])
    glm.synchronize()

    assert torch.equal(flat[0], paged[0])
    assert torch.equal(flat[1], paged[1])
    assert torch.equal(flat[2], paged[2])
