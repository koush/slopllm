"""Tests for the fused indexer_score_topk kernel.

Compares the fused score+topk kernel against the separate indexer_score + torch.topk reference.
The fused kernel uses a min-heap internally, so output indices are unordered — we compare sets.
"""

import pytest
import torch
import ctypes
from helpers import GlmOps
from test_indexer import indexer_score_torch


def _setup_random_kv(B, seq_lens, n_heads, head_dim, page_size, device, seed=42):
    torch.manual_seed(seed)
    total_q = sum(seq_lens)
    num_pages_total = sum((s + page_size - 1) // page_size for s in seq_lens)
    max_pages = num_pages_total + 16

    k_paged = torch.randn(max_pages, page_size, head_dim, dtype=torch.bfloat16, device=device)
    q = torch.randn(total_q, n_heads, head_dim, dtype=torch.bfloat16, device=device)
    weights = torch.randn(total_q, n_heads, dtype=torch.bfloat16, device=device)
    scale = head_dim ** -0.5

    page_indices_list = []
    page_indptr = [0]
    last_page_len_list = []
    page_id = 0
    for s in seq_lens:
        num_pages_s = (s + page_size - 1) // page_size
        page_indices_list.extend(range(page_id, page_id + num_pages_s))
        page_id += num_pages_s
        page_indptr.append(page_indptr[-1] + num_pages_s)
        last_page_len_list.append(s - (num_pages_s - 1) * page_size)

    qo_indptr = [0]
    for s in seq_lens:
        qo_indptr.append(qo_indptr[-1] + s)

    page_indices_t = torch.tensor(page_indices_list, dtype=torch.int32, device=device)
    page_indptr_t = torch.tensor(page_indptr, dtype=torch.int32, device=device)
    last_page_len_t = torch.tensor(last_page_len_list, dtype=torch.int32, device=device)
    qo_indptr_t = torch.tensor(qo_indptr, dtype=torch.int32, device=device)

    return (q, k_paged, weights, page_indices_list, page_indptr, last_page_len_list,
            qo_indptr, page_indices_t, page_indptr_t, last_page_len_t, qo_indptr_t,
            scale, total_q, max_pages)


def _get_valid_kv_len(seq_idx, qi, seq_lens, page_indptr, last_page_len_list,
                      page_size, qo_indptr, causal):
    q_start, q_end = qo_indptr[seq_idx], qo_indptr[seq_idx + 1]
    num_queries = q_end - q_start
    page_start, page_end = page_indptr[seq_idx], page_indptr[seq_idx + 1]
    num_pages = page_end - page_start
    kv_len = (num_pages - 1) * page_size + last_page_len_list[seq_idx] if num_pages > 0 else 0
    prefix_len = max(0, kv_len - num_queries)
    causal_limit = prefix_len + qi if causal else kv_len - 1
    return min(kv_len, causal_limit + 1)


def _run_fused_topk_test(glm, device, B, seq_lens, n_heads, head_dim, page_size, topk, causal, seed=42):
    (q, k_paged, weights, page_indices_list, page_indptr, last_page_len_list,
     qo_indptr, page_indices_t, page_indptr_t, last_page_len_t, qo_indptr_t,
     scale, total_q, max_pages) = _setup_random_kv(B, seq_lens, n_heads, head_dim, page_size, device, seed)

    max_kv_len = max(seq_lens)

    # Reference: indexer_score + torch.topk
    ref_scores = indexer_score_torch(
        q, k_paged, weights, page_indices_list, page_indptr,
        last_page_len_list, qo_indptr, scale, page_size, max_kv_len, causal)

    # CUDA fused kernel
    cuda_indices = torch.full((total_q, topk), -1, dtype=torch.int32, device=device)
    glm.indexer_score_topk(cuda_indices, q, k_paged, weights,
                           page_indices_t, page_indptr_t, last_page_len_t, qo_indptr_t,
                           scale, total_q, n_heads, head_dim, page_size, topk, causal)

    # Compare: for each query, verify the CUDA topk indices have scores matching
    # the reference top-k threshold. Ties in bfloat16 may cause different index
    # selection, so we compare scores rather than exact index sets.
    for seq_idx in range(B):
        q_start, q_end = qo_indptr[seq_idx], qo_indptr[seq_idx + 1]
        for qi in range(q_end - q_start):
            q_idx = q_start + qi
            valid_k = _get_valid_kv_len(seq_idx, qi, seq_lens, page_indptr,
                                        last_page_len_list, page_size, qo_indptr, causal)
            if valid_k <= 0:
                assert torch.all(cuda_indices[q_idx] == -1), f"q{q_idx}: expected all -1"
                continue

            effective_k = min(topk, valid_k)
            ref_scores_row = ref_scores[q_idx, :valid_k].float()
            ref_topk_vals, ref_topk_indices = torch.topk(ref_scores_row, effective_k)
            threshold = ref_topk_vals[-1].item()  # k-th highest score

            cuda_row = cuda_indices[q_idx].cpu().numpy()
            cuda_valid = [int(x) for x in cuda_row if x >= 0]

            assert len(cuda_valid) == effective_k, \
                f"q{q_idx}: expected {effective_k} valid, got {len(cuda_valid)}"

            if valid_k <= topk:
                assert set(cuda_valid) == set(range(valid_k)), \
                    f"q{q_idx} (kvLen={valid_k} <= topk={topk}): expected all positions"
            else:
                cuda_scores = ref_scores_row[cuda_valid]
                assert torch.all(cuda_scores >= threshold - 0.5), \
                    f"q{q_idx}: CUDA indices have scores below threshold {threshold:.4f}"
                ref_set = set(int(x) for x in ref_topk_indices)
                cuda_set = set(cuda_valid)
                if cuda_set != ref_set:
                    diff = cuda_set.symmetric_difference(ref_set)
                    for d in diff:
                        score = ref_scores_row[d].item()
                        assert abs(score - threshold) < 1.0, \
                            f"q{q_idx}: index {d} score {score:.4f} far from threshold {threshold:.4f}"


def test_fused_topk_single_seq_short(glm, device):
    _run_fused_topk_test(glm, device, B=1, seq_lens=[16], n_heads=4, head_dim=64,
                         page_size=8, topk=32, causal=True)


def test_fused_topk_single_seq_long(glm, device):
    _run_fused_topk_test(glm, device, B=1, seq_lens=[128], n_heads=4, head_dim=64,
                         page_size=8, topk=32, causal=True)


def test_fused_topk_multi_seq(glm, device):
    _run_fused_topk_test(glm, device, B=3, seq_lens=[16, 32, 8], n_heads=4, head_dim=64,
                         page_size=8, topk=16, causal=True)


def test_fused_topk_noncausal(glm, device):
    _run_fused_topk_test(glm, device, B=1, seq_lens=[64], n_heads=4, head_dim=64,
                         page_size=8, topk=32, causal=False)


def test_fused_topk_kvlen_equals_topk(glm, device):
    _run_fused_topk_test(glm, device, B=1, seq_lens=[32], n_heads=4, head_dim=64,
                         page_size=8, topk=32, causal=True)


def test_fused_topk_kvlen_less_than_topk(glm, device):
    _run_fused_topk_test(glm, device, B=2, seq_lens=[5, 10], n_heads=4, head_dim=64,
                         page_size=8, topk=64, causal=True)


def test_fused_topk_decode_pattern(glm, device):
    _run_fused_topk_test(glm, device, B=4, seq_lens=[1, 1, 1, 1], n_heads=4, head_dim=64,
                         page_size=8, topk=4, causal=False)


def test_fused_topk_uneven_pages(glm, device):
    _run_fused_topk_test(glm, device, B=2, seq_lens=[20, 33], n_heads=4, head_dim=64,
                         page_size=8, topk=16, causal=True)


def test_fused_topk_large_topk(glm, device):
    _run_fused_topk_test(glm, device, B=1, seq_lens=[256], n_heads=8, head_dim=64,
                         page_size=16, topk=128, causal=True)


def test_fused_topk_multi_page_noncausal(glm, device):
    _run_fused_topk_test(glm, device, B=2, seq_lens=[48, 64], n_heads=4, head_dim=64,
                         page_size=16, topk=32, causal=False)
