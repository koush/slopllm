"""Shared helpers for indexer_score_topk v2/prefill tests.

The original v1 fused topk kernel was removed; these setup helpers remain
for use by test_indexer_score_topk_v2.py and test_indexer_score_topk_prefill.py.
"""

import torch
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
