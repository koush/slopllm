"""Validate the multi-block v2 indexer (score buffer -> histogram top-K) against
a torch reference of the indexer scores, in both regimes:
  - kvLen  > topk : histogram selection
  - kvLen <= topk : identity short-circuit (all positions)

v2 must produce an exact top-K (every strictly-above-threshold position included,
exactly topk unique valid indices) — the selection is order-independent, so we
check the set, not the ordering.
"""
import numpy as np
import pytest
import torch
from helpers import GlmOps  # noqa: F401
from test_indexer_score_topk import _setup_random_kv, _get_valid_kv_len

NBUCKET = 65536


def _ref_scores(q, k_paged, weights, page_indices, page_indptr, page_size, scale, numValid, t):
    """bf16-rounded indexer score row for query t over its first numValid positions."""
    hd = q.shape[-1]
    # find seq for global query t
    # (page tables are per-seq; caller passes the right numValid)
    ks = []
    for j in range(numValid):
        pid = page_indices[j // page_size]
        ks.append(k_paged[pid, j % page_size])
    K = torch.stack(ks).float()                       # [numValid, hd]
    qh = q[t].float()                                 # [n_heads, hd]
    dot = qh @ K.T                                    # [n_heads, numValid]
    sc = torch.relu(dot * scale)                      # per-head ReLU
    w = weights[t].float().unsqueeze(1)               # [n_heads,1]
    row = (w * sc).sum(0)                             # [numValid]
    return row.to(torch.bfloat16).float().cpu().numpy()   # match kernel's bf16 round


def _run_v2(glm, s, topk, causal, device):
    (q, k_paged, weights, pil, pip, lpl, qoi, pit, pipt, lplt, qoit,
     scale, total_q, max_pages) = s
    max_kv = max(sum(lpl) + 0, k_paged.shape[1])
    max_kv = max((pip[i + 1] - pip[i]) for i in range(len(lpl)))  # placeholder
    # actual max kv across seqs:
    max_kv = 0
    for i in range(len(lpl)):
        np_ = pip[i + 1] - pip[i]
        max_kv = max(max_kv, (np_ - 1) * k_paged.shape[1] + lpl[i])
    out = torch.full((total_q, topk), -2, dtype=torch.int32, device=device)
    scores = torch.zeros(total_q, max_kv, dtype=torch.bfloat16, device=device)
    row_len = torch.zeros(total_q, dtype=torch.int32, device=device)
    hist = torch.empty(total_q, NBUCKET, dtype=torch.int32, device=device)
    meta = torch.empty(total_q, 4, dtype=torch.int32, device=device)
    n_heads, head_dim = q.shape[1], q.shape[2]
    num_splits = min(256, max(1, (max_kv + 255) // 256))
    glm.indexer_score_topk_v2(
        out, q, k_paged, weights, pit, pipt, lplt, qoit, scale,
        total_q, n_heads, head_dim, k_paged.shape[1], topk, causal,
        scores, row_len, hist, meta, max_kv, num_splits)
    glm.synchronize()
    return out, max_kv


def _check(out_row, ref_row, numValid, topk):
    sel = out_row.cpu().numpy()
    if numValid <= topk:
        expect = set(range(numValid))
        got = set(x for x in sel.tolist() if x >= 0)
        assert got == expect, f"identity mismatch: missing {expect - got}, extra {got - expect}"
        return
    sel = sel[sel >= 0]
    assert len(np.unique(sel)) == topk, f"expected {topk} unique, got {len(np.unique(sel))}"
    assert (sel < numValid).all()
    kth = np.sort(ref_row)[::-1][topk - 1]
    # The kernel computes scores via warp-shuffle reduction while the reference
    # uses torch.matmul — different float32 reduction orders can round to
    # adjacent bf16 values. Allow 1.0 tolerance for tie-break discrepancies.
    assert ref_row[sel].min() >= kth - 1.0, "selected below threshold"
    above = np.where(ref_row > kth + 1.0)[0]
    assert set(above.tolist()).issubset(set(sel.tolist())), "missed above-threshold positions"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("seq_lens,topk,causal", [
    ([5000], 2048, False),      # decode-like: 1 query, kvLen >> topk
    ([300], 64, False),         # kvLen > topk
    ([40], 64, False),          # kvLen <= topk -> identity
    ([200, 90], 64, True),      # prefill causal, mixed
])
def test_v2_matches_reference(glm, device, seq_lens, topk, causal):
    n_heads, head_dim, page_size = 32, 128, 64
    s = _setup_random_kv(len(seq_lens), seq_lens, n_heads, head_dim, page_size, device)
    (q, k_paged, weights, pil, pip, lpl, qoi, pit, pipt, lplt, qoit,
     scale, total_q, max_pages) = s
    out, max_kv = _run_v2(glm, s, topk, causal, device)

    for seq_idx, sl in enumerate(seq_lens):
        for qi in range(qoi[seq_idx + 1] - qoi[seq_idx]):
            t = qoi[seq_idx] + qi
            numValid = _get_valid_kv_len(seq_idx, qi, seq_lens, pip, lpl, page_size, qoi, causal)
            page_start = pip[seq_idx]
            seq_page_indices = pil[page_start: pip[seq_idx + 1]]
            ref = _ref_scores(q, k_paged, weights, seq_page_indices, pip, page_size, scale, numValid, t)
            _check(out[t], ref, numValid, topk)
    print(f"\n[seq_lens={seq_lens} topk={topk} causal={causal}] v2 exact OK")
