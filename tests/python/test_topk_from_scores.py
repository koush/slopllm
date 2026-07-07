"""Exact large-K top-K selector over non-negative bf16 scores (histogram).

Validates glm_topk_from_scores against the exact top-K definition (under bf16
ties) and measures throughput at long context. This is the selection half of the
multi-block indexer rewrite; scores are fed directly so the selector is tested
in isolation.
"""
import time
import numpy as np
import pytest
import torch
from helpers import GlmOps  # noqa: F401

NBUCKET = 65536


def _select(glm, scores_bf16, N_per_row, topk, num_splits):
    batch, stride = scores_bf16.shape
    dev = scores_bf16.device
    out = torch.full((batch, topk), -2, dtype=torch.int32, device=dev)
    hist = torch.empty(batch, NBUCKET, dtype=torch.int32, device=dev)
    meta = torch.empty(batch, 4, dtype=torch.int32, device=dev)
    row_len = None
    if N_per_row is not None:
        row_len = torch.tensor(N_per_row, dtype=torch.int32, device=dev)
    glm.topk_from_scores(out, scores_bf16, row_len, hist, meta,
                         batch, stride, topk, num_splits)
    glm.synchronize()
    return out


def _check_exact_topk(sel, scores_row, N, topk):
    """sel: 1D int32 [topk] selected indices. scores_row: fp32 [stride]."""
    sel = sel.cpu().numpy()
    s = scores_row.float().cpu().numpy()[:N]
    assert (sel >= 0).all() and (sel < N).all(), f"out-of-range indices: {sel[(sel<0)|(sel>=N)][:5]}"
    assert len(np.unique(sel)) == topk, f"duplicates: {topk - len(np.unique(sel))}"
    kth = np.sort(s)[::-1][topk - 1]           # the K-th largest value = threshold
    sel_scores = s[sel]
    assert sel_scores.min() >= kth - 1e-9, f"selected below threshold: {sel_scores.min()} < {kth}"
    # every strictly-greater-than-threshold element must be selected
    above = np.where(s > kth + 1e-9)[0]
    assert set(above.tolist()).issubset(set(sel.tolist())), \
        f"missed {len(set(above.tolist()) - set(sel.tolist()))} elements above threshold"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("N", [4096, 65536, 200000])
@pytest.mark.parametrize("batch", [1, 4])
def test_topk_from_scores_exact(glm, device, N, batch):
    topk = 2048
    torch.manual_seed(N + batch)
    # non-negative (ReLU'd) bf16 scores, like the indexer produces
    scores = torch.rand(batch, N, device=device).relu().to(torch.bfloat16)
    num_splits = min(256, (N + 4095) // 4096)
    sel = _select(glm, scores, None, topk, num_splits)
    for b in range(batch):
        _check_exact_topk(sel[b], scores[b], N, topk)
    print(f"\n[N={N} batch={batch}] exact top-{topk} OK")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_topk_from_scores_perf(glm, device):
    """Throughput at 200k context, batch 1 (the decode-critical case)."""
    N, topk, batch = 200000, 2048, 1
    torch.manual_seed(0)
    scores = torch.rand(batch, N, device=device).relu().to(torch.bfloat16)
    num_splits = 256
    out = torch.full((batch, topk), -2, dtype=torch.int32, device=device)
    hist = torch.empty(batch, NBUCKET, dtype=torch.int32, device=device)
    meta = torch.empty(batch, 4, dtype=torch.int32, device=device)
    def call():
        glm.topk_from_scores(out, scores, None, hist, meta, batch, N, topk, num_splits)
    for _ in range(3):
        call()
    glm.synchronize()
    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        call()
    glm.synchronize()
    us = (time.perf_counter() - t0) / iters * 1e6
    print(f"\n[perf] select top-{topk} of {N}: {us:.1f} us/call (batch={batch})")
