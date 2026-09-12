"""Exact large-K top-K selector over bf16 scores (two-pass radix selection).

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

TOPK_SCRATCH_I32 = 1056


def _select(glm, scores_bf16, N_per_row, topk, num_splits):
    out, _ = _select_with_scores(glm, scores_bf16, N_per_row, topk, num_splits)
    return out


def _select_with_scores(glm, scores_bf16, N_per_row, topk, num_splits):
    batch, stride = scores_bf16.shape
    dev = scores_bf16.device
    out = torch.full((batch, topk), -2, dtype=torch.int32, device=dev)
    out_scores = torch.full((batch, topk), float('nan'), dtype=torch.bfloat16, device=dev)
    hist = torch.empty(batch, TOPK_SCRATCH_I32, dtype=torch.int32, device=dev)
    meta = torch.empty(batch, 4, dtype=torch.int32, device=dev)
    row_len = None
    if N_per_row is not None:
        row_len = torch.tensor(N_per_row, dtype=torch.int32, device=dev)
    glm.topk_from_scores(out, out_scores, scores_bf16, row_len, hist, meta,
                         batch, stride, topk, num_splits)
    glm.synchronize()
    return out, out_scores


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
@pytest.mark.parametrize("N,num_splits,batch", [
    (16384, 1, 3),   # single-block radix path
    (16384, 32, 4),  # CP merge: parallel radix below the general cutoff
    (16384, 32, 8),
    (4096, 8, 4),    # short full row: single-block radix
    (8192, 16, 16),  # enough scan work saved to use parallel radix
    (8192, 2, 32),   # two partitions do not amortize the extra launches here
    (12288, 2, 32),  # crossover with two partitions
    (16384, 32, 1),
    (16384, 32, 2),
    (16384, 32, 16),
    (16384, 16, 64),
    (32768, 1, 4),   # no histogram parallelism: keep a single-block scan
    (65536, 16, 3),  # multi-block, so the per-block prefix has to line up
    (200000, 256, 3),
])
def test_topk_from_scores_deterministic(glm, device, N, num_splits, batch):
    """Repeated selection over identical scores must be bit-identical.

    The selection feeds topk_to_slots, which compacts in input order, which sets
    the order sparse MLA accumulates candidate tiles in -- so any permutation
    here moves the logits. Under CP every rank runs this over identical data and
    the ranks must agree on the tie subset, or gatherTopkCkv leaves flat slots
    that no rank wrote. Ties at the threshold are the interesting part, so use a
    small value range (heavy bf16 tie bucket) plus a -inf pad block like the CP
    merge produces.
    """
    topk = 2048
    torch.manual_seed(N)
    scores = (torch.randint(0, 24, (batch, N), device=device).float() / 8).to(torch.bfloat16)
    scores[:, : N // 2] = float('-inf')   # pad block, as the per-rank merge emits

    ref_idx, ref_val = _select_with_scores(glm, scores, None, topk, num_splits)
    ref_idx, ref_val = ref_idx.clone(), ref_val.clone()
    if num_splits > 1:
        single_idx, single_val = _select_with_scores(glm, scores, None, topk, 1)
        assert torch.equal(ref_idx, single_idx)
        assert torch.equal(ref_val.view(torch.int16), single_val.view(torch.int16))
    for it in range(8):
        idx, val = _select_with_scores(glm, scores, None, topk, num_splits)
        assert torch.equal(idx, ref_idx), f"iteration {it}: indices differ from first run"
        assert torch.equal(val.view(torch.int16), ref_val.view(torch.int16)), \
            f"iteration {it}: scores differ from first run"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_topk_from_scores_pairs_scores_with_indices(glm, device):
    """out_scores[j] must be the score at out_idx[j] (the merge relies on it)."""
    N, topk, batch, num_splits = 65536, 2048, 2, 16
    torch.manual_seed(7)
    scores = torch.rand(batch, N, device=device).relu().to(torch.bfloat16)
    idx, val = _select_with_scores(glm, scores, None, topk, num_splits)
    for b in range(batch):
        expect = scores[b][idx[b].long()]
        assert torch.equal(val[b].view(torch.int16), expect.view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_topk_from_scores_long_stride_short_row(glm, device):
    """The long-row dispatch must bypass radix scans for short live rows."""
    N, topk = 65536, 2048
    torch.manual_seed(19)
    scores = torch.randn(2, N, device=device).to(torch.bfloat16)
    row_len = [731, N]
    idx, val = _select_with_scores(glm, scores, row_len, topk, 64)

    expected = torch.arange(row_len[0], device=device, dtype=torch.int32)
    assert torch.equal(idx[0, :row_len[0]], expected)
    assert torch.all(idx[0, row_len[0]:] == -1)
    assert torch.equal(
        val[0, :row_len[0]].view(torch.int16),
        scores[0, :row_len[0]].view(torch.int16),
    )
    _check_exact_topk(idx[1], scores[1], N, topk)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_topk_from_scores_perf(glm, device):
    """Throughput at 200k context, batch 1 (the decode-critical case)."""
    N, topk, batch = 200000, 2048, 1
    torch.manual_seed(0)
    scores = torch.rand(batch, N, device=device).relu().to(torch.bfloat16)
    num_splits = 256
    out = torch.full((batch, topk), -2, dtype=torch.int32, device=device)
    out_scores = torch.empty(batch, topk, dtype=torch.bfloat16, device=device)
    hist = torch.empty(batch, TOPK_SCRATCH_I32, dtype=torch.int32, device=device)
    meta = torch.empty(batch, 4, dtype=torch.int32, device=device)
    def call():
        glm.topk_from_scores(out, out_scores, scores, None, hist, meta,
                             batch, N, topk, num_splits)
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
