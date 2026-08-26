"""Validate the fused two-level prefill indexer against a torch reference.

The prefill kernel uses a 5-pass pipeline (score+coarse-hist, coarse-threshold,
score+fine-hist, fine-threshold, score+gather) with no materialized score buffer.
It must produce the same top-K selection as the reference: every strictly-above-
threshold position included, exactly topk unique valid indices (or all positions
if numValid <= topk).
"""
import pytest
import torch
from helpers import GlmOps  # noqa: F401
from test_indexer_score_topk import _setup_random_kv, _get_valid_kv_len
from test_indexer_score_topk_v2 import _ref_scores, _check


def _run_prefill(glm, s, topk, causal, device):
    (q, k_paged, k_scale_paged, weights, pil, pip, lpl, qoi, pit, pipt, lplt, qoit,
     scale, total_q, max_pages) = s
    n_heads, head_dim = q.shape[1], q.shape[2]
    page_size = k_paged.shape[1]

    max_kv = 0
    for i in range(len(lpl)):
        np_ = pip[i + 1] - pip[i]
        max_kv = max(max_kv, (np_ - 1) * page_size + lpl[i])

    num_splits = min(256, max(1, (max_kv + 255) // 256))
    out = torch.full((total_q, topk), -2, dtype=torch.int32, device=device)
    out_scores = torch.full((total_q, topk), float('-inf'), dtype=torch.bfloat16, device=device)
    scores = torch.empty(total_q, max_kv, dtype=torch.bfloat16, device=device)
    row_len = torch.empty(total_q, dtype=torch.int32, device=device)
    coarse_hist = torch.empty(total_q, 1024, dtype=torch.int32, device=device)
    fine_hist = torch.empty(total_q, 64, dtype=torch.int32, device=device)
    meta = torch.empty(total_q, 4, dtype=torch.int32, device=device)

    glm.indexer_score_topk_prefill(
        out, out_scores, q, k_paged, k_scale_paged, weights, pit, pipt, lplt, qoit, scale,
        total_q, n_heads, head_dim, page_size, topk, causal,
        scores, row_len, max_kv,
        coarse_hist, fine_hist, meta, num_splits,
    )
    glm.synchronize()
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("seq_lens,topk,causal", [
    ([5000], 2048, False),
    ([300], 64, False),
    ([40], 64, False),
    ([200, 90], 64, True),
    ([128], 32, True),
    ([16], 32, True),
    ([1000, 500, 250], 128, True),
    ([64, 64, 64, 64], 32, False),
])
def test_prefill_matches_reference(glm, device, seq_lens, topk, causal):
    n_heads, head_dim, page_size = 32, 128, 64
    s = _setup_random_kv(len(seq_lens), seq_lens, n_heads, head_dim, page_size, device)
    (q, k_paged, k_scale_paged, weights, pil, pip, lpl, qoi, pit, pipt, lplt, qoit,
     scale, total_q, max_pages) = s
    out = _run_prefill(glm, s, topk, causal, device)

    for seq_idx, sl in enumerate(seq_lens):
        for qi in range(qoi[seq_idx + 1] - qoi[seq_idx]):
            t = qoi[seq_idx] + qi
            numValid = _get_valid_kv_len(seq_idx, qi, seq_lens, pip, lpl, page_size, qoi, causal)
            if numValid <= 0:
                assert torch.all(out[t] == -1), f"q{t}: expected all -1"
                continue
            page_start = pip[seq_idx]
            seq_page_indices = pil[page_start: pip[seq_idx + 1]]
            ref = _ref_scores(q, k_paged, k_scale_paged, weights, seq_page_indices,
                              pip, page_size, scale, numValid, t)
            _check(out[t], ref, numValid, topk)
    print(f"\n[seq_lens={seq_lens} topk={topk} causal={causal}] prefill OK")
