"""Batched query slices retain global sequence, causal, and mask coordinates."""

import pytest
import torch

from test_indexer_flat import _make_caches
from test_indexer_cp_remap import _build_window_mask


@pytest.mark.parametrize("kernel", ["v2", "prefill"])
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_batched_query_shards(glm, device, monkeypatch, kernel, fp8, flat, masked):
    monkeypatch.setenv("GLM_INDEXER_DECODE_FP8_MMA", "1" if fp8 else "0")
    query_lengths = [4, 4] if kernel == "v2" else [35, 70, 29]
    kv_lengths = [length + 73 + i * 31 for i, length in enumerate(query_lengths)]
    caches = _make_caches(device, kv_lengths, 64, 128)
    total = sum(query_lengths)
    q = torch.randn(total, 32, 128, dtype=torch.bfloat16, device=device)
    weights = torch.randn(total, 32, dtype=torch.bfloat16, device=device)
    boundaries = [0]
    for length in query_lengths:
        boundaries.append(boundaries[-1] + length)
    qo_indptr = torch.tensor(boundaries, dtype=torch.int32, device=device)
    masks = [_build_window_mask(length, length, device) for length in query_lengths]
    mask_offsets = [0]
    for mask in masks:
        mask_offsets.append(mask_offsets[-1] + mask["data"].numel())
    common = dict(
        kv_token_indptr=caches["kv_token_indptr"] if flat else None,
        custom_mask=torch.cat([mask["data"] for mask in masks]) if masked else None,
        mask_indptr=torch.tensor(mask_offsets, dtype=torch.int32, device=device) if masked else None,
        mask_kv_len=torch.tensor(query_lengths, dtype=torch.int32, device=device) if masked else None,
    )
    max_kv, topk = max(kv_lengths), 23

    def run(start, count):
        indices = torch.full((count, topk), -2, dtype=torch.int32, device=device)
        values = torch.full((count, topk), -torch.inf, dtype=torch.bfloat16, device=device)
        scores = torch.full((count, max_kv), 8192, dtype=torch.bfloat16, device=device)
        lengths = torch.full((count,), -1, dtype=torch.int32, device=device)
        meta = torch.empty(count, 4, dtype=torch.int32, device=device)
        args = (indices, values, q[start:start + count],
                caches["flat" if flat else "paged"],
                caches["flat_scales" if flat else "paged_scales"], weights[start:start + count],
                caches["page_indices"], caches["page_indptr"], caches["last_page_len"],
                qo_indptr, 128 ** -0.5, count, 32, 128, 64, topk, True)
        if kernel == "v2":
            hist = torch.empty(count, 1056, dtype=torch.int32, device=device)
            glm.indexer_score_topk_v2(*args, scores, lengths, hist, meta, max_kv, 1,
                                     q_global_start=start, **common)
        else:
            coarse = torch.empty(count, 1024, dtype=torch.int32, device=device)
            fine = torch.empty(count, 64, dtype=torch.int32, device=device)
            glm.indexer_score_topk_prefill(*args, scores, lengths, max_kv, coarse, fine, meta, 1,
                                          q_global_start=start, **common)
        glm.synchronize()
        return indices, values, scores, lengths

    full = run(0, total)
    # M=8/TP8 ownership, plus a shard spanning the sequence boundary. Larger
    # slices exercise tiled kernels starting/ending inside different sequences.
    slices = [(i, 1) for i in range(total)] + [(2, 4)] if kernel == "v2" else [(0, 20), (20, 65), (85, 49)]
    for start, count in slices:
        indices, values, scores, lengths = run(start, count)
        for local in range(count):
            global_row = start + local
            seq = next(i for i in range(len(query_lengths)) if global_row < boundaries[i + 1])
            expected_length = kv_lengths[seq] - query_lengths[seq] + global_row - boundaries[seq] + 1
            assert lengths[local].item() == expected_length
            torch.testing.assert_close(scores[local, :expected_length],
                                       full[2][global_row, :expected_length], rtol=0, atol=0)
            # Atomic gather may reorder the same selected entries.
            selected = indices[local].long()
            assert selected.unique().numel() == topk
            assert torch.all((selected >= 0) & (selected < expected_length))
            torch.testing.assert_close(values[local], scores[local, selected], rtol=0, atol=0)
            torch.testing.assert_close(values[local].sort().values,
                                       full[1][global_row].sort().values, rtol=0, atol=0)
