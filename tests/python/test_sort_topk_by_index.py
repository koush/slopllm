"""Sort a top-k row ascending by index, -1 padding last.

The CP merge concatenates per-shard top-k lists rank-major, so it emits
positions ordered by (P % W, P / W) rather than by global position P. Whenever
the merge and the replicated path pick the same set -- every context up to topk
-- that ordering is the only difference, and it is enough to move the logits,
because topk_to_slots preserves order and sparse MLA sums candidate tiles in
slot order. This kernel restores ascending order so the two paths agree.
"""
import numpy as np
import pytest
import torch
from helpers import GlmOps  # noqa: F401


def _run(glm, idx_np, val_np, device):
    batch, topk = idx_np.shape
    idx = torch.from_numpy(idx_np).to(device)
    val = torch.from_numpy(val_np).to(torch.bfloat16).to(device)
    glm.sort_topk_by_index(idx, val, batch, topk)
    glm.synchronize()
    return idx.cpu().numpy(), val.float().cpu().numpy()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("topk", [8, 64, 2048, 4096])
@pytest.mark.parametrize("n_valid_frac", [1.0, 0.5, 0.05, 0.0])
def test_sort_ascending_padding_last(glm, device, topk, n_valid_frac):
    batch = 3
    rng = np.random.default_rng(topk + int(n_valid_frac * 100))
    n_valid = int(topk * n_valid_frac)
    idx = np.full((batch, topk), -1, dtype=np.int32)
    val = np.full((batch, topk), -np.inf, dtype=np.float32)
    for b in range(batch):
        # Distinct positions in rank-major-ish scrambled order, like the merge emits.
        chosen = rng.choice(100000, size=n_valid, replace=False).astype(np.int32)
        idx[b, :n_valid] = chosen
        val[b, :n_valid] = (chosen % 37).astype(np.float32)   # payload tied to the key

    got_idx, got_val = _run(glm, idx.copy(), val.copy(), device)

    for b in range(batch):
        want = np.sort(idx[b][idx[b] >= 0])
        assert np.array_equal(got_idx[b, :n_valid], want), f"batch {b}: not ascending"
        assert (got_idx[b, n_valid:] == -1).all(), f"batch {b}: padding not parked at end"
        # Payload must travel with its key.
        assert np.array_equal(got_val[b, :n_valid], (want % 37).astype(np.float32)), \
            f"batch {b}: value/index pairing broken by the sort"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sort_is_idempotent(glm, device):
    batch, topk = 2, 2048
    rng = np.random.default_rng(0)
    idx = np.full((batch, topk), -1, dtype=np.int32)
    val = np.full((batch, topk), -np.inf, dtype=np.float32)
    n = 1500
    for b in range(batch):
        c = rng.choice(50000, size=n, replace=False).astype(np.int32)
        idx[b, :n] = c
        val[b, :n] = (c % 13).astype(np.float32)
    once_i, once_v = _run(glm, idx.copy(), val.copy(), device)
    twice_i, twice_v = _run(glm, once_i.copy(), once_v.astype(np.float32), device)
    assert np.array_equal(once_i, twice_i)
    assert np.array_equal(once_v, twice_v)
