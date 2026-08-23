"""Validate that cpWorldSize/cpRank params remap indexer top-k positions correctly.

The kernel applies cp_remap(pos) = pos * cpWorldSize + cpRank when cpWorldSize > 0,
and identity (pos) when cpWorldSize == 0. We verify by running with CP params,
reverse-mapping the output indices back to local positions, and checking them
against the same reference scoring used by the non-CP tests.
"""
import numpy as np
import pytest
import torch
from helpers import GlmOps, pack_indexer_k  # noqa: F401
from test_indexer_score_topk import _setup_random_kv, _get_valid_kv_len
from test_indexer_score_topk_v2 import _ref_scores, _check, _run_v2
from test_indexer_score_topk_prefill import _run_prefill

NBUCKET = 65536


def _run_v2_cp(glm, s, topk, causal, device, cp_world_size, cp_rank):
    (q, k_paged, weights, pil, pip, lpl, qoi, pit, pipt, lplt, qoit,
     scale, total_q, max_pages) = s
    max_kv = 0
    for i in range(len(lpl)):
        np_ = pip[i + 1] - pip[i]
        max_kv = max(max_kv, (np_ - 1) * k_paged.shape[1] + lpl[i])
    out = torch.full((total_q, topk), -2, dtype=torch.int32, device=device)
    out_scores = torch.full((total_q, topk), float('-inf'), dtype=torch.bfloat16, device=device)
    scores = torch.zeros(total_q, max_kv, dtype=torch.bfloat16, device=device)
    row_len = torch.zeros(total_q, dtype=torch.int32, device=device)
    hist = torch.empty(total_q, NBUCKET, dtype=torch.int32, device=device)
    meta = torch.empty(total_q, 4, dtype=torch.int32, device=device)
    n_heads, head_dim = q.shape[1], q.shape[2]
    num_splits = min(256, max(1, (max_kv + 255) // 256))
    glm.indexer_score_topk_v2(
        out, out_scores, q, k_paged, weights, pit, pipt, lplt, qoit, scale,
        total_q, n_heads, head_dim, k_paged.shape[1], topk, causal,
        scores, row_len, hist, meta, max_kv, num_splits,
        cp_world_size=cp_world_size, cp_rank=cp_rank)
    glm.synchronize()
    return out


def _run_prefill_cp(glm, s, topk, causal, device, cp_world_size, cp_rank):
    (q, k_paged, weights, pil, pip, lpl, qoi, pit, pipt, lplt, qoit,
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
        out, out_scores, q, k_paged, weights, pit, pipt, lplt, qoit, scale,
        total_q, n_heads, head_dim, page_size, topk, causal,
        scores, row_len, max_kv,
        coarse_hist, fine_hist, meta, num_splits,
        cp_world_size=cp_world_size, cp_rank=cp_rank)
    glm.synchronize()
    return out


def _reverse_map(out_cp, cp_world_size, cp_rank):
    """Reverse-map CP indices: (idx - cpRank) / cpWorldSize. -1 stays -1."""
    out = out_cp.cpu().numpy().copy()
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            if out[i, j] >= 0:
                out[i, j] = (out[i, j] - cp_rank) // cp_world_size
    return torch.tensor(out, dtype=torch.int32, device=out_cp.device)


def _check_cp_remap(out_cp, ref_row, numValid, topk, cp_world_size, cp_rank, q_idx):
    """Reverse-map CP output and validate against reference scoring."""
    local = _reverse_map(out_cp, cp_world_size, cp_rank)
    _check(local[q_idx], ref_row, numValid, topk)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("cp_world_size,cp_rank", [
    (4, 0),
    (4, 1),
    (4, 3),
    (8, 5),
])
@pytest.mark.parametrize("seq_lens,topk,causal", [
    ([300], 64, False),      # histogram selection (kvLen > topk)
    ([40], 64, False),       # identity (kvLen <= topk)
    ([200, 90], 64, True),   # causal, multi-seq
])
def test_v2_cp_remap(glm, device, seq_lens, topk, causal, cp_world_size, cp_rank):
    n_heads, head_dim, page_size = 32, 128, 64
    s = _setup_random_kv(len(seq_lens), seq_lens, n_heads, head_dim, page_size, device)
    (q, k_paged, weights, pil, pip, lpl, qoi, pit, pipt, lplt, qoit,
     scale, total_q, max_pages) = s
    out_cp = _run_v2_cp(glm, s, topk, causal, device, cp_world_size, cp_rank)

    for seq_idx, sl in enumerate(seq_lens):
        for qi in range(qoi[seq_idx + 1] - qoi[seq_idx]):
            t = qoi[seq_idx] + qi
            numValid = _get_valid_kv_len(seq_idx, qi, seq_lens, pip, lpl, page_size, qoi, causal)
            page_start = pip[seq_idx]
            seq_page_indices = pil[page_start: pip[seq_idx + 1]]
            ref = _ref_scores(q, k_paged, weights, seq_page_indices, pip, page_size, scale, numValid, t)
            _check_cp_remap(out_cp, ref, numValid, topk, cp_world_size, cp_rank, t)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("cp_world_size,cp_rank", [
    (4, 0),
    (4, 2),
    (8, 7),
])
@pytest.mark.parametrize("seq_lens,topk,causal", [
    ([300], 64, False),      # histogram selection (kvLen > topk)
    ([40], 64, False),       # identity (kvLen <= topk)
    ([128], 32, True),       # causal
    ([64, 64, 64, 64], 32, False),  # multi-seq
])
def test_prefill_cp_remap(glm, device, seq_lens, topk, causal, cp_world_size, cp_rank):
    n_heads, head_dim, page_size = 32, 128, 64
    s = _setup_random_kv(len(seq_lens), seq_lens, n_heads, head_dim, page_size, device)
    (q, k_paged, weights, pil, pip, lpl, qoi, pit, pipt, lplt, qoit,
     scale, total_q, max_pages) = s
    out_cp = _run_prefill_cp(glm, s, topk, causal, device, cp_world_size, cp_rank)

    for seq_idx, sl in enumerate(seq_lens):
        for qi in range(qoi[seq_idx + 1] - qoi[seq_idx]):
            t = qoi[seq_idx] + qi
            numValid = _get_valid_kv_len(seq_idx, qi, seq_lens, pip, lpl, page_size, qoi, causal)
            if numValid <= 0:
                local = _reverse_map(out_cp, cp_world_size, cp_rank)
                assert torch.all(local[t] == -1), f"q{t}: expected all -1"
                continue
            page_start = pip[seq_idx]
            seq_page_indices = pil[page_start: pip[seq_idx + 1]]
            ref = _ref_scores(q, k_paged, weights, seq_page_indices, pip, page_size, scale, numValid, t)
            _check_cp_remap(out_cp, ref, numValid, topk, cp_world_size, cp_rank, t)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("seq_lens,topk", [
    ([300], 64),
    ([40], 64),
])
def test_cp_world_size_zero_is_identity(glm, device, seq_lens, topk):
    """cpWorldSize=0 with any cpRank must produce the same valid set as defaults.

    Both runs use the same seed, so the only difference is cpWorldSize/cpRank.
    Since the gather kernel uses atomics, we compare sorted sets of valid indices.
    Ties at the threshold boundary may resolve differently, so we check that the
    sets match up to the existing score tolerance.
    """
    n_heads, head_dim, page_size = 32, 128, 64
    s = _setup_random_kv(len(seq_lens), seq_lens, n_heads, head_dim, page_size, device)
    (q, k_paged, weights, pil, pip, lpl, qoi, pit, pipt, lplt, qoit,
     scale, total_q, max_pages) = s
    out_nocp, _ = _run_v2(glm, s, topk, False, device)
    out_zero = _run_v2_cp(glm, s, topk, False, device, 0, 3)

    for seq_idx, sl in enumerate(seq_lens):
        for qi in range(qoi[seq_idx + 1] - qoi[seq_idx]):
            t = qoi[seq_idx] + qi
            numValid = _get_valid_kv_len(seq_idx, qi, seq_lens, pip, lpl, page_size, qoi, False)
            page_start = pip[seq_idx]
            seq_page_indices = pil[page_start: pip[seq_idx + 1]]
            ref = _ref_scores(q, k_paged, weights, seq_page_indices, pip, page_size, scale, numValid, t)
            # Both should pass the same reference check
            _check(out_nocp[t], ref, numValid, topk)
            _check(out_zero[t], ref, numValid, topk)


# ---------------------------------------------------------------------------
# CP causal limit: the shard-local causal bound must be derived from the GLOBAL
# sequence position, and every consumer of it (rowLen, the tile prune, the
# epilogue's per-element check) must agree on one value.
#
# The tests above pass cp_world_size/cp_rank but no global_last_page_len, so the
# kernels take the non-CP branch and this path went uncovered. When the prefill
# epilogue used its own local-kvLen formula it stopped short of what rowLen
# advertised: the top-k then scanned score slots the MMA pass never wrote and
# selected whatever the recycled buffer held. POISON makes that visible.
# ---------------------------------------------------------------------------

# Far above any real indexer score, so an unwritten slot wins the top-k. Must be
# exactly representable in bf16 (8 mantissa bits) or the readback compares against
# a value the buffer never holds -- 1e4, for instance, stores as 9984.
POISON = 8192.0


def _setup_cp_shard(L_global, W, r, n_queries, n_heads, head_dim, local_page_size,
                    device, seed=1234):
    """Build the paged-KV view one CP rank sees: every W-th token of an L_global sequence.

    Local position p holds global position p*W + r. The page table is global (a
    logical page is local_page_size*W tokens) while the shard's physical pages
    hold local_page_size tokens each.
    """
    torch.manual_seed(seed)
    full_page = local_page_size * W
    num_pages = (L_global + full_page - 1) // full_page
    global_lpl = L_global - (num_pages - 1) * full_page
    L_local = len(range(r, L_global, W))
    local_lpl = L_local - (num_pages - 1) * local_page_size
    # Skip the degenerate slice where this rank owns none of the final logical
    # page; last_page_len cannot express that and it is not what we're testing.
    assert 1 <= local_lpl <= local_page_size, f"unsupported slice: local_lpl={local_lpl}"

    max_pages = num_pages + 4
    k_paged = pack_indexer_k(torch.randn(
        max_pages, local_page_size, head_dim, dtype=torch.bfloat16, device=device))
    q = torch.randn(n_queries, n_heads, head_dim, dtype=torch.bfloat16, device=device)
    weights = torch.randn(n_queries, n_heads, dtype=torch.bfloat16, device=device)

    t = lambda xs: torch.tensor(xs, dtype=torch.int32, device=device)
    return dict(
        q=q, k_paged=k_paged, weights=weights,
        page_indices=t(list(range(num_pages))), page_indptr=t([0, num_pages]),
        last_page_len=t([local_lpl]), global_last_page_len=t([global_lpl]),
        qo_indptr=t([0, n_queries]),
        scale=head_dim ** -0.5, L_local=L_local, L_global=L_global,
        n_queries=n_queries, W=W, r=r, page_size=local_page_size,
    )


def _expected_num_valid(s, qi):
    """Local causal limit + 1, computed the way the kernel must: globally, then mapped back.

    FLOOR division, not truncation toward zero. When global_limit < cp_rank this
    rank owns nothing at or below the limit (its first token is global position
    cp_rank), so the count is 0. Truncation yields 0 for the limit instead, which
    hands every rank its local position 0 -- query q then sees global positions
    0..W-1 rather than just 0, on the first W-1 rows of every sequence.
    """
    global_prefix = max(0, s["L_global"] - s["n_queries"])
    global_limit = global_prefix + qi                      # causal
    if global_limit < s["r"]:
        return 0
    local_limit = (global_limit - s["r"]) // s["W"]
    return min(local_limit, s["L_local"] - 1) + 1


def _run_cp_shard(glm, s, topk, device, kernel, mask=None):
    max_kv = s["L_local"]
    total_q = s["n_queries"]
    out = torch.full((total_q, topk), -2, dtype=torch.int32, device=device)
    out_scores = torch.full((total_q, topk), float('-inf'), dtype=torch.bfloat16, device=device)
    scores = torch.full((total_q, max_kv), POISON, dtype=torch.bfloat16, device=device)
    row_len = torch.zeros(total_q, dtype=torch.int32, device=device)
    meta = torch.empty(total_q, 4, dtype=torch.int32, device=device)
    num_splits = min(256, max(1, (max_kv + 255) // 256))
    common = dict(custom_mask=None if mask is None else mask["data"],
                  mask_indptr=None if mask is None else mask["indptr"],
                  mask_kv_len=None if mask is None else mask["kv_len"],
                  cp_world_size=s["W"], cp_rank=s["r"],
                  global_last_page_len=s["global_last_page_len"])
    args = (out, out_scores, s["q"], s["k_paged"], s["weights"],
            s["page_indices"], s["page_indptr"], s["last_page_len"], s["qo_indptr"],
            s["scale"], total_q, s["q"].shape[1], s["q"].shape[2], s["page_size"], topk, True)
    if kernel == "v2":
        hist = torch.empty(total_q, NBUCKET, dtype=torch.int32, device=device)
        glm.indexer_score_topk_v2(*args, scores, row_len, hist, meta, max_kv, num_splits, **common)
    else:
        coarse = torch.empty(total_q, 1024, dtype=torch.int32, device=device)
        fine = torch.empty(total_q, 64, dtype=torch.int32, device=device)
        glm.indexer_score_topk_prefill(*args, scores, row_len, max_kv,
                                       coarse, fine, meta, num_splits, **common)
    glm.synchronize()
    return out, out_scores, row_len, scores


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kernel", ["prefill", "v2"])
@pytest.mark.parametrize("cp_world_size,cp_rank", [(4, 0), (4, 3), (8, 5), (8, 0)])
@pytest.mark.parametrize("L_global,n_queries", [
    (1200, 128),   # chunked prefill onto a long prefix
    (13, 13),      # full prefill of a short prompt: early rows have limit < cp_rank
    (64, 64),
])
def test_cp_causal_limit_never_selects_unwritten_scores(glm, device, kernel,
                                                        cp_world_size, cp_rank,
                                                        L_global, n_queries):
    n_heads, head_dim, page_size = 32, 128, 64
    topk = 64
    s = _setup_cp_shard(L_global, cp_world_size, cp_rank, n_queries,
                        n_heads, head_dim, page_size, device)
    out, out_scores, row_len, _ = _run_cp_shard(glm, s, topk, device, kernel)

    row_len_h = row_len.cpu().numpy()
    idx_h = out.cpu().numpy()
    scores_h = out_scores.float().cpu().numpy()

    for qi in range(n_queries):
        want = _expected_num_valid(s, qi)
        assert row_len_h[qi] == want, \
            f"q{qi}: rowLen={row_len_h[qi]} expected {want} (local causal limit from global position)"

        sel = idx_h[qi]
        valid = sel[sel >= 0]
        # Indices are cp-remapped to global; map back to this shard's local space.
        local = (valid - cp_rank) // cp_world_size
        assert ((valid - cp_rank) % cp_world_size == 0).all(), \
            f"q{qi}: selected a position this rank does not own"
        assert (local < want).all(), \
            f"q{qi}: selected local pos {local.max()} past causal limit {want - 1}"
        # The decisive check: a slot the score pass skipped still holds POISON.
        assert not (scores_h[qi][sel >= 0] >= POISON).any(), \
            f"q{qi}: top-k selected a score slot the kernel never wrote"


# ---------------------------------------------------------------------------
# CP custom (tree) mask: the mask bitmap is built over GLOBAL kv columns, so the
# lookup must convert this shard's local position to its global one. Indexing it
# with the local position gates entirely the wrong tokens -- and since MTP
# verification is the only caller that passes a CausalCustom mask, that shows up
# as a drop in draft acceptance rather than as visibly broken output.
# ---------------------------------------------------------------------------

MASK_STRIDE = 3   # query q may see global window column c iff c % 3 == q % 3


def _build_window_mask(n_queries, mask_kv_len, device):
    total_bits = n_queries * mask_kv_len
    data = np.zeros((total_bits + 7) // 8, dtype=np.uint8)
    for q in range(n_queries):
        for c in range(mask_kv_len):
            if c % MASK_STRIDE == q % MASK_STRIDE:      # 1 = visible
                bit = q * mask_kv_len + c
                data[bit >> 3] |= np.uint8(1 << (bit & 7))
    return dict(
        data=torch.from_numpy(data).to(device),
        indptr=torch.tensor([0, data.size], dtype=torch.int32, device=device),
        kv_len=torch.tensor([mask_kv_len], dtype=torch.int32, device=device),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kernel", ["prefill", "v2"])
@pytest.mark.parametrize("cp_world_size,cp_rank", [(4, 0), (4, 3), (8, 5), (8, 0)])
def test_cp_custom_mask_uses_global_columns(glm, device, kernel, cp_world_size, cp_rank):
    n_heads, head_dim, page_size = 32, 128, 64
    L_global, n_queries, topk, mask_kv_len = 1200, 128, 64, 64
    W, r = cp_world_size, cp_rank
    s = _setup_cp_shard(L_global, W, r, n_queries, n_heads, head_dim, page_size, device)
    mask = _build_window_mask(n_queries, mask_kv_len, device)
    _, _, row_len, scores = _run_cp_shard(glm, s, topk, device, kernel, mask=mask)

    rl = row_len.cpu().numpy()
    sc = scores.float().cpu().numpy()
    window_start = L_global - mask_kv_len
    checked = 0
    for qi in range(n_queries):
        for pos in range(rl[qi]):
            P = pos * W + r                       # this shard's local pos -> global
            got_masked = np.isneginf(sc[qi, pos])
            if P < window_start:
                want_masked = False               # outside the window: always visible
            else:
                want_masked = ((P - window_start) % MASK_STRIDE) != (qi % MASK_STRIDE)
                checked += 1
            assert got_masked == want_masked, (
                f"q{qi} local pos {pos} (global {P}): masked={got_masked} "
                f"expected {want_masked} -- mask read at the wrong column")
    # Guard against a vacuous pass: the causal range must actually reach the window.
    assert checked > 0, "no in-window positions were exercised"
