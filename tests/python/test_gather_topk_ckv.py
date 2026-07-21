"""Tests for glm_gather_topk_ckv — sparse topk-driven P2P CKV gather.

Verifies that the kernel:
  * reads BPT-byte tokens from this rank's local paged KV cache once per token,
  * writes those bytes to all N peer flat-format output buffers at
    flat_slot = topk_idx[entry] (i.e. the output flat slot given directly, in
    the same format topk_to_slots produces in its flat / cp_world_size==1 mode:
    flat_slot = kv_token_indptr[seq] + token_pos),
  * drops tokens not owned by this rank (recovered token_pos % cp_world_size
    != cp_rank),
  * leaves non-topk flat slots untouched (stale data preserved).

INPUT CONTRACT: the kernel takes flat slots as topk_idx — NOT raw KV
positions. Tests construct raw positions for readability and pre-flatten via
_to_flat_slots() before invoking the kernel, mirroring how the production
caller passes sharedSlots.value (the output of topkToSlots in flat mode)
directly.

The "cross-GPU" semantics are simulated locally by allocating N independent
peer flat buffers on the test GPU (per the user's note that 'pointers are
arbitrary' — no real P2P / multi-GPU needed for verification). For the full
multi-rank fan-out test we model each cp_rank's local KV cache independently
and run the kernel once per rank with all N flat pointers; afterwards every
peer buffer must contain the union of topk tokens copied from each rank.
"""

import numpy as np
import pytest
import torch

PAGE_SIZE = 64
KV_LORA_RANK = 512
NUM_TILES = KV_LORA_RANK // 128          # 4
PE_DIM = 64
BPT = KV_LORA_RANK + NUM_TILES * 4 + PE_DIM * 2   # 656



def _alloc_scratch(device, padded_kv_len, num_tokens, topk):
    """Dedup scratch, owned by the caller (mirrors what GlmOps allocates from the
    TS workspace). Returned as a dict of kwargs plus the tensors themselves, which
    must stay alive for the duration of the call."""
    bitmap_words = (padded_kv_len + 31) // 32
    max_entries = num_tokens * topk
    bitmap = torch.zeros(bitmap_words, dtype=torch.int32, device=device)
    unique = torch.zeros(max_entries * 2, dtype=torch.int32, device=device)
    counter = torch.zeros(1, dtype=torch.int32, device=device)
    return bitmap, unique, counter, dict(
        scratch_bitmap=bitmap.data_ptr(),
        scratch_unique=unique.data_ptr(),
        scratch_counter=counter.data_ptr(),
    )


def _build_global_page_table(seq_lens, page_size, start_page_id=0):
    page_indices = []
    page_indptr = [0]
    last_page_len = []
    page_id = start_page_id
    for sl in seq_lens:
        npages = (sl + page_size - 1) // page_size
        page_indices.extend(range(page_id, page_id + npages))
        page_id += npages
        page_indptr.append(page_indptr[-1] + npages)
        rem = sl % page_size
        last_page_len.append(rem if rem > 0 else page_size)
    return (
        np.array(page_indices, dtype=np.int32),
        np.array(page_indptr, dtype=np.int32),
        np.array(last_page_len, dtype=np.int32),
        page_id,
    )


def _token_bytes(rank, abs_page, offset_in_page, bpt):
    """Deterministic BPT-byte pattern unique to (rank, abs_page, offset)."""
    base = (rank * 1_000_003 + abs_page * 7_001 + offset_in_page * 97) & 0xFFFF
    return np.array([(base + i * 13) & 0xFF for i in range(bpt)], dtype=np.uint8)


def _expected_rank(pos, cp_world_size):
    return pos % cp_world_size if cp_world_size > 1 else 0


def _local_slot(rank, page_indices_np, page_indptr_np, seq, pos, cp_world_size, eff_page_size):
    """Map global (seq, pos) to this rank's (slot, abs_page, offset_in_page),
    or None if pos doesn't live on this rank."""
    if cp_world_size > 1:
        if pos % cp_world_size != rank:
            return None
        local_pos = (pos - rank) // cp_world_size
    else:
        local_pos = pos
    page_base = int(page_indptr_np[seq])
    page_in_seq = local_pos // eff_page_size
    offset_in_page = local_pos - page_in_seq * eff_page_size
    abs_page = int(page_indices_np[page_base + page_in_seq])
    return abs_page * eff_page_size + offset_in_page, abs_page, offset_in_page


def _to_flat_slots(topk_idx_np, batch_indices_np, kv_token_indptr):
    """Convert raw KV positions [num_tokens, topk] to output flat slots
    (mirroring topk_to_slots' flat / cp_world_size==1 mode):
        flat_slot = kv_token_indptr[seq] + token_pos,  when token_pos >= 0
        flat_slot = -1,                                 when token_pos < 0
    The kernel reads these as direct output slots and reverse-derives
    token_pos = flat_slot - kv_token_indptr[seq] for the CP filter + paged
    src_slot lookup. Tests construct raw positions for readability and
    pre-flatten via this helper before invoking gather_topk_ckv.
    """
    flat = np.empty_like(topk_idx_np)
    for t in range(topk_idx_np.shape[0]):
        seq = int(batch_indices_np[t])
        flat[t] = np.where(
            topk_idx_np[t] >= 0,
            kv_token_indptr[seq] + topk_idx_np[t],
            -1,
        )
    return flat


def test_gather_topk_ckv_local_mirror_non_cp(glm, device):
    """cp_world_size=0 (no CP — kernel takes the straight-line non-CP path),
    single rank, single peer buffer. Basic correctness."""
    torch.manual_seed(0)
    seq_lens = [100]
    page_indices, page_indptr, _, next_page = _build_global_page_table(seq_lens, PAGE_SIZE)
    max_pages = next_page + 4
    eff_page_size = PAGE_SIZE
    cp_world_size = 0
    cp_rank = 0
    total_tokens = seq_lens[0]
    kv_token_indptr = np.cumsum([0] + seq_lens).astype(np.int32)

    local_kv_np = np.zeros((max_pages * eff_page_size, BPT), dtype=np.uint8)
    for pos in range(seq_lens[0]):
        slot, abs_page, off = _local_slot(cp_rank, page_indices, page_indptr, 0, pos,
                                          cp_world_size, eff_page_size)
        local_kv_np[slot] = _token_bytes(cp_rank, abs_page, off, BPT)
    local_kv = torch.from_numpy(local_kv_np).to(device)

    sentinel = 0xDE
    flat_buf = torch.full((total_tokens, BPT), sentinel, dtype=torch.uint8, device=device)

    topk = 32
    num_tokens = 4
    topk_idx_np = np.random.randint(0, seq_lens[0], size=(num_tokens, topk)).astype(np.int32)
    batch_indices_np = np.zeros(num_tokens, dtype=np.int32)
    topk_flat_np = _to_flat_slots(topk_idx_np, batch_indices_np, kv_token_indptr)
    topk_idx = torch.from_numpy(topk_flat_np).to(device)
    batch_indices = torch.from_numpy(batch_indices_np).to(device)
    page_indices_t = torch.from_numpy(page_indices).to(device)
    page_indptr_t = torch.from_numpy(page_indptr).to(device)
    kv_token_indptr_t = torch.from_numpy(kv_token_indptr).to(device)

    _sb, _su, _sc, _scratch = _alloc_scratch(device, total_tokens, num_tokens, topk)
    glm.gather_topk_ckv(
        flat_ptrs=[flat_buf.data_ptr()],
        local_kv_cache=local_kv.data_ptr(),
        topk_idx=topk_idx.data_ptr(),
        batch_indices=batch_indices.data_ptr(),
        page_indices=page_indices_t.data_ptr(),
        page_indptr=page_indptr_t.data_ptr(),
        kv_token_indptr=kv_token_indptr_t.data_ptr(),
        N=1, cp_world_size=cp_world_size, cp_rank=cp_rank,
        eff_page_size=eff_page_size, bpt_bytes=BPT,
        num_tokens=num_tokens, topk=topk, padded_kv_len=total_tokens, **_scratch,
    )
    torch.cuda.synchronize(device)

    flat_buf_np = flat_buf.cpu().numpy()

    written_slots = set()
    for t in range(num_tokens):
        for k in range(topk):
            pos = int(topk_idx_np[t, k])
            if pos < 0:
                continue
            flat_slot = int(kv_token_indptr[int(batch_indices_np[t])]) + pos
            slot, abs_page, off = _local_slot(cp_rank, page_indices, page_indptr,
                                              int(batch_indices_np[t]), pos,
                                              cp_world_size, eff_page_size)
            expected_bytes = _token_bytes(cp_rank, abs_page, off, BPT)
            assert np.array_equal(flat_buf_np[flat_slot], expected_bytes), (
                f"flat_slot={flat_slot} (pos={pos}, abs_page={abs_page}, off={off}) "
                f"bytes mismatch;\n got={flat_buf_np[flat_slot][:16]}\n exp={expected_bytes[:16]}"
            )
            written_slots.add(flat_slot)

    non_topk = set(range(total_tokens)) - written_slots
    assert non_topk, "Test should have non-topk slots to verify stale preservation"
    for s in list(non_topk)[:50]:
        assert (flat_buf_np[s] == sentinel).all(), f"non-topk slot {s} was overwritten"


@pytest.mark.parametrize("cp_world_size", [1, 2, 4, 8])
@pytest.mark.parametrize("num_seqs", [1, 2])
def test_gather_topk_ckv_multi_rank_fanout(glm, device, cp_world_size, num_seqs):
    """Simulate N ranks — each with its own local paged KV cache holding only
    the global positions `pos % cp_world_size == rank`. Run the kernel once per
    rank passing all N peer flat buffers. After all N runs every peer must
    contain the union of all topk tokens at the correct flat slots,
    byte-identical across peers (fan-out property). Non-topk slots retain the
    sentinel."""
    torch.manual_seed(42)
    seq_lens = [128, 200][:num_seqs]
    page_indices, page_indptr, _, next_page = _build_global_page_table(seq_lens, PAGE_SIZE)
    max_pages = next_page + 4
    eff_page_size = PAGE_SIZE // cp_world_size
    total_tokens = sum(seq_lens)
    kv_token_indptr = np.cumsum([0] + seq_lens).astype(np.int32)

    # Per-rank local KV cache. Fill EVERY (abs_page, offset) cell with a
    # deterministic pattern keyed on (rank, abs_page, off) so cross-rank bugs
    # (one rank accidentally emitting another rank's token) stand out.
    per_rank_local_kv = []
    for rank in range(cp_world_size):
        buf_np = np.zeros((max_pages * eff_page_size, BPT), dtype=np.uint8)
        for abs_page in range(max_pages):
            for off in range(eff_page_size):
                slot = abs_page * eff_page_size + off
                buf_np[slot] = _token_bytes(rank, abs_page, off, BPT)
        per_rank_local_kv.append(torch.from_numpy(np.ascontiguousarray(buf_np)).to(device))

    sentinel = 0xCD
    flat_bufs = [torch.full((total_tokens, BPT), sentinel, dtype=torch.uint8, device=device)
                 for _ in range(cp_world_size)]

    topk = 64
    num_tokens = 6
    topk_idx_np = np.empty((num_tokens, topk), dtype=np.int32)
    batch_indices_np = np.empty(num_tokens, dtype=np.int32)
    for t in range(num_tokens):
        seq = t % num_seqs
        batch_indices_np[t] = seq
        topk_idx_np[t] = np.random.randint(0, seq_lens[seq], size=topk)
    # Inject a few -1 (invalid) entries — kernel must silently skip them.
    for t in range(num_tokens):
        topk_idx_np[t, topk - 1] = -1
        topk_idx_np[t, topk - 2] = -1

    topk_flat_np = _to_flat_slots(topk_idx_np, batch_indices_np, kv_token_indptr)
    topk_idx = torch.from_numpy(topk_flat_np).to(device)
    batch_indices = torch.from_numpy(batch_indices_np).to(device)
    page_indices_t = torch.from_numpy(page_indices).to(device)
    page_indptr_t = torch.from_numpy(page_indptr).to(device)
    kv_token_indptr_t = torch.from_numpy(kv_token_indptr).to(device)

    for rank in range(cp_world_size):
        _sb, _su, _sc, _scratch = _alloc_scratch(device, total_tokens, num_tokens, topk)
        glm.gather_topk_ckv(
            flat_ptrs=[b.data_ptr() for b in flat_bufs],
            local_kv_cache=per_rank_local_kv[rank].data_ptr(),
            topk_idx=topk_idx.data_ptr(),
            batch_indices=batch_indices.data_ptr(),
            page_indices=page_indices_t.data_ptr(),
            page_indptr=page_indptr_t.data_ptr(),
            kv_token_indptr=kv_token_indptr_t.data_ptr(),
            N=cp_world_size, cp_world_size=cp_world_size, cp_rank=rank,
            eff_page_size=eff_page_size, bpt_bytes=BPT,
            num_tokens=num_tokens, topk=topk, padded_kv_len=total_tokens, **_scratch,
        )
    torch.cuda.synchronize(device)

    flat_bufs_np = [b.cpu().numpy() for b in flat_bufs]

    # Build expected: flat_slot -> (owning_rank, abs_page, off, bytes)
    expected = {}
    for t in range(num_tokens):
        for k in range(topk):
            pos = int(topk_idx_np[t, k])
            if pos < 0:
                continue
            seq = int(batch_indices_np[t])
            flat_slot = int(kv_token_indptr[seq]) + pos
            owning_rank = _expected_rank(pos, cp_world_size)
            slot, abs_page, off = _local_slot(owning_rank, page_indices, page_indptr,
                                              seq, pos, cp_world_size, eff_page_size)
            expected[flat_slot] = (owning_rank, abs_page, off,
                                   _token_bytes(owning_rank, abs_page, off, BPT))

    # Each peer buffer must contain exactly the expected bytes at every flat slot
    # written by any query's topk.
    for j, fb_np in enumerate(flat_bufs_np):
        for flat_slot, (rank, abs_page, off, expected_bytes) in expected.items():
            assert np.array_equal(fb_np[flat_slot], expected_bytes), (
                f"peer={j} flat_slot={flat_slot} (rank={rank}, abs_page={abs_page}, off={off}) "
                f"bytes mismatch;\n got={fb_np[flat_slot][:16]}\n exp={expected_bytes[:16]}"
            )

    # Fan-out: all peer buffers must be byte-identical (every rank writes the
    # same bytes to every peer). This is the core "read once, write many" check.
    for j in range(1, cp_world_size):
        assert np.array_equal(flat_bufs_np[0], flat_bufs_np[j]), (
            f"peer {j} differs from peer 0 — fan-out didn't write to all peers"
        )

    # Non-topk slots must retain sentinel (i.e. kernel only wrote topk slots).
    written_slots = set(expected.keys())
    non_topk = set(range(total_tokens)) - written_slots
    assert non_topk, "Test should have non-topk slots to verify stale preservation"
    for s in list(non_topk)[:50]:
        for j, fb_np in enumerate(flat_bufs_np):
            assert (fb_np[s] == sentinel).all(), (
                f"peer={j} non-topk slot {s} was overwritten (expected sentinel {sentinel})"
            )


def test_gather_topk_ckv_arbitrary_pointers_eight_peers(glm, device):
    """The 8 peer pointers are arbitrary — here they all live on the same GPU
    (no P2P) and each gets a *different* sentinel pre-fill. With cp_world_size=1
    and N=8, the single rank writes its topk tokens to all 8 buffers; non-topk
    slots must keep each peer's distinct sentinel ( confirming the kernel did
    NOT broadcast a single peer's contents to the others)."""
    torch.manual_seed(7)
    seq_lens = [80]
    page_indices, page_indptr, _, next_page = _build_global_page_table(seq_lens, PAGE_SIZE)
    max_pages = next_page + 4
    eff_page_size = PAGE_SIZE
    cp_world_size = 0
    cp_rank = 0
    total_tokens = seq_lens[0]
    kv_token_indptr = np.cumsum([0] + seq_lens).astype(np.int32)

    local_kv_np = np.zeros((max_pages * eff_page_size, BPT), dtype=np.uint8)
    for pos in range(seq_lens[0]):
        slot, abs_page, off = _local_slot(cp_rank, page_indices, page_indptr, 0, pos,
                                          cp_world_size, eff_page_size)
        local_kv_np[slot] = _token_bytes(cp_rank, abs_page, off, BPT)
    local_kv = torch.from_numpy(local_kv_np).to(device)

    N = 8
    flat_bufs = [torch.full((total_tokens, BPT), 0xA0 + i, dtype=torch.uint8, device=device)
                 for i in range(N)]

    topk = 16
    num_tokens = 3
    topk_idx_np = np.random.randint(0, seq_lens[0], size=(num_tokens, topk)).astype(np.int32)
    batch_indices_np = np.zeros(num_tokens, dtype=np.int32)
    topk_flat_np = _to_flat_slots(topk_idx_np, batch_indices_np, kv_token_indptr)
    topk_idx = torch.from_numpy(topk_flat_np).to(device)
    batch_indices = torch.from_numpy(batch_indices_np).to(device)
    page_indices_t = torch.from_numpy(page_indices).to(device)
    page_indptr_t = torch.from_numpy(page_indptr).to(device)
    kv_token_indptr_t = torch.from_numpy(kv_token_indptr).to(device)

    _sb, _su, _sc, _scratch = _alloc_scratch(device, total_tokens, num_tokens, topk)
    glm.gather_topk_ckv(
        flat_ptrs=[b.data_ptr() for b in flat_bufs],
        local_kv_cache=local_kv.data_ptr(),
        topk_idx=topk_idx.data_ptr(),
        batch_indices=batch_indices.data_ptr(),
        page_indices=page_indices_t.data_ptr(),
        page_indptr=page_indptr_t.data_ptr(),
        kv_token_indptr=kv_token_indptr_t.data_ptr(),
        N=N, cp_world_size=cp_world_size, cp_rank=cp_rank,
        eff_page_size=eff_page_size, bpt_bytes=BPT,
        num_tokens=num_tokens, topk=topk, padded_kv_len=total_tokens, **_scratch,
    )
    torch.cuda.synchronize(device)

    written_slots = set()
    for t in range(num_tokens):
        for k in range(topk):
            pos = int(topk_idx_np[t, k])
            if pos < 0:
                continue
            flat_slot = int(kv_token_indptr[0]) + pos
            slot, abs_page, off = _local_slot(cp_rank, page_indices, page_indptr, 0, pos,
                                              cp_world_size, eff_page_size)
            expected_bytes = _token_bytes(cp_rank, abs_page, off, BPT)
            written_slots.add(flat_slot)
            for j, fb in enumerate(flat_bufs):
                fb_np = fb.cpu().numpy()
                assert np.array_equal(fb_np[flat_slot], expected_bytes), (
                    f"peer={j} flat_slot={flat_slot} bytes mismatch"
                )

    # At any non-topk slot each peer must still carry its DISTINCT sentinel —
    # i.e. writes were only at topk slots, and the kernel didn't touch / sync
    # across peers at untouched slots.
    non_topk = set(range(total_tokens)) - written_slots
    assert non_topk
    s = next(iter(non_topk))
    sentinels = [int(fb.cpu().numpy()[s, 0]) for fb in flat_bufs]
    assert len(set(sentinels)) == N, (
        f"Expected N={N} distinct sentinels at non-topk slot {s}; got {sentinels}"
    )


def test_gather_topk_ckv_matches_gather_pages_on_topk_subset(glm, device):
    """Direct A/B equivalence vs gatherPages: for every (seq, pos) referenced
    by any query's topk, gatherTopkCkv must produce the SAME bytes at
    flat_slot = kv_token_indptr[seq] + token_pos that gatherPages produces at
    that same flat_slot. gatherPages writes ALL slots (full gather); the
    sparse kernel writes ONLY topk-referenced slots and leaves the rest
    untouched (sentinel preservation).

    Uses cp_world_size=0 (non-CP) and N=1 since the comparison is against the
    non-CP gatherPages path; gatherPages CP goes through a separate per-rank-
    shard-then-deinterleave pipeline that is not a 1:1 byte mapping with
    gatherTopkCkv's CP fan-out semantics. KV cache content is RANDOM (not a
    deterministic pattern) so this is a pure cross-check between two
    independent read paths that must agree on the topk subset — eliminating
    any coupling between the expected-bytes computation in earlier tests and
    the kernel's reading semantics.
    """
    torch.manual_seed(1234)
    np.random.seed(1234)
    seq_lens = [90, 130]  # multi-sequence with partial last pages
    page_indices, page_indptr, last_page_len, next_page = _build_global_page_table(seq_lens, PAGE_SIZE)
    max_pages = next_page + 4
    total_tokens = sum(seq_lens)
    kv_token_indptr = np.cumsum([0] + seq_lens).astype(np.int32)

    # Random KV cache content — identical memory, two views:
    #   [max_pages, PAGE_SIZE, BPT] for gatherPages (it indexes page-then-offset)
    #   [max_pages * PAGE_SIZE, BPT] for gatherTopkCkv (it indexes src_slot flat)
    # Both kernels read the same bytes at the same (abs_page, offset_in_page).
    kv_cache_np = np.random.randint(
        0, 256, (max_pages, PAGE_SIZE, BPT), dtype=np.uint8,
    )
    kv_cache_3d = torch.from_numpy(kv_cache_np).to(device)

    topk = 32
    num_tokens = 8
    topk_idx_np = np.empty((num_tokens, topk), dtype=np.int32)
    batch_indices_np = np.empty(num_tokens, dtype=np.int32)
    for t in range(num_tokens):
        seq = t % len(seq_lens)
        batch_indices_np[t] = seq
        topk_idx_np[t] = np.random.randint(0, seq_lens[seq], size=topk)
    # Inject duplicates across queries — same (seq, pos) written by multiple
    # warps must produce identical bytes (idempotent write check).
    for t in range(2, num_tokens, 2):
        topk_idx_np[t, :4] = topk_idx_np[0, :4]
    # Inject -1 invalid entries — kernel must silently skip them.
    for t in range(num_tokens):
        topk_idx_np[t, -1] = -1

    # === Reference: full gather via gatherPages ===
    full = torch.empty(total_tokens, BPT, dtype=torch.uint8, device=device)
    page_indices_t = torch.from_numpy(page_indices).to(device)
    page_indptr_t = torch.from_numpy(page_indptr).to(device)
    last_page_len_t = torch.from_numpy(last_page_len).to(device)
    glm.gather_pages(
        full, kv_cache_3d, page_indices_t, page_indptr_t, last_page_len_t,
        max_pages, len(seq_lens), PAGE_SIZE, BPT,
    )

    # === Sparse gather via gatherTopkCkv ===
    sentinel = 0xB0
    sparse = torch.full((total_tokens, BPT), sentinel, dtype=torch.uint8, device=device)
    topk_flat_np = _to_flat_slots(topk_idx_np, batch_indices_np, kv_token_indptr)
    topk_idx = torch.from_numpy(topk_flat_np).to(device)
    batch_indices = torch.from_numpy(batch_indices_np).to(device)
    kv_token_indptr_t = torch.from_numpy(kv_token_indptr).to(device)

    _sb, _su, _sc, _scratch = _alloc_scratch(device, total_tokens, num_tokens, topk)
    glm.gather_topk_ckv(
        flat_ptrs=[sparse.data_ptr()],
        local_kv_cache=kv_cache_3d.data_ptr(),
        topk_idx=topk_idx.data_ptr(),
        batch_indices=batch_indices.data_ptr(),
        page_indices=page_indices_t.data_ptr(),
        page_indptr=page_indptr_t.data_ptr(),
        kv_token_indptr=kv_token_indptr_t.data_ptr(),
        N=1, cp_world_size=0, cp_rank=0,
        eff_page_size=PAGE_SIZE, bpt_bytes=BPT,
        num_tokens=num_tokens, topk=topk, padded_kv_len=total_tokens, **_scratch,
    )
    torch.cuda.synchronize(device)

    full_np = full.cpu().numpy()
    sparse_np = sparse.cpu().numpy()

    # === Topk subset must EXACTLY match gatherPages output ===
    written_slots = set()
    for t in range(num_tokens):
        for k in range(topk):
            pos = int(topk_idx_np[t, k])
            if pos < 0:
                continue
            seq = int(batch_indices_np[t])
            flat_slot = int(kv_token_indptr[seq]) + pos
            written_slots.add(flat_slot)
            assert np.array_equal(sparse_np[flat_slot], full_np[flat_slot]), (
                f"topk (t={t}, k={k}) pos={pos} seq={seq} flat_slot={flat_slot}: "
                f"sparse != full (gatherPages)\n"
                f" sparse[:16]={sparse_np[flat_slot][:16]}\n"
                f" full  [:16]={full_np[flat_slot][:16]}"
            )

    # === Non-topk slots must retain sentinel (sparse only wrote topk) ===
    non_topk = set(range(total_tokens)) - written_slots
    assert non_topk, "Test must have non-topk slots for sentinel check"
    for s in list(non_topk)[:50]:
        assert (sparse_np[s] == sentinel).all(), (
            f"non-topk slot {s} was overwritten; expected sentinel 0x{sentinel:02x}"
        )


# ---------------------------------------------------------------------------
# Dedup path (num_tokens > 1): the mark/compact kernels collapse duplicate
# flat_slots so each unique token is fanned out once instead of once per query
# that selected it. Two properties matter and neither is covered above:
#
#   1. Heavy duplication across queries still produces every selected token.
#   2. The bitmap is restored between calls. It is zeroed once at allocation
#      and thereafter cleared by the fanout kernel, so a stale bit would make a
#      later call silently drop that slot. Only repeated invocation catches it.
# ---------------------------------------------------------------------------
def test_gather_topk_ckv_dedup_identical_rows_repeated_calls(glm, device):
    """All queries select the SAME topk set — the maximal-duplication case that
    MTP verification produces. Runs the gather three times over a re-primed
    buffer; every call must reproduce the full result."""
    torch.manual_seed(7)
    seq_lens = [100]
    page_indices, page_indptr, _, next_page = _build_global_page_table(seq_lens, PAGE_SIZE)
    max_pages = next_page + 4
    eff_page_size = PAGE_SIZE
    cp_world_size = 0
    cp_rank = 0
    total_tokens = seq_lens[0]
    kv_token_indptr = np.cumsum([0] + seq_lens).astype(np.int32)

    local_kv_np = np.zeros((max_pages * eff_page_size, BPT), dtype=np.uint8)
    expected = {}
    for pos in range(seq_lens[0]):
        slot, abs_page, off = _local_slot(cp_rank, page_indices, page_indptr, 0, pos,
                                          cp_world_size, eff_page_size)
        local_kv_np[slot] = _token_bytes(cp_rank, abs_page, off, BPT)
        expected[pos] = local_kv_np[slot].copy()
    local_kv = torch.from_numpy(local_kv_np).to(device)

    topk = 32
    num_tokens = 4
    # One row of distinct positions, replicated across all queries: every entry
    # beyond the first row is a duplicate, so 128 entries collapse to 32.
    row = np.random.choice(seq_lens[0], size=topk, replace=False).astype(np.int32)
    topk_idx_np = np.tile(row, (num_tokens, 1))
    batch_indices_np = np.zeros(num_tokens, dtype=np.int32)
    topk_flat_np = _to_flat_slots(topk_idx_np, batch_indices_np, kv_token_indptr)

    topk_idx = torch.from_numpy(topk_flat_np).to(device)
    batch_indices = torch.from_numpy(batch_indices_np).to(device)
    page_indices_t = torch.from_numpy(page_indices).to(device)
    page_indptr_t = torch.from_numpy(page_indptr).to(device)
    kv_token_indptr_t = torch.from_numpy(kv_token_indptr).to(device)

    selected = set(int(p) for p in row)
    # Scratch is deliberately REUSED across the three calls and is pre-dirtied
    # below, mirroring the workspace recycler handing back an arbitrary block.
    # The kernel must clear it on entry; a stale bitmap bit would silently drop
    # that slot on calls 2 and 3.
    _sb, _su, _sc, _scratch = _alloc_scratch(device, total_tokens, num_tokens, topk)
    for call in range(3):
        _sb.fill_(-1); _sc.fill_(12345)   # dirty the scratch before every call
        sentinel = 0xD0 + call
        flat_buf = torch.full((total_tokens, BPT), sentinel, dtype=torch.uint8, device=device)
        glm.gather_topk_ckv(
            flat_ptrs=[flat_buf.data_ptr()],
            local_kv_cache=local_kv.data_ptr(),
            topk_idx=topk_idx.data_ptr(),
            batch_indices=batch_indices.data_ptr(),
            page_indices=page_indices_t.data_ptr(),
            page_indptr=page_indptr_t.data_ptr(),
            kv_token_indptr=kv_token_indptr_t.data_ptr(),
            N=1, cp_world_size=cp_world_size, cp_rank=cp_rank,
            eff_page_size=eff_page_size, bpt_bytes=BPT,
            num_tokens=num_tokens, topk=topk, padded_kv_len=total_tokens, **_scratch,
        )
        torch.cuda.synchronize(device)
        flat_np = flat_buf.cpu().numpy()

        for pos in sorted(selected):
            assert np.array_equal(flat_np[pos], expected[pos]), (
                f"call {call}: selected slot {pos} not written "
                f"(stale bitmap bit would drop it)\n"
                f" got     [:16]={flat_np[pos][:16]}\n"
                f" expected[:16]={expected[pos][:16]}"
            )
        for pos in range(total_tokens):
            if pos not in selected:
                assert (flat_np[pos] == sentinel).all(), (
                    f"call {call}: unselected slot {pos} was overwritten"
                )


def test_gather_topk_ckv_dedup_matches_non_dedup_multi_rank(glm, device):
    """The dedup path (num_tokens > 1) must agree byte-for-byte with the
    single-query path (num_tokens == 1) run once per query row, under CP fan-out
    with duplicate slots across rows."""
    torch.manual_seed(11)
    cp_world_size = 4
    seq_lens = [70, 45]
    eff_page_size = PAGE_SIZE // cp_world_size
    page_indices, page_indptr, _, next_page = _build_global_page_table(
        [(_l + cp_world_size - 1) // cp_world_size for _l in seq_lens], eff_page_size)
    max_pages = next_page + 4
    total_tokens = sum(seq_lens)
    kv_token_indptr = np.cumsum([0] + seq_lens).astype(np.int32)

    per_rank_local_kv = []
    for rank in range(cp_world_size):
        buf_np = np.zeros((max_pages * eff_page_size, BPT), dtype=np.uint8)
        for seq, slen in enumerate(seq_lens):
            for pos in range(slen):
                if _expected_rank(pos, cp_world_size) != rank:
                    continue
                slot, abs_page, off = _local_slot(rank, page_indices, page_indptr, seq,
                                                  pos, cp_world_size, eff_page_size)
                buf_np[slot] = _token_bytes(rank, abs_page, off, BPT)
        per_rank_local_kv.append(torch.from_numpy(np.ascontiguousarray(buf_np)).to(device))

    topk = 16
    num_tokens = 6
    topk_idx_np = np.empty((num_tokens, topk), dtype=np.int32)
    batch_indices_np = np.empty(num_tokens, dtype=np.int32)
    for t in range(num_tokens):
        batch_indices_np[t] = t % len(seq_lens)
    # Rows sharing a sequence get the SAME positions, so they duplicate.
    per_seq_rows = {}
    for seq in range(len(seq_lens)):
        per_seq_rows[seq] = np.random.choice(seq_lens[seq], size=topk, replace=False)
    for t in range(num_tokens):
        topk_idx_np[t] = per_seq_rows[int(batch_indices_np[t])]

    topk_flat_np = _to_flat_slots(topk_idx_np, batch_indices_np, kv_token_indptr)
    page_indices_t = torch.from_numpy(page_indices).to(device)
    page_indptr_t = torch.from_numpy(page_indptr).to(device)
    kv_token_indptr_t = torch.from_numpy(kv_token_indptr).to(device)
    batch_indices = torch.from_numpy(batch_indices_np).to(device)
    topk_idx = torch.from_numpy(topk_flat_np).to(device)

    sentinel = 0x5A

    def run(nt, rows_topk_idx, rows_batch_indices):
        bufs = [torch.full((total_tokens, BPT), sentinel, dtype=torch.uint8, device=device)
                for _ in range(cp_world_size)]
        ptrs = [b.data_ptr() for b in bufs]
        for rank in range(cp_world_size):
            _sb, _su, _sc, _scratch = _alloc_scratch(device, total_tokens, num_tokens, topk)
            glm.gather_topk_ckv(
                flat_ptrs=ptrs,
                local_kv_cache=per_rank_local_kv[rank].data_ptr(),
                topk_idx=rows_topk_idx.data_ptr(),
                batch_indices=rows_batch_indices.data_ptr(),
                page_indices=page_indices_t.data_ptr(),
                page_indptr=page_indptr_t.data_ptr(),
                kv_token_indptr=kv_token_indptr_t.data_ptr(),
                N=cp_world_size, cp_world_size=cp_world_size, cp_rank=rank,
                eff_page_size=eff_page_size, bpt_bytes=BPT,
                num_tokens=nt, topk=topk, padded_kv_len=total_tokens, **_scratch,
            )
        torch.cuda.synchronize(device)
        return [b.cpu().numpy() for b in bufs]

    dedup = run(num_tokens, topk_idx, batch_indices)

    # Reference: same work driven one query row at a time, which takes the
    # num_tokens == 1 path and never deduplicates. Accumulate into one buffer
    # set by replaying every row against the same destination.
    ref = [np.full((total_tokens, BPT), sentinel, dtype=np.uint8) for _ in range(cp_world_size)]
    for t in range(num_tokens):
        row_idx = torch.from_numpy(np.ascontiguousarray(topk_flat_np[t:t + 1])).to(device)
        row_bi = torch.from_numpy(np.ascontiguousarray(batch_indices_np[t:t + 1])).to(device)
        got = run(1, row_idx, row_bi)
        for j in range(cp_world_size):
            written = (got[j] != sentinel).any(axis=1)
            ref[j][written] = got[j][written]

    for j in range(cp_world_size):
        assert np.array_equal(dedup[j], ref[j]), (
            f"peer {j}: dedup path disagrees with per-row non-dedup reference"
        )
