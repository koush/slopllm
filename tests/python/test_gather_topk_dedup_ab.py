"""Byte-for-byte: dedup path vs original path, production-shaped CP=8 config."""
import os, sys
import numpy as np, torch, pytest

from helpers import *  # noqa
from test_gather_topk_ckv import (_build_global_page_table, _token_bytes, _expected_rank,
                                  _local_slot, _to_flat_slots, _alloc_scratch, PAGE_SIZE, BPT)

@pytest.mark.parametrize("num_tokens,cp_world_size", [(4, 8), (4, 4), (2, 8), (6, 8)])
def test_dedup_matches_original_bytes(glm, device, num_tokens, cp_world_size):
    torch.manual_seed(3)
    np.random.seed(3)
    seq_lens = [301]                      # deliberately not a page multiple
    eff_page_size = PAGE_SIZE // cp_world_size
    page_indices, page_indptr, _, next_page = _build_global_page_table(
        [(l + cp_world_size - 1) // cp_world_size for l in seq_lens], eff_page_size)
    max_pages = next_page + 4
    total_tokens = sum(seq_lens)
    kv_token_indptr = np.cumsum([0] + seq_lens).astype(np.int32)

    per_rank = []
    for rank in range(cp_world_size):
        buf = np.zeros((max_pages * eff_page_size, BPT), dtype=np.uint8)
        for pos in range(seq_lens[0]):
            if _expected_rank(pos, cp_world_size) != rank:
                continue
            slot, ap, off = _local_slot(rank, page_indices, page_indptr, 0, pos,
                                        cp_world_size, eff_page_size)
            buf[slot] = _token_bytes(rank, ap, off, BPT)
        per_rank.append(torch.from_numpy(np.ascontiguousarray(buf)).to(device))

    topk = 128
    # heavy overlap across queries, plus -1 sentinels, like MTP verification rows
    base = np.random.choice(seq_lens[0], size=topk, replace=False)
    topk_idx_np = np.tile(base, (num_tokens, 1)).astype(np.int32)
    for t in range(1, num_tokens):                 # perturb a few per row
        topk_idx_np[t, :8] = np.random.choice(seq_lens[0], size=8, replace=False)
    topk_idx_np[:, -1] = -1
    batch_indices_np = np.zeros(num_tokens, dtype=np.int32)
    flat = _to_flat_slots(topk_idx_np, batch_indices_np, kv_token_indptr)

    ti = torch.from_numpy(flat).to(device)
    bi = torch.from_numpy(batch_indices_np).to(device)
    pi = torch.from_numpy(page_indices).to(device)
    pp = torch.from_numpy(page_indptr).to(device)
    kti = torch.from_numpy(kv_token_indptr).to(device)

    def run(no_dedup):
        os.environ["GLM_GATHER_NO_DEDUP"] = "1" if no_dedup else "0"
        bufs = [torch.full((total_tokens, BPT), 0x77, dtype=torch.uint8, device=device)
                for _ in range(cp_world_size)]
        ptrs = [b.data_ptr() for b in bufs]
        for rank in range(cp_world_size):
            _sb, _su, _sc, sc = _alloc_scratch(device, total_tokens, num_tokens, topk)
            _sb.fill_(-1); _sc.fill_(999)      # dirty scratch, as the recycler would
            glm.gather_topk_ckv(
                flat_ptrs=ptrs, local_kv_cache=per_rank[rank].data_ptr(),
                topk_idx=ti.data_ptr(), batch_indices=bi.data_ptr(),
                page_indices=pi.data_ptr(), page_indptr=pp.data_ptr(),
                kv_token_indptr=kti.data_ptr(),
                N=cp_world_size, cp_world_size=cp_world_size, cp_rank=rank,
                eff_page_size=eff_page_size, bpt_bytes=BPT,
                num_tokens=num_tokens, topk=topk, padded_kv_len=total_tokens, **sc)
        torch.cuda.synchronize(device)
        return [b.cpu().numpy() for b in bufs]

    ref = run(no_dedup=True)
    got = run(no_dedup=False)
    os.environ.pop("GLM_GATHER_NO_DEDUP", None)
    for j in range(cp_world_size):
        if not np.array_equal(ref[j], got[j]):
            bad = np.where((ref[j] != got[j]).any(axis=1))[0]
            pytest.fail(f"peer {j}: {len(bad)} slots differ, first={bad[:12].tolist()}")
