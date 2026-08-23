"""Integration test: quantize KV → indexer score → topk → topk_to_slots → verify slots.

Chains the four Phase 4-6 kernels end-to-end and verifies that:
1. FP8 quantization writes to the correct paged slots
2. Indexer scores are computed from the correct K cache positions
3. Topk selects token positions within the sequence
4. topk_to_slots converts positions to physical slots that point to the
   correct FP8-quantized KV data in the packed cache
"""
import torch
import pytest
import numpy as np
from helpers import GlmOps, pack_indexer_k
from test_ds_mla_quant import ref_quantize_ds_mla, ref_dequantize_ds_mla

PAGE_SIZE = 64
FP8_MAX = 448.0


def _build_page_table(seq_lens, page_size, device, start_page_id=0):
    """Build page indices, indptr, last_page_len, batch_indices, positions."""
    page_indices = []
    page_indptr = [0]
    last_page_len = []
    batch_indices = []
    positions = []
    page_id = start_page_id
    for b, sl in enumerate(seq_lens):
        npages = (sl + page_size - 1) // page_size
        page_indices.extend(range(page_id, page_id + npages))
        page_id += npages
        page_indptr.append(page_indptr[-1] + npages)
        rem = sl % page_size
        last_page_len.append(rem if rem > 0 else page_size)
        batch_indices.extend([b] * sl)
        positions.extend(range(sl))
    return (
        torch.tensor(page_indices, dtype=torch.int32, device=device),
        torch.tensor(page_indptr, dtype=torch.int32, device=device),
        torch.tensor(last_page_len, dtype=torch.int32, device=device),
        torch.tensor(batch_indices, dtype=torch.int32, device=device),
        torch.tensor(positions, dtype=torch.int32, device=device),
        page_id,  # next free page
    )


def _slot_for_token(batch, pos, page_indices_np, page_indptr_np, page_size):
    """Compute physical slot for a (batch, pos) pair."""
    page_in_seq = pos // page_size
    offset = pos % page_size
    page_id = page_indices_np[page_indptr_np[batch] + page_in_seq]
    return page_id * page_size + offset


@pytest.mark.parametrize("kv_lora_rank,pe_dim", [
    (512, 64),
    (128, 64),
])
def test_sparse_mla_flow_quantize_score_topk_slots(glm, device, kv_lora_rank, pe_dim):
    """Full pipeline: quantize → score → topk → slots → verify data at slots."""
    torch.manual_seed(42)
    page_size = 64
    idx_head_dim = 128
    idx_n_heads = 32
    topk = 64
    scale = idx_head_dim ** -0.5

    seq_lens = [100, 200, 50]
    nnz = sum(seq_lens)
    max_kv_len = max(seq_lens)

    # Build page table
    page_indices_t, page_indptr_t, last_page_len_t, batch_indices_t, positions_t, next_page = \
        _build_page_table(seq_lens, page_size, device)
    page_indices_np = page_indices_t.cpu().numpy()
    page_indptr_np = page_indptr_t.cpu().numpy()

    num_pages_total = len(page_indices_np)
    max_pages = num_pages_total + 8

    # --- 1. Create and quantize MLA KV ---
    ckv = torch.randn(nnz, kv_lora_rank, dtype=torch.bfloat16, device=device)
    kpe = torch.randn(nnz, pe_dim, dtype=torch.bfloat16, device=device)
    num_tiles = kv_lora_rank // 128
    bpt = kv_lora_rank + num_tiles * 4 + pe_dim * 2
    kv_cache = torch.zeros(max_pages, page_size, bpt, dtype=torch.uint8, device=device)

    glm.concat_and_cache_ds_mla(
        kv_cache.data_ptr(), ckv.data_ptr(), kpe.data_ptr(),
        page_indices_t.data_ptr(), page_indptr_t.data_ptr(),
        batch_indices_t.data_ptr(), positions_t.data_ptr(),
        nnz, page_size, kv_lora_rank, pe_dim,
        kv_lora_rank, pe_dim,
    )

    # --- 2. Fill indexer K cache at same physical slots ---
    k_data_bf16 = torch.randn(max_pages, page_size, idx_head_dim, dtype=torch.bfloat16, device=device)
    # Write K vectors at the correct slots (same page table as MLA KV)
    for i in range(nnz):
        b = int(batch_indices_t[i])
        pos = int(positions_t[i])
        slot = _slot_for_token(b, pos, page_indices_np, page_indptr_np, page_size)
        k_data_bf16[slot // page_size, slot % page_size, :] = torch.randn(idx_head_dim, dtype=torch.bfloat16, device=device)
    k_data = pack_indexer_k(k_data_bf16)

    # --- 3. Indexer score (non-causal = decode mode) ---
    total_q = nnz
    idx_q = torch.randn(total_q, idx_n_heads, idx_head_dim, dtype=torch.bfloat16, device=device)
    idx_weights = torch.randn(total_q, idx_n_heads, dtype=torch.bfloat16, device=device)

    qo_indptr = torch.tensor([0] + list(np.cumsum(seq_lens)), dtype=torch.int32, device=device)
    index_scores = torch.full((total_q, max_kv_len), float('-inf'), dtype=torch.bfloat16, device=device)

    glm.indexer_score(
        index_scores, idx_q, k_data, idx_weights,
        page_indices_t, page_indptr_t, last_page_len_t, qo_indptr,
        scale, total_q, idx_n_heads, idx_head_dim,
        page_size, max_kv_len, False,
    )

    # --- 4. Topk ---
    topk_values = torch.empty(total_q, topk, dtype=torch.bfloat16, device=device)
    topk_indices = torch.empty(total_q, topk, dtype=torch.int32, device=device)
    glm.topk(topk_values, topk_indices, index_scores, topk, max_kv_len, total_q)

    # --- 5. Topk to slots ---
    slots = torch.full((total_q, topk), -1, dtype=torch.int32, device=device)
    glm.topk_to_slots(
        slots, topk_indices,
        page_indices_t, page_indptr_t, last_page_len_t,
        batch_indices_t, total_q, topk, page_size,
    )

    # --- 6. Verify slots point to correctly quantized KV data ---
    ref_packed = ref_quantize_ds_mla(ckv, kpe, kv_lora_rank, pe_dim)

    # Check a sample of query tokens and their top-k entries
    sample_tokens = range(0, total_q, max(1, total_q // 20))
    for t in sample_tokens:
        b = int(batch_indices_t[t])
        seq_start = int(np.sum(seq_lens[:b]))

        for k in range(min(8, topk)):
            token_pos = int(topk_indices[t, k])
            slot = int(slots[t, k])

            if token_pos < 0 or slot < 0:
                continue

            # Verify slot matches manual computation
            expected_slot = _slot_for_token(b, token_pos, page_indices_np, page_indptr_np, page_size)
            assert slot == expected_slot, \
                f"token {t} topk {k}: slot {slot} != expected {expected_slot} (pos={token_pos}, batch={b})"

            # Read FP8 data at the slot
            page_id = slot // page_size
            offset = slot % page_size
            actual_packed = kv_cache[page_id, offset]

            # Compare with reference quantization of the original token
            global_token_idx = seq_start + token_pos
            ref_token = ref_packed[global_token_idx]

            ckv_a, kpe_a = ref_dequantize_ds_mla(actual_packed.unsqueeze(0), kv_lora_rank, pe_dim)
            ckv_r, kpe_r = ref_dequantize_ds_mla(ref_token.unsqueeze(0), kv_lora_rank, pe_dim)

            assert torch.allclose(ckv_a, ckv_r, atol=1e-1, rtol=1e-1), \
                f"token {t} topk {k}: ckv mismatch at slot {slot}"
            assert torch.equal(kpe_a, kpe_r), \
                f"token {t} topk {k}: kpe mismatch at slot {slot}"


@pytest.mark.parametrize("cp_world_size,cp_rank", [
    (2, 0),
    (2, 1),
    (4, 2),
])
def test_sparse_mla_flow_cp_filter(glm, device, cp_world_size, cp_rank):
    """Verify CP filtering: only tokens at positions % cp_world_size == cp_rank get valid slots."""
    torch.manual_seed(42)
    page_size = 64
    kv_lora_rank = 512
    pe_dim = 64
    idx_head_dim = 128
    idx_n_heads = 32
    topk = 64
    scale = idx_head_dim ** -0.5

    seq_lens = [150]
    nnz = sum(seq_lens)
    max_kv_len = seq_lens[0]

    page_indices_t, page_indptr_t, last_page_len_t, batch_indices_t, positions_t, next_page = \
        _build_page_table(seq_lens, page_size, device)
    page_indices_np = page_indices_t.cpu().numpy()
    page_indptr_np = page_indptr_t.cpu().numpy()
    max_pages = len(page_indices_np) + 8

    # Quantize MLA KV
    ckv = torch.randn(nnz, kv_lora_rank, dtype=torch.bfloat16, device=device)
    kpe = torch.randn(nnz, pe_dim, dtype=torch.bfloat16, device=device)
    num_tiles = kv_lora_rank // 128
    bpt = kv_lora_rank + num_tiles * 4 + pe_dim * 2
    kv_cache = torch.zeros(max_pages, page_size, bpt, dtype=torch.uint8, device=device)

    glm.concat_and_cache_ds_mla(
        kv_cache.data_ptr(), ckv.data_ptr(), kpe.data_ptr(),
        page_indices_t.data_ptr(), page_indptr_t.data_ptr(),
        batch_indices_t.data_ptr(), positions_t.data_ptr(),
        nnz, page_size, kv_lora_rank, pe_dim,
        kv_lora_rank, pe_dim,
    )

    # Indexer K + Q + weights
    k_data = pack_indexer_k(torch.randn(
        max_pages, page_size, idx_head_dim, dtype=torch.bfloat16, device=device))
    idx_q = torch.randn(nnz, idx_n_heads, idx_head_dim, dtype=torch.bfloat16, device=device)
    idx_weights = torch.randn(nnz, idx_n_heads, dtype=torch.bfloat16, device=device)
    qo_indptr = torch.tensor([0, nnz], dtype=torch.int32, device=device)

    index_scores = torch.full((nnz, max_kv_len), float('-inf'), dtype=torch.bfloat16, device=device)
    glm.indexer_score(
        index_scores, idx_q, k_data, idx_weights,
        page_indices_t, page_indptr_t, last_page_len_t, qo_indptr,
        scale, nnz, idx_n_heads, idx_head_dim,
        page_size, max_kv_len, False,
    )

    topk_values = torch.empty(nnz, topk, dtype=torch.bfloat16, device=device)
    topk_indices = torch.empty(nnz, topk, dtype=torch.int32, device=device)
    glm.topk(topk_values, topk_indices, index_scores, topk, max_kv_len, nnz)

    slots = torch.full((nnz, topk), -1, dtype=torch.int32, device=device)
    glm.topk_to_slots(
        slots, topk_indices,
        page_indices_t, page_indptr_t, last_page_len_t,
        batch_indices_t, nnz, topk, page_size,
        cp_world_size, cp_rank,
    )

    # Verify: compacted slots match expected (valid entries at front, in input order).
    # topk_to_slots compacts valid slots to a contiguous prefix [0, count) preserving
    # the relative order from topk_indices, so slots[t, k] does NOT correspond to
    # topk_indices[t, k] when invalid entries were filtered out (e.g. wrong CP rank).
    slots_cpu = slots.cpu().numpy()
    topk_cpu = topk_indices.cpu().numpy()
    last_page_len_np = last_page_len_t.cpu().numpy()
    eps = page_size // cp_world_size

    valid_count = 0
    for t in range(nnz):
        # Build expected compacted slots from topk_indices
        expected = []
        for k in range(topk):
            token_pos = int(topk_cpu[t, k])
            if token_pos < 0:
                continue
            if token_pos % cp_world_size != cp_rank:
                continue
            local_pos = (token_pos - cp_rank) // cp_world_size
            num_pages = page_indptr_np[1] - page_indptr_np[0]
            local_kv_len = (num_pages - 1) * eps + int(last_page_len_np[0])
            if local_pos >= local_kv_len:
                continue
            page_idx_in_seq = local_pos // eps
            offset_in_page = local_pos % eps
            abs_page = page_indices_np[page_indptr_np[0] + page_idx_in_seq]
            expected.append(abs_page * eps + offset_in_page)

        # Compare actual vs expected
        for k in range(len(expected)):
            assert int(slots_cpu[t, k]) == expected[k], \
                f"token {t} k {k}: slot {int(slots_cpu[t, k])} != expected {expected[k]}"
        for k in range(len(expected), topk):
            assert int(slots_cpu[t, k]) == -1, \
                f"token {t} k {k}: expected -1 but got {int(slots_cpu[t, k])}"

        valid_count += len(expected)

    # Roughly 1/cp_world_size of entries should be valid
    total_entries = nnz * topk
    min_expected = total_entries // (cp_world_size * 2)
    assert valid_count > min_expected, \
        f"Too few valid slots: {valid_count}/{total_entries} (expected ~{total_entries // cp_world_size})"


def test_sparse_mla_flow_topk_ordering(glm, device):
    """Verify that topk indices are in score-descending order and slots follow."""
    torch.manual_seed(42)
    page_size = 64
    kv_lora_rank = 512
    pe_dim = 64
    idx_head_dim = 128
    idx_n_heads = 16
    topk = 32
    scale = idx_head_dim ** -0.5

    seq_lens = [80]
    nnz = seq_lens[0]
    max_kv_len = nnz

    page_indices_t, page_indptr_t, last_page_len_t, batch_indices_t, positions_t, _ = \
        _build_page_table(seq_lens, page_size, device)
    page_indices_np = page_indices_t.cpu().numpy()
    page_indptr_np = page_indptr_t.cpu().numpy()
    max_pages = len(page_indices_np) + 4

    k_data = pack_indexer_k(torch.randn(
        max_pages, page_size, idx_head_dim, dtype=torch.bfloat16, device=device))
    idx_q = torch.randn(nnz, idx_n_heads, idx_head_dim, dtype=torch.bfloat16, device=device)
    idx_weights = torch.randn(nnz, idx_n_heads, dtype=torch.bfloat16, device=device)
    qo_indptr = torch.tensor([0, nnz], dtype=torch.int32, device=device)

    index_scores = torch.full((nnz, max_kv_len), float('-inf'), dtype=torch.bfloat16, device=device)
    glm.indexer_score(
        index_scores, idx_q, k_data, idx_weights,
        page_indices_t, page_indptr_t, last_page_len_t, qo_indptr,
        scale, nnz, idx_n_heads, idx_head_dim,
        page_size, max_kv_len, False,
    )

    topk_values = torch.empty(nnz, topk, dtype=torch.bfloat16, device=device)
    topk_indices = torch.empty(nnz, topk, dtype=torch.int32, device=device)
    glm.topk(topk_values, topk_indices, index_scores, topk, max_kv_len, nnz)

    slots = torch.full((nnz, topk), -1, dtype=torch.int32, device=device)
    glm.topk_to_slots(
        slots, topk_indices,
        page_indices_t, page_indptr_t, last_page_len_t,
        batch_indices_t, nnz, topk, page_size,
    )

    # Verify ordering: scores should be non-increasing
    vals = topk_values.float().cpu().numpy()
    for t in range(nnz):
        for k in range(topk - 1):
            assert vals[t, k] >= vals[t, k + 1], \
                f"token {t}: topk not ordered at k={k}: {vals[t, k]} < {vals[t, k+1]}"

    # Verify all slots are valid (single seq, all positions < kv_len)
    slots_cpu = slots.cpu().numpy()
    assert np.all(slots_cpu >= 0), "All slots should be valid for single full sequence"
