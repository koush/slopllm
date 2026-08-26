"""Correctness test for sparse MLA pipeline with production model dimensions.

Tests the full pipeline: concatAndCacheDsMla -> indexerScoreTopk -> topkToSlots
-> sparseMlaDecode, using GLM-5.1 production dimensions:
  NUM_HEADS=64, KV_LORA_RANK=512, PE_DIM=64, INDEX_N_HEADS=32, INDEX_HEAD_DIM=128,
  TOPK=2048, PAGE_SIZE=64, SM_SCALE=256**-0.5
"""
import torch
import pytest
import numpy as np
from helpers import GlmOps, pack_indexer_k, unpack_indexer_k  # noqa: F401
from test_ds_mla_quant import ref_quantize_ds_mla, ref_dequantize_ds_mla
from test_sparse_mla_sm120 import _build_page_table, _slot_for_token, ref_sparse_mla_prefill

PAGE_SIZE = 64
KV_LORA_RANK = 512
PE_DIM = 64
D_QK = KV_LORA_RANK + PE_DIM   # 576
D_V = KV_LORA_RANK             # 512
BPT = KV_LORA_RANK + (KV_LORA_RANK // 128) * 4 + PE_DIM * 2  # 656
TOPK = 2048

# Production model dimensions
NUM_HEADS = 64
INDEX_N_HEADS = 32
INDEX_HEAD_DIM = 128
INDEX_ROPE_DIM = PE_DIM  # 64
INDEX_NOPE_DIM = INDEX_HEAD_DIM - INDEX_ROPE_DIM  # 64
SM_SCALE = 256.0 ** -0.5  # production: qk_head_dim=256


def _build_page_table_np(seq_lens, page_size, start_page_id=0):
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
        np.array(page_indices, dtype=np.int32),
        np.array(page_indptr, dtype=np.int32),
        np.array(last_page_len, dtype=np.int32),
        np.array(batch_indices, dtype=np.int32),
        np.array(positions, dtype=np.int32),
        page_id,
    )


def ref_indexer_score_topk(q_idx, k_data, k_scale_data, weights, page_indices_np, page_indptr_np,
                           last_page_len_np, qo_indptr_np, scale, idx_n_heads, idx_head_dim,
                           page_size, topk, causal, device):
    """PyTorch reference for indexer score+topk."""
    total_q = q_idx.shape[0]
    if k_data.dtype == torch.uint8:
        k_data = unpack_indexer_k(k_data, k_scale_data)
    out_idx = torch.full((total_q, topk), -1, dtype=torch.int32, device=device)

    for qi in range(total_q):
        seq = 0
        while qo_indptr_np[seq + 1] <= qi:
            seq += 1
        q_local_pos = qi - qo_indptr_np[seq]

        page_start = page_indptr_np[seq]
        page_end = page_indptr_np[seq + 1]
        num_pages = page_end - page_start
        kv_len = (num_pages - 1) * page_size + last_page_len_np[seq] if num_pages > 0 else 0
        num_queries = qo_indptr_np[seq + 1] - qo_indptr_np[seq]
        prefix_len = max(0, kv_len - num_queries)
        causal_limit = prefix_len + q_local_pos if causal else kv_len - 1

        scores_heap = []  # (score, pos)
        q_heads = q_idx[qi]  # [idx_n_heads, idx_head_dim]

        for pos in range(min(kv_len, causal_limit + 1)):
            page_in_seq = pos // page_size
            offset = pos % page_size
            page_id = page_indices_np[page_start + page_in_seq]
            k_token = k_data[page_id, offset]  # [idx_head_dim]

            head_scores = torch.zeros(idx_n_heads, device=device, dtype=torch.float32)
            for h in range(idx_n_heads):
                dot = (q_heads[h].float() * k_token.float()).sum()
                dot = dot * scale
                dot = torch.clamp(dot, min=0.0)  # ReLU
                head_scores[h] = dot

            w = weights[qi].float()  # [idx_n_heads]
            index_score = (w * head_scores).sum()
            index_score = index_score.to(torch.bfloat16).float().item()

            scores_heap.append((index_score, pos))

        # Top-k
        scores_heap.sort(key=lambda x: -x[0])
        for i in range(min(topk, len(scores_heap))):
            out_idx[qi, i] = scores_heap[i][1]

    return out_idx


@pytest.fixture
def prod_setup(glm, device):
    """Create KV cache, indexer K cache, Q, weights for production-dimension test."""
    torch.manual_seed(42)
    seq_len = 200  # > 64 to span multiple pages
    num_q = 1  # decode-like

    page_indices_np, page_indptr_np, last_page_len_np, batch_indices_np, positions_np, next_page = \
        _build_page_table_np([seq_len], PAGE_SIZE)
    max_pages = next_page + 4

    page_indices = torch.tensor(page_indices_np, device=device)
    page_indptr = torch.tensor(page_indptr_np, device=device)
    last_page_len = torch.tensor(last_page_len_np, device=device)
    batch_indices = torch.tensor(batch_indices_np, device=device)
    positions = torch.tensor(positions_np, device=device)
    qo_indptr = torch.tensor([0, num_q], dtype=torch.int32, device=device)

    # Create and quantize KV cache (main MLA)
    ckv = torch.randn(seq_len, KV_LORA_RANK, dtype=torch.bfloat16, device=device)
    kpe = torch.randn(seq_len, PE_DIM, dtype=torch.bfloat16, device=device)
    kv_cache = torch.zeros(max_pages, PAGE_SIZE, BPT, dtype=torch.uint8, device=device)

    glm.concat_and_cache_ds_mla(
        kv_cache.data_ptr(), ckv.data_ptr(), kpe.data_ptr(),
        page_indices.data_ptr(), page_indptr.data_ptr(),
        batch_indices.data_ptr(), positions.data_ptr(),
        seq_len, PAGE_SIZE, KV_LORA_RANK, PE_DIM,
        KV_LORA_RANK, PE_DIM,
    )

    # Create indexer K cache: [max_pages, PAGE_SIZE, INDEX_HEAD_DIM]
    k_data_bf16 = torch.randn(max_pages, PAGE_SIZE, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=device)
    # Fill only the valid positions
    for pos in range(seq_len):
        page_in_seq = pos // PAGE_SIZE
        offset = pos % PAGE_SIZE
        page_id = page_indices_np[page_indptr_np[0] + page_in_seq]
        k_data_bf16[page_id, offset] = torch.randn(INDEX_HEAD_DIM, dtype=torch.bfloat16, device=device)
    k_data, k_scale_data = pack_indexer_k(k_data_bf16)

    # Create indexer Q: [num_q, INDEX_N_HEADS, INDEX_HEAD_DIM]
    q_idx = torch.randn(num_q, INDEX_N_HEADS, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=device)

    # Create indexer weights: [num_q, INDEX_N_HEADS]
    weights = torch.randn(num_q, INDEX_N_HEADS, dtype=torch.bfloat16, device=device)
    # Apply scale: sqrt(1/n_heads)
    weights = weights * (INDEX_N_HEADS ** -0.5)

    # Create main MLA Q: [num_q, NUM_HEADS, D_QK]
    q_mla = torch.randn(num_q, NUM_HEADS, D_QK, dtype=torch.bfloat16, device=device)

    return {
        'glm': glm, 'device': device,
        'kv_cache': kv_cache, 'k_data': k_data, 'k_scale_data': k_scale_data,
        'q_idx': q_idx, 'weights': weights, 'q_mla': q_mla,
        'page_indices': page_indices, 'page_indptr': page_indptr,
        'last_page_len': last_page_len, 'batch_indices': batch_indices,
        'positions': positions, 'qo_indptr': qo_indptr,
        'page_indices_np': page_indices_np, 'page_indptr_np': page_indptr_np,
        'last_page_len_np': last_page_len_np,
        'seq_len': seq_len, 'num_q': num_q, 'max_pages': max_pages,
    }


def test_indexer_score_topk_prod_dims(prod_setup, device):
    """Test indexer score+topk with production dimensions (32 heads, 128 dim)."""
    s = prod_setup
    glm = s['glm']
    total_q = s['num_q']

    out_idx = torch.full((total_q, TOPK), -1, dtype=torch.int32, device=device)
    out_scores = torch.full((total_q, TOPK), float('-inf'), dtype=torch.bfloat16, device=device)

    max_kv = s['seq_len']
    TOPK_SCRATCH_I32 = 1056
    scores = torch.empty(total_q, max_kv, dtype=torch.bfloat16, device=device)
    row_len = torch.zeros(total_q, dtype=torch.int32, device=device)
    hist = torch.empty(total_q, TOPK_SCRATCH_I32, dtype=torch.int32, device=device)
    meta = torch.empty(total_q, 4, dtype=torch.int32, device=device)
    num_splits = min(256, max(1, (max_kv + 255) // 256))

    glm.indexer_score_topk_v2(
        out_idx, out_scores, s['q_idx'], s['k_data'], s['k_scale_data'], s['weights'],
        s['page_indices'], s['page_indptr'], s['last_page_len'], s['qo_indptr'],
        INDEX_HEAD_DIM ** -0.5, total_q, INDEX_N_HEADS, INDEX_HEAD_DIM,
        PAGE_SIZE, TOPK, False,
        scores, row_len, hist, meta, max_kv, num_splits,
    )
    torch.cuda.synchronize()

    # Reference
    ref_idx = ref_indexer_score_topk(
        s['q_idx'], s['k_data'], s['k_scale_data'], s['weights'],
        s['page_indices_np'], s['page_indptr_np'], s['last_page_len_np'],
        np.array([0, total_q], dtype=np.int32),
        INDEX_HEAD_DIM ** -0.5, INDEX_N_HEADS, INDEX_HEAD_DIM,
        PAGE_SIZE, TOPK, False, device,
    )

    # Compare top-k indices: compare SETS of valid (non -1) entries
    actual_valid = set(out_idx[out_idx >= 0].cpu().numpy().tolist())
    expected_valid = set(ref_idx[ref_idx >= 0].cpu().numpy().tolist())
    overlap = len(actual_valid & expected_valid)
    overlap_ratio = overlap / len(expected_valid) if expected_valid else 1.0

    print(f"Indexer topk overlap: {overlap}/{len(expected_valid)} = {overlap_ratio:.2%}")

    assert overlap_ratio > 0.95, f"Indexer topk overlap too low: {overlap_ratio:.2%}"


def test_topk_to_slots_prod(prod_setup, device):
    """Test topkToSlots with production dimensions."""
    s = prod_setup
    glm = s['glm']

    # Create topk indices: all positions 0..seq_len-1, then -1 padding
    topk_idx = torch.full((s['num_q'], TOPK), -1, dtype=torch.int32, device=device)
    for i in range(s['seq_len']):
        topk_idx[0, i] = i

    slots = torch.full((s['num_q'], TOPK), -1, dtype=torch.int32, device=device)

    glm.topk_to_slots(
        slots.data_ptr(), topk_idx.data_ptr(),
        s['page_indices'].data_ptr(), s['page_indptr'].data_ptr(),
        s['last_page_len'].data_ptr(), s['batch_indices'].data_ptr(),
        s['num_q'], TOPK, PAGE_SIZE,
    )
    torch.cuda.synchronize()

    # topk_to_slots compacts valid slots to the front in arbitrary order, so
    # compare the set of valid slots and check the rest is -1 padded.
    expected = {_slot_for_token(0, pos, s['page_indices_np'], s['page_indptr_np'], PAGE_SIZE)
                for pos in range(s['seq_len'])}
    slots_np = slots[0].cpu().numpy()
    got = set(int(x) for x in slots_np if x >= 0)
    assert got == expected, f"slot set mismatch: missing {expected - got}, extra {got - expected}"
    assert set(int(x) for x in slots_np[s['seq_len']:]) == {-1}, "expected -1 padding after valid slots"

    print(f"topkToSlots: all {s['seq_len']} slots correct")


def test_full_pipeline_decode_prod(prod_setup, device):
    """Test full pipeline: indexer -> topkToSlots -> sparseMlaDecode with production dims."""
    s = prod_setup
    glm = s['glm']
    total_q = s['num_q']

    # Step 1: Indexer score+topk
    out_idx = torch.full((total_q, TOPK), -1, dtype=torch.int32, device=device)
    out_scores = torch.full((total_q, TOPK), float('-inf'), dtype=torch.bfloat16, device=device)
    max_kv = s['seq_len']
    TOPK_SCRATCH_I32 = 1056
    scores = torch.empty(total_q, max_kv, dtype=torch.bfloat16, device=device)
    row_len = torch.zeros(total_q, dtype=torch.int32, device=device)
    hist = torch.empty(total_q, TOPK_SCRATCH_I32, dtype=torch.int32, device=device)
    meta = torch.empty(total_q, 4, dtype=torch.int32, device=device)
    num_splits = min(256, max(1, (max_kv + 255) // 256))
    glm.indexer_score_topk_v2(
        out_idx, out_scores, s['q_idx'], s['k_data'], s['k_scale_data'], s['weights'],
        s['page_indices'], s['page_indptr'], s['last_page_len'], s['qo_indptr'],
        INDEX_HEAD_DIM ** -0.5, total_q, INDEX_N_HEADS, INDEX_HEAD_DIM,
        PAGE_SIZE, TOPK, False,
        scores, row_len, hist, meta, max_kv, num_splits,
    )
    torch.cuda.synchronize()

    # Step 2: topkToSlots
    slots = torch.full((total_q, TOPK), -1, dtype=torch.int32, device=device)
    glm.topk_to_slots(
        slots.data_ptr(), out_idx.data_ptr(),
        s['page_indices'].data_ptr(), s['page_indptr'].data_ptr(),
        s['last_page_len'].data_ptr(), s['batch_indices'].data_ptr(),
        total_q, TOPK, PAGE_SIZE,
    )
    torch.cuda.synchronize()

    # Verify slots: exactly seq_len valid (non -1) slots
    valid_count = (slots[0] >= 0).sum().item()
    assert valid_count == s['seq_len'], f"Expected {s['seq_len']} valid slots, got {valid_count}"

    # Step 3: Sparse MLA decode
    num_splits = (TOPK + 63) // 64  # 32
    mid_out = torch.zeros(total_q, NUM_HEADS, num_splits, D_V, dtype=torch.bfloat16, device=device)
    mid_lse = torch.zeros(total_q, NUM_HEADS, num_splits, dtype=torch.float32, device=device)
    output = torch.zeros(total_q, NUM_HEADS, D_V, dtype=torch.bfloat16, device=device)
    out_lse = torch.zeros(total_q, NUM_HEADS, dtype=torch.float32, device=device)
    stride_kv_block = PAGE_SIZE * BPT

    glm.sparse_mla_decode(
        s['q_mla'].data_ptr(), s['kv_cache'].data_ptr(), slots.data_ptr(),
        mid_out.data_ptr(), mid_lse.data_ptr(),
        output.data_ptr(), out_lse.data_ptr(),
        total_q, NUM_HEADS, TOPK, num_splits,
        SM_SCALE, stride_kv_block, 0,
    )
    torch.cuda.synchronize()

    assert output.abs().max().item() > 0, "Decode output is all zeros"

    # Reference: use all slots (kernel processes all TOPK slots, masking -1 to -inf)
    o_ref, lse_ref = ref_sparse_mla_prefill(
        s['q_mla'], s['kv_cache'], slots, None,
        SM_SCALE, KV_LORA_RANK, PE_DIM,
    )

    output_f = output.float()
    o_ref_f = o_ref.float()
    max_diff = (output_f - o_ref_f).abs().max().item()
    rel_diff = max_diff / (o_ref_f.abs().max().item() + 1e-6)

    print(f"Full pipeline decode max diff: {max_diff}")
    print(f"Full pipeline decode rel diff: {rel_diff:.4f}")
    print(f"Output[0,0,:8]: {output_f[0,0,:8].tolist()}")
    print(f"Ref[0,0,:8]:    {o_ref_f[0,0,:8].tolist()}")

    assert max_diff < 3.0, f"Full pipeline output diff too large: {max_diff}"


def test_sparse_mla_decode_64_heads(prod_setup, device):
    """Test SM120 decode kernel with NUM_HEADS=64 (production non-TP)."""
    s = prod_setup
    glm = s['glm']
    total_q = s['num_q']

    # Build slots: all positions
    slots = torch.full((total_q, TOPK), -1, dtype=torch.int32, device=device)
    for pos in range(s['seq_len']):
        slot = _slot_for_token(0, pos, s['page_indices_np'], s['page_indptr_np'], PAGE_SIZE)
        slots[0, pos] = slot

    num_splits = (TOPK + 63) // 64
    mid_out = torch.zeros(total_q, NUM_HEADS, num_splits, D_V, dtype=torch.bfloat16, device=device)
    mid_lse = torch.zeros(total_q, NUM_HEADS, num_splits, dtype=torch.float32, device=device)
    output = torch.zeros(total_q, NUM_HEADS, D_V, dtype=torch.bfloat16, device=device)
    out_lse = torch.zeros(total_q, NUM_HEADS, dtype=torch.float32, device=device)
    stride_kv_block = PAGE_SIZE * BPT

    glm.sparse_mla_decode(
        s['q_mla'].data_ptr(), s['kv_cache'].data_ptr(), slots.data_ptr(),
        mid_out.data_ptr(), mid_lse.data_ptr(),
        output.data_ptr(), out_lse.data_ptr(),
        total_q, NUM_HEADS, TOPK, num_splits,
        SM_SCALE, stride_kv_block, 0,
    )
    torch.cuda.synchronize()

    assert output.abs().max().item() > 0, "Output is all zeros"

    num_valid = torch.tensor([s['seq_len']], dtype=torch.int32, device=device)
    o_ref, lse_ref = ref_sparse_mla_prefill(
        s['q_mla'], s['kv_cache'], slots, num_valid,
        SM_SCALE, KV_LORA_RANK, PE_DIM,
    )

    output_f = output.float()
    o_ref_f = o_ref.float()
    max_diff = (output_f - o_ref_f).abs().max().item()

    print(f"64-head decode max diff: {max_diff}")
    print(f"Output[0,0,:8]: {output_f[0,0,:8].tolist()}")
    print(f"Ref[0,0,:8]:    {o_ref_f[0,0,:8].tolist()}")

    assert max_diff < 2.0, f"64-head decode diff too large: {max_diff}"


def test_sparse_mla_prefill_64_heads(prod_setup, device):
    """Test SM120 prefill kernel with NUM_HEADS=64 and production scale."""
    s = prod_setup
    glm = s['glm']

    # Use multiple query tokens for prefill
    num_q = 5
    q_prefill = torch.randn(num_q, NUM_HEADS, D_QK, dtype=torch.bfloat16, device=device)

    # Build slots: each query attends to all tokens (causal)
    slots_list = []
    for t in range(num_q):
        token_slots = []
        for pos in range(s['seq_len']):
            slot = _slot_for_token(0, pos, s['page_indices_np'], s['page_indptr_np'], PAGE_SIZE)
            token_slots.append(slot)
        while len(token_slots) < TOPK:
            token_slots.append(-1)
        slots_list.append(token_slots)

    indices = torch.tensor(slots_list, dtype=torch.int32, device=device)

    output = torch.zeros(num_q, NUM_HEADS, D_V, dtype=torch.bfloat16, device=device)
    out_lse = torch.zeros(num_q, NUM_HEADS, dtype=torch.float32, device=device)
    stride_kv_block = PAGE_SIZE * BPT

    glm.sparse_mla_prefill(
        q_prefill.data_ptr(), s['kv_cache'].data_ptr(), indices.data_ptr(),
        output.data_ptr(), out_lse.data_ptr(),
        num_q, NUM_HEADS, TOPK, PAGE_SIZE,
        SM_SCALE, stride_kv_block,
    )
    torch.cuda.synchronize()

    assert output.abs().max().item() > 0, "Prefill output is all zeros"

    num_valid = torch.tensor([s['seq_len']] * num_q, dtype=torch.int32, device=device)
    o_ref, lse_ref = ref_sparse_mla_prefill(
        q_prefill, s['kv_cache'], indices, num_valid,
        SM_SCALE, KV_LORA_RANK, PE_DIM,
    )

    output_f = output.float()
    o_ref_f = o_ref.float()
    max_diff = (output_f - o_ref_f).abs().max().item()

    print(f"64-head prefill max diff: {max_diff}")
    print(f"Output[0,0,:8]: {output_f[0,0,:8].tolist()}")
    print(f"Ref[0,0,:8]:    {o_ref_f[0,0,:8].tolist()}")

    assert max_diff < 2.0, f"64-head prefill diff too large: {max_diff}"


@pytest.mark.parametrize("num_q", [16, 32])
def test_sparse_mla_prefill_split_q_64_heads(prod_setup, device, num_q):
    """Production MG prefill must treat split Q identically to contiguous Q."""
    s = prod_setup
    glm = s['glm']
    q = torch.randn(num_q, NUM_HEADS, D_QK, dtype=torch.bfloat16, device=device)
    q_nope = q[:, :, :KV_LORA_RANK].contiguous()
    q_rope = q[:, :, KV_LORA_RANK:].contiguous()

    token_slots = [
        _slot_for_token(0, pos, s['page_indices_np'], s['page_indptr_np'], PAGE_SIZE)
        for pos in range(s['seq_len'])
    ]
    token_slots.extend([-1] * (TOPK - len(token_slots)))
    indices = torch.tensor([token_slots] * num_q, dtype=torch.int32, device=device)
    num_valid = torch.full((num_q,), s['seq_len'], dtype=torch.int32, device=device)

    expected = torch.empty(num_q, NUM_HEADS, D_V, dtype=torch.bfloat16, device=device)
    expected_lse = torch.empty(num_q, NUM_HEADS, dtype=torch.float32, device=device)
    actual = torch.empty_like(expected)
    actual_lse = torch.empty_like(expected_lse)
    stride_kv_block = PAGE_SIZE * BPT

    glm.sparse_mla_prefill(
        q.data_ptr(), s['kv_cache'].data_ptr(), indices.data_ptr(),
        expected.data_ptr(), expected_lse.data_ptr(), num_q, NUM_HEADS, TOPK,
        PAGE_SIZE, SM_SCALE, stride_kv_block, num_valid.data_ptr(),
    )
    glm.sparse_mla_prefill_split_q(
        q_nope.data_ptr(), q_rope.data_ptr(), s['kv_cache'].data_ptr(), indices.data_ptr(),
        actual.data_ptr(), actual_lse.data_ptr(), num_q, NUM_HEADS, TOPK,
        SM_SCALE, stride_kv_block, num_valid.data_ptr(),
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)
    assert torch.equal(actual_lse, expected_lse)
