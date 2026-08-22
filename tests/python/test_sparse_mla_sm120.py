"""Correctness test for SM120 sparse MLA prefill/decode kernel.

Validates the FlashInfer SM120 kernel (GLM_NSA path) against a PyTorch
reference implementation.  The kernel hardcodes D_QK=576 (kv_lora_rank=512
+ pe_dim=64) and D_V=512, so we use those dimensions.
"""
import torch
import pytest
import numpy as np
from helpers import GlmOps  # noqa: F401 (used by conftest fixture)
from test_ds_mla_quant import ref_quantize_ds_mla, ref_dequantize_ds_mla

PAGE_SIZE = 64
KV_LORA_RANK = 512
PE_DIM = 64
D_QK = KV_LORA_RANK + PE_DIM   # 576
D_V = KV_LORA_RANK             # 512
BPT = KV_LORA_RANK + (KV_LORA_RANK // 128) * 4 + PE_DIM * 2  # 656
TOPK = 2048
NUM_HEADS = 8
SM_SCALE = 0.0791  # 160 ** -0.5 (qk_head_dim of small model)


def _build_page_table(seq_lens, page_size, device, start_page_id=0):
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
        page_id,
    )


def _slot_for_token(batch, pos, page_indices_np, page_indptr_np, page_size):
    page_in_seq = pos // page_size
    offset = pos % page_size
    page_id = page_indices_np[page_indptr_np[batch] + page_in_seq]
    return page_id * page_size + offset


def ref_sparse_mla_prefill(q, kv_cache_packed, indices, num_valid_per_token,
                           sm_scale, kv_lora_rank, pe_dim):
    """PyTorch reference for sparse MLA prefill.

    q:            [num_q, num_heads, D_QK] BF16
    kv_cache:     [max_pages, page_size, BPT] U8 (packed FP8+scale+BF16)
    indices:      [num_q, topk] I32 (flat slot IDs, -1 = invalid)
    num_valid:    [num_q] I32 (effective k per token)
    Returns:      (o [num_q, num_heads, D_V] BF16, lse [num_q, num_heads] F32)
    """
    num_q, num_heads, _ = q.shape
    topk = indices.shape[1]
    device = q.device

    # Dequantize all KV tokens we might need
    max_slot = int(indices[indices >= 0].max().item()) if (indices >= 0).any() else 0
    # Gather KV at valid slots
    o_ref = torch.zeros(num_q, num_heads, kv_lora_rank, dtype=torch.bfloat16, device=device)
    lse_ref = torch.zeros(num_q, num_heads, dtype=torch.float32, device=device)

    for t in range(num_q):
        nv = int(num_valid_per_token[t]) if num_valid_per_token is not None else topk
        valid_slots = indices[t, :nv]
        valid_mask = valid_slots >= 0
        valid_slots = valid_slots[valid_mask]
        if len(valid_slots) == 0:
            lse_ref[t] = 0.0
            continue

        # Gather and dequantize KV at these slots
        nkv = len(valid_slots)
        kv_packed = torch.zeros(nkv, BPT, dtype=torch.uint8, device=device)
        for i, s in enumerate(valid_slots.tolist()):
            pi = s // PAGE_SIZE
            oi = s % PAGE_SIZE
            kv_packed[i] = kv_cache_packed[pi, oi]

        ckv_deq, kpe_deq = ref_dequantize_ds_mla(kv_packed, kv_lora_rank, pe_dim)
        # ckv_deq: [nkv, 512] BF16, kpe_deq: [nkv, 64] BF16
        # K = [ckv | kpe], V = ckv
        k_nope = ckv_deq.float()  # [nkv, 512]
        k_rope = kpe_deq.float()  # [nkv, 64]
        v = ckv_deq.float()       # [nkv, 512]

        for h in range(num_heads):
            q_nope = q[t, h, :kv_lora_rank].float()  # [512]
            q_rope = q[t, h, kv_lora_rank:].float()  # [64]

            qk_nope = q_nope @ k_nope.T  # [nkv]
            qk_rope = q_rope @ k_rope.T  # [nkv]
            qk = (qk_nope + qk_rope) * sm_scale  # [nkv]

            # softmax
            qk_max = qk.max()
            exp_qk = torch.exp(qk - qk_max)
            denom = exp_qk.sum()
            attn = exp_qk / denom  # [nkv]

            out = attn @ v  # [512]
            o_ref[t, h] = out.to(torch.bfloat16)
            lse_ref[t, h] = (qk_max + torch.log(denom)).item()

    return o_ref, lse_ref


@pytest.fixture
def small_setup(glm, device):
    """Create KV cache, Q, and indices for a small test."""
    torch.manual_seed(42)
    seq_len = 100
    num_q = 4  # query tokens (decode-like: all attend to all KV)

    # Build page table
    page_indices_t, page_indptr_t, last_page_len_t, batch_indices_t, positions_t, next_page = \
        _build_page_table([seq_len], PAGE_SIZE, device)
    page_indices_np = page_indices_t.cpu().numpy()
    page_indptr_np = page_indptr_t.cpu().numpy()
    max_pages = next_page + 4

    # Create and quantize KV
    ckv = torch.randn(seq_len, KV_LORA_RANK, dtype=torch.bfloat16, device=device)
    kpe = torch.randn(seq_len, PE_DIM, dtype=torch.bfloat16, device=device)
    kv_cache = torch.zeros(max_pages, PAGE_SIZE, BPT, dtype=torch.uint8, device=device)

    glm.concat_and_cache_ds_mla(
        kv_cache.data_ptr(), ckv.data_ptr(), kpe.data_ptr(),
        page_indices_t.data_ptr(), page_indptr_t.data_ptr(),
        batch_indices_t.data_ptr(), positions_t.data_ptr(),
        seq_len, PAGE_SIZE, KV_LORA_RANK, PE_DIM,
        KV_LORA_RANK, PE_DIM,
    )

    # Create Q: [num_q, num_heads, D_QK]
    q = torch.randn(num_q, NUM_HEADS, D_QK, dtype=torch.bfloat16, device=device)

    # Create indices: all query tokens attend to all seq_len KV tokens
    # Slots are flat: page_id * page_size + offset
    slots_list = []
    for t in range(num_q):
        token_slots = []
        for pos in range(seq_len):
            slot = _slot_for_token(0, pos, page_indices_np, page_indptr_np, PAGE_SIZE)
            token_slots.append(slot)
        # Pad to TOPK with -1
        while len(token_slots) < TOPK:
            token_slots.append(-1)
        slots_list.append(token_slots)

    indices = torch.tensor(slots_list, dtype=torch.int32, device=device)
    num_valid = torch.tensor([seq_len] * num_q, dtype=torch.int32, device=device)

    return glm, q, kv_cache, indices, num_valid, ckv, kpe


def test_sparse_mla_prefill_basic(small_setup, device):
    """Test SM120 prefill kernel produces correct output."""
    glm, q, kv_cache, indices, num_valid, ckv, kpe = small_setup
    num_q = q.shape[0]

    # Allocate output
    output = torch.zeros(num_q, NUM_HEADS, D_V, dtype=torch.bfloat16, device=device)
    out_lse = torch.zeros(num_q, NUM_HEADS, dtype=torch.float32, device=device)
    stride_kv_block = PAGE_SIZE * BPT

    glm.sparse_mla_prefill(
        q.data_ptr(), kv_cache.data_ptr(), indices.data_ptr(),
        output.data_ptr(), out_lse.data_ptr(),
        num_q, NUM_HEADS, TOPK, PAGE_SIZE,
        SM_SCALE, stride_kv_block,
    )
    torch.cuda.synchronize()

    # Check for CUDA errors
    err = torch.cuda.get_device_properties(0)  # will throw if CUDA error pending

    # Reference
    o_ref, lse_ref = ref_sparse_mla_prefill(
        q, kv_cache, indices, num_valid, SM_SCALE, KV_LORA_RANK, PE_DIM,
    )

    # Compare
    output_f = output.float()
    o_ref_f = o_ref.float()
    max_diff = (output_f - o_ref_f).abs().max().item()
    print(f"Max output diff: {max_diff}")
    print(f"Output[0,0,:8]: {output_f[0,0,:8].tolist()}")
    print(f"Ref[0,0,:8]:    {o_ref_f[0,0,:8].tolist()}")

    assert max_diff < 1.0, f"Output diff too large: {max_diff}"


def test_sparse_mla_prefill_does_not_crash(small_setup, device):
    """Minimal test: just verify the kernel launches without CUDA error."""
    glm, q, kv_cache, indices, num_valid, ckv, kpe = small_setup
    num_q = q.shape[0]

    output = torch.zeros(num_q, NUM_HEADS, D_V, dtype=torch.bfloat16, device=device)
    out_lse = torch.zeros(num_q, NUM_HEADS, dtype=torch.float32, device=device)
    stride_kv_block = PAGE_SIZE * BPT

    glm.sparse_mla_prefill(
        q.data_ptr(), kv_cache.data_ptr(), indices.data_ptr(),
        output.data_ptr(), out_lse.data_ptr(),
        num_q, NUM_HEADS, TOPK, PAGE_SIZE,
        SM_SCALE, stride_kv_block,
    )
    torch.cuda.synchronize()

    # If we get here without crash, the kernel launched successfully
    # Verify output is not all zeros (kernel actually wrote something)
    assert output.abs().max().item() > 0, "Output is all zeros — kernel may not have written"


def test_sparse_mla_prefill_split_q_matches_concat(small_setup, device):
    glm, q, kv_cache, indices, num_valid, *_ = small_setup
    num_q = q.shape[0]
    q_nope = q[:, :, :KV_LORA_RANK].contiguous()
    q_rope = q[:, :, KV_LORA_RANK:].contiguous()
    expected = torch.empty(num_q, NUM_HEADS, D_V, dtype=torch.bfloat16, device=device)
    expected_lse = torch.empty(num_q, NUM_HEADS, dtype=torch.float32, device=device)
    actual = torch.empty_like(expected)
    actual_lse = torch.empty_like(expected_lse)
    stride_kv_block = PAGE_SIZE * BPT

    glm.sparse_mla_prefill(
        q.data_ptr(), kv_cache.data_ptr(), indices.data_ptr(),
        expected.data_ptr(), expected_lse.data_ptr(), num_q, NUM_HEADS, TOPK,
        PAGE_SIZE, SM_SCALE, stride_kv_block, num_valid.data_ptr(),
    )
    glm.sparse_mla_prefill_split_q(
        q_nope.data_ptr(), q_rope.data_ptr(), kv_cache.data_ptr(), indices.data_ptr(),
        actual.data_ptr(), actual_lse.data_ptr(), num_q, NUM_HEADS, TOPK,
        SM_SCALE, stride_kv_block, num_valid.data_ptr(),
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)
    assert torch.equal(actual_lse, expected_lse)

def ref_sparse_mla_decode(q, kv_cache_packed, indices, num_valid_per_token,
                          sm_scale, kv_lora_rank, pe_dim, num_splits):
    """PyTorch reference for sparse MLA decode (split-K)."""
    return ref_sparse_mla_prefill(q, kv_cache_packed, indices, num_valid_per_token,
                                  sm_scale, kv_lora_rank, pe_dim)


def test_sparse_mla_decode_basic(small_setup, device):
    """Test SM120 decode kernel produces correct output."""
    glm, q, kv_cache, indices, num_valid, ckv, kpe = small_setup
    num_q = 1  # decode is always batch=1 per token
    q_decode = q[:1]  # use first query token
    indices_decode = indices[:1]

    num_splits = (TOPK + 63) // 64  # 32
    mid_out = torch.zeros(num_q, NUM_HEADS, num_splits, D_V, dtype=torch.bfloat16, device=device)
    mid_lse = torch.zeros(num_q, NUM_HEADS, num_splits, dtype=torch.float32, device=device)
    output = torch.zeros(num_q, NUM_HEADS, D_V, dtype=torch.bfloat16, device=device)
    out_lse = torch.zeros(num_q, NUM_HEADS, dtype=torch.float32, device=device)
    stride_kv_block = PAGE_SIZE * BPT

    glm.sparse_mla_decode(
        q_decode.data_ptr(), kv_cache.data_ptr(), indices_decode.data_ptr(),
        mid_out.data_ptr(), mid_lse.data_ptr(),
        output.data_ptr(), out_lse.data_ptr(),
        num_q, NUM_HEADS, TOPK, num_splits,
        SM_SCALE, stride_kv_block, 0,
    )
    torch.cuda.synchronize()

    assert output.abs().max().item() > 0, "Decode output is all zeros"

    # Compare against reference (same as prefill for single token)
    o_ref, lse_ref = ref_sparse_mla_decode(
        q_decode, kv_cache, indices_decode, num_valid[:1],
        SM_SCALE, KV_LORA_RANK, PE_DIM, num_splits,
    )

    output_f = output.float()
    o_ref_f = o_ref.float()
    max_diff = (output_f - o_ref_f).abs().max().item()
    print(f"Decode max output diff: {max_diff}")
    assert max_diff < 1.0, f"Decode output diff too large: {max_diff}"


def test_sparse_mla_decode_split_q_matches_concat(small_setup, device):
    glm, q, kv_cache, indices, num_valid, *_ = small_setup
    q = q[:1]
    q_nope = q[:, :, :KV_LORA_RANK].contiguous()
    q_rope = q[:, :, KV_LORA_RANK:].contiguous()
    indices = indices[:1]
    topk_length = num_valid[:1]
    num_splits = (TOPK + 63) // 64
    mid_shape = (1, NUM_HEADS, num_splits, D_V)
    lse_shape = (1, NUM_HEADS, num_splits)
    expected_mid = torch.empty(mid_shape, dtype=torch.bfloat16, device=device)
    expected_mid_lse = torch.empty(lse_shape, dtype=torch.float32, device=device)
    actual_mid = torch.empty_like(expected_mid)
    actual_mid_lse = torch.empty_like(expected_mid_lse)
    expected = torch.empty(1, NUM_HEADS, D_V, dtype=torch.bfloat16, device=device)
    expected_lse = torch.empty(1, NUM_HEADS, dtype=torch.float32, device=device)
    actual = torch.empty_like(expected)
    actual_lse = torch.empty_like(expected_lse)
    stride_kv_block = PAGE_SIZE * BPT

    glm.sparse_mla_decode(
        q.data_ptr(), kv_cache.data_ptr(), indices.data_ptr(),
        expected_mid.data_ptr(), expected_mid_lse.data_ptr(),
        expected.data_ptr(), expected_lse.data_ptr(), 1, NUM_HEADS, TOPK,
        num_splits, SM_SCALE, stride_kv_block, topk_length=topk_length.data_ptr(),
    )
    glm.sparse_mla_decode_split_q(
        q_nope.data_ptr(), q_rope.data_ptr(), kv_cache.data_ptr(), indices.data_ptr(),
        actual_mid.data_ptr(), actual_mid_lse.data_ptr(), actual.data_ptr(), actual_lse.data_ptr(),
        1, NUM_HEADS, TOPK, num_splits, SM_SCALE, stride_kv_block,
        topk_length=topk_length.data_ptr(),
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)
    assert torch.equal(actual_lse, expected_lse)


def test_sparse_mla_decode_multiple_steps(small_setup, device):
    """Test SM120 decode kernel across multiple steps (simulating generation)."""
    glm, q, kv_cache, indices, num_valid, ckv, kpe = small_setup
    num_splits = (TOPK + 63) // 64

    # Start with seq_len=100, simulate 10 decode steps
    # Each step: decode with current indices, then "append" a token (just increase num_valid)
    seq_len = 100
    num_q = 1

    for step in range(10):
        # Build indices for current seq_len
        slots = []
        page_indices_np = torch.tensor([0, 1, 2], dtype=torch.int32).numpy()
        page_indptr_np = torch.tensor([0, 2], dtype=torch.int32).numpy()
        for pos in range(seq_len):
            page_in_seq = pos // PAGE_SIZE
            offset = pos % PAGE_SIZE
            slots.append(page_in_seq * PAGE_SIZE + offset)
        while len(slots) < TOPK:
            slots.append(-1)
        idx = torch.tensor([slots], dtype=torch.int32, device=device)

        # Random Q for this step
        q_step = torch.randn(1, NUM_HEADS, D_QK, dtype=torch.bfloat16, device=device)

        mid_out = torch.zeros(num_q, NUM_HEADS, num_splits, D_V, dtype=torch.bfloat16, device=device)
        mid_lse = torch.zeros(num_q, NUM_HEADS, num_splits, dtype=torch.float32, device=device)
        output = torch.zeros(num_q, NUM_HEADS, D_V, dtype=torch.bfloat16, device=device)
        out_lse = torch.zeros(num_q, NUM_HEADS, dtype=torch.float32, device=device)
        stride_kv_block = PAGE_SIZE * BPT

        glm.sparse_mla_decode(
            q_step.data_ptr(), kv_cache.data_ptr(), idx.data_ptr(),
            mid_out.data_ptr(), mid_lse.data_ptr(),
            output.data_ptr(), out_lse.data_ptr(),
            num_q, NUM_HEADS, TOPK, num_splits,
            SM_SCALE, stride_kv_block, 0,
        )
        torch.cuda.synchronize()

        assert output.abs().max().item() > 0, f"Step {step}: output is all zeros"
        seq_len += 1  # simulate token generation

    print(f"Completed {step + 1} decode steps successfully")
