"""Test FlashInfer MLA kernel: compare 5-token prefill vs 13-token prefill for positions 0-4.
This isolates whether the kernel itself produces different results when the
total token count changes, even though causal masking should give the same
result for positions 0-4.
"""
import torch
import ctypes
import math
import os
import sys
sys.path.insert(0, 'tests/python')
from helpers import GlmOps

HEAD_DIM_CKV = 128
HEAD_DIM_KPE = 64
PAGE_SIZE = 16
num_heads = 8


def _alloc_workspace(glm, float_mb=32, int_mb=8):
    float_ws = glm.alloc(float_mb * 1024 * 1024)
    int_ws = glm.alloc(int_mb * 1024 * 1024)
    pinned_int_ws = glm.alloc_pinned(int_mb * 1024 * 1024)
    return float_ws, int_ws, pinned_int_ws


def run_mla_prefill(glm, device, q_nope, q_pe_rope, ckv_paged, kpe_paged, kv_indices,
                     qo_len, kv_len, num_heads, sm_scale, causal=True):
    """Run MLA prefill via FlashInfer and return output."""
    B = 1
    float_ws, int_ws, pinned_int_ws = _alloc_workspace(glm)

    qo_indptr_h = (ctypes.c_int32 * 2)(0, qo_len)
    kv_indptr_h = (ctypes.c_int32 * 2)(0, kv_len)
    kv_len_h = (ctypes.c_int32 * 1)(kv_len)
    plan_info = (ctypes.c_int64 * 19)()

    glm.mla_prefill_plan(
        float_ws, 32 * 1024 * 1024,
        int_ws, pinned_int_ws, 8 * 1024 * 1024,
        ctypes.addressof(plan_info),
        ctypes.addressof(qo_indptr_h),
        ctypes.addressof(kv_indptr_h),
        ctypes.addressof(kv_len_h),
        B, num_heads, HEAD_DIM_CKV, causal)

    o = torch.empty(qo_len, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)

    glm.mla_prefill_run(
        q_nope.data_ptr(), q_pe_rope.data_ptr(),
        ckv_paged.data_ptr(), kpe_paged.data_ptr(),
        kv_indices.data_ptr(),
        o.data_ptr(),
        float_ws, int_ws, ctypes.addressof(plan_info),
        num_heads, PAGE_SIZE, 1 if causal else 0, sm_scale,
        num_heads * HEAD_DIM_CKV, HEAD_DIM_CKV,
        num_heads * HEAD_DIM_KPE, HEAD_DIM_KPE,
        PAGE_SIZE * HEAD_DIM_CKV, HEAD_DIM_CKV,
        PAGE_SIZE * HEAD_DIM_KPE, HEAD_DIM_KPE,
        num_heads * HEAD_DIM_CKV, HEAD_DIM_CKV,
        HEAD_DIM_CKV, HEAD_DIM_KPE,
        None, 0, 0)

    glm.synchronize()
    glm.free_buf(float_ws)
    glm.free_buf(int_ws)
    glm.free_pinned(pinned_int_ws)
    return o


def mla_prefill_reference(q_nope, q_pe_rope, ckv, kpe_rope, sm_scale, prefix_len=None):
    """Reference MLA attention in PyTorch (float32)."""
    q_len = q_nope.shape[0]
    H = q_nope.shape[1]
    kv_len = ckv.shape[1] if ckv.ndim == 3 else ckv.shape[0]

    if ckv.ndim == 2:
        ckv = ckv.unsqueeze(0)
    if kpe_rope.ndim == 2:
        kpe_rope = kpe_rope.unsqueeze(0)

    ckv_exp = ckv.expand(H, kv_len, HEAD_DIM_CKV)
    kpe_exp = kpe_rope.expand(H, kv_len, -1)

    q_nope_t = q_nope.transpose(0, 1).float()
    q_pe_t = q_pe_rope.transpose(0, 1).float()
    ckv_exp_f = ckv_exp.float()
    kpe_exp_f = kpe_exp.float()

    score_nope = torch.bmm(q_nope_t, ckv_exp_f.transpose(1, 2))
    score_pe = torch.bmm(q_pe_t, kpe_exp_f.transpose(1, 2))
    score = (score_nope + score_pe) * sm_scale

    mask = torch.zeros(q_len, kv_len, dtype=torch.float32, device=score.device)
    if prefix_len is None:
        for i in range(q_len):
            for j in range(kv_len):
                if j > i:
                    mask[i, j] = float('-inf')
    else:
        for i in range(q_len):
            for j in range(kv_len):
                if j > prefix_len + i:
                    mask[i, j] = float('-inf')
    score = score + mask.unsqueeze(0)

    attn = torch.nn.functional.softmax(score, dim=-1).to(q_nope.dtype)
    output = torch.bmm(attn.float(), ckv_exp_f)
    return output.transpose(0, 1)


def pad_kv(ckv, kpe, kv_len, page_size=PAGE_SIZE):
    """Pad KV to page boundary and reshape to paged format."""
    num_pages = (kv_len + page_size - 1) // page_size
    padded_len = num_pages * page_size
    ckv_padded = torch.zeros(padded_len, HEAD_DIM_CKV, dtype=ckv.dtype, device=ckv.device)
    ckv_padded[:kv_len] = ckv
    kpe_padded = torch.zeros(padded_len, HEAD_DIM_KPE, dtype=kpe.dtype, device=kpe.device)
    kpe_padded[:kv_len] = kpe
    ckv_paged = ckv_padded.reshape(num_pages, page_size, HEAD_DIM_CKV)
    kpe_paged = kpe_padded.reshape(num_pages, page_size, HEAD_DIM_KPE)
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device=ckv.device)
    return ckv_paged, kpe_paged, kv_indices


def test_prefix_chunk_vs_full():
    """Compare prefix chunk alone vs full prefill for positions 0-4."""
    device = torch.device('cuda')
    glm = GlmOps('/mnt/storage2/glm.js/build/Release/libglm_ops.so', device_id=0)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM_CKV + HEAD_DIM_KPE)

    S_full = 13
    torch.manual_seed(42)

    # Generate random Q, K, V for the full sequence
    q_nope_full = torch.randn(S_full, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    q_pe_full = torch.randn(S_full, num_heads, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
    ckv_full = torch.randn(S_full, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    kpe_full = torch.randn(S_full, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)

    # Apply RoPE to k_pe
    cos_full, sin_full = _make_rotary_embed(glm, device, HEAD_DIM_KPE // 2, 1, S_full)
    kpe_4d = kpe_full.reshape(1, 1, S_full, HEAD_DIM_KPE)
    kpe_rope_4d = torch.empty_like(kpe_4d)
    glm.apply_rotary_pos_emb(kpe_rope_4d, kpe_4d, cos_full, sin_full, HEAD_DIM_KPE, 1, S_full, 1, 1, interleaved=True)
    kpe_rope_full = kpe_rope_4d.reshape(S_full, HEAD_DIM_KPE)

    # Apply RoPE to q_pe
    q_pe_4d = q_pe_full.reshape(1, num_heads, S_full, HEAD_DIM_KPE)
    q_pe_rope_4d = torch.empty_like(q_pe_4d)
    glm.apply_rotary_pos_emb(q_pe_rope_4d, q_pe_4d, cos_full, sin_full, HEAD_DIM_KPE, num_heads, S_full, 1, 1, interleaved=True)
    q_pe_rope_full = q_pe_rope_4d.reshape(S_full, num_heads, HEAD_DIM_KPE)

    # Test various chunk sizes
    for chunk_size in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
        # Prefix chunk: first chunk_size tokens
        q_nope_chunk = q_nope_full[:chunk_size]
        q_pe_chunk = q_pe_rope_full[:chunk_size]
        ckv_chunk = ckv_full[:chunk_size]
        kpe_chunk = kpe_rope_full[:chunk_size]

        ckv_paged, kpe_paged, kv_indices = pad_kv(ckv_chunk, kpe_chunk, chunk_size)

        o_chunk = run_mla_prefill(glm, device, q_nope_chunk, q_pe_chunk,
                                   ckv_paged, kpe_paged, kv_indices,
                                   chunk_size, chunk_size, num_heads, sm_scale, causal=True)

        # Full prefill
        ckv_paged_full, kpe_paged_full, kv_indices_full = pad_kv(ckv_full, kpe_rope_full, S_full)
        o_full = run_mla_prefill(glm, device, q_nope_full, q_pe_rope_full,
                                  ckv_paged_full, kpe_paged_full, kv_indices_full,
                                  S_full, S_full, num_heads, sm_scale, causal=True)

        # Compare positions 0..chunk_size-1
        o_full_prefix = o_full[:chunk_size]
        max_diff = (o_chunk.cpu().float() - o_full_prefix.cpu().float()).abs().max().item()
        packed_qo = chunk_size * num_heads
        cluster = 2 if packed_qo > 64 else 1
        print(f"chunk={chunk_size:2d} packed_qo={packed_qo:3d} cluster={cluster}: maxDiff={max_diff:.8f}")

    # Also compare against PyTorch reference
    print("\n--- Reference comparison (FlashInfer vs PyTorch) ---")
    for chunk_size in [4, 5, 8, 13]:
        q_nope_chunk = q_nope_full[:chunk_size]
        q_pe_chunk = q_pe_rope_full[:chunk_size]
        ckv_chunk = ckv_full[:chunk_size]
        kpe_chunk = kpe_rope_full[:chunk_size]

        ckv_paged, kpe_paged, kv_indices = pad_kv(ckv_chunk, kpe_chunk, chunk_size)

        o_chunk = run_mla_prefill(glm, device, q_nope_chunk, q_pe_chunk,
                                   ckv_paged, kpe_paged, kv_indices,
                                   chunk_size, chunk_size, num_heads, sm_scale, causal=True)

        ref_chunk = mla_prefill_reference(q_nope_chunk, q_pe_chunk,
                                           ckv_chunk.unsqueeze(0), kpe_chunk.unsqueeze(0),
                                           sm_scale)

        max_diff_ref = (o_chunk.cpu().float() - ref_chunk.cpu().float()).abs().max().item()
        print(f"chunk={chunk_size:2d}: FlashInfer vs PyTorch ref: maxDiff={max_diff_ref:.8f}")

    glm.free_buf(glm.alloc(1))


def _make_rotary_embed(glm, device, dim_half, batch, seq_len, theta=1000000.0):
    inv_freq = (1.0 / (theta ** (torch.arange(0, dim_half * 2, 2, dtype=torch.float32, device=device) / (dim_half * 2)))).to(torch.bfloat16)
    position_ids = torch.arange(seq_len, dtype=torch.int32, device=device).unsqueeze(0).expand(batch, -1)
    cos_out = torch.empty(batch, seq_len, dim_half * 2, dtype=torch.bfloat16, device=device)
    sin_out = torch.empty(batch, seq_len, dim_half * 2, dtype=torch.bfloat16, device=device)
    glm.rotary_embedding(cos_out, sin_out, inv_freq, position_ids, dim_half, batch, seq_len)
    return cos_out, sin_out


if __name__ == '__main__':
    test_prefix_chunk_vs_full()
