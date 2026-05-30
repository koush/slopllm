import torch
import pytest
import ctypes
import math
import numpy as np
from helpers import ATOL, RTOL, GlmOps

HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
PAGE_SIZE = 1

MASK_MODE_CAUSAL = 1
MASK_MODE_CUSTOM = 2
MASK_MODE_CAUSAL_CUSTOM = 4


def _alloc_workspace(glm, float_mb=32, int_mb=8):
    float_ws = glm.alloc(float_mb * 1024 * 1024)
    int_ws = glm.alloc(int_mb * 1024 * 1024)
    pinned_int_ws = glm.alloc_pinned(int_mb * 1024 * 1024)
    return float_ws, int_ws, pinned_int_ws


def pack_bits_little_endian(mask_flat, qo_len, kv_len):
    total_bits = qo_len * kv_len
    byte_len = (total_bits + 7) // 8
    data = np.zeros(byte_len, dtype=np.uint8)
    for q in range(qo_len):
        for k in range(kv_len):
            offset = q * kv_len + k
            if mask_flat[offset]:
                data[offset >> 3] |= np.uint8(1 << (offset & 7))
    return data


def build_tree_suffix_mask(parents, num_prefill_tokens):
    mask = np.zeros(num_prefill_tokens * num_prefill_tokens, dtype=bool)
    for q in range(num_prefill_tokens):
        mask[q * num_prefill_tokens + q] = True
        cur = q
        while parents[cur]:
            cur = parents[cur][0]
            mask[q * num_prefill_tokens + cur] = True
    return mask


def build_full_mask_from_suffix(suffix_mask, qo_len, kv_len):
    prefix_len = kv_len - qo_len
    mask = np.zeros(qo_len * kv_len, dtype=bool)
    for q in range(qo_len):
        mask[q * kv_len : q * kv_len + prefix_len] = True
        for k in range(qo_len):
            mask[q * kv_len + prefix_len + k] = suffix_mask[q * qo_len + k]
    return mask


def build_causal_suffix_mask(qo_len):
    mask = np.zeros(qo_len * qo_len, dtype=bool)
    for q in range(qo_len):
        for k in range(q + 1):
            mask[q * qo_len + k] = True
    return mask


def run_mla_prefill(glm, device, q_nope_bf16, q_pe_bf16, ckv_bf16, kpe_bf16,
                    batch_size, seq_len, kv_len, num_heads, causal,
                    custom_mask_data=None, mask_indptr_data=None, mask_mode=None):
    num_pages = math.ceil(kv_len / PAGE_SIZE)
    float_ws, int_ws, pinned_int_ws = _alloc_workspace(glm)

    qo_indptr_h = (ctypes.c_int32 * (batch_size + 1))(*([0] + [seq_len] * batch_size))
    kv_indptr_h = (ctypes.c_int32 * (batch_size + 1))(*([0] + [num_pages] * batch_size))
    kv_len_h = (ctypes.c_int32 * batch_size)(*([kv_len] * batch_size))
    kv_indices = torch.arange(num_pages * batch_size, dtype=torch.int32, device=device)
    plan_info = (ctypes.c_int64 * 19)()

    glm.mla_prefill_plan(
        float_ws, 32 * 1024 * 1024,
        int_ws, pinned_int_ws, 8 * 1024 * 1024,
        ctypes.addressof(plan_info),
        ctypes.addressof(qo_indptr_h),
        ctypes.addressof(kv_indptr_h),
        ctypes.addressof(kv_len_h),
        batch_size, num_heads, HEAD_DIM_CKV, causal)

    o = torch.empty(batch_size * seq_len, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)

    sm_scale = 1.0 / math.sqrt(HEAD_DIM_CKV + HEAD_DIM_KPE)

    if mask_mode is None:
        if custom_mask_data is not None:
            mask_mode = MASK_MODE_CUSTOM
        else:
            mask_mode = MASK_MODE_CAUSAL if causal else 0

    custom_mask_ptr = 0
    mask_indptr_ptr = 0
    if custom_mask_data is not None:
        mask_tensor = torch.from_numpy(custom_mask_data).to(device)
        custom_mask_ptr = mask_tensor.data_ptr()
    if mask_indptr_data is not None:
        indptr_tensor = torch.from_numpy(mask_indptr_data).to(device)
        mask_indptr_ptr = indptr_tensor.data_ptr()

    glm.mla_prefill_run(
        q_nope_bf16.data_ptr(), q_pe_bf16.data_ptr(),
        ckv_bf16.data_ptr(), kpe_bf16.data_ptr(),
        kv_indices.data_ptr(),
        o.data_ptr(),
        float_ws, int_ws, ctypes.addressof(plan_info),
        num_heads, PAGE_SIZE, mask_mode, sm_scale,
        num_heads * HEAD_DIM_CKV, HEAD_DIM_CKV,
        num_heads * HEAD_DIM_KPE, HEAD_DIM_KPE,
        PAGE_SIZE * HEAD_DIM_CKV, HEAD_DIM_CKV,
        PAGE_SIZE * HEAD_DIM_KPE, HEAD_DIM_KPE,
        num_heads * HEAD_DIM_CKV, HEAD_DIM_CKV,
        HEAD_DIM_CKV, HEAD_DIM_KPE,
        None, 0, 0,
        custom_mask_ptr, mask_indptr_ptr)

    glm.synchronize()
    glm.free_buf(float_ws)
    glm.free_buf(int_ws)
    glm.free_pinned(pinned_int_ws)

    return o


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_causal_custom_tree_mask_with_prefix(glm, device):
    B = 1
    num_heads = 4
    prefix_len = 32
    num_prefill_tokens = 15
    qo_len = num_prefill_tokens
    kv_len = prefix_len + num_prefill_tokens
    num_pages = math.ceil(kv_len / PAGE_SIZE)

    torch.manual_seed(42)
    q_nope = torch.randn(B * qo_len, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(B * qo_len, num_heads, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
    ckv = torch.randn(num_pages * PAGE_SIZE, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    kpe = torch.randn(num_pages * PAGE_SIZE, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)

    parents = [[], [0], [0], [1], [1], [2], [2], [3], [3], [4], [4], [5], [5], [6], [6]]

    suffix_mask = build_tree_suffix_mask(parents, num_prefill_tokens)
    packed_suffix = pack_bits_little_endian(suffix_mask, qo_len, qo_len)
    suffix_indptr = np.array([0, len(packed_suffix)], dtype=np.int32)

    full_mask = build_full_mask_from_suffix(suffix_mask, qo_len, kv_len)
    packed_full = pack_bits_little_endian(full_mask, qo_len, kv_len)
    full_indptr = np.array([0, len(packed_full)], dtype=np.int32)

    o_cc = run_mla_prefill(glm, device, q_nope, q_pe, ckv, kpe,
                           B, qo_len, kv_len, num_heads, causal=True,
                           custom_mask_data=packed_suffix,
                           mask_indptr_data=suffix_indptr,
                           mask_mode=MASK_MODE_CAUSAL_CUSTOM)

    o_custom = run_mla_prefill(glm, device, q_nope, q_pe, ckv, kpe,
                               B, qo_len, kv_len, num_heads, causal=False,
                               custom_mask_data=packed_full,
                               mask_indptr_data=full_indptr,
                               mask_mode=MASK_MODE_CUSTOM)

    torch.testing.assert_close(o_cc.cpu(), o_custom.cpu(), atol=0.05, rtol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_causal_custom_causal_suffix_matches_builtin_causal(glm, device):
    B = 1
    num_heads = 4
    prefix_len = 32
    num_prefill_tokens = 7
    qo_len = num_prefill_tokens
    kv_len = prefix_len + num_prefill_tokens
    num_pages = math.ceil(kv_len / PAGE_SIZE)

    torch.manual_seed(42)
    q_nope = torch.randn(B * qo_len, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(B * qo_len, num_heads, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
    ckv = torch.randn(num_pages * PAGE_SIZE, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    kpe = torch.randn(num_pages * PAGE_SIZE, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)

    suffix_mask = build_causal_suffix_mask(qo_len)
    packed_suffix = pack_bits_little_endian(suffix_mask, qo_len, qo_len)
    suffix_indptr = np.array([0, len(packed_suffix)], dtype=np.int32)

    o_cc = run_mla_prefill(glm, device, q_nope, q_pe, ckv, kpe,
                           B, qo_len, kv_len, num_heads, causal=True,
                           custom_mask_data=packed_suffix,
                           mask_indptr_data=suffix_indptr,
                           mask_mode=MASK_MODE_CAUSAL_CUSTOM)

    o_causal = run_mla_prefill(glm, device, q_nope, q_pe, ckv, kpe,
                               B, qo_len, kv_len, num_heads, causal=True)

    torch.testing.assert_close(o_cc.cpu(), o_causal.cpu(), atol=0.05, rtol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_causal_custom_no_crash(glm, device):
    B = 1
    num_heads = 4
    prefix_len = 32
    num_prefill_tokens = 15
    qo_len = num_prefill_tokens
    kv_len = prefix_len + num_prefill_tokens
    num_pages = math.ceil(kv_len / PAGE_SIZE)

    torch.manual_seed(42)
    q_nope = torch.randn(B * qo_len, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(B * qo_len, num_heads, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
    ckv = torch.randn(num_pages * PAGE_SIZE, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    kpe = torch.randn(num_pages * PAGE_SIZE, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)

    parents = [[], [0], [0], [1], [1], [2], [2], [3], [3], [4], [4], [5], [5], [6], [6]]
    suffix_mask = build_tree_suffix_mask(parents, num_prefill_tokens)
    packed_suffix = pack_bits_little_endian(suffix_mask, qo_len, qo_len)
    suffix_indptr = np.array([0, len(packed_suffix)], dtype=np.int32)

    o = run_mla_prefill(glm, device, q_nope, q_pe, ckv, kpe,
                        B, qo_len, kv_len, num_heads, causal=True,
                        custom_mask_data=packed_suffix,
                        mask_indptr_data=suffix_indptr,
                        mask_mode=MASK_MODE_CAUSAL_CUSTOM)

    assert o.shape == (B * qo_len, num_heads, HEAD_DIM_CKV)
    assert torch.isfinite(o).all(), "output contains non-finite values"
    assert (o != 0).any(), "output should not be all zeros"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_causal_custom_all_ones_suffix_matches_custom_all_ones(glm, device):
    B = 1
    num_heads = 4
    prefix_len = 32
    num_prefill_tokens = 7
    qo_len = num_prefill_tokens
    kv_len = prefix_len + num_prefill_tokens
    num_pages = math.ceil(kv_len / PAGE_SIZE)

    torch.manual_seed(42)
    q_nope = torch.randn(B * qo_len, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(B * qo_len, num_heads, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
    ckv = torch.randn(num_pages * PAGE_SIZE, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    kpe = torch.randn(num_pages * PAGE_SIZE, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)

    all_ones_suffix = np.ones(qo_len * qo_len, dtype=bool)
    packed_suffix = pack_bits_little_endian(all_ones_suffix, qo_len, qo_len)
    suffix_indptr = np.array([0, len(packed_suffix)], dtype=np.int32)

    all_ones_full = np.ones(qo_len * kv_len, dtype=bool)
    packed_full = pack_bits_little_endian(all_ones_full, qo_len, kv_len)
    full_indptr = np.array([0, len(packed_full)], dtype=np.int32)

    o_cc = run_mla_prefill(glm, device, q_nope, q_pe, ckv, kpe,
                           B, qo_len, kv_len, num_heads, causal=True,
                           custom_mask_data=packed_suffix,
                           mask_indptr_data=suffix_indptr,
                           mask_mode=MASK_MODE_CAUSAL_CUSTOM)

    o_custom = run_mla_prefill(glm, device, q_nope, q_pe, ckv, kpe,
                               B, qo_len, kv_len, num_heads, causal=False,
                               custom_mask_data=packed_full,
                               mask_indptr_data=full_indptr,
                               mask_mode=MASK_MODE_CUSTOM)

    torch.testing.assert_close(o_cc.cpu(), o_custom.cpu(), atol=0.05, rtol=0.02)
