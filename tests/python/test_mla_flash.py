import torch
import pytest
import ctypes
import math
from helpers import ATOL, RTOL, GlmOps

HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64
PAGE_SIZE = 1


def _make_rotary_embed(glm, device, dim_half, batch, seq_len, theta=1000000.0):
    inv_freq = (1.0 / (theta ** (torch.arange(0, dim_half * 2, 2, dtype=torch.float32, device=device) / (dim_half * 2)))).to(torch.bfloat16)
    position_ids = torch.arange(seq_len, dtype=torch.int32, device=device).unsqueeze(0).expand(batch, -1)
    cos_out = torch.empty(batch, seq_len, dim_half * 2, dtype=torch.bfloat16, device=device)
    sin_out = torch.empty(batch, seq_len, dim_half * 2, dtype=torch.bfloat16, device=device)
    glm.rotary_embedding(cos_out, sin_out, inv_freq, position_ids, dim_half, batch, seq_len)
    return cos_out, sin_out


def apply_rotary_pos_emb_torch(x, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def apply_rotary_pos_emb_torch_interleaved(x, cos, sin, unsqueeze_dim=1):
    dim_half = cos.shape[-1] // 2
    cos_half = cos[..., :dim_half]
    sin_half = sin[..., :dim_half]
    if unsqueeze_dim == 1:
        cos_half = cos_half.unsqueeze(1)
        sin_half = sin_half.unsqueeze(1)
    elif unsqueeze_dim == 2:
        cos_half = cos_half.unsqueeze(2)
        sin_half = sin_half.unsqueeze(2)
    x1 = x[..., 0::2].float()
    x2 = x[..., 1::2].float()
    o1 = x1 * cos_half.float() - x2 * sin_half.float()
    o2 = x2 * cos_half.float() + x1 * sin_half.float()
    return torch.stack((o1, o2), dim=-1).flatten(-2).to(x.dtype)


def _apply_interleaved_rope(x, cos, sin):
    dim_half = cos.shape[-1] // 2
    cos_half = cos[..., :dim_half]
    sin_half = sin[..., :dim_half]
    while cos_half.ndim < x.ndim:
        cos_half = cos_half.unsqueeze(0)
        sin_half = sin_half.unsqueeze(0)
    x1 = x[..., 0::2].float()
    x2 = x[..., 1::2].float()
    o1 = x1 * cos_half.float() - x2 * sin_half.float()
    o2 = x2 * cos_half.float() + x1 * sin_half.float()
    return torch.stack((o1, o2), dim=-1).flatten(-2).to(x.dtype)


def mla_prefill_reference(q_nope, q_pe_rope, ckv, kpe_rope, sm_scale, causal=True):
    BS, H, D_CKV = q_nope.shape
    S = ckv.shape[1]
    D_KPE = q_pe_rope.shape[-1]

    ckv_exp = ckv.expand(H, S, D_CKV)
    kpe_exp = kpe_rope.expand(H, S, D_KPE)

    score_nope = torch.einsum('bhd,hkd->bhk', q_nope, ckv_exp)
    score_pe = torch.einsum('bhd,hkd->bhk', q_pe_rope, kpe_exp)
    score = (score_nope + score_pe) * sm_scale

    if causal:
        mask = torch.triu(torch.full((BS, S), float('-inf'), device=score.device, dtype=score.dtype), diagonal=1)
        score = score + mask.unsqueeze(1)

    attn = torch.nn.functional.softmax(score.float(), dim=-1).to(q_nope.dtype)
    output = torch.einsum('bhk,hkd->bhd', attn, ckv_exp)
    return output


def mla_decode_reference(q_nope_absorbed, q_pe_rope, ckv, kpe, positions, sm_scale, rope_theta=1000000.0):
    B, H, D_CKV = q_nope_absorbed.shape
    S = ckv.shape[1]
    D_KPE = q_pe_rope.shape[-1]

    dim_half = D_KPE // 2
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, D_KPE, 2, dtype=torch.float32, device=ckv.device) / D_KPE))
    freqs = torch.outer(positions.float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos_emb = emb.cos().to(ckv.dtype)
    sin_emb = emb.sin().to(ckv.dtype)

    kpe_cos_half = cos_emb[:, :dim_half].unsqueeze(0)
    kpe_sin_half = sin_emb[:, :dim_half].unsqueeze(0)
    kpe_x1 = kpe[..., 0::2].float()
    kpe_x2 = kpe[..., 1::2].float()
    kpe_o1 = kpe_x1 * kpe_cos_half.float() - kpe_x2 * kpe_sin_half.float()
    kpe_o2 = kpe_x2 * kpe_cos_half.float() + kpe_x1 * kpe_sin_half.float()
    kpe_rope = torch.stack((kpe_o1, kpe_o2), dim=-1).flatten(-2).to(kpe.dtype)

    q_nope_3d = q_nope_absorbed.reshape(B * H, 1, D_CKV)
    q_pe_3d = q_pe_rope.reshape(B * H, 1, D_KPE)
    ckv_3d = ckv.reshape(B, 1, S, D_CKV).expand(B, H, S, D_CKV).reshape(B * H, S, D_CKV)
    kpe_3d = kpe_rope.reshape(B, 1, S, D_KPE).expand(B, H, S, D_KPE).reshape(B * H, S, D_KPE)

    score = torch.bmm(q_nope_3d, ckv_3d.transpose(1, 2)) + torch.bmm(q_pe_3d, kpe_3d.transpose(1, 2))
    score = score * sm_scale
    attn = torch.nn.functional.softmax(score.float(), dim=-1).to(ckv.dtype)
    output = torch.bmm(attn, ckv_3d)
    return output.reshape(B, H, D_CKV)


def _alloc_workspace(glm, float_mb=32, int_mb=8):
    float_ws = glm.alloc(float_mb * 1024 * 1024)
    int_ws = glm.alloc(int_mb * 1024 * 1024)
    pinned_int_ws = glm.alloc_pinned(int_mb * 1024 * 1024)
    return float_ws, int_ws, pinned_int_ws


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mla_prefill_causal(glm, device):
    B = 1
    S = 8
    num_heads = 4
    sm_scale = 1.0 / math.sqrt(HEAD_DIM_CKV + HEAD_DIM_KPE)

    torch.manual_seed(42)
    q_nope = torch.randn(B * S, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(B * S, num_heads, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
    ckv = torch.randn(S, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    kpe = torch.randn(S, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)

    cos, sin = _make_rotary_embed(glm, device, HEAD_DIM_KPE // 2, B, S)
    q_pe_4d = q_pe.reshape(B, num_heads, S, HEAD_DIM_KPE)
    q_pe_rope = torch.empty_like(q_pe_4d)
    glm.apply_rotary_pos_emb(q_pe_rope, q_pe_4d, cos, sin, HEAD_DIM_KPE, num_heads, S, B, 1, interleaved=True)
    q_pe_rope = q_pe_rope.reshape(B * S, num_heads, HEAD_DIM_KPE)

    kpe_4d = kpe.reshape(B, 1, S, HEAD_DIM_KPE)
    kpe_rope_4d = torch.empty_like(kpe_4d)
    glm.apply_rotary_pos_emb(kpe_rope_4d, kpe_4d, cos, sin, HEAD_DIM_KPE, 1, S, B, 1, interleaved=True)
    kpe_rope = kpe_rope_4d.reshape(S, HEAD_DIM_KPE)

    ckv_paged = ckv.reshape(S, PAGE_SIZE, HEAD_DIM_CKV)
    kpe_paged = kpe_rope.reshape(S, PAGE_SIZE, HEAD_DIM_KPE)

    qo_indptr_h = (ctypes.c_int32 * 2)(0, S)
    kv_indptr_h = (ctypes.c_int32 * 2)(0, S)
    kv_len_h = (ctypes.c_int32 * 1)(S)
    kv_indices = torch.arange(S, dtype=torch.int32, device=device)
    plan_info = (ctypes.c_int64 * 19)()

    float_ws, int_ws, pinned_int_ws = _alloc_workspace(glm)

    glm.mla_prefill_plan(
        float_ws, 32 * 1024 * 1024,
        int_ws, pinned_int_ws, 8 * 1024 * 1024,
        ctypes.addressof(plan_info),
        ctypes.addressof(qo_indptr_h),
        ctypes.addressof(kv_indptr_h),
        ctypes.addressof(kv_len_h),
        B, num_heads, HEAD_DIM_CKV, True)

    o = torch.empty(B * S, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)

    glm.mla_prefill_run(
        q_nope.data_ptr(), q_pe_rope.data_ptr(),
        ckv_paged.data_ptr(), kpe_paged.data_ptr(),
        kv_indices.data_ptr(),
        o.data_ptr(),
        float_ws, int_ws, ctypes.addressof(plan_info),
        num_heads, PAGE_SIZE, 1, sm_scale,
        num_heads * HEAD_DIM_CKV, HEAD_DIM_CKV,
        num_heads * HEAD_DIM_KPE, HEAD_DIM_KPE,
        PAGE_SIZE * HEAD_DIM_CKV, HEAD_DIM_CKV,
        PAGE_SIZE * HEAD_DIM_KPE, HEAD_DIM_KPE,
        num_heads * HEAD_DIM_CKV, HEAD_DIM_CKV,
        HEAD_DIM_CKV, HEAD_DIM_KPE,
        None, 0, 0)

    glm.synchronize()

    ref_output = mla_prefill_reference(
        q_nope, q_pe_rope,
        ckv.unsqueeze(0), kpe_rope.unsqueeze(0),
        sm_scale, causal=True)

    torch.testing.assert_close(o.cpu(), ref_output.cpu(), atol=0.05, rtol=0.02)

    glm.free_buf(float_ws)
    glm.free_buf(int_ws)
    glm.free_pinned(pinned_int_ws)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mla_prefill_noncausal(glm, device):
    B = 1
    S = 8
    num_heads = 4
    sm_scale = 1.0 / math.sqrt(HEAD_DIM_CKV + HEAD_DIM_KPE)

    torch.manual_seed(42)
    q_nope = torch.randn(B * S, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(B * S, num_heads, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
    ckv = torch.randn(S, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    kpe = torch.randn(S, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)

    cos, sin = _make_rotary_embed(glm, device, HEAD_DIM_KPE // 2, B, S)
    q_pe_4d = q_pe.reshape(B, num_heads, S, HEAD_DIM_KPE)
    q_pe_rope = torch.empty_like(q_pe_4d)
    glm.apply_rotary_pos_emb(q_pe_rope, q_pe_4d, cos, sin, HEAD_DIM_KPE, num_heads, S, B, 1, interleaved=True)
    q_pe_rope = q_pe_rope.reshape(B * S, num_heads, HEAD_DIM_KPE)

    kpe_4d = kpe.reshape(B, 1, S, HEAD_DIM_KPE)
    kpe_rope_4d = torch.empty_like(kpe_4d)
    glm.apply_rotary_pos_emb(kpe_rope_4d, kpe_4d, cos, sin, HEAD_DIM_KPE, 1, S, B, 1, interleaved=True)
    kpe_rope = kpe_rope_4d.reshape(S, HEAD_DIM_KPE)

    ckv_paged = ckv.reshape(S, PAGE_SIZE, HEAD_DIM_CKV)
    kpe_paged = kpe_rope.reshape(S, PAGE_SIZE, HEAD_DIM_KPE)

    qo_indptr_h = (ctypes.c_int32 * 2)(0, S)
    kv_indptr_h = (ctypes.c_int32 * 2)(0, S)
    kv_len_h = (ctypes.c_int32 * 1)(S)
    kv_indices = torch.arange(S, dtype=torch.int32, device=device)
    plan_info = (ctypes.c_int64 * 19)()

    float_ws, int_ws, pinned_int_ws = _alloc_workspace(glm)

    glm.mla_prefill_plan(
        float_ws, 32 * 1024 * 1024,
        int_ws, pinned_int_ws, 8 * 1024 * 1024,
        ctypes.addressof(plan_info),
        ctypes.addressof(qo_indptr_h),
        ctypes.addressof(kv_indptr_h),
        ctypes.addressof(kv_len_h),
        B, num_heads, HEAD_DIM_CKV, False)

    o = torch.empty(B * S, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)

    glm.mla_prefill_run(
        q_nope.data_ptr(), q_pe_rope.data_ptr(),
        ckv_paged.data_ptr(), kpe_paged.data_ptr(),
        kv_indices.data_ptr(),
        o.data_ptr(),
        float_ws, int_ws, ctypes.addressof(plan_info),
        num_heads, PAGE_SIZE, 0, sm_scale,
        num_heads * HEAD_DIM_CKV, HEAD_DIM_CKV,
        num_heads * HEAD_DIM_KPE, HEAD_DIM_KPE,
        PAGE_SIZE * HEAD_DIM_CKV, HEAD_DIM_CKV,
        PAGE_SIZE * HEAD_DIM_KPE, HEAD_DIM_KPE,
        num_heads * HEAD_DIM_CKV, HEAD_DIM_CKV,
        HEAD_DIM_CKV, HEAD_DIM_KPE,
        None, 0, 0)

    glm.synchronize()

    ref_output = mla_prefill_reference(
        q_nope, q_pe_rope,
        ckv.unsqueeze(0), kpe_rope.unsqueeze(0),
        sm_scale, causal=False)

    torch.testing.assert_close(o.cpu(), ref_output.cpu(), atol=0.05, rtol=0.02)

    glm.free_buf(float_ws)
    glm.free_buf(int_ws)
    glm.free_pinned(pinned_int_ws)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mla_decode_single(glm, device):
    B = 1
    S = 8
    num_heads = 4
    sm_scale = 1.0 / math.sqrt(HEAD_DIM_CKV + HEAD_DIM_KPE)
    rope_theta = 1000000.0

    torch.manual_seed(42)
    q_nope = torch.randn(B, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(B, num_heads, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
    ckv = torch.randn(S, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    kpe = torch.randn(S, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)

    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, HEAD_DIM_KPE, 2, dtype=torch.float32, device=device) / HEAD_DIM_KPE))

    decode_pos = torch.tensor([S - 1], dtype=torch.float32, device=device)
    freqs_q = torch.outer(decode_pos, inv_freq)
    emb_q = torch.cat([freqs_q, freqs_q], dim=-1)
    cos_q = emb_q.cos().to(torch.bfloat16).reshape(1, 1, HEAD_DIM_KPE)
    sin_q = emb_q.sin().to(torch.bfloat16).reshape(1, 1, HEAD_DIM_KPE)
    q_pe_rope = _apply_interleaved_rope(q_pe, cos_q, sin_q)

    positions_k = torch.arange(S, dtype=torch.float32, device=device)
    freqs_k = torch.outer(positions_k, inv_freq)
    emb_k = torch.cat([freqs_k, freqs_k], dim=-1)
    cos_k = emb_k.cos().to(torch.bfloat16)
    sin_k = emb_k.sin().to(torch.bfloat16)
    kpe_rope = _apply_interleaved_rope(kpe, cos_k, sin_k)

    ckv_paged = ckv.reshape(S, PAGE_SIZE, HEAD_DIM_CKV).contiguous()
    kpe_paged = kpe_rope.reshape(S, PAGE_SIZE, HEAD_DIM_KPE).contiguous()

    num_pages = S
    indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    indptr_h = (ctypes.c_int32 * 2)(0, num_pages)
    last_page_len_d = torch.tensor([PAGE_SIZE], dtype=torch.int32, device=device)

    float_ws, int_ws, pinned_int_ws = _alloc_workspace(glm)
    plan_info = (ctypes.c_int64 * 10)()

    glm.mla_decode_plan(
        float_ws, 32 * 1024 * 1024,
        int_ws, pinned_int_ws, 8 * 1024 * 1024,
        ctypes.addressof(plan_info),
        ctypes.addressof(indptr_h),
        B, num_heads, PAGE_SIZE, False,
        head_dim_ckv=HEAD_DIM_CKV, head_dim_kpe=HEAD_DIM_KPE)

    o = torch.empty(B, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    indptr_d = torch.tensor([0, num_pages], dtype=torch.int32, device=device)

    glm.mla_decode_run(
        q_nope.data_ptr(), q_pe_rope.data_ptr(),
        ckv_paged.data_ptr(), kpe_paged.data_ptr(),
        indices.data_ptr(), indptr_d.data_ptr(), last_page_len_d.data_ptr(),
        o.data_ptr(),
        float_ws, int_ws, ctypes.addressof(plan_info),
        B, num_heads, PAGE_SIZE, sm_scale,
        head_dim_ckv=HEAD_DIM_CKV, head_dim_kpe=HEAD_DIM_KPE)

    glm.synchronize()

    positions = torch.arange(S, dtype=torch.int32, device=device)
    ref_output = mla_decode_reference(q_nope, q_pe_rope, ckv.unsqueeze(0), kpe.unsqueeze(0), positions, sm_scale)

    torch.testing.assert_close(o.cpu(), ref_output.cpu(), atol=0.05, rtol=0.02)

    glm.free_buf(float_ws)
    glm.free_buf(int_ws)
    glm.free_pinned(pinned_int_ws)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mla_decode_batch(glm, device):
    B = 3
    seq_lens = [4, 7, 2]
    num_heads = 4
    sm_scale = 1.0 / math.sqrt(HEAD_DIM_CKV + HEAD_DIM_KPE)
    rope_theta = 1000000.0

    torch.manual_seed(123)
    max_S = max(seq_lens)

    q_nope = torch.randn(B, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(B, num_heads, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)

    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, HEAD_DIM_KPE, 2, dtype=torch.float32, device=device) / HEAD_DIM_KPE))

    for b in range(B):
        pos = seq_lens[b]
        decode_pos = torch.tensor([pos - 1], dtype=torch.float32, device=device)
        freqs_q = torch.outer(decode_pos, inv_freq)
        emb_q = torch.cat([freqs_q, freqs_q], dim=-1)
        cos_q = emb_q.cos().to(torch.bfloat16).reshape(1, 1, HEAD_DIM_KPE)
        sin_q = emb_q.sin().to(torch.bfloat16).reshape(1, 1, HEAD_DIM_KPE)
        q_pe[b] = _apply_interleaved_rope(q_pe[b:b+1], cos_q, sin_q)

    all_ckv = []
    all_kpe = []
    all_kpe_raw = []
    for b in range(B):
        all_ckv.append(torch.randn(seq_lens[b], HEAD_DIM_CKV, dtype=torch.bfloat16, device=device))
        raw_kpe = torch.randn(seq_lens[b], HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
        all_kpe_raw.append(raw_kpe)
        positions_k = torch.arange(seq_lens[b], dtype=torch.float32, device=device)
        freqs_k = torch.outer(positions_k, inv_freq)
        emb_k = torch.cat([freqs_k, freqs_k], dim=-1)
        cos_k = emb_k.cos().to(torch.bfloat16)
        sin_k = emb_k.sin().to(torch.bfloat16)
        kpe_rope = _apply_interleaved_rope(raw_kpe, cos_k, sin_k)
        all_kpe.append(kpe_rope)

    total_pages = sum(seq_lens)
    ckv_paged = torch.cat(all_ckv, dim=0).reshape(total_pages, PAGE_SIZE, HEAD_DIM_CKV).contiguous()
    kpe_paged = torch.cat(all_kpe, dim=0).reshape(total_pages, PAGE_SIZE, HEAD_DIM_KPE).contiguous()

    indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    indptr_h_data = [0]
    for sl in seq_lens:
        indptr_h_data.append(indptr_h_data[-1] + sl)
    indptr_h = (ctypes.c_int32 * (B + 1))(*indptr_h_data)
    last_page_len_d = torch.tensor([PAGE_SIZE] * B, dtype=torch.int32, device=device)
    indptr_d = torch.tensor(indptr_h_data, dtype=torch.int32, device=device)

    float_ws, int_ws, pinned_int_ws = _alloc_workspace(glm)
    plan_info = (ctypes.c_int64 * 10)()

    glm.mla_decode_plan(
        float_ws, 32 * 1024 * 1024,
        int_ws, pinned_int_ws, 8 * 1024 * 1024,
        ctypes.addressof(plan_info),
        ctypes.addressof(indptr_h),
        B, num_heads, PAGE_SIZE, False,
        head_dim_ckv=HEAD_DIM_CKV, head_dim_kpe=HEAD_DIM_KPE)

    o = torch.empty(B, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)

    glm.mla_decode_run(
        q_nope.data_ptr(), q_pe.data_ptr(),
        ckv_paged.data_ptr(), kpe_paged.data_ptr(),
        indices.data_ptr(), indptr_d.data_ptr(), last_page_len_d.data_ptr(),
        o.data_ptr(),
        float_ws, int_ws, ctypes.addressof(plan_info),
        B, num_heads, PAGE_SIZE, sm_scale,
        head_dim_ckv=HEAD_DIM_CKV, head_dim_kpe=HEAD_DIM_KPE)

    glm.synchronize()

    ref_outputs = []
    for b in range(B):
        positions = torch.arange(seq_lens[b], dtype=torch.int32, device=device)
        ref_out = mla_decode_reference(
            q_nope[b:b+1], q_pe[b:b+1],
            all_ckv[b].unsqueeze(0), all_kpe_raw[b].unsqueeze(0),
            positions, sm_scale)
        ref_outputs.append(ref_out.squeeze(0))
    ref_output = torch.stack(ref_outputs, dim=0)

    torch.testing.assert_close(o.cpu(), ref_output.cpu(), atol=0.05, rtol=0.02)

    glm.free_buf(float_ws)
    glm.free_buf(int_ws)
    glm.free_pinned(pinned_int_ws)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mla_kv_cache_append(glm, device):
    B = 2
    S = 4
    num_heads = 4
    page_size = 1

    torch.manual_seed(42)
    append_ckv = torch.randn(B * S, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    append_kpe = torch.randn(B * S, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)

    num_pages = B * S
    ckv_cache = torch.zeros(num_pages, page_size, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    kpe_cache = torch.zeros(num_pages, page_size, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)

    indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    indptr = torch.tensor([0, S, B * S], dtype=torch.int32, device=device)
    last_page_len = torch.tensor([page_size] * B, dtype=torch.int32, device=device)

    batch_indices = torch.cat([torch.full((S,), b, dtype=torch.int32, device=device) for b in range(B)])
    positions = torch.cat([torch.arange(S, dtype=torch.int32, device=device) for _ in range(B)])

    glm.mla_kv_cache_append(
        ckv_cache.data_ptr(), kpe_cache.data_ptr(),
        indices.data_ptr(), indptr.data_ptr(), last_page_len.data_ptr(),
        append_ckv.data_ptr(), append_kpe.data_ptr(),
        batch_indices.data_ptr(), positions.data_ptr(),
        B * S, page_size,
        HEAD_DIM_CKV, HEAD_DIM_KPE,
        HEAD_DIM_CKV, HEAD_DIM_KPE)

    glm.synchronize()

    expected_ckv = append_ckv.reshape(B * S, 1, HEAD_DIM_CKV)
    expected_kpe = append_kpe.reshape(B * S, 1, HEAD_DIM_KPE)
    torch.testing.assert_close(ckv_cache.cpu(), expected_ckv.cpu(), atol=0.0, rtol=0.0)
    torch.testing.assert_close(kpe_cache.cpu(), expected_kpe.cpu(), atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mla_decode_with_append(glm, device):
    B = 1
    num_pages = 8
    num_heads = 4
    sm_scale = 1.0 / math.sqrt(HEAD_DIM_CKV + HEAD_DIM_KPE)
    rope_theta = 1000000.0

    torch.manual_seed(99)
    ckv_tokens = torch.randn(num_pages, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    kpe_tokens = torch.randn(num_pages, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)

    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, HEAD_DIM_KPE, 2, dtype=torch.float32, device=device) / HEAD_DIM_KPE))

    positions_k = torch.arange(num_pages, dtype=torch.float32, device=device)
    freqs_k = torch.outer(positions_k, inv_freq)
    emb_k = torch.cat([freqs_k, freqs_k], dim=-1)
    cos_k = emb_k.cos().to(torch.bfloat16)
    sin_k = emb_k.sin().to(torch.bfloat16)
    kpe_tokens_rope = _apply_interleaved_rope(kpe_tokens, cos_k, sin_k)

    ckv_cache = torch.zeros(num_pages, PAGE_SIZE, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    kpe_cache = torch.zeros(num_pages, PAGE_SIZE, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)

    indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    last_page_len = torch.tensor([PAGE_SIZE], dtype=torch.int32, device=device)
    batch_indices = torch.zeros(num_pages, dtype=torch.int32, device=device)
    positions = torch.arange(num_pages, dtype=torch.int32, device=device)

    glm.mla_kv_cache_append(
        ckv_cache.data_ptr(), kpe_cache.data_ptr(),
        indices.data_ptr(), indptr.data_ptr(), last_page_len.data_ptr(),
        ckv_tokens.data_ptr(), kpe_tokens_rope.data_ptr(),
        batch_indices.data_ptr(), positions.data_ptr(),
        num_pages, PAGE_SIZE,
        HEAD_DIM_CKV, HEAD_DIM_KPE,
        HEAD_DIM_CKV, HEAD_DIM_KPE)

    q_nope = torch.randn(B, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(B, num_heads, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)

    decode_pos = torch.tensor([num_pages - 1], dtype=torch.float32, device=device)
    freqs_q = torch.outer(decode_pos, inv_freq)
    emb_q = torch.cat([freqs_q, freqs_q], dim=-1)
    cos_q = emb_q.cos().to(torch.bfloat16).reshape(1, 1, HEAD_DIM_KPE)
    sin_q = emb_q.sin().to(torch.bfloat16).reshape(1, 1, HEAD_DIM_KPE)
    q_pe_rope = _apply_interleaved_rope(q_pe, cos_q, sin_q)

    indptr_h = (ctypes.c_int32 * 2)(0, num_pages)
    float_ws, int_ws, pinned_int_ws = _alloc_workspace(glm)
    plan_info = (ctypes.c_int64 * 10)()

    glm.mla_decode_plan(
        float_ws, 32 * 1024 * 1024,
        int_ws, pinned_int_ws, 8 * 1024 * 1024,
        ctypes.addressof(plan_info),
        ctypes.addressof(indptr_h),
        B, num_heads, PAGE_SIZE, False,
        head_dim_ckv=HEAD_DIM_CKV, head_dim_kpe=HEAD_DIM_KPE)

    o = torch.empty(B, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    indptr_d = torch.tensor([0, num_pages], dtype=torch.int32, device=device)

    glm.mla_decode_run(
        q_nope.data_ptr(), q_pe_rope.data_ptr(),
        ckv_cache.data_ptr(), kpe_cache.data_ptr(),
        indices.data_ptr(), indptr_d.data_ptr(), last_page_len.data_ptr(),
        o.data_ptr(),
        float_ws, int_ws, ctypes.addressof(plan_info),
        B, num_heads, PAGE_SIZE, sm_scale,
        head_dim_ckv=HEAD_DIM_CKV, head_dim_kpe=HEAD_DIM_KPE)

    glm.synchronize()

    ref_positions = torch.arange(num_pages, dtype=torch.int32, device=device)
    ref_output = mla_decode_reference(
        q_nope, q_pe_rope,
        ckv_tokens.unsqueeze(0), kpe_tokens.unsqueeze(0),
        ref_positions, sm_scale)

    torch.testing.assert_close(o.cpu(), ref_output.cpu(), atol=0.05, rtol=0.02)

    glm.free_buf(float_ws)
    glm.free_buf(int_ws)
    glm.free_pinned(pinned_int_ws)


# FlashInfer LSE is log-base-2: lse = log2(sum(exp(sm_scale * logits)))
# Reference: score.logsumexp(dim=-1) * log2(e)  converts base-e to base-2

def mla_prefill_reference_with_lse(q_nope, q_pe_rope, ckv, kpe_rope, sm_scale, causal=True):
    BS, H, D_CKV = q_nope.shape
    S = ckv.shape[1]
    D_KPE = q_pe_rope.shape[-1]

    ckv_exp = ckv.expand(H, S, D_CKV)
    kpe_exp = kpe_rope.expand(H, S, D_KPE)

    score_nope = torch.einsum('bhd,hkd->bhk', q_nope, ckv_exp)
    score_pe = torch.einsum('bhd,hkd->bhk', q_pe_rope, kpe_exp)
    score = (score_nope + score_pe) * sm_scale

    if causal:
        mask = torch.triu(torch.full((BS, S), float('-inf'), device=score.device, dtype=score.dtype), diagonal=1)
        score = score + mask.unsqueeze(1)

    score_f = score.float()
    # FlashInfer LSE is base-2: log2(sum(exp(score)))
    lse = score_f.logsumexp(dim=-1) * math.log2(math.e)

    attn = torch.nn.functional.softmax(score_f, dim=-1).to(q_nope.dtype)
    output = torch.einsum('bhk,hkd->bhd', attn, ckv_exp)
    return output, lse


def mla_decode_reference_with_lse(q_nope_absorbed, q_pe_rope, ckv, kpe, positions, sm_scale, rope_theta=1000000.0):
    B, H, D_CKV = q_nope_absorbed.shape
    S = ckv.shape[1]
    D_KPE = q_pe_rope.shape[-1]

    dim_half = D_KPE // 2
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, D_KPE, 2, dtype=torch.float32, device=ckv.device) / D_KPE))
    freqs = torch.outer(positions.float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos_emb = emb.cos().to(ckv.dtype)
    sin_emb = emb.sin().to(ckv.dtype)

    kpe_cos_half = cos_emb[:, :dim_half].unsqueeze(0)
    kpe_sin_half = sin_emb[:, :dim_half].unsqueeze(0)
    kpe_x1 = kpe[..., 0::2].float()
    kpe_x2 = kpe[..., 1::2].float()
    kpe_o1 = kpe_x1 * kpe_cos_half.float() - kpe_x2 * kpe_sin_half.float()
    kpe_o2 = kpe_x2 * kpe_cos_half.float() + kpe_x1 * kpe_sin_half.float()
    kpe_rope = torch.stack((kpe_o1, kpe_o2), dim=-1).flatten(-2).to(kpe.dtype)

    q_nope_3d = q_nope_absorbed.reshape(B * H, 1, D_CKV)
    q_pe_3d = q_pe_rope.reshape(B * H, 1, D_KPE)
    ckv_3d = ckv.reshape(B, 1, S, D_CKV).expand(B, H, S, D_CKV).reshape(B * H, S, D_CKV)
    kpe_3d = kpe_rope.reshape(B, 1, S, D_KPE).expand(B, H, S, D_KPE).reshape(B * H, S, D_KPE)

    score = torch.bmm(q_nope_3d, ckv_3d.transpose(1, 2)) + torch.bmm(q_pe_3d, kpe_3d.transpose(1, 2))
    score = score * sm_scale
    score_f = score.float()
    # FlashInfer LSE is base-2: log2(sum(exp(score)))
    lse = score_f.logsumexp(dim=-1) * math.log2(math.e)

    attn = torch.nn.functional.softmax(score_f, dim=-1).to(ckv.dtype)
    output = torch.bmm(attn, ckv_3d)
    return output.reshape(B, H, D_CKV), lse.reshape(B, H)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mla_prefill_lse(glm, device):
    B = 1
    S = 8
    num_heads = 4
    sm_scale = 1.0 / math.sqrt(HEAD_DIM_CKV + HEAD_DIM_KPE)

    torch.manual_seed(42)
    q_nope = torch.randn(B * S, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(B * S, num_heads, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
    ckv = torch.randn(S, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    kpe = torch.randn(S, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)

    cos, sin = _make_rotary_embed(glm, device, HEAD_DIM_KPE // 2, B, S)
    q_pe_4d = q_pe.reshape(B, num_heads, S, HEAD_DIM_KPE)
    q_pe_rope = torch.empty_like(q_pe_4d)
    glm.apply_rotary_pos_emb(q_pe_rope, q_pe_4d, cos, sin, HEAD_DIM_KPE, num_heads, S, B, 1, interleaved=True)
    q_pe_rope = q_pe_rope.reshape(B * S, num_heads, HEAD_DIM_KPE)

    kpe_4d = kpe.reshape(B, 1, S, HEAD_DIM_KPE)
    kpe_rope_4d = torch.empty_like(kpe_4d)
    glm.apply_rotary_pos_emb(kpe_rope_4d, kpe_4d, cos, sin, HEAD_DIM_KPE, 1, S, B, 1, interleaved=True)
    kpe_rope = kpe_rope_4d.reshape(S, HEAD_DIM_KPE)

    ckv_paged = ckv.reshape(S, PAGE_SIZE, HEAD_DIM_CKV)
    kpe_paged = kpe_rope.reshape(S, PAGE_SIZE, HEAD_DIM_KPE)

    qo_indptr_h = (ctypes.c_int32 * 2)(0, S)
    kv_indptr_h = (ctypes.c_int32 * 2)(0, S)
    kv_len_h = (ctypes.c_int32 * 1)(S)
    kv_indices = torch.arange(S, dtype=torch.int32, device=device)
    plan_info = (ctypes.c_int64 * 19)()

    float_ws, int_ws, pinned_int_ws = _alloc_workspace(glm)

    glm.mla_prefill_plan(
        float_ws, 32 * 1024 * 1024,
        int_ws, pinned_int_ws, 8 * 1024 * 1024,
        ctypes.addressof(plan_info),
        ctypes.addressof(qo_indptr_h),
        ctypes.addressof(kv_indptr_h),
        ctypes.addressof(kv_len_h),
        B, num_heads, HEAD_DIM_CKV, True)

    o = torch.empty(B * S, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    # FlashInfer prefill LSE layout: [totalTokens, numHeads]
    lse = torch.empty(B * S, num_heads, dtype=torch.float32, device=device)

    glm.mla_prefill_run(
        q_nope.data_ptr(), q_pe_rope.data_ptr(),
        ckv_paged.data_ptr(), kpe_paged.data_ptr(),
        kv_indices.data_ptr(),
        o.data_ptr(),
        float_ws, int_ws, ctypes.addressof(plan_info),
        num_heads, PAGE_SIZE, 1, sm_scale,
        num_heads * HEAD_DIM_CKV, HEAD_DIM_CKV,
        num_heads * HEAD_DIM_KPE, HEAD_DIM_KPE,
        PAGE_SIZE * HEAD_DIM_CKV, HEAD_DIM_CKV,
        PAGE_SIZE * HEAD_DIM_KPE, HEAD_DIM_KPE,
        num_heads * HEAD_DIM_CKV, HEAD_DIM_CKV,
        HEAD_DIM_CKV, HEAD_DIM_KPE,
        lse.data_ptr(), 0, 0)

    glm.synchronize()

    ref_output, ref_lse = mla_prefill_reference_with_lse(
        q_nope, q_pe_rope,
        ckv.unsqueeze(0), kpe_rope.unsqueeze(0),
        sm_scale, causal=True)

    torch.testing.assert_close(o.cpu(), ref_output.cpu(), atol=0.05, rtol=0.02)
    torch.testing.assert_close(lse.cpu(), ref_lse.cpu(), atol=0.1, rtol=0.05)

    glm.free_buf(float_ws)
    glm.free_buf(int_ws)
    glm.free_pinned(pinned_int_ws)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mla_decode_lse(glm, device):
    B = 1
    S = 8
    num_heads = 4
    sm_scale = 1.0 / math.sqrt(HEAD_DIM_CKV + HEAD_DIM_KPE)
    rope_theta = 1000000.0

    torch.manual_seed(42)
    q_nope = torch.randn(B, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(B, num_heads, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
    ckv = torch.randn(S, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    kpe = torch.randn(S, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)

    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, HEAD_DIM_KPE, 2, dtype=torch.float32, device=device) / HEAD_DIM_KPE))

    decode_pos = torch.tensor([S - 1], dtype=torch.float32, device=device)
    freqs_q = torch.outer(decode_pos, inv_freq)
    emb_q = torch.cat([freqs_q, freqs_q], dim=-1)
    cos_q = emb_q.cos().to(torch.bfloat16).reshape(1, 1, HEAD_DIM_KPE)
    sin_q = emb_q.sin().to(torch.bfloat16).reshape(1, 1, HEAD_DIM_KPE)
    q_pe_rope = _apply_interleaved_rope(q_pe, cos_q, sin_q)

    positions_k = torch.arange(S, dtype=torch.float32, device=device)
    freqs_k = torch.outer(positions_k, inv_freq)
    emb_k = torch.cat([freqs_k, freqs_k], dim=-1)
    cos_k = emb_k.cos().to(torch.bfloat16)
    sin_k = emb_k.sin().to(torch.bfloat16)
    kpe_rope = _apply_interleaved_rope(kpe, cos_k, sin_k)

    ckv_paged = ckv.reshape(S, PAGE_SIZE, HEAD_DIM_CKV).contiguous()
    kpe_paged = kpe_rope.reshape(S, PAGE_SIZE, HEAD_DIM_KPE).contiguous()

    num_pages = S
    indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    indptr_h = (ctypes.c_int32 * 2)(0, num_pages)
    last_page_len_d = torch.tensor([PAGE_SIZE], dtype=torch.int32, device=device)

    float_ws, int_ws, pinned_int_ws = _alloc_workspace(glm)
    plan_info = (ctypes.c_int64 * 10)()

    glm.mla_decode_plan(
        float_ws, 32 * 1024 * 1024,
        int_ws, pinned_int_ws, 8 * 1024 * 1024,
        ctypes.addressof(plan_info),
        ctypes.addressof(indptr_h),
        B, num_heads, PAGE_SIZE, False,
        head_dim_ckv=HEAD_DIM_CKV, head_dim_kpe=HEAD_DIM_KPE)

    o = torch.empty(B, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    # FlashInfer decode LSE layout: [batchSize, numHeads]
    lse = torch.empty(B, num_heads, dtype=torch.float32, device=device)
    indptr_d = torch.tensor([0, num_pages], dtype=torch.int32, device=device)

    glm.mla_decode_run(
        q_nope.data_ptr(), q_pe_rope.data_ptr(),
        ckv_paged.data_ptr(), kpe_paged.data_ptr(),
        indices.data_ptr(), indptr_d.data_ptr(), last_page_len_d.data_ptr(),
        o.data_ptr(),
        float_ws, int_ws, ctypes.addressof(plan_info),
        B, num_heads, PAGE_SIZE, sm_scale,
        head_dim_ckv=HEAD_DIM_CKV, head_dim_kpe=HEAD_DIM_KPE,
        lse=lse.data_ptr())

    glm.synchronize()

    positions = torch.arange(S, dtype=torch.int32, device=device)
    ref_output, ref_lse = mla_decode_reference_with_lse(q_nope, q_pe_rope, ckv.unsqueeze(0), kpe.unsqueeze(0), positions, sm_scale)

    torch.testing.assert_close(o.cpu(), ref_output.cpu(), atol=0.05, rtol=0.02)
    torch.testing.assert_close(lse.cpu(), ref_lse.cpu(), atol=0.1, rtol=0.05)

    glm.free_buf(float_ws)
    glm.free_buf(int_ws)
    glm.free_pinned(pinned_int_ws)


def mla_prefill_reference_custom_mask(q_nope, q_pe_rope, ckv, kpe_rope, sm_scale, custom_mask, prefix_len=0):
    """Reference MLA prefill with custom mask support.

    Args:
        custom_mask: boolean array of shape [qo_len, kv_len] or [qo_len, suffix_len].
                     If prefix_len > 0, custom_mask is [qo_len, suffix_len] and
                     prefix positions are always attended (CausalCustom semantics).
        prefix_len: number of prefix KV positions (always attended).
    """
    BS, H, D_CKV = q_nope.shape
    S = ckv.shape[1]
    D_KPE = q_pe_rope.shape[-1]

    ckv_exp = ckv.expand(H, S, D_CKV)
    kpe_exp = kpe_rope.expand(H, S, D_KPE)

    score_nope = torch.einsum('bhd,hkd->bhk', q_nope, ckv_exp)
    score_pe = torch.einsum('bhd,hkd->bhk', q_pe_rope, kpe_exp)
    score = (score_nope + score_pe) * sm_scale

    if prefix_len > 0:
        suffix_len = S - prefix_len
        mask = torch.full((BS, S), float('-inf'), device=score.device, dtype=score.dtype)
        for q in range(BS):
            for k in range(prefix_len):
                mask[q, k] = 0.0
            for k in range(suffix_len):
                if custom_mask[q * suffix_len + k]:
                    mask[q, prefix_len + k] = 0.0
        score = score + mask.unsqueeze(1)
    else:
        mask = torch.full((BS, S), float('-inf'), device=score.device, dtype=score.dtype)
        for q in range(BS):
            for k in range(S):
                if custom_mask[q * S + k]:
                    mask[q, k] = 0.0
        score = score + mask.unsqueeze(1)

    attn = torch.nn.functional.softmax(score.float(), dim=-1).to(q_nope.dtype)
    output = torch.einsum('bhk,hkd->bhd', attn, ckv_exp)
    return output


def _pack_mask(mask_bool, qo_len, kv_len):
    total_bits = qo_len * kv_len
    byte_len = (total_bits + 7) // 8
    data = bytearray(byte_len)
    for q in range(qo_len):
        for k in range(kv_len):
            if mask_bool[q * kv_len + k]:
                offset = q * kv_len + k
                data[offset >> 3] |= 1 << (offset & 7)
    indptr = (ctypes.c_int32 * 2)(0, byte_len)
    return bytes(data), indptr


def _run_mla_prefill_with_mask(glm, device, q_nope, q_pe_rope, ckv_paged, kpe_paged,
                                kv_indices, qo_len, kv_len, num_heads,
                                causal, custom_mask_data=None, mask_indptr=None,
                                mask_mode=0, page_size=1):
    sm_scale = 1.0 / math.sqrt(HEAD_DIM_CKV + HEAD_DIM_KPE)
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

    o = torch.empty(B * qo_len, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)

    custom_mask_ptr = None
    mask_indptr_ptr = None
    if custom_mask_data is not None:
        custom_mask_ptr = (ctypes.c_uint8 * len(custom_mask_data))(*custom_mask_data)
        mask_indptr_ptr = mask_indptr

    mask_mode_int = mask_mode

    glm.mla_prefill_run(
        q_nope.data_ptr(), q_pe_rope.data_ptr(),
        ckv_paged.data_ptr(), kpe_paged.data_ptr(),
        kv_indices.data_ptr(),
        o.data_ptr(),
        float_ws, int_ws, ctypes.addressof(plan_info),
        num_heads, page_size, mask_mode_int, sm_scale,
        num_heads * HEAD_DIM_CKV, HEAD_DIM_CKV,
        num_heads * HEAD_DIM_KPE, HEAD_DIM_KPE,
        page_size * HEAD_DIM_CKV, HEAD_DIM_CKV,
        page_size * HEAD_DIM_KPE, HEAD_DIM_KPE,
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
def test_mla_prefill_causal_custom_tree_mask(glm, device):
    """Test CausalCustom mask with tree-structured attention mask and prefix.

    This directly tests the mask pattern used by MTP tree verification:
    - A prefix of existing KV cache entries (always attended)
    - A suffix of tree tokens with a tree-shaped custom mask
    - Each tree node attends to itself and its ancestors
    """
    B = 1
    prefix_len = 16
    num_prefill = 7
    qo_len = num_prefill
    kv_len = prefix_len + num_prefill
    num_heads = 4
    page_size = 1
    num_pages = kv_len

    torch.manual_seed(123)

    q_nope = torch.randn(B * qo_len, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(B * qo_len, num_heads, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
    ckv = torch.randn(kv_len, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    kpe = torch.randn(kv_len, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)

    cos, sin = _make_rotary_embed(glm, device, HEAD_DIM_KPE // 2, B, qo_len)
    q_pe_4d = q_pe.reshape(B, num_heads, qo_len, HEAD_DIM_KPE)
    q_pe_rope = torch.empty_like(q_pe_4d)
    glm.apply_rotary_pos_emb(q_pe_rope, q_pe_4d, cos, sin, HEAD_DIM_KPE, num_heads, qo_len, B, 1, interleaved=True)
    q_pe_rope = q_pe_rope.reshape(B * qo_len, num_heads, HEAD_DIM_KPE)

    kpe_4d = kpe.reshape(kv_len, 1, HEAD_DIM_KPE)
    kpe_rope_4d = torch.empty_like(kpe_4d)
    kpe_cos = cos[:, :kv_len, :]
    kpe_sin = sin[:, :kv_len, :]
    glm.apply_rotary_pos_emb(kpe_rope_4d, kpe_4d.expand(B, kv_len, HEAD_DIM_KPE),
                              kpe_cos, kpe_sin, HEAD_DIM_KPE, 1, kv_len, B, 1, interleaved=True)
    kpe_rope = kpe_rope_4d.reshape(kv_len, HEAD_DIM_KPE)

    ckv_paged = ckv.reshape(num_pages, page_size, HEAD_DIM_CKV)
    kpe_paged = kpe_rope.reshape(num_pages, page_size, HEAD_DIM_KPE)
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)

    # Build tree mask: binary tree with depth 3 (7 nodes)
    parents = [[], [0], [0], [1], [1], [2], [2]]
    suffix_mask = [False] * (qo_len * qo_len)
    for q in range(qo_len):
        suffix_mask[q * qo_len + q] = True
        cur = q
        while len(parents[cur]) > 0:
            cur = parents[cur][0]
            suffix_mask[q * qo_len + cur] = True

    packed_mask, mask_indptr = _pack_mask(suffix_mask, qo_len, qo_len)

    # Run with CausalCustom (mask_mode=4)
    o_cc = _run_mla_prefill_with_mask(
        glm, device, q_nope, q_pe_rope, ckv_paged, kpe_paged,
        kv_indices, qo_len, kv_len, num_heads,
        causal=True,
        custom_mask_data=packed_mask, mask_indptr=mask_indptr,
        mask_mode=4,
        page_size=page_size)

    # Build full custom mask for reference
    full_mask = [False] * (qo_len * kv_len)
    for q in range(qo_len):
        for k in range(prefix_len):
            full_mask[q * kv_len + k] = True
        for k in range(qo_len):
            full_mask[q * kv_len + prefix_len + k] = suffix_mask[q * qo_len + k]

    ref_output = mla_prefill_reference_custom_mask(
        q_nope, q_pe_rope,
        ckv.unsqueeze(0), kpe_rope.unsqueeze(0),
        1.0 / math.sqrt(HEAD_DIM_CKV + HEAD_DIM_KPE),
        full_mask, prefix_len=prefix_len)

    torch.testing.assert_close(o_cc.cpu(), ref_output.cpu(), atol=0.05, rtol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mla_prefill_causal_custom_matches_full_custom(glm, device):
    """Test that CausalCustom produces the same output as full Custom mask (15-node tree)."""
    B = 1
    prefix_len = 32
    num_prefill = 15
    qo_len = num_prefill
    kv_len = prefix_len + num_prefill
    num_heads = 4
    page_size = 1
    num_pages = kv_len

    torch.manual_seed(456)

    q_nope = torch.randn(B * qo_len, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(B * qo_len, num_heads, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
    ckv = torch.randn(kv_len, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    kpe = torch.randn(kv_len, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)

    cos, sin = _make_rotary_embed(glm, device, HEAD_DIM_KPE // 2, B, qo_len)
    q_pe_4d = q_pe.reshape(B, num_heads, qo_len, HEAD_DIM_KPE)
    q_pe_rope = torch.empty_like(q_pe_4d)
    glm.apply_rotary_pos_emb(q_pe_rope, q_pe_4d, cos, sin, HEAD_DIM_KPE, num_heads, qo_len, B, 1, interleaved=True)
    q_pe_rope = q_pe_rope.reshape(B * qo_len, num_heads, HEAD_DIM_KPE)

    kpe_4d = kpe.reshape(kv_len, 1, HEAD_DIM_KPE)
    kpe_rope_4d = torch.empty_like(kpe_4d)
    kpe_cos = cos[:, :kv_len, :]
    kpe_sin = sin[:, :kv_len, :]
    glm.apply_rotary_pos_emb(kpe_rope_4d, kpe_4d.expand(B, kv_len, HEAD_DIM_KPE),
                              kpe_cos, kpe_sin, HEAD_DIM_KPE, 1, kv_len, B, 1, interleaved=True)
    kpe_rope = kpe_rope_4d.reshape(kv_len, HEAD_DIM_KPE)

    ckv_paged = ckv.reshape(num_pages, page_size, HEAD_DIM_CKV)
    kpe_paged = kpe_rope.reshape(num_pages, page_size, HEAD_DIM_KPE)
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)

    # Build tree mask (binary tree depth 4, 15 nodes)
    parents = [[], [0], [0], [1], [1], [2], [2], [3], [3], [4], [4], [5], [5], [6], [6]]
    suffix_mask = [False] * (qo_len * qo_len)
    for q in range(qo_len):
        suffix_mask[q * qo_len + q] = True
        cur = q
        while len(parents[cur]) > 0:
            cur = parents[cur][0]
            suffix_mask[q * qo_len + cur] = True

    # CausalCustom mask
    packed_suffix, suffix_indptr = _pack_mask(suffix_mask, qo_len, qo_len)
    o_cc = _run_mla_prefill_with_mask(
        glm, device, q_nope, q_pe_rope, ckv_paged, kpe_paged,
        kv_indices, qo_len, kv_len, num_heads,
        causal=True,
        custom_mask_data=packed_suffix, mask_indptr=suffix_indptr,
        mask_mode=4,
        page_size=page_size)

    # Full Custom mask (prefix always attended + tree mask for suffix)
    full_mask = [False] * (qo_len * kv_len)
    for q in range(qo_len):
        for k in range(prefix_len):
            full_mask[q * kv_len + k] = True
        for k in range(qo_len):
            full_mask[q * kv_len + prefix_len + k] = suffix_mask[q * qo_len + k]

    packed_full, full_indptr = _pack_mask(full_mask, qo_len, kv_len)
    o_full = _run_mla_prefill_with_mask(
        glm, device, q_nope, q_pe_rope, ckv_paged, kpe_paged,
        kv_indices, qo_len, kv_len, num_heads,
        causal=False,
        custom_mask_data=packed_full, mask_indptr=full_indptr,
        mask_mode=2,
        page_size=page_size)

    torch.testing.assert_close(o_cc.cpu(), o_full.cpu(), atol=0.01, rtol=0.005)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mla_prefill_causal_custom_causal_suffix_matches_causal(glm, device):
    """Test that CausalCustom with a causal suffix mask matches built-in causal attention."""
    B = 1
    prefix_len = 32
    num_prefill = 7
    qo_len = num_prefill
    kv_len = prefix_len + num_prefill
    num_heads = 4
    page_size = 1
    num_pages = kv_len

    torch.manual_seed(789)

    q_nope = torch.randn(B * qo_len, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    q_pe = torch.randn(B * qo_len, num_heads, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)
    ckv = torch.randn(kv_len, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device)
    kpe = torch.randn(kv_len, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device)

    cos, sin = _make_rotary_embed(glm, device, HEAD_DIM_KPE // 2, B, qo_len)
    q_pe_4d = q_pe.reshape(B, num_heads, qo_len, HEAD_DIM_KPE)
    q_pe_rope = torch.empty_like(q_pe_4d)
    glm.apply_rotary_pos_emb(q_pe_rope, q_pe_4d, cos, sin, HEAD_DIM_KPE, num_heads, qo_len, B, 1, interleaved=True)
    q_pe_rope = q_pe_rope.reshape(B * qo_len, num_heads, HEAD_DIM_KPE)

    kpe_4d = kpe.reshape(kv_len, 1, HEAD_DIM_KPE)
    kpe_rope_4d = torch.empty_like(kpe_4d)
    kpe_cos = cos[:, :kv_len, :]
    kpe_sin = sin[:, :kv_len, :]
    glm.apply_rotary_pos_emb(kpe_rope_4d, kpe_4d.expand(B, kv_len, HEAD_DIM_KPE),
                              kpe_cos, kpe_sin, HEAD_DIM_KPE, 1, kv_len, B, 1, interleaved=True)
    kpe_rope = kpe_rope_4d.reshape(kv_len, HEAD_DIM_KPE)

    ckv_paged = ckv.reshape(num_pages, page_size, HEAD_DIM_CKV)
    kpe_paged = kpe_rope.reshape(num_pages, page_size, HEAD_DIM_KPE)
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)

    # Causal suffix mask: q[i] attends to suffix[0..i]
    suffix_mask = [False] * (qo_len * qo_len)
    for q in range(qo_len):
        for k in range(q + 1):
            suffix_mask[q * qo_len + k] = True

    packed_suffix, suffix_indptr = _pack_mask(suffix_mask, qo_len, qo_len)
    o_cc = _run_mla_prefill_with_mask(
        glm, device, q_nope, q_pe_rope, ckv_paged, kpe_paged,
        kv_indices, qo_len, kv_len, num_heads,
        causal=True,
        custom_mask_data=packed_suffix, mask_indptr=suffix_indptr,
        mask_mode=4,
        page_size=page_size)

    # Built-in causal
    o_causal = _run_mla_prefill_with_mask(
        glm, device, q_nope, q_pe_rope, ckv_paged, kpe_paged,
        kv_indices, qo_len, kv_len, num_heads,
        causal=True,
        page_size=page_size)

    torch.testing.assert_close(o_cc.cpu(), o_causal.cpu(), atol=0.05, rtol=0.02)
