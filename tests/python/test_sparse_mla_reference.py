"""Sparse MLA versus independent FP32 attention on unquantized KV.

When kvLen < topk, sparse attention selects every KV position and must agree
with full attention within FP8 quantization tolerance. Both prefill and decode
are checked against PyTorch, with kpe treated as already position-encoded.
"""
import math
import pytest
import torch

from helpers import GlmOps  # noqa: F401 (used by conftest `glm`/`device` fixtures)

KV_LORA_RANK = 512
PE_DIM = 64
D_QK = KV_LORA_RANK + PE_DIM      # 576
D_V = KV_LORA_RANK                # 512
PAGE_SIZE = 64                    # SM120 sparse kernel requires 64
TOPK = 2048
BPT = KV_LORA_RANK + (KV_LORA_RANK // 128) * 4 + PE_DIM * 2  # 656 packed bytes/token

# Production GLM-5.2 scale: qk_head_dim = qk_nope(192)+qk_rope(64) = 256.
SM_SCALE = 256 ** -0.5            # 0.0625


def _reference_mla(q, ckv, kpe, sm_scale):
    """fp32 ground-truth absorbed-MLA decode: 1 query attends all S KV (no mask).

    q:   [H, D_QK]  ([nope(512) | rope(64)])
    ckv: [S, 512]   (also serves as V — absorbed MLA)
    kpe: [S, 64]
    returns o: [H, 512] fp32
    """
    q_nope = q[:, :KV_LORA_RANK].float()   # [H,512]
    q_pe = q[:, KV_LORA_RANK:].float()     # [H,64]
    ckv_f = ckv.float()
    kpe_f = kpe.float()
    score = (q_nope @ ckv_f.T + q_pe @ kpe_f.T) * sm_scale  # [H,S]
    attn = torch.softmax(score, dim=-1)                     # [H,S]
    return attn @ ckv_f                                     # [H,512]


def _cos(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


def _run_sparse(glm, device, q, ckv, kpe, num_heads, seq_len):
    """Sparse MLA decode over the FP8-packed cache. Returns [H,512] bf16.

    Packs the same ckv/kpe via glm.concat_and_cache_ds_mla, builds slots for a
    single decode query attending all `seq_len` positions (rest -1 padded to
    TOPK), and runs glm.sparse_mla_decode.
    """
    num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    max_pages = num_pages + 2
    kv_cache = torch.zeros(max_pages, PAGE_SIZE, BPT, dtype=torch.uint8, device=device)

    # Page table for a single sequence of length seq_len.
    page_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    page_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    batch_indices = torch.zeros(seq_len, dtype=torch.int32, device=device)
    positions = torch.arange(seq_len, dtype=torch.int32, device=device)

    glm.concat_and_cache_ds_mla(
        kv_cache.data_ptr(), ckv.contiguous().data_ptr(), kpe.contiguous().data_ptr(),
        page_indices.data_ptr(), page_indptr.data_ptr(),
        batch_indices.data_ptr(), positions.data_ptr(),
        seq_len, PAGE_SIZE, KV_LORA_RANK, PE_DIM,
        KV_LORA_RANK, PE_DIM,
    )

    # Slots: single decode query attends every stored position; pad to TOPK.
    slots = torch.full((1, TOPK), -1, dtype=torch.int32, device=device)
    for pos in range(seq_len):
        page = pos // PAGE_SIZE
        off = pos % PAGE_SIZE
        slots[0, pos] = int(page_indices[page].item()) * PAGE_SIZE + off

    q_in = q.reshape(1, num_heads, D_QK).contiguous()
    num_splits = (TOPK + 63) // 64  # 32, matches the model
    mid_out = torch.zeros(1, num_heads, num_splits, KV_LORA_RANK, dtype=torch.bfloat16, device=device)
    mid_lse = torch.zeros(1, num_heads, num_splits, dtype=torch.float32, device=device)
    output = torch.zeros(1, num_heads, KV_LORA_RANK, dtype=torch.bfloat16, device=device)
    out_lse = torch.zeros(1, num_heads, dtype=torch.float32, device=device)
    stride_kv_block = PAGE_SIZE * BPT

    glm.sparse_mla_decode(
        q_in.data_ptr(), kv_cache.data_ptr(), slots.data_ptr(),
        mid_out.data_ptr(), mid_lse.data_ptr(),
        output.data_ptr(), out_lse.data_ptr(),
        1, num_heads, TOPK, num_splits,
        SM_SCALE, stride_kv_block,
    )
    glm.synchronize()
    return output.reshape(num_heads, KV_LORA_RANK)


def _reference_mla_prefill(q, ckv, kpe, sm_scale):
    """fp32 causal ground truth: query t attends KV positions 0..t.

    q:   [S, H, D_QK]   ckv/kpe: [S, *]   returns o: [S, H, 512] fp32
    """
    S, H, _ = q.shape
    q_nope = q[:, :, :KV_LORA_RANK].float()   # [S,H,512]
    q_pe = q[:, :, KV_LORA_RANK:].float()     # [S,H,64]
    ckv_f = ckv.float()
    kpe_f = kpe.float()
    # score[t,h,j] = (q_nope[t,h]·ckv[j] + q_pe[t,h]·kpe[j]) * scale
    score = (torch.einsum('thd,jd->thj', q_nope, ckv_f)
             + torch.einsum('thd,jd->thj', q_pe, kpe_f)) * sm_scale  # [S,H,S]
    causal = torch.triu(torch.full((S, S), float('-inf'), device=score.device), diagonal=1)
    score = score + causal.unsqueeze(1)
    attn = torch.softmax(score, dim=-1)
    return torch.einsum('thj,jd->thd', attn, ckv_f)  # [S,H,512]


def _run_sparse_prefill(glm, device, q, ckv, kpe, num_heads, seq_len):
    """Sparse MLA prefill: S causal queries over the FP8-packed cache. [S,H,512]."""
    num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    max_pages = num_pages + 2
    kv_cache = torch.zeros(max_pages, PAGE_SIZE, BPT, dtype=torch.uint8, device=device)

    page_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    page_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    batch_indices = torch.zeros(seq_len, dtype=torch.int32, device=device)
    positions = torch.arange(seq_len, dtype=torch.int32, device=device)

    glm.concat_and_cache_ds_mla(
        kv_cache.data_ptr(), ckv.contiguous().data_ptr(), kpe.contiguous().data_ptr(),
        page_indices.data_ptr(), page_indptr.data_ptr(),
        batch_indices.data_ptr(), positions.data_ptr(),
        seq_len, PAGE_SIZE, KV_LORA_RANK, PE_DIM,
        KV_LORA_RANK, PE_DIM,
    )

    def slot(pos):
        return int(page_indices[pos // PAGE_SIZE].item()) * PAGE_SIZE + (pos % PAGE_SIZE)

    # Causal slots: query t attends positions 0..t, padded to TOPK with -1.
    slots = torch.full((seq_len, TOPK), -1, dtype=torch.int32, device=device)
    for t in range(seq_len):
        for pos in range(t + 1):
            slots[t, pos] = slot(pos)

    q_in = q.reshape(seq_len, num_heads, D_QK).contiguous()
    output = torch.zeros(seq_len, num_heads, KV_LORA_RANK, dtype=torch.bfloat16, device=device)
    out_lse = torch.zeros(seq_len, num_heads, dtype=torch.float32, device=device)
    stride_kv_block = PAGE_SIZE * BPT

    glm.sparse_mla_prefill(
        q_in.data_ptr(), kv_cache.data_ptr(), slots.data_ptr(),
        output.data_ptr(), out_lse.data_ptr(),
        seq_len, num_heads, TOPK, PAGE_SIZE,
        SM_SCALE, stride_kv_block,
    )
    glm.synchronize()
    return output.reshape(seq_len, num_heads, KV_LORA_RANK)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("num_heads", [8, 64])
@pytest.mark.parametrize("seq_len", [13, 37, 200])
def test_sparse_prefill_matches_reference(glm, device, num_heads, seq_len):
    """Sparse MLA PREFILL (causal, S queries) vs fp32 causal reference.

    This is the path the 13-token prompt hits first; a bug here corrupts the
    very first generated token.
    """
    torch.manual_seed(7000 + seq_len + num_heads)

    ckv = torch.randn(seq_len, KV_LORA_RANK, dtype=torch.bfloat16, device=device)
    kpe = torch.randn(seq_len, PE_DIM, dtype=torch.bfloat16, device=device)
    q = torch.randn(seq_len, num_heads, D_QK, dtype=torch.bfloat16, device=device)

    ref = _reference_mla_prefill(q, ckv, kpe, SM_SCALE)                    # [S,H,512]
    sparse = _run_sparse_prefill(glm, device, q, ckv, kpe, num_heads, seq_len).float()

    ref_scale = ref.abs().max().item() + 1e-6
    rel = (sparse - ref).abs().max().item() / ref_scale
    cos = _cos(sparse, ref)
    # Per-query-token worst case (the last token feeds the next-token logits).
    last_cos = _cos(sparse[-1], ref[-1])

    print(f"\n[PREFILL H={num_heads} S={seq_len}] rel={rel:.4f} cos={cos:.5f} "
          f"last_tok_cos={last_cos:.5f}")
    print(f"  ref[-1,0,:6]:    {ref[-1,0,:6].tolist()}")
    print(f"  sparse[-1,0,:6]: {sparse[-1,0,:6].tolist()}")

    assert cos > 0.99, (
        f"SPARSE PREFILL disagrees with reference (cos={cos}, rel={rel}) "
        f"— reproduces the model garbage-output bug")
    assert last_cos > 0.99, f"last-token prefill mismatch (cos={last_cos})"
    assert rel < 0.12, f"sparse prefill rel error too high: {rel}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("num_heads", [8, 64])
@pytest.mark.parametrize("seq_len", [13, 37, 200])
def test_sparse_decode_matches_reference(glm, device, num_heads, seq_len):
    """Sparse MLA over all KV positions agrees with the FP32 reference."""
    torch.manual_seed(1234 + seq_len + num_heads)

    ckv = torch.randn(seq_len, KV_LORA_RANK, dtype=torch.bfloat16, device=device)
    kpe = torch.randn(seq_len, PE_DIM, dtype=torch.bfloat16, device=device)
    q = torch.randn(num_heads, D_QK, dtype=torch.bfloat16, device=device)

    ref = _reference_mla(q, ckv, kpe, SM_SCALE)                       # [H,512] fp32
    sparse = _run_sparse(glm, device, q, ckv, kpe, num_heads, seq_len).float()

    ref_scale = ref.abs().max().item() + 1e-6
    sparse_rel = (sparse - ref).abs().max().item() / ref_scale
    sparse_cos = _cos(sparse, ref)

    print(f"\n[H={num_heads} S={seq_len}] "
          f"sparse: rel={sparse_rel:.4f} cos={sparse_cos:.5f}")
    print(f"  ref[0,:6]:    {ref[0,:6].tolist()}")
    print(f"  sparse[0,:6]: {sparse[0,:6].tolist()}")

    # The actual bug check: sparse must match within FP8 tolerance.
    assert sparse_cos > 0.99, (
        f"SPARSE disagrees with reference (cos={sparse_cos}, rel={sparse_rel}) "
        f"— reproduces the model garbage-output bug")
    assert sparse_rel < 0.10, f"sparse rel error too high: {sparse_rel}"


def _run_sparse_prefill_and_decode(glm, device, q, ckv, kpe, num_heads, seq_len):
    """Pack once, then run BOTH the prefill kernel and the split-K decode kernel
    on identical q/cache/causal-slots. Returns (prefill[S,H,512], decode[S,H,512])."""
    num_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    max_pages = num_pages + 2
    kv_cache = torch.zeros(max_pages, PAGE_SIZE, BPT, dtype=torch.uint8, device=device)
    page_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    page_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    batch_indices = torch.zeros(seq_len, dtype=torch.int32, device=device)
    positions = torch.arange(seq_len, dtype=torch.int32, device=device)
    glm.concat_and_cache_ds_mla(
        kv_cache.data_ptr(), ckv.contiguous().data_ptr(), kpe.contiguous().data_ptr(),
        page_indices.data_ptr(), page_indptr.data_ptr(),
        batch_indices.data_ptr(), positions.data_ptr(),
        seq_len, PAGE_SIZE, KV_LORA_RANK, PE_DIM, KV_LORA_RANK, PE_DIM,
    )
    slots = torch.full((seq_len, TOPK), -1, dtype=torch.int32, device=device)
    for t in range(seq_len):
        for pos in range(t + 1):
            slots[t, pos] = int(page_indices[pos // PAGE_SIZE].item()) * PAGE_SIZE + (pos % PAGE_SIZE)
    q_in = q.reshape(seq_len, num_heads, D_QK).contiguous()
    stride_kv_block = PAGE_SIZE * BPT

    pre = torch.zeros(seq_len, num_heads, KV_LORA_RANK, dtype=torch.bfloat16, device=device)
    pre_lse = torch.zeros(seq_len, num_heads, dtype=torch.float32, device=device)
    glm.sparse_mla_prefill(
        q_in.data_ptr(), kv_cache.data_ptr(), slots.data_ptr(),
        pre.data_ptr(), pre_lse.data_ptr(),
        seq_len, num_heads, TOPK, PAGE_SIZE, SM_SCALE, stride_kv_block,
    )

    num_splits = (TOPK + 63) // 64
    mid_out = torch.zeros(seq_len, num_heads, num_splits, KV_LORA_RANK, dtype=torch.bfloat16, device=device)
    mid_lse = torch.zeros(seq_len, num_heads, num_splits, dtype=torch.float32, device=device)
    dec = torch.zeros(seq_len, num_heads, KV_LORA_RANK, dtype=torch.bfloat16, device=device)
    dec_lse = torch.zeros(seq_len, num_heads, dtype=torch.float32, device=device)
    glm.sparse_mla_decode(
        q_in.data_ptr(), kv_cache.data_ptr(), slots.data_ptr(),
        mid_out.data_ptr(), mid_lse.data_ptr(), dec.data_ptr(), dec_lse.data_ptr(),
        seq_len, num_heads, TOPK, num_splits, SM_SCALE, stride_kv_block,
    )
    glm.synchronize()
    return pre.reshape(seq_len, num_heads, KV_LORA_RANK), dec.reshape(seq_len, num_heads, KV_LORA_RANK)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("num_heads", [8, 64])
@pytest.mark.parametrize("seq_len", [13, 37, 200])
def test_prefill_decode_equivalence(glm, device, num_heads, seq_len):
    """The split-K decode kernel must produce the same result as the prefill
    kernel on identical multi-token causal inputs (this is what the GlmOps
    sparseMlaPrefill->decode carveout relies on)."""
    torch.manual_seed(9000 + seq_len + num_heads)
    ckv = torch.randn(seq_len, KV_LORA_RANK, dtype=torch.bfloat16, device=device)
    kpe = torch.randn(seq_len, PE_DIM, dtype=torch.bfloat16, device=device)
    q = torch.randn(seq_len, num_heads, D_QK, dtype=torch.bfloat16, device=device)

    pre, dec = _run_sparse_prefill_and_decode(glm, device, q, ckv, kpe, num_heads, seq_len)
    pre = pre.float(); dec = dec.float()
    rel = (pre - dec).abs().max().item() / (pre.abs().max().item() + 1e-6)
    cos = _cos(pre, dec)
    last_cos = _cos(pre[-1], dec[-1])
    print(f"\n[prefill vs decode H={num_heads} S={seq_len}] cos={cos:.5f} last={last_cos:.5f} rel={rel:.4f}")
    assert cos > 0.999, f"prefill vs decode mismatch: cos={cos}, rel={rel}"
    assert rel < 0.02, f"prefill vs decode rel too high: {rel}"
