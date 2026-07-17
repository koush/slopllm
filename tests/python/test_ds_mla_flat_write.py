"""Tests for concat_and_cache_ds_mla flat-write mode (indices=None).

When indices is None, the kernel skips page indirection and writes sequentially
at slot = indptr[batch] * page_size + pos. This is the flat gathered-buffer path
used by the cross-layer CKV gather pipeline.

These tests verify:
  1. Flat write produces the same FP8 quantization as the paged path.
  2. Flat write places tokens at sequential positions (no page scattering).
  3. Multi-sequence flat write respects per-batch page-aligned offsets.
  4. Flat write matches paged write with identity indices (equivalence).
  5. Flat write with scattered paged indices diverges correctly.
"""
import torch
import pytest
from helpers import GlmOps
from test_ds_mla_quant import ref_quantize_ds_mla, ref_dequantize_ds_mla

PAGE_SIZE = 64


@pytest.mark.parametrize("kv_lora_rank,pe_dim", [
    (512, 64),
    (128, 64),
])
def test_flat_write_single_seq(glm, device, kv_lora_rank, pe_dim):
    """Flat write quantizes correctly and writes at sequential slots."""
    nnz = 128
    num_pages = (nnz + PAGE_SIZE - 1) // PAGE_SIZE
    num_tiles = kv_lora_rank // 128
    bpt = kv_lora_rank + num_tiles * 4 + pe_dim * 2

    ckv = torch.randn(nnz, kv_lora_rank, dtype=torch.bfloat16, device=device)
    kpe = torch.randn(nnz, pe_dim, dtype=torch.bfloat16, device=device)

    flat_cache = torch.zeros(num_pages * PAGE_SIZE, bpt, dtype=torch.uint8, device=device)
    indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    batch_indices = torch.zeros(nnz, dtype=torch.int32, device=device)
    positions = torch.arange(nnz, dtype=torch.int32, device=device)

    glm.concat_and_cache_ds_mla(
        flat_cache.data_ptr(), ckv.data_ptr(), kpe.data_ptr(),
        None, indptr.data_ptr(),
        batch_indices.data_ptr(), positions.data_ptr(),
        nnz, PAGE_SIZE, kv_lora_rank, pe_dim,
        kv_lora_rank, pe_dim,
    )
    torch.cuda.synchronize(device)

    ref = ref_quantize_ds_mla(ckv, kpe, kv_lora_rank, pe_dim)

    actual = flat_cache[:nnz]
    ckv_deq_actual, kpe_actual = ref_dequantize_ds_mla(actual, kv_lora_rank, pe_dim)
    ckv_deq_ref, kpe_ref = ref_dequantize_ds_mla(ref, kv_lora_rank, pe_dim)

    assert torch.allclose(ckv_deq_actual, ckv_deq_ref, atol=1e-1, rtol=1e-1), \
        f"ckv dequant mismatch: max diff {(ckv_deq_actual.float() - ckv_deq_ref.float()).abs().max().item()}"
    assert torch.equal(kpe_actual, kpe), "kpe should be exact BF16 copy"

    beyond = flat_cache[nnz:]
    assert (beyond == 0).all(), "tokens beyond nnz should be zero"


@pytest.mark.parametrize("kv_lora_rank,pe_dim", [
    (512, 64),
])
def test_flat_write_multi_seq(glm, device, kv_lora_rank, pe_dim):
    """Multi-sequence flat write respects per-batch page-aligned offsets."""
    num_tiles = kv_lora_rank // 128
    bpt = kv_lora_rank + num_tiles * 4 + pe_dim * 2

    seq_lens = [64, 30, 100]
    nnz = sum(seq_lens)
    num_pages = sum((sl + PAGE_SIZE - 1) // PAGE_SIZE for sl in seq_lens)

    ckv = torch.randn(nnz, kv_lora_rank, dtype=torch.bfloat16, device=device)
    kpe = torch.randn(nnz, pe_dim, dtype=torch.bfloat16, device=device)
    flat_cache = torch.zeros(num_pages * PAGE_SIZE, bpt, dtype=torch.uint8, device=device)

    indptr = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=device)
    for i, sl in enumerate(seq_lens):
        indptr[i + 1] = indptr[i] + (sl + PAGE_SIZE - 1) // PAGE_SIZE

    batch_indices = torch.zeros(nnz, dtype=torch.int32, device=device)
    positions = torch.zeros(nnz, dtype=torch.int32, device=device)
    idx = 0
    for b, sl in enumerate(seq_lens):
        batch_indices[idx:idx + sl] = b
        positions[idx:idx + sl] = torch.arange(sl, dtype=torch.int32, device=device)
        idx += sl

    glm.concat_and_cache_ds_mla(
        flat_cache.data_ptr(), ckv.data_ptr(), kpe.data_ptr(),
        None, indptr.data_ptr(),
        batch_indices.data_ptr(), positions.data_ptr(),
        nnz, PAGE_SIZE, kv_lora_rank, pe_dim,
        kv_lora_rank, pe_dim,
    )
    torch.cuda.synchronize(device)

    ref = ref_quantize_ds_mla(ckv, kpe, kv_lora_rank, pe_dim)

    idx = 0
    for b, sl in enumerate(seq_lens):
        pages_for_seq = (sl + PAGE_SIZE - 1) // PAGE_SIZE
        seq_start_slot = int(indptr[b]) * PAGE_SIZE
        for p in range(pages_for_seq):
            page_start = p * PAGE_SIZE
            n = min(PAGE_SIZE, sl - page_start)
            slot = seq_start_slot + page_start
            actual = flat_cache[slot:slot + n]
            ref_slice = ref[idx:idx + n]
            assert actual.shape == ref_slice.shape, \
                f"seq {b} page {p}: shape {actual.shape} vs {ref_slice.shape}"
            ckv_a, kpe_a = ref_dequantize_ds_mla(actual, kv_lora_rank, pe_dim)
            ckv_r, kpe_r = ref_dequantize_ds_mla(ref_slice, kv_lora_rank, pe_dim)
            assert torch.allclose(ckv_a, ckv_r, atol=1e-1, rtol=1e-1), \
                f"seq {b} page {p}: ckv mismatch"
            assert torch.equal(kpe_a, kpe_r), f"seq {b} page {p}: kpe mismatch"
            idx += n

        pad_start = seq_start_slot + sl
        pad_end = seq_start_slot + pages_for_seq * PAGE_SIZE
        if pad_end > pad_start:
            assert (flat_cache[pad_start:pad_end] == 0).all(), \
                f"seq {b}: padding slots should be zero"


@pytest.mark.parametrize("kv_lora_rank,pe_dim", [
    (512, 64),
    (128, 64),
])
def test_flat_write_matches_identity_paged(glm, device, kv_lora_rank, pe_dim):
    """Flat write (indices=None) matches paged write with identity indices."""
    nnz = 200
    num_pages = (nnz + PAGE_SIZE - 1) // PAGE_SIZE
    num_tiles = kv_lora_rank // 128
    bpt = kv_lora_rank + num_tiles * 4 + pe_dim * 2

    ckv = torch.randn(nnz, kv_lora_rank, dtype=torch.bfloat16, device=device)
    kpe = torch.randn(nnz, pe_dim, dtype=torch.bfloat16, device=device)

    identity_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    batch_indices = torch.zeros(nnz, dtype=torch.int32, device=device)
    positions = torch.arange(nnz, dtype=torch.int32, device=device)

    paged_cache = torch.zeros(num_pages, PAGE_SIZE, bpt, dtype=torch.uint8, device=device)
    glm.concat_and_cache_ds_mla(
        paged_cache.data_ptr(), ckv.data_ptr(), kpe.data_ptr(),
        identity_indices.data_ptr(), indptr.data_ptr(),
        batch_indices.data_ptr(), positions.data_ptr(),
        nnz, PAGE_SIZE, kv_lora_rank, pe_dim,
        kv_lora_rank, pe_dim,
    )

    flat_cache = torch.zeros(num_pages * PAGE_SIZE, bpt, dtype=torch.uint8, device=device)
    glm.concat_and_cache_ds_mla(
        flat_cache.data_ptr(), ckv.data_ptr(), kpe.data_ptr(),
        None, indptr.data_ptr(),
        batch_indices.data_ptr(), positions.data_ptr(),
        nnz, PAGE_SIZE, kv_lora_rank, pe_dim,
        kv_lora_rank, pe_dim,
    )
    torch.cuda.synchronize(device)

    paged_flat = paged_cache.reshape(num_pages * PAGE_SIZE, bpt)
    assert torch.equal(paged_flat[:nnz], flat_cache[:nnz]), \
        "flat write should match identity-indexed paged write"


@pytest.mark.parametrize("kv_lora_rank,pe_dim", [
    (512, 64),
])
def test_flat_write_vs_scattered_paged(glm, device, kv_lora_rank, pe_dim):
    """Flat write ignores page scattering; paged write respects non-identity indices."""
    nnz = 64
    num_pages = 1
    num_tiles = kv_lora_rank // 128
    bpt = kv_lora_rank + num_tiles * 4 + pe_dim * 2

    ckv = torch.randn(nnz, kv_lora_rank, dtype=torch.bfloat16, device=device)
    kpe = torch.randn(nnz, pe_dim, dtype=torch.bfloat16, device=device)

    scattered_indices = torch.tensor([5], dtype=torch.int32, device=device)
    indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    batch_indices = torch.zeros(nnz, dtype=torch.int32, device=device)
    positions = torch.arange(nnz, dtype=torch.int32, device=device)

    paged_cache = torch.zeros(10, PAGE_SIZE, bpt, dtype=torch.uint8, device=device)
    glm.concat_and_cache_ds_mla(
        paged_cache.data_ptr(), ckv.data_ptr(), kpe.data_ptr(),
        scattered_indices.data_ptr(), indptr.data_ptr(),
        batch_indices.data_ptr(), positions.data_ptr(),
        nnz, PAGE_SIZE, kv_lora_rank, pe_dim,
        kv_lora_rank, pe_dim,
    )

    flat_cache = torch.zeros(10 * PAGE_SIZE, bpt, dtype=torch.uint8, device=device)
    glm.concat_and_cache_ds_mla(
        flat_cache.data_ptr(), ckv.data_ptr(), kpe.data_ptr(),
        None, indptr.data_ptr(),
        batch_indices.data_ptr(), positions.data_ptr(),
        nnz, PAGE_SIZE, kv_lora_rank, pe_dim,
        kv_lora_rank, pe_dim,
    )
    torch.cuda.synchronize(device)

    paged_data = paged_cache[5, :nnz]
    flat_data = flat_cache[:nnz]
    assert torch.equal(paged_data, flat_data), \
        "scattered paged write at page 5 should match flat write at page 0"

    assert (paged_cache[0, :nnz] == 0).all(), \
        "page 0 in paged cache should be empty (data went to page 5)"
    assert (flat_cache[5 * PAGE_SIZE:5 * PAGE_SIZE + nnz] == 0).all(), \
        "flat write should not touch page 5 slots"


@pytest.mark.parametrize("kv_lora_rank,pe_dim", [
    (512, 64),
])
def test_flat_write_multi_seq_matches_paged(glm, device, kv_lora_rank, pe_dim):
    """Multi-sequence flat write matches identity-indexed paged write."""
    num_tiles = kv_lora_rank // 128
    bpt = kv_lora_rank + num_tiles * 4 + pe_dim * 2

    seq_lens = [64, 30, 100]
    nnz = sum(seq_lens)
    num_pages = sum((sl + PAGE_SIZE - 1) // PAGE_SIZE for sl in seq_lens)

    ckv = torch.randn(nnz, kv_lora_rank, dtype=torch.bfloat16, device=device)
    kpe = torch.randn(nnz, pe_dim, dtype=torch.bfloat16, device=device)

    identity_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    indptr = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=device)
    for i, sl in enumerate(seq_lens):
        indptr[i + 1] = indptr[i] + (sl + PAGE_SIZE - 1) // PAGE_SIZE

    batch_indices = torch.zeros(nnz, dtype=torch.int32, device=device)
    positions = torch.zeros(nnz, dtype=torch.int32, device=device)
    idx = 0
    for b, sl in enumerate(seq_lens):
        batch_indices[idx:idx + sl] = b
        positions[idx:idx + sl] = torch.arange(sl, dtype=torch.int32, device=device)
        idx += sl

    paged_cache = torch.zeros(num_pages, PAGE_SIZE, bpt, dtype=torch.uint8, device=device)
    glm.concat_and_cache_ds_mla(
        paged_cache.data_ptr(), ckv.data_ptr(), kpe.data_ptr(),
        identity_indices.data_ptr(), indptr.data_ptr(),
        batch_indices.data_ptr(), positions.data_ptr(),
        nnz, PAGE_SIZE, kv_lora_rank, pe_dim,
        kv_lora_rank, pe_dim,
    )

    flat_cache = torch.zeros(num_pages * PAGE_SIZE, bpt, dtype=torch.uint8, device=device)
    glm.concat_and_cache_ds_mla(
        flat_cache.data_ptr(), ckv.data_ptr(), kpe.data_ptr(),
        None, indptr.data_ptr(),
        batch_indices.data_ptr(), positions.data_ptr(),
        nnz, PAGE_SIZE, kv_lora_rank, pe_dim,
        kv_lora_rank, pe_dim,
    )
    torch.cuda.synchronize(device)

    paged_flat = paged_cache.reshape(num_pages * PAGE_SIZE, bpt)
    assert torch.equal(paged_flat, flat_cache), \
        "multi-seq flat write should match identity-indexed paged write"
