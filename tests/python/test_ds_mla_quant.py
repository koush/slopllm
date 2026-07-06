import torch
import pytest
import numpy as np
from helpers import GlmOps

PAGE_SIZE = 64
FP8_MAX = 448.0


def ref_quantize_ds_mla(ckv_bf16, kpe_bf16, kv_lora_rank, pe_dim):
    """Reference: per-128-block FP8 e4m3 quantization with FP32 scales, BF16 kpe copy."""
    nnz = ckv_bf16.shape[0]
    num_tiles = kv_lora_rank // 128
    bpt = kv_lora_rank + num_tiles * 4 + pe_dim * 2

    out = torch.zeros(nnz, bpt, dtype=torch.uint8, device=ckv_bf16.device)

    ckv_f32 = ckv_bf16.float()
    for t in range(num_tiles):
        tile = ckv_f32[:, t * 128:(t + 1) * 128]
        max_abs = tile.abs().amax(dim=-1, keepdim=True)
        scale = (max_abs / FP8_MAX).clamp(min=1e-38)
        quantized = (tile / scale).clamp(-FP8_MAX, FP8_MAX)
        fp8 = quantized.to(torch.float8_e4m3fn)
        out[:, t * 128:(t + 1) * 128] = fp8.view(torch.uint8)
        scale_bytes = scale.view(torch.uint8).view(-1, 4).expand(nnz, -1)
        out[:, kv_lora_rank + t * 4:kv_lora_rank + (t + 1) * 4] = scale_bytes

    kpe_offset = kv_lora_rank + num_tiles * 4
    kpe_bytes = kpe_bf16.view(torch.uint8).reshape(nnz, pe_dim * 2)
    out[:, kpe_offset:kpe_offset + pe_dim * 2] = kpe_bytes

    return out


def ref_dequantize_ds_mla(packed, kv_lora_rank, pe_dim):
    """Dequantize packed format back to BF16 for comparison."""
    nnz = packed.shape[0]
    num_tiles = kv_lora_rank // 128
    kpe_offset = kv_lora_rank + num_tiles * 4

    ckv_deq = torch.zeros(nnz, kv_lora_rank, dtype=torch.bfloat16, device=packed.device)
    for t in range(num_tiles):
        fp8_bytes = packed[:, t * 128:(t + 1) * 128]
        fp8 = fp8_bytes.view(torch.float8_e4m3fn)
        scale_bytes = packed[:, kv_lora_rank + t * 4:kv_lora_rank + (t + 1) * 4]
        scale = scale_bytes.view(torch.uint8).view(torch.float32)
        ckv_deq[:, t * 128:(t + 1) * 128] = (fp8.float() * scale).to(torch.bfloat16)

    kpe_bytes = packed[:, kpe_offset:kpe_offset + pe_dim * 2]
    kpe = kpe_bytes.view(torch.bfloat16).reshape(nnz, pe_dim)

    return ckv_deq, kpe


@pytest.mark.parametrize("kv_lora_rank,pe_dim", [
    (512, 64),
    (128, 64),
])
def test_ds_mla_quant_correctness(glm, device, kv_lora_rank, pe_dim):
    nnz = 128
    num_pages = (nnz + PAGE_SIZE - 1) // PAGE_SIZE
    num_tiles = kv_lora_rank // 128
    bpt = kv_lora_rank + num_tiles * 4 + pe_dim * 2

    ckv = torch.randn(nnz, kv_lora_rank, dtype=torch.bfloat16, device=device)
    kpe = torch.randn(nnz, pe_dim, dtype=torch.bfloat16, device=device)

    kv_cache = torch.zeros(num_pages, PAGE_SIZE, bpt, dtype=torch.uint8, device=device)

    indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    batch_indices = torch.zeros(nnz, dtype=torch.int32, device=device)
    positions = torch.arange(nnz, dtype=torch.int32, device=device)

    glm.concat_and_cache_ds_mla(
        kv_cache.data_ptr(), ckv.data_ptr(), kpe.data_ptr(),
        indices.data_ptr(), indptr.data_ptr(),
        batch_indices.data_ptr(), positions.data_ptr(),
        nnz, PAGE_SIZE, kv_lora_rank, pe_dim,
        kv_lora_rank, pe_dim,
    )
    torch.cuda.synchronize(device)

    ref = ref_quantize_ds_mla(ckv, kpe, kv_lora_rank, pe_dim)
    ref_flat = ref.reshape(num_pages * PAGE_SIZE, bpt)[:nnz]
    actual_flat = kv_cache.reshape(num_pages * PAGE_SIZE, bpt)[:nnz]

    ckv_deq_actual, kpe_actual = ref_dequantize_ds_mla(actual_flat, kv_lora_rank, pe_dim)
    ckv_deq_ref, kpe_ref = ref_dequantize_ds_mla(ref_flat, kv_lora_rank, pe_dim)

    assert torch.allclose(ckv_deq_actual, ckv_deq_ref, atol=1e-1, rtol=1e-1), \
        f"ckv dequant mismatch: max diff {(ckv_deq_actual.float() - ckv_deq_ref.float()).abs().max().item()}"
    assert torch.equal(kpe_actual, kpe), "kpe should be exact BF16 copy"


@pytest.mark.parametrize("kv_lora_rank,pe_dim", [
    (512, 64),
])
def test_ds_mla_quant_multi_seq(glm, device, kv_lora_rank, pe_dim):
    num_tiles = kv_lora_rank // 128
    bpt = kv_lora_rank + num_tiles * 4 + pe_dim * 2

    seq_lens = [64, 30, 100]
    nnz = sum(seq_lens)
    num_pages = sum((sl + PAGE_SIZE - 1) // PAGE_SIZE for sl in seq_lens)

    ckv = torch.randn(nnz, kv_lora_rank, dtype=torch.bfloat16, device=device)
    kpe = torch.randn(nnz, pe_dim, dtype=torch.bfloat16, device=device)
    kv_cache = torch.zeros(num_pages, PAGE_SIZE, bpt, dtype=torch.uint8, device=device)

    indices = torch.arange(num_pages, dtype=torch.int32, device=device)
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
        kv_cache.data_ptr(), ckv.data_ptr(), kpe.data_ptr(),
        indices.data_ptr(), indptr.data_ptr(),
        batch_indices.data_ptr(), positions.data_ptr(),
        nnz, PAGE_SIZE, kv_lora_rank, pe_dim,
        kv_lora_rank, pe_dim,
    )
    torch.cuda.synchronize(device)

    ref = ref_quantize_ds_mla(ckv, kpe, kv_lora_rank, pe_dim)

    idx = 0
    for b, sl in enumerate(seq_lens):
        pages_for_seq = (sl + PAGE_SIZE - 1) // PAGE_SIZE
        for p in range(pages_for_seq):
            page_id = int(indptr[b]) + p
            start = p * PAGE_SIZE
            end = min(start + PAGE_SIZE, sl)
            n = end - start
            actual = kv_cache[page_id, :n]
            ref_slice = ref[idx:idx + n]
            assert actual.shape == ref_slice.shape, f"seq {b} page {p}: shape {actual.shape} vs {ref_slice.shape}"
            ckv_a, kpe_a = ref_dequantize_ds_mla(actual, kv_lora_rank, pe_dim)
            ckv_r, kpe_r = ref_dequantize_ds_mla(ref_slice, kv_lora_rank, pe_dim)
            assert torch.allclose(ckv_a, ckv_r, atol=1e-1, rtol=1e-1), \
                f"seq {b} page {p}: ckv mismatch"
            assert torch.equal(kpe_a, kpe_r), f"seq {b} page {p}: kpe mismatch"
            idx += n


@pytest.mark.parametrize("kv_lora_rank,pe_dim", [
    (512, 64),
])
def test_ds_mla_quant_zero_input(glm, device, kv_lora_rank, pe_dim):
    """Zero input should produce zero FP8 values and FLT_MIN scales."""
    nnz = 16
    num_tiles = kv_lora_rank // 128
    bpt = kv_lora_rank + num_tiles * 4 + pe_dim * 2
    num_pages = 1

    ckv = torch.zeros(nnz, kv_lora_rank, dtype=torch.bfloat16, device=device)
    kpe = torch.zeros(nnz, pe_dim, dtype=torch.bfloat16, device=device)
    kv_cache = torch.zeros(num_pages, PAGE_SIZE, bpt, dtype=torch.uint8, device=device)

    indices = torch.zeros(1, dtype=torch.int32, device=device)
    indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    batch_indices = torch.zeros(nnz, dtype=torch.int32, device=device)
    positions = torch.arange(nnz, dtype=torch.int32, device=device)

    glm.concat_and_cache_ds_mla(
        kv_cache.data_ptr(), ckv.data_ptr(), kpe.data_ptr(),
        indices.data_ptr(), indptr.data_ptr(),
        batch_indices.data_ptr(), positions.data_ptr(),
        nnz, PAGE_SIZE, kv_lora_rank, pe_dim,
        kv_lora_rank, pe_dim,
    )
    torch.cuda.synchronize(device)

    actual = kv_cache.reshape(-1, bpt)[:nnz]
    fp8_data = actual[:, :kv_lora_rank]
    assert (fp8_data == 0).all(), "Zero input should produce zero FP8 values"
    kpe_data = actual[:, kv_lora_rank + num_tiles * 4:].view(torch.bfloat16).reshape(nnz, pe_dim)
    assert (kpe_data == 0).all(), "Zero kpe should produce zero BF16 values"


@pytest.mark.parametrize("kv_lora_rank,pe_dim", [
    (512, 64),
])
def test_ds_mla_quant_layout(glm, device, kv_lora_rank, pe_dim):
    """Verify exact byte layout: [FP8 ckv | FP32 scales | BF16 kpe]."""
    nnz = 1
    num_tiles = kv_lora_rank // 128
    bpt = kv_lora_rank + num_tiles * 4 + pe_dim * 2
    assert bpt == 656 if kv_lora_rank == 512 else bpt == 260

    ckv = torch.randn(nnz, kv_lora_rank, dtype=torch.bfloat16, device=device)
    kpe = torch.randn(nnz, pe_dim, dtype=torch.bfloat16, device=device)
    kv_cache = torch.zeros(1, PAGE_SIZE, bpt, dtype=torch.uint8, device=device)

    indices = torch.zeros(1, dtype=torch.int32, device=device)
    indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    batch_indices = torch.zeros(nnz, dtype=torch.int32, device=device)
    positions = torch.zeros(nnz, dtype=torch.int32, device=device)

    glm.concat_and_cache_ds_mla(
        kv_cache.data_ptr(), ckv.data_ptr(), kpe.data_ptr(),
        indices.data_ptr(), indptr.data_ptr(),
        batch_indices.data_ptr(), positions.data_ptr(),
        nnz, PAGE_SIZE, kv_lora_rank, pe_dim,
        kv_lora_rank, pe_dim,
    )
    torch.cuda.synchronize(device)

    slot = kv_cache[0, 0]

    fp8_region = slot[:kv_lora_rank]
    scale_region = slot[kv_lora_rank:kv_lora_rank + num_tiles * 4].view(torch.float32)
    kpe_region = slot[kv_lora_rank + num_tiles * 4:].view(torch.bfloat16)

    for t in range(num_tiles):
        tile = ckv[0, t * 128:(t + 1) * 128].float()
        expected_max = tile.abs().max().item()
        expected_scale = max(expected_max / FP8_MAX, 1e-38)
        actual_scale = scale_region[t].item()
        assert abs(actual_scale - expected_scale) < 1e-5, \
            f"tile {t}: scale {actual_scale} vs expected {expected_scale}"

    ref_fp8 = (ckv[0].float().reshape(num_tiles, 128) / scale_region.unsqueeze(1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).flatten()
    actual_fp8 = fp8_region.view(torch.float8_e4m3fn)
    assert torch.equal(actual_fp8, ref_fp8), "FP8 data should match reference"

    assert torch.equal(kpe_region, kpe[0]), "kpe should be exact copy"
