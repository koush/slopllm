"""
Microbenchmark for nvfp4_mul_mat_id at MoE decode shapes (BS=1, TP=8).

  gate/up_proj: N=256 (2048/8), K=6144, count=8, top_k=8
  down_proj:    N=6144,         K=256 (2048/8), count=8, top_k=1
"""
import time
import ctypes
import torch
import pytest

from helpers import GlmOps, GpuPtrs

# NVFP4 quantization
QUANT_GROUP = 16


def quantize_nvfp4(weight_bf16, device):
    """Fake NVFP4 quantization: pack two e2m1 values per byte, fp8e4m3 per-group scale."""
    N, K = weight_bf16.shape
    num_groups = K // QUANT_GROUP
    weight_float = weight_bf16.float()

    # Per-group max for scale
    groups = weight_float.view(N, num_groups, QUANT_GROUP)
    group_max = groups.abs().amax(dim=2, keepdim=True)  # [N, num_groups, 1]

    # FP4 range: ±6.0 (e2m1 max value)
    fp4_max = 6.0
    scale = group_max / fp4_max
    scale = scale.clamp(min=1e-8)
    scale_fp8 = scale.to(torch.float8_e4m3fn).to(torch.float32)
    # Per-tensor scale2 = 1.0
    scale2 = torch.tensor([1.0], dtype=torch.float32, device=device)

    # Quantize to FP4 e2m1
    scaled = groups / scale_fp8
    # FP4 e2m1 values: {0, 1, 2, 3, 4, 5, 6, 7} -> {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    # Round to nearest FP4 value
    fp4_lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device=device)
    # Clamp and round
    scaled_abs = scaled.abs().clamp(max=6.0)
    # Find nearest FP4 value
    signs = torch.sign(scaled)
    indices = torch.argmin((scaled_abs.unsqueeze(-1) - fp4_lut).abs(), dim=-1)
    quantized = signs * fp4_lut[indices]

    # Pack two FP4 values per byte
    # Convert to nibbles: indices give 0-7 for abs value, add sign bit
    nibbles = indices * 2  # shift left by 1 for sign
    nibbles = torch.where(signs < 0, nibbles | 1, nibbles)
    nibbles = nibbles.to(torch.uint8)

    # Pack: lo nibble in bits 0-3, hi nibble in bits 4-7
    packed = nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)
    # packed shape: [N, K/2]
    packed = packed.contiguous()

    # Scale per-row: [N, num_groups] as fp8e4m3
    scale_per_row = scale_fp8.squeeze(2).to(torch.float8_e4m3fn).contiguous()

    return packed, scale_per_row, scale2


def bench_nvfp4_mul_mat_id(glm, device, N, K, count, top_k, label):
    """Benchmark nvfp4_mul_mat_id for given shape."""
    # Create input: [batch, K] where batch = count / top_k
    batch = count // top_k
    input_bf16 = torch.randn(batch, K, dtype=torch.bfloat16, device=device)

    # Create 256 expert weights (only need a few, but create enough for pointer array)
    num_experts = 256
    weights = []
    scales = []
    for _ in range(num_experts):
        w = torch.randn(N, K, dtype=torch.bfloat16, device=device)
        packed, scale_per_row, _ = quantize_nvfp4(w, device)
        weights.append(packed)
        scales.append(scale_per_row)

    # Scale2 for each expert (all 1.0)
    scale2_vals = torch.ones(num_experts, dtype=torch.float32, device=device)

    # Expert IDs
    expert_ids = torch.randint(0, num_experts, (count,), dtype=torch.int32, device=device)

    # Output
    output = torch.empty(count, N, dtype=torch.bfloat16, device=device)

    # Pointer arrays
    weight_ptrs = GpuPtrs([w.data_ptr() for w in weights], device)
    scale_ptrs = GpuPtrs([s.data_ptr() for s in scales], device)
    scale2_ptrs = GpuPtrs([scale2_vals.data_ptr()] * num_experts, device)

    # Warmup
    for _ in range(50):
        glm.lib.glm_nvfp4_mul_mat_id(
            glm.ctx,
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_void_p(input_bf16.data_ptr()),
            ctypes.c_void_p(weight_ptrs.data_ptr),
            ctypes.c_void_p(scale_ptrs.data_ptr),
            ctypes.c_void_p(scale2_ptrs.data_ptr),
            ctypes.c_void_p(expert_ids.data_ptr()),
            top_k,
            count, N, K
        )
    glm.synchronize()

    # Time with perf_counter
    iters = 1000
    t0 = time.perf_counter()
    for _ in range(iters):
        glm.lib.glm_nvfp4_mul_mat_id(
            glm.ctx,
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_void_p(input_bf16.data_ptr()),
            ctypes.c_void_p(weight_ptrs.data_ptr),
            ctypes.c_void_p(scale_ptrs.data_ptr),
            ctypes.c_void_p(scale2_ptrs.data_ptr),
            ctypes.c_void_p(expert_ids.data_ptr()),
            top_k,
            count, N, K
        )
    glm.synchronize()
    elapsed = (time.perf_counter() - t0) / iters * 1e9  # ns

    # Data read: weight (count experts * N * K/2 bytes) + input (batch * K * 2) + scale (count * N * K/16 * 1)
    weight_bytes = count * N * (K // 2)
    input_bytes = batch * K * 2
    scale_bytes = count * N * (K // QUANT_GROUP)
    total_bytes = weight_bytes + input_bytes + scale_bytes
    bw_gbps = total_bytes / elapsed  # GB/s (bytes / ns)

    print(f"  {label:20s} N={N:5d} K={K:5d} count={count}  {elapsed:8.1f} ns  {bw_gbps:7.1f} GB/s  grid={'splitk' if N < 1024 else 'rpw8'}")
    return elapsed


@pytest.mark.parametrize("N,K,count,top_k,label", [
    (256, 6144, 8, 8, "gate/up_proj"),
    (6144, 256, 8, 1, "down_proj"),
])
def test_bench_nvfp4_mul_mat_id(glm, device, N, K, count, top_k, label):
    bench_nvfp4_mul_mat_id(glm, device, N, K, count, top_k, label)
