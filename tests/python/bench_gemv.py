"""
Microbenchmark for the BF16 GEMV kernel at the exact shapes used by
Qwen3-32B with tensor-parallel = 8.

  Q-proj  : N=1024 (8192/8) K=5120
  K/V-proj: N=128  (1024/8) K=5120
  O-proj  : N=5120          K=1024 (8192/8)  -- row parallel
  gate/up : N=3456 (27648/8) K=5120
  down    : N=5120          K=3456 (27648/8) -- row parallel
  lm_head : N=18992 (151936/8) K=5120
"""
import time
import torch
import pytest

GLM_SHAPES = [
    ("Q",       1024, 5120),
    ("K",       128,  5120),
    ("V",       128,  5120),
    ("O",       5120, 1024),
    ("gate",    3456, 5120),
    ("up",      3456, 5120),
    ("down",    5120, 3456),
    ("lm_head", 18992,5120),
]


@pytest.mark.parametrize("name,n,k", GLM_SHAPES)
def test_bench_gemv(glm, device, name, n, k):
    x = torch.randn(1, k, dtype=torch.bfloat16, device=device)
    w = torch.randn(n, k, dtype=torch.bfloat16, device=device)
    out = torch.empty(1, n, dtype=torch.bfloat16, device=device)

    # Warm
    for _ in range(20):
        glm.linear(out, x, w, 1, n, k)
    torch.cuda.synchronize()

    # Time
    iters = 1000
    t0 = time.perf_counter()
    for _ in range(iters):
        glm.linear(out, x, w, 1, n, k)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / iters * 1e6  # µs

    bytes_read = (n * k * 2) + (k * 2)  # weight + input
    bw_gbps = bytes_read / 1e9 / (elapsed / 1e6)

    print(f"  {name:8s} N={n:5d} K={k:5d}  {elapsed:7.2f} us  {bw_gbps:7.1f} GB/s")
