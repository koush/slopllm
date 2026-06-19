#!/usr/bin/env python3
"""Benchmark the PC kernel with configurable defines.

Usage:
  python scratchpad/bench_pc.py                     # full pipeline
  python scratchpad/bench_pc.py --define PRODUCER_ONLY
  python scratchpad/bench_pc.py --define CONSUMER_NOOP
  python scratchpad/bench_pc.py --define SKIP_WRITEBACK
  python scratchpad/bench_pc.py --compare-mma
  python scratchpad/bench_pc.py --small              # smaller dims for quick iteration
"""

import argparse
import ctypes
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests", "python"))
from helpers import GlmOps, GpuPtrs
from test_nvfp4 import quantize_weight_nvfp4, GROUP_SIZE


def build_kernel(defines=None):
    """Rebuild the .so with optional -D flags."""
    defines = defines or []
    nvcc_flags = " ".join(f"-D{d}" for d in defines)
    obj = "build/Release/glm_mma_moe_pc.o"
    so = "build/Release/libglm_ops.so"
    subprocess.run(["rm", "-f", obj, so], check=False)
    compile_cmd = (
        f"nvcc -O2 -std=c++20 -Xcompiler -fPIC -Icsrc "
        f"-gencode arch=compute_120a,code=sm_120a "
        f"--expt-relaxed-constexpr --extended-lambda "
        f"{nvcc_flags} -c csrc/glm_mma_moe_pc.cu -o {obj}"
    )
    r = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"COMPILE ERROR:\n{r.stderr}")
        sys.exit(1)
    link_cmd = (
        f"nvcc -O2 -std=c++20 -Xcompiler -fPIC "
        f"-gencode arch=compute_120a,code=sm_120a "
        f"--expt-relaxed-constexpr --extended-lambda -shared "
        f"-o {so} build/Release/glm_ops.o build/Release/glm_flash.o "
        f"build/Release/glm_gemv.o build/Release/glm_gdn.o "
        f"build/Release/glm_sampling.o build/Release/glm_p2p.o "
        f"build/Release/glm_context_parallel.o "
        f"build/Release/glm_grouped_moe.o build/Release/glm_mma_moe.o "
        f"build/Release/glm_mma_moe_pc.o build/Release/glm_nccl.o "
        f"build/Release/glm_device.o "
        f"-Ivendor/flashinfer/include "
        f"-Ivendor/flashinfer/3rdparty/cccl/libcudacxx/include "
        f"-I/usr/local/cuda/include "
        f"-L/usr/local/cuda/lib64 -lcublas -lcudart -lnccl"
    )
    r = subprocess.run(link_cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"LINK ERROR:\n{r.stderr}")
        sys.exit(1)


def setup_weights(glm, device_id, N, K, num_experts):
    weight_ptrs_list, scale_ptrs_list, scale2_ptrs_list = [], [], []
    for _ in range(num_experts):
        w = torch.randn(N, K, dtype=torch.bfloat16, device=f"cuda:{device_id}") * 0.5
        packed, scale, scale2 = quantize_weight_nvfp4(w.cpu())
        fp4_gpu = glm.alloc(N * (K // 2))
        nkg = K // GROUP_SIZE
        scale_gpu = glm.alloc(N * nkg)
        scale2_gpu = glm.alloc(4)
        glm.h2d(fp4_gpu, packed.view(torch.uint8).numpy().ctypes.data_as(ctypes.c_void_p), N * (K // 2))
        glm.h2d(scale_gpu, scale.view(torch.uint8).numpy().ctypes.data_as(ctypes.c_void_p), N * nkg)
        glm.h2d(scale2_gpu, scale2.numpy().ctypes.data_as(ctypes.c_void_p), 4)
        weight_ptrs_list.append(fp4_gpu)
        scale_ptrs_list.append(scale_gpu)
        scale2_ptrs_list.append(scale2_gpu)
    wp = GpuPtrs(weight_ptrs_list, device=f"cuda:{device_id}")
    sp = GpuPtrs(scale_ptrs_list, device=f"cuda:{device_id}")
    s2p = GpuPtrs(scale2_ptrs_list, device=f"cuda:{device_id}")
    return wp, sp, s2p


def bench_pc(glm, device_id, batch, K, N, num_experts, topK, weight_ptrs, scale_ptrs, scale2_ptrs,
             warmup=5, iters=20):
    device = f"cuda:{device_id}"
    count = batch * topK
    input_bf16 = torch.randn(batch, K, dtype=torch.bfloat16, device=device) * 0.5
    expert_ids = torch.randint(0, num_experts, (batch, topK), dtype=torch.int32, device=device)
    expert_ids_flat = expert_ids.reshape(-1)
    out = torch.empty(count, N, dtype=torch.bfloat16, device=device)
    ws_size = glm.lib.glm_mma_moe_pc_workspace_size(count, N, K, num_experts)
    ws_ptr = glm.alloc(ws_size)

    def run():
        glm.lib.glm_nvfp4_mul_mat_id_grouped_mma_pc(
            glm.ctx,
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(input_bf16.data_ptr()),
            ctypes.c_void_p(weight_ptrs.data_ptr),
            ctypes.c_void_p(scale_ptrs.data_ptr),
            ctypes.c_void_p(scale2_ptrs.data_ptr),
            ctypes.c_void_p(expert_ids_flat.data_ptr()),
            topK, count, N, K, num_experts,
            ctypes.c_void_p(ws_ptr),
        )

    for _ in range(warmup):
        run()
    glm.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run()
        glm.synchronize()
        times.append(time.perf_counter() - t0)

    gmem = count * (K + N) * 2 + count * N * 2
    avg = np.mean(times) * 1000
    p50 = np.percentile(times, 50) * 1000
    mn = np.min(times) * 1000
    bw = gmem / np.mean(times) / 1e9
    return {"avg_ms": avg, "p50_ms": p50, "min_ms": mn, "bw_gbs": bw}


def bench_mma(glm, device_id, batch, K, N, num_experts, topK, weight_ptrs, scale_ptrs, scale2_ptrs,
              warmup=5, iters=20):
    device = f"cuda:{device_id}"
    count = batch * topK
    input_bf16 = torch.randn(batch, K, dtype=torch.bfloat16, device=device) * 0.5
    expert_ids = torch.randint(0, num_experts, (batch, topK), dtype=torch.int32, device=device)
    expert_ids_flat = expert_ids.reshape(-1)
    out = torch.empty(count, N, dtype=torch.bfloat16, device=device)
    ws_size = glm.lib.glm_mma_moe_workspace_size(count, N, K, num_experts)
    ws_ptr = glm.alloc(ws_size)

    def run():
        glm.lib.glm_nvfp4_mul_mat_id_grouped_mma(
            glm.ctx,
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(input_bf16.data_ptr()),
            ctypes.c_void_p(weight_ptrs.data_ptr),
            ctypes.c_void_p(scale_ptrs.data_ptr),
            ctypes.c_void_p(scale2_ptrs.data_ptr),
            ctypes.c_void_p(expert_ids_flat.data_ptr()),
            topK, count, N, K, num_experts,
            ctypes.c_void_p(ws_ptr),
        )

    for _ in range(warmup):
        run()
    glm.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run()
        glm.synchronize()
        times.append(time.perf_counter() - t0)

    gmem = count * (K + N) * 2 + count * N * 2
    avg = np.mean(times) * 1000
    p50 = np.percentile(times, 50) * 1000
    mn = np.min(times) * 1000
    bw = gmem / np.mean(times) / 1e9
    return {"avg_ms": avg, "p50_ms": p50, "min_ms": mn, "bw_gbs": bw}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--define", "-D", action="append", default=[], help="Add -D flag to nvcc")
    parser.add_argument("--compare-mma", action="store_true", help="Also benchmark the MMA kernel")
    parser.add_argument("--small", action="store_true", help="Use smaller dims (batch=64)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--no-rebuild", action="store_true", help="Skip rebuild step")
    args = parser.parse_args()

    device_id = args.gpu
    defines = args.define

    if not args.no_rebuild:
        label = "+".join(defines) if defines else "default"
        print(f"Building kernel [{label}]...", flush=True)
        build_kernel(defines)

    glm = GlmOps(device_id=device_id)

    if args.small:
        batch, K, N, num_experts, topK = 64, 7168, 2048, 256, 4
    else:
        batch, K, N, num_experts, topK = 4096, 7168, 2048, 256, 4

    count = batch * topK
    gmem = count * (K + N) * 2 + count * N * 2
    print(f"Config: batch={batch} K={K} N={N} experts={num_experts} topK={topK} count={count}")
    print(f"GMEM traffic: {gmem/1e9:.1f} GB  (peak HBM: 1792 GB/s)")

    print(f"Quantizing {num_experts} experts...", flush=True)
    wp, sp, s2p = setup_weights(glm, device_id, N, K, num_experts)

    label = "+".join(defines) if defines else "full pipeline"
    print(f"\n--- PC kernel [{label}] ---")
    r = bench_pc(glm, device_id, batch, K, N, num_experts, topK, wp, sp, s2p,
                 warmup=args.warmup, iters=args.iters)
    print(f"  avg={r['avg_ms']:.1f}ms  p50={r['p50_ms']:.1f}ms  min={r['min_ms']:.1f}ms  BW={r['bw_gbs']:.0f} GB/s")

    if args.compare_mma:
        print(f"\n--- MMA kernel (existing) ---")
        r_mma = bench_mma(glm, device_id, batch, K, N, num_experts, topK, wp, sp, s2p,
                           warmup=args.warmup, iters=args.iters)
        print(f"  avg={r_mma['avg_ms']:.1f}ms  p50={r_mma['p50_ms']:.1f}ms  min={r_mma['min_ms']:.1f}ms  BW={r_mma['bw_gbs']:.0f} GB/s")
        print(f"  Speedup: {r_mma['avg_ms']/r['avg_ms']:.2f}x")

    # Cleanup
    glm.free_buf(wp.base if hasattr(wp, 'base') else 0)


if __name__ == "__main__":
    main()
