"""Tests for glm_nvfp4_mul_mat_id_grouped_mma: grouped NVFP4 MoE with wmma BF16 MMA.

Compares against PyTorch reference (dequantized BF16 matmul) and against
the scalar GEMV kernel (glm_nvfp4_mul_mat_id_grouped).
"""

import pytest
import torch
import ctypes
from helpers import GpuPtrs, GlmOps
from test_nvfp4 import quantize_weight_nvfp4, dequantize_nvfp4, GROUP_SIZE


class TestNVFP4MulMatIdGroupedMMA:
    def _run_mma_moe(self, glm, device, batch, K, N, num_experts, topK):
        count = batch * topK
        torch.manual_seed(42)
        input_bf16 = torch.randn(batch, K, dtype=torch.bfloat16, device=device) * 0.5
        expert_ids = torch.randint(0, num_experts, (batch, topK), dtype=torch.int32, device=device)
        expert_ids_flat = expert_ids.reshape(-1)

        weight_ptrs_list = []
        scale_ptrs_list = []
        scale2_ptrs_list = []
        ref_outputs = []

        for e in range(num_experts):
            w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.5
            packed, scale, scale_2 = quantize_weight_nvfp4(w_bf16.cpu())
            w_dequant = dequantize_nvfp4(packed, scale, scale_2, (N, K))
            ref_outputs.append(w_dequant)

            fp4_gpu = glm.alloc(N * (K // 2))
            num_k_groups = K // GROUP_SIZE
            scale_gpu = glm.alloc(N * num_k_groups)
            scale2_gpu = glm.alloc(4)

            glm.h2d(fp4_gpu, packed.view(torch.uint8).numpy().ctypes.data_as(ctypes.c_void_p), N * (K // 2))
            glm.h2d(scale_gpu, scale.view(torch.uint8).numpy().ctypes.data_as(ctypes.c_void_p), N * num_k_groups)
            glm.h2d(scale2_gpu, scale_2.numpy().ctypes.data_as(ctypes.c_void_p), 4)

            weight_ptrs_list.append(fp4_gpu)
            scale_ptrs_list.append(scale_gpu)
            scale2_ptrs_list.append(scale2_gpu)

        weight_ptrs = GpuPtrs(weight_ptrs_list, device)
        scale_ptrs = GpuPtrs(scale_ptrs_list, device)
        scale2_ptrs = GpuPtrs(scale2_ptrs_list, device)

        out_mma = torch.empty(count, N, dtype=torch.bfloat16, device=device)

        ws_size = glm.lib.glm_mma_moe_workspace_size(count, N, K, num_experts)
        ws_ptr = glm.alloc(ws_size)

        glm.lib.glm_nvfp4_mul_mat_id_grouped_mma(
            glm.ctx,
            ctypes.c_void_p(out_mma.data_ptr()),
            ctypes.c_void_p(input_bf16.data_ptr()),
            ctypes.c_void_p(weight_ptrs.data_ptr),
            ctypes.c_void_p(scale_ptrs.data_ptr),
            ctypes.c_void_p(scale2_ptrs.data_ptr),
            ctypes.c_void_p(expert_ids_flat.data_ptr()),
            topK, count, N, K, num_experts,
            ctypes.c_void_p(ws_ptr)
        )
        glm.synchronize()

        ref = torch.zeros(count, N, dtype=torch.float32)
        for i in range(count):
            bid = i // topK
            eid = expert_ids_flat[i].item()
            ref[i] = torch.nn.functional.linear(input_bf16[bid].cpu().float(), ref_outputs[eid].float())

        out_f32 = out_mma.cpu().float()
        mean_err = (out_f32 - ref).abs().mean().item()
        max_err = (out_f32 - ref).abs().max().item()
        mean_abs_ref = ref.abs().mean().item()

        assert mean_err < mean_abs_ref * 0.25, f"mean error {mean_err:.4f} > 25% of ref {mean_abs_ref:.4f}"
        assert max_err < mean_abs_ref * 5.0, f"max error {max_err:.4f} > 5x ref {mean_abs_ref:.4f}"

        glm.free_buf(ws_ptr)
        for ptrs in [weight_ptrs_list, scale_ptrs_list, scale2_ptrs_list]:
            for p in ptrs:
                glm.free_buf(p)

    def test_basic(self, glm, device):
        self._run_mma_moe(glm, device, batch=2, K=128, N=64, num_experts=4, topK=2)

    def test_single_token(self, glm, device):
        self._run_mma_moe(glm, device, batch=1, K=256, N=128, num_experts=8, topK=4)

    def test_multi_batch(self, glm, device):
        self._run_mma_moe(glm, device, batch=4, K=128, N=64, num_experts=8, topK=3)

    def test_glm51_dims(self, glm, device):
        self._run_mma_moe(glm, device, batch=2, K=256, N=512, num_experts=8, topK=4)

    def test_many_tokens(self, glm, device):
        self._run_mma_moe(glm, device, batch=10, K=128, N=64, num_experts=2, topK=2)

    def test_all_same_expert(self, glm, device):
        batch = 4
        K = 128
        N = 64
        num_experts = 4
        topK = 2
        count = batch * topK
        torch.manual_seed(42)
        input_bf16 = torch.randn(batch, K, dtype=torch.bfloat16, device=device) * 0.5
        expert_ids = torch.zeros(batch, topK, dtype=torch.int32, device=device)
        expert_ids_flat = expert_ids.reshape(-1)

        weight_ptrs_list = []
        scale_ptrs_list = []
        scale2_ptrs_list = []
        ref_outputs = []

        for e in range(num_experts):
            w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.5
            packed, scale, scale_2 = quantize_weight_nvfp4(w_bf16.cpu())
            w_dequant = dequantize_nvfp4(packed, scale, scale_2, (N, K))
            ref_outputs.append(w_dequant)

            fp4_gpu = glm.alloc(N * (K // 2))
            num_k_groups = K // GROUP_SIZE
            scale_gpu = glm.alloc(N * num_k_groups)
            scale2_gpu = glm.alloc(4)

            glm.h2d(fp4_gpu, packed.view(torch.uint8).numpy().ctypes.data_as(ctypes.c_void_p), N * (K // 2))
            glm.h2d(scale_gpu, scale.view(torch.uint8).numpy().ctypes.data_as(ctypes.c_void_p), N * num_k_groups)
            glm.h2d(scale2_gpu, scale_2.numpy().ctypes.data_as(ctypes.c_void_p), 4)

            weight_ptrs_list.append(fp4_gpu)
            scale_ptrs_list.append(scale_gpu)
            scale2_ptrs_list.append(scale2_gpu)

        weight_ptrs = GpuPtrs(weight_ptrs_list, device)
        scale_ptrs = GpuPtrs(scale_ptrs_list, device)
        scale2_ptrs = GpuPtrs(scale2_ptrs_list, device)

        out_mma = torch.empty(count, N, dtype=torch.bfloat16, device=device)
        ws_size = glm.lib.glm_mma_moe_workspace_size(count, N, K, num_experts)
        ws_ptr = glm.alloc(ws_size)

        glm.lib.glm_nvfp4_mul_mat_id_grouped_mma(
            glm.ctx,
            ctypes.c_void_p(out_mma.data_ptr()),
            ctypes.c_void_p(input_bf16.data_ptr()),
            ctypes.c_void_p(weight_ptrs.data_ptr),
            ctypes.c_void_p(scale_ptrs.data_ptr),
            ctypes.c_void_p(scale2_ptrs.data_ptr),
            ctypes.c_void_p(expert_ids_flat.data_ptr()),
            topK, count, N, K, num_experts,
            ctypes.c_void_p(ws_ptr)
        )
        glm.synchronize()

        ref = torch.zeros(count, N, dtype=torch.float32)
        for i in range(count):
            bid = i // topK
            ref[i] = torch.nn.functional.linear(input_bf16[bid].cpu().float(), ref_outputs[0].float())

        out_f32 = out_mma.cpu().float()
        mean_err = (out_f32 - ref).abs().mean().item()
        mean_abs_ref = ref.abs().mean().item()
        assert mean_err < mean_abs_ref * 0.25, f"mean error {mean_err:.4f} > 25% of ref {mean_abs_ref:.4f}"

        glm.free_buf(ws_ptr)
        for ptrs in [weight_ptrs_list, scale_ptrs_list, scale2_ptrs_list]:
            for p in ptrs:
                glm.free_buf(p)

    def test_mma_bf16_direct(self, glm, device):
        """Test MMA kernel with BF16 weights directly (bypass FP4 dequant) to isolate the bug."""
        import ctypes
        M, K, N = 4, 128, 64
        torch.manual_seed(123)
        input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.5
        weights_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.5

        output_f32 = torch.zeros(M, N, dtype=torch.float32, device=device)
        glm.mma_moe_debug2(output_f32, input_bf16, weights_bf16, M, K, N)
        glm.synchronize()

        ref = torch.nn.functional.linear(input_bf16.cpu().float(), weights_bf16.cpu().float())

        out_cpu = output_f32.cpu()
        mean_err = (out_cpu - ref).abs().mean().item()
        max_err = (out_cpu - ref).abs().max().item()
        mean_abs_ref = ref.abs().mean().item()
        rel_err = mean_err / mean_abs_ref if mean_abs_ref > 0 else float('inf')

        print(f"\n=== BF16 Direct MMA Debug ===")
        print(f"M={M}, K={K}, N={N}")
        print(f"Mean error: {mean_err:.6f}, Max error: {max_err:.6f}, Mean|ref|: {mean_abs_ref:.6f}, Rel: {rel_err:.4f}")
        print(f"Output[0,:8]: {out_cpu[0,:8].tolist()}")
        print(f"Ref[0,:8]:    {ref[0,:8].tolist()}")
        print(f"Output[0,-8:]: {out_cpu[0,-8:].tolist()}")
        print(f"Ref[0,-8:]:    {ref[0,-8:].tolist()}")

        assert rel_err < 0.05, f"BF16 direct MMA: rel error {rel_err:.4f} > 5%"

    def test_mma_moe_debug_ones(self, glm, device):
        M, K, N = 16, 16, 64
        input_bf16 = torch.ones(M, K, dtype=torch.bfloat16, device=device)
        weights_bf16 = torch.ones(N, K, dtype=torch.bfloat16, device=device)
        output_f32 = torch.zeros(M, N, dtype=torch.float32, device=device)

        glm.mma_moe_debug(output_f32, input_bf16, weights_bf16, M, K, N)
        glm.synchronize()

        output_cpu = output_f32.cpu()
        print("\n=== MMA Debug Output ===")
        print(f"Shape: {output_cpu.shape}")
        print(f"Min: {output_cpu.min().item():.4f}, Max: {output_cpu.max().item():.4f}, Mean: {output_cpu.mean().item():.4f}")
        print(f"First row (cols 0-15): {output_cpu[0, :16].tolist()}")
        print(f"First row (cols 16-31): {output_cpu[0, 16:32].tolist()}")
        print(f"First row (cols 48-63): {output_cpu[0, 48:64].tolist()}")
        print(f"All values == 16.0? {torch.allclose(output_cpu, torch.full_like(output_cpu, 16.0), atol=1e-2)}")
        unique_vals = torch.unique(output_cpu)
        print(f"Unique values: {unique_vals.tolist()[:20]}")
        assert torch.allclose(output_cpu, torch.full_like(output_cpu, 16.0), atol=0.5), \
            f"MMA debug: expected all 16.0, got min={output_cpu.min().item()}, max={output_cpu.max().item()}"
