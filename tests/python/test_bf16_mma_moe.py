"""Tests for glm_bf16_mul_mat_id_grouped_mma: grouped BF16 MoE with Tensor Core MMA.

Compares against PyTorch reference and against the scalar GEMV kernel
(glm_mul_mat_id_grouped).
"""

import pytest
import torch
import ctypes
from helpers import GpuPtrs, GlmOps


class TestBf16MulMatIdGroupedMMA:
    def _run_mma_moe(self, glm, device, batch, K, N, num_experts, topK):
        count = batch * topK
        torch.manual_seed(42)
        input_bf16 = torch.randn(batch, K, dtype=torch.bfloat16, device=device) * 0.5
        expert_ids = torch.randint(0, num_experts, (batch, topK), dtype=torch.int32, device=device)
        expert_ids_flat = expert_ids.reshape(-1)

        weights = [torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.5 for _ in range(num_experts)]
        weight_ptrs = GpuPtrs([w.data_ptr() for w in weights], device)

        out_mma = torch.empty(count, N, dtype=torch.bfloat16, device=device)

        ws_size = glm.lib.glm_mma_moe_workspace_size(count, N, K, num_experts)
        ws_ptr = glm.alloc(ws_size)

        glm.lib.glm_bf16_mul_mat_id_grouped_mma(
            glm.ctx,
            ctypes.c_void_p(out_mma.data_ptr()),
            ctypes.c_void_p(input_bf16.data_ptr()),
            ctypes.c_void_p(weight_ptrs.data_ptr),
            ctypes.c_void_p(expert_ids_flat.data_ptr()),
            topK, count, N, K, num_experts,
            ctypes.c_void_p(ws_ptr)
        )
        glm.synchronize()

        ref = torch.zeros(count, N, dtype=torch.float32)
        for i in range(count):
            bid = i // topK
            eid = expert_ids_flat[i].item()
            ref[i] = torch.nn.functional.linear(input_bf16[bid].cpu().float(), weights[eid].cpu().float())

        out_f32 = out_mma.cpu().float()
        mean_err = (out_f32 - ref).abs().mean().item()
        max_err = (out_f32 - ref).abs().max().item()
        mean_abs_ref = ref.abs().mean().item()

        assert mean_err < mean_abs_ref * 0.01, f"mean error {mean_err:.6f} > 1% of ref {mean_abs_ref:.6f}"
        assert max_err < mean_abs_ref * 0.1, f"max error {max_err:.6f} > 10% of ref {mean_abs_ref:.6f}"

        glm.free_buf(ws_ptr)

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

        weights = [torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.5 for _ in range(num_experts)]
        weight_ptrs = GpuPtrs([w.data_ptr() for w in weights], device)

        out_mma = torch.empty(count, N, dtype=torch.bfloat16, device=device)
        ws_size = glm.lib.glm_mma_moe_workspace_size(count, N, K, num_experts)
        ws_ptr = glm.alloc(ws_size)

        glm.lib.glm_bf16_mul_mat_id_grouped_mma(
            glm.ctx,
            ctypes.c_void_p(out_mma.data_ptr()),
            ctypes.c_void_p(input_bf16.data_ptr()),
            ctypes.c_void_p(weight_ptrs.data_ptr),
            ctypes.c_void_p(expert_ids_flat.data_ptr()),
            topK, count, N, K, num_experts,
            ctypes.c_void_p(ws_ptr)
        )
        glm.synchronize()

        ref = torch.zeros(count, N, dtype=torch.float32)
        for i in range(count):
            bid = i // topK
            ref[i] = torch.nn.functional.linear(input_bf16[bid].cpu().float(), weights[0].cpu().float())

        out_f32 = out_mma.cpu().float()
        mean_err = (out_f32 - ref).abs().mean().item()
        mean_abs_ref = ref.abs().mean().item()
        assert mean_err < mean_abs_ref * 0.01, f"mean error {mean_err:.6f} > 1% of ref {mean_abs_ref:.6f}"

        glm.free_buf(ws_ptr)

    def test_vs_gemv(self, glm, device):
        """Compare MMA output against scalar GEMV kernel."""
        batch = 4
        K = 128
        N = 64
        num_experts = 4
        topK = 2
        count = batch * topK
        torch.manual_seed(99)
        input_bf16 = torch.randn(batch, K, dtype=torch.bfloat16, device=device) * 0.5
        expert_ids = torch.randint(0, num_experts, (batch, topK), dtype=torch.int32, device=device)
        expert_ids_flat = expert_ids.reshape(-1)

        weights = [torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.5 for _ in range(num_experts)]
        weight_ptrs = GpuPtrs([w.data_ptr() for w in weights], device)

        out_mma = torch.empty(count, N, dtype=torch.bfloat16, device=device)
        out_gemv = torch.empty(count, N, dtype=torch.bfloat16, device=device)

        ws_size = glm.lib.glm_mma_moe_workspace_size(count, N, K, num_experts)
        ws_ptr = glm.alloc(ws_size)

        glm.lib.glm_bf16_mul_mat_id_grouped_mma(
            glm.ctx,
            ctypes.c_void_p(out_mma.data_ptr()),
            ctypes.c_void_p(input_bf16.data_ptr()),
            ctypes.c_void_p(weight_ptrs.data_ptr),
            ctypes.c_void_p(expert_ids_flat.data_ptr()),
            topK, count, N, K, num_experts,
            ctypes.c_void_p(ws_ptr)
        )

        ws_size_gemv = glm.grouped_moe_workspace_size(count, N, K, num_experts)
        ws_ptr_gemv = glm.alloc(ws_size_gemv)

        glm.mul_mat_id_grouped(
            out_gemv.data_ptr(),
            input_bf16.data_ptr(),
            weight_ptrs.data_ptr,
            expert_ids_flat.data_ptr(),
            topK, count, N, K, num_experts,
            ws_ptr_gemv
        )
        glm.synchronize()

        mma_f32 = out_mma.cpu().float()
        gemv_f32 = out_gemv.cpu().float()
        diff = (mma_f32 - gemv_f32).abs().max().item()
        assert diff < 0.01, f"MMA vs GEMV max diff {diff:.6f} > 0.01"

        glm.free_buf(ws_ptr)
        glm.free_buf(ws_ptr_gemv)
