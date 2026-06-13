"""Tests for glm_mul_mat_id_grouped: grouped MoE with sort-by-expert pipeline.

For each (token, expert) pair, computes:
  output[i] = input[i // top_k] @ weights[expert_ids[i]].T

The grouped path sorts entries by expert for weight reuse, then runs
batched GEMV per expert, then unscatters output.

Tests correctness against PyTorch reference and against the original
per-entry GEMV kernel.
"""

import pytest
import torch
import ctypes
from helpers import GpuBuffer, GpuPtrs, GlmOps


class TestMulMatIdGrouped:
    def _run_grouped(self, glm, device, batch, K, N, num_experts, topK, scale=1.0):
        count = batch * topK
        input_bf16 = torch.randn(batch, K, dtype=torch.bfloat16, device=device) * scale
        weights = [torch.randn(N, K, dtype=torch.bfloat16, device=device) * scale for _ in range(num_experts)]
        expert_ids = torch.randint(0, num_experts, (batch, topK), dtype=torch.int32, device=device)
        expert_ids_flat = expert_ids.reshape(-1)

        output_bf16 = torch.empty(count, N, dtype=torch.bfloat16, device=device)
        output_grouped = torch.empty(count, N, dtype=torch.bfloat16, device=device)

        weight_ptrs = GpuPtrs([w.data_ptr() for w in weights], device)

        # Original per-entry GEMV
        glm.mul_mat_id(
            output_bf16.data_ptr(),
            input_bf16.data_ptr(),
            weight_ptrs.data_ptr,
            expert_ids_flat.data_ptr(),
            topK, count, N, K
        )

        # Grouped path
        ws_size = glm.grouped_moe_workspace_size(count, N, K, num_experts)
        ws_ptr = glm.alloc(ws_size)

        glm.mul_mat_id_grouped(
            output_grouped.data_ptr(),
            input_bf16.data_ptr(),
            weight_ptrs.data_ptr,
            expert_ids_flat.data_ptr(),
            topK, count, N, K, num_experts,
            ws_ptr
        )
        glm.synchronize()

        # PyTorch reference
        ref = torch.zeros(count, N, dtype=torch.float32, device=device)
        for i in range(count):
            bid = i // topK
            eid = expert_ids_flat[i].item()
            ref[i] = input_bf16[bid].float() @ weights[eid].float().T

        # Check grouped path against reference
        torch.testing.assert_close(output_grouped.cpu().float(), ref.cpu(), atol=2e-2, rtol=2e-2)

        # Check grouped path matches original per-entry GEMV
        torch.testing.assert_close(output_grouped.cpu().float(), output_bf16.cpu().float(), atol=1e-3, rtol=1e-3)

        glm.free_buf(ws_ptr)

    def test_grouped_basic(self, glm, device):
        self._run_grouped(glm, device, batch=2, K=128, N=64, num_experts=4, topK=2)

    def test_grouped_single_token(self, glm, device):
        self._run_grouped(glm, device, batch=1, K=256, N=128, num_experts=8, topK=4)

    def test_grouped_multi_batch(self, glm, device):
        self._run_grouped(glm, device, batch=4, K=128, N=64, num_experts=8, topK=3)

    def test_grouped_same_expert(self, glm, device):
        """Multiple tokens selecting the same expert."""
        batch = 3
        K = 128
        N = 64
        num_experts = 4
        topK = 2
        count = batch * topK

        input_bf16 = torch.randn(batch, K, dtype=torch.bfloat16, device=device)
        weights = [torch.randn(N, K, dtype=torch.bfloat16, device=device) for _ in range(num_experts)]
        expert_ids = torch.tensor([[0, 1], [0, 2], [0, 3]], dtype=torch.int32, device=device)
        expert_ids_flat = expert_ids.reshape(-1)

        output_grouped = torch.empty(count, N, dtype=torch.bfloat16, device=device)

        weight_ptrs = GpuPtrs([w.data_ptr() for w in weights], device)
        ws_size = glm.grouped_moe_workspace_size(count, N, K, num_experts)
        ws_ptr = glm.alloc(ws_size)

        glm.mul_mat_id_grouped(
            output_grouped.data_ptr(),
            input_bf16.data_ptr(),
            weight_ptrs.data_ptr,
            expert_ids_flat.data_ptr(),
            topK, count, N, K, num_experts,
            ws_ptr
        )
        glm.synchronize()

        ref = torch.zeros(count, N, dtype=torch.float32, device=device)
        for i in range(count):
            bid = i // topK
            eid = expert_ids_flat[i].item()
            ref[i] = input_bf16[bid].float() @ weights[eid].float().T

        torch.testing.assert_close(output_grouped.cpu().float(), ref.cpu(), atol=2e-2, rtol=2e-2)
        glm.free_buf(ws_ptr)

    def test_grouped_glm51_gate_dims(self, glm, device):
        """Test with GLM-5.1 gate/up dimensions (N=1408, K=4096)."""
        self._run_grouped(glm, device, batch=2, K=4096, N=1408, num_experts=8, topK=4)

    def test_grouped_glm51_down_dims(self, glm, device):
        """Test with GLM-5.1 down_proj dimensions (N=4096, K=1408)."""
        self._run_grouped(glm, device, batch=2, K=1408, N=4096, num_experts=8, topK=4)

    def test_grouped_large_batch(self, glm, device):
        """Test with larger batch to exercise multi-entry-per-expert path."""
        self._run_grouped(glm, device, batch=8, K=256, N=512, num_experts=16, topK=4, scale=0.01)

    def test_grouped_all_same_expert(self, glm, device):
        """All entries select the same expert."""
        batch = 4
        K = 128
        N = 64
        num_experts = 4
        topK = 2
        count = batch * topK

        input_bf16 = torch.randn(batch, K, dtype=torch.bfloat16, device=device)
        weights = [torch.randn(N, K, dtype=torch.bfloat16, device=device) for _ in range(num_experts)]
        expert_ids = torch.zeros(batch, topK, dtype=torch.int32, device=device)
        expert_ids_flat = expert_ids.reshape(-1)

        output_grouped = torch.empty(count, N, dtype=torch.bfloat16, device=device)

        weight_ptrs = GpuPtrs([w.data_ptr() for w in weights], device)
        ws_size = glm.grouped_moe_workspace_size(count, N, K, num_experts)
        ws_ptr = glm.alloc(ws_size)

        glm.mul_mat_id_grouped(
            output_grouped.data_ptr(),
            input_bf16.data_ptr(),
            weight_ptrs.data_ptr,
            expert_ids_flat.data_ptr(),
            topK, count, N, K, num_experts,
            ws_ptr
        )
        glm.synchronize()

        ref = torch.zeros(count, N, dtype=torch.float32, device=device)
        for i in range(count):
            bid = i // topK
            ref[i] = input_bf16[bid].float() @ weights[0].float().T

        torch.testing.assert_close(output_grouped.cpu().float(), ref.cpu(), atol=2e-2, rtol=2e-2)
        glm.free_buf(ws_ptr)

    def test_grouped_small_N(self, glm, device):
        """Test with N < 1024 (split-K path)."""
        self._run_grouped(glm, device, batch=2, K=256, N=512, num_experts=4, topK=2)
