"""Tests for glm_mul_mat_id: indexed matrix-vector multiplication for MoE expert dispatch.

For each (token, expert) pair, computes:
  output[i] = input[batch_ids[i]] @ weights[expert_ids[i]].T

Tests correctness against PyTorch reference using BF16 throughout.
"""

import pytest
import torch
from helpers import GpuBuffer, GpuPtrs


class TestMulMatId:
    def test_mul_mat_id_basic(self, glm, device):
        batch = 2
        K = 128
        N = 64
        num_experts = 4
        topK = 2
        count = batch * topK

        input_bf16 = torch.randn(batch, K, dtype=torch.bfloat16, device=device)
        weights = [torch.randn(N, K, dtype=torch.bfloat16, device=device) for _ in range(num_experts)]

        expert_ids = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32, device=device)
        batch_ids = torch.arange(batch).unsqueeze(1).expand(batch, topK).reshape(-1).int().to(device)
        expert_ids_flat = expert_ids.reshape(-1)

        output_bf16 = torch.empty(count, N, dtype=torch.bfloat16, device=device)

        weight_ptrs = GpuPtrs([w.data_ptr() for w in weights], device)

        glm.mul_mat_id(
            output_bf16.data_ptr(),
            input_bf16.data_ptr(),
            weight_ptrs.data_ptr,
            expert_ids_flat.data_ptr(),
            batch_ids.data_ptr(),
            count, N, K
        )

        ref = torch.zeros(count, N, dtype=torch.float32, device=device)
        for i in range(count):
            bid = batch_ids[i].item()
            eid = expert_ids_flat[i].item()
            ref[i] = input_bf16[bid].float() @ weights[eid].float().T

        torch.testing.assert_close(output_bf16.cpu().float(), ref.cpu(), atol=1e-2, rtol=1e-2)

    def test_mul_mat_id_single_token(self, glm, device):
        batch = 1
        K = 256
        N = 128
        num_experts = 8
        topK = 4
        count = batch * topK

        input_bf16 = torch.randn(batch, K, dtype=torch.bfloat16, device=device)
        weights = [torch.randn(N, K, dtype=torch.bfloat16, device=device) for _ in range(num_experts)]

        expert_ids = torch.tensor([[0, 2, 5, 7]], dtype=torch.int32, device=device)
        batch_ids = torch.zeros(count, dtype=torch.int32, device=device)
        expert_ids_flat = expert_ids.reshape(-1)

        output_bf16 = torch.empty(count, N, dtype=torch.bfloat16, device=device)

        weight_ptrs = GpuPtrs([w.data_ptr() for w in weights], device)

        glm.mul_mat_id(
            output_bf16.data_ptr(),
            input_bf16.data_ptr(),
            weight_ptrs.data_ptr,
            expert_ids_flat.data_ptr(),
            batch_ids.data_ptr(),
            count, N, K
        )

        ref = torch.zeros(count, N, dtype=torch.float32, device=device)
        for i in range(count):
            eid = expert_ids_flat[i].item()
            ref[i] = input_bf16[0].float() @ weights[eid].float().T

        torch.testing.assert_close(output_bf16.cpu().float(), ref.cpu(), atol=1e-2, rtol=1e-2)

    def test_mul_mat_id_large_dim(self, glm, device):
        batch = 1
        K = 512
        N = 2048
        num_experts = 4
        topK = 2
        count = batch * topK

        input_bf16 = torch.randn(batch, K, dtype=torch.bfloat16, device=device)
        weights = [torch.randn(N, K, dtype=torch.bfloat16, device=device) for _ in range(num_experts)]

        expert_ids = torch.tensor([[1, 3]], dtype=torch.int32, device=device)
        batch_ids = torch.zeros(count, dtype=torch.int32, device=device)
        expert_ids_flat = expert_ids.reshape(-1)

        output_bf16 = torch.empty(count, N, dtype=torch.bfloat16, device=device)

        weight_ptrs = GpuPtrs([w.data_ptr() for w in weights], device)

        glm.mul_mat_id(
            output_bf16.data_ptr(),
            input_bf16.data_ptr(),
            weight_ptrs.data_ptr,
            expert_ids_flat.data_ptr(),
            batch_ids.data_ptr(),
            count, N, K
        )

        ref = torch.zeros(count, N, dtype=torch.float32, device=device)
        for i in range(count):
            eid = expert_ids_flat[i].item()
            ref[i] = input_bf16[0].float() @ weights[eid].float().T

        torch.testing.assert_close(output_bf16.cpu().float(), ref.cpu(), atol=1e-2, rtol=1e-2)

    def test_mul_mat_id_multi_batch(self, glm, device):
        batch = 4
        K = 128
        N = 64
        num_experts = 8
        topK = 3
        count = batch * topK

        input_bf16 = torch.randn(batch, K, dtype=torch.bfloat16, device=device)
        weights = [torch.randn(N, K, dtype=torch.bfloat16, device=device) for _ in range(num_experts)]

        expert_ids = torch.randint(0, num_experts, (batch, topK), dtype=torch.int32, device=device)
        batch_ids = torch.arange(batch).unsqueeze(1).expand(batch, topK).reshape(-1).int().to(device)
        expert_ids_flat = expert_ids.reshape(-1)

        output_bf16 = torch.empty(count, N, dtype=torch.bfloat16, device=device)

        weight_ptrs = GpuPtrs([w.data_ptr() for w in weights], device)

        glm.mul_mat_id(
            output_bf16.data_ptr(),
            input_bf16.data_ptr(),
            weight_ptrs.data_ptr,
            expert_ids_flat.data_ptr(),
            batch_ids.data_ptr(),
            count, N, K
        )

        ref = torch.zeros(count, N, dtype=torch.float32, device=device)
        for i in range(count):
            bid = batch_ids[i].item()
            eid = expert_ids_flat[i].item()
            ref[i] = input_bf16[bid].float() @ weights[eid].float().T

        torch.testing.assert_close(output_bf16.cpu().float(), ref.cpu(), atol=1e-2, rtol=1e-2)

    def test_mul_mat_id_same_expert(self, glm, device):
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
        batch_ids = torch.arange(batch).unsqueeze(1).expand(batch, topK).reshape(-1).int().to(device)
        expert_ids_flat = expert_ids.reshape(-1)

        output_bf16 = torch.empty(count, N, dtype=torch.bfloat16, device=device)

        weight_ptrs = GpuPtrs([w.data_ptr() for w in weights], device)

        glm.mul_mat_id(
            output_bf16.data_ptr(),
            input_bf16.data_ptr(),
            weight_ptrs.data_ptr,
            expert_ids_flat.data_ptr(),
            batch_ids.data_ptr(),
            count, N, K
        )

        ref = torch.zeros(count, N, dtype=torch.float32, device=device)
        for i in range(count):
            bid = batch_ids[i].item()
            eid = expert_ids_flat[i].item()
            ref[i] = input_bf16[bid].float() @ weights[eid].float().T

        torch.testing.assert_close(output_bf16.cpu().float(), ref.cpu(), atol=1e-2, rtol=1e-2)

    def test_mul_mat_id_glm51_dims(self, glm, device):
        """Test with GLM-5.1 small model dimensions (hidden=128, moe_intermediate=256)."""
        batch = 1
        K = 128
        N = 256
        num_experts = 8
        topK = 4
        count = batch * topK

        input_bf16 = torch.randn(batch, K, dtype=torch.bfloat16, device=device)
        weights = [torch.randn(N, K, dtype=torch.bfloat16, device=device) for _ in range(num_experts)]

        expert_ids = torch.tensor([[0, 2, 5, 7]], dtype=torch.int32, device=device)
        batch_ids = torch.zeros(count, dtype=torch.int32, device=device)
        expert_ids_flat = expert_ids.reshape(-1)

        output_bf16 = torch.empty(count, N, dtype=torch.bfloat16, device=device)

        weight_ptrs = GpuPtrs([w.data_ptr() for w in weights], device)

        glm.mul_mat_id(
            output_bf16.data_ptr(),
            input_bf16.data_ptr(),
            weight_ptrs.data_ptr,
            expert_ids_flat.data_ptr(),
            batch_ids.data_ptr(),
            count, N, K
        )

        ref = torch.zeros(count, N, dtype=torch.float32, device=device)
        for i in range(count):
            eid = expert_ids_flat[i].item()
            ref[i] = input_bf16[0].float() @ weights[eid].float().T

        torch.testing.assert_close(output_bf16.cpu().float(), ref.cpu(), atol=1e-2, rtol=1e-2)

    def test_mul_mat_id_down_proj(self, glm, device):
        """Test down_proj dimension: [hidden_size, moe_intermediate_size] -> [hidden_size]."""
        batch = 1
        K = 256
        N = 128
        num_experts = 8
        topK = 4
        count = batch * topK

        input_bf16 = torch.randn(batch, K, dtype=torch.bfloat16, device=device)
        weights = [torch.randn(N, K, dtype=torch.bfloat16, device=device) for _ in range(num_experts)]

        expert_ids = torch.tensor([[1, 3, 5, 7]], dtype=torch.int32, device=device)
        batch_ids = torch.zeros(count, dtype=torch.int32, device=device)
        expert_ids_flat = expert_ids.reshape(-1)

        output_bf16 = torch.empty(count, N, dtype=torch.bfloat16, device=device)

        weight_ptrs = GpuPtrs([w.data_ptr() for w in weights], device)

        glm.mul_mat_id(
            output_bf16.data_ptr(),
            input_bf16.data_ptr(),
            weight_ptrs.data_ptr,
            expert_ids_flat.data_ptr(),
            batch_ids.data_ptr(),
            count, N, K
        )

        ref = torch.zeros(count, N, dtype=torch.float32, device=device)
        for i in range(count):
            bid = batch_ids[i].item()
            eid = expert_ids_flat[i].item()
            ref[i] = input_bf16[bid].float() @ weights[eid].float().T

        torch.testing.assert_close(output_bf16.cpu().float(), ref.cpu(), atol=1e-2, rtol=1e-2)
