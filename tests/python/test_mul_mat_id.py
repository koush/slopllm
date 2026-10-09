"""Tests for glm_mul_mat_id: indexed matrix-vector multiplication for MoE expert dispatch.

For each (token, expert) pair, computes:
  output[i] = input[i // top_k] @ weights[expert_ids[i]].T

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
        expert_ids_flat = expert_ids.reshape(-1)

        output_bf16 = torch.empty(count, N, dtype=torch.bfloat16, device=device)

        weight_ptrs = GpuPtrs([w.data_ptr() for w in weights], device)

        glm.mul_mat_id(
            output_bf16.data_ptr(),
            input_bf16.data_ptr(),
            weight_ptrs.data_ptr,
            expert_ids_flat.data_ptr(),
            topK, count, N, K
        )

        ref = torch.zeros(count, N, dtype=torch.float32, device=device)
        for i in range(count):
            bid = i // topK
            eid = expert_ids_flat[i].item()
            ref[i] = input_bf16[bid].float() @ weights[eid].float().T

        torch.testing.assert_close(output_bf16.cpu().float(), ref.cpu(), atol=2e-2, rtol=2e-2)

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
        expert_ids_flat = expert_ids.reshape(-1)

        output_bf16 = torch.empty(count, N, dtype=torch.bfloat16, device=device)

        weight_ptrs = GpuPtrs([w.data_ptr() for w in weights], device)

        glm.mul_mat_id(
            output_bf16.data_ptr(),
            input_bf16.data_ptr(),
            weight_ptrs.data_ptr,
            expert_ids_flat.data_ptr(),
            topK, count, N, K
        )

        ref = torch.zeros(count, N, dtype=torch.float32, device=device)
        for i in range(count):
            eid = expert_ids_flat[i].item()
            ref[i] = input_bf16[0].float() @ weights[eid].float().T

        torch.testing.assert_close(output_bf16.cpu().float(), ref.cpu(), atol=2e-2, rtol=2e-2)

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
        expert_ids_flat = expert_ids.reshape(-1)

        output_bf16 = torch.empty(count, N, dtype=torch.bfloat16, device=device)

        weight_ptrs = GpuPtrs([w.data_ptr() for w in weights], device)

        glm.mul_mat_id(
            output_bf16.data_ptr(),
            input_bf16.data_ptr(),
            weight_ptrs.data_ptr,
            expert_ids_flat.data_ptr(),
            topK, count, N, K
        )

        ref = torch.zeros(count, N, dtype=torch.float32, device=device)
        for i in range(count):
            eid = expert_ids_flat[i].item()
            ref[i] = input_bf16[0].float() @ weights[eid].float().T

        torch.testing.assert_close(output_bf16.cpu().float(), ref.cpu(), atol=2e-2, rtol=2e-2)

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
        expert_ids_flat = expert_ids.reshape(-1)

        output_bf16 = torch.empty(count, N, dtype=torch.bfloat16, device=device)

        weight_ptrs = GpuPtrs([w.data_ptr() for w in weights], device)

        glm.mul_mat_id(
            output_bf16.data_ptr(),
            input_bf16.data_ptr(),
            weight_ptrs.data_ptr,
            expert_ids_flat.data_ptr(),
            topK, count, N, K
        )

        ref = torch.zeros(count, N, dtype=torch.float32, device=device)
        for i in range(count):
            bid = i // topK
            eid = expert_ids_flat[i].item()
            ref[i] = input_bf16[bid].float() @ weights[eid].float().T

        torch.testing.assert_close(output_bf16.cpu().float(), ref.cpu(), atol=2e-2, rtol=2e-2)

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
        expert_ids_flat = expert_ids.reshape(-1)

        output_bf16 = torch.empty(count, N, dtype=torch.bfloat16, device=device)

        weight_ptrs = GpuPtrs([w.data_ptr() for w in weights], device)

        glm.mul_mat_id(
            output_bf16.data_ptr(),
            input_bf16.data_ptr(),
            weight_ptrs.data_ptr,
            expert_ids_flat.data_ptr(),
            topK, count, N, K
        )

        ref = torch.zeros(count, N, dtype=torch.float32, device=device)
        for i in range(count):
            bid = i // topK
            eid = expert_ids_flat[i].item()
            ref[i] = input_bf16[bid].float() @ weights[eid].float().T

        torch.testing.assert_close(output_bf16.cpu().float(), ref.cpu(), atol=2e-2, rtol=2e-2)

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
        expert_ids_flat = expert_ids.reshape(-1)

        output_bf16 = torch.empty(count, N, dtype=torch.bfloat16, device=device)

        weight_ptrs = GpuPtrs([w.data_ptr() for w in weights], device)

        glm.mul_mat_id(
            output_bf16.data_ptr(),
            input_bf16.data_ptr(),
            weight_ptrs.data_ptr,
            expert_ids_flat.data_ptr(),
            topK, count, N, K
        )

        ref = torch.zeros(count, N, dtype=torch.float32, device=device)
        for i in range(count):
            eid = expert_ids_flat[i].item()
            ref[i] = input_bf16[0].float() @ weights[eid].float().T

        torch.testing.assert_close(output_bf16.cpu().float(), ref.cpu(), atol=2e-2, rtol=2e-2)

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
        expert_ids_flat = expert_ids.reshape(-1)

        output_bf16 = torch.empty(count, N, dtype=torch.bfloat16, device=device)

        weight_ptrs = GpuPtrs([w.data_ptr() for w in weights], device)

        glm.mul_mat_id(
            output_bf16.data_ptr(),
            input_bf16.data_ptr(),
            weight_ptrs.data_ptr,
            expert_ids_flat.data_ptr(),
            topK, count, N, K
        )

        ref = torch.zeros(count, N, dtype=torch.float32, device=device)
        for i in range(count):
            bid = i // topK
            eid = expert_ids_flat[i].item()
            ref[i] = input_bf16[bid].float() @ weights[eid].float().T

        torch.testing.assert_close(output_bf16.cpu().float(), ref.cpu(), atol=2e-2, rtol=2e-2)

    def test_mul_mat_id_large_batch_small_N(self, glm, device):
        batch = 15
        K = 768
        N = 256
        num_experts = 8
        topK = 8
        count = batch * topK

        input_bf16 = torch.randn(batch, K, dtype=torch.bfloat16, device=device)
        weights = [torch.randn(N, K, dtype=torch.bfloat16, device=device) for _ in range(num_experts)]

        expert_ids = torch.randint(0, num_experts, (batch, topK), dtype=torch.int32, device=device)
        expert_ids_flat = expert_ids.reshape(-1)

        output_bf16 = torch.empty(count, N, dtype=torch.bfloat16, device=device)

        weight_ptrs = GpuPtrs([w.data_ptr() for w in weights], device)

        glm.mul_mat_id(
            output_bf16.data_ptr(),
            input_bf16.data_ptr(),
            weight_ptrs.data_ptr,
            expert_ids_flat.data_ptr(),
            topK, count, N, K
        )

        ref = torch.zeros(count, N, dtype=torch.float32, device=device)
        for i in range(count):
            bid = i // topK
            eid = expert_ids_flat[i].item()
            ref[i] = input_bf16[bid].float() @ weights[eid].float().T

        torch.testing.assert_close(output_bf16.cpu().float(), ref.cpu(), atol=2e-2, rtol=2e-2)

    def test_mul_mat_id_large_batch_down_proj(self, glm, device):
        batch = 15
        K = 256
        N = 768
        num_experts = 8
        topK = 8
        count = batch * topK

        input_bf16 = torch.randn(batch, K, dtype=torch.bfloat16, device=device)
        weights = [torch.randn(N, K, dtype=torch.bfloat16, device=device) for _ in range(num_experts)]

        expert_ids = torch.randint(0, num_experts, (batch, topK), dtype=torch.int32, device=device)
        expert_ids_flat = expert_ids.reshape(-1)

        output_bf16 = torch.empty(count, N, dtype=torch.bfloat16, device=device)

        weight_ptrs = GpuPtrs([w.data_ptr() for w in weights], device)

        glm.mul_mat_id(
            output_bf16.data_ptr(),
            input_bf16.data_ptr(),
            weight_ptrs.data_ptr,
            expert_ids_flat.data_ptr(),
            topK, count, N, K
        )

        ref = torch.zeros(count, N, dtype=torch.float32, device=device)
        for i in range(count):
            bid = i // topK
            eid = expert_ids_flat[i].item()
            ref[i] = input_bf16[bid].float() @ weights[eid].float().T

        torch.testing.assert_close(output_bf16.cpu().float(), ref.cpu(), atol=2e-2, rtol=2e-2)

class TestScatterAddRows:
    @pytest.mark.parametrize("rows,dim,topk,input_offset,output_offset", [
        (128, 6144, 8, 0, 0),
        (129, 1028, 8, 0, 0),  # Partial final vectorized block.
        (128, 1027, 8, 0, 0),  # Odd row stride uses the scalar path.
        (128, 1028, 8, 1, 0),  # Unaligned input view.
        (128, 1028, 8, 0, 1),  # Unaligned output view.
        (127, 1028, 8, 0, 0),  # Below the prefill dispatch threshold.
        (128, 1028, 4, 0, 0),
    ])
    def test_scatter_add_rows_prefill(self, glm, device, rows, dim, topk, input_offset, output_offset):
        torch.manual_seed(42)
        storage = torch.randn(rows * topk * dim + input_offset, dtype=torch.bfloat16, device=device)
        inputs = storage[input_offset:].view(rows, topk, dim)
        scales = torch.randn(rows, topk, dtype=torch.bfloat16, device=device)
        output_storage = torch.full((rows * dim + output_offset,), 99, dtype=torch.bfloat16, device=device)
        output = output_storage[output_offset:].view(rows, dim)

        expected = torch.zeros(rows, dim, dtype=torch.float32, device=device)
        for j in range(topk):
            # A product of two finite BF16 values in this range is exact in
            # FP32; sequential FP32 additions match the kernel's FMA order.
            expected += inputs[:, j].float() * scales[:, j, None].float()
        expected = expected.to(torch.bfloat16)
        glm.scatter_add_rows(output.data_ptr(), inputs.data_ptr(), scales.data_ptr(), topk, dim, rows, 0)
        glm.synchronize()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        if output_offset:
            assert output_storage[0].item() == 99

    def test_scatter_add_rows_basic(self, glm, device):
        rows_out = 2
        dim = 64
        topK = 2
        count = rows_out * topK

        out = torch.zeros(rows_out, dim, dtype=torch.bfloat16, device=device)
        input_bf16 = torch.randn(count, dim, dtype=torch.bfloat16, device=device)
        scales = torch.randn(count, dtype=torch.bfloat16, device=device)

        glm.scatter_add_rows(
            out.data_ptr(),
            input_bf16.data_ptr(),
            scales.data_ptr(),
            topK, dim, rows_out,
            0  # workspace (unused)
        )

        ref = torch.zeros(rows_out, dim, dtype=torch.float32, device=device)
        for i in range(count):
            bid = i // topK
            ref[bid] += scales[i].float() * input_bf16[i].float()

        torch.testing.assert_close(out.cpu().float(), ref.cpu(), atol=1e-2, rtol=1e-2)

    def test_scatter_add_rows_single_batch(self, glm, device):
        rows_out = 1
        dim = 128
        topK = 8
        count = rows_out * topK

        out = torch.zeros(rows_out, dim, dtype=torch.bfloat16, device=device)
        input_bf16 = torch.randn(count, dim, dtype=torch.bfloat16, device=device)
        scales = torch.randn(count, dtype=torch.bfloat16, device=device)

        glm.scatter_add_rows(
            out.data_ptr(),
            input_bf16.data_ptr(),
            scales.data_ptr(),
            topK, dim, rows_out,
            0  # workspace (unused)
        )

        ref = torch.zeros(rows_out, dim, dtype=torch.float32, device=device)
        for i in range(count):
            ref[0] += scales[i].float() * input_bf16[i].float()

        torch.testing.assert_close(out.cpu().float(), ref.cpu(), atol=1e-2, rtol=1e-2)

    def test_scatter_add_rows_moe_routing(self, glm, device):
        BS = 4
        topK = 4
        hs = 128
        count = BS * topK

        out = torch.zeros(BS, hs, dtype=torch.bfloat16, device=device)
        input_bf16 = torch.randn(count, hs, dtype=torch.bfloat16, device=device)
        scales = torch.rand(count, dtype=torch.bfloat16, device=device) + 0.1

        glm.scatter_add_rows(
            out.data_ptr(),
            input_bf16.data_ptr(),
            scales.data_ptr(),
            topK, hs, BS,
            0  # workspace (unused)
        )

        ref = torch.zeros(BS, hs, dtype=torch.float32, device=device)
        for i in range(count):
            bid = i // topK
            ref[bid] += scales[i].float() * input_bf16[i].float()

        torch.testing.assert_close(out.cpu().float(), ref.cpu(), atol=1e-2, rtol=1e-2)
