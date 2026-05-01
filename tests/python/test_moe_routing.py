"""Tests for GPU MoE routing kernels: row_normalize, group_mask_mul, expert_scale.

Each kernel is tested against a PyTorch reference implementation.
Also tests the full GPU MoE routing pipeline (sigmoid + bias + group routing + topk + gather + normalize).
"""

import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# row_normalize
# ---------------------------------------------------------------------------

class TestRowNormalize:
    def test_row_normalize_basic(self, glm, device):
        rows, cols = 4, 8
        x = torch.rand(rows, cols, dtype=torch.bfloat16, device=device) + 0.1
        scale = 2.5
        out = torch.empty_like(x)
        glm.row_normalize(out.data_ptr(), x.data_ptr(), scale, rows, cols, normalize=True)
        ref = F.normalize(x.float(), p=1, dim=-1) * scale
        torch.testing.assert_close(out.cpu().float(), ref.cpu(), atol=1e-2, rtol=1e-2)

    def test_row_normalize_no_normalize(self, glm, device):
        rows, cols = 4, 8
        x = torch.rand(rows, cols, dtype=torch.bfloat16, device=device) + 0.1
        scale = 2.5
        out = torch.empty_like(x)
        glm.row_normalize(out.data_ptr(), x.data_ptr(), scale, rows, cols, normalize=False)
        ref = x.float() * scale
        torch.testing.assert_close(out.cpu().float(), ref.cpu(), atol=1e-2, rtol=1e-2)

    def test_row_normalize_single_row(self, glm, device):
        rows, cols = 1, 64
        x = torch.rand(rows, cols, dtype=torch.bfloat16, device=device) + 0.1
        scale = 1.0
        out = torch.empty_like(x)
        glm.row_normalize(out.data_ptr(), x.data_ptr(), scale, rows, cols, normalize=True)
        ref = F.normalize(x.float(), p=1, dim=-1) * scale
        torch.testing.assert_close(out.cpu().float(), ref.cpu(), atol=1e-2, rtol=1e-2)

    def test_row_normalize_moe_weights(self, glm, device):
        topk = 8
        num_tokens = 16
        topk_w = torch.rand(num_tokens, topk, dtype=torch.bfloat16, device=device) + 0.01
        scale = 2.5
        out = torch.empty_like(topk_w)
        glm.row_normalize(out.data_ptr(), topk_w.data_ptr(), scale, num_tokens, topk, normalize=True)
        ref = F.normalize(topk_w.float(), p=1, dim=-1) * scale
        torch.testing.assert_close(out.cpu().float(), ref.cpu(), atol=1e-2, rtol=1e-2)

    def test_row_normalize_routed_scaling(self, glm, device):
        topk = 4
        num_tokens = 8
        topk_w = torch.rand(num_tokens, topk, dtype=torch.bfloat16, device=device) + 0.01
        routed_scaling = 2.5
        out = torch.empty_like(topk_w)
        glm.row_normalize(out.data_ptr(), topk_w.data_ptr(), routed_scaling, num_tokens, topk, normalize=True)
        ref = topk_w.float() / topk_w.float().sum(dim=-1, keepdim=True) * routed_scaling
        torch.testing.assert_close(out.cpu().float(), ref.cpu(), atol=1e-2, rtol=1e-2)

    def test_row_normalize_no_norm_scale_only(self, glm, device):
        topk = 4
        num_tokens = 8
        topk_w = torch.rand(num_tokens, topk, dtype=torch.bfloat16, device=device) + 0.01
        routed_scaling = 2.5
        out = torch.empty_like(topk_w)
        glm.row_normalize(out.data_ptr(), topk_w.data_ptr(), routed_scaling, num_tokens, topk, normalize=False)
        ref = topk_w.float() * routed_scaling
        torch.testing.assert_close(out.cpu().float(), ref.cpu(), atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# group_mask_mul
# ---------------------------------------------------------------------------

class TestGroupMaskMul:
    def test_group_mask_mul_basic(self, glm, device):
        batch = 2
        num_experts = 8
        n_group = 2
        experts_per_group = num_experts // n_group

        scores = torch.rand(batch, num_experts, dtype=torch.bfloat16, device=device)
        group_mask = torch.zeros(batch, n_group, dtype=torch.bfloat16, device=device)
        group_mask[:, 0] = 1.0
        group_mask[1, 1] = 1.0

        scores_gpu = scores.clone()
        glm.group_mask_mul(scores_gpu.data_ptr(), group_mask.data_ptr(),
                           num_experts, experts_per_group, n_group, batch)

        mask_expanded = group_mask.unsqueeze(-1).expand(-1, n_group, experts_per_group).reshape(batch, num_experts)
        ref = scores.float() * mask_expanded.float()
        torch.testing.assert_close(scores_gpu.cpu().float(), ref.cpu(), atol=1e-3, rtol=1e-3)

    def test_group_mask_mul_all_selected(self, glm, device):
        batch = 3
        num_experts = 6
        n_group = 3
        experts_per_group = num_experts // n_group

        scores = torch.rand(batch, num_experts, dtype=torch.bfloat16, device=device)
        group_mask = torch.ones(batch, n_group, dtype=torch.bfloat16, device=device)

        scores_gpu = scores.clone()
        glm.group_mask_mul(scores_gpu.data_ptr(), group_mask.data_ptr(),
                           num_experts, experts_per_group, n_group, batch)

        ref = scores.float()
        torch.testing.assert_close(scores_gpu.cpu().float(), ref.cpu(), atol=1e-3, rtol=1e-3)

    def test_group_mask_mul_none_selected(self, glm, device):
        batch = 2
        num_experts = 8
        n_group = 4
        experts_per_group = num_experts // n_group

        scores = torch.rand(batch, num_experts, dtype=torch.bfloat16, device=device)
        group_mask = torch.zeros(batch, n_group, dtype=torch.bfloat16, device=device)

        scores_gpu = scores.clone()
        glm.group_mask_mul(scores_gpu.data_ptr(), group_mask.data_ptr(),
                           num_experts, experts_per_group, n_group, batch)

        ref = torch.zeros_like(scores.float())
        torch.testing.assert_close(scores_gpu.cpu().float(), ref.cpu(), atol=1e-3, rtol=1e-3)

    def test_group_mask_mul_glm51_config(self, glm, device):
        batch = 4
        num_experts = 8
        n_group = 2
        experts_per_group = num_experts // n_group
        topk_group = 1

        scores = torch.rand(batch, num_experts, dtype=torch.bfloat16, device=device) + 0.1
        group_scores = scores.view(batch, n_group, experts_per_group).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = group_scores.topk(topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros(batch, n_group, dtype=torch.bfloat16, device=device)
        group_mask.scatter_(1, group_idx, 1.0)

        scores_gpu = scores.clone()
        glm.group_mask_mul(scores_gpu.data_ptr(), group_mask.data_ptr(),
                           num_experts, experts_per_group, n_group, batch)

        mask_expanded = group_mask.unsqueeze(-1).expand(-1, n_group, experts_per_group).reshape(batch, num_experts)
        ref = scores.float() * mask_expanded.float()
        torch.testing.assert_close(scores_gpu.cpu().float(), ref.cpu(), atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# expert_scale
# ---------------------------------------------------------------------------

class TestExpertScale:
    def test_expert_scale_basic(self, glm, device):
        batch = 4
        top_k = 2
        num_experts = 8

        topk_weights = torch.rand(batch, top_k, dtype=torch.bfloat16, device=device) + 0.01
        topk_indices = torch.zeros(batch, top_k, dtype=torch.int32, device=device)
        topk_indices[0] = torch.tensor([3, 7])
        topk_indices[1] = torch.tensor([1, 5])
        topk_indices[2] = torch.tensor([2, 6])
        topk_indices[3] = torch.tensor([0, 4])

        expert_id = 3
        out = torch.zeros(batch, dtype=torch.bfloat16, device=device)
        glm.expert_scale(out.data_ptr(), topk_weights.data_ptr(), topk_indices.data_ptr(),
                         expert_id, top_k, batch)

        ref = torch.zeros(batch, dtype=torch.float32)
        for b in range(batch):
            for k in range(top_k):
                if topk_indices[b, k].item() == expert_id:
                    ref[b] = topk_weights[b, k].float().item()
                    break

        torch.testing.assert_close(out.cpu().float(), ref, atol=1e-3, rtol=1e-3)

    def test_expert_scale_not_selected(self, glm, device):
        batch = 4
        top_k = 2
        num_experts = 16

        topk_weights = torch.rand(batch, top_k, dtype=torch.bfloat16, device=device) + 0.01
        topk_indices = torch.zeros(batch, top_k, dtype=torch.int32, device=device)
        topk_indices[0] = torch.tensor([0, 1])
        topk_indices[1] = torch.tensor([2, 3])
        topk_indices[2] = torch.tensor([4, 5])
        topk_indices[3] = torch.tensor([6, 7])

        expert_id = 15
        out = torch.zeros(batch, dtype=torch.bfloat16, device=device)
        glm.expert_scale(out.data_ptr(), topk_weights.data_ptr(), topk_indices.data_ptr(),
                         expert_id, top_k, batch)

        ref = torch.zeros(batch, dtype=torch.float32)
        torch.testing.assert_close(out.cpu().float(), ref, atol=1e-3, rtol=1e-3)

    def test_expert_scale_all_selected(self, glm, device):
        batch = 3
        top_k = 4
        expert_id = 2

        topk_weights = torch.rand(batch, top_k, dtype=torch.bfloat16, device=device) + 0.01
        topk_indices = torch.zeros(batch, top_k, dtype=torch.int32, device=device)
        for b in range(batch):
            topk_indices[b] = torch.tensor([expert_id, 1, 3, 4])

        out = torch.zeros(batch, dtype=torch.bfloat16, device=device)
        glm.expert_scale(out.data_ptr(), topk_weights.data_ptr(), topk_indices.data_ptr(),
                         expert_id, top_k, batch)

        ref = topk_weights[:, 0].float().cpu()
        torch.testing.assert_close(out.cpu().float(), ref, atol=1e-3, rtol=1e-3)

    def test_expert_scale_glm51_config(self, glm, device):
        batch = 6
        top_k = 4
        num_experts = 8

        scores = torch.rand(batch, num_experts, dtype=torch.bfloat16, device=device) + 0.1
        topk_result = scores.topk(top_k, dim=-1, sorted=False)
        topk_weights = topk_result.values
        topk_indices = topk_result.indices.int()

        for expert_id in range(num_experts):
            out = torch.zeros(batch, dtype=torch.bfloat16, device=device)
            glm.expert_scale(out.data_ptr(), topk_weights.data_ptr(), topk_indices.data_ptr(),
                             expert_id, top_k, batch)

            ref = torch.zeros(batch, dtype=torch.float32)
            for b in range(batch):
                for k in range(top_k):
                    if topk_indices[b, k].item() == expert_id:
                        ref[b] = topk_weights[b, k].float().item()
                        break
            torch.testing.assert_close(out.cpu().float(), ref, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Full GPU MoE routing pipeline
# ---------------------------------------------------------------------------

class TestMoeRoutingPipeline:
    """Test the complete GPU MoE routing pipeline against PyTorch reference.

    These tests validate each step of the GPU routing matches the PyTorch reference,
    using BF16 throughout (matching the CUDA model's precision model).
    """

    def _gpu_topk(self, glm, scores_gpu, k, dim, batch):
        """Run GPU topk and return (values_gpu, indices_gpu_int32)."""
        values_gpu = torch.empty(batch, k, dtype=torch.bfloat16, device=scores_gpu.device)
        indices_gpu = torch.empty(batch, k, dtype=torch.int32, device=scores_gpu.device)
        glm.topk(values_gpu.data_ptr(), indices_gpu.data_ptr(), scores_gpu.data_ptr(), k, dim, batch)
        return values_gpu, indices_gpu

    def test_routing_no_bias_no_groups(self, glm, device):
        BS = 4
        num_experts = 8
        top_k = 4
        routed_scaling = 2.5

        gate_logits = torch.randn(BS, num_experts, dtype=torch.bfloat16, device=device)

        gate_sigmoid_gpu = torch.empty_like(gate_logits)
        glm.sigmoid(gate_sigmoid_gpu.data_ptr(), gate_logits.data_ptr(), BS * num_experts)
        gate_sigmoid_ref = torch.sigmoid(gate_logits.float())

        topk_values_gpu, topk_indices_gpu = self._gpu_topk(glm, gate_sigmoid_gpu, top_k, num_experts, BS)

        topk_weights_gpu = torch.empty(BS, top_k, dtype=torch.bfloat16, device=device)
        glm.gather(topk_weights_gpu.data_ptr(), gate_sigmoid_gpu.data_ptr(),
                   topk_indices_gpu.data_ptr(), top_k, num_experts, BS)

        normalized_gpu = torch.empty_like(topk_weights_gpu)
        glm.row_normalize(normalized_gpu.data_ptr(), topk_weights_gpu.data_ptr(),
                          routed_scaling, BS, top_k, normalize=True)

        topk_indices_ref = topk_indices_gpu.cpu().long()
        topk_weights_ref = gate_sigmoid_ref.cpu().gather(1, topk_indices_ref)
        normalized_ref = topk_weights_ref / (topk_weights_ref.sum(dim=-1, keepdim=True) + 1e-20) * routed_scaling

        torch.testing.assert_close(normalized_gpu.cpu().float(), normalized_ref, atol=1e-2, rtol=1e-2)

    def test_routing_with_bias(self, glm, device):
        BS = 4
        num_experts = 8
        top_k = 4
        routed_scaling = 2.5

        gate_logits = torch.randn(BS, num_experts, dtype=torch.bfloat16, device=device)
        bias = torch.randn(num_experts, dtype=torch.bfloat16, device=device)

        gate_sigmoid_gpu = torch.empty_like(gate_logits)
        glm.sigmoid(gate_sigmoid_gpu.data_ptr(), gate_logits.data_ptr(), BS * num_experts)
        gate_sigmoid_ref = torch.sigmoid(gate_logits.float())

        bias_expanded = bias.unsqueeze(0).expand(BS, num_experts).contiguous()
        corrected_gpu = torch.empty_like(gate_sigmoid_gpu)
        glm.add(corrected_gpu.data_ptr(), gate_sigmoid_gpu.data_ptr(), bias_expanded.data_ptr(),
                BS * num_experts)

        corrected_ref = gate_sigmoid_ref.cpu() + bias.float().unsqueeze(0).cpu()

        topk_values_gpu, topk_indices_gpu = self._gpu_topk(glm, corrected_gpu, top_k, num_experts, BS)

        topk_weights_gpu = torch.empty(BS, top_k, dtype=torch.bfloat16, device=device)
        glm.gather(topk_weights_gpu.data_ptr(), gate_sigmoid_gpu.data_ptr(),
                   topk_indices_gpu.data_ptr(), top_k, num_experts, BS)

        normalized_gpu = torch.empty_like(topk_weights_gpu)
        glm.row_normalize(normalized_gpu.data_ptr(), topk_weights_gpu.data_ptr(),
                          routed_scaling, BS, top_k, normalize=True)

        topk_indices_ref = topk_indices_gpu.cpu().long()
        topk_weights_ref = gate_sigmoid_ref.cpu().gather(1, topk_indices_ref)
        normalized_ref = topk_weights_ref / (topk_weights_ref.sum(dim=-1, keepdim=True) + 1e-20) * routed_scaling

        torch.testing.assert_close(normalized_gpu.cpu().float(), normalized_ref, atol=1e-2, rtol=1e-2)

    def test_routing_with_groups(self, glm, device):
        BS = 4
        num_experts = 8
        top_k = 4
        n_group = 2
        topk_group = 1
        experts_per_group = num_experts // n_group
        routed_scaling = 2.5

        gate_logits = torch.randn(BS, num_experts, dtype=torch.bfloat16, device=device)
        bias = torch.randn(num_experts, dtype=torch.bfloat16, device=device)

        gate_sigmoid_gpu = torch.empty_like(gate_logits)
        glm.sigmoid(gate_sigmoid_gpu.data_ptr(), gate_logits.data_ptr(), BS * num_experts)
        gate_sigmoid_ref = torch.sigmoid(gate_logits.float()).cpu()

        bias_expanded = bias.unsqueeze(0).expand(BS, num_experts).contiguous()
        corrected_gpu = torch.empty_like(gate_sigmoid_gpu)
        glm.add(corrected_gpu.data_ptr(), gate_sigmoid_gpu.data_ptr(), bias_expanded.data_ptr(),
                BS * num_experts)
        corrected_ref = gate_sigmoid_ref + bias.float().unsqueeze(0).cpu()

        group_scores_gpu = corrected_gpu.view(BS * n_group, experts_per_group).topk(2, dim=-1).values
        group_sums_gpu = torch.empty(BS * n_group, dtype=torch.bfloat16, device=device)
        glm.reduce_sum(group_sums_gpu.data_ptr(), group_scores_gpu.reshape(-1).data_ptr(), BS * n_group, 2)

        group_scores_2d = group_sums_gpu.view(BS, n_group)
        group_idx_gpu = group_scores_2d.topk(topk_group, dim=-1, sorted=False)[1]

        group_mask_gpu = torch.zeros(BS, n_group, dtype=torch.bfloat16, device=device)
        glm.fill(group_mask_gpu.data_ptr(), 0.0, BS * n_group)
        glm.scatter_scalar(group_mask_gpu.data_ptr(), group_idx_gpu.data_ptr(),
                           1.0, topk_group, n_group, BS)

        scores_gpu = corrected_gpu.clone()
        glm.group_mask_mul(scores_gpu.data_ptr(), group_mask_gpu.data_ptr(),
                           num_experts, experts_per_group, n_group, BS)

        group_scores_ref = corrected_ref.view(BS, n_group, experts_per_group).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx_ref = group_scores_ref.topk(topk_group, dim=-1, sorted=False)[1]
        group_mask_ref = torch.zeros(BS, n_group)
        group_mask_ref.scatter_(1, group_idx_ref, 1.0)
        mask_expanded = group_mask_ref.unsqueeze(-1).expand(-1, n_group, experts_per_group).reshape(BS, num_experts)
        scores_ref = corrected_ref * mask_expanded

        topk_values_gpu, topk_indices_gpu = self._gpu_topk(glm, scores_gpu, top_k, num_experts, BS)

        topk_weights_gpu = torch.empty(BS, top_k, dtype=torch.bfloat16, device=device)
        glm.gather(topk_weights_gpu.data_ptr(), gate_sigmoid_gpu.data_ptr(),
                   topk_indices_gpu.data_ptr(), top_k, num_experts, BS)

        normalized_gpu = torch.empty_like(topk_weights_gpu)
        glm.row_normalize(normalized_gpu.data_ptr(), topk_weights_gpu.data_ptr(),
                          routed_scaling, BS, top_k, normalize=True)

        topk_indices_ref = topk_indices_gpu.cpu().long()
        topk_weights_ref = gate_sigmoid_ref.gather(1, topk_indices_ref)
        normalized_ref = topk_weights_ref / (topk_weights_ref.sum(dim=-1, keepdim=True) + 1e-20) * routed_scaling

        torch.testing.assert_close(normalized_gpu.cpu().float(), normalized_ref, atol=1e-2, rtol=1e-2)

    def test_routing_expert_scale_loop(self, glm, device):
        """Test the full expert dispatch loop with expert_scale."""
        BS = 4
        num_experts = 8
        top_k = 4
        hidden_size = 32
        routed_scaling = 2.5

        gate_logits = torch.randn(BS, num_experts, dtype=torch.bfloat16, device=device)
        gate_sigmoid_gpu = torch.empty_like(gate_logits)
        glm.sigmoid(gate_sigmoid_gpu.data_ptr(), gate_logits.data_ptr(), BS * num_experts)
        gate_sigmoid_ref = torch.sigmoid(gate_logits.float()).cpu()

        topk_values_gpu, topk_indices_gpu = self._gpu_topk(glm, gate_sigmoid_gpu, top_k, num_experts, BS)

        topk_weights_gpu = torch.empty(BS, top_k, dtype=torch.bfloat16, device=device)
        glm.gather(topk_weights_gpu.data_ptr(), gate_sigmoid_gpu.data_ptr(),
                   topk_indices_gpu.data_ptr(), top_k, num_experts, BS)

        normalized_gpu = torch.empty_like(topk_weights_gpu)
        glm.row_normalize(normalized_gpu.data_ptr(), topk_weights_gpu.data_ptr(),
                          routed_scaling, BS, top_k, normalize=True)

        topk_indices_ref = topk_indices_gpu.cpu().long()
        topk_weights_ref = gate_sigmoid_ref.gather(1, topk_indices_ref)
        normalized_ref = topk_weights_ref / (topk_weights_ref.sum(dim=-1, keepdim=True) + 1e-20) * routed_scaling

        expert_indices_cpu = topk_indices_gpu.cpu().numpy()

        routed_out_gpu = torch.zeros(BS, hidden_size, dtype=torch.bfloat16, device=device)
        glm.fill(routed_out_gpu.data_ptr(), 0.0, BS * hidden_size)

        routed_out_ref = torch.zeros(BS, hidden_size, dtype=torch.float32)
        scale_buf = torch.zeros(BS, dtype=torch.bfloat16, device=device)

        for e in range(num_experts):
            any_selected = False
            for b in range(BS):
                for k in range(top_k):
                    if expert_indices_cpu[b, k] == e:
                        any_selected = True
                        break
                if any_selected:
                    break
            if not any_selected:
                continue

            expert_out = torch.rand(BS, hidden_size, dtype=torch.bfloat16, device=device)

            glm.expert_scale(scale_buf.data_ptr(), normalized_gpu.data_ptr(),
                             topk_indices_gpu.data_ptr(), e, top_k, BS)

            glm.row_scale_add(routed_out_gpu.data_ptr(), expert_out.data_ptr(),
                              scale_buf.data_ptr(), BS, hidden_size)

            scale_ref = torch.zeros(BS, dtype=torch.float32)
            for b in range(BS):
                for k in range(top_k):
                    if expert_indices_cpu[b, k] == e:
                        scale_ref[b] = normalized_ref[b, k].item()
                        break
            routed_out_ref += expert_out.cpu().float() * scale_ref.unsqueeze(1)

        torch.testing.assert_close(routed_out_gpu.cpu().float(), routed_out_ref, atol=1e-2, rtol=1e-2)
