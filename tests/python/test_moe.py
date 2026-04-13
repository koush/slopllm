import torch
import torch.nn.functional as F
import pytest
from helpers import ATOL, RTOL


def moe_route_torch(hidden_states, gate_weight, e_score_correction_bias,
                    n_group, topk_group, top_k, num_experts,
                    norm_topk_prob, routed_scaling_factor):
    x = hidden_states.view(-1, hidden_states.shape[-1])
    logits = F.linear(x, gate_weight).sigmoid()
    logits_corrected = logits + e_score_correction_bias.float().to(torch.bfloat16)
    experts_per_group = num_experts // n_group
    group_scores = logits_corrected.view(-1, n_group, experts_per_group).topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = group_scores.topk(topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
    score_mask = group_mask.unsqueeze(-1).expand(-1, n_group, experts_per_group).reshape(-1, num_experts)
    scores_masked = logits_corrected.masked_fill(~score_mask.bool(), 0.0)
    topk_idx = scores_masked.topk(top_k, dim=-1, sorted=False)[1]
    topk_w = logits.gather(1, topk_idx)
    if norm_topk_prob:
        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-20)
    topk_w = topk_w * routed_scaling_factor
    return topk_idx, topk_w


def moe_forward_torch(hidden_states, gate_weight, e_score_correction_bias,
                      expert_gates, expert_ups, expert_downs,
                      shared_gate_w, shared_up_w, shared_down_w,
                      n_group, topk_group, top_k, num_experts,
                      norm_topk_prob, routed_scaling_factor, hidden, moe_inter):
    residual = hidden_states
    orig_shape = hidden_states.shape
    topk_idx, topk_w = moe_route_torch(
        hidden_states, gate_weight, e_score_correction_bias,
        n_group, topk_group, top_k, num_experts,
        norm_topk_prob, routed_scaling_factor)

    h = hidden_states.view(-1, hidden_states.shape[-1])
    out = torch.zeros_like(h)
    expert_mask = F.one_hot(topk_idx, num_experts).permute(2, 1, 0)

    for i in range(num_experts):
        top_k_pos, token_idx = torch.where(expert_mask[i])
        if token_idx.numel() == 0:
            continue
        cur = h[token_idx]
        gate = F.linear(cur, expert_gates[i])
        up = F.linear(cur, expert_ups[i])
        gate_up = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
        down = F.linear(gate_up, expert_downs[i])
        out.index_add_(0, token_idx, (down * topk_w[token_idx, top_k_pos, None]))

    h = out.view(*orig_shape)
    shared_gate = F.linear(residual, shared_gate_w)
    shared_up = F.linear(residual, shared_up_w)
    shared_gate_up = (F.silu(shared_gate.float()) * shared_up.float()).to(torch.bfloat16)
    shared = F.linear(shared_gate_up, shared_down_w)
    return h + shared


def moe_route_cuda(glm, device, hidden_states_flat, gate_weight, e_score_correction_bias,
                   n_group, topk_group, top_k, num_experts,
                   norm_topk_prob, routed_scaling_factor):
    N = hidden_states_flat.shape[0]
    hidden = hidden_states_flat.shape[1]
    experts_per_group = num_experts // n_group

    logits = torch.empty(N, num_experts, dtype=torch.bfloat16, device=device)
    glm.linear(logits.reshape(-1), hidden_states_flat.reshape(-1), gate_weight,
               N, num_experts, hidden)
    glm.sigmoid(logits.reshape(-1), logits.reshape(-1), N * num_experts)

    logits_corrected = logits + e_score_correction_bias.unsqueeze(0)

    logits_grouped = logits_corrected.view(N, n_group, experts_per_group).contiguous()
    top2_values = torch.empty(N * n_group, 2, dtype=torch.bfloat16, device=device)
    top2_indices = torch.empty(N * n_group, 2, dtype=torch.int32, device=device)
    glm.topk(top2_values, top2_indices, logits_grouped.reshape(N * n_group, experts_per_group),
             2, experts_per_group, N * n_group)

    group_scores = torch.empty(N * n_group, dtype=torch.bfloat16, device=device)
    glm.reduce_sum(group_scores, top2_values.reshape(-1), N * n_group, 2)
    group_scores = group_scores.reshape(N, n_group)

    group_idx = torch.empty(N, topk_group, dtype=torch.int32, device=device)
    group_idx_values = torch.empty(N, topk_group, dtype=torch.bfloat16, device=device)
    glm.topk(group_idx_values, group_idx, group_scores.reshape(N, n_group),
             topk_group, n_group, N)

    group_mask = torch.empty(N, n_group, dtype=torch.bfloat16, device=device)
    glm.fill(group_mask.reshape(-1), 0.0, N * n_group)
    glm.scatter_scalar(group_mask.reshape(-1), group_idx.reshape(-1),
                       1.0, topk_group, n_group, N)

    score_mask = group_mask.unsqueeze(-1).expand(N, n_group, experts_per_group).contiguous()
    score_mask = score_mask.reshape(N, num_experts)

    scores_masked = torch.empty(N, num_experts, dtype=torch.bfloat16, device=device)
    glm.mul(scores_masked.reshape(-1), logits_corrected.reshape(-1),
            score_mask.reshape(-1), N * num_experts)

    topk_values = torch.empty(N, top_k, dtype=torch.bfloat16, device=device)
    topk_idx = torch.empty(N, top_k, dtype=torch.int32, device=device)
    glm.topk(topk_values, topk_idx, scores_masked.reshape(N, num_experts),
             top_k, num_experts, N)

    topk_w = torch.empty(N, top_k, dtype=torch.bfloat16, device=device)
    glm.gather(topk_w.reshape(-1), logits.reshape(-1), topk_idx.reshape(-1),
               top_k, num_experts, N)

    if norm_topk_prob:
        topk_w_sum = torch.empty(N, dtype=torch.bfloat16, device=device)
        glm.reduce_sum(topk_w_sum, topk_w.reshape(-1), N, top_k)
        topk_w_sum = topk_w_sum + 1e-20
        topk_w_sum_recip = (1.0 / topk_w_sum.float()).to(torch.bfloat16)
        topk_w_sum_expanded = topk_w_sum_recip.unsqueeze(1).expand(N, top_k).contiguous()
        glm.mul(topk_w.reshape(-1), topk_w.reshape(-1),
                topk_w_sum_expanded.reshape(-1), N * top_k)

    glm.scale(topk_w.reshape(-1), topk_w.reshape(-1), routed_scaling_factor, N * top_k)

    return topk_idx, topk_w


def moe_forward_cuda(glm, device, hidden_states, gate_weight, e_score_correction_bias,
                     expert_gates, expert_ups, expert_downs,
                     shared_gate_w, shared_up_w, shared_down_w,
                     n_group, topk_group, top_k, num_experts,
                     norm_topk_prob, routed_scaling_factor, hidden, moe_inter):
    B, S, H = hidden_states.shape
    flat = hidden_states.reshape(B * S, H)

    topk_idx, topk_w = moe_route_cuda(
        glm, device, flat, gate_weight, e_score_correction_bias,
        n_group, topk_group, top_k, num_experts,
        norm_topk_prob, routed_scaling_factor)

    out = torch.zeros(B * S, H, dtype=torch.bfloat16, device=device)
    glm.fill(out.reshape(-1), 0.0, B * S * H)

    expert_mask = F.one_hot(topk_idx.cpu().long(), num_experts).permute(2, 1, 0)

    for i in range(num_experts):
        top_k_pos, token_idx = torch.where(expert_mask[i])
        if token_idx.numel() == 0:
            continue
        token_idx_gpu = token_idx.to(device)
        top_k_pos_gpu = top_k_pos.to(device)
        n_tokens = token_idx_gpu.shape[0]

        cur = torch.empty(n_tokens, H, dtype=torch.bfloat16, device=device)
        glm.index_select(cur.reshape(-1), flat.reshape(-1),
                         token_idx_gpu.to(torch.int32), H, n_tokens)

        gate = torch.empty(n_tokens, moe_inter, dtype=torch.bfloat16, device=device)
        glm.linear(gate.reshape(-1), cur.reshape(-1), expert_gates[i],
                   n_tokens, moe_inter, H)
        up = torch.empty(n_tokens, moe_inter, dtype=torch.bfloat16, device=device)
        glm.linear(up.reshape(-1), cur.reshape(-1), expert_ups[i],
                   n_tokens, moe_inter, H)
        gate_up = torch.empty(n_tokens, moe_inter, dtype=torch.bfloat16, device=device)
        glm.silu_and_mul(gate_up.reshape(-1), gate.reshape(-1), up.reshape(-1),
                         moe_inter, n_tokens)
        down = torch.empty(n_tokens, H, dtype=torch.bfloat16, device=device)
        glm.linear(down.reshape(-1), gate_up.reshape(-1), expert_downs[i],
                   n_tokens, H, moe_inter)

        weights = topk_w[token_idx_gpu.long(), top_k_pos_gpu.long()].unsqueeze(1)
        glm.mul(down.reshape(-1), down.reshape(-1),
                weights.expand(n_tokens, H).contiguous().reshape(-1), n_tokens * H)

        glm.index_add(out.reshape(-1), token_idx_gpu.to(torch.int32),
                      down.reshape(-1), n_tokens, H)

    shared_gate = torch.empty(B * S, moe_inter, dtype=torch.bfloat16, device=device)
    glm.linear(shared_gate.reshape(-1), hidden_states.reshape(-1), shared_gate_w,
               B * S, moe_inter, H)
    shared_up = torch.empty(B * S, moe_inter, dtype=torch.bfloat16, device=device)
    glm.linear(shared_up.reshape(-1), hidden_states.reshape(-1), shared_up_w,
               B * S, moe_inter, H)
    shared_gate_up = torch.empty(B * S, moe_inter, dtype=torch.bfloat16, device=device)
    glm.silu_and_mul(shared_gate_up.reshape(-1), shared_gate.reshape(-1),
                     shared_up.reshape(-1), moe_inter, B * S)
    shared_down = torch.empty(B * S, H, dtype=torch.bfloat16, device=device)
    glm.linear(shared_down.reshape(-1), shared_gate_up.reshape(-1), shared_down_w,
               B * S, H, moe_inter)

    glm.add(out.reshape(-1), out.reshape(-1), shared_down.reshape(-1), B * S * H)

    return out.reshape(B, S, H)


def test_moe_route_small(glm, device):
    num_experts = 8
    top_k = 2
    n_group = 2
    topk_group = 1
    hidden = 32
    num_tokens = 4
    norm_topk_prob = True
    routed_scaling_factor = 1.0

    torch.manual_seed(42)
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    gate_weight = torch.randn(num_experts, hidden, dtype=torch.bfloat16, device=device)
    e_score_correction_bias = torch.randn(num_experts, dtype=torch.bfloat16, device=device)

    ref_idx, ref_w = moe_route_torch(
        x, gate_weight, e_score_correction_bias,
        n_group, topk_group, top_k, num_experts,
        norm_topk_prob, routed_scaling_factor)

    cuda_idx, cuda_w = moe_route_cuda(
        glm, device, x.reshape(num_tokens, hidden), gate_weight, e_score_correction_bias,
        n_group, topk_group, top_k, num_experts,
        norm_topk_prob, routed_scaling_factor)

    ref_idx_i = ref_idx.cpu().long()
    cuda_idx_i = cuda_idx.cpu().long()

    ref_w_sorted, _ = torch.sort(ref_w.float(), dim=-1, descending=True)
    cuda_w_sorted, _ = torch.sort(cuda_w.float(), dim=-1, descending=True)
    torch.testing.assert_close(cuda_w_sorted.cpu(), ref_w_sorted.cpu(), atol=1e-2, rtol=1e-1)


def test_moe_forward_small(glm, device):
    num_experts = 4
    top_k = 2
    n_group = 2
    topk_group = 1
    hidden = 32
    moe_inter = 64
    norm_topk_prob = True
    routed_scaling_factor = 1.0
    B, S = 1, 4

    torch.manual_seed(99)
    hidden_states = torch.randn(B, S, hidden, dtype=torch.bfloat16, device=device)
    gate_weight = torch.randn(num_experts, hidden, dtype=torch.bfloat16, device=device)
    e_score_correction_bias = torch.randn(num_experts, dtype=torch.bfloat16, device=device)
    expert_gates = [torch.randn(moe_inter, hidden, dtype=torch.bfloat16, device=device) for _ in range(num_experts)]
    expert_ups = [torch.randn(moe_inter, hidden, dtype=torch.bfloat16, device=device) for _ in range(num_experts)]
    expert_downs = [torch.randn(hidden, moe_inter, dtype=torch.bfloat16, device=device) for _ in range(num_experts)]
    shared_gate_w = torch.randn(moe_inter, hidden, dtype=torch.bfloat16, device=device)
    shared_up_w = torch.randn(moe_inter, hidden, dtype=torch.bfloat16, device=device)
    shared_down_w = torch.randn(hidden, moe_inter, dtype=torch.bfloat16, device=device)

    ref = moe_forward_torch(
        hidden_states, gate_weight, e_score_correction_bias,
        expert_gates, expert_ups, expert_downs,
        shared_gate_w, shared_up_w, shared_down_w,
        n_group, topk_group, top_k, num_experts,
        norm_topk_prob, routed_scaling_factor, hidden, moe_inter)

    cuda = moe_forward_cuda(
        glm, device, hidden_states, gate_weight, e_score_correction_bias,
        expert_gates, expert_ups, expert_downs,
        shared_gate_w, shared_up_w, shared_down_w,
        n_group, topk_group, top_k, num_experts,
        norm_topk_prob, routed_scaling_factor, hidden, moe_inter)

    torch.testing.assert_close(cuda.cpu(), ref.cpu(), atol=0.5, rtol=1e-1)


def test_moe_forward_no_norm(glm, device):
    num_experts = 4
    top_k = 2
    n_group = 2
    topk_group = 1
    hidden = 32
    moe_inter = 64
    norm_topk_prob = False
    routed_scaling_factor = 2.0
    B, S = 2, 3

    torch.manual_seed(77)
    hidden_states = torch.randn(B, S, hidden, dtype=torch.bfloat16, device=device)
    gate_weight = torch.randn(num_experts, hidden, dtype=torch.bfloat16, device=device)
    e_score_correction_bias = torch.randn(num_experts, dtype=torch.bfloat16, device=device)
    expert_gates = [torch.randn(moe_inter, hidden, dtype=torch.bfloat16, device=device) for _ in range(num_experts)]
    expert_ups = [torch.randn(moe_inter, hidden, dtype=torch.bfloat16, device=device) for _ in range(num_experts)]
    expert_downs = [torch.randn(hidden, moe_inter, dtype=torch.bfloat16, device=device) for _ in range(num_experts)]
    shared_gate_w = torch.randn(moe_inter, hidden, dtype=torch.bfloat16, device=device)
    shared_up_w = torch.randn(moe_inter, hidden, dtype=torch.bfloat16, device=device)
    shared_down_w = torch.randn(hidden, moe_inter, dtype=torch.bfloat16, device=device)

    ref = moe_forward_torch(
        hidden_states, gate_weight, e_score_correction_bias,
        expert_gates, expert_ups, expert_downs,
        shared_gate_w, shared_up_w, shared_down_w,
        n_group, topk_group, top_k, num_experts,
        norm_topk_prob, routed_scaling_factor, hidden, moe_inter)

    cuda = moe_forward_cuda(
        glm, device, hidden_states, gate_weight, e_score_correction_bias,
        expert_gates, expert_ups, expert_downs,
        shared_gate_w, shared_up_w, shared_down_w,
        n_group, topk_group, top_k, num_experts,
        norm_topk_prob, routed_scaling_factor, hidden, moe_inter)

    torch.testing.assert_close(cuda.cpu(), ref.cpu(), atol=0.5, rtol=1e-1)
