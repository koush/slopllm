import torch
import pytest
import math
import os
from helpers import GlmOps, ATOL, RTOL


def torch_merge_partial_attn(partial_v_outs, partial_lses):
    """Reference merge using online softmax (state_t::merge semantics)."""
    num_shards = len(partial_v_outs)
    B, H, D = partial_v_outs[0].shape
    device = partial_v_outs[0].device

    merged_o = torch.zeros(B, H, D, dtype=torch.float32, device=device)
    merged_m = torch.full((B, H), float('-inf'), dtype=torch.float32, device=device)
    merged_d = torch.ones(B, H, dtype=torch.float32, device=device)

    for s in range(num_shards):
        v = partial_v_outs[s].float()
        lse = partial_lses[s]
        m_prev = merged_m.clone()
        d_prev = merged_d.clone()
        merged_m = torch.max(merged_m, lse)
        merged_d = d_prev * torch.exp2(m_prev - merged_m) + torch.exp2(lse - merged_m)
        merged_o = merged_o * torch.exp2(m_prev - merged_m).unsqueeze(-1) + \
                   v * torch.exp2(lse - merged_m).unsqueeze(-1)

    merged_o = merged_o / merged_d.unsqueeze(-1)
    merged_lse = merged_m + torch.log2(merged_d)

    return merged_o.bfloat16(), merged_lse


def torch_merge_partial_attn_heads(partial_v_outs, partial_lses,
                                   num_heads, shard_n_heads, head_offset):
    """Reference head-grouped merge: merge only heads [head_offset, head_offset + shard_n_heads).

    Inputs have full [B, num_heads, D] layout. Output is [B, shard_n_heads, D].
    """
    num_shards = len(partial_v_outs)
    B = partial_v_outs[0].shape[0]
    D = partial_v_outs[0].shape[2]
    device = partial_v_outs[0].device

    merged_o = torch.zeros(B, shard_n_heads, D, dtype=torch.float32, device=device)
    merged_m = torch.full((B, shard_n_heads), float('-inf'), dtype=torch.float32, device=device)
    merged_d = torch.ones(B, shard_n_heads, dtype=torch.float32, device=device)

    for s in range(num_shards):
        v = partial_v_outs[s][:, head_offset:head_offset + shard_n_heads, :].float()
        lse = partial_lses[s][:, head_offset:head_offset + shard_n_heads]
        m_prev = merged_m.clone()
        d_prev = merged_d.clone()
        merged_m = torch.max(merged_m, lse)
        merged_d = d_prev * torch.exp2(m_prev - merged_m) + torch.exp2(lse - merged_m)
        merged_o = merged_o * torch.exp2(m_prev - merged_m).unsqueeze(-1) + \
                   v * torch.exp2(lse - merged_m).unsqueeze(-1)

    merged_o = merged_o / merged_d.unsqueeze(-1)
    merged_lse = merged_m + torch.log2(merged_d)

    return merged_o.bfloat16(), merged_lse


NUM_GPUS = min(torch.cuda.device_count(), 2)
SKIP_REASON = f"Tests require 1 GPU, found {torch.cuda.device_count()}"


@pytest.mark.skipif(NUM_GPUS < 1, reason=SKIP_REASON)
class TestP2PCpMerge:
    def test_cp_merge_2_shards(self, glm, device):
        """2-shard cp_merge_tree merge."""
        B, H, D = 1, 4, 128
        num_shards = 2
        torch.manual_seed(42)

        partial_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        partial_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]

        merged_v_ref, merged_lse_ref = torch_merge_partial_attn(partial_v_outs, partial_lses)

        merged_v_out = torch.empty(B, H, D, dtype=torch.bfloat16, device=device)
        merged_lse_out = torch.empty(B, H, dtype=torch.float32, device=device)

        glm.cp_merge_tree(
            [t.data_ptr() for t in partial_v_outs],
            [t.data_ptr() for t in partial_lses],
            num_shards,
            merged_v_out.data_ptr(),
            merged_lse_out.data_ptr(),
            B * H * D, B, H, D
        )
        glm.synchronize()

        torch.testing.assert_close(merged_v_out.cpu(), merged_v_ref.cpu(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(merged_lse_out.cpu(), merged_lse_ref.cpu(), atol=1e-3, rtol=1e-3)

    def test_cp_merge_nullptr_lse(self, glm, device):
        """cp_merge_tree with merged_lse=nullptr."""
        B, H, D = 1, 4, 128
        num_shards = 2
        torch.manual_seed(123)

        partial_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        partial_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]

        merged_v_ref, _ = torch_merge_partial_attn(partial_v_outs, partial_lses)

        merged_v_out = torch.empty(B, H, D, dtype=torch.bfloat16, device=device)

        glm.cp_merge_tree(
            [t.data_ptr() for t in partial_v_outs],
            [t.data_ptr() for t in partial_lses],
            num_shards,
            merged_v_out.data_ptr(),
            None,
            B * H * D, B, H, D
        )
        glm.synchronize()

        torch.testing.assert_close(merged_v_out.cpu(), merged_v_ref.cpu(), atol=1e-3, rtol=1e-3)

    def test_cp_merge_matches_reference(self, glm, device):
        """cp_merge_tree should match PyTorch reference merge."""
        B, H, D = 1, 4, 128
        num_shards = 2
        torch.manual_seed(99)

        partial_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        partial_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]

        merged_v = torch.empty(B, H, D, dtype=torch.bfloat16, device=device)
        merged_lse = torch.empty(B, H, dtype=torch.float32, device=device)

        glm.cp_merge_tree(
            [t.data_ptr() for t in partial_v_outs],
            [t.data_ptr() for t in partial_lses],
            num_shards,
            merged_v.data_ptr(),
            merged_lse.data_ptr(),
            B * H * D, B, H, D
        )
        glm.synchronize()

        merged_v_ref, merged_lse_ref = torch_merge_partial_attn(partial_v_outs, partial_lses)

        torch.testing.assert_close(merged_v.cpu(), merged_v_ref.cpu(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(merged_lse.cpu(), merged_lse_ref.cpu(), atol=1e-3, rtol=1e-3)

    def test_cp_merge_commutativity(self, glm, device):
        """merge(a,b) == merge(b,a) — swap shard order."""
        B, H, D = 1, 4, 128
        torch.manual_seed(77)

        v0 = torch.randn(B, H, D, dtype=torch.bfloat16, device=device)
        v1 = torch.randn(B, H, D, dtype=torch.bfloat16, device=device)
        lse0 = torch.randn(B, H, dtype=torch.float32, device=device)
        lse1 = torch.randn(B, H, dtype=torch.float32, device=device)

        merged_v_ab = torch.empty(B, H, D, dtype=torch.bfloat16, device=device)
        merged_lse_ab = torch.empty(B, H, dtype=torch.float32, device=device)

        glm.cp_merge_tree(
            [v0.data_ptr(), v1.data_ptr()],
            [lse0.data_ptr(), lse1.data_ptr()],
            2, merged_v_ab.data_ptr(), merged_lse_ab.data_ptr(),
            B * H * D, B, H, D
        )
        glm.synchronize()

        merged_v_ba = torch.empty(B, H, D, dtype=torch.bfloat16, device=device)
        merged_lse_ba = torch.empty(B, H, dtype=torch.float32, device=device)

        glm.cp_merge_tree(
            [v1.data_ptr(), v0.data_ptr()],
            [lse1.data_ptr(), lse0.data_ptr()],
            2, merged_v_ba.data_ptr(), merged_lse_ba.data_ptr(),
            B * H * D, B, H, D
        )
        glm.synchronize()

        torch.testing.assert_close(merged_v_ab.cpu(), merged_v_ba.cpu(), atol=0, rtol=0)
        torch.testing.assert_close(merged_lse_ab.cpu(), merged_lse_ba.cpu(), atol=1e-6, rtol=1e-6)

    def test_cp_merge_head_dim_256(self, glm, device):
        """cp_merge_tree with head_dim=256."""
        B, H, D = 1, 8, 256
        num_shards = 2
        torch.manual_seed(42)

        partial_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        partial_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]

        merged_v_ref, merged_lse_ref = torch_merge_partial_attn(partial_v_outs, partial_lses)

        merged_v_out = torch.empty(B, H, D, dtype=torch.bfloat16, device=device)
        merged_lse_out = torch.empty(B, H, dtype=torch.float32, device=device)

        glm.cp_merge_tree(
            [t.data_ptr() for t in partial_v_outs],
            [t.data_ptr() for t in partial_lses],
            num_shards,
            merged_v_out.data_ptr(),
            merged_lse_out.data_ptr(),
            B * H * D, B, H, D
        )
        glm.synchronize()

        torch.testing.assert_close(merged_v_out.cpu(), merged_v_ref.cpu(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(merged_lse_out.cpu(), merged_lse_ref.cpu(), atol=1e-3, rtol=1e-3)

    def test_cp_merge_heads_first_half(self, glm, device):
        """Head-grouped merge: only first half of heads.

        Uses shard-layout v_out with head_offset=0 and input_n_heads=shard_n_heads.
        """
        B, H, D = 1, 8, 128
        shard_n_heads = H // 2
        head_offset = 0
        num_shards = 2
        torch.manual_seed(42)

        full_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        partial_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]
        shard_v_outs = [v[:, head_offset:head_offset + shard_n_heads, :].contiguous() for v in full_v_outs]

        merged_v_ref, merged_lse_ref = torch_merge_partial_attn(full_v_outs, partial_lses)
        merged_v_ref_slice = merged_v_ref[:, head_offset:head_offset + shard_n_heads, :]
        merged_lse_ref_slice = merged_lse_ref[:, head_offset:head_offset + shard_n_heads]

        merged_v = torch.empty(B, shard_n_heads, D, dtype=torch.bfloat16, device=device)
        merged_lse = torch.empty(B, shard_n_heads, dtype=torch.float32, device=device)

        glm.cp_merge_tree(
            [t.data_ptr() for t in shard_v_outs],
            [t.data_ptr() for t in partial_lses],
            num_shards,
            merged_v.data_ptr(),
            merged_lse.data_ptr(),
            B * shard_n_heads * D, B, H, D,
            shard_n_heads, head_offset, shard_n_heads
        )
        glm.synchronize()

        torch.testing.assert_close(merged_v.cpu(), merged_v_ref_slice.cpu(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(merged_lse.cpu(), merged_lse_ref_slice.cpu(), atol=1e-3, rtol=1e-3)

    def test_cp_merge_heads_two_tp_shards(self, glm, device):
        """Head-grouped merge simulating 2-way TP: merge heads [0,H/2) and [H/2,H) separately.

        Uses full-layout v_out with input_n_heads=num_heads because head_offset > 0 for second group.
        """
        B, H, D = 1, 8, 128
        shard_n_heads = H // 2
        num_shards = 2
        torch.manual_seed(99)

        full_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        partial_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]

        merged_v_ref, merged_lse_ref = torch_merge_partial_attn(full_v_outs, partial_lses)

        for tp_rank in range(2):
            offset = tp_rank * shard_n_heads
            merged_v = torch.empty(B, shard_n_heads, D, dtype=torch.bfloat16, device=device)
            merged_lse = torch.empty(B, shard_n_heads, dtype=torch.float32, device=device)

            glm.cp_merge_tree(
                [t.data_ptr() for t in full_v_outs],
                [t.data_ptr() for t in partial_lses],
                num_shards,
                merged_v.data_ptr(),
                merged_lse.data_ptr(),
                B * shard_n_heads * D, B, H, D,
                shard_n_heads, offset, H
            )
            glm.synchronize()

            torch.testing.assert_close(
                merged_v.cpu(), merged_v_ref[:, offset:offset + shard_n_heads, :].cpu(),
                atol=1e-3, rtol=1e-3)
            torch.testing.assert_close(
                merged_lse.cpu(), merged_lse_ref[:, offset:offset + shard_n_heads].cpu(),
                atol=1e-3, rtol=1e-3)

    def test_cp_merge_heads_matches_full_merge(self, glm, device):
        """Per-head merge should match full merge when all head groups are combined.

        Uses full-layout v_out with input_n_heads=num_heads because head_offset > 0.
        """
        B, H, D = 1, 8, 128
        shard_n_heads = H // 2
        head_offset = H // 2
        num_shards = 2
        torch.manual_seed(77)

        full_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        partial_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]

        # Full merge
        merged_v_full = torch.empty(B, H, D, dtype=torch.bfloat16, device=device)
        merged_lse_full = torch.empty(B, H, dtype=torch.float32, device=device)

        glm.cp_merge_tree(
            [t.data_ptr() for t in full_v_outs],
            [t.data_ptr() for t in partial_lses],
            num_shards,
            merged_v_full.data_ptr(),
            merged_lse_full.data_ptr(),
            B * H * D, B, H, D
        )
        glm.synchronize()

        # Per-head merge: first half and second half
        merged_v_parts = []
        merged_lse_parts = []
        for half in range(2):
            offset = half * shard_n_heads
            merged_v = torch.empty(B, shard_n_heads, D, dtype=torch.bfloat16, device=device)
            merged_lse = torch.empty(B, shard_n_heads, dtype=torch.float32, device=device)

            glm.cp_merge_tree(
                [t.data_ptr() for t in full_v_outs],
                [t.data_ptr() for t in partial_lses],
                num_shards,
                merged_v.data_ptr(),
                merged_lse.data_ptr(),
                B * shard_n_heads * D, B, H, D,
                shard_n_heads, offset, H
            )
            glm.synchronize()
            merged_v_parts.append(merged_v)
            merged_lse_parts.append(merged_lse)

        # Concatenate per-head results and compare with full merge
        cat_v = torch.cat(merged_v_parts, dim=1)
        cat_lse = torch.cat(merged_lse_parts, dim=1)

        torch.testing.assert_close(cat_v.cpu(), merged_v_full.cpu(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(cat_lse.cpu(), merged_lse_full.cpu(), atol=1e-3, rtol=1e-3)
