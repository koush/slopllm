import torch
import pytest
from helpers import GlmOps


def torch_merge_partial_attn(partial_v_outs, partial_lses):
    """Reference merge using online softmax (state_t::merge semantics).

    Args:
        partial_v_outs: list of [B, H, D] BF16 tensors
        partial_lses: list of [B, H] F32 tensors (base-2 log-sum-exp)

    Returns:
        merged_v: [B, H, D] BF16
        merged_lse: [B, H] F32
    """
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


def shard_v_outs_from_full(full_v_outs, head_offset, shard_n_heads):
    """Extract shard-layout v_outs [B, snh, D] from full-layout [B, H, D]."""
    return [v[:, head_offset:head_offset + shard_n_heads, :].contiguous() for v in full_v_outs]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestContextParallelMergeHeads:

    def test_heads_matches_full_merge(self, glm, device):
        """Head-grouped merge with shard_n_heads=num_heads, head_offset=0 matches full merge."""
        B, H, D = 2, 8, 128
        num_shards = 2
        torch.manual_seed(42)

        partial_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        partial_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]

        merged_v_ref, merged_lse_ref = torch_merge_partial_attn(partial_v_outs, partial_lses)

        merged_v_heads = torch.empty(B, H, D, dtype=torch.bfloat16, device=device)
        merged_lse_heads = torch.empty(B, H, dtype=torch.float32, device=device)

        glm.cp_merge_tree(
            [t.data_ptr() for t in partial_v_outs],
            [t.data_ptr() for t in partial_lses],
            num_shards,
            merged_v_heads.data_ptr(),
            merged_lse_heads.data_ptr(),
            B * H * D, B, H, D, H, 0, H
        )
        glm.synchronize()

        torch.testing.assert_close(merged_v_heads.cpu(), merged_v_ref.cpu(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(merged_lse_heads.cpu(), merged_lse_ref.cpu(), atol=1e-3, rtol=1e-3)

    def test_heads_first_half(self, glm, device):
        """Merge only first half of heads, verify matches slice of full merge.

        Uses shard-layout v_out [B, snh, D] with head_offset=0 and input_n_heads=shard_n_heads.
        """
        B, H, D = 2, 8, 128
        shard_n_heads = H // 2
        head_offset = 0
        num_shards = 2
        torch.manual_seed(42)

        partial_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        partial_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]

        merged_v_ref, merged_lse_ref = torch_merge_partial_attn(partial_v_outs, partial_lses)
        merged_v_ref_slice = merged_v_ref[:, head_offset:head_offset + shard_n_heads, :]
        merged_lse_ref_slice = merged_lse_ref[:, head_offset:head_offset + shard_n_heads]

        shard_v_outs = shard_v_outs_from_full(partial_v_outs, head_offset, shard_n_heads)
        merged_v_heads = torch.empty(B, shard_n_heads, D, dtype=torch.bfloat16, device=device)
        merged_lse_heads = torch.empty(B, shard_n_heads, dtype=torch.float32, device=device)

        glm.cp_merge_tree(
            [t.data_ptr() for t in shard_v_outs],
            [t.data_ptr() for t in partial_lses],
            num_shards,
            merged_v_heads.data_ptr(),
            merged_lse_heads.data_ptr(),
            B * shard_n_heads * D, B, H, D, shard_n_heads, head_offset, shard_n_heads
        )
        glm.synchronize()

        torch.testing.assert_close(merged_v_heads.cpu(), merged_v_ref_slice.cpu(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(merged_lse_heads.cpu(), merged_lse_ref_slice.cpu(), atol=1e-3, rtol=1e-3)

    def test_heads_second_half(self, glm, device):
        """Merge only second half of heads, verify matches slice of full merge.

        Uses full-layout v_out [B, H, D] with input_n_heads=num_heads because head_offset>0.
        """
        B, H, D = 2, 8, 128
        shard_n_heads = H // 2
        head_offset = H // 2
        num_shards = 2
        torch.manual_seed(123)

        partial_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        partial_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]

        merged_v_ref, merged_lse_ref = torch_merge_partial_attn(partial_v_outs, partial_lses)
        merged_v_ref_slice = merged_v_ref[:, head_offset:head_offset + shard_n_heads, :]
        merged_lse_ref_slice = merged_lse_ref[:, head_offset:head_offset + shard_n_heads]

        merged_v_heads = torch.empty(B, shard_n_heads, D, dtype=torch.bfloat16, device=device)
        merged_lse_heads = torch.empty(B, shard_n_heads, dtype=torch.float32, device=device)

        glm.cp_merge_tree(
            [t.data_ptr() for t in partial_v_outs],
            [t.data_ptr() for t in partial_lses],
            num_shards,
            merged_v_heads.data_ptr(),
            merged_lse_heads.data_ptr(),
            B * shard_n_heads * D, B, H, D, shard_n_heads, head_offset, H
        )
        glm.synchronize()

        torch.testing.assert_close(merged_v_heads.cpu(), merged_v_ref_slice.cpu(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(merged_lse_heads.cpu(), merged_lse_ref_slice.cpu(), atol=1e-3, rtol=1e-3)

    def test_heads_two_tp_shards_reconstruct(self, glm, device):
        """Simulate 2-way TP: merge heads [0,H/2) and [H/2,H) separately, combine = full merge.

        Uses full-layout v_out with input_n_heads=num_heads because head_offset>0 for second group.
        """
        B, H, D = 2, 8, 128
        shard_n_heads = H // 2
        num_shards = 2
        torch.manual_seed(99)

        partial_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        partial_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]

        merged_v_ref, merged_lse_ref = torch_merge_partial_attn(partial_v_outs, partial_lses)

        merged_v_shard0 = torch.empty(B, shard_n_heads, D, dtype=torch.bfloat16, device=device)
        merged_lse_shard0 = torch.empty(B, shard_n_heads, dtype=torch.float32, device=device)
        glm.cp_merge_tree(
            [t.data_ptr() for t in partial_v_outs],
            [t.data_ptr() for t in partial_lses],
            num_shards,
            merged_v_shard0.data_ptr(),
            merged_lse_shard0.data_ptr(),
            B * shard_n_heads * D, B, H, D, shard_n_heads, 0, H
        )

        merged_v_shard1 = torch.empty(B, shard_n_heads, D, dtype=torch.bfloat16, device=device)
        merged_lse_shard1 = torch.empty(B, shard_n_heads, dtype=torch.float32, device=device)
        glm.cp_merge_tree(
            [t.data_ptr() for t in partial_v_outs],
            [t.data_ptr() for t in partial_lses],
            num_shards,
            merged_v_shard1.data_ptr(),
            merged_lse_shard1.data_ptr(),
            B * shard_n_heads * D, B, H, D, shard_n_heads, shard_n_heads, H
        )

        glm.synchronize()

        torch.testing.assert_close(merged_v_shard0.cpu(), merged_v_ref[:, :shard_n_heads, :].cpu(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(merged_v_shard1.cpu(), merged_v_ref[:, shard_n_heads:, :].cpu(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(merged_lse_shard0.cpu(), merged_lse_ref[:, :shard_n_heads].cpu(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(merged_lse_shard1.cpu(), merged_lse_ref[:, shard_n_heads:].cpu(), atol=1e-3, rtol=1e-3)

    def test_heads_four_way_tp(self, glm, device):
        """Simulate 4-way TP: merge 4 head groups separately, combine = full merge.

        Uses full-layout v_out with input_n_heads=num_heads.
        """
        B, H, D = 1, 8, 128
        tp_size = 4
        shard_n_heads = H // tp_size
        num_shards = 2
        torch.manual_seed(77)

        partial_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        partial_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]

        merged_v_ref, merged_lse_ref = torch_merge_partial_attn(partial_v_outs, partial_lses)

        merged_v_shards = []
        merged_lse_shards = []
        for tp_rank in range(tp_size):
            offset = tp_rank * shard_n_heads
            merged_v = torch.empty(B, shard_n_heads, D, dtype=torch.bfloat16, device=device)
            merged_lse = torch.empty(B, shard_n_heads, dtype=torch.float32, device=device)
            glm.cp_merge_tree(
                [t.data_ptr() for t in partial_v_outs],
                [t.data_ptr() for t in partial_lses],
                num_shards,
                merged_v.data_ptr(),
                merged_lse.data_ptr(),
                B * shard_n_heads * D, B, H, D, shard_n_heads, offset, H
            )
            merged_v_shards.append(merged_v)
            merged_lse_shards.append(merged_lse)

        glm.synchronize()

        reconstructed_v = torch.cat(merged_v_shards, dim=1)
        reconstructed_lse = torch.cat(merged_lse_shards, dim=1)

        torch.testing.assert_close(reconstructed_v.cpu(), merged_v_ref.cpu(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(reconstructed_lse.cpu(), merged_lse_ref.cpu(), atol=1e-3, rtol=1e-3)

    def test_heads_single_head(self, glm, device):
        """Merge a single head (shard_n_heads=1) at various offsets.

        Uses full-layout v_out with input_n_heads=num_heads.
        """
        B, H, D = 2, 8, 128
        num_shards = 2
        torch.manual_seed(42)

        partial_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        partial_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]

        merged_v_ref, merged_lse_ref = torch_merge_partial_attn(partial_v_outs, partial_lses)

        for h in range(H):
            merged_v = torch.empty(B, 1, D, dtype=torch.bfloat16, device=device)
            merged_lse = torch.empty(B, 1, dtype=torch.float32, device=device)
            glm.cp_merge_tree(
                [t.data_ptr() for t in partial_v_outs],
                [t.data_ptr() for t in partial_lses],
                num_shards,
                merged_v.data_ptr(),
                merged_lse.data_ptr(),
                B * 1 * D, B, H, D, 1, h, H
            )
            glm.synchronize()
            torch.testing.assert_close(merged_v.cpu(), merged_v_ref[:, h:h+1, :].cpu(), atol=1e-3, rtol=1e-3)
            torch.testing.assert_close(merged_lse.cpu(), merged_lse_ref[:, h:h+1].cpu(), atol=1e-3, rtol=1e-3)

    def test_heads_4_shards(self, glm, device):
        """Head-grouped merge with 4 CP shards.

        Uses full-layout v_out with input_n_heads=num_heads because head_offset>0.
        """
        B, H, D = 2, 8, 128
        shard_n_heads = H // 2
        head_offset = shard_n_heads
        num_shards = 4
        torch.manual_seed(42)

        partial_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        partial_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]

        merged_v_ref, merged_lse_ref = torch_merge_partial_attn(partial_v_outs, partial_lses)
        merged_v_ref_slice = merged_v_ref[:, head_offset:head_offset + shard_n_heads, :]
        merged_lse_ref_slice = merged_lse_ref[:, head_offset:head_offset + shard_n_heads]

        merged_v_heads = torch.empty(B, shard_n_heads, D, dtype=torch.bfloat16, device=device)
        merged_lse_heads = torch.empty(B, shard_n_heads, dtype=torch.float32, device=device)

        glm.cp_merge_tree(
            [t.data_ptr() for t in partial_v_outs],
            [t.data_ptr() for t in partial_lses],
            num_shards,
            merged_v_heads.data_ptr(),
            merged_lse_heads.data_ptr(),
            B * shard_n_heads * D, B, H, D, shard_n_heads, head_offset, H
        )
        glm.synchronize()

        torch.testing.assert_close(merged_v_heads.cpu(), merged_v_ref_slice.cpu(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(merged_lse_heads.cpu(), merged_lse_ref_slice.cpu(), atol=1e-3, rtol=1e-3)

    def test_heads_8_shards(self, glm, device):
        """Head-grouped merge with 8 CP shards.

        Uses full-layout v_out with input_n_heads=num_heads because head_offset>0.
        """
        B, H, D = 1, 16, 64
        shard_n_heads = H // 4
        head_offset = shard_n_heads * 2
        num_shards = 8
        torch.manual_seed(55)

        partial_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        partial_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]

        merged_v_ref, merged_lse_ref = torch_merge_partial_attn(partial_v_outs, partial_lses)
        merged_v_ref_slice = merged_v_ref[:, head_offset:head_offset + shard_n_heads, :]
        merged_lse_ref_slice = merged_lse_ref[:, head_offset:head_offset + shard_n_heads]

        merged_v_heads = torch.empty(B, shard_n_heads, D, dtype=torch.bfloat16, device=device)
        merged_lse_heads = torch.empty(B, shard_n_heads, dtype=torch.float32, device=device)

        glm.cp_merge_tree(
            [t.data_ptr() for t in partial_v_outs],
            [t.data_ptr() for t in partial_lses],
            num_shards,
            merged_v_heads.data_ptr(),
            merged_lse_heads.data_ptr(),
            B * shard_n_heads * D, B, H, D, shard_n_heads, head_offset, H
        )
        glm.synchronize()

        torch.testing.assert_close(merged_v_heads.cpu(), merged_v_ref_slice.cpu(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(merged_lse_heads.cpu(), merged_lse_ref_slice.cpu(), atol=1e-3, rtol=1e-3)

    def test_heads_head_dim_256(self, glm, device):
        """Head-grouped merge with v_head_dim=256 (Qwen3.5 full attention).

        Uses shard-layout v_out with head_offset=0 and input_n_heads=shard_n_heads.
        """
        B, H, D = 1, 8, 256
        shard_n_heads = H // 2
        head_offset = 0
        num_shards = 2
        torch.manual_seed(42)

        partial_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        partial_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]

        merged_v_ref, merged_lse_ref = torch_merge_partial_attn(partial_v_outs, partial_lses)
        merged_v_ref_slice = merged_v_ref[:, head_offset:head_offset + shard_n_heads, :]
        merged_lse_ref_slice = merged_lse_ref[:, head_offset:head_offset + shard_n_heads]

        shard_v = shard_v_outs_from_full(partial_v_outs, head_offset, shard_n_heads)
        merged_v_heads = torch.empty(B, shard_n_heads, D, dtype=torch.bfloat16, device=device)
        merged_lse_heads = torch.empty(B, shard_n_heads, dtype=torch.float32, device=device)

        glm.cp_merge_tree(
            [t.data_ptr() for t in shard_v],
            [t.data_ptr() for t in partial_lses],
            num_shards,
            merged_v_heads.data_ptr(),
            merged_lse_heads.data_ptr(),
            B * shard_n_heads * D, B, H, D, shard_n_heads, head_offset, shard_n_heads
        )
        glm.synchronize()

        torch.testing.assert_close(merged_v_heads.cpu(), merged_v_ref_slice.cpu(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(merged_lse_heads.cpu(), merged_lse_ref_slice.cpu(), atol=1e-3, rtol=1e-3)

    def test_heads_head_dim_512(self, glm, device):
        """Head-grouped merge with v_head_dim=512 (GLM-5.1 merge before v_expand).

        Uses full-layout v_out with input_n_heads=num_heads because head_offset>0.
        """
        B, H, D = 1, 4, 512
        shard_n_heads = H // 2
        head_offset = shard_n_heads
        num_shards = 2
        torch.manual_seed(42)

        partial_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        partial_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]

        merged_v_ref, merged_lse_ref = torch_merge_partial_attn(partial_v_outs, partial_lses)
        merged_v_ref_slice = merged_v_ref[:, head_offset:head_offset + shard_n_heads, :]
        merged_lse_ref_slice = merged_lse_ref[:, head_offset:head_offset + shard_n_heads]

        merged_v_heads = torch.empty(B, shard_n_heads, D, dtype=torch.bfloat16, device=device)
        merged_lse_heads = torch.empty(B, shard_n_heads, dtype=torch.float32, device=device)

        glm.cp_merge_tree(
            [t.data_ptr() for t in partial_v_outs],
            [t.data_ptr() for t in partial_lses],
            num_shards,
            merged_v_heads.data_ptr(),
            merged_lse_heads.data_ptr(),
            B * shard_n_heads * D, B, H, D, shard_n_heads, head_offset, H
        )
        glm.synchronize()

        torch.testing.assert_close(merged_v_heads.cpu(), merged_v_ref_slice.cpu(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(merged_lse_heads.cpu(), merged_lse_ref_slice.cpu(), atol=1e-3, rtol=1e-3)

    def test_heads_nullptr_lse(self, glm, device):
        """Head-grouped merge with merged_lse=nullptr (skip LSE output).

        Uses full-layout v_out with input_n_heads=num_heads because head_offset>0.
        """
        B, H, D = 2, 8, 128
        shard_n_heads = H // 2
        head_offset = H // 2
        num_shards = 2
        torch.manual_seed(42)

        partial_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        partial_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]

        merged_v_ref, _ = torch_merge_partial_attn(partial_v_outs, partial_lses)
        merged_v_ref_slice = merged_v_ref[:, head_offset:head_offset + shard_n_heads, :]

        merged_v_heads = torch.empty(B, shard_n_heads, D, dtype=torch.bfloat16, device=device)

        glm.cp_merge_tree(
            [t.data_ptr() for t in partial_v_outs],
            [t.data_ptr() for t in partial_lses],
            num_shards,
            merged_v_heads.data_ptr(),
            None,
            B * shard_n_heads * D, B, H, D, shard_n_heads, head_offset, H
        )
        glm.synchronize()

        torch.testing.assert_close(merged_v_heads.cpu(), merged_v_ref_slice.cpu(), atol=1e-3, rtol=1e-3)

    def test_heads_glm51_like(self, glm, device):
        """GLM-5.1-like config: 128 heads, 8-way TP (16 heads each), 2 CP shards, v_head_dim=128.

        Uses full-layout v_out with input_n_heads=num_heads.
        """
        B, H, D = 1, 128, 128
        tp_size = 8
        shard_n_heads = H // tp_size
        num_shards = 2
        torch.manual_seed(42)

        partial_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        partial_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]

        merged_v_ref, merged_lse_ref = torch_merge_partial_attn(partial_v_outs, partial_lses)

        for tp_rank in range(tp_size):
            offset = tp_rank * shard_n_heads
            merged_v = torch.empty(B, shard_n_heads, D, dtype=torch.bfloat16, device=device)
            merged_lse = torch.empty(B, shard_n_heads, dtype=torch.float32, device=device)
            glm.cp_merge_tree(
                [t.data_ptr() for t in partial_v_outs],
                [t.data_ptr() for t in partial_lses],
                num_shards,
                merged_v.data_ptr(),
                merged_lse.data_ptr(),
                B * shard_n_heads * D, B, H, D, shard_n_heads, offset, H
            )
            glm.synchronize()
            torch.testing.assert_close(
                merged_v.cpu(), merged_v_ref[:, offset:offset + shard_n_heads, :].cpu(),
                atol=1e-3, rtol=1e-3)
            torch.testing.assert_close(
                merged_lse.cpu(), merged_lse_ref[:, offset:offset + shard_n_heads].cpu(),
                atol=1e-3, rtol=1e-3)

    def test_heads_non_divisible_batch(self, glm, device):
        """Head-grouped merge with batch_size > 1 and non-zero offset.

        Uses full-layout v_out with input_n_heads=num_heads because head_offset>0.
        """
        B, H, D = 3, 6, 128
        shard_n_heads = 2
        head_offset = 2
        num_shards = 4
        torch.manual_seed(42)

        partial_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        partial_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]

        merged_v_ref, merged_lse_ref = torch_merge_partial_attn(partial_v_outs, partial_lses)
        merged_v_ref_slice = merged_v_ref[:, head_offset:head_offset + shard_n_heads, :]
        merged_lse_ref_slice = merged_lse_ref[:, head_offset:head_offset + shard_n_heads]

        merged_v_heads = torch.empty(B, shard_n_heads, D, dtype=torch.bfloat16, device=device)
        merged_lse_heads = torch.empty(B, shard_n_heads, dtype=torch.float32, device=device)

        glm.cp_merge_tree(
            [t.data_ptr() for t in partial_v_outs],
            [t.data_ptr() for t in partial_lses],
            num_shards,
            merged_v_heads.data_ptr(),
            merged_lse_heads.data_ptr(),
            B * shard_n_heads * D, B, H, D, shard_n_heads, head_offset, H
        )
        glm.synchronize()

        torch.testing.assert_close(merged_v_heads.cpu(), merged_v_ref_slice.cpu(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(merged_lse_heads.cpu(), merged_lse_ref_slice.cpu(), atol=1e-3, rtol=1e-3)

    def test_heads_commutativity(self, glm, device):
        """Head-grouped merge should be consistent regardless of which heads are merged first.

        Uses full-layout v_out with input_n_heads=num_heads.
        """
        B, H, D = 2, 8, 128
        shard_n_heads = H // 2
        num_shards = 2
        torch.manual_seed(42)

        partial_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        partial_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]

        merged_v0 = torch.empty(B, shard_n_heads, D, dtype=torch.bfloat16, device=device)
        merged_v1 = torch.empty(B, shard_n_heads, D, dtype=torch.bfloat16, device=device)

        glm.cp_merge_tree(
            [t.data_ptr() for t in partial_v_outs],
            [t.data_ptr() for t in partial_lses],
            num_shards,
            merged_v0.data_ptr(), None,
            B * shard_n_heads * D, B, H, D, shard_n_heads, 0, H
        )
        glm.cp_merge_tree(
            [t.data_ptr() for t in partial_v_outs],
            [t.data_ptr() for t in partial_lses],
            num_shards,
            merged_v1.data_ptr(), None,
            B * shard_n_heads * D, B, H, D, shard_n_heads, shard_n_heads, H
        )
        glm.synchronize()

        full_v_ref, _ = torch_merge_partial_attn(partial_v_outs, partial_lses)
        torch.testing.assert_close(merged_v0.cpu(), full_v_ref[:, :shard_n_heads, :].cpu(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(merged_v1.cpu(), full_v_ref[:, shard_n_heads:, :].cpu(), atol=1e-3, rtol=1e-3)

    def test_heads_1_shard_identity(self, glm, device):
        """1-shard head-grouped merge should return the input unchanged.

        Uses full-layout v_out with input_n_heads=num_heads because head_offset>0.
        """
        B, H, D = 2, 8, 128
        shard_n_heads = H // 2
        head_offset = shard_n_heads
        torch.manual_seed(42)

        v = torch.randn(B, H, D, dtype=torch.bfloat16, device=device)
        lse = torch.randn(B, H, dtype=torch.float32, device=device)

        merged_v_heads = torch.empty(B, shard_n_heads, D, dtype=torch.bfloat16, device=device)
        merged_lse_heads = torch.empty(B, shard_n_heads, dtype=torch.float32, device=device)

        glm.cp_merge_tree(
            [v.data_ptr()],
            [lse.data_ptr()],
            1,
            merged_v_heads.data_ptr(),
            merged_lse_heads.data_ptr(),
            B * shard_n_heads * D, B, H, D, shard_n_heads, head_offset, H
        )
        glm.synchronize()

        expected_v = v[:, head_offset:head_offset + shard_n_heads, :]
        expected_lse = lse[:, head_offset:head_offset + shard_n_heads]

        torch.testing.assert_close(merged_v_heads.cpu(), expected_v.cpu(), atol=0, rtol=0)
        torch.testing.assert_close(merged_lse_heads.cpu(), expected_lse.cpu(), atol=1e-6, rtol=1e-6)

    def test_heads_head_dim_32(self, glm, device):
        """Head-grouped merge with v_head_dim=32.

        Uses full-layout v_out with input_n_heads=num_heads because head_offset>0.
        """
        B, H, D = 2, 4, 32
        shard_n_heads = 2
        head_offset = 1
        num_shards = 2
        torch.manual_seed(42)

        partial_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        partial_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]

        merged_v_ref, merged_lse_ref = torch_merge_partial_attn(partial_v_outs, partial_lses)
        merged_v_ref_slice = merged_v_ref[:, head_offset:head_offset + shard_n_heads, :]
        merged_lse_ref_slice = merged_lse_ref[:, head_offset:head_offset + shard_n_heads]

        merged_v_heads = torch.empty(B, shard_n_heads, D, dtype=torch.bfloat16, device=device)
        merged_lse_heads = torch.empty(B, shard_n_heads, dtype=torch.float32, device=device)

        glm.cp_merge_tree(
            [t.data_ptr() for t in partial_v_outs],
            [t.data_ptr() for t in partial_lses],
            num_shards,
            merged_v_heads.data_ptr(),
            merged_lse_heads.data_ptr(),
            B * shard_n_heads * D, B, H, D, shard_n_heads, head_offset, H
        )
        glm.synchronize()

        torch.testing.assert_close(merged_v_heads.cpu(), merged_v_ref_slice.cpu(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(merged_lse_heads.cpu(), merged_lse_ref_slice.cpu(), atol=1e-3, rtol=1e-3)

    def test_heads_head_dim_64(self, glm, device):
        """Head-grouped merge with v_head_dim=64.

        Uses full-layout v_out with input_n_heads=num_heads because head_offset>0.
        """
        B, H, D = 2, 8, 64
        shard_n_heads = 4
        head_offset = 2
        num_shards = 2
        torch.manual_seed(42)

        partial_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        partial_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]

        merged_v_ref, merged_lse_ref = torch_merge_partial_attn(partial_v_outs, partial_lses)
        merged_v_ref_slice = merged_v_ref[:, head_offset:head_offset + shard_n_heads, :]
        merged_lse_ref_slice = merged_lse_ref[:, head_offset:head_offset + shard_n_heads]

        merged_v_heads = torch.empty(B, shard_n_heads, D, dtype=torch.bfloat16, device=device)
        merged_lse_heads = torch.empty(B, shard_n_heads, dtype=torch.float32, device=device)

        glm.cp_merge_tree(
            [t.data_ptr() for t in partial_v_outs],
            [t.data_ptr() for t in partial_lses],
            num_shards,
            merged_v_heads.data_ptr(),
            merged_lse_heads.data_ptr(),
            B * shard_n_heads * D, B, H, D, shard_n_heads, head_offset, H
        )
        glm.synchronize()

        torch.testing.assert_close(merged_v_heads.cpu(), merged_v_ref_slice.cpu(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(merged_lse_heads.cpu(), merged_lse_ref_slice.cpu(), atol=1e-3, rtol=1e-3)

    def test_heads_output_contiguous_column_layout(self, glm, device):
        """Verify output is contiguous [B, shard_n_heads, D] (Row-parallel layout).

        Uses shard-layout v_out with head_offset=0 and input_n_heads=shard_n_heads.
        """
        B, H, D = 2, 8, 128
        shard_n_heads = H // 2
        head_offset = 0
        num_shards = 2
        torch.manual_seed(42)

        partial_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        partial_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]

        shard_v = shard_v_outs_from_full(partial_v_outs, head_offset, shard_n_heads)
        merged_v_heads = torch.empty(B, shard_n_heads, D, dtype=torch.bfloat16, device=device)

        glm.cp_merge_tree(
            [t.data_ptr() for t in shard_v],
            [t.data_ptr() for t in partial_lses],
            num_shards,
            merged_v_heads.data_ptr(),
            None,
            B * shard_n_heads * D, B, H, D, shard_n_heads, head_offset, shard_n_heads
        )
        glm.synchronize()

        assert merged_v_heads.is_contiguous(), "Output should be contiguous"
        assert merged_v_heads.shape == (B, shard_n_heads, D), \
            f"Expected shape ({B}, {shard_n_heads}, {D}), got {merged_v_heads.shape}"
        assert merged_v_heads.stride() == (shard_n_heads * D, D, 1), \
            f"Expected stride ({shard_n_heads * D}, {D}, 1), got {merged_v_heads.stride()}"
