"""Test cp_merge_tree with shard-layout v_out input and per-head merge.

When input_n_heads=shard_n_heads (column-parallel v_proj output), each shard's
v_out is [B, shard_n_heads, D]. This is only valid when head_offset=0 because
the kernel indexes v_out as (b * input_n_heads + local_h + head_offset) * D.

For head_offset > 0, full-layout v_out with input_n_heads=num_heads must be used.
"""
import torch
import pytest
from helpers import GlmOps


def torch_merge_partial_attn_shard_layout(partial_v_outs_shard, partial_lses_full,
                                           num_heads, shard_n_heads, head_offset):
    """Reference merge for shard-layout v_out + full-layout lse.

    Args:
        partial_v_outs_shard: list of [B, shard_n_heads, D] BF16 tensors (shard layout)
        partial_lses_full: list of [B, num_heads] F32 tensors (full layout)
        num_heads: total number of heads
        shard_n_heads: heads per shard
        head_offset: which head group this merge targets (must be 0 for shard layout)

    Returns:
        merged_v: [B, shard_n_heads, D] BF16
        merged_lse: [B, shard_n_heads] F32
    """
    num_shards = len(partial_v_outs_shard)
    B = partial_v_outs_shard[0].shape[0]
    D = partial_v_outs_shard[0].shape[2]
    device = partial_v_outs_shard[0].device

    merged_o = torch.zeros(B, shard_n_heads, D, dtype=torch.float32, device=device)
    merged_m = torch.full((B, shard_n_heads), float('-inf'), dtype=torch.float32, device=device)
    merged_d = torch.ones(B, shard_n_heads, dtype=torch.float32, device=device)

    for s in range(num_shards):
        v = partial_v_outs_shard[s].float()
        lse = partial_lses_full[s][:, head_offset:head_offset + shard_n_heads]
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
    """Reference merge for full-layout v_out + full-layout lse.

    Args:
        partial_v_outs: list of [B, num_heads, D] BF16 tensors (full layout)
        partial_lses: list of [B, num_heads] F32 tensors (full layout)
        num_heads: total number of heads
        shard_n_heads: heads per shard
        head_offset: which head group this merge targets

    Returns:
        merged_v: [B, shard_n_heads, D] BF16
        merged_lse: [B, shard_n_heads] F32
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestCpMergeShardLayout:

    def test_shard_layout_2_shards_first_half(self, glm, device):
        """Shard-layout v_out [B, snh, D] with head_offset=0, 2 CP shards.

        input_n_heads=shard_n_heads is valid when head_offset=0.
        """
        B, H, D = 2, 8, 128
        snh = H // 2
        head_offset = 0
        num_shards = 2
        torch.manual_seed(42)

        full_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        full_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]

        shard_v_outs = [v[:, head_offset:head_offset + snh, :].contiguous() for v in full_v_outs]

        merged_v_ref, merged_lse_ref = torch_merge_partial_attn_shard_layout(
            shard_v_outs, full_lses, H, snh, head_offset)

        merged_v = torch.empty(B, snh, D, dtype=torch.bfloat16, device=device)
        merged_lse = torch.empty(B, snh, dtype=torch.float32, device=device)

        glm.cp_merge_tree(
            [t.data_ptr() for t in shard_v_outs],
            [t.data_ptr() for t in full_lses],
            num_shards,
            merged_v.data_ptr(),
            merged_lse.data_ptr(),
            B * snh * D, B, H, D,
            snh, head_offset, snh
        )
        glm.synchronize()

        torch.testing.assert_close(merged_v.cpu(), merged_v_ref.cpu(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(merged_lse.cpu(), merged_lse_ref.cpu(), atol=1e-3, rtol=1e-3)

    def test_shard_layout_2_shards_second_half(self, glm, device):
        """Full-layout v_out [B, H, D] with head_offset=H/2, 2 CP shards.

        head_offset > 0 requires full-layout v_out with input_n_heads=num_heads.
        """
        B, H, D = 2, 8, 128
        snh = H // 2
        head_offset = H // 2
        num_shards = 2
        torch.manual_seed(123)

        full_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        full_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]

        merged_v_ref, merged_lse_ref = torch_merge_partial_attn_heads(
            full_v_outs, full_lses, H, snh, head_offset)

        merged_v = torch.empty(B, snh, D, dtype=torch.bfloat16, device=device)
        merged_lse = torch.empty(B, snh, dtype=torch.float32, device=device)

        glm.cp_merge_tree(
            [t.data_ptr() for t in full_v_outs],
            [t.data_ptr() for t in full_lses],
            num_shards,
            merged_v.data_ptr(),
            merged_lse.data_ptr(),
            B * snh * D, B, H, D,
            snh, head_offset, H
        )
        glm.synchronize()

        torch.testing.assert_close(merged_v.cpu(), merged_v_ref.cpu(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(merged_lse.cpu(), merged_lse_ref.cpu(), atol=1e-3, rtol=1e-3)

    def test_shard_layout_4_shards(self, glm, device):
        """Full-layout v_out with 4 CP shards, head_offset=H/2.

        head_offset > 0 requires full-layout v_out with input_n_heads=num_heads.
        """
        B, H, D = 2, 8, 128
        snh = H // 2
        head_offset = snh
        num_shards = 4
        torch.manual_seed(77)

        full_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        full_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]

        merged_v_ref, merged_lse_ref = torch_merge_partial_attn_heads(
            full_v_outs, full_lses, H, snh, head_offset)

        merged_v = torch.empty(B, snh, D, dtype=torch.bfloat16, device=device)
        merged_lse = torch.empty(B, snh, dtype=torch.float32, device=device)

        glm.cp_merge_tree(
            [t.data_ptr() for t in full_v_outs],
            [t.data_ptr() for t in full_lses],
            num_shards,
            merged_v.data_ptr(),
            merged_lse.data_ptr(),
            B * snh * D, B, H, D,
            snh, head_offset, H
        )
        glm.synchronize()

        torch.testing.assert_close(merged_v.cpu(), merged_v_ref.cpu(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(merged_lse.cpu(), merged_lse_ref.cpu(), atol=1e-3, rtol=1e-3)

    def test_shard_layout_tp_reconstruct(self, glm, device):
        """2-way TP: merge both head groups from full-layout v_out, cat = full merge.

        Uses full-layout v_out with input_n_heads=num_heads for head_offset > 0.
        """
        B, H, D = 2, 8, 128
        tp_size = 2
        snh = H // tp_size
        num_shards = 2
        torch.manual_seed(99)

        full_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        full_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]

        full_v_ref, full_lse_ref = torch_merge_partial_attn_heads(
            full_v_outs, full_lses, H, H, 0)

        merged_v_shards = []
        merged_lse_shards = []
        for tp_rank in range(tp_size):
            offset = tp_rank * snh

            merged_v = torch.empty(B, snh, D, dtype=torch.bfloat16, device=device)
            merged_lse = torch.empty(B, snh, dtype=torch.float32, device=device)
            glm.cp_merge_tree(
                [t.data_ptr() for t in full_v_outs],
                [t.data_ptr() for t in full_lses],
                num_shards,
                merged_v.data_ptr(),
                merged_lse.data_ptr(),
                B * snh * D, B, H, D,
                snh, offset, H
            )
            merged_v_shards.append(merged_v)
            merged_lse_shards.append(merged_lse)

        glm.synchronize()

        reconstructed_v = torch.cat(merged_v_shards, dim=1)
        reconstructed_lse = torch.cat(merged_lse_shards, dim=1)

        torch.testing.assert_close(reconstructed_v.cpu(), full_v_ref.cpu(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(reconstructed_lse.cpu(), full_lse_ref.cpu(), atol=1e-3, rtol=1e-3)

    def test_shard_layout_glm51_like(self, glm, device):
        """GLM-5.1 config: 128 heads, 8-way TP (16 heads each), 2 CP shards, D=128.

        Uses full-layout v_out with input_n_heads=num_heads.
        """
        B, H, D = 1, 128, 128
        tp_size = 8
        snh = H // tp_size
        num_shards = 2
        torch.manual_seed(42)

        full_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        full_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]

        full_v_ref, full_lse_ref = torch_merge_partial_attn_heads(
            full_v_outs, full_lses, H, H, 0)

        for tp_rank in range(tp_size):
            offset = tp_rank * snh

            merged_v = torch.empty(B, snh, D, dtype=torch.bfloat16, device=device)
            merged_lse = torch.empty(B, snh, dtype=torch.float32, device=device)
            glm.cp_merge_tree(
                [t.data_ptr() for t in full_v_outs],
                [t.data_ptr() for t in full_lses],
                num_shards,
                merged_v.data_ptr(),
                merged_lse.data_ptr(),
                B * snh * D, B, H, D,
                snh, offset, H
            )
            glm.synchronize()
            torch.testing.assert_close(
                merged_v.cpu(), full_v_ref[:, offset:offset + snh, :].cpu(),
                atol=1e-3, rtol=1e-3)
            torch.testing.assert_close(
                merged_lse.cpu(), full_lse_ref[:, offset:offset + snh].cpu(),
                atol=1e-3, rtol=1e-3)

    def test_shard_layout_batch_gt1(self, glm, device):
        """Full-layout v_out with B=3 and head_offset=2.

        head_offset > 0 requires full-layout v_out with input_n_heads=num_heads.
        """
        B, H, D = 3, 6, 128
        snh = 2
        head_offset = 2
        num_shards = 2
        torch.manual_seed(42)

        full_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        full_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]

        merged_v_ref, merged_lse_ref = torch_merge_partial_attn_heads(
            full_v_outs, full_lses, H, snh, head_offset)

        merged_v = torch.empty(B, snh, D, dtype=torch.bfloat16, device=device)
        merged_lse = torch.empty(B, snh, dtype=torch.float32, device=device)

        glm.cp_merge_tree(
            [t.data_ptr() for t in full_v_outs],
            [t.data_ptr() for t in full_lses],
            num_shards,
            merged_v.data_ptr(),
            merged_lse.data_ptr(),
            B * snh * D, B, H, D,
            snh, head_offset, H
        )
        glm.synchronize()

        torch.testing.assert_close(merged_v.cpu(), merged_v_ref.cpu(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(merged_lse.cpu(), merged_lse_ref.cpu(), atol=1e-3, rtol=1e-3)

    def test_shard_layout_head_dim_512(self, glm, device):
        """Full-layout v_out with v_head_dim=512 (GLM-5.1 kvLoraRank).

        head_offset > 0 requires full-layout v_out with input_n_heads=num_heads.
        """
        B, H, D = 1, 4, 512
        snh = H // 2
        head_offset = snh
        num_shards = 2
        torch.manual_seed(42)

        full_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        full_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]

        merged_v_ref, merged_lse_ref = torch_merge_partial_attn_heads(
            full_v_outs, full_lses, H, snh, head_offset)

        merged_v = torch.empty(B, snh, D, dtype=torch.bfloat16, device=device)
        merged_lse = torch.empty(B, snh, dtype=torch.float32, device=device)

        glm.cp_merge_tree(
            [t.data_ptr() for t in full_v_outs],
            [t.data_ptr() for t in full_lses],
            num_shards,
            merged_v.data_ptr(),
            merged_lse.data_ptr(),
            B * snh * D, B, H, D,
            snh, head_offset, H
        )
        glm.synchronize()

        torch.testing.assert_close(merged_v.cpu(), merged_v_ref.cpu(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(merged_lse.cpu(), merged_lse_ref.cpu(), atol=1e-3, rtol=1e-3)

    def test_shard_layout_1_shard_identity(self, glm, device):
        """1-shard full-layout merge should return the input slice unchanged.

        Uses full-layout v_out with input_n_heads=num_heads because head_offset > 0.
        """
        B, H, D = 2, 8, 128
        snh = H // 2
        head_offset = snh
        torch.manual_seed(42)

        full_v = torch.randn(B, H, D, dtype=torch.bfloat16, device=device)
        full_lse = torch.randn(B, H, dtype=torch.float32, device=device)

        merged_v = torch.empty(B, snh, D, dtype=torch.bfloat16, device=device)
        merged_lse = torch.empty(B, snh, dtype=torch.float32, device=device)

        glm.cp_merge_tree(
            [full_v.data_ptr()],
            [full_lse.data_ptr()],
            1,
            merged_v.data_ptr(),
            merged_lse.data_ptr(),
            B * snh * D, B, H, D,
            snh, head_offset, H
        )
        glm.synchronize()

        expected_v = full_v[:, head_offset:head_offset + snh, :]
        expected_lse = full_lse[:, head_offset:head_offset + snh]

        torch.testing.assert_close(merged_v.cpu(), expected_v.cpu(), atol=0, rtol=0)
        torch.testing.assert_close(merged_lse.cpu(), expected_lse.cpu(), atol=1e-6, rtol=1e-6)

    def test_shard_layout_matches_reference(self, glm, device):
        """Verify kernel output matches PyTorch reference for head_offset > 0.

        Uses full-layout v_out with input_n_heads=num_heads because head_offset > 0.
        """
        B, H, D = 2, 8, 128
        snh = H // 2
        head_offset = snh
        num_shards = 2
        torch.manual_seed(42)

        full_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        full_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]

        merged_v_ref, merged_lse_ref = torch_merge_partial_attn_heads(
            full_v_outs, full_lses, H, snh, head_offset)

        merged_v = torch.empty(B, snh, D, dtype=torch.bfloat16, device=device)
        merged_lse = torch.empty(B, snh, dtype=torch.float32, device=device)

        glm.cp_merge_tree(
            [t.data_ptr() for t in full_v_outs],
            [t.data_ptr() for t in full_lses],
            num_shards,
            merged_v.data_ptr(),
            merged_lse.data_ptr(),
            B * snh * D, B, H, D,
            snh, head_offset, H
        )
        glm.synchronize()

        torch.testing.assert_close(merged_v.cpu(), merged_v_ref.cpu(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(merged_lse.cpu(), merged_lse_ref.cpu(), atol=1e-3, rtol=1e-3)

    def test_shard_layout_shard_v_offset0(self, glm, device):
        """Shard-layout v_out [B, snh, D] with head_offset=0 verifies input_n_heads=shard_n_heads.

        This tests the column-parallel v_proj output path where each shard's v_out
        contains only snh heads (no head_offset needed in v_out indexing).
        """
        B, H, D = 2, 8, 128
        snh = H // 2
        head_offset = 0
        num_shards = 2
        torch.manual_seed(42)

        full_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        full_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]

        shard_v_outs = [v[:, head_offset:head_offset + snh, :].contiguous() for v in full_v_outs]

        merged_v_ref, merged_lse_ref = torch_merge_partial_attn_shard_layout(
            shard_v_outs, full_lses, H, snh, head_offset)

        merged_v = torch.empty(B, snh, D, dtype=torch.bfloat16, device=device)
        merged_lse = torch.empty(B, snh, dtype=torch.float32, device=device)

        glm.cp_merge_tree(
            [t.data_ptr() for t in shard_v_outs],
            [t.data_ptr() for t in full_lses],
            num_shards,
            merged_v.data_ptr(),
            merged_lse.data_ptr(),
            B * snh * D, B, H, D,
            snh, head_offset, snh
        )
        glm.synchronize()

        torch.testing.assert_close(merged_v.cpu(), merged_v_ref.cpu(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(merged_lse.cpu(), merged_lse_ref.cpu(), atol=1e-3, rtol=1e-3)
