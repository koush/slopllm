import torch
import pytest
import math
from helpers import ATOL, RTOL, GlmOps


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
    
    # Initialize state: o=0, m=-inf, d=1
    merged_o = torch.zeros(B, H, D, dtype=torch.float32, device=device)
    merged_m = torch.full((B, H), float('-inf'), dtype=torch.float32, device=device)
    merged_d = torch.ones(B, H, dtype=torch.float32, device=device)
    
    for s in range(num_shards):
        v = partial_v_outs[s].float()  # [B, H, D]
        lse = partial_lses[s]          # [B, H]
        m_prev = merged_m.clone()
        d_prev = merged_d.clone()
        merged_m = torch.max(merged_m, lse)
        merged_d = d_prev * torch.exp2(m_prev - merged_m) + torch.exp2(lse - merged_m)
        merged_o = merged_o * torch.exp2(m_prev - merged_m).unsqueeze(-1) + \
                   v * torch.exp2(lse - merged_m).unsqueeze(-1)
    
    # Normalize
    merged_o = merged_o / merged_d.unsqueeze(-1)
    merged_lse = merged_m + torch.log2(merged_d)
    
    return merged_o.bfloat16(), merged_lse


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestContextParallelMerge:
    def test_merge_2_shards(self, glm, device):
        B, H, D = 2, 4, 128
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

    def test_merge_4_shards(self, glm, device):
        B, H, D = 2, 4, 128
        num_shards = 4
        torch.manual_seed(123)
        
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

    def test_merge_8_shards(self, glm, device):
        B, H, D = 1, 8, 128
        num_shards = 8
        torch.manual_seed(99)
        
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

    def test_merge_8_shards(self, glm, device):
        B, H, D = 1, 4, 128
        num_shards = 8
        torch.manual_seed(77)
        
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

    def test_merge_1_shard_identity(self, glm, device):
        """Merging 1 shard should return the input unchanged."""
        B, H, D = 2, 4, 128
        torch.manual_seed(42)
        
        v = torch.randn(B, H, D, dtype=torch.bfloat16, device=device)
        lse = torch.randn(B, H, dtype=torch.float32, device=device)
        
        merged_v_out = torch.empty(B, H, D, dtype=torch.bfloat16, device=device)
        merged_lse_out = torch.empty(B, H, dtype=torch.float32, device=device)
        
        glm.cp_merge_tree(
            [v.data_ptr()],
            [lse.data_ptr()],
            1,
            merged_v_out.data_ptr(),
            merged_lse_out.data_ptr(),
            B * H * D, B, H, D
        )
        glm.synchronize()
        
        torch.testing.assert_close(merged_v_out.cpu(), v.cpu(), atol=0, rtol=0)
        torch.testing.assert_close(merged_lse_out.cpu(), lse.cpu(), atol=1e-6, rtol=1e-6)

    def test_merge_commutativity(self, glm, device):
        """Merge(a, b) == Merge(b, a)."""
        B, H, D = 2, 4, 128
        torch.manual_seed(42)
        
        v0 = torch.randn(B, H, D, dtype=torch.bfloat16, device=device)
        v1 = torch.randn(B, H, D, dtype=torch.bfloat16, device=device)
        lse0 = torch.randn(B, H, dtype=torch.float32, device=device)
        lse1 = torch.randn(B, H, dtype=torch.float32, device=device)
        
        merged_ab = torch.empty(B, H, D, dtype=torch.bfloat16, device=device)
        merged_ba = torch.empty(B, H, D, dtype=torch.bfloat16, device=device)
        lse_ab = torch.empty(B, H, dtype=torch.float32, device=device)
        lse_ba = torch.empty(B, H, dtype=torch.float32, device=device)
        
        glm.cp_merge_tree(
            [v0.data_ptr(), v1.data_ptr()],
            [lse0.data_ptr(), lse1.data_ptr()],
            2, merged_ab.data_ptr(), lse_ab.data_ptr(),
            B * H * D, B, H, D
        )
        glm.cp_merge_tree(
            [v1.data_ptr(), v0.data_ptr()],
            [lse1.data_ptr(), lse0.data_ptr()],
            2, merged_ba.data_ptr(), lse_ba.data_ptr(),
            B * H * D, B, H, D
        )
        glm.synchronize()
        
        torch.testing.assert_close(merged_ab.cpu(), merged_ba.cpu(), atol=0, rtol=0)
        torch.testing.assert_close(lse_ab.cpu(), lse_ba.cpu(), atol=1e-6, rtol=1e-6)

    def test_merge_associativity(self, glm, device):
        """merge(a, merge(b, c)) == merge(merge(a, b), c)."""
        B, H, D = 1, 4, 128
        torch.manual_seed(42)
        
        v0 = torch.randn(B, H, D, dtype=torch.bfloat16, device=device)
        v1 = torch.randn(B, H, D, dtype=torch.bfloat16, device=device)
        v2 = torch.randn(B, H, D, dtype=torch.bfloat16, device=device)
        lse0 = torch.randn(B, H, dtype=torch.float32, device=device)
        lse1 = torch.randn(B, H, dtype=torch.float32, device=device)
        lse2 = torch.randn(B, H, dtype=torch.float32, device=device)
        
        # merge(a, b) first, then merge with c
        merged_ab_v = torch.empty(B, H, D, dtype=torch.bfloat16, device=device)
        merged_ab_lse = torch.empty(B, H, dtype=torch.float32, device=device)
        glm.cp_merge_tree(
            [v0.data_ptr(), v1.data_ptr()],
            [lse0.data_ptr(), lse1.data_ptr()],
            2, merged_ab_v.data_ptr(), merged_ab_lse.data_ptr(),
            B * H * D, B, H, D
        )
        
        merged_ab_c_v = torch.empty(B, H, D, dtype=torch.bfloat16, device=device)
        merged_ab_c_lse = torch.empty(B, H, dtype=torch.float32, device=device)
        glm.cp_merge_tree(
            [merged_ab_v.data_ptr(), v2.data_ptr()],
            [merged_ab_lse.data_ptr(), lse2.data_ptr()],
            2, merged_ab_c_v.data_ptr(), merged_ab_c_lse.data_ptr(),
            B * H * D, B, H, D
        )
        
        # merge(b, c) first, then merge with a
        merged_bc_v = torch.empty(B, H, D, dtype=torch.bfloat16, device=device)
        merged_bc_lse = torch.empty(B, H, dtype=torch.float32, device=device)
        glm.cp_merge_tree(
            [v1.data_ptr(), v2.data_ptr()],
            [lse1.data_ptr(), lse2.data_ptr()],
            2, merged_bc_v.data_ptr(), merged_bc_lse.data_ptr(),
            B * H * D, B, H, D
        )
        
        merged_a_bc_v = torch.empty(B, H, D, dtype=torch.bfloat16, device=device)
        merged_a_bc_lse = torch.empty(B, H, dtype=torch.float32, device=device)
        glm.cp_merge_tree(
            [v0.data_ptr(), merged_bc_v.data_ptr()],
            [lse0.data_ptr(), merged_bc_lse.data_ptr()],
            2, merged_a_bc_v.data_ptr(), merged_a_bc_lse.data_ptr(),
            B * H * D, B, H, D
        )
        
        glm.synchronize()
        
        # Also verify against 3-shard direct merge
        merged_3_v = torch.empty(B, H, D, dtype=torch.bfloat16, device=device)
        merged_3_lse = torch.empty(B, H, dtype=torch.float32, device=device)
        glm.cp_merge_tree(
            [v0.data_ptr(), v1.data_ptr(), v2.data_ptr()],
            [lse0.data_ptr(), lse1.data_ptr(), lse2.data_ptr()],
            3, merged_3_v.data_ptr(), merged_3_lse.data_ptr(),
            B * H * D, B, H, D
        )
        glm.synchronize()
        
        # Two-step merge compounds BF16 roundtrip errors (BF16->FP32->normalize->BF16->FP32->normalize->BF16)
        torch.testing.assert_close(merged_ab_c_v.cpu(), merged_a_bc_v.cpu(), atol=0.02, rtol=0.05)
        # Compare both against direct 3-shard merge (also BF16 roundtrip)
        torch.testing.assert_close(merged_ab_c_v.cpu(), merged_3_v.cpu(), atol=0.02, rtol=0.05)
        torch.testing.assert_close(merged_ab_c_lse.cpu(), merged_3_lse.cpu(), atol=0.02, rtol=0.05)

    def test_merge_v_expand_linearity(self, glm, device):
        """v_expand(scale * attn) = scale * v_expand(attn).
        Merge-then-expand should equal expand-then-merge."""
        B, H, kv_lora_rank, v_head_dim = 1, 4, 128, 128
        num_shards = 2
        torch.manual_seed(42)
        
        partial_attn_outs = [torch.randn(B, H, kv_lora_rank, dtype=torch.bfloat16, device=device) for _ in range(num_shards)]
        partial_lses = [torch.randn(B, H, dtype=torch.float32, device=device) for _ in range(num_shards)]
        
        # Per-head v_expand matrix: [H, kv_lora_rank, v_head_dim]
        v_proj = torch.randn(H, kv_lora_rank, v_head_dim, dtype=torch.bfloat16, device=device)
        
        # Path 1: merge in kvLoraRank space, then v_expand
        merged_attn_v, merged_attn_lse = torch_merge_partial_attn(partial_attn_outs, partial_lses)
        merged_then_expand = torch.einsum('bhd,hdv->bhv', merged_attn_v.float(), v_proj.float()).bfloat16()
        
        # Path 2: v_expand first, then merge in vHeadDim space
        partial_v_outs = [torch.einsum('bhd,hdv->bhv', a.float(), v_proj.float()).bfloat16() for a in partial_attn_outs]
        expand_then_merge_v, expand_then_merge_lse = torch_merge_partial_attn(partial_v_outs, partial_lses)
        
        # Path 3: CUDA kernel merge in vHeadDim space
        merged_v_cuda = torch.empty(B, H, v_head_dim, dtype=torch.bfloat16, device=device)
        merged_lse_cuda = torch.empty(B, H, dtype=torch.float32, device=device)
        glm.cp_merge_tree(
            [t.data_ptr() for t in partial_v_outs],
            [t.data_ptr() for t in partial_lses],
            num_shards,
            merged_v_cuda.data_ptr(),
            merged_lse_cuda.data_ptr(),
            B * H * v_head_dim, B, H, v_head_dim
        )
        glm.synchronize()
        
        # BF16 roundtrip through different operation orderings compounds rounding errors
        torch.testing.assert_close(merged_then_expand.cpu(), expand_then_merge_v.cpu(), atol=0.15, rtol=0.15)
        torch.testing.assert_close(merged_v_cuda.cpu(), expand_then_merge_v.cpu(), atol=1e-3, rtol=1e-3)

    def test_merge_head_dim_512(self, glm, device):
        """Test with kvLoraRank=512 (merge before v_expand)."""
        B, H, D = 1, 4, 512
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

    def test_merge_nullptr_lse(self, glm, device):
        """Test with merged_lse=nullptr (skip LSE output)."""
        B, H, D = 1, 4, 128
        num_shards = 2
        torch.manual_seed(42)
        
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
