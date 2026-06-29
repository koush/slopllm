"""Tests for the fused rope_transpose_gather kernel.

Simulates multi-GPU Row-parallel AllGather + RoPE + head transpose on a single GPU
by allocating separate input buffers per shard and passing all pointers to the kernel.
"""
import torch
import pytest
import ctypes


def _shard_column_heads(full_data, total_rows, n_heads, head_dim, shard_idx, world_size):
    """Extract one shard's heads from full [total_rows, n_heads * head_dim] data."""
    shard_heads = n_heads // world_size
    result = torch.empty(total_rows, shard_heads * head_dim, dtype=full_data.dtype, device=full_data.device)
    for r in range(total_rows):
        for h in range(shard_heads):
            src_off = r * n_heads * head_dim + (shard_idx * shard_heads + h) * head_dim
            dst_off = r * shard_heads * head_dim + h * head_dim
            result.view(-1)[dst_off:dst_off + head_dim] = full_data.view(-1)[src_off:src_off + head_dim]
    return result


def _gather_rope_transpose_output(shard_outs, batch, n_heads, seq_len, head_dim, world_size):
    """Gather [BS, shardNHeads, head_dim] per shard → [BS, nHeads, head_dim]."""
    shard_heads = n_heads // world_size
    result = torch.empty(batch * seq_len, n_heads, head_dim, dtype=shard_outs[0].dtype, device=shard_outs[0].device)
    for b in range(batch):
        for s in range(seq_len):
            for g in range(world_size):
                for h in range(shard_heads):
                    global_h = g * shard_heads + h
                    src = shard_outs[g]
                    src_idx = (b * seq_len + s) * shard_heads + h
                    dst_idx = (b * seq_len + s) * n_heads + global_h
                    result[dst_idx] = src[src_idx]
    return result


class TestRopeTransposeGather:

    def _run_gather(self, glm, device, batch, n_heads, seq_len, head_dim, rope_dim, world_size, interleaved=False):
        """Run rope_transpose_gather and compare against single-GPU ropeTranspose per-shard + manual gather."""
        total_rows = batch * seq_len
        shard_heads = n_heads // world_size
        in_stride = head_dim

        # Full input (as if all heads were on one GPU)
        x_full = torch.randn(total_rows, n_heads * in_stride, dtype=torch.bfloat16, device=device)

        # Shard the input: each shard gets [total_rows, shard_heads * in_stride]
        shards = [_shard_column_heads(x_full, total_rows, n_heads, head_dim, g, world_size) for g in range(world_size)]

        # cos/sin
        if rope_dim > 0:
            inv_freq = torch.zeros(rope_dim // 2, dtype=torch.bfloat16, device=device)
            for i in range(rope_dim // 2):
                inv_freq[i] = 1.0 / (10000.0 ** (2.0 * i / rope_dim))
            position_ids = torch.arange(seq_len, dtype=torch.int32, device=device).unsqueeze(0).expand(batch, -1).contiguous()
            cos = torch.empty(batch, seq_len, rope_dim, dtype=torch.bfloat16, device=device)
            sin = torch.empty(batch, seq_len, rope_dim, dtype=torch.bfloat16, device=device)
            glm.rotary_embedding(cos, sin, inv_freq, position_ids, rope_dim // 2, batch, seq_len)
        else:
            cos = None
            sin = None

        # Reference: run ropeTranspose on full input (single-GPU equivalent)
        ref_out = torch.empty(total_rows, n_heads, head_dim, dtype=torch.bfloat16, device=device)
        glm.ropeTranspose(ref_out, x_full, cos, sin, rope_dim, head_dim, n_heads, seq_len, batch, in_stride, interleaved)

        # Run rope_transpose_gather: pass all shard pointers
        out = torch.empty(total_rows, n_heads, head_dim, dtype=torch.bfloat16, device=device)
        shard_ptrs = [shard.data_ptr() for shard in shards]
        glm.rope_transpose_gather(
            shard_ptrs, out, cos, sin, world_size,
            rope_dim, head_dim, n_heads, shard_heads, in_stride,
            seq_len, batch, interleaved
        )

        torch.testing.assert_close(out.cpu(), ref_out.cpu(), atol=1e-2, rtol=1e-2)

    def test_gather_no_rope_2_shards(self, glm, device):
        self._run_gather(glm, device, batch=2, n_heads=4, seq_len=3, head_dim=16, rope_dim=0, world_size=2)

    def test_gather_no_rope_4_shards(self, glm, device):
        self._run_gather(glm, device, batch=2, n_heads=8, seq_len=3, head_dim=16, rope_dim=0, world_size=4)

    def test_gather_with_rope_2_shards(self, glm, device):
        self._run_gather(glm, device, batch=1, n_heads=4, seq_len=4, head_dim=16, rope_dim=8, world_size=2)

    def test_gather_with_rope_interleaved_2_shards(self, glm, device):
        self._run_gather(glm, device, batch=1, n_heads=4, seq_len=4, head_dim=16, rope_dim=8, world_size=2, interleaved=True)

    def test_gather_no_rope_large_head_dim(self, glm, device):
        """Simulate qAbsorbedR-like: rope_dim=0, head_dim=512."""
        self._run_gather(glm, device, batch=2, n_heads=8, seq_len=4, head_dim=512, rope_dim=0, world_size=2)

    def test_gather_no_rope_large_head_dim_4_shards(self, glm, device):
        self._run_gather(glm, device, batch=1, n_heads=8, seq_len=2, head_dim=512, rope_dim=0, world_size=4)

    def test_gather_with_rope_large(self, glm, device):
        """Simulate qPeR-like: rope_dim=64, head_dim=64."""
        self._run_gather(glm, device, batch=2, n_heads=8, seq_len=4, head_dim=64, rope_dim=64, world_size=2)

    def test_gather_single_shard(self, glm, device):
        """1 shard: degenerate case, should match ropeTranspose."""
        self._run_gather(glm, device, batch=2, n_heads=4, seq_len=3, head_dim=16, rope_dim=0, world_size=1)
