"""Push-based CP merge: cp_merge_scatter + cp_merge_local.

The two kernels are exercised on a single GPU by simulating all `world_size`
ranks' buffers locally: each simulated rank runs the scatter with its own
rank-rotated destination pointers, then merges its own staging slot. That
covers the head slicing, the rank/block rotation, the staging layout and the
online-softmax merge math. The cross-device barrier between the phases is
structural and not modelled here.
"""

import ctypes
import numpy as np
import pytest
import torch

from helpers import GlmOps

SKIP_REASON = f"Requires 1 GPU, found {torch.cuda.device_count()}"


def bf16_bytes_to_tensor(raw_bytes, count):
    u8 = np.frombuffer(raw_bytes[: count * 2], dtype=np.uint8)
    return torch.from_numpy(u8.view(np.uint16).copy()).view(torch.bfloat16)


def read_back(ops, gpu_ptr, nbytes):
    buf = ctypes.create_string_buffer(nbytes)
    ops.d2h(buf, gpu_ptr, nbytes)
    return buf.raw


def zero_fill(ops, gpu_ptr, nbytes):
    ops.h2d(gpu_ptr, b"\x00" * nbytes, nbytes)


def reference_merge(v_outs, lses):
    """Online-softmax merge in float64.

    Each rank's partial v is already normalized within its own KV shard, so the
    merge is a weighted average with weights 2^(lse_i - max_lse).
    """
    v = torch.stack([t.to(torch.float64) for t in v_outs])      # [N, B, H, D]
    lse = torch.stack([t.to(torch.float64) for t in lses])      # [N, B, H]
    m = lse.max(dim=0, keepdim=True).values
    w = torch.exp2(lse - m)                                     # [N, B, H]
    num = (v * w.unsqueeze(-1)).sum(dim=0)
    den = w.sum(dim=0)
    merged_v = num / den.unsqueeze(-1)
    merged_lse = m.squeeze(0) + torch.log2(den)
    return merged_v, merged_lse


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason=SKIP_REASON)
@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("batch,num_heads,v_head_dim", [(1, 8, 128), (3, 8, 128), (2, 16, 512), (4, 32, 64)])
def test_cp_merge_push_matches_reference(world_size, batch, num_heads, v_head_dim):
    if num_heads % world_size != 0:
        pytest.skip("num_heads must be divisible by world_size")

    ops = GlmOps(device_id=0)
    try:
        torch.manual_seed(1234 + world_size * 97 + num_heads)
        shard_n_heads = num_heads // world_size

        # Per-rank full-width partials, as produced by attention on each KV shard.
        v_outs = [torch.randn(batch, num_heads, v_head_dim, dtype=torch.bfloat16) for _ in range(world_size)]
        lses = [torch.randn(batch, num_heads, dtype=torch.float32) * 4.0 for _ in range(world_size)]

        # Upload each rank's partials.
        v_ptrs, lse_ptrs = [], []
        for v, lse in zip(v_outs, lses):
            p = ops.alloc(batch * num_heads * v_head_dim * 2)
            ops.h2d(p, v.view(torch.int16).numpy().tobytes())
            v_ptrs.append(p)
            q = ops.alloc(batch * num_heads * 4)
            ops.h2d(q, lse.numpy().astype(np.float32).tobytes())
            lse_ptrs.append(q)

        # Per-rank staging: [world, batch, shard_n_heads, v_head_dim] + [world, batch, shard_n_heads].
        stage_v_bytes = world_size * batch * shard_n_heads * v_head_dim * 2
        stage_lse_bytes = world_size * batch * shard_n_heads * 4
        stage_v = [ops.alloc(stage_v_bytes) for _ in range(world_size)]
        stage_lse = [ops.alloc(stage_lse_bytes) for _ in range(world_size)]
        for p in stage_v:
            zero_fill(ops, p, stage_v_bytes)
        for p in stage_lse:
            zero_fill(ops, p, stage_lse_bytes)

        # Phase 1: every rank scatters, with destinations rotated by its rank.
        for rank in range(world_size):
            dv = [stage_v[(rank + k) % world_size] for k in range(world_size)]
            dl = [stage_lse[(rank + k) % world_size] for k in range(world_size)]
            ops.cp_merge_scatter(
                v_ptrs[rank], lse_ptrs[rank], dv, dl, world_size,
                batch, shard_n_heads, v_head_dim, num_heads, num_heads, rank,
            )

        # Phase 2: every rank merges its own staging slot.
        out_v_bytes = batch * shard_n_heads * v_head_dim * 2
        out_lse_bytes = batch * shard_n_heads * 4
        out_v = [ops.alloc(out_v_bytes) for _ in range(world_size)]
        out_lse = [ops.alloc(out_lse_bytes) for _ in range(world_size)]
        for rank in range(world_size):
            ops.cp_merge_local(
                stage_v[rank], stage_lse[rank], out_v[rank], out_lse[rank],
                world_size, batch, shard_n_heads, v_head_dim,
            )
        ops.synchronize()

        ref_v, ref_lse = reference_merge(v_outs, lses)

        # Rank r owns heads [r*shard_n_heads, (r+1)*shard_n_heads).
        for rank in range(world_size):
            got_v = bf16_bytes_to_tensor(
                read_back(ops, out_v[rank], out_v_bytes), batch * shard_n_heads * v_head_dim
            ).view(batch, shard_n_heads, v_head_dim).to(torch.float64)
            got_lse = torch.from_numpy(
                np.frombuffer(read_back(ops, out_lse[rank], out_lse_bytes), dtype=np.float32).copy()
            ).view(batch, shard_n_heads).to(torch.float64)

            h0, h1 = rank * shard_n_heads, (rank + 1) * shard_n_heads
            exp_v = ref_v[:, h0:h1, :]
            exp_lse = ref_lse[:, h0:h1]

            torch.testing.assert_close(got_v, exp_v, rtol=3e-2, atol=3e-2,
                                       msg=f"v mismatch on rank {rank}")
            torch.testing.assert_close(got_lse, exp_lse, rtol=1e-5, atol=1e-4,
                                       msg=f"lse mismatch on rank {rank}")
    finally:
        del ops
        torch.cuda.empty_cache()


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason=SKIP_REASON)
def test_cp_merge_push_matches_pull():
    """Push and pull paths must agree; both merge the same N partials."""
    ops = GlmOps(device_id=0)
    try:
        torch.manual_seed(7)
        world_size, batch, num_heads, v_head_dim = 4, 8, 128, 128
        shard_n_heads = num_heads // world_size

        v_outs = [torch.randn(batch, num_heads, v_head_dim, dtype=torch.bfloat16) for _ in range(world_size)]
        lses = [torch.randn(batch, num_heads, dtype=torch.float32) * 3.0 for _ in range(world_size)]

        v_ptrs, lse_ptrs = [], []
        for v, lse in zip(v_outs, lses):
            p = ops.alloc(batch * num_heads * v_head_dim * 2)
            ops.h2d(p, v.view(torch.int16).numpy().tobytes())
            v_ptrs.append(p)
            q = ops.alloc(batch * num_heads * 4)
            ops.h2d(q, lse.numpy().astype(np.float32).tobytes())
            lse_ptrs.append(q)

        out_v_bytes = batch * shard_n_heads * v_head_dim * 2

        # --- pull: each rank reads all peers' partials for its own head slice.
        pull_v = []
        for rank in range(world_size):
            o = ops.alloc(out_v_bytes)
            ops.cp_merge_tree(
                v_ptrs, lse_ptrs, world_size, o, None,
                batch * num_heads * v_head_dim, batch, num_heads, v_head_dim,
                shard_n_heads=shard_n_heads, head_offset=rank * shard_n_heads,
                input_n_heads=num_heads,
            )
            pull_v.append(o)

        # --- push: scatter then merge locally.
        stage_v_bytes = world_size * batch * shard_n_heads * v_head_dim * 2
        stage_lse_bytes = world_size * batch * shard_n_heads * 4
        stage_v = [ops.alloc(stage_v_bytes) for _ in range(world_size)]
        stage_lse = [ops.alloc(stage_lse_bytes) for _ in range(world_size)]
        for rank in range(world_size):
            dv = [stage_v[(rank + k) % world_size] for k in range(world_size)]
            dl = [stage_lse[(rank + k) % world_size] for k in range(world_size)]
            ops.cp_merge_scatter(
                v_ptrs[rank], lse_ptrs[rank], dv, dl, world_size,
                batch, shard_n_heads, v_head_dim, num_heads, num_heads, rank,
            )
        push_v = []
        for rank in range(world_size):
            o = ops.alloc(out_v_bytes)
            ops.cp_merge_local(
                stage_v[rank], stage_lse[rank], o, None,
                world_size, batch, shard_n_heads, v_head_dim,
            )
            push_v.append(o)
        ops.synchronize()

        n = batch * shard_n_heads * v_head_dim
        for rank in range(world_size):
            a = bf16_bytes_to_tensor(read_back(ops, pull_v[rank], out_v_bytes), n).to(torch.float32)
            b = bf16_bytes_to_tensor(read_back(ops, push_v[rank], out_v_bytes), n).to(torch.float32)
            # Both merge in fp32 but in different shard orders, so allow bf16 ulp drift.
            torch.testing.assert_close(a, b, rtol=2e-2, atol=2e-2,
                                       msg=f"push/pull mismatch on rank {rank}")
    finally:
        del ops
        torch.cuda.empty_cache()
