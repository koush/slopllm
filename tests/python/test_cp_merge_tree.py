import torch
import pytest
import ctypes
import numpy as np
from helpers import GlmOps

NUM_GPUS = min(torch.cuda.device_count(), 2)
SKIP_REASON = f"Requires 1 GPU, found {torch.cuda.device_count()}"


def bf16_bytes_to_tensor(raw_bytes, count):
    u8 = np.frombuffer(raw_bytes[:count * 2], dtype=np.uint8)
    u16 = u8.view(np.uint16)
    return torch.from_numpy(u16.copy()).view(torch.bfloat16)


def f32_bytes_to_tensor(raw_bytes, count):
    return torch.from_numpy(np.frombuffer(raw_bytes[:count * 4], dtype=np.float32).copy())


@pytest.mark.skipif(NUM_GPUS < 1, reason=SKIP_REASON)
class TestCpMergeTree:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ops = GlmOps(device_id=0)
        yield
        del self.ops
        torch.cuda.empty_cache()

    def _cp_merge_tree(self, v_outs_bf16, lses_f32, num_heads, v_head_dim,
                       shard_n_heads=None, head_offset=0, input_n_heads=None):
        N = len(v_outs_bf16)
        batch = v_outs_bf16[0].shape[0]
        if shard_n_heads is None:
            shard_n_heads = num_heads
        if input_n_heads is None:
            input_n_heads = shard_n_heads if head_offset == 0 else num_heads
        out_numel = batch * shard_n_heads * v_head_dim
        in_numel = batch * input_n_heads * v_head_dim

        gpu_v_ptrs = []
        gpu_lse_ptrs = []
        for v, lse in zip(v_outs_bf16, lses_f32):
            v_ptr = self.ops.alloc(in_numel * 2)
            cpu_v = v.cpu().view(torch.int16).numpy().tobytes()
            self.ops.h2d(v_ptr, cpu_v)
            gpu_v_ptrs.append(v_ptr)

            lse_ptr = self.ops.alloc(batch * num_heads * 4)
            cpu_lse = lse.cpu().numpy().astype(np.float32).tobytes()
            self.ops.h2d(lse_ptr, cpu_lse)
            gpu_lse_ptrs.append(lse_ptr)

        out_v_ptr = self.ops.alloc(out_numel * 2)
        out_lse_ptr = self.ops.alloc(batch * shard_n_heads * 4)

        v_args = [ctypes.c_void_p(int(p)) for p in gpu_v_ptrs] + [ctypes.c_void_p(0)] * (8 - N)
        lse_args = [ctypes.c_void_p(int(p)) for p in gpu_lse_ptrs] + [ctypes.c_void_p(0)] * (8 - N)

        self.ops.lib.glm_cp_merge_tree.restype = None
        self.ops.lib.glm_cp_merge_tree.argtypes = (
            [ctypes.c_void_p] +
            [ctypes.c_void_p] * 8 +
            [ctypes.c_void_p] * 8 +
            [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int,
             ctypes.c_int, ctypes.c_int, ctypes.c_int]
        )
        self.ops.lib.glm_cp_merge_tree(
            self.ops.ctx,
            *v_args,
            *lse_args,
            N,
            ctypes.c_void_p(int(out_v_ptr)),
            ctypes.c_void_p(int(out_lse_ptr)),
            out_numel, batch, num_heads, v_head_dim,
            shard_n_heads, head_offset, input_n_heads,
        )

        self.ops.synchronize()

        out_v_bytes = ctypes.create_string_buffer(out_numel * 2)
        self.ops.d2h(out_v_bytes, out_v_ptr, out_numel * 2)
        out_v = bf16_bytes_to_tensor(out_v_bytes.raw, out_numel).reshape(batch, shard_n_heads * v_head_dim)

        out_lse_bytes = ctypes.create_string_buffer(batch * shard_n_heads * 4)
        self.ops.d2h(out_lse_bytes, out_lse_ptr, batch * shard_n_heads * 4)
        out_lse = f32_bytes_to_tensor(out_lse_bytes.raw, batch * shard_n_heads).reshape(batch, shard_n_heads)

        for p in gpu_v_ptrs + gpu_lse_ptrs:
            self.ops.free_buf(p)
        self.ops.free_buf(out_v_ptr)
        self.ops.free_buf(out_lse_ptr)

        return out_v, out_lse

    def _reference_merge(self, v_outs, lses, num_heads, v_head_dim):
        batch = v_outs[0].shape[0]
        max_lse = torch.stack([lse for lse in lses], dim=0).max(dim=0).values
        d = torch.stack([torch.exp2(lse - max_lse) for lse in lses], dim=0).sum(dim=0)
        merged_v = torch.zeros(batch, num_heads, v_head_dim, dtype=torch.float32)
        for v, lse in zip(v_outs, lses):
            v_3d = v.float().reshape(batch, num_heads, v_head_dim)
            scale = torch.exp2(lse - max_lse).unsqueeze(-1)
            merged_v += v_3d * scale
        merged_v = merged_v / d.unsqueeze(-1)
        merged_lse = max_lse + torch.log2(d)
        return merged_v.reshape(batch, num_heads * v_head_dim), merged_lse

    def test_two_shards(self):
        torch.manual_seed(42)
        batch, num_heads, v_head_dim = 4, 8, 64
        v0 = torch.randn(batch, num_heads * v_head_dim, dtype=torch.bfloat16)
        v1 = torch.randn(batch, num_heads * v_head_dim, dtype=torch.bfloat16)
        lse0 = torch.randn(batch, num_heads, dtype=torch.float32)
        lse1 = torch.randn(batch, num_heads, dtype=torch.float32)

        merged_v, merged_lse = self._cp_merge_tree(
            [v0, v1], [lse0, lse1], num_heads, v_head_dim)

        ref_v, ref_lse = self._reference_merge(
            [v0.float(), v1.float()], [lse0, lse1], num_heads, v_head_dim)

        assert torch.allclose(merged_v.float(), ref_v.bfloat16().float(), atol=1e-2, rtol=1e-2), \
            f"v_out max err = {(merged_v.float() - ref_v.bfloat16().float()).abs().max():.6f}"
        assert torch.allclose(merged_lse, ref_lse, atol=1e-3, rtol=1e-3), \
            f"lse max err = {(merged_lse - ref_lse).abs().max():.6f}"

    def test_four_shards(self):
        torch.manual_seed(42)
        batch, num_heads, v_head_dim = 2, 4, 128
        shards_v = [torch.randn(batch, num_heads * v_head_dim, dtype=torch.bfloat16) for _ in range(4)]
        shards_lse = [torch.randn(batch, num_heads, dtype=torch.float32) for _ in range(4)]

        merged_v, merged_lse = self._cp_merge_tree(shards_v, shards_lse, num_heads, v_head_dim)

        ref_v, ref_lse = self._reference_merge(
            [v.float() for v in shards_v], shards_lse, num_heads, v_head_dim)

        assert torch.allclose(merged_v.float(), ref_v.bfloat16().float(), atol=1e-2, rtol=1e-2), \
            f"v_out max err = {(merged_v.float() - ref_v.bfloat16().float()).abs().max():.6f}"
        assert torch.allclose(merged_lse, ref_lse, atol=1e-3, rtol=1e-3), \
            f"lse max err = {(merged_lse - ref_lse).abs().max():.6f}"

    def test_single_shard(self):
        torch.manual_seed(42)
        batch, num_heads, v_head_dim = 1, 2, 256
        v0 = torch.randn(batch, num_heads * v_head_dim, dtype=torch.bfloat16)
        lse0 = torch.randn(batch, num_heads, dtype=torch.float32)

        merged_v, merged_lse = self._cp_merge_tree([v0], [lse0], num_heads, v_head_dim)

        ref_v, ref_lse = self._reference_merge([v0.float()], [lse0], num_heads, v_head_dim)

        assert torch.allclose(merged_v.float(), ref_v.bfloat16().float(), atol=1e-2, rtol=1e-2), \
            f"v_out max err = {(merged_v.float() - ref_v.bfloat16().float()).abs().max():.6f}"

    def test_in_place(self):
        torch.manual_seed(42)
        batch, num_heads, v_head_dim = 2, 4, 64
        v0 = torch.randn(batch, num_heads * v_head_dim, dtype=torch.bfloat16)
        v1 = torch.randn(batch, num_heads * v_head_dim, dtype=torch.bfloat16)
        lse0 = torch.randn(batch, num_heads, dtype=torch.float32)
        lse1 = torch.randn(batch, num_heads, dtype=torch.float32)

        merged_v, merged_lse = self._cp_merge_tree(
            [v0, v1], [lse0, lse1], num_heads, v_head_dim)

        ref_v, ref_lse = self._reference_merge(
            [v0.float(), v1.float()], [lse0, lse1], num_heads, v_head_dim)

        assert torch.allclose(merged_v.float(), ref_v.bfloat16().float(), atol=1e-2, rtol=1e-2), \
            f"v_out max err = {(merged_v.float() - ref_v.bfloat16().float()).abs().max():.6f}"

    def test_large_v_head_dim(self):
        torch.manual_seed(42)
        batch, num_heads, v_head_dim = 1, 64, 256
        v0 = torch.randn(batch, num_heads * v_head_dim, dtype=torch.bfloat16)
        v1 = torch.randn(batch, num_heads * v_head_dim, dtype=torch.bfloat16)
        lse0 = torch.randn(batch, num_heads, dtype=torch.float32)
        lse1 = torch.randn(batch, num_heads, dtype=torch.float32)

        merged_v, merged_lse = self._cp_merge_tree(
            [v0, v1], [lse0, lse1], num_heads, v_head_dim)

        ref_v, ref_lse = self._reference_merge(
            [v0.float(), v1.float()], [lse0, lse1], num_heads, v_head_dim)

        assert torch.allclose(merged_v.float(), ref_v.bfloat16().float(), atol=1e-2, rtol=1e-2), \
            f"v_out max err = {(merged_v.float() - ref_v.bfloat16().float()).abs().max():.6f}"

    def test_eight_shards(self):
        torch.manual_seed(42)
        batch, num_heads, v_head_dim = 1, 4, 64
        shards_v = [torch.randn(batch, num_heads * v_head_dim, dtype=torch.bfloat16) for _ in range(8)]
        shards_lse = [torch.randn(batch, num_heads, dtype=torch.float32) for _ in range(8)]

        merged_v, merged_lse = self._cp_merge_tree(shards_v, shards_lse, num_heads, v_head_dim)

        ref_v, ref_lse = self._reference_merge(
            [v.float() for v in shards_v], shards_lse, num_heads, v_head_dim)

        assert torch.allclose(merged_v.float(), ref_v.bfloat16().float(), atol=1e-2, rtol=1e-2), \
            f"v_out max err = {(merged_v.float() - ref_v.bfloat16().float()).abs().max():.6f}"
        assert torch.allclose(merged_lse, ref_lse, atol=1e-2, rtol=1e-2), \
            f"lse max err = {(merged_lse - ref_lse).abs().max():.6f}"

    def test_per_head_matches_full_merge(self):
        """Per-head merge (split heads into groups) concatenated matches full-head merge."""
        torch.manual_seed(42)
        batch, num_heads, v_head_dim = 2, 8, 128
        shards_v = [torch.randn(batch, num_heads * v_head_dim, dtype=torch.bfloat16) for _ in range(4)]
        shards_lse = [torch.randn(batch, num_heads, dtype=torch.float32) for _ in range(4)]

        # Full merge: all heads at once
        full_v, full_lse = self._cp_merge_tree(shards_v, shards_lse, num_heads, v_head_dim)

        # Per-head merge: split into first half and second half
        shard_n_heads = num_heads // 2
        half_v_parts = []
        half_lse_parts = []
        for half in range(2):
            head_offset = half * shard_n_heads
            half_v, half_lse = self._cp_merge_tree(
                shards_v, shards_lse, num_heads, v_head_dim,
                shard_n_heads=shard_n_heads, head_offset=head_offset,
                input_n_heads=num_heads)
            half_v_parts.append(half_v)
            half_lse_parts.append(half_lse)

        # Concatenate per-head results and compare with full merge
        cat_v = torch.cat(half_v_parts, dim=1)
        cat_lse = torch.cat(half_lse_parts, dim=1)

        assert torch.allclose(full_v.float(), cat_v.float(), atol=1e-2, rtol=1e-2), \
            f"v_out max err = {(full_v.float() - cat_v.float()).abs().max():.6f}"
        assert torch.allclose(full_lse, cat_lse, atol=1e-2, rtol=1e-2), \
            f"lse max err = {(full_lse - cat_lse).abs().max():.6f}"
