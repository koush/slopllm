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

    def _cp_merge_tree(self, v_outs_bf16, lses_f32, num_heads, v_head_dim):
        N = len(v_outs_bf16)
        batch = v_outs_bf16[0].shape[0]
        numel = batch * num_heads * v_head_dim

        gpu_v_ptrs = []
        gpu_lse_ptrs = []
        for v, lse in zip(v_outs_bf16, lses_f32):
            v_ptr = self.ops.alloc(numel * 2)
            cpu_v = v.cpu().view(torch.int16).numpy().tobytes()
            self.ops.h2d(v_ptr, cpu_v)
            gpu_v_ptrs.append(v_ptr)

            lse_ptr = self.ops.alloc(batch * num_heads * 4)
            cpu_lse = lse.cpu().numpy().astype(np.float32).tobytes()
            self.ops.h2d(lse_ptr, cpu_lse)
            gpu_lse_ptrs.append(lse_ptr)

        out_v_ptr = self.ops.alloc(numel * 2)
        out_lse_ptr = self.ops.alloc(batch * num_heads * 4)

        v_args = [ctypes.c_void_p(int(p)) for p in gpu_v_ptrs] + [ctypes.c_void_p(0)] * (16 - N)
        lse_args = [ctypes.c_void_p(int(p)) for p in gpu_lse_ptrs] + [ctypes.c_void_p(0)] * (16 - N)

        self.ops.lib.glm_cp_merge_tree.restype = None
        self.ops.lib.glm_cp_merge_tree.argtypes = (
            [ctypes.c_void_p] +
            [ctypes.c_void_p] * 16 +
            [ctypes.c_void_p] * 16 +
            [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        )
        self.ops.lib.glm_cp_merge_tree(
            self.ops.ctx,
            *v_args,
            *lse_args,
            N,
            ctypes.c_void_p(int(out_v_ptr)),
            ctypes.c_void_p(int(out_lse_ptr)),
            numel, batch, num_heads, v_head_dim,
        )

        self.ops.synchronize()

        out_v_bytes = ctypes.create_string_buffer(numel * 2)
        self.ops.d2h(out_v_bytes, out_v_ptr, numel * 2)
        out_v = bf16_bytes_to_tensor(out_v_bytes.raw, numel).reshape(batch, num_heads * v_head_dim)

        out_lse_bytes = ctypes.create_string_buffer(batch * num_heads * 4)
        self.ops.d2h(out_lse_bytes, out_lse_ptr, batch * num_heads * 4)
        out_lse = f32_bytes_to_tensor(out_lse_bytes.raw, batch * num_heads).reshape(batch, num_heads)

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

    def test_matches_existing_cp_merge(self):
        torch.manual_seed(42)
        batch, num_heads, v_head_dim = 2, 8, 128
        shards_v = [torch.randn(batch, num_heads * v_head_dim, dtype=torch.bfloat16) for _ in range(4)]
        shards_lse = [torch.randn(batch, num_heads, dtype=torch.float32) for _ in range(4)]

        tree_v, tree_lse = self._cp_merge_tree(shards_v, shards_lse, num_heads, v_head_dim)

        numel = batch * num_heads * v_head_dim
        gpu_v_ptrs = []
        gpu_lse_ptrs = []
        for v, lse in zip(shards_v, shards_lse):
            v_ptr = self.ops.alloc(numel * 2)
            cpu_v = v.cpu().view(torch.int16).numpy().tobytes()
            self.ops.h2d(v_ptr, cpu_v)
            gpu_v_ptrs.append(v_ptr)
            lse_ptr = self.ops.alloc(batch * num_heads * 4)
            cpu_lse = lse.cpu().numpy().astype(np.float32).tobytes()
            self.ops.h2d(lse_ptr, cpu_lse)
            gpu_lse_ptrs.append(lse_ptr)

        merged_v_ptr = self.ops.alloc(numel * 2)
        merged_lse_ptr = self.ops.alloc(batch * num_heads * 4)

        self.ops.context_parallel_merge(
            gpu_v_ptrs, gpu_lse_ptrs, 4,
            merged_v_ptr, merged_lse_ptr,
            batch, num_heads, v_head_dim)

        self.ops.synchronize()

        ref_v_bytes = ctypes.create_string_buffer(numel * 2)
        self.ops.d2h(ref_v_bytes, merged_v_ptr, numel * 2)
        ref_v = bf16_bytes_to_tensor(ref_v_bytes.raw, numel).reshape(batch, num_heads * v_head_dim)

        ref_lse_bytes = ctypes.create_string_buffer(batch * num_heads * 4)
        self.ops.d2h(ref_lse_bytes, merged_lse_ptr, batch * num_heads * 4)
        ref_lse = f32_bytes_to_tensor(ref_lse_bytes.raw, batch * num_heads).reshape(batch, num_heads)

        assert torch.allclose(tree_v.float(), ref_v.float(), atol=1e-2, rtol=1e-2), \
            f"v_out max err vs existing = {(tree_v.float() - ref_v.float()).abs().max():.6f}"
        assert torch.allclose(tree_lse, ref_lse, atol=1e-2, rtol=1e-2), \
            f"lse max err vs existing = {(tree_lse - ref_lse).abs().max():.6f}"

        for p in gpu_v_ptrs + gpu_lse_ptrs:
            self.ops.free_buf(p)
        self.ops.free_buf(merged_v_ptr)
        self.ops.free_buf(merged_lse_ptr)
