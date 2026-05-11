import torch
import pytest
import math
import os
import ctypes
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


NUM_GPUS = min(torch.cuda.device_count(), 2)
SKIP_REASON = f"P2P CP merge requires 2 GPUs, found {NUM_GPUS}"


@pytest.mark.skipif(NUM_GPUS < 2, reason=SKIP_REASON)
class TestP2PCpMerge:
    @pytest.fixture(autouse=True)
    def setup_p2p(self):
        self.ops = []
        self.instances = []
        self.data_ptrs = []
        for rank in range(NUM_GPUS):
            self.ops.append(GlmOps(device_id=rank))
        yield
        for data_ptrs in self.data_ptrs:
            for rank, ptr in enumerate(data_ptrs):
                self.ops[rank].free_buf(ptr)
        for inst in self.instances:
            for rank in range(NUM_GPUS):
                self.ops[rank].p2p_destroy_instance(inst[rank])
        for ops in self.ops:
            del ops
        torch.cuda.empty_cache()

    def _setup_p2p(self, max_bytes):
        """Create P2P instances and enable peer access."""
        world_size = NUM_GPUS
        instances = []
        for rank in range(world_size):
            for peer in range(world_size):
                if peer != rank:
                    rc = self.ops[rank].p2p_enable_peer_access(peer)
                    assert rc == 0, f"Peer access failed: GPU {rank} -> {peer}"
            inst = self.ops[rank].p2p_create_instance(rank, world_size)
            assert inst is not None and inst != 0, f"p2p_create_instance failed for rank {rank}"
            instances.append(inst)

        # Allocate data buffers (2 * max_bytes for double buffering).
        buf_bytes = max_bytes * 2
        data_ptrs = []
        for rank in range(world_size):
            ptr = self.ops[rank].alloc(buf_bytes)
            data_ptrs.append(ptr)

        flag_ptrs = [self.ops[r].p2p_get_flag_ptr(instances[r]) for r in range(world_size)]

        for rank in range(world_size):
            self.ops[rank].p2p_set_max_bytes(instances[rank], max_bytes)
            self.ops[rank].p2p_set_peers(
                instances[rank],
                data_ptrs,
                flag_ptrs,
                world_size,
            )

        self.instances.append(instances)
        self.data_ptrs.append(data_ptrs)
        return instances

    def test_p2p_cp_merge_2_gpus(self):
        """2-GPU P2P CP merge with 2 shards."""
        B, H, D = 1, 4, 128
        num_shards = NUM_GPUS
        torch.manual_seed(42)

        v_out_bytes = B * H * D * 2
        lse_bytes = B * H * 4
        max_bytes = v_out_bytes + lse_bytes + 256
        instances = self._setup_p2p(max_bytes)

        partial_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=f"cuda:{r}") for r in range(num_shards)]
        partial_lses = [torch.randn(B, H, dtype=torch.float32, device=f"cuda:{r}") for r in range(num_shards)]

        all_v_on_gpu0 = [t.to("cuda:0") for t in partial_v_outs]
        all_lse_on_gpu0 = [t.to("cuda:0") for t in partial_lses]
        merged_v_ref, merged_lse_ref = torch_merge_partial_attn(all_v_on_gpu0, all_lse_on_gpu0)

        merged_v_outs = [torch.empty(B, H, D, dtype=torch.bfloat16, device=f"cuda:{r}") for r in range(num_shards)]
        merged_lse_outs = [torch.empty(B, H, dtype=torch.float32, device=f"cuda:{r}") for r in range(num_shards)]

        for rank in range(num_shards):
            self.ops[rank].p2p_cp_merge(
                instances[rank],
                partial_v_outs[rank].data_ptr(),
                partial_lses[rank].data_ptr(),
                merged_v_outs[rank].data_ptr(),
                merged_lse_outs[rank].data_ptr(),
                num_shards, B, H, D,
            )

        for rank in range(num_shards):
            self.ops[rank].synchronize()

        torch.testing.assert_close(merged_v_outs[0].cpu(), merged_v_ref.cpu(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(merged_lse_outs[0].cpu(), merged_lse_ref.cpu(), atol=1e-3, rtol=1e-3)

        torch.testing.assert_close(merged_v_outs[1].cpu(), merged_v_outs[0].cpu(), atol=0, rtol=0)

    def test_p2p_cp_merge_nullptr_lse(self):
        """P2P CP merge with merged_lse=nullptr."""
        B, H, D = 1, 4, 128
        num_shards = NUM_GPUS
        torch.manual_seed(123)

        v_out_bytes = B * H * D * 2
        lse_bytes = B * H * 4
        max_bytes = v_out_bytes + lse_bytes + 256
        instances = self._setup_p2p(max_bytes)

        partial_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=f"cuda:{r}") for r in range(num_shards)]
        partial_lses = [torch.randn(B, H, dtype=torch.float32, device=f"cuda:{r}") for r in range(num_shards)]

        all_v_on_gpu0 = [t.to("cuda:0") for t in partial_v_outs]
        all_lse_on_gpu0 = [t.to("cuda:0") for t in partial_lses]
        merged_v_ref, _ = torch_merge_partial_attn(all_v_on_gpu0, all_lse_on_gpu0)

        merged_v_outs = [torch.empty(B, H, D, dtype=torch.bfloat16, device=f"cuda:{r}") for r in range(num_shards)]

        for rank in range(num_shards):
            self.ops[rank].p2p_cp_merge(
                instances[rank],
                partial_v_outs[rank].data_ptr(),
                partial_lses[rank].data_ptr(),
                merged_v_outs[rank].data_ptr(),
                None,
                num_shards, B, H, D,
            )

        for rank in range(num_shards):
            self.ops[rank].synchronize()

        torch.testing.assert_close(merged_v_outs[0].cpu(), merged_v_ref.cpu(), atol=1e-3, rtol=1e-3)

    def test_p2p_cp_match_single_gpu_merge(self):
        """P2P merge should match single-GPU context_parallel_merge."""
        B, H, D = 1, 4, 128
        num_shards = NUM_GPUS
        torch.manual_seed(99)

        v_out_bytes = B * H * D * 2
        lse_bytes = B * H * 4
        max_bytes = v_out_bytes + lse_bytes + 256
        instances = self._setup_p2p(max_bytes)

        partial_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=f"cuda:{r}") for r in range(num_shards)]
        partial_lses = [torch.randn(B, H, dtype=torch.float32, device=f"cuda:{r}") for r in range(num_shards)]

        merged_v_p2p = [torch.empty(B, H, D, dtype=torch.bfloat16, device=f"cuda:{r}") for r in range(num_shards)]
        merged_lse_p2p = [torch.empty(B, H, dtype=torch.float32, device=f"cuda:{r}") for r in range(num_shards)]

        for rank in range(num_shards):
            self.ops[rank].p2p_cp_merge(
                instances[rank],
                partial_v_outs[rank].data_ptr(),
                partial_lses[rank].data_ptr(),
                merged_v_p2p[rank].data_ptr(),
                merged_lse_p2p[rank].data_ptr(),
                num_shards, B, H, D,
            )

        for rank in range(num_shards):
            self.ops[rank].synchronize()

        all_v_on_gpu0 = [partial_v_outs[r].to("cuda:0") for r in range(num_shards)]
        all_lse_on_gpu0 = [partial_lses[r].to("cuda:0") for r in range(num_shards)]

        merged_v_single = torch.empty(B, H, D, dtype=torch.bfloat16, device="cuda:0")
        merged_lse_single = torch.empty(B, H, dtype=torch.float32, device="cuda:0")

        self.ops[0].context_parallel_merge(
            [t.data_ptr() for t in all_v_on_gpu0],
            [t.data_ptr() for t in all_lse_on_gpu0],
            num_shards,
            merged_v_single.data_ptr(),
            merged_lse_single.data_ptr(),
            B, H, D,
        )
        self.ops[0].synchronize()

        torch.testing.assert_close(merged_v_p2p[0].cpu(), merged_v_single.cpu(), atol=0, rtol=0)
        torch.testing.assert_close(merged_lse_p2p[0].cpu(), merged_lse_single.cpu(), atol=1e-6, rtol=1e-6)

    def test_p2p_cp_merge_commutativity(self):
        """P2P merge(a,b) == merge(b,a) — swap which data is on which rank."""
        B, H, D = 1, 4, 128
        torch.manual_seed(77)

        v_out_bytes = B * H * D * 2
        lse_bytes = B * H * 4
        max_bytes = v_out_bytes + lse_bytes + 256
        instances_ab = self._setup_p2p(max_bytes)

        v0 = torch.randn(B, H, D, dtype=torch.bfloat16, device="cuda:0")
        v1 = torch.randn(B, H, D, dtype=torch.bfloat16, device="cuda:1")
        lse0 = torch.randn(B, H, dtype=torch.float32, device="cuda:0")
        lse1 = torch.randn(B, H, dtype=torch.float32, device="cuda:1")

        merged_v_ab = [torch.empty(B, H, D, dtype=torch.bfloat16, device=f"cuda:{r}") for r in range(2)]
        merged_lse_ab = [torch.empty(B, H, dtype=torch.float32, device=f"cuda:{r}") for r in range(2)]

        self.ops[0].p2p_cp_merge(
            instances_ab[0], v0.data_ptr(), lse0.data_ptr(),
            merged_v_ab[0].data_ptr(), merged_lse_ab[0].data_ptr(),
            2, B, H, D,
        )
        self.ops[1].p2p_cp_merge(
            instances_ab[1], v1.data_ptr(), lse1.data_ptr(),
            merged_v_ab[1].data_ptr(), merged_lse_ab[1].data_ptr(),
            2, B, H, D,
        )
        self.ops[0].synchronize()
        self.ops[1].synchronize()

        # Swap: put v1 on GPU 0 and v0 on GPU 1, merge again
        # Since merge is commutative, the result should be identical
        merged_v_ba = [torch.empty(B, H, D, dtype=torch.bfloat16, device=f"cuda:{r}") for r in range(2)]
        merged_lse_ba = [torch.empty(B, H, dtype=torch.float32, device=f"cuda:{r}") for r in range(2)]

        v1_on_0 = v1.to("cuda:0")
        lse1_on_0 = lse1.to("cuda:0")
        v0_on_1 = v0.to("cuda:1")
        lse0_on_1 = lse0.to("cuda:1")

        instances_ba = self._setup_p2p(max_bytes)

        self.ops[0].p2p_cp_merge(
            instances_ba[0], v1_on_0.data_ptr(), lse1_on_0.data_ptr(),
            merged_v_ba[0].data_ptr(), merged_lse_ba[0].data_ptr(),
            2, B, H, D,
        )
        self.ops[1].p2p_cp_merge(
            instances_ba[1], v0_on_1.data_ptr(), lse0_on_1.data_ptr(),
            merged_v_ba[1].data_ptr(), merged_lse_ba[1].data_ptr(),
            2, B, H, D,
        )
        self.ops[0].synchronize()
        self.ops[1].synchronize()

        torch.testing.assert_close(merged_v_ab[0].cpu(), merged_v_ba[0].cpu(), atol=0, rtol=0)
        torch.testing.assert_close(merged_lse_ab[0].cpu(), merged_lse_ba[0].cpu(), atol=1e-6, rtol=1e-6)

    def test_p2p_cp_merge_head_dim_256(self):
        """P2P CP merge with head_dim=256."""
        B, H, D = 1, 8, 256
        num_shards = NUM_GPUS
        torch.manual_seed(42)

        v_out_bytes = B * H * D * 2
        lse_bytes = B * H * 4
        max_bytes = v_out_bytes + lse_bytes + 256
        instances = self._setup_p2p(max_bytes)

        partial_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=f"cuda:{r}") for r in range(num_shards)]
        partial_lses = [torch.randn(B, H, dtype=torch.float32, device=f"cuda:{r}") for r in range(num_shards)]

        all_v_on_gpu0 = [t.to("cuda:0") for t in partial_v_outs]
        all_lse_on_gpu0 = [t.to("cuda:0") for t in partial_lses]
        merged_v_ref, merged_lse_ref = torch_merge_partial_attn(all_v_on_gpu0, all_lse_on_gpu0)

        merged_v_outs = [torch.empty(B, H, D, dtype=torch.bfloat16, device=f"cuda:{r}") for r in range(num_shards)]
        merged_lse_outs = [torch.empty(B, H, dtype=torch.float32, device=f"cuda:{r}") for r in range(num_shards)]

        for rank in range(num_shards):
            self.ops[rank].p2p_cp_merge(
                instances[rank],
                partial_v_outs[rank].data_ptr(),
                partial_lses[rank].data_ptr(),
                merged_v_outs[rank].data_ptr(),
                merged_lse_outs[rank].data_ptr(),
                num_shards, B, H, D,
            )

        for rank in range(num_shards):
            self.ops[rank].synchronize()

        torch.testing.assert_close(merged_v_outs[0].cpu(), merged_v_ref.cpu(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(merged_lse_outs[0].cpu(), merged_lse_ref.cpu(), atol=1e-3, rtol=1e-3)

    def test_p2p_cp_merge_heads_first_half(self):
        """P2P head-grouped merge: only first half of heads."""
        B, H, D = 1, 8, 128
        shard_n_heads = H // 2
        head_offset = 0
        num_shards = NUM_GPUS
        torch.manual_seed(42)

        v_out_bytes = B * shard_n_heads * D * 2
        lse_bytes = B * H * 4
        max_bytes = v_out_bytes + lse_bytes + 256
        instances = self._setup_p2p(max_bytes)

        full_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=f"cuda:{r}") for r in range(num_shards)]
        partial_lses = [torch.randn(B, H, dtype=torch.float32, device=f"cuda:{r}") for r in range(num_shards)]
        shard_v_outs = [v[:, head_offset:head_offset + shard_n_heads, :].contiguous() for v in full_v_outs]

        all_v_on_gpu0 = [v.to("cuda:0") for v in full_v_outs]
        all_lse_on_gpu0 = [t.to("cuda:0") for t in partial_lses]
        merged_v_ref, merged_lse_ref = torch_merge_partial_attn(all_v_on_gpu0, all_lse_on_gpu0)
        merged_v_ref_slice = merged_v_ref[:, head_offset:head_offset + shard_n_heads, :]
        merged_lse_ref_slice = merged_lse_ref[:, head_offset:head_offset + shard_n_heads]

        merged_v_outs = [torch.empty(B, shard_n_heads, D, dtype=torch.bfloat16, device=f"cuda:{r}") for r in range(num_shards)]
        merged_lse_outs = [torch.empty(B, shard_n_heads, dtype=torch.float32, device=f"cuda:{r}") for r in range(num_shards)]

        for rank in range(num_shards):
            self.ops[rank].p2p_cp_merge_heads(
                instances[rank],
                shard_v_outs[rank].data_ptr(),
                partial_lses[rank].data_ptr(),
                merged_v_outs[rank].data_ptr(),
                merged_lse_outs[rank].data_ptr(),
                num_shards, B, H, shard_n_heads, head_offset, D,
            )

        for rank in range(num_shards):
            self.ops[rank].synchronize()

        torch.testing.assert_close(merged_v_outs[0].cpu(), merged_v_ref_slice.cpu(), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(merged_lse_outs[0].cpu(), merged_lse_ref_slice.cpu(), atol=1e-3, rtol=1e-3)

    def test_p2p_cp_merge_heads_two_tp_shards(self):
        """P2P head-grouped merge simulating 2-way TP: merge heads [0,H/2) and [H/2,H) separately."""
        B, H, D = 1, 8, 128
        shard_n_heads = H // 2
        num_shards = NUM_GPUS
        torch.manual_seed(99)

        v_out_bytes = B * shard_n_heads * D * 2
        lse_bytes = B * H * 4
        max_bytes = v_out_bytes + lse_bytes + 256
        instances = self._setup_p2p(max_bytes)

        full_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=f"cuda:{r}") for r in range(num_shards)]
        partial_lses = [torch.randn(B, H, dtype=torch.float32, device=f"cuda:{r}") for r in range(num_shards)]

        all_v_on_gpu0 = [t.to("cuda:0") for t in full_v_outs]
        all_lse_on_gpu0 = [t.to("cuda:0") for t in partial_lses]
        merged_v_ref, merged_lse_ref = torch_merge_partial_attn(all_v_on_gpu0, all_lse_on_gpu0)

        for tp_rank in range(2):
            offset = tp_rank * shard_n_heads
            shard_v_outs = [v[:, offset:offset + shard_n_heads, :].contiguous() for v in full_v_outs]
            merged_v_outs = [torch.empty(B, shard_n_heads, D, dtype=torch.bfloat16, device=f"cuda:{r}") for r in range(num_shards)]
            merged_lse_outs = [torch.empty(B, shard_n_heads, dtype=torch.float32, device=f"cuda:{r}") for r in range(num_shards)]

            for rank in range(num_shards):
                self.ops[rank].p2p_cp_merge_heads(
                    instances[rank],
                    shard_v_outs[rank].data_ptr(),
                    partial_lses[rank].data_ptr(),
                    merged_v_outs[rank].data_ptr(),
                    merged_lse_outs[rank].data_ptr(),
                    num_shards, B, H, shard_n_heads, offset, D,
                )

            for rank in range(num_shards):
                self.ops[rank].synchronize()

            torch.testing.assert_close(
                merged_v_outs[0].cpu(), merged_v_ref[:, offset:offset + shard_n_heads, :].cpu(),
                atol=1e-3, rtol=1e-3)
            torch.testing.assert_close(
                merged_lse_outs[0].cpu(), merged_lse_ref[:, offset:offset + shard_n_heads].cpu(),
                atol=1e-3, rtol=1e-3)

    def test_p2p_cp_merge_heads_matches_single_gpu(self):
        """P2P head-grouped merge should match single-GPU context_parallel_merge_heads."""
        B, H, D = 1, 8, 128
        shard_n_heads = H // 2
        head_offset = H // 2
        num_shards = NUM_GPUS
        torch.manual_seed(77)

        v_out_bytes = B * shard_n_heads * D * 2
        lse_bytes = B * H * 4
        max_bytes = v_out_bytes + lse_bytes + 256
        instances = self._setup_p2p(max_bytes)

        full_v_outs = [torch.randn(B, H, D, dtype=torch.bfloat16, device=f"cuda:{r}") for r in range(num_shards)]
        partial_lses = [torch.randn(B, H, dtype=torch.float32, device=f"cuda:{r}") for r in range(num_shards)]
        shard_v_outs = [v[:, head_offset:head_offset + shard_n_heads, :].contiguous() for v in full_v_outs]

        merged_v_p2p = [torch.empty(B, shard_n_heads, D, dtype=torch.bfloat16, device=f"cuda:{r}") for r in range(num_shards)]
        merged_lse_p2p = [torch.empty(B, shard_n_heads, dtype=torch.float32, device=f"cuda:{r}") for r in range(num_shards)]

        for rank in range(num_shards):
            self.ops[rank].p2p_cp_merge_heads(
                instances[rank],
                shard_v_outs[rank].data_ptr(),
                partial_lses[rank].data_ptr(),
                merged_v_p2p[rank].data_ptr(),
                merged_lse_p2p[rank].data_ptr(),
                num_shards, B, H, shard_n_heads, head_offset, D,
            )

        for rank in range(num_shards):
            self.ops[rank].synchronize()

        all_shard_v_on_gpu0 = [v.to("cuda:0") for v in shard_v_outs]
        all_lse_on_gpu0 = [t.to("cuda:0") for t in partial_lses]

        merged_v_single = torch.empty(B, shard_n_heads, D, dtype=torch.bfloat16, device="cuda:0")
        merged_lse_single = torch.empty(B, shard_n_heads, dtype=torch.float32, device="cuda:0")

        self.ops[0].context_parallel_merge_heads(
            [t.data_ptr() for t in all_shard_v_on_gpu0],
            [t.data_ptr() for t in all_lse_on_gpu0],
            num_shards,
            merged_v_single.data_ptr(),
            merged_lse_single.data_ptr(),
            B, H, shard_n_heads, head_offset, D,
        )
        self.ops[0].synchronize()

        torch.testing.assert_close(merged_v_p2p[0].cpu(), merged_v_single.cpu(), atol=0, rtol=0)
        torch.testing.assert_close(merged_lse_p2p[0].cpu(), merged_lse_single.cpu(), atol=1e-6, rtol=1e-6)
