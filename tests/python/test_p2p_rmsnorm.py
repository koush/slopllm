import torch
import pytest
import math
from helpers import GlmOps, ATOL, RTOL


NUM_GPUS = min(torch.cuda.device_count(), 2)
SKIP_REASON = f"P2P RMSNorm requires 2 GPUs, found {NUM_GPUS}"


@pytest.mark.skipif(NUM_GPUS < 2, reason=SKIP_REASON)
class TestP2PRmsnorm:
    @pytest.fixture(autouse=True)
    def setup_p2p(self):
        self.ops = []
        self.instances = []
        for rank in range(NUM_GPUS):
            self.ops.append(GlmOps(device_id=rank))
        yield
        for inst in self.instances:
            for rank in range(NUM_GPUS):
                self.ops[rank].p2p_destroy_instance(inst[rank])
        for ops in self.ops:
            del ops
        torch.cuda.empty_cache()

    def _setup_p2p(self, max_bytes):
        world_size = NUM_GPUS
        instances = []
        for rank in range(world_size):
            for peer in range(world_size):
                if peer != rank:
                    rc = self.ops[rank].p2p_enable_peer_access(peer)
                    assert rc == 0, f"Peer access failed: GPU {rank} -> {peer}"
            inst = self.ops[rank].p2p_create_instance(rank, world_size, max_bytes)
            assert inst is not None and inst != 0, f"p2p_create_instance failed for rank {rank}"
            instances.append(inst)

        data_ptrs = [self.ops[r].p2p_get_data_ptr(instances[r]) for r in range(world_size)]
        flag_ptrs = [self.ops[r].p2p_get_flag_ptr(instances[r]) for r in range(world_size)]

        for rank in range(world_size):
            self.ops[rank].p2p_set_peers(
                instances[rank],
                data_ptrs,
                flag_ptrs,
                world_size,
            )

        self.instances.append(instances)
        return instances

    def _synchronize_all(self):
        for rank in range(NUM_GPUS):
            self.ops[rank].synchronize()
            # torch.cuda.synchronize(rank)

    def _torch_rmsnorm(self, x, weight, eps):
        x_f = x.float()
        variance = x_f.pow(2).mean(-1, keepdim=True)
        inv_rms = torch.rsqrt(variance + eps)
        return (x_f * inv_rms * weight.float()).to(torch.bfloat16)

    def test_p2p_rmsnorm_basic(self):
        """Basic 2-GPU P2P RMSNorm: each GPU has half the hidden dim."""
        world_size = NUM_GPUS
        full_dim = 256
        shard_dim = full_dim // world_size
        batch = 4
        eps = 1e-6
        torch.manual_seed(42)

        max_bytes = batch * 4 + 256
        instances = self._setup_p2p(max_bytes)

        full_x = torch.randn(batch, full_dim, dtype=torch.bfloat16, device="cpu")
        full_weight = torch.randn(full_dim, dtype=torch.bfloat16, device="cpu")
        expected = self._torch_rmsnorm(full_x, full_weight, eps)

        shards = []
        for rank in range(world_size):
            x_shard = full_x[:, rank * shard_dim:(rank + 1) * shard_dim].to(f"cuda:{rank}").contiguous()
            w_shard = full_weight[rank * shard_dim:(rank + 1) * shard_dim].to(f"cuda:{rank}").contiguous()
            out_shard = torch.empty(batch, shard_dim, dtype=torch.bfloat16, device=f"cuda:{rank}")
            shards.append((x_shard, w_shard, out_shard))

        outputs = []
        for rank in range(world_size):
            x_shard, w_shard, out_shard = shards[rank]
            self.ops[rank].p2p_rmsnorm(
                instances[rank],
                x_shard.data_ptr(),
                w_shard.data_ptr(),
                out_shard.data_ptr(),
                eps, shard_dim, full_dim, batch,
            )
            outputs.append(out_shard)

        self._synchronize_all()

        full_output = torch.cat([o.cpu() for o in outputs], dim=1)
        torch.testing.assert_close(full_output, expected.cpu(), atol=1e-2, rtol=1e-2)

    def test_p2p_rmsnorm_matches_single_gpu(self):
        """P2P RMSNorm should match single-GPU RMSNorm exactly."""
        world_size = NUM_GPUS
        full_dim = 512
        shard_dim = full_dim // world_size
        batch = 8
        eps = 1e-5
        torch.manual_seed(123)

        max_bytes = batch * 4 + 256
        instances = self._setup_p2p(max_bytes)

        full_x = torch.randn(batch, full_dim, dtype=torch.bfloat16, device="cuda:0")
        full_weight = torch.randn(full_dim, dtype=torch.bfloat16, device="cuda:0")

        # Compute reference on single GPU
        single_out = torch.empty(batch, full_dim, dtype=torch.bfloat16, device="cuda:0")
        self.ops[0].rmsnorm(single_out.data_ptr(), full_x.data_ptr(), full_weight.data_ptr(), eps, full_dim, batch)
        self.ops[0].synchronize()

        # Stage all shards to target GPUs before launching any P2P kernels
        shards = []
        for rank in range(world_size):
            x_shard = full_x[:, rank * shard_dim:(rank + 1) * shard_dim].to(f"cuda:{rank}").contiguous()
            w_shard = full_weight[rank * shard_dim:(rank + 1) * shard_dim].to(f"cuda:{rank}").contiguous()
            out_shard = torch.empty(batch, shard_dim, dtype=torch.bfloat16, device=f"cuda:{rank}")
            shards.append((x_shard, w_shard, out_shard))

        # Launch all P2P kernels (no cross-GPU copies after this point)
        outputs = []
        for rank in range(world_size):
            x_shard, w_shard, out_shard = shards[rank]
            self.ops[rank].p2p_rmsnorm(
                instances[rank],
                x_shard.data_ptr(),
                w_shard.data_ptr(),
                out_shard.data_ptr(),
                eps, shard_dim, full_dim, batch,
            )
            outputs.append(out_shard)

        self._synchronize_all()
        full_p2p_out = torch.cat([o.cpu() for o in outputs], dim=1)
        torch.testing.assert_close(full_p2p_out, single_out.cpu(), atol=1e-2, rtol=1e-2)

    def test_p2p_rmsnorm_single_row(self):
        """P2P RMSNorm with batch=1 (single row)."""
        world_size = NUM_GPUS
        full_dim = 1024
        shard_dim = full_dim // world_size
        batch = 1
        eps = 1e-6
        torch.manual_seed(99)

        max_bytes = batch * 4 + 256
        instances = self._setup_p2p(max_bytes)

        full_x = torch.randn(batch, full_dim, dtype=torch.bfloat16, device="cpu")
        full_weight = torch.randn(full_dim, dtype=torch.bfloat16, device="cpu")
        expected = self._torch_rmsnorm(full_x, full_weight, eps)

        shards = []
        for rank in range(world_size):
            x_shard = full_x[:, rank * shard_dim:(rank + 1) * shard_dim].to(f"cuda:{rank}").contiguous()
            w_shard = full_weight[rank * shard_dim:(rank + 1) * shard_dim].to(f"cuda:{rank}").contiguous()
            out_shard = torch.empty(batch, shard_dim, dtype=torch.bfloat16, device=f"cuda:{rank}")
            shards.append((x_shard, w_shard, out_shard))

        outputs = []
        for rank in range(world_size):
            x_shard, w_shard, out_shard = shards[rank]
            self.ops[rank].p2p_rmsnorm(
                instances[rank],
                x_shard.data_ptr(),
                w_shard.data_ptr(),
                out_shard.data_ptr(),
                eps, shard_dim, full_dim, batch,
            )
            outputs.append(out_shard)

        self._synchronize_all()

        full_output = torch.cat([o.cpu() for o in outputs], dim=1)
        torch.testing.assert_close(full_output, expected.cpu(), atol=1e-2, rtol=1e-2)

    def test_p2p_rmsnorm_large_batch(self):
        """P2P RMSNorm with larger batch size."""
        world_size = NUM_GPUS
        full_dim = 128
        shard_dim = full_dim // world_size
        batch = 32
        eps = 1e-6
        torch.manual_seed(7)

        max_bytes = batch * 4 + 256
        instances = self._setup_p2p(max_bytes)

        full_x = torch.randn(batch, full_dim, dtype=torch.bfloat16, device="cpu")
        full_weight = torch.randn(full_dim, dtype=torch.bfloat16, device="cpu")
        expected = self._torch_rmsnorm(full_x, full_weight, eps)

        shards = []
        for rank in range(world_size):
            x_shard = full_x[:, rank * shard_dim:(rank + 1) * shard_dim].to(f"cuda:{rank}").contiguous()
            w_shard = full_weight[rank * shard_dim:(rank + 1) * shard_dim].to(f"cuda:{rank}").contiguous()
            out_shard = torch.empty(batch, shard_dim, dtype=torch.bfloat16, device=f"cuda:{rank}")
            shards.append((x_shard, w_shard, out_shard))

        outputs = []
        for rank in range(world_size):
            x_shard, w_shard, out_shard = shards[rank]
            self.ops[rank].p2p_rmsnorm(
                instances[rank],
                x_shard.data_ptr(),
                w_shard.data_ptr(),
                out_shard.data_ptr(),
                eps, shard_dim, full_dim, batch,
            )
            outputs.append(out_shard)

        self._synchronize_all()

        full_output = torch.cat([o.cpu() for o in outputs], dim=1)
        torch.testing.assert_close(full_output, expected.cpu(), atol=1e-2, rtol=1e-2)

    def test_p2p_rmsnorm_ones_weight(self):
        """P2P RMSNorm with all-ones weight (pure normalization)."""
        world_size = NUM_GPUS
        full_dim = 256
        shard_dim = full_dim // world_size
        batch = 4
        eps = 1e-6
        torch.manual_seed(55)

        max_bytes = batch * 4 + 256
        instances = self._setup_p2p(max_bytes)

        full_x = torch.randn(batch, full_dim, dtype=torch.bfloat16, device="cpu")
        full_weight = torch.ones(full_dim, dtype=torch.bfloat16, device="cpu")
        expected = self._torch_rmsnorm(full_x, full_weight, eps)

        shards = []
        for rank in range(world_size):
            x_shard = full_x[:, rank * shard_dim:(rank + 1) * shard_dim].to(f"cuda:{rank}").contiguous()
            w_shard = full_weight[rank * shard_dim:(rank + 1) * shard_dim].to(f"cuda:{rank}").contiguous()
            out_shard = torch.empty(batch, shard_dim, dtype=torch.bfloat16, device=f"cuda:{rank}")
            shards.append((x_shard, w_shard, out_shard))

        outputs = []
        for rank in range(world_size):
            x_shard, w_shard, out_shard = shards[rank]
            self.ops[rank].p2p_rmsnorm(
                instances[rank],
                x_shard.data_ptr(),
                w_shard.data_ptr(),
                out_shard.data_ptr(),
                eps, shard_dim, full_dim, batch,
            )
            outputs.append(out_shard)

        self._synchronize_all()

        full_output = torch.cat([o.cpu() for o in outputs], dim=1)
        torch.testing.assert_close(full_output, expected.cpu(), atol=1e-2, rtol=1e-2)

    def test_p2p_rmsnorm_odd_dim(self):
        """P2P RMSNorm with odd shard dimension."""
        world_size = NUM_GPUS
        shard_dim = 127
        full_dim = shard_dim * world_size
        batch = 2
        eps = 1e-6
        torch.manual_seed(200)

        max_bytes = batch * 4 + 256
        instances = self._setup_p2p(max_bytes)

        full_x = torch.randn(batch, full_dim, dtype=torch.bfloat16, device="cpu")
        full_weight = torch.randn(full_dim, dtype=torch.bfloat16, device="cpu")
        expected = self._torch_rmsnorm(full_x, full_weight, eps)

        shards = []
        for rank in range(world_size):
            start = rank * shard_dim
            end = start + shard_dim
            x_shard = full_x[:, start:end].to(f"cuda:{rank}").contiguous()
            w_shard = full_weight[start:end].to(f"cuda:{rank}").contiguous()
            out_shard = torch.empty(batch, shard_dim, dtype=torch.bfloat16, device=f"cuda:{rank}")
            shards.append((x_shard, w_shard, out_shard))

        outputs = []
        for rank in range(world_size):
            x_shard, w_shard, out_shard = shards[rank]
            self.ops[rank].p2p_rmsnorm(
                instances[rank],
                x_shard.data_ptr(),
                w_shard.data_ptr(),
                out_shard.data_ptr(),
                eps, shard_dim, full_dim, batch,
            )
            outputs.append(out_shard)

        self._synchronize_all()

        full_output = torch.cat([o.cpu() for o in outputs], dim=1)
        torch.testing.assert_close(full_output, expected.cpu(), atol=1e-2, rtol=1e-2)
