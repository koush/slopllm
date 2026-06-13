import torch
import pytest
from helpers import GlmOps

NUM_GPUS = min(torch.cuda.device_count(), 2)
SKIP_REASON = f"P2P barrier requires 2 GPUs, found {NUM_GPUS}"


@pytest.mark.skipif(NUM_GPUS < 2, reason=SKIP_REASON)
class TestP2PBarrier:
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

    def _synchronize_all(self):
        for rank in range(NUM_GPUS):
            self.ops[rank].synchronize()

    def test_p2p_barrier_basic(self):
        """Barrier completes without error on all ranks."""
        instances = self._setup_p2p(1024)
        for rank in range(NUM_GPUS):
            self.ops[rank].p2p_barrier(instances[rank])
        self._synchronize_all()

    def test_p2p_barrier_repeated(self):
        """Consecutive barriers work correctly (double-buffer seq counter)."""
        instances = self._setup_p2p(1024)
        for _ in range(10):
            for rank in range(NUM_GPUS):
                self.ops[rank].p2p_barrier(instances[rank])
        self._synchronize_all()

    def test_p2p_barrier_interleaved_with_allreduce(self):
        """Barrier and allreduce can share the same instance without conflict."""
        instances = self._setup_p2p(4096)
        count = 1024

        for rank in range(NUM_GPUS):
            self.ops[rank].p2p_barrier(instances[rank])

        for rank in range(NUM_GPUS):
            x = torch.randn(count, dtype=torch.bfloat16, device=f'cuda:{rank}')
            self.ops[rank].p2p_allreduce(instances[rank], x, x, count)

        for rank in range(NUM_GPUS):
            self.ops[rank].p2p_barrier(instances[rank])

        for rank in range(NUM_GPUS):
            x = torch.randn(count, dtype=torch.bfloat16, device=f'cuda:{rank}')
            self.ops[rank].p2p_allreduce(instances[rank], x, x, count)

        self._synchronize_all()

    def test_p2p_barrier_ordering(self):
        """Barrier enforces ordering: writes before barrier are visible after."""
        instances = self._setup_p2p(4096)
        count = 1024

        for rank in range(NUM_GPUS):
            x = torch.full((count,), rank, dtype=torch.bfloat16, device=f'cuda:{rank}')
            self.ops[rank].p2p_allreduce(instances[rank], x, x, count)

        self._synchronize_all()

        for rank in range(NUM_GPUS):
            self.ops[rank].p2p_barrier(instances[rank])

        for rank in range(NUM_GPUS):
            y = torch.randn(count, dtype=torch.bfloat16, device=f'cuda:{rank}')
            self.ops[rank].p2p_allreduce(instances[rank], y, y, count)

        self._synchronize_all()
