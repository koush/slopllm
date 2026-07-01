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
        for rank in range(NUM_GPUS):
            self.ops.append(GlmOps(device_id=rank))
        yield
        for inst in self.instances:
            for rank in range(NUM_GPUS):
                self.ops[rank].p2p_destroy_instance(inst[rank])
        for ops in self.ops:
            del ops
        torch.cuda.empty_cache()

    def _setup_p2p(self):
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

        flag_ptrs = [self.ops[r].p2p_get_flag_ptr(instances[r]) for r in range(world_size)]

        for rank in range(world_size):
            self.ops[rank].p2p_set_peers(
                instances[rank],
                flag_ptrs,
                world_size,
            )

        self.instances.append(instances)
        return instances

    def _synchronize_all(self):
        for rank in range(NUM_GPUS):
            self.ops[rank].synchronize()

    def test_p2p_barrier_basic(self):
        """Barrier completes without error on all ranks."""
        instances = self._setup_p2p()
        for rank in range(NUM_GPUS):
            self.ops[rank].p2p_barrier(instances[rank])
        self._synchronize_all()

    def test_p2p_barrier_repeated(self):
        """Consecutive barriers work correctly (double-buffer seq counter)."""
        instances = self._setup_p2p()
        for _ in range(10):
            for rank in range(NUM_GPUS):
                self.ops[rank].p2p_barrier(instances[rank])
        self._synchronize_all()
