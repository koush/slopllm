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
        device_ids = [self.ops[r].device for r in range(world_size)]
        instances = []
        for rank in range(world_size):
            for peer in range(world_size):
                if peer != rank:
                    rc = self.ops[rank].p2p_enable_peer_access(device_ids[peer])
                    assert rc == 0, f"Peer access failed: GPU {rank} -> {peer}"
            inst = self.ops[rank].p2p_create_instance(rank, world_size, device_ids)
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

    def test_p2p_arrive_wait_basic(self):
        """Split barrier: arrive on all ranks, then wait on all ranks."""
        instances = self._setup_p2p()
        for rank in range(NUM_GPUS):
            self.ops[rank].p2p_arrive(instances[rank])
        for rank in range(NUM_GPUS):
            self.ops[rank].p2p_wait(instances[rank])
        self._synchronize_all()

    def test_p2p_arrive_wait_repeated(self):
        """Repeated split barriers: alternate arrive-all / wait-all rounds."""
        instances = self._setup_p2p()
        for _ in range(10):
            for rank in range(NUM_GPUS):
                self.ops[rank].p2p_arrive(instances[rank])
            for rank in range(NUM_GPUS):
                self.ops[rank].p2p_wait(instances[rank])
        self._synchronize_all()

    def test_p2p_arrive_wait_decoupled(self):
        """arrive and wait are genuinely separate: a full host sync between the
        arrive phase and the wait phase must still complete (flags persist in
        device memory; wait recovers the target from my_seq_counter)."""
        instances = self._setup_p2p()
        for round_idx in range(5):
            for rank in range(NUM_GPUS):
                self.ops[rank].p2p_arrive(instances[rank])
            self._synchronize_all()  # arrive fully landed on every GPU before any wait
            for rank in range(NUM_GPUS):
                self.ops[rank].p2p_wait(instances[rank])
            self._synchronize_all()
            round_idx  # silence unused

    def test_p2p_arrive_wait_interleaved(self):
        """Per-rank arrive-then-wait (the original barrier ordering) completes."""
        instances = self._setup_p2p()
        for _ in range(10):
            for rank in range(NUM_GPUS):
                self.ops[rank].p2p_arrive(instances[rank])
                self.ops[rank].p2p_wait(instances[rank])
        self._synchronize_all()

    def test_p2p_arrive_wait_partial(self):
        """Partial split barrier: each rank syncs with one peer via peer_rank."""
        instances = self._setup_p2p()
        for _ in range(10):
            for rank in range(NUM_GPUS):
                peer = 1 - rank
                self.ops[rank].p2p_arrive(instances[rank], peer)
            for rank in range(NUM_GPUS):
                peer = 1 - rank
                self.ops[rank].p2p_wait(instances[rank], peer)
        self._synchronize_all()

    def test_p2p_barrier_matches_split(self):
        """p2pBarrier (arrive+wait fused at the host) and the explicit arrive/wait
        pair are interchangeable: interleaving them across rounds completes."""
        instances = self._setup_p2p()
        for round_idx in range(10):
            if round_idx % 2 == 0:
                for rank in range(NUM_GPUS):
                    self.ops[rank].p2p_barrier(instances[rank])
            else:
                for rank in range(NUM_GPUS):
                    self.ops[rank].p2p_arrive(instances[rank])
                for rank in range(NUM_GPUS):
                    self.ops[rank].p2p_wait(instances[rank])
        self._synchronize_all()
