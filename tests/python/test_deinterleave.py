import torch
import pytest


def interleave(x, world_size):
    """Interleave: out[r * shard_len + i] = x[i * world_size + r] where shard_len = ceil(total_len / world_size)"""
    total_len, D = x.shape
    shard_len = (total_len + world_size - 1) // world_size
    result = torch.empty(shard_len * world_size, D, dtype=x.dtype, device=x.device)
    for r in range(world_size):
        shard = x[r::world_size]
        slen = min(shard_len, shard.shape[0])
        result[r * shard_len : r * shard_len + slen] = shard[:slen]
    return result[:shard_len * world_size]


def interleave_with_gaps(x, world_size, chunk_len):
    """Simulate CP gather → all-gather: each rank's valid tokens packed at front of a chunk_len-sized slot.
    chunk_len can be larger than ceil(total_len / world_size) — remaining rows are garbage (gaps)."""
    total_len, D = x.shape
    global_len = chunk_len * world_size
    result = torch.randn(global_len, D, dtype=x.dtype, device=x.device)  # garbage fill
    for r in range(world_size):
        shard = x[r::world_size]
        result[r * chunk_len : r * chunk_len + shard.shape[0]] = shard
    return result


@pytest.mark.parametrize("total_len,ws,D", [
    (4096, 8, 576),
    (4090, 8, 576),
    (1, 8, 576),
    (7, 8, 576),
    (4096, 1, 576),
    (4096, 2, 576),
    (100, 3, 576),
    (4096, 4, 576),
    (132096, 8, 576),
    (4096, 8, 512),
    (4096, 8, 64),
    (4096, 8, 100),
    (4096, 8, 3),
])
def test_deinterleave_correctness(glm, device, total_len, ws, D):
    torch.manual_seed(42)
    x = torch.randn(total_len, D, dtype=torch.bfloat16, device=device)
    gathered = interleave(x, ws)
    out = torch.empty(gathered.shape[0], D, dtype=torch.bfloat16, device=device)

    kv_indptr = torch.tensor([0, total_len], dtype=torch.int32, device=device)
    glm.deinterleave(out, gathered, ws, total_len, gathered.shape[0], kv_indptr, 1, D)
    glm.synchronize()

    torch.testing.assert_close(out[:total_len].cpu(), x.cpu(), atol=0, rtol=0)


def test_deinterleave_d576(glm, device):
    """D=576 = 72*8, vec-aligned, no remainder"""
    total_len, ws, D = 8192, 8, 576
    x = torch.randn(total_len, D, dtype=torch.bfloat16, device=device)
    gathered = interleave(x, ws)
    out = torch.empty(gathered.shape[0], D, dtype=torch.bfloat16, device=device)

    kv_indptr = torch.tensor([0, total_len], dtype=torch.int32, device=device)
    glm.deinterleave(out, gathered, ws, total_len, gathered.shape[0], kv_indptr, 1, D)
    glm.synchronize()

    torch.testing.assert_close(out[:total_len].cpu(), x.cpu(), atol=0, rtol=0)


def test_deinterleave_d100(glm, device):
    """D=100 = 12*8 + 4, tests remainder path"""
    total_len, ws, D = 4096, 8, 100
    x = torch.randn(total_len, D, dtype=torch.bfloat16, device=device)
    gathered = interleave(x, ws)
    out = torch.empty(gathered.shape[0], D, dtype=torch.bfloat16, device=device)

    kv_indptr = torch.tensor([0, total_len], dtype=torch.int32, device=device)
    glm.deinterleave(out, gathered, ws, total_len, gathered.shape[0], kv_indptr, 1, D)
    glm.synchronize()

    torch.testing.assert_close(out[:total_len].cpu(), x.cpu(), atol=0, rtol=0)


def test_deinterleave_d3(glm, device):
    """D=3, tiny remainder"""
    total_len, ws, D = 4096, 8, 3
    x = torch.randn(total_len, D, dtype=torch.bfloat16, device=device)
    gathered = interleave(x, ws)
    out = torch.empty(gathered.shape[0], D, dtype=torch.bfloat16, device=device)

    kv_indptr = torch.tensor([0, total_len], dtype=torch.int32, device=device)
    glm.deinterleave(out, gathered, ws, total_len, gathered.shape[0], kv_indptr, 1, D)
    glm.synchronize()

    torch.testing.assert_close(out[:total_len].cpu(), x.cpu(), atol=0, rtol=0)


def test_deinterleave_large_128k(glm, device):
    """128k context + 4k chunk"""
    total_len, ws, D = 132096, 8, 576
    x = torch.randn(total_len, D, dtype=torch.bfloat16, device=device)
    gathered = interleave(x, ws)
    out = torch.empty(gathered.shape[0], D, dtype=torch.bfloat16, device=device)

    kv_indptr = torch.tensor([0, total_len], dtype=torch.int32, device=device)
    glm.deinterleave(out, gathered, ws, total_len, gathered.shape[0], kv_indptr, 1, D)
    glm.synchronize()

    torch.testing.assert_close(out.cpu(), x.cpu(), atol=0, rtol=0)


def test_deinterleave_ws1(glm, device):
    """world_size=1 is identity"""
    total_len, ws, D = 4096, 1, 576
    x = torch.randn(total_len, D, dtype=torch.bfloat16, device=device)
    gathered = interleave(x, ws)
    out = torch.empty(gathered.shape[0], D, dtype=torch.bfloat16, device=device)

    kv_indptr = torch.tensor([0, total_len], dtype=torch.int32, device=device)
    glm.deinterleave(out, gathered, ws, total_len, gathered.shape[0], kv_indptr, 1, D)
    glm.synchronize()

    torch.testing.assert_close(out[:total_len].cpu(), gathered[:total_len].cpu(), atol=0, rtol=0)


def test_deinterleave_single_token(glm, device):
    """Single token"""
    total_len, ws, D = 1, 8, 576
    x = torch.randn(total_len, D, dtype=torch.bfloat16, device=device)
    gathered = interleave(x, ws)
    out = torch.empty(gathered.shape[0], D, dtype=torch.bfloat16, device=device)

    kv_indptr = torch.tensor([0, total_len], dtype=torch.int32, device=device)
    glm.deinterleave(out, gathered, ws, total_len, gathered.shape[0], kv_indptr, 1, D)
    glm.synchronize()

    torch.testing.assert_close(out[:total_len].cpu(), x.cpu(), atol=0, rtol=0)


def test_deinterleave_non_divisible_7(glm, device):
    """7 tokens with ws=8: some shards empty"""
    total_len, ws, D = 7, 8, 576
    x = torch.randn(total_len, D, dtype=torch.bfloat16, device=device)
    gathered = interleave(x, ws)
    out = torch.empty(gathered.shape[0], D, dtype=torch.bfloat16, device=device)

    kv_indptr = torch.tensor([0, total_len], dtype=torch.int32, device=device)
    glm.deinterleave(out, gathered, ws, total_len, gathered.shape[0], kv_indptr, 1, D)
    glm.synchronize()

    torch.testing.assert_close(out[:total_len].cpu(), x.cpu(), atol=0, rtol=0)


def test_deinterleave_random(glm, device):
    """Random sizes and world sizes"""
    torch.manual_seed(123)
    for _ in range(20):
        total_len = torch.randint(1, 20000, (1,)).item()
        ws = torch.randint(1, 9, (1,)).item()
        D = torch.randint(1, 600, (1,)).item()
        x = torch.randn(total_len, D, dtype=torch.bfloat16, device=device)
        gathered = interleave(x, ws)
        out = torch.empty(gathered.shape[0], D, dtype=torch.bfloat16, device=device)

        kv_indptr = torch.tensor([0, total_len], dtype=torch.int32, device=device)
        glm.deinterleave(out, gathered, ws, total_len, gathered.shape[0], kv_indptr, 1, D)
        glm.synchronize()

        torch.testing.assert_close(out[:total_len].cpu(), x.cpu(), atol=0, rtol=0)


def test_deinterleave_gaps(glm, device):
    """Simulate CP: chunk_len larger than ceil(total_len/ws) — gaps between shards from page padding"""
    torch.manual_seed(456)
    for _ in range(20):
        total_len = torch.randint(1, 2000, (1,)).item()
        ws = torch.randint(1, 9, (1,)).item()
        D = torch.randint(1, 600, (1,)).item()
        min_chunk = (total_len + ws - 1) // ws
        chunk_len = min_chunk + torch.randint(0, 20, (1,)).item()  # extra padding
        x = torch.randn(total_len, D, dtype=torch.bfloat16, device=device)
        gathered = interleave_with_gaps(x, ws, chunk_len)
        out = torch.empty(gathered.shape[0], D, dtype=torch.bfloat16, device=device)

        kv_indptr = torch.tensor([0, total_len], dtype=torch.int32, device=device)
        glm.deinterleave(out, gathered, ws, total_len, gathered.shape[0], kv_indptr, 1, D)
        glm.synchronize()

        torch.testing.assert_close(out[:total_len].cpu(), x.cpu(), atol=0, rtol=0)
