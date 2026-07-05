import torch
import pytest


def gather_pages_ref(src, page_indices, page_indptr, last_page_len, batch_size, page_size):
    """Reference implementation: gather pages for all sequences into contiguous buffer."""
    out_len = 0
    for seq in range(batch_size):
        seq_pages = page_indptr[seq].item()
        seq_page_end = page_indptr[seq + 1].item()
        num_pages = seq_page_end - seq_pages
        seq_len = (num_pages - 1) * page_size + last_page_len[seq].item()
        out_len += seq_len
    out = torch.empty(out_len, src.shape[-1], dtype=src.dtype, device=src.device)
    out_offset = 0
    for seq in range(batch_size):
        seq_pages = page_indptr[seq].item()
        seq_page_end = page_indptr[seq + 1].item()
        num_pages = seq_page_end - seq_pages
        for p in range(num_pages):
            src_page = page_indices[seq_pages + p].item()
            copy_len = last_page_len[seq].item() if (p == num_pages - 1) else page_size
            out[out_offset : out_offset + copy_len] = src[src_page, :copy_len]
            out_offset += copy_len
    return out


@pytest.mark.parametrize("num_pages,page_size,D", [
    (1, 16, 576),
    (4, 16, 576),
    (8, 16, 576),
    (100, 16, 576),
    (1, 16, 512),
    (1, 16, 64),
    (8, 32, 576),
    (8, 256, 576),
    (4, 16, 100),
    (4, 16, 3),
])
def test_gather_pages_correctness(glm, device, num_pages, page_size, D):
    torch.manual_seed(42)
    max_pages = num_pages + 10
    src = torch.randn(max_pages, page_size, D, dtype=torch.bfloat16, device=device)
    page_indices = torch.randperm(max_pages)[:num_pages].to(torch.int32).to(device)
    page_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    last_page_len = torch.tensor([page_size], dtype=torch.int32, device=device)

    out = torch.empty(num_pages * page_size, D, dtype=torch.bfloat16, device=device)
    glm.gather_pages(out, src, page_indices, page_indptr, last_page_len, num_pages, 1, page_size, D)
    glm.synchronize()

    ref = src[page_indices.cpu().numpy()].reshape(num_pages * page_size, D).to(device)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=0, rtol=0)


def test_gather_pages_single_page(glm, device):
    num_pages, page_size, D = 1, 16, 576
    src = torch.randn(10, page_size, D, dtype=torch.bfloat16, device=device)
    page_indices = torch.tensor([3], dtype=torch.int32, device=device)
    page_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    last_page_len = torch.tensor([page_size], dtype=torch.int32, device=device)
    out = torch.empty(page_size, D, dtype=torch.bfloat16, device=device)
    glm.gather_pages(out, src, page_indices, page_indptr, last_page_len, num_pages, 1, page_size, D)
    glm.synchronize()
    torch.testing.assert_close(out.cpu(), src[3].cpu(), atol=0, rtol=0)


def test_gather_pages_sequential(glm, device):
    """Sequential page indices — should be identity"""
    num_pages, page_size, D = 8, 16, 576
    src = torch.randn(num_pages, page_size, D, dtype=torch.bfloat16, device=device)
    page_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    page_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    last_page_len = torch.tensor([page_size], dtype=torch.int32, device=device)
    out = torch.empty(num_pages * page_size, D, dtype=torch.bfloat16, device=device)
    glm.gather_pages(out, src, page_indices, page_indptr, last_page_len, num_pages, 1, page_size, D)
    glm.synchronize()
    ref = src.reshape(num_pages * page_size, D)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=0, rtol=0)


def test_gather_pages_large(glm, device):
    """Large: 128k tokens = 512 pages × 256 page_size"""
    num_pages, page_size, D = 512, 256, 576
    max_pages = num_pages + 50
    src = torch.randn(max_pages, page_size, D, dtype=torch.bfloat16, device=device)
    page_indices = torch.randperm(max_pages)[:num_pages].to(torch.int32).to(device)
    page_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    last_page_len = torch.tensor([page_size], dtype=torch.int32, device=device)
    out = torch.empty(num_pages * page_size, D, dtype=torch.bfloat16, device=device)
    glm.gather_pages(out, src, page_indices, page_indptr, last_page_len, num_pages, 1, page_size, D)
    glm.synchronize()
    ref = src[page_indices.cpu().numpy()].reshape(num_pages * page_size, D).to(device)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=0, rtol=0)


def test_gather_pages_d3(glm, device):
    """D=3, tests scalar fallback path"""
    num_pages, page_size, D = 4, 16, 3
    src = torch.randn(10, page_size, D, dtype=torch.bfloat16, device=device)
    page_indices = torch.tensor([7, 2, 5, 0], dtype=torch.int32, device=device)
    page_indptr = torch.tensor([0, 4], dtype=torch.int32, device=device)
    last_page_len = torch.tensor([16], dtype=torch.int32, device=device)
    out = torch.empty(64, D, dtype=torch.bfloat16, device=device)
    glm.gather_pages(out, src, page_indices, page_indptr, last_page_len, num_pages, 1, page_size, D)
    glm.synchronize()
    ref = src[page_indices.cpu().numpy()].reshape(num_pages * page_size, D).to(device)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=0, rtol=0)


def test_gather_pages_d100(glm, device):
    """D=100 = 12*8 + 4, tests scalar fallback path"""
    num_pages, page_size, D = 4, 16, 100
    src = torch.randn(10, page_size, D, dtype=torch.bfloat16, device=device)
    page_indices = torch.tensor([7, 2, 5, 0], dtype=torch.int32, device=device)
    page_indptr = torch.tensor([0, 4], dtype=torch.int32, device=device)
    last_page_len = torch.tensor([16], dtype=torch.int32, device=device)
    out = torch.empty(64, D, dtype=torch.bfloat16, device=device)
    glm.gather_pages(out, src, page_indices, page_indptr, last_page_len, num_pages, 1, page_size, D)
    glm.synchronize()
    ref = src[page_indices.cpu().numpy()].reshape(num_pages * page_size, D).to(device)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=0, rtol=0)


def test_gather_pages_partial_last_page(glm, device):
    """Last page is partially filled — only last_page_len tokens should be copied"""
    num_pages, page_size, D = 5, 16, 576
    last_page_len_val = 7
    total_len = (num_pages - 1) * page_size + last_page_len_val
    max_pages = num_pages + 10
    src = torch.randn(max_pages, page_size, D, dtype=torch.bfloat16, device=device)
    page_indices = torch.randperm(max_pages)[:num_pages].to(torch.int32).to(device)
    page_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    last_page_len = torch.tensor([last_page_len_val], dtype=torch.int32, device=device)
    out = torch.empty(total_len, D, dtype=torch.bfloat16, device=device)
    glm.gather_pages(out, src, page_indices, page_indptr, last_page_len, num_pages, 1, page_size, D)
    glm.synchronize()
    ref = gather_pages_ref(src, page_indices, page_indptr, last_page_len, 1, page_size)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=0, rtol=0)


def test_gather_pages_multi_sequence(glm, device):
    """Multiple sequences with different page counts and partial last pages"""
    torch.manual_seed(123)
    page_size, D = 16, 576
    seq_page_counts = [3, 5, 1]
    seq_last_page_lens = [5, 16, 1]
    num_pages = sum(seq_page_counts)
    total_len = sum((c - 1) * page_size + l for c, l in zip(seq_page_counts, seq_last_page_lens))
    max_pages = num_pages + 10
    src = torch.randn(max_pages, page_size, D, dtype=torch.bfloat16, device=device)
    page_indices = torch.randperm(max_pages)[:num_pages].to(torch.int32).to(device)
    page_indptr = torch.tensor([0, 3, 8, 9], dtype=torch.int32, device=device)
    last_page_len = torch.tensor(seq_last_page_lens, dtype=torch.int32, device=device)
    out = torch.empty(total_len, D, dtype=torch.bfloat16, device=device)
    glm.gather_pages(out, src, page_indices, page_indptr, last_page_len, num_pages, 3, page_size, D)
    glm.synchronize()
    ref = gather_pages_ref(src, page_indices, page_indptr, last_page_len, 3, page_size)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=0, rtol=0)


def test_gather_pages_random(glm, device):
    """Random sizes with multi-sequence and partial pages"""
    torch.manual_seed(999)
    for _ in range(20):
        batch_size = torch.randint(1, 5, (1,)).item()
        page_size = torch.randint(1, 257, (1,)).item()
        D = torch.randint(1, 600, (1,)).item()
        seq_page_counts = [torch.randint(1, 20, (1,)).item() for _ in range(batch_size)]
        seq_last_page_lens = [torch.randint(1, page_size + 1, (1,)).item() for _ in range(batch_size)]
        num_pages = sum(seq_page_counts)
        total_len = sum((c - 1) * page_size + l for c, l in zip(seq_page_counts, seq_last_page_lens))
        max_pages = num_pages + 10
        src = torch.randn(max_pages, page_size, D, dtype=torch.bfloat16, device=device)
        page_indices = torch.randperm(max_pages)[:num_pages].to(torch.int32).to(device)
        page_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        for i in range(batch_size):
            page_indptr[i + 1] = page_indptr[i] + seq_page_counts[i]
        last_page_len = torch.tensor(seq_last_page_lens, dtype=torch.int32, device=device)
        out = torch.empty(total_len, D, dtype=torch.bfloat16, device=device)
        glm.gather_pages(out, src, page_indices, page_indptr, last_page_len, num_pages, batch_size, page_size, D)
        glm.synchronize()
        ref = gather_pages_ref(src, page_indices, page_indptr, last_page_len, batch_size, page_size)
        torch.testing.assert_close(out.cpu(), ref.cpu(), atol=0, rtol=0)


def test_gather_pages_cp_pipeline(glm, device):
    """End-to-end CP simulation: multi-sequence gather → all-gather → deinterleave on single GPU.

    Simulates world_size GPUs by:
    1. Splitting each page across ws shards (effPageSize = pageSize / ws)
    2. Each "rank" gathers its shard from the page table into a local buffer
    3. Concatenating local buffers (simulates NCCL all-gather)
    4. Deinterleaving to recover sequential token order

    Tests multiple sequences with partial last pages and gaps in the gathered buffer.
    """
    torch.manual_seed(789)
    for _ in range(20):
        ws = torch.randint(2, 9, (1,)).item()
        page_size = ws * torch.randint(4, 33, (1,)).item()  # must be divisible by ws
        D = torch.randint(1, 600, (1,)).item()
        batch_size = torch.randint(1, 4, (1,)).item()
        seq_page_counts = [torch.randint(1, 20, (1,)).item() for _ in range(batch_size)]
        seq_last_page_lens = [torch.randint(1, page_size + 1, (1,)).item() for _ in range(batch_size)]
        num_pages = sum(seq_page_counts)
        seq_lens = [(c - 1) * page_size + l for c, l in zip(seq_page_counts, seq_last_page_lens)]
        total_len = sum(seq_lens)
        max_pages = num_pages + 10
        src = torch.randn(max_pages, page_size, D, dtype=torch.bfloat16, device=device)
        page_indices = torch.randperm(max_pages)[:num_pages].to(torch.int32).to(device)
        page_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        for i in range(batch_size):
            page_indptr[i + 1] = page_indptr[i] + seq_page_counts[i]
        last_page_len = torch.tensor(seq_last_page_lens, dtype=torch.int32, device=device)
        kv_token_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        for i in range(batch_size):
            kv_token_indptr[i + 1] = kv_token_indptr[i] + seq_lens[i]

        eff_page_size = page_size // ws
        local_len = num_pages * eff_page_size

        # Per-rank paged data: each rank stores every ws-th slot within each page
        local_bufs = []
        for r in range(ws):
            rank_src = src[:, r::ws, :].contiguous()
            rank_last_page_len = torch.tensor(
                [(l + ws - 1 - r) // ws if l > r else 0 for l in seq_last_page_lens],
                dtype=torch.int32, device=device,
            )
            local_out = torch.empty(local_len, D, dtype=torch.bfloat16, device=device)
            glm.gather_pages(local_out, rank_src, page_indices, page_indptr, rank_last_page_len,
                             num_pages, batch_size, eff_page_size, D)
            local_bufs.append(local_out)

        glm.synchronize()

        # Simulate all-gather: concatenate local buffers
        gathered = torch.cat(local_bufs, dim=0)
        global_len = gathered.shape[0]

        # Deinterleave
        out = torch.empty(global_len, D, dtype=torch.bfloat16, device=device)
        glm.deinterleave(out, gathered, ws, total_len, global_len, kv_token_indptr, batch_size, D)
        glm.synchronize()

        # Reference: standard gather_pages on the full (non-sharded) source
        ref = gather_pages_ref(src, page_indices, page_indptr, last_page_len, batch_size, page_size)
        torch.testing.assert_close(out[:total_len].cpu(), ref.cpu(), atol=0, rtol=0)
