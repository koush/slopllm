import torch
import pytest
import numpy as np
from helpers import GlmOps

PAGE_SIZE = 64


def ref_topk_to_slots(topk_idx, page_indices, page_indptr, last_page_len,
                      batch_indices, topk, page_size, cp_world_size=1, cp_rank=0):
    """Reference implementation on CPU."""
    num_tokens = topk_idx.shape[0]
    slots = np.full((num_tokens, topk), -1, dtype=np.int32)

    for t in range(num_tokens):
        seq = batch_indices[t]
        num_pages = page_indptr[seq + 1] - page_indptr[seq]
        eps = page_size // cp_world_size if cp_world_size > 1 else page_size
        local_kv_len = (num_pages - 1) * eps + last_page_len[seq]

        for k in range(topk):
            token_pos = topk_idx[t, k]
            if token_pos < 0:
                continue

            if cp_world_size > 1:
                if token_pos % cp_world_size != cp_rank:
                    continue
                local_pos = (token_pos - cp_rank) // cp_world_size
            else:
                local_pos = token_pos

            if local_pos >= local_kv_len:
                continue

            page_idx_in_seq = local_pos // eps
            offset_in_page = local_pos % eps
            abs_page = page_indices[page_indptr[seq] + page_idx_in_seq]
            slots[t, k] = abs_page * page_size + offset_in_page

    return slots


def valid_set(arr):
    """Per-row set of valid (>=0) slots. topk_to_slots now compacts valid slots to
    the front in arbitrary order, so only the set is well-defined, not position."""
    return [set(int(x) for x in row if x >= 0) for row in np.asarray(arr)]


@pytest.mark.parametrize("cp_world_size,cp_rank", [
    (1, 0),
    (2, 0),
    (2, 1),
    (4, 1),
    (8, 3),
])
def test_topk_to_slots_basic(glm, device, cp_world_size, cp_rank):
    """Basic correctness: single sequence, verify slot conversion."""
    page_size = 64
    eps = page_size // cp_world_size if cp_world_size > 1 else page_size
    num_pages = 5
    kv_len = num_pages * eps  # full pages

    page_indices = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32, device=device)
    page_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    last_page_len = torch.tensor([eps], dtype=torch.int32, device=device)
    batch_indices = torch.zeros(1, dtype=torch.int32, device=device)

    topk = 8
    global_kv_len = kv_len * cp_world_size
    topk_idx = torch.tensor([[0, 1, 3, cp_world_size, cp_world_size+1, 2*cp_world_size, global_kv_len-1, global_kv_len]],
                            dtype=torch.int32, device=device)

    slots = torch.full((1, topk), -1, dtype=torch.int32, device=device)

    glm.topk_to_slots(
        slots.data_ptr(), topk_idx.data_ptr(),
        page_indices.data_ptr(), page_indptr.data_ptr(),
        last_page_len.data_ptr(), batch_indices.data_ptr(),
        1, topk, page_size, cp_world_size, cp_rank,
    )
    torch.cuda.synchronize(device)

    ref = ref_topk_to_slots(
        topk_idx.cpu().numpy(), page_indices.cpu().numpy(),
        page_indptr.cpu().numpy(), last_page_len.cpu().numpy(),
        batch_indices.cpu().numpy(), topk, page_size, cp_world_size, cp_rank,
    )

    actual = slots.cpu().numpy()
    assert valid_set(actual) == valid_set(ref), f"mismatch:\n actual={actual}\n ref={ref}"


def test_topk_to_slots_multi_seq(glm, device):
    """Multiple sequences with different page tables."""
    page_size = 64
    seq_lens = [128, 30, 200]
    page_indptr_h = [0]
    page_indices_h = []
    last_page_len_h = []

    for sl in seq_lens:
        npages = (sl + page_size - 1) // page_size
        page_indices_h.extend(range(100 + page_indptr_h[-1], 100 + page_indptr_h[-1] + npages))
        page_indptr_h.append(page_indptr_h[-1] + npages)
        rem = sl % page_size
        last_page_len_h.append(rem if rem > 0 else page_size)

    num_tokens = 3
    topk = 16
    batch_indices_h = [0, 1, 2]
    topk_idx_h = []
    for i, sl in enumerate(seq_lens):
        idxs = np.random.randint(0, sl, size=topk)
        topk_idx_h.append(idxs)

    page_indices = torch.tensor(page_indices_h, dtype=torch.int32, device=device)
    page_indptr = torch.tensor(page_indptr_h, dtype=torch.int32, device=device)
    last_page_len = torch.tensor(last_page_len_h, dtype=torch.int32, device=device)
    batch_indices = torch.tensor(batch_indices_h, dtype=torch.int32, device=device)
    topk_idx = torch.tensor(np.stack(topk_idx_h), dtype=torch.int32, device=device)
    slots = torch.full((num_tokens, topk), -1, dtype=torch.int32, device=device)

    glm.topk_to_slots(
        slots.data_ptr(), topk_idx.data_ptr(),
        page_indices.data_ptr(), page_indptr.data_ptr(),
        last_page_len.data_ptr(), batch_indices.data_ptr(),
        num_tokens, topk, page_size, 1, 0,
    )
    torch.cuda.synchronize(device)

    ref = ref_topk_to_slots(
        topk_idx.cpu().numpy(), page_indices.cpu().numpy(),
        page_indptr.cpu().numpy(), last_page_len.cpu().numpy(),
        batch_indices.cpu().numpy(), topk, page_size, 1, 0,
    )

    actual = slots.cpu().numpy()
    assert valid_set(actual) == valid_set(ref), f"mismatch:\n actual={actual}\n ref={ref}"


def test_topk_to_slots_invalid(glm, device):
    """Invalid positions (negative, beyond kvLen) should produce -1."""
    page_size = 64
    num_pages = 2
    kv_len = num_pages * page_size

    page_indices = torch.tensor([5, 10], dtype=torch.int32, device=device)
    page_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    last_page_len = torch.tensor([page_size], dtype=torch.int32, device=device)
    batch_indices = torch.zeros(1, dtype=torch.int32, device=device)

    topk = 4
    topk_idx = torch.tensor([[-1, kv_len, kv_len + 10, 0]], dtype=torch.int32, device=device)
    slots = torch.full((1, topk), 0, dtype=torch.int32, device=device)

    glm.topk_to_slots(
        slots.data_ptr(), topk_idx.data_ptr(),
        page_indices.data_ptr(), page_indptr.data_ptr(),
        last_page_len.data_ptr(), batch_indices.data_ptr(),
        1, topk, page_size, 1, 0,
    )
    torch.cuda.synchronize(device)

    actual = slots.cpu().numpy()
    # Only pos 0 is valid; the rest (-1, >=kv_len) are dropped.
    assert valid_set(actual) == [{5 * page_size}], f"got {actual}"


def test_topk_to_slots_partial_page(glm, device):
    """Sequence with a partial last page."""
    page_size = 64
    kv_len = 100  # 1 full page + 36 tokens in second page
    page_indices = torch.tensor([7, 3], dtype=torch.int32, device=device)
    page_indptr = torch.tensor([0, 2], dtype=torch.int32, device=device)
    last_page_len = torch.tensor([36], dtype=torch.int32, device=device)
    batch_indices = torch.zeros(1, dtype=torch.int32, device=device)

    topk = 4
    topk_idx = torch.tensor([[0, 63, 64, 99]], dtype=torch.int32, device=device)
    slots = torch.full((1, topk), -1, dtype=torch.int32, device=device)

    glm.topk_to_slots(
        slots.data_ptr(), topk_idx.data_ptr(),
        page_indices.data_ptr(), page_indptr.data_ptr(),
        last_page_len.data_ptr(), batch_indices.data_ptr(),
        1, topk, page_size, 1, 0,
    )
    torch.cuda.synchronize(device)

    actual = slots.cpu().numpy()
    assert valid_set(actual) == [{7*64+0, 7*64+63, 3*64+0, 3*64+35}], f"got {actual}"
