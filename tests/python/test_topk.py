import torch
import pytest


@pytest.mark.parametrize("batch,dim,k", [(1, 16, 4), (4, 32, 8), (2, 64, 3)])
def test_topk_basic(glm, device, batch, dim, k):
    torch.manual_seed(42 + batch * 100 + dim + k)
    x = torch.randn(batch, dim, dtype=torch.bfloat16, device=device)
    out_vals = torch.empty(batch, k, dtype=torch.bfloat16, device=device)
    out_idxs = torch.empty(batch, k, dtype=torch.int32, device=device)

    glm.topk(out_vals, out_idxs, x, k, dim, batch)

    ref_vals, ref_idxs = torch.topk(x.float(), k, dim=-1)
    ref_vals = ref_vals.to(torch.bfloat16)

    for b in range(batch):
        cuda_vals_sorted, _ = out_vals[b].cpu().sort(descending=True)
        ref_vals_sorted, _ = ref_vals[b].cpu().sort(descending=True)
        assert torch.equal(cuda_vals_sorted.float(), ref_vals_sorted.float()), \
            f"Top-k values mismatch at batch {b}"

        cuda_set = set(out_idxs[b].cpu().tolist())
        ref_set = set(ref_idxs[b].cpu().tolist())
        assert cuda_set == ref_set, \
            f"Top-k indices mismatch at batch {b}: cuda={cuda_set}, ref={ref_set}"


def test_topk_small_dim(glm, device):
    batch, dim, k = 2, 8, 3
    x = torch.randn(batch, dim, dtype=torch.bfloat16, device=device)
    out_vals = torch.empty(batch, k, dtype=torch.bfloat16, device=device)
    out_idxs = torch.empty(batch, k, dtype=torch.int32, device=device)

    glm.topk(out_vals, out_idxs, x, k, dim, batch)

    ref_vals, ref_idxs = torch.topk(x.float(), k, dim=-1)
    ref_vals = ref_vals.to(torch.bfloat16)

    for b in range(batch):
        for i in range(k):
            assert out_idxs[b, i].item() == ref_idxs[b, i].item()
            assert torch.equal(out_vals[b, i:i+1].cpu(), ref_vals[b, i:i+1].cpu()), \
                f"Value mismatch at batch {b}, position {i}"


def test_topk_moe_routing(glm, device):
    batch = 4
    num_experts = 256
    top_k = 8
    logits = torch.randn(batch, num_experts, dtype=torch.bfloat16, device=device)
    out_vals = torch.empty(batch, top_k, dtype=torch.bfloat16, device=device)
    out_idxs = torch.empty(batch, top_k, dtype=torch.int32, device=device)

    glm.topk(out_vals, out_idxs, logits, top_k, num_experts, batch)

    ref_vals, ref_idxs = torch.topk(logits.float(), top_k, dim=-1, sorted=True)
    ref_vals = ref_vals.to(torch.bfloat16)

    for b in range(batch):
        cuda_vals_sorted, _ = out_vals[b].cpu().sort(descending=True)
        ref_vals_sorted, _ = ref_vals[b].cpu().sort(descending=True)
        assert torch.equal(cuda_vals_sorted.float(), ref_vals_sorted.float()), \
            f"Top-k values mismatch at batch {b}"

        cuda_set = set(out_idxs[b].cpu().tolist())
        ref_set = set(ref_idxs[b].cpu().tolist())
        overlap = len(cuda_set & ref_set)
        assert overlap >= top_k - 1, \
            f"Top-k indices overlap too small at batch {b}: {overlap}/{top_k} (cuda={cuda_set}, ref={ref_set})"


def test_topk_tiebreaking(glm, device):
    batch, dim, k = 1, 8, 4
    x_vals = [3.0, 1.0, 3.0, 2.0, 3.0, 0.5, 2.0, 0.25]
    x = torch.tensor([x_vals], dtype=torch.bfloat16, device=device)
    out_vals = torch.empty(batch, k, dtype=torch.bfloat16, device=device)
    out_idxs = torch.empty(batch, k, dtype=torch.int32, device=device)

    glm.topk(out_vals, out_idxs, x, k, dim, batch)

    ref_vals, ref_idxs = torch.topk(x.float(), k, dim=-1, sorted=False)
    ref_vals = ref_vals.to(torch.bfloat16)

    cuda_set = set(out_idxs[0].cpu().tolist())
    ref_set = set(ref_idxs[0].cpu().tolist())
    assert cuda_set == ref_set, f"Index set mismatch: cuda={cuda_set}, ref={ref_set}"

    for tied_val in [3.0, 2.0]:
        tied_cuda_idxs = [out_idxs[0, i].item() for i in range(k)
                          if abs(out_vals[0, i].item() - tied_val) < 0.01]
        assert tied_cuda_idxs == sorted(tied_cuda_idxs), \
            f"Tied value {tied_val}: indices should be ascending, got {tied_cuda_idxs}"


def test_topk_k_equals_dim(glm, device):
    batch, dim = 2, 8
    x = torch.randn(batch, dim, dtype=torch.bfloat16, device=device)
    out_vals = torch.empty(batch, dim, dtype=torch.bfloat16, device=device)
    out_idxs = torch.empty(batch, dim, dtype=torch.int32, device=device)

    glm.topk(out_vals, out_idxs, x, dim, dim, batch)

    ref_vals, ref_idxs = torch.topk(x.float(), dim, dim=-1)
    ref_vals = ref_vals.to(torch.bfloat16)

    cuda_sorted_vals, _ = out_vals[0].cpu().sort(descending=True)
    ref_sorted_vals, _ = ref_vals[0].cpu().sort(descending=True)
    assert torch.equal(cuda_sorted_vals, ref_sorted_vals)


@pytest.mark.parametrize("offset", [0, 100, 512])
def test_topk_with_offset(glm, device, offset):
    batch, dim, k = 2, 16, 4
    torch.manual_seed(42 + offset)
    x = torch.randn(batch, dim, dtype=torch.bfloat16, device=device)
    out_vals = torch.empty(batch, k, dtype=torch.bfloat16, device=device)
    out_idxs = torch.empty(batch, k, dtype=torch.int32, device=device)

    glm.topk(out_vals, out_idxs, x, k, dim, batch, offset)

    ref_vals, ref_idxs = torch.topk(x.float(), k, dim=-1)
    ref_vals = ref_vals.to(torch.bfloat16)
    ref_idxs_shifted = ref_idxs + offset

    for b in range(batch):
        cuda_vals_sorted, _ = out_vals[b].cpu().sort(descending=True)
        ref_vals_sorted, _ = ref_vals[b].cpu().sort(descending=True)
        assert torch.equal(cuda_vals_sorted.float(), ref_vals_sorted.float()), \
            f"Top-k values mismatch at batch {b}, offset={offset}"

        cuda_set = set(out_idxs[b].cpu().tolist())
        ref_set = set(ref_idxs_shifted[b].cpu().tolist())
        assert cuda_set == ref_set, \
            f"Top-k indices mismatch at batch {b}, offset={offset}: cuda={cuda_set}, ref={ref_set}"
