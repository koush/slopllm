import torch
import pytest


@pytest.mark.parametrize("batch,dim", [(1, 16), (4, 32), (2, 64), (8, 128)])
def test_max_basic(glm, device, batch, dim):
    torch.manual_seed(42 + batch * 100 + dim)
    x = torch.randn(batch, dim, dtype=torch.bfloat16, device=device)
    out_vals = torch.empty(batch, dtype=torch.bfloat16, device=device)
    out_idxs = torch.empty(batch, dtype=torch.int32, device=device)

    glm.max(out_vals, out_idxs, x, dim, batch)

    ref_vals, ref_idxs = x.float().max(dim=-1)

    for b in range(batch):
        assert out_idxs[b].item() == ref_idxs[b].item(), \
            f"Index mismatch at batch {b}: cuda={out_idxs[b].item()}, ref={ref_idxs[b].item()}"
        assert abs(out_vals[b].float().item() - ref_vals[b].item()) < 0.01, \
            f"Value mismatch at batch {b}: cuda={out_vals[b].float().item()}, ref={ref_vals[b].item()}"


def test_max_consistent_with_argmax(glm, device):
    batch, dim = 8, 256
    torch.manual_seed(123)
    x = torch.randn(batch, dim, dtype=torch.bfloat16, device=device)
    out_vals = torch.empty(batch, dtype=torch.bfloat16, device=device)
    out_idxs = torch.empty(batch, dtype=torch.int32, device=device)
    argmax_idxs = torch.empty(batch, dtype=torch.int32, device=device)

    glm.max(out_vals, out_idxs, x, dim, batch)
    glm.argmax(argmax_idxs, x, dim, batch)

    for b in range(batch):
        assert out_idxs[b].item() == argmax_idxs[b].item(), \
            f"max index != argmax at batch {b}: max={out_idxs[b].item()}, argmax={argmax_idxs[b].item()}"


def test_max_tiebreaking(glm, device):
    batch, dim = 1, 8
    x_vals = [3.0, 1.0, 3.0, 2.0, 3.0, 0.5, 2.0, 0.25]
    x = torch.tensor([x_vals], dtype=torch.bfloat16, device=device)
    out_vals = torch.empty(batch, dtype=torch.bfloat16, device=device)
    out_idxs = torch.empty(batch, dtype=torch.int32, device=device)

    glm.max(out_vals, out_idxs, x, dim, batch)

    ref_val, ref_idx = x.float().max(dim=-1)
    assert out_idxs[0].item() == ref_idx.item(), \
        f"Tiebreak index mismatch: cuda={out_idxs[0].item()}, ref={ref_idx.item()}"
    assert abs(out_vals[0].float().item() - ref_val.item()) < 0.01, \
        f"Value mismatch: cuda={out_vals[0].float().item()}, ref={ref_val.item()}"


def test_max_single_element(glm, device):
    batch, dim = 1, 1
    x = torch.tensor([[5.0]], dtype=torch.bfloat16, device=device)
    out_vals = torch.empty(batch, dtype=torch.bfloat16, device=device)
    out_idxs = torch.empty(batch, dtype=torch.int32, device=device)

    glm.max(out_vals, out_idxs, x, dim, batch)

    assert out_idxs[0].item() == 0
    assert abs(out_vals[0].float().item() - 5.0) < 0.01


def test_max_negative_values(glm, device):
    batch, dim = 3, 16
    torch.manual_seed(99)
    x = -torch.abs(torch.randn(batch, dim, dtype=torch.bfloat16, device=device))
    out_vals = torch.empty(batch, dtype=torch.bfloat16, device=device)
    out_idxs = torch.empty(batch, dtype=torch.int32, device=device)

    glm.max(out_vals, out_idxs, x, dim, batch)

    ref_vals, ref_idxs = x.float().max(dim=-1)

    for b in range(batch):
        assert out_idxs[b].item() == ref_idxs[b].item(), \
            f"Index mismatch at batch {b}: cuda={out_idxs[b].item()}, ref={ref_idxs[b].item()}"
        assert abs(out_vals[b].float().item() - ref_vals[b].item()) < 0.01, \
            f"Value mismatch at batch {b}: cuda={out_vals[b].float().item()}, ref={ref_vals[b].item()}"


@pytest.mark.parametrize("offset", [0, 5, 100, 256])
def test_max_offset(glm, device, offset):
    batch, dim = 4, 32
    torch.manual_seed(42 + offset)
    x = torch.randn(batch, dim, dtype=torch.bfloat16, device=device)
    out_vals = torch.empty(batch, dtype=torch.bfloat16, device=device)
    out_idxs = torch.empty(batch, dtype=torch.int32, device=device)

    glm.max(out_vals, out_idxs, x, dim, batch, offset=offset)

    ref_vals, ref_idxs = x.float().max(dim=-1)

    for b in range(batch):
        assert out_idxs[b].item() == ref_idxs[b].item() + offset, \
            f"Offset index mismatch at batch {b}: cuda={out_idxs[b].item()}, ref={ref_idxs[b].item() + offset}, offset={offset}"
        assert abs(out_vals[b].float().item() - ref_vals[b].item()) < 0.01, \
            f"Value mismatch at batch {b}: cuda={out_vals[b].float().item()}, ref={ref_vals[b].item()}"
