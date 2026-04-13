import torch
import pytest


@pytest.mark.parametrize("batch,in_dim,k", [(1, 16, 4), (4, 32, 8), (2, 64, 3)])
def test_gather_basic(glm, device, batch, in_dim, k):
    x = torch.randn(batch, in_dim, dtype=torch.bfloat16, device=device)
    indices = torch.randint(0, in_dim, (batch, k), device=device, dtype=torch.int32)
    out = torch.empty(batch, k, dtype=torch.bfloat16, device=device)

    glm.gather(out, x, indices, k, in_dim, batch)

    ref = torch.gather(x, 1, indices)
    assert torch.equal(out.cpu(), ref.cpu())


def test_gather_single_element(glm, device):
    x = torch.randn(1, 10, dtype=torch.bfloat16, device=device)
    indices = torch.tensor([[5]], device=device, dtype=torch.int32)
    out = torch.empty(1, 1, dtype=torch.bfloat16, device=device)

    glm.gather(out, x, indices, 1, 10, 1)
    assert torch.equal(out.cpu(), x[:, 5:6].cpu())


def test_gather_all_indices(glm, device):
    batch, dim = 2, 8
    x = torch.randn(batch, dim, dtype=torch.bfloat16, device=device)
    indices = torch.arange(dim, device=device, dtype=torch.int32).unsqueeze(0).expand(batch, -1).contiguous()
    out = torch.empty(batch, dim, dtype=torch.bfloat16, device=device)

    glm.gather(out, x, indices, dim, dim, batch)
    assert torch.equal(out.cpu(), x.cpu())
