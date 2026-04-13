import torch
import pytest


@pytest.mark.parametrize("batch,out_dim,k,value", [
    (1, 16, 4, 0.0),
    (2, 32, 8, 1.0),
    (4, 64, 3, -5.0),
])
def test_scatter_scalar_basic(glm, device, batch, out_dim, k, value):
    out = torch.zeros(batch, out_dim, dtype=torch.bfloat16, device=device)
    indices = torch.randint(0, out_dim, (batch, k), device=device, dtype=torch.int32)

    glm.scatter_scalar(out, indices, value, k, out_dim, batch)

    ref = torch.zeros(batch, out_dim, dtype=torch.bfloat16, device=device)
    ref.scatter_(1, indices, value)
    assert torch.equal(out.cpu(), ref.cpu())


def test_scatter_scalar_neg_inf(glm, device):
    batch, seq_len, topk = 2, 16, 4
    out = torch.zeros(batch, seq_len, dtype=torch.bfloat16, device=device)
    indices = torch.randint(0, seq_len, (batch, topk), device=device, dtype=torch.int32)

    glm.scatter_scalar(out, indices, float('-inf'), topk, seq_len, batch)

    for b in range(batch):
        for i in range(topk):
            idx = indices[b, i].item()
            assert out[b, idx].item() == float('-inf') or out[b, idx].isinf()


def test_scatter_scalar_overwrite(glm, device):
    batch, out_dim, k = 1, 8, 3
    out = torch.full((batch, out_dim), 99.0, dtype=torch.bfloat16, device=device)
    indices = torch.tensor([[0, 3, 7]], device=device, dtype=torch.int32)

    glm.scatter_scalar(out, indices, 0.0, k, out_dim, batch)

    assert out[0, 0].item() == 0.0
    assert out[0, 3].item() == 0.0
    assert out[0, 7].item() == 0.0
    assert out[0, 1].item() == pytest.approx(99.0, abs=1e-2)
