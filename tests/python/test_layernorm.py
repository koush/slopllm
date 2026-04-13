import torch
import pytest


def torch_layernorm(x, weight, bias, eps):
    return torch.nn.functional.layer_norm(x.float(), [x.shape[-1]],
                                          weight.float(), bias.float() if bias is not None else None,
                                          eps).to(torch.bfloat16)


@pytest.mark.parametrize("batch,dim", [(1, 64), (4, 128), (2, 512), (8, 256)])
def test_layernorm_with_bias(glm, device, batch, dim):
    x = torch.randn(batch, dim, dtype=torch.bfloat16, device=device)
    w = torch.randn(dim, dtype=torch.bfloat16, device=device)
    b = torch.randn(dim, dtype=torch.bfloat16, device=device)
    eps = 1e-5
    out = torch.empty_like(x)

    glm.layernorm(out, x, w, b, eps, dim, batch)
    ref = torch_layernorm(x, w, b, eps)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("batch,dim", [(1, 64), (4, 128)])
def test_layernorm_no_bias(glm, device, batch, dim):
    x = torch.randn(batch, dim, dtype=torch.bfloat16, device=device)
    w = torch.randn(dim, dtype=torch.bfloat16, device=device)
    eps = 1e-5
    out = torch.empty_like(x)

    glm.layernorm(out, x, w, None, eps, dim, batch)
    ref = torch_layernorm(x, w, None, eps)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=5e-3, rtol=5e-3)


def test_layernorm_model_weight(glm, device):
    hidden = 64
    batch = 2
    x = torch.randn(batch, hidden, dtype=torch.bfloat16, device=device) * 2.0
    w = torch.ones(hidden, dtype=torch.bfloat16, device=device)
    b = torch.zeros(hidden, dtype=torch.bfloat16, device=device)
    eps = 1e-6
    out = torch.empty_like(x)

    glm.layernorm(out, x, w, b, eps, hidden, batch)
    ref = torch_layernorm(x, w, b, eps)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=5e-3, rtol=5e-3)
