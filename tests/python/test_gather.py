import torch
import pytest

BF16 = 2
F32 = 4
I32 = 4


@pytest.mark.parametrize("batch,in_dim,k", [(1, 16, 4), (4, 32, 8), (2, 64, 3), (3, 10, 10)])
def test_gather_bf16(glm, device, batch, in_dim, k):
    torch.manual_seed(42 + batch * 100 + in_dim + k)
    x = torch.randn(batch, in_dim, dtype=torch.bfloat16, device=device)
    indices = torch.randint(0, in_dim, (batch, k), dtype=torch.int32, device=device)
    out = torch.empty(batch, k, dtype=torch.bfloat16, device=device)

    glm.gather(out, x, indices, k, in_dim, batch, BF16)

    ref = torch.gather(x, 1, indices)
    assert torch.equal(out.cpu(), ref.cpu()), f"BF16 gather mismatch"


@pytest.mark.parametrize("batch,in_dim,k", [(1, 16, 4), (4, 32, 8), (2, 64, 3)])
def test_gather_i32(glm, device, batch, in_dim, k):
    torch.manual_seed(42 + batch * 100 + in_dim + k)
    x = torch.randint(0, 1000, (batch, in_dim), dtype=torch.int32, device=device)
    indices = torch.randint(0, in_dim, (batch, k), dtype=torch.int32, device=device)
    out = torch.empty(batch, k, dtype=torch.int32, device=device)

    glm.gather(out, x, indices, k, in_dim, batch, I32)

    ref = torch.gather(x, 1, indices)
    assert torch.equal(out.cpu(), ref.cpu()), f"I32 gather mismatch"


@pytest.mark.parametrize("batch,in_dim,k", [(1, 16, 4), (4, 32, 8), (2, 64, 3)])
def test_gather_f32(glm, device, batch, in_dim, k):
    torch.manual_seed(42 + batch * 100 + in_dim + k)
    x = torch.randn(batch, in_dim, dtype=torch.float32, device=device)
    indices = torch.randint(0, in_dim, (batch, k), dtype=torch.int32, device=device)
    out = torch.empty(batch, k, dtype=torch.float32, device=device)

    glm.gather(out, x, indices, k, in_dim, batch, F32)

    ref = torch.gather(x, 1, indices)
    assert torch.equal(out.cpu(), ref.cpu()), f"F32 gather mismatch"


def test_gather_simple(glm, device):
    a = torch.tensor([[10, 11, 12, 13]], dtype=torch.bfloat16, device=device)
    b = torch.tensor([[0, 0, 1, 1]], dtype=torch.int32, device=device)
    out = torch.empty(1, 4, dtype=torch.bfloat16, device=device)

    glm.gather(out, a, b, 4, 4, 1, BF16)

    expected = torch.tensor([[10, 10, 11, 11]], dtype=torch.bfloat16, device=device)
    assert torch.equal(out.cpu(), expected.cpu()), f"Simple gather mismatch"


def test_gather_diverse_indices_bf16(glm, device):
    a = torch.tensor([
        [3.5, -1.0, 0.0, 7.25, 100.0, -50.0, 0.125, 1.0],
    ], dtype=torch.bfloat16, device=device)
    indices = torch.tensor([[7, 3, 0, 5, 2, 1, 6, 4]], dtype=torch.int32, device=device)
    out = torch.empty(1, 8, dtype=torch.bfloat16, device=device)

    glm.gather(out, a, indices, 8, 8, 1, BF16)

    ref = torch.gather(a, 1, indices)
    assert torch.equal(out.cpu(), ref.cpu()), f"Diverse indices BF16 mismatch"


def test_gather_diverse_indices_i32(glm, device):
    a = torch.tensor([
        [42, -999, 0, 1, 2147483647, -2147483648, 12345, -1],
    ], dtype=torch.int32, device=device)
    indices = torch.tensor([[5, 0, 7, 4, 1, 3, 6, 2]], dtype=torch.int32, device=device)
    out = torch.empty(1, 8, dtype=torch.int32, device=device)

    glm.gather(out, a, indices, 8, 8, 1, I32)

    ref = torch.gather(a, 1, indices)
    assert torch.equal(out.cpu(), ref.cpu()), f"Diverse indices I32 mismatch"


def test_gather_diverse_indices_f32(glm, device):
    a = torch.tensor([
        [1e-6, -1e6, 3.14159, 2.71828, 0.0, -0.0, 1e38, -1e38],
    ], dtype=torch.float32, device=device)
    indices = torch.tensor([[6, 2, 0, 7, 4, 1, 5, 3]], dtype=torch.int32, device=device)
    out = torch.empty(1, 8, dtype=torch.float32, device=device)

    glm.gather(out, a, indices, 8, 8, 1, F32)

    ref = torch.gather(a, 1, indices)
    assert torch.equal(out.cpu(), ref.cpu()), f"Diverse indices F32 mismatch"


def test_gather_duplicated_indices(glm, device):
    a = torch.tensor([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [10.0, 20.0, 30.0, 40.0, 50.0],
    ], dtype=torch.bfloat16, device=device)
    indices = torch.tensor([
        [4, 4, 0, 0, 2, 2],
        [1, 1, 3, 3, 0, 0],
    ], dtype=torch.int32, device=device)
    out = torch.empty(2, 6, dtype=torch.bfloat16, device=device)

    glm.gather(out, a, indices, 6, 5, 2, BF16)

    ref = torch.gather(a, 1, indices)
    assert torch.equal(out.cpu(), ref.cpu()), f"Duplicated indices mismatch"


def test_gather_last_index(glm, device):
    a = torch.tensor([
        [0.0, 0.0, 0.0, 99.0],
        [0.0, 0.0, 0.0, -1.0],
    ], dtype=torch.bfloat16, device=device)
    indices = torch.tensor([[3], [3]], dtype=torch.int32, device=device)
    out = torch.empty(2, 1, dtype=torch.bfloat16, device=device)

    glm.gather(out, a, indices, 1, 4, 2, BF16)

    ref = torch.gather(a, 1, indices)
    assert torch.equal(out.cpu(), ref.cpu()), f"Last index gather mismatch"


def test_gather_k_equals_dim(glm, device):
    batch, dim = 2, 8
    x = torch.randn(batch, dim, dtype=torch.bfloat16, device=device)
    perm = torch.randperm(dim).unsqueeze(0).expand(batch, -1).clone().to(device).int()
    out = torch.empty(batch, dim, dtype=torch.bfloat16, device=device)

    glm.gather(out, x, perm, dim, dim, batch, BF16)

    ref = torch.gather(x, 1, perm)
    assert torch.equal(out.cpu(), ref.cpu()), f"k==dim gather mismatch"


def test_gather_single_index(glm, device):
    batch, in_dim = 4, 32
    k = 1
    x = torch.randn(batch, in_dim, dtype=torch.bfloat16, device=device)
    indices = torch.randint(0, in_dim, (batch, k), dtype=torch.int32, device=device)
    out = torch.empty(batch, k, dtype=torch.bfloat16, device=device)

    glm.gather(out, x, indices, k, in_dim, batch, BF16)

    ref = torch.gather(x, 1, indices)
    assert torch.equal(out.cpu(), ref.cpu()), f"Single index gather mismatch"
