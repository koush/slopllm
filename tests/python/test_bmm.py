import torch
import pytest


def test_bmm_transB_small(glm, device):
    batch, M, N, K = 2, 4, 4, 8
    A = torch.randn(batch, M, K, dtype=torch.bfloat16, device=device)
    B = torch.randn(batch, N, K, dtype=torch.bfloat16, device=device)
    C_cuda = torch.empty(batch, M, N, dtype=torch.bfloat16, device=device)
    glm.bmm(C_cuda, A, B, 1.0, 0.0, batch, M, N, K, transB=1)
    C_ref = A @ B.transpose(-1, -2)
    torch.testing.assert_close(C_cuda.cpu(), C_ref.cpu(), atol=5e-2, rtol=5e-3)


def test_bmm_no_transB_small(glm, device):
    batch, M, N, K = 2, 4, 4, 8
    A = torch.randn(batch, M, K, dtype=torch.bfloat16, device=device)
    B = torch.randn(batch, K, N, dtype=torch.bfloat16, device=device)
    C_cuda = torch.empty(batch, M, N, dtype=torch.bfloat16, device=device)
    glm.bmm(C_cuda, A, B, 1.0, 0.0, batch, M, N, K, transB=0)
    C_ref = A @ B
    torch.testing.assert_close(C_cuda.cpu(), C_ref.cpu(), atol=5e-2, rtol=5e-3)


def test_bmm_transB_attention_dims(glm, device):
    batch, n_heads, seq_len, head_dim = 1, 4, 8, 16
    M = seq_len
    N = seq_len
    K = head_dim
    Q = torch.randn(batch, n_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    K_tensor = torch.randn(batch, n_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    Q_2d = Q.reshape(batch * n_heads, M, K).contiguous()
    K_2d = K_tensor.reshape(batch * n_heads, N, K).contiguous()
    attn_cuda = torch.empty(batch * n_heads, M, N, dtype=torch.bfloat16, device=device)
    glm.bmm(attn_cuda, Q_2d, K_2d, 1.0, 0.0, batch * n_heads, M, N, K, transB=1)
    attn_ref = Q_2d @ K_2d.transpose(-1, -2)
    torch.testing.assert_close(attn_cuda.cpu(), attn_ref.cpu(), atol=5e-2, rtol=5e-3)


def test_bmm_transB_with_beta(glm, device):
    batch, M, N, K = 1, 4, 4, 8
    A = torch.randn(batch, M, K, dtype=torch.bfloat16, device=device)
    B = torch.randn(batch, N, K, dtype=torch.bfloat16, device=device)
    C_init = torch.randn(batch, M, N, dtype=torch.bfloat16, device=device)
    C_cuda = C_init.clone()
    glm.bmm(C_cuda, A, B, 1.0, 1.0, batch, M, N, K, transB=1)
    C_ref = A @ B.transpose(-1, -2) + C_init
    torch.testing.assert_close(C_cuda.cpu(), C_ref.cpu(), atol=5e-2, rtol=5e-3)


def test_bmm_transB_with_scale(glm, device):
    batch, M, N, K = 2, 4, 4, 16
    A = torch.randn(batch, M, K, dtype=torch.bfloat16, device=device)
    B = torch.randn(batch, N, K, dtype=torch.bfloat16, device=device)
    C_cuda = torch.empty(batch, M, N, dtype=torch.bfloat16, device=device)
    scale = 0.125
    glm.bmm(C_cuda, A, B, scale, 0.0, batch, M, N, K, transB=1)
    C_ref = scale * (A @ B.transpose(-1, -2))
    torch.testing.assert_close(C_cuda.cpu(), C_ref.cpu(), atol=5e-2, rtol=5e-3)


def test_bmm_batch1(glm, device):
    M, N, K = 8, 8, 16
    A = torch.randn(1, M, K, dtype=torch.bfloat16, device=device)
    B = torch.randn(1, N, K, dtype=torch.bfloat16, device=device)
    C_cuda = torch.empty(1, M, N, dtype=torch.bfloat16, device=device)
    glm.bmm(C_cuda, A, B, 1.0, 0.0, 1, M, N, K, transB=1)
    C_ref = A @ B.transpose(-1, -2)
    torch.testing.assert_close(C_cuda.cpu(), C_ref.cpu(), atol=5e-2, rtol=5e-3)


def test_bmm_transA_absorbed_weight(glm, device):
    batch, M, N, K = 4, 512, 256, 192
    A = torch.randn(batch, K, M, dtype=torch.bfloat16, device=device)
    B = torch.randn(batch, K, N, dtype=torch.bfloat16, device=device)
    C_cuda = torch.empty(batch, M, N, dtype=torch.bfloat16, device=device)
    glm.bmm(C_cuda, A, B, 1.0, 0.0, batch, M, N, K, transA=1, transB=0)
    C_ref = A.transpose(-1, -2) @ B
    torch.testing.assert_close(C_cuda.cpu(), C_ref.cpu(), atol=5e-2, rtol=5e-3)


def test_bmm_transA_transB(glm, device):
    batch, M, N, K = 2, 4, 4, 8
    A = torch.randn(batch, K, M, dtype=torch.bfloat16, device=device)
    B = torch.randn(batch, N, K, dtype=torch.bfloat16, device=device)
    C_cuda = torch.empty(batch, M, N, dtype=torch.bfloat16, device=device)
    glm.bmm(C_cuda, A, B, 1.0, 0.0, batch, M, N, K, transA=1, transB=1)
    C_ref = A.transpose(-1, -2) @ B.transpose(-1, -2)
    torch.testing.assert_close(C_cuda.cpu(), C_ref.cpu(), atol=5e-2, rtol=5e-3)


@pytest.mark.parametrize("M", [1, 4, 8, 65, 512])
@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_bmm_token_major(glm, device, M, beta):
    torch.manual_seed(132)
    heads, K, N = 8, 192, 512
    A = torch.randn(M, heads, K, dtype=torch.bfloat16, device=device)
    B = torch.randn(heads, K, N, dtype=torch.bfloat16, device=device)
    initial = torch.randn(M, heads, N, dtype=torch.bfloat16, device=device)
    actual = initial.clone()
    glm.bmm(actual, A, B, 0.25, beta, heads, M, N, K, token_major=True)
    glm.synchronize()
    expected = (0.25 * torch.bmm(A.float().transpose(0, 1), B.float()).transpose(0, 1)
                + beta * initial.float()).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-3)
