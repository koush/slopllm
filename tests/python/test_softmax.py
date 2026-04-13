import torch
import pytest


@pytest.mark.parametrize("batch,dim", [(1, 16), (4, 32), (2, 128)])
def test_softmax_no_mask(glm, device, batch, dim):
    x = torch.randn(batch, dim, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(x)

    glm.softmax(out, x, None, dim, batch)
    ref = torch.softmax(x.float(), dim=-1).to(torch.bfloat16)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=5e-3, rtol=5e-3)


def test_softmax_with_causal_mask(glm, device):
    seq_len = 8
    x = torch.randn(seq_len, seq_len, dtype=torch.bfloat16, device=device)
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), dtype=torch.bfloat16, device=device), diagonal=1)
    out = torch.empty_like(x)

    glm.softmax(out, x, mask, seq_len, seq_len)
    ref = torch.softmax((x + mask).float(), dim=-1).to(torch.bfloat16)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=5e-3, rtol=5e-3)


def test_softmax_rows_sum_to_one(glm, device):
    x = torch.randn(4, 32, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(x)

    glm.softmax(out, x, None, 32, 4)
    row_sums = out.cpu().float().sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones(4), atol=1e-3, rtol=1e-3)


def test_softmax_large_values(glm, device):
    x = torch.randn(2, 16, dtype=torch.bfloat16, device=device) * 10.0
    out = torch.empty_like(x)

    glm.softmax(out, x, None, 16, 2)
    ref = torch.softmax(x.float(), dim=-1).to(torch.bfloat16)
    torch.testing.assert_close(out.cpu(), ref.cpu(), atol=5e-3, rtol=5e-3)


def test_softmax_masked_positions_near_zero(glm, device):
    seq_len = 4
    x = torch.zeros(seq_len, seq_len, dtype=torch.bfloat16, device=device)
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), dtype=torch.bfloat16, device=device), diagonal=1)
    out = torch.empty_like(x)

    glm.softmax(out, x, mask, seq_len, seq_len)
    for i in range(seq_len):
        row_sum = out[i, :i + 1].float().sum().item()
        assert abs(row_sum - 1.0) < 5e-3, f"Row {i} sum = {row_sum}"
        if i < seq_len - 1:
            assert (out[i, i + 1:] == 0).all(), f"Row {i} masked positions should be 0"
