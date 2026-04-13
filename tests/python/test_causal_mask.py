import torch
import pytest
from helpers import ATOL, RTOL


@pytest.mark.parametrize("seq_len", [4, 8, 16, 64])
def test_causal_mask_shape(glm, device, seq_len):
    out = torch.empty(seq_len, seq_len, dtype=torch.bfloat16, device=device)

    glm.causal_mask(out, seq_len)

    ref = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1).bfloat16()
    torch.testing.assert_close(out.cpu(), ref, atol=0.0, rtol=0.0)


def test_causal_mask_lower_triangle_zero(glm, device):
    seq_len = 16
    out = torch.empty(seq_len, seq_len, dtype=torch.bfloat16, device=device)

    glm.causal_mask(out, seq_len)

    lower = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    lower_vals = out.cpu().float()[lower]
    assert (lower_vals == 0.0).all(), "Lower triangle should be 0"


def test_causal_mask_upper_triangle_neginf(glm, device):
    seq_len = 16
    out = torch.empty(seq_len, seq_len, dtype=torch.bfloat16, device=device)

    glm.causal_mask(out, seq_len)

    upper = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    assert (out.cpu().float()[upper] == float('-inf')).all(), "Upper triangle should be -inf"


def test_causal_mask_diagonal_is_zero(glm, device):
    seq_len = 8
    out = torch.empty(seq_len, seq_len, dtype=torch.bfloat16, device=device)

    glm.causal_mask(out, seq_len)

    diag = out.cpu().float().diagonal()
    assert (diag == 0.0).all(), "Diagonal should be 0"
