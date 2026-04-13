import torch
import pytest


def torch_embedding(table, ids):
    return torch.nn.functional.embedding(ids, table)


@pytest.mark.parametrize("hidden,seq_len", [(64, 8), (512, 16), (6144, 4)])
def test_embedding_random(glm, device, hidden, seq_len):
    vocab = 1000
    table = torch.randn(vocab, hidden, dtype=torch.bfloat16, device=device)
    ids = torch.randint(0, vocab, (seq_len,), dtype=torch.int32, device=device)
    out = torch.empty(seq_len, hidden, dtype=torch.bfloat16, device=device)

    glm.embedding(out, table, ids, hidden, seq_len)
    ref = torch_embedding(table, ids)
    assert torch.equal(out.cpu(), ref.cpu())


def test_embedding_single_token(glm, device):
    vocab = 100
    hidden = 128
    table = torch.randn(vocab, hidden, dtype=torch.bfloat16, device=device)
    ids = torch.tensor([42], dtype=torch.int32, device=device)
    out = torch.empty(1, hidden, dtype=torch.bfloat16, device=device)

    glm.embedding(out, table, ids, hidden, 1)
    ref = torch_embedding(table, ids)
    assert torch.equal(out.cpu(), ref.cpu())


def test_embedding_same_id_repeated(glm, device):
    vocab = 50
    hidden = 64
    table = torch.randn(vocab, hidden, dtype=torch.bfloat16, device=device)
    ids = torch.tensor([7, 7, 7, 7], dtype=torch.int32, device=device)
    out = torch.empty(4, hidden, dtype=torch.bfloat16, device=device)

    glm.embedding(out, table, ids, hidden, 4)
    ref = torch_embedding(table, ids)
    assert torch.equal(out.cpu(), ref.cpu())
    assert torch.equal(out[0].cpu(), out[1].cpu())
    assert torch.equal(out[0].cpu(), out[2].cpu())
    assert torch.equal(out[0].cpu(), out[3].cpu())
