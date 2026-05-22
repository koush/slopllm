import torch
import pytest
import ctypes


def test_rotate_input_ids_single_sequence(glm, device):
    input_ids = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32, device=device)
    qo_indptr = torch.tensor([0, 5], dtype=torch.int32, device=device)
    new_tokens = torch.tensor([99], dtype=torch.int32, device=device)
    output_ids = torch.empty_like(input_ids)

    glm.rotate_input_ids(output_ids, input_ids, qo_indptr, new_tokens, batch_size=1)

    expected = torch.tensor([20, 30, 40, 50, 99], dtype=torch.int32, device=device)
    assert torch.equal(output_ids, expected)


def test_rotate_input_ids_multi_sequence(glm, device):
    input_ids = torch.tensor([1, 2, 3, 10, 20, 30, 40], dtype=torch.int32, device=device)
    qo_indptr = torch.tensor([0, 3, 7], dtype=torch.int32, device=device)
    new_tokens = torch.tensor([100, 200], dtype=torch.int32, device=device)
    output_ids = torch.empty_like(input_ids)

    glm.rotate_input_ids(output_ids, input_ids, qo_indptr, new_tokens, batch_size=2)

    expected = torch.tensor([2, 3, 100, 20, 30, 40, 200], dtype=torch.int32, device=device)
    assert torch.equal(output_ids, expected)


def test_rotate_input_ids_length_one(glm, device):
    input_ids = torch.tensor([42], dtype=torch.int32, device=device)
    qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    new_tokens = torch.tensor([99], dtype=torch.int32, device=device)
    output_ids = torch.empty_like(input_ids)

    glm.rotate_input_ids(output_ids, input_ids, qo_indptr, new_tokens, batch_size=1)

    expected = torch.tensor([99], dtype=torch.int32, device=device)
    assert torch.equal(output_ids, expected)


def test_rotate_input_ids_length_two(glm, device):
    input_ids = torch.tensor([10, 20], dtype=torch.int32, device=device)
    qo_indptr = torch.tensor([0, 2], dtype=torch.int32, device=device)
    new_tokens = torch.tensor([99], dtype=torch.int32, device=device)
    output_ids = torch.empty_like(input_ids)

    glm.rotate_input_ids(output_ids, input_ids, qo_indptr, new_tokens, batch_size=1)

    expected = torch.tensor([20, 99], dtype=torch.int32, device=device)
    assert torch.equal(output_ids, expected)


def test_rotate_input_ids_mixed_lengths(glm, device):
    input_ids = torch.tensor([5], dtype=torch.int32, device=device)
    qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    new_tokens = torch.tensor([88], dtype=torch.int32, device=device)
    output_ids = torch.empty_like(input_ids)

    glm.rotate_input_ids(output_ids, input_ids, qo_indptr, new_tokens, batch_size=1)

    expected = torch.tensor([88], dtype=torch.int32, device=device)
    assert torch.equal(output_ids, expected)


def test_rotate_input_ids_three_sequences(glm, device):
    input_ids = torch.tensor([1, 2, 3, 4, 10, 20, 100], dtype=torch.int32, device=device)
    qo_indptr = torch.tensor([0, 4, 6, 7], dtype=torch.int32, device=device)
    new_tokens = torch.tensor([50, 60, 70], dtype=torch.int32, device=device)
    output_ids = torch.empty_like(input_ids)

    glm.rotate_input_ids(output_ids, input_ids, qo_indptr, new_tokens, batch_size=3)

    expected = torch.tensor([2, 3, 4, 50, 20, 60, 70], dtype=torch.int32, device=device)
    assert torch.equal(output_ids, expected)


def test_rotate_input_ids_chain(glm, device):
    input_ids = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32, device=device)
    qo_indptr = torch.tensor([0, 5], dtype=torch.int32, device=device)
    output_ids = torch.empty_like(input_ids)

    glm.rotate_input_ids(output_ids, input_ids, qo_indptr,
                         torch.tensor([99], dtype=torch.int32, device=device), 1)
    assert torch.equal(output_ids, torch.tensor([20, 30, 40, 50, 99], dtype=torch.int32, device=device))

    glm.rotate_input_ids(output_ids, output_ids.clone(), qo_indptr,
                         torch.tensor([88], dtype=torch.int32, device=device), 1)
    assert torch.equal(output_ids, torch.tensor([30, 40, 50, 99, 88], dtype=torch.int32, device=device))

    glm.rotate_input_ids(output_ids, output_ids.clone(), qo_indptr,
                         torch.tensor([77], dtype=torch.int32, device=device), 1)
    assert torch.equal(output_ids, torch.tensor([40, 50, 99, 88, 77], dtype=torch.int32, device=device))


def test_rotate_input_ids_large_sequence(glm, device):
    seq_len = 1024
    input_ids = torch.arange(seq_len, dtype=torch.int32, device=device)
    qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    new_tokens = torch.tensor([9999], dtype=torch.int32, device=device)
    output_ids = torch.empty_like(input_ids)

    glm.rotate_input_ids(output_ids, input_ids, qo_indptr, new_tokens, batch_size=1)

    expected = torch.arange(1, seq_len, dtype=torch.int32, device=device)
    expected = torch.cat([expected, torch.tensor([9999], dtype=torch.int32, device=device)])
    assert torch.equal(output_ids, expected)
