"""Regression coverage for RNG used by the native batch sampling kernel."""

import ctypes

import pytest
import torch


@pytest.mark.parametrize("seed", [0, 1_000_000_000, 2**32 - 2048])
@pytest.mark.parametrize("top_k", [8, 20])
def test_sample_batch_rng_uniform_and_adjacent_repeats(glm, device, seed, top_k):
    sample_batch = glm.lib.glm_sample_batch
    sample_batch.restype = None
    sample_batch.argtypes = [
        ctypes.c_void_p,  # ctx
        ctypes.c_void_p,  # out_tokens
        ctypes.c_void_p,  # topk_vals
        ctypes.c_void_p,  # topk_idxs
        ctypes.c_void_p,  # workspace
        ctypes.c_void_p,  # logits
        ctypes.c_void_p,  # penalty_tokens
        ctypes.c_void_p,  # penalty_count
        ctypes.c_int,     # max_window
        ctypes.c_int,     # vocab_size
        ctypes.c_int,     # batch_size
        ctypes.c_void_p,  # temperatures
        ctypes.c_void_p,  # repetition_penalties
        ctypes.c_void_p,  # presence_penalties
        ctypes.c_void_p,  # top_ks
        ctypes.c_void_p,  # top_ps
        ctypes.c_void_p,  # step_counter (uint32)
        ctypes.c_int,     # max_effective_k
    ]

    draws, batch = 4096, 1
    logits = torch.zeros((batch, top_k), dtype=torch.bfloat16, device=device)
    outputs = torch.empty((draws, batch), dtype=torch.int32, device=device)
    # Match the MAX_K=8/32 dispatch and SAMPLING_BLOCK_SIZE=256 scratch layout.
    max_k = 8 if top_k <= 8 else 32
    topk_vals = torch.empty((batch, max_k * 256), dtype=torch.float32, device=device)
    topk_idxs = torch.empty_like(topk_vals, dtype=torch.int32)
    workspace = torch.empty((batch, top_k), dtype=torch.float32, device=device)
    penalty_tokens = torch.zeros(batch, dtype=torch.int32, device=device)
    penalty_count = torch.zeros_like(penalty_tokens)
    temperatures = torch.ones(batch, dtype=torch.float32, device=device)
    repetition_penalties = torch.ones_like(temperatures)
    presence_penalties = torch.zeros_like(temperatures)
    top_ks = torch.full((batch,), top_k, dtype=torch.int32, device=device)
    top_ps = torch.ones_like(temperatures)
    step_counter = torch.tensor([seed], dtype=torch.uint32, device=device)

    # Finish PyTorch initialization before the native context consumes buffers.
    torch.cuda.synchronize(device)
    out_ptr = outputs.data_ptr()
    row_bytes = batch * outputs.element_size()
    for draw in range(draws):
        sample_batch(
            glm.ctx, out_ptr + draw * row_bytes,
            topk_vals.data_ptr(), topk_idxs.data_ptr(), workspace.data_ptr(),
            logits.data_ptr(), penalty_tokens.data_ptr(), penalty_count.data_ptr(),
            0, top_k, batch,
            temperatures.data_ptr(), repetition_penalties.data_ptr(),
            presence_penalties.data_ptr(), top_ks.data_ptr(), top_ps.data_ptr(),
            step_counter.data_ptr(), top_k,
        )
    glm.synchronize()
    tokens = outputs.cpu().flatten().long()
    assert step_counter.cpu().item() == (seed + draws) % 2**32
    assert ((tokens >= 0) & (tokens < top_k)).all(), f"seed={seed}: invalid token ID"

    frequencies = torch.bincount(tokens, minlength=top_k).float() / draws
    repeat_rate = (tokens[1:] == tokens[:-1]).float().mean().item()
    # Both expectations are 1/top_k. The tolerance is over seven standard errors;
    # xorshift(counter) instead produces long runs of nearly identical draws.
    expected = 1.0 / top_k
    assert (frequencies - expected).abs().max().item() < 0.04, (
        f"seed={seed}: nonuniform token frequencies {frequencies.tolist()}"
    )
    assert abs(repeat_rate - expected) < 0.04, (
        f"seed={seed}: adjacent repeat rate {repeat_rate:.4f}, expected {expected}"
    )
