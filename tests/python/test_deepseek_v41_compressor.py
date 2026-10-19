"""Compression keeps FP32 pending pairs and is invariant to prefill chunk boundaries."""
import ctypes

import pytest
import torch


@pytest.mark.parametrize("ratio", [1, 2])
def test_compressor_chunk_continuation_and_missing_state(glm, device, ratio):
    torch.manual_seed(123)
    lengths = [7, 0, 4]
    raw = torch.randn(sum(lengths), 512 * ratio, device=device)
    if ratio == 2:
        raw[0, 512:] *= 1000
    weight = torch.randn(512, device=device).bfloat16()
    state = torch.zeros(len(lengths), 1024, device=device)
    state_position = torch.full((len(lengths),), -1, dtype=torch.int32, device=device)
    fn = glm.lib.glm_dsv41_compress
    fn.restype = None
    fn.argtypes = [ctypes.c_void_p] * 11 + [ctypes.c_int] * 2 + [ctypes.c_float]

    def run(indices, positions, cu):
        nonlocal state, state_position
        inputs = raw[indices].contiguous()
        pos = torch.tensor(positions, dtype=torch.int32, device=device)
        boundaries = torch.tensor(cu, dtype=torch.int32, device=device)
        latent = torch.empty(len(indices), 512, dtype=torch.bfloat16, device=device)
        valid = torch.empty(len(indices), dtype=torch.int32, device=device)
        next_state, next_position = torch.empty_like(state), torch.empty_like(state_position)
        old_state, old_position = state.clone(), state_position.clone()
        torch.cuda.synchronize(device)
        fn(glm.ctx, latent.data_ptr(), valid.data_ptr(), state.data_ptr(), state_position.data_ptr(),
           next_state.data_ptr(), next_position.data_ptr(),
           inputs.data_ptr(), pos.data_ptr(), boundaries.data_ptr(), weight.data_ptr(), len(lengths), ratio, 1e-20)
        torch.cuda.synchronize(device)
        torch.testing.assert_close(state, old_state, atol=0, rtol=0)
        torch.testing.assert_close(state_position, old_position, atol=0, rtol=0)
        if ratio == 2:
            state, state_position = next_state, next_position
        return latent, valid

    cu = [0, 7, 7, 11]
    whole, valid = run(list(range(11)), list(range(7)) + list(range(4)), cu)
    reference = torch.zeros_like(whole)
    valid_reference = torch.zeros_like(valid)
    for start, end in zip(cu[:-1], cu[1:]):
        for t in range(start, end):
            if ratio == 2 and (t - start) % 2 == 0:
                continue
            if ratio == 2:
                pair = raw[t-1:t+1]
                pooled = (pair[:, :512] * pair[:, 512:].softmax(0)).sum(0)
            else:
                pooled = raw[t]
            reference[t] = pooled * torch.rsqrt(pooled.square().mean() + 1e-20) * weight.float()
            valid_reference[t] = 1
    torch.testing.assert_close(whole, reference, atol=.008, rtol=.008)
    torch.testing.assert_close(valid, valid_reference, atol=0, rtol=0)
    state.zero_()
    state_position.fill_(-1)
    # Both requests split on an odd boundary; the first token of the second
    # chunk must consume the preceding chunk's saved FP32 row.
    first, first_valid = run([0, 1, 2, 7], [0, 1, 2, 0], [0, 3, 3, 4])
    retained_state, retained_position = state, state_position
    expected_state, expected_position = state.clone(), state_position.clone()
    second, second_valid = run([3, 4, 5, 6, 8, 9, 10], [3, 4, 5, 6, 1, 2, 3], [0, 4, 4, 7])
    torch.testing.assert_close(retained_state, expected_state, atol=0, rtol=0)
    torch.testing.assert_close(retained_position, expected_position, atol=0, rtol=0)
    torch.testing.assert_close(first, whole[[0, 1, 2, 7]], atol=0, rtol=0)
    torch.testing.assert_close(second, whole[[3, 4, 5, 6, 8, 9, 10]], atol=0, rtol=0)
    torch.testing.assert_close(first_valid, valid[[0, 1, 2, 7]], atol=0, rtol=0)
    torch.testing.assert_close(second_valid, valid[[3, 4, 5, 6, 8, 9, 10]], atol=0, rtol=0)
    if ratio == 2:
        state, state_position = retained_state, retained_position
        replay, replay_valid = run([3, 8], [3, 1], [0, 1, 1, 2])
        torch.testing.assert_close(replay, whole[[3, 8]], atol=0, rtol=0)
        assert replay_valid.tolist() == [1, 1]
        state_position.fill_(-1)
        _, missing = run([3], [3], [0, 1, 1, 1])
        assert missing.item() == -1
