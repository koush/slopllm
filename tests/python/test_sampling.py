"""Native GPU sampling distributions, compact supports, and counter regression tests."""

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
        ctypes.c_void_p,  # optional compact probabilities
        ctypes.c_void_p,  # optional compact token IDs
        ctypes.c_int,     # support_capacity
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
            step_counter.data_ptr(), top_k, None, None, 0,
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


def lowbias32_uniform(counter):
    """Match uint32 overflow and the kernel's exactly representable 24-bit draw."""
    value = counter & 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    value ^= value >> 16
    return (value >> 8) / 2**24


def _sample_buffers(glm, device, logits, top_ks, temperatures, top_ps, seed,
                    capacity=256, export="both"):
    sample = glm.lib.glm_sample_batch
    sample.restype = None
    sample.argtypes = ([ctypes.c_void_p] * 8 + [ctypes.c_int] * 3
                       + [ctypes.c_void_p] * 6 + [ctypes.c_int]
                       + [ctypes.c_void_p] * 2 + [ctypes.c_int])
    batch, vocab = logits.shape
    effective = [1 if t <= 0 else min(k, vocab) if k > 0 else 32
                 for k, t in zip(top_ks, temperatures)]
    max_k = max(effective)
    scratch_k = next((k for k in (2, 8, 16, 32, 64) if k >= max_k), max_k)
    scratch_size = scratch_k * 256 if max_k <= 64 else max_k
    logits = logits.to(device=device, dtype=torch.bfloat16)
    vals = torch.empty((batch, scratch_size), device=device)
    idxs = torch.empty_like(vals, dtype=torch.int32)
    workspace = torch.empty((batch, vocab), device=device)
    out = torch.full((batch,), -99, device=device, dtype=torch.int32)
    probs = torch.full((batch, capacity), -99.0, device=device)
    ids = torch.full((batch, capacity), -99, device=device, dtype=torch.int32)
    counter = torch.tensor([seed], device=device, dtype=torch.uint32)
    penalties = torch.zeros(batch, device=device, dtype=torch.int32)
    counts = torch.zeros_like(penalties)
    temps = torch.tensor(temperatures, device=device, dtype=torch.float32)
    ks = torch.tensor(top_ks, device=device, dtype=torch.int32)
    ps = torch.tensor(top_ps, device=device, dtype=torch.float32)
    repetition = torch.ones_like(temps)
    presence = torch.zeros_like(temps)

    def launch():
        sample(glm.ctx, out.data_ptr(), vals.data_ptr(), idxs.data_ptr(),
               workspace.data_ptr(), logits.data_ptr(), penalties.data_ptr(),
               counts.data_ptr(), 0, vocab, batch, temps.data_ptr(),
               repetition.data_ptr(), presence.data_ptr(), ks.data_ptr(), ps.data_ptr(),
               counter.data_ptr(), max_k,
               probs.data_ptr() if export in ("both", "probs") else None,
               ids.data_ptr() if export in ("both", "ids") else None,
               capacity if export != "neither" else 0)

    torch.cuda.synchronize(device)
    return launch, out, probs, ids, counter


@pytest.mark.parametrize("vocab", [1024, 154880])
@pytest.mark.parametrize("top_ks", [
    [1], [8, 1], [16, 8, 1], [20, 8, 0, -1, 1],
    [64, 20, 0, 1], [65, 20, 0, 1], [256, 65, 20, 0, 1],
])
def test_sample_batch_compact_distribution(glm, device, vocab, top_ks):
    generator = torch.Generator().manual_seed(712)
    batch = len(top_ks)
    logits = -8 + torch.randn((batch, vocab), generator=generator) * 0.25
    # Distinct BF16-representable leaders avoid ambiguous top-k boundary ties.
    # Scatter them across a realistic vocabulary, not just the first k tokens.
    for row in range(batch):
        leaders = torch.randperm(vocab, generator=generator)[:256]
        logits[row, leaders] = torch.cat((
            4 - torch.arange(32).float() / 8,
            -torch.arange(224).float() / 128,
        ))
    logits = logits.bfloat16()
    temperatures = [0.75 if row % 2 == 0 else 1.5 for row in range(batch)]
    temperatures[-1] = 0.0
    top_ps = [0.55] * batch
    seed = 2**32 - 3
    launch, out, probs, ids, counter = _sample_buffers(
        glm, device, logits, top_ks, temperatures, top_ps, seed)
    for iteration in range(3):
        launch()
        glm.synchronize()
        actual_probs, actual_ids = probs.cpu(), ids.cpu().long()
        tokens = out.cpu()
        for row, (k, temperature) in enumerate(zip(top_ks, temperatures)):
            effective = 1 if temperature <= 0 else k if k > 0 else 32
            values, expected_ids = torch.topk(
                logits[row].float() / (temperature if temperature > 0 else 1), effective)
            expected_probs = values.softmax(0)
            if k <= 0 and top_ps[row] < 1:
                cutoff = int(torch.nonzero(expected_probs.cumsum(0) > top_ps[row])[0]) + 1
                expected_probs[cutoff:] = 0
                expected_probs /= expected_probs.sum()
            assert set(actual_ids[row, :effective].tolist()) == set(expected_ids.tolist())
            # Sorting by ID compares distributions independently of kernel ordering.
            order = actual_ids[row, :effective].argsort()
            torch.testing.assert_close(
                actual_probs[row, :effective][order],
                expected_probs[expected_ids.argsort()], rtol=2e-5, atol=2e-7)
            assert (actual_ids[row, effective:] == -1).all()
            assert (actual_probs[row, effective:] == 0).all()
            assert actual_probs[row].sum().item() == pytest.approx(1, abs=2e-6)
            uniform = lowbias32_uniform(seed + iteration * batch + row)
            cumulative = 0.0
            weights = actual_probs[row, :effective].tolist()
            threshold = uniform * sum(weights)
            expected_token = int(actual_ids[row, effective - 1])
            for token, probability in zip(actual_ids[row, :effective], weights):
                cumulative += probability
                if probability > 0 and threshold < cumulative:
                    expected_token = int(token)
                    break
            assert tokens[row].item() == expected_token
        assert counter.cpu().item() == (seed + (iteration + 1) * batch) % 2**32


@pytest.mark.parametrize("export", ["both", "probs", "ids", "neither"])
def test_sample_batch_counter_graph_replay(glm, device, export):
    batch, k, seed = 37, 20, 2**32 - 19
    logits = torch.full((batch, 1024), -10.0)
    logits[:, :k] = -torch.arange(k).float() / 8
    launch, out, probs, ids, counter = _sample_buffers(
        glm, device, logits, [k] * batch, [1.0] * batch, [0.1] * batch,
        seed, capacity=32, export=export)
    expected_probs = logits[0, :k].softmax(0)
    # Warm up on the native stream, then capture that stream, not Torch's stream.
    launch()
    glm.synchronize()
    glm.graph_begin_capture()
    launch()
    graph = glm.graph_end_capture()
    executable = None
    try:
        executable = glm.graph_instantiate(graph)
        assert counter.cpu().item() == (seed + batch) % 2**32
        for iteration in range(1, 5):
            glm.graph_launch(executable)
            glm.synchronize()
            expected = [int(torch.searchsorted(
                expected_probs.cumsum(0), lowbias32_uniform(seed + iteration * batch + row),
                right=True)) for row in range(batch)]
            assert out.cpu().tolist() == expected
            assert counter.cpu().item() == (seed + (iteration + 1) * batch) % 2**32
        if export in ("both", "probs"):
            torch.testing.assert_close(probs.cpu()[:, :k], expected_probs.expand(batch, k))
            assert (probs.cpu()[:, k:] == 0).all()
        else:
            assert (probs.cpu() == -99).all()
        if export in ("both", "ids"):
            assert ids.cpu()[:, :k].tolist() == [list(range(k))] * batch
            assert (ids.cpu()[:, k:] == -1).all()
        else:
            assert (ids.cpu() == -99).all()
    finally:
        if executable is not None:
            glm.graph_exec_destroy(executable)
        glm.graph_destroy(graph)


@pytest.mark.parametrize("top_k", [8, 65])
@pytest.mark.parametrize("case", ["exhausted", "empty", "nan", "positive_inf"])
def test_sample_batch_exhausted_support(glm, device, top_k, case):
    logits = torch.full((2, 128), float("nan") if case == "nan" else -40.0)
    if case == "exhausted":
        logits[:, 73] = 0
    elif case == "positive_inf":
        logits[:, [73, 91]] = 40
    launch, out, probs, ids, _ = _sample_buffers(
        glm, device, logits, [top_k] * 2, [1e-37] * 2, [1.0] * 2, 71,
        capacity=80)
    launch()
    glm.synchronize()
    probs, ids, out = probs.cpu(), ids.cpu(), out.cpu()
    support = [73, 91] if case == "positive_inf" else [73] if case == "exhausted" else [0]
    size = len(support)
    assert torch.isfinite(probs).all()
    assert (probs[:, :size] == 1 / size).all()
    assert (probs[:, size:] == 0).all()
    assert (ids[:, size:] == -1).all()
    for row in range(2):
        assert set(ids[row, :size].tolist()) == set(support)
        assert out[row].item() in support


def _candidate_buffers(glm, device, values, candidate_ids, top_ks, temperatures,
                       top_ps, seed, capacity=256):
    sample = glm.lib.glm_sample_candidates
    sample.restype = None
    sample.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 3
    batch, count = values.shape
    values = values.to(device=device, dtype=torch.bfloat16).contiguous()
    candidate_ids = candidate_ids.to(device=device, dtype=torch.int32).contiguous()
    temps = torch.tensor(temperatures, device=device, dtype=torch.float32)
    ks = torch.tensor(top_ks, device=device, dtype=torch.int32)
    ps = torch.tensor(top_ps, device=device, dtype=torch.float32)
    out = torch.full((batch,), -99, device=device, dtype=torch.int32)
    probs = torch.full((batch, capacity), -99.0, device=device)
    ids = torch.full((batch, capacity), -99, device=device, dtype=torch.int32)
    counter = torch.tensor([seed], device=device, dtype=torch.uint32)

    def launch():
        sample(glm.ctx, out.data_ptr(), probs.data_ptr(), ids.data_ptr(),
               values.data_ptr(), candidate_ids.data_ptr(), temps.data_ptr(),
               ks.data_ptr(), ps.data_ptr(), counter.data_ptr(), batch, count, capacity)

    torch.cuda.synchronize(device)
    return launch, out, probs, ids, counter


@pytest.mark.parametrize("candidate_count", [64, 256])
@pytest.mark.parametrize("shards", [1, 4])
@pytest.mark.parametrize("sorted_candidates", [False, True])
@pytest.mark.parametrize("seed", [0, 1_000_000_000, 2**32 - 3])
def test_sample_candidates_full_logits_reference(
        glm, device, candidate_count, shards, sorted_candidates, seed):
    # Small full vocabulary keeps even the original K>64 fallback inexpensive.
    batch, vocab, capacity = 8, 2048, 256
    generator = torch.Generator().manual_seed(819)
    background = -torch.cat([
        (1 + torch.arange(128).float() / 128) * 2**exponent
        for exponent in range(16)
    ])
    logits = torch.empty((batch, vocab), dtype=torch.bfloat16)
    for row in range(batch):
        logits[row] = background[torch.randperm(vocab, generator=generator)]
        leaders = 512 + torch.randperm(vocab - 512, generator=generator)[:256]
        logits[row, leaders] = (2 - torch.arange(256).float() / 128).bfloat16()
        assert logits[row].unique().numel() == vocab

    full_values, full_ids = torch.topk(logits, candidate_count, sorted=False)
    if shards == 1:
        values, candidate_ids = full_values, full_ids
    else:
        local_values, local_ids = [], []
        for rank, shard in enumerate(logits.chunk(shards, dim=1)):
            vals, ids = torch.topk(shard, candidate_count, sorted=False)
            local_values.append(vals)
            local_ids.append(ids + rank * (vocab // shards))
        merged_values = torch.cat(local_values, dim=1)
        merged_ids = torch.cat(local_ids, dim=1)
        values, slots = torch.topk(merged_values, candidate_count, sorted=False)
        candidate_ids = merged_ids.gather(1, slots)
        torch.testing.assert_close(candidate_ids.sort(1).values, full_ids.sort(1).values)
        torch.testing.assert_close(values, logits.gather(1, candidate_ids))
    order = (values.argsort(dim=1, descending=True) if sorted_candidates else
             torch.stack([torch.randperm(candidate_count, generator=generator)
                          for _ in range(batch)]))
    values, candidate_ids = values.gather(1, order), candidate_ids.gather(1, order)
    assert (candidate_ids >= 512).all()  # No global ID can accidentally be a slot.
    if not sorted_candidates:
        assert not (values[:, :-1] > values[:, 1:]).all()

    ks = [candidate_count, 1, 20, 0, 20, 20, -1,
          65 if candidate_count > 64 else 32]
    temps = [1.0, 0.75, 1.5, 1.0, 0.0, -0.5, 0.75, 1.0]
    ps = [0.6] * batch
    compact = _candidate_buffers(glm, device, values, candidate_ids, ks, temps, ps,
                                 seed, capacity)
    original = _sample_buffers(glm, device, logits, ks, temps, ps, seed, capacity)
    expected_probs = torch.zeros((batch, capacity))
    expected_ids = torch.full((batch, capacity), -1, dtype=torch.int32)
    for row, (k, temperature) in enumerate(zip(ks, temps)):
        effective = 1 if temperature <= 0 else k if k > 0 else 32
        vals, ids = torch.topk(logits[row].float(), effective, sorted=True)
        weights = (vals / (temperature if temperature > 0 else 1)).softmax(0)
        if k <= 0:
            cutoff = int(torch.nonzero(weights.cumsum(0) > ps[row])[0]) + 1
            weights[cutoff:] = 0
            weights /= weights.sum()
        expected_probs[row, :effective] = weights
        expected_ids[row, :effective] = ids.int()

    # Warm up both native paths, then replay against the same independent counters.
    compact[0]()
    original[0]()
    glm.synchronize()
    glm.graph_begin_capture()
    compact[0]()
    original[0]()
    graph = glm.graph_end_capture()
    executable = None
    try:
        executable = glm.graph_instantiate(graph)
        for iteration in range(4):
            if iteration:
                glm.graph_launch(executable)
                glm.synchronize()
            for result in (compact, original):
                _, out, probs, ids, counter = result
                actual_probs = probs.cpu()
                torch.testing.assert_close(ids.cpu(), expected_ids)
                torch.testing.assert_close(actual_probs, expected_probs,
                                           rtol=2e-5, atol=2e-7)
                assert counter.cpu().item() == (seed + (iteration + 1) * batch) % 2**32
                expected_tokens = []
                for row in range(batch):
                    # Native sparse draw accumulates exported FP32 q weights in FP64.
                    cumulative = actual_probs[row].double().cumsum(0)
                    threshold = lowbias32_uniform(seed + iteration * batch + row) * cumulative[-1]
                    slot = int(torch.searchsorted(cumulative, threshold, right=True))
                    expected_tokens.append(int(expected_ids[row, slot]))
                assert out.cpu().tolist() == expected_tokens
            torch.testing.assert_close(compact[1].cpu(), original[1].cpu())
    finally:
        if executable is not None:
            glm.graph_exec_destroy(executable)
        glm.graph_destroy(graph)


@pytest.mark.parametrize("candidate_count", [7, 64, 256])
@pytest.mark.parametrize("temperature", [1.0, 0.0, -1.0])
def test_sample_candidates_invalid_and_infinite_support(glm, device, candidate_count, temperature):
    # Invalid IDs must mask even +inf; NaN/-inf values cannot enter the support.
    values = torch.tensor([
        [float("inf"), 2, float("nan"), float("-inf"), 1, -1, 0],
        [float("inf"), float("nan"), float("-inf"), 3, 2, 1, 0],
        [float("-inf")] * 7,
        [float("nan")] * 7,
        [float("inf"), float("inf"), float("-inf"), 2, float("nan"), 0, -1],
    ], dtype=torch.bfloat16)
    candidate_ids = torch.tensor([
        [-1, 901, 700, 800, 154879, -7, -1],
        [-1, 701, 801, -1, -1, -1, -1],
        [901, 700, 800, 1000, 154879, 900, 999],
        [901, 700, 800, 1000, 154879, 900, 999],
        [154879, 901, 800, 1000, 700, -1, -1],
    ], dtype=torch.int32)
    batch = values.shape[0]
    padded_values = torch.full((batch, candidate_count), float("inf"), dtype=torch.bfloat16)
    padded_ids = torch.full((batch, candidate_count), -1, dtype=torch.int32)
    padded_values[:, :7], padded_ids[:, :7] = values, candidate_ids
    launch, out, probs, ids, counter = _candidate_buffers(
        glm, device, padded_values, padded_ids, [20] * batch,
        [temperature] * batch, [0.6] * batch, 71)
    launch()
    glm.synchronize()
    expected_ids = torch.full((batch, 256), -1, dtype=torch.int32)
    expected_probs = torch.zeros((batch, 256))
    expected_ids[1:4, 0] = 0
    expected_probs[1:4, 0] = 1
    if temperature <= 0:
        expected_ids[[0, 4], 0] = 901
        expected_probs[[0, 4], 0] = 1
    else:
        expected_ids[0, :2] = torch.tensor([901, 154879])
        expected_probs[0, :2] = torch.tensor([2.0, 1.0]).softmax(0)
        expected_ids[4, :3] = torch.tensor([901, 154879, 1000])
        expected_probs[4, :2] = 0.5
    torch.testing.assert_close(ids.cpu(), expected_ids)
    torch.testing.assert_close(probs.cpu(), expected_probs)
    assert torch.isfinite(probs.cpu()).all()
    for row, token in enumerate(out.cpu().tolist()):
        assert expected_probs[row, expected_ids[row] == token].sum() > 0
    assert counter.cpu().item() == 71 + batch


def test_sample_candidates_exported_q_rejection_global_lookup(glm, device):
    launch, drafts, q_probs, q_ids, _ = _candidate_buffers(
        glm, device, torch.tensor([[0.0, 2.0, 1.0]] * 2),
        torch.tensor([[154879, 901, 70001]] * 2), [20, 20], [1.0, 1.0],
        [0.6, 0.6], 71, capacity=3)
    launch()
    glm.synchronize()
    # Row 0: identical p/q with permuted slots must accept. Row 1: disjoint p
    # forces rejection and a correction in global vocabulary coordinates.
    p_probs = q_probs.flip(1)[:, None, :].repeat(1, 2, 1).contiguous()
    p_ids = q_ids.flip(1)[:, None, :].repeat(1, 2, 1).contiguous()
    p_probs[1] = torch.tensor([1.0, 0.0, 0.0], device=device)
    p_ids[1] = torch.tensor([123456, -1, -1], device=device, dtype=torch.int32)
    out = torch.full((2, 2), -99, device=device, dtype=torch.int32)
    accepted = torch.full((2,), -99, device=device, dtype=torch.int32)
    counter = torch.tensor([17], device=device, dtype=torch.uint32)
    reject = glm.lib.glm_spec_reject_linear
    reject.restype = None
    reject.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int] * 3
    torch.cuda.synchronize(device)
    reject(glm.ctx, out.data_ptr(), accepted.data_ptr(), drafts.data_ptr(),
           q_probs.data_ptr(), q_ids.data_ptr(), p_probs.data_ptr(), p_ids.data_ptr(),
           counter.data_ptr(), 2, 1, 3)
    glm.synchronize()
    assert accepted.cpu().tolist() == [1, 0]
    assert out.cpu()[0, 0].item() == drafts.cpu()[0].item()
    assert out.cpu()[0, 1].item() in [154879, 901, 70001]
    assert out.cpu()[1].tolist() == [123456, 123456]
    assert counter.cpu().item() == 25
