"""GPU linear speculative rejection against an exact counter-based CPU oracle."""

import ctypes
import random

import pytest
import torch

from test_sampling import _sample_buffers, lowbias32_uniform


def _draw(probs, ids, uniform):
    support = [(p, token) for p, token in zip(probs, ids) if token >= 0]
    threshold = uniform * sum(p for p, _ in support)
    cumulative = 0.0
    sampled = support[0][1]
    for probability, token in support:
        if probability <= 0:
            continue
        sampled = token
        cumulative += probability
        if threshold < cumulative:
            break
    return sampled


def _reference(drafts, q_probs, q_ids, p_probs, p_ids, seed):
    batch, depth = drafts.shape
    stride = 2 * depth + 2
    outputs, accepted = [], []
    arrays = [x.tolist() for x in (drafts, q_probs, q_ids, p_probs, p_ids)]
    for row, (tokens, qs, qis, ps, pis) in enumerate(zip(*arrays)):
        base = seed + row * stride
        out = [_draw(p, pi, lowbias32_uniform(base + d))
               for d, (p, pi) in enumerate(zip(ps, pis))]
        count = depth
        for d, token in enumerate(tokens):
            p = {token: prob for token, prob in zip(pis[d], ps[d]) if token >= 0}
            q = {token: prob for token, prob in zip(qis[d], qs[d]) if token >= 0}
            p_total, q_total = sum(p.values()), sum(q.values())
            px, qx = p.get(token, 0.0), q.get(token, 0.0)
            ratio = px * q_total / (qx * p_total) if qx > 0 else 0.0
            if qx > 0 and lowbias32_uniform(base + depth + 1 + d) < min(1.0, ratio):
                out[d] = token
                continue
            count = d
            residual = [max(0.0, prob / p_total - q.get(token_id, 0.0) / q_total)
                        for prob, token_id in zip(ps[d], pis[d])]
            if sum(residual) <= 0:
                residual = ps[d]
            out[d] = _draw(residual, pis[d], lowbias32_uniform(base + stride - 1))
            break
        outputs.append(out)
        accepted.append(count)
    return outputs, accepted


def _reject_buffers(glm, device, drafts, q_probs, q_ids, p_probs, p_ids, seed):
    reject = glm.lib.glm_spec_reject_linear
    reject.restype = None
    reject.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int] * 3
    batch, depth = drafts.shape
    capacity = q_probs.shape[-1]
    buffers = [x.to(device).contiguous() for x in (drafts, q_probs, q_ids, p_probs, p_ids)]
    out = torch.full((batch, depth + 1), -99, dtype=torch.int32, device=device)
    accepted = torch.full((batch,), -99, dtype=torch.int32, device=device)
    counter = torch.tensor([seed], dtype=torch.uint32, device=device)

    def launch():
        reject(glm.ctx, out.data_ptr(), accepted.data_ptr(),
               *(x.data_ptr() for x in buffers), counter.data_ptr(), batch, depth, capacity)

    torch.cuda.synchronize(device)
    return launch, out, accepted, counter


@pytest.mark.parametrize("seed", [0, 1_000_000_000, 2**32 - 13])
@pytest.mark.parametrize("capacity", [4, 256])
def test_spec_reject_linear_exact_and_graph(glm, device, seed, capacity):
    batch, depth = 12, 3
    drafts = torch.empty((batch, depth), dtype=torch.int32)
    q_probs = torch.zeros((batch, depth, capacity), dtype=torch.float32)
    q_ids = torch.full(q_probs.shape, -1, dtype=torch.int32)
    p_probs = torch.zeros((batch, depth + 1, capacity), dtype=torch.float32)
    p_ids = torch.full(p_probs.shape, -1, dtype=torch.int32)
    for row in range(batch):
        # Same distribution but different sparse order; IDs are not slot indices.
        p_probs[row, :, :3] = torch.tensor([0.5, 0.25, 0.25])
        p_ids[row, :, :3] = torch.tensor([901, 17, 154879])
        q_probs[row, :, :3] = torch.tensor([0.25, 0.5, 0.25])
        q_ids[row, :, :3] = torch.tensor([154879, 901, 17])
        drafts[row] = torch.tensor([154879, 17, 901])
        mode = row % 4
        if mode in (0, 1):
            # Force first/middle rejection with disjoint q and p supports.
            d = 0 if mode == 0 else 1
            q_ids[row, d, :3] = torch.tensor([2, 3, 4])
            drafts[row, d] = 3
        elif mode == 3:
            # Nontrivial acceptance ratios and positive-part residual, not just p.
            q_probs[row, :, :3] = torch.tensor([0.75, 0.125, 0.125])
    launch, out, accepted, counter = _reject_buffers(
        glm, device, drafts, q_probs, q_ids, p_probs, p_ids, seed)
    stride = 2 * depth + 2
    launch()
    glm.synchronize()
    expected, counts = _reference(drafts, q_probs, q_ids, p_probs, p_ids, seed)
    assert out.cpu().tolist() == expected
    assert accepted.cpu().tolist() == counts
    assert counts[0::4] == [0] * 3
    assert counts[1::4] == [1] * 3
    assert counts[2::4] == [depth] * 3
    assert counter.cpu().item() == (seed + batch * stride) % 2**32

    glm.graph_begin_capture()
    launch()
    graph = glm.graph_end_capture()
    executable = None
    try:
        executable = glm.graph_instantiate(graph)
        assert counter.cpu().item() == (seed + batch * stride) % 2**32
        for iteration in range(1, 5):
            glm.graph_launch(executable)
            glm.synchronize()
            expected, counts = _reference(
                drafts, q_probs, q_ids, p_probs, p_ids, seed + iteration * batch * stride)
            # Also check unused suffix target draws, fixing every reserved row offset.
            assert out.cpu().tolist() == expected
            assert accepted.cpu().tolist() == counts
            assert counter.cpu().item() == (seed + (iteration + 1) * batch * stride) % 2**32
    finally:
        if executable is not None:
            glm.graph_exec_destroy(executable)
        glm.graph_destroy(graph)


@pytest.mark.parametrize("seed", [71, 2**32 - 4096])
@pytest.mark.parametrize("scales", [(1.0, 1.0), (7.0, 0.125)])
def test_spec_reject_linear_target_marginal(glm, device, seed, scales):
    batch, depth, capacity = 8192, 1, 4
    # Independent CPU proposal RNG: reusing the kernel's acceptance counter would
    # correlate proposal and acceptance draws and invalidate this marginal test.
    rng = random.Random(20260908)
    support = [7, 901, 154879]
    target = torch.tensor([0.625, 0.25, 0.125])
    proposal = [0.125, 0.375, 0.5]
    drafts = torch.tensor(rng.choices(support, weights=proposal, k=batch),
                          dtype=torch.int32).reshape(batch, depth)
    p_probs = torch.tensor([0.625, 0.25, 0.125, 0.0]).repeat(batch, depth + 1, 1)
    p_ids = torch.tensor(support + [-1], dtype=torch.int32).repeat(batch, depth + 1, 1)
    q_probs = torch.tensor([0.5, 0.125, 0.375, 0.0]).repeat(batch, depth, 1)
    q_ids = torch.tensor([154879, 7, 901, -1], dtype=torch.int32).repeat(batch, depth, 1)
    p_probs *= scales[0]
    q_probs *= scales[1]
    launch, out, accepted, counter = _reject_buffers(
        glm, device, drafts, q_probs, q_ids, p_probs, p_ids, seed)
    launch()
    glm.synchronize()
    expected, counts = _reference(drafts, q_probs, q_ids, p_probs, p_ids, seed)
    assert out.cpu().tolist() == expected
    assert accepted.cpu().tolist() == counts
    assert counter.cpu().item() == (seed + batch * (2 * depth + 2)) % 2**32
    first = out.cpu()[:, 0]
    assert torch.isin(first, torch.tensor(support)).all()
    frequencies = torch.tensor([(first == token).float().mean() for token in support])
    # Seven binomial standard errors, plus rounding slack; rejection is frequent.
    tolerance = 7 * torch.sqrt(target * (1 - target) / batch) + 1 / batch
    assert ((frequencies - target).abs() < tolerance).all(), frequencies.tolist()
    assert 0.4 < accepted.cpu().float().mean().item() < 0.6


@pytest.mark.parametrize("top_k", [20, 65])
def test_gpu_proposals_equal_target(glm, device, top_k):
    batch, seed, capacity = 256, 713, 80
    logits = torch.full((batch, 128), -40.0)
    logits[:, :top_k] = -torch.arange(top_k).float() / 32
    sample, drafts, probs, ids, _ = _sample_buffers(
        glm, device, logits, [top_k] * batch, [1.0] * batch,
        [1.0] * batch, seed, capacity=capacity)
    sample()
    glm.synchronize()
    drafts, probs, ids = drafts.cpu(), probs.cpu(), ids.cpu()
    expected = [_draw(p, pi, lowbias32_uniform(seed + row))
                for row, (p, pi) in enumerate(zip(probs.tolist(), ids.tolist()))]
    assert drafts.tolist() == expected
    q_probs, q_ids = probs[:, None, :], ids[:, None, :]
    # Different totals, identical normalized distributions of exported F32 weights.
    p_probs = q_probs.repeat(1, 2, 1) * 8
    p_ids = q_ids.repeat(1, 2, 1)
    launch, out, accepted, _ = _reject_buffers(
        glm, device, drafts[:, None], q_probs, q_ids, p_probs, p_ids, seed + batch)
    launch()
    glm.synchronize()
    expected, counts = _reference(
        drafts[:, None], q_probs, q_ids, p_probs, p_ids, seed + batch)
    assert out.cpu().tolist() == expected
    assert accepted.cpu().tolist() == counts == [1] * batch
    assert torch.equal(out.cpu()[:, 0], drafts)
