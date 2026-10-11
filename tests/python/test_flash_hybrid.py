"""GLM Flash primitives against independent FP32 PyTorch math (CPU oracle)."""
import ctypes as ct

import numpy as np
import pytest
import torch

from helpers import GlmOps


@pytest.fixture(scope="module")
def glm():
    ops = GlmOps()
    signatures = {
        "glm_kda_scan": [ct.c_void_p] * 11 + [ct.c_int, ct.c_int, ct.c_float],
        "glm_kda_norm": [ct.c_void_p] * 5 + [ct.c_int, ct.c_float],
        "glm_mhc_pre": [ct.c_void_p] * 8 + [ct.c_int, ct.c_int, ct.c_float, ct.c_float, ct.c_int],
        "glm_mhc_post": [ct.c_void_p] * 6 + [ct.c_int, ct.c_int],
        "glm_mhc_streams": [ct.c_void_p] * 3 + [ct.c_int] * 3,
    }
    for name, args in signatures.items():
        fn = getattr(ops.lib, name)
        fn.argtypes, fn.restype = args, None
    yield ops
    ops.lib.glm_free(ops.ctx)
    ops.ctx = None


@pytest.fixture
def buffers(glm):
    allocated = []

    def upload(t):
        data = t.contiguous().view(torch.uint8).numpy().tobytes()
        ptr = glm.alloc(len(data))
        allocated.append(ptr)
        glm.h2d(ptr, data)
        return ptr

    yield upload
    glm.synchronize()
    for ptr in allocated:
        glm.free_buf(ptr)


def download(glm, ptr, shape, dtype):
    size = int(np.prod(shape)) * (2 if dtype == torch.bfloat16 else 4)
    data = ct.create_string_buffer(size)
    glm.synchronize()
    glm.d2h(data, ptr, size)
    return torch.frombuffer(bytearray(data.raw), dtype=dtype).reshape(shape).float()


def kda_reference(q, k, v, f, beta, a_log, dt, cu, state):
    q = q.float() / (q.float().square().sum(-1, keepdim=True) + 1e-6).sqrt() / 128**0.5
    k = k.float() / (k.float().square().sum(-1, keepdim=True) + 1e-6).sqrt()
    decay = (-5 * torch.sigmoid(a_log.exp()[None, :, None] * (f.float() + dt))).exp()
    beta = beta.sigmoid().float()
    state = state.clone()
    out = torch.empty_like(v)
    for b in range(len(cu) - 1):
        for t in range(cu[b], cu[b + 1]):
            s = state[b] * decay[t, :, :, None]
            delta = beta[t, :, None] * (v[t].float() - torch.einsum("hk,hkv->hv", k[t], s))
            state[b] = s + k[t, :, :, None] * delta[:, None, :]
            out[t] = torch.einsum("hk,hkv->hv", q[t], state[b])
    return out.float(), state


@pytest.mark.parametrize("lengths", [[1, 1], [3, 0, 65, 4]])
def test_kda_packed_scan_and_chunk_continuation(glm, buffers, lengths):
    torch.manual_seed(61)
    heads, tokens = 2, sum(lengths)
    q, k, v, f = [torch.randn(tokens, heads, 128).bfloat16() for _ in range(4)]
    q[0] = 0
    k[-1] *= 1e-5  # catches normalization epsilon mismatches
    f[0, 0] = torch.linspace(-30, 30, 128).bfloat16()  # channel-dependent saturated gates
    beta = torch.randn(tokens, heads).bfloat16()
    a_log, dt = torch.randn(heads), torch.randn(heads, 128)
    initial = torch.randn(len(lengths), heads, 128, 128) * .1
    cu = [0, *np.cumsum(lengths).tolist()]
    ref, ref_state = kda_reference(q, k, v, f, beta, a_log, dt, cu, initial)
    sp = buffers(initial)
    ap, dp = buffers(a_log), buffers(dt)

    def run(data, boundaries, state):
        qp, kp, vp, fp, bp = [buffers(x) for x in data]
        op = buffers(torch.full_like(data[2], float("nan")))
        cp = buffers(torch.tensor(boundaries, dtype=torch.int32))
        glm.lib.glm_kda_scan(glm.ctx, op, state, qp, kp, vp, fp, bp, ap, dp, cp, heads, len(lengths), -5.)
        return download(glm, op, data[2].shape, torch.bfloat16)

    actual = run([q, k, v, f, beta], cu, sp)
    actual_state = download(glm, sp, initial.shape, torch.float32)
    torch.testing.assert_close(actual, ref, atol=2e-3, rtol=1e-2)
    torch.testing.assert_close(actual_state, ref_state, atol=2e-6, rtol=2e-4)

    # Split every sequence independently, preserving empty slots and the state slot mapping.
    sp = buffers(initial)
    result = torch.empty_like(actual)
    for phase in [0, 1]:
        indices, lens = [], []
        for b, length in enumerate(lengths):
            split = min(2, length)
            begin, end = (0, split) if phase == 0 else (split, length)
            indices.extend(range(cu[b] + begin, cu[b] + end))
            lens.append(end - begin)
        if indices:
            result[indices] = run([x[indices] for x in [q, k, v, f, beta]], [0, *np.cumsum(lens).tolist()], sp)
    torch.testing.assert_close(result, actual, atol=0, rtol=0)
    torch.testing.assert_close(download(glm, sp, initial.shape, torch.float32), actual_state, atol=0, rtol=0)


def test_kda_graph_replay_advances_state(glm, buffers):
    torch.manual_seed(82)
    heads = 2
    q, k, v, f = [torch.randn(1, heads, 128).bfloat16() for _ in range(4)]
    beta = torch.randn(1, heads).bfloat16()
    a_log, dt = torch.randn(heads), torch.randn(heads, 128)
    state = torch.randn(1, heads, 128, 128) * .1
    tensors = [q, k, v, f, beta, a_log, dt, torch.tensor([0, 1], dtype=torch.int32)]
    pointers = [buffers(t) for t in tensors]
    sp, out = buffers(state), buffers(torch.empty_like(v))
    glm.synchronize()
    glm.graph_begin_capture()
    glm.lib.glm_kda_scan(glm.ctx, out, sp, *pointers, heads, 1, -5.)
    graph = glm.graph_end_capture()
    executable = glm.graph_instantiate(graph)
    try:
        for _ in range(3):
            glm.graph_launch(executable)
            expected, state = kda_reference(q, k, v, f, beta, a_log, dt, [0, 1], state)
            torch.testing.assert_close(download(glm, out, v.shape, torch.bfloat16), expected, atol=2e-3, rtol=1e-2)
            torch.testing.assert_close(download(glm, sp, state.shape, torch.float32), state, atol=2e-6, rtol=2e-4)
    finally:
        glm.graph_exec_destroy(executable)
        glm.graph_destroy(graph)


def test_kda_sigmoid_norm(glm, buffers):
    torch.manual_seed(4)
    x, gate = [torch.randn(11, 128).bfloat16() for _ in range(2)]
    weight = torch.randn(128).bfloat16()
    out = buffers(torch.empty_like(x))
    glm.lib.glm_kda_norm(glm.ctx, out, buffers(x), buffers(gate), buffers(weight), 11, 1e-5)
    expected = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
                * weight.float() * gate.float().sigmoid()).bfloat16().float()
    torch.testing.assert_close(download(glm, out, x.shape, torch.bfloat16), expected, atol=2e-3, rtol=8e-3)


@pytest.mark.parametrize("dim,iters", [(37, 1), (128, 20), (4096, 20)])
def test_mhc_projection_sinkhorn_mix_and_graph(glm, buffers, dim, iters):
    torch.manual_seed(18)
    rows = 3
    x = torch.randn(rows, 4, dim).bfloat16()
    fn = (torch.randn(24, 4 * dim) / (4 * dim)**.5).bfloat16()
    base, scale = torch.randn(24), torch.tensor([.3, .2, .4])
    flat = x.flatten(1).float()
    logits = (flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-5)) @ fn.float().T
    pre = torch.sigmoid(logits[:, :4] * scale[0] + base[:4]) + 1e-6
    post = 2 * torch.sigmoid(logits[:, 4:8] * scale[1] + base[4:8])
    comb = (logits[:, 8:] * scale[2] + base[8:]).reshape(rows, 4, 4).softmax(-1) + 1e-6
    comb /= comb.sum(-2, keepdim=True) + 1e-6
    for _ in range(iters - 1):
        comb /= comb.sum(-1, keepdim=True) + 1e-6
        comb /= comb.sum(-2, keepdim=True) + 1e-6
    collapsed = (pre[:, :, None] * x.float()).sum(1).bfloat16().float()
    block = torch.randn(rows, dim).bfloat16()
    mixed = (post[:, :, None] * block.float()[:, None, :] + comb.transpose(1, 2) @ x.float()).bfloat16().float()

    xp, fp, bp, sp, blk = [buffers(t) for t in [x, fn, base, scale, block]]
    cp = buffers(torch.empty(rows, dim, dtype=torch.bfloat16))
    pp, mp = buffers(torch.empty_like(post)), buffers(torch.empty_like(comb))
    out = buffers(torch.empty_like(x))

    def launch():
        glm.lib.glm_mhc_pre(glm.ctx, cp, pp, mp, xp, fp, bp, sp, rows, dim, 1e-5, 1e-6, iters)
        glm.lib.glm_mhc_post(glm.ctx, out, xp, blk, pp, mp, rows, dim)

    launch()
    glm.synchronize()
    glm.graph_begin_capture()
    launch()
    graph = glm.graph_end_capture()
    executable = glm.graph_instantiate(graph)
    try:
        for _ in range(2):
            glm.graph_launch(executable)
            torch.testing.assert_close(download(glm, cp, collapsed.shape, torch.bfloat16), collapsed, atol=.008, rtol=.008)
            torch.testing.assert_close(download(glm, pp, post.shape, torch.float32), post, atol=2e-6, rtol=2e-5)
            actual_comb = download(glm, mp, comb.shape, torch.float32)
            torch.testing.assert_close(actual_comb, comb, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(download(glm, out, mixed.shape, torch.bfloat16), mixed, atol=.008, rtol=.008)
            if iters == 20:
                torch.testing.assert_close(actual_comb.sum(-1), torch.ones(rows, 4), atol=5e-6, rtol=0)
                torch.testing.assert_close(actual_comb.sum(-2), torch.ones(rows, 4), atol=5e-6, rtol=0)
    finally:
        glm.graph_exec_destroy(executable)
        glm.graph_destroy(graph)

    # GLM's final head is a mean, not a learned hyper-head.
    glm.lib.glm_mhc_streams(glm.ctx, cp, xp, rows, dim, 0)
    torch.testing.assert_close(download(glm, cp, (rows, dim), torch.bfloat16), x.float().mean(1).bfloat16().float(), atol=0, rtol=0)
    glm.lib.glm_mhc_streams(glm.ctx, out, blk, rows, dim, 1)
    torch.testing.assert_close(download(glm, out, x.shape, torch.bfloat16), block.float()[:, None, :].expand_as(x), atol=0, rtol=0)
