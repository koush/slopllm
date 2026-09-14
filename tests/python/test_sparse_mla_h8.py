"""Native H8 versus generic decode and FP32 attention on quantized inputs."""
import math

import pytest
import torch


@pytest.mark.parametrize("mode", ["packed", "split", "fp8"])
@pytest.mark.parametrize("cpb", [1, 2, 3, 32, "prefill"])
def test_native_h8(glm, device, monkeypatch, mode, cpb):
    torch.manual_seed(782)
    rows, heads, topk, splits = (32 if cpb == "prefill" else 8), 8, 2048, 32
    switch = "FLASHINFER_GLM_H8_PREFILL_SWAP_QK" if cpb == "prefill" else "FLASHINFER_GLM_H8_NATIVE"
    q = (torch.randn(rows, heads, 512, device=device) * .3).to(torch.float8_e4m3fn)
    qr = (torch.randn(rows, heads, 64, device=device) * .3).bfloat16()
    qs = 2.0 ** torch.randint(-2, 2, (rows, heads, 4), device=device).float()
    qd = (q.float().view(rows, heads, 4, 128) * qs[..., None]).view(rows, heads, 512)
    k = (torch.randn(3072, 512, device=device) * .3).to(torch.float8_e4m3fn)
    ks = torch.rand(3072, 4, device=device) * 1.5 + .1
    kr = (torch.randn(3072, 64, device=device) * .3).bfloat16()
    packed = torch.empty(3072, 656, dtype=torch.uint8, device=device)
    packed[:, :512] = k.view(torch.uint8)
    packed[:, 512:528] = ks.view(torch.uint8)
    packed[:, 528:] = kr.view(torch.uint8)
    kd = (k.float().view(-1, 4, 128) * ks[:, :, None]).view(-1, 512)
    indices = torch.full((rows, topk), -1, dtype=torch.int32, device=device)
    lengths = [0, 1, 63, 64, 65, 127, 1023, 2048] * (rows // 8)
    for i, n in enumerate(lengths):
        indices[i, :n] = torch.randperm(3072, device=device)[:n].int()
    # Holes within active chunks, plus an entirely invalid first chunk.
    indices[6, :64] = -1
    indices[7, 13::29] = -1
    valid = torch.tensor(lengths, dtype=torch.int32, device=device)
    mid = torch.full((rows, heads, splits, 512), float("nan"), dtype=torch.bfloat16, device=device)
    mid_lse = torch.empty(rows, heads, splits, device=device)
    out = torch.empty(rows, heads, 512, dtype=torch.bfloat16, device=device)
    lse = torch.empty(rows, heads, device=device)
    qb = qd.bfloat16()
    qp = torch.cat([qb, qr], -1)

    def run():
        if cpb == "prefill" and mode == "packed":
            glm.sparse_mla_prefill(
                qp.data_ptr(), packed.data_ptr(), indices.data_ptr(), out.data_ptr(), lse.data_ptr(),
                rows, heads, topk, 64, 1 / 16, 64 * 656, topk_length=valid.data_ptr())
        elif cpb == "prefill":
            glm.sparse_mla_prefill_split_q(
                (q if mode == "fp8" else qb).data_ptr(), qr.data_ptr(), packed.data_ptr(),
                indices.data_ptr(), out.data_ptr(), lse.data_ptr(), rows, heads, topk,
                1 / 16, 64 * 656, topk_length=valid.data_ptr(),
                q_scales=qs.data_ptr() if mode == "fp8" else None)
        elif mode == "packed":
            glm.sparse_mla_decode(
                qp.data_ptr(), packed.data_ptr(), indices.data_ptr(), mid.data_ptr(), mid_lse.data_ptr(),
                out.data_ptr(), lse.data_ptr(), rows, heads, topk, splits, 1 / 16, 64 * 656,
                chunks_per_block=cpb, topk_length=valid.data_ptr())
        else:
            glm.sparse_mla_decode_split_q(
                (q if mode == "fp8" else qb).data_ptr(), qr.data_ptr(), packed.data_ptr(),
                indices.data_ptr(), mid.data_ptr(), mid_lse.data_ptr(), out.data_ptr(), lse.data_ptr(),
                rows, heads, topk, splits, 1 / 16, 64 * 656,
                chunks_per_block=cpb, topk_length=valid.data_ptr(),
                q_scales=qs.data_ptr() if mode == "fp8" else 0)

    torch.cuda.synchronize(device)
    monkeypatch.setenv(switch, "0")
    run()
    glm.synchronize()
    generic, generic_lse = out.clone(), lse.clone()
    monkeypatch.setenv(switch, "1")
    run()
    glm.synchronize()
    torch.testing.assert_close(out, generic, atol=.002, rtol=.025)
    torch.testing.assert_close(lse, generic_lse, atol=.002, rtol=.001)
    for i, n in enumerate(lengths):
        slots = indices[i, :n].long()
        slots = slots[slots >= 0]
        if not slots.numel():
            assert torch.count_nonzero(out[i]) == 0
            continue
        scores = (qd[i] @ kd[slots].T + qr[i].float() @ kr[slots].float().T) / 16
        torch.testing.assert_close(out[i].float(), scores.softmax(-1) @ kd[slots], atol=.004, rtol=.04)
        torch.testing.assert_close(lse[i], scores.logsumexp(-1) / math.log(2), atol=.003, rtol=.001)

    glm.graph_begin_capture()
    run()
    graph = glm.graph_end_capture()
    executable = glm.graph_instantiate(graph)
    try:
        # Replay with changed masks/lengths and poisoned scratch/output. All
        # pointers remain stable; a previously empty row becomes nonempty.
        indices[0, :65] = indices[7, :65]
        valid[0] = 65
        valid[7] = 0
        out.fill_(float("nan"))
        mid.fill_(float("nan"))
        torch.cuda.synchronize(device)
        glm.graph_launch(executable)
        glm.synchronize()
        replay, replay_lse = out.clone(), lse.clone()
        monkeypatch.setenv(switch, "0")
        torch.cuda.synchronize(device)
        run()
        glm.synchronize()
        torch.testing.assert_close(replay, out, atol=.002, rtol=.025)
        torch.testing.assert_close(replay_lse, lse, atol=.002, rtol=.001)
        assert torch.count_nonzero(replay[0]) > 0
        assert torch.count_nonzero(replay[7]) == 0
    finally:
        glm.graph_exec_destroy(executable)
        glm.graph_destroy(graph)
