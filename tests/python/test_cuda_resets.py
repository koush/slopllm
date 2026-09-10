"""Scratch initialization must survive dirty buffers and repeated graph replay."""
import ctypes

import pytest
import torch

from helpers import pack_indexer_k

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _replay_dirty(glm, run, buffers, check):
    torch.cuda.synchronize(glm.device)
    run()  # warm any lazy native setup before capture
    glm.synchronize()
    glm.graph_begin_capture()
    run()
    graph = glm.graph_end_capture()
    executable = glm.graph_instantiate(graph)
    try:
        for poison in (1.0, -3.0, 7.0):
            for buffer in buffers:
                # Fill uses BF16 element counts; +0 also clears I32/F32 bits.
                glm.fill(buffer, poison, buffer.numel() * buffer.element_size() // 2)
            glm.graph_launch(executable)
            glm.synchronize()
            check()
    finally:
        glm.graph_exec_destroy(executable)
        glm.graph_destroy(graph)


@pytest.mark.parametrize("stride", [32768, 32769, 65536])
@pytest.mark.parametrize("rows", [1, 3])
def test_topk_dirty_scratch_graph(glm, device, stride, rows):
    torch.manual_seed(801)
    topk = 32
    scores = torch.randn(rows, stride, dtype=torch.bfloat16, device=device)
    lengths = torch.tensor([stride] if rows == 1 else [0, 17, stride], dtype=torch.int32, device=device)
    indices = torch.empty(rows, topk, dtype=torch.int32, device=device)
    values = torch.empty(rows, topk, dtype=torch.bfloat16, device=device)
    hist = torch.empty(rows, 1056, dtype=torch.int32, device=device)
    meta = torch.empty(rows, 4, dtype=torch.int32, device=device)
    expected = torch.topk(scores[-1].float(), topk).values.sort().values

    def run():
        glm.topk_from_scores(indices, values, scores, lengths, hist, meta,
                             rows, stride, topk, 128)

    def check():
        if rows == 3:
            assert torch.all(indices[0] == -1)
            assert torch.equal(indices[1, :17], torch.arange(17, device=device))
            assert torch.all(indices[1, 17:] == -1)
        selected = indices[-1].long()
        assert selected.unique().numel() == topk
        torch.testing.assert_close(values[-1].float().sort().values, expected, rtol=0, atol=0)
        torch.testing.assert_close(values[-1], scores[-1, selected], rtol=0, atol=0)

    _replay_dirty(glm, run, [hist, meta, indices, values], check)


@pytest.mark.parametrize("fp8", [False, True])
def test_prefill_overwrites_dirty_histograms(glm, device, monkeypatch, fp8):
    monkeypatch.setenv("GLM_INDEXER_DECODE_FP8_MMA", "1" if fp8 else "0")
    torch.manual_seed(802)
    rows, length, topk = 65, 65, 32
    q = torch.randn(rows, 32, 128, dtype=torch.bfloat16, device=device)
    weights = torch.randn(rows, 32, dtype=torch.bfloat16, device=device)
    k, scales = pack_indexer_k(torch.randn(2, 64, 128, dtype=torch.bfloat16, device=device))
    pages = torch.tensor([0, 1], dtype=torch.int32, device=device)
    indptr = torch.tensor([0, 2], dtype=torch.int32, device=device)
    last = torch.tensor([1], dtype=torch.int32, device=device)
    qo = torch.tensor([0, rows], dtype=torch.int32, device=device)
    scores = torch.empty(rows, length, dtype=torch.bfloat16, device=device)
    lengths = torch.empty(rows, dtype=torch.int32, device=device)
    coarse = torch.empty(rows, 1024, dtype=torch.int32, device=device)
    fine = torch.empty(rows, 64, dtype=torch.int32, device=device)
    meta = torch.empty(rows, 4, dtype=torch.int32, device=device)
    indices = torch.empty(rows, topk, dtype=torch.int32, device=device)
    values = torch.empty(rows, topk, dtype=torch.bfloat16, device=device)

    def run():
        glm.indexer_score_topk_prefill(indices, values, q, k, scales, weights,
            pages, indptr, last, qo, 128 ** -0.5, rows, 32, 128, 64, topk, True,
            scores, lengths, length, coarse, fine, meta, 2)

    def check():
        torch.testing.assert_close(lengths, torch.arange(1, rows + 1, dtype=torch.int32, device=device))
        for row in (0, 16, 31, 32, 64):
            valid = row + 1
            if valid <= topk:
                assert torch.equal(indices[row, :valid], torch.arange(valid, device=device))
                assert torch.all(indices[row, valid:] == -1)
            else:
                selected = indices[row].long()
                assert selected.unique().numel() == topk
                expected = torch.topk(scores[row, :valid].float(), topk).values.sort().values
                torch.testing.assert_close(values[row].float().sort().values, expected, rtol=0, atol=0)

    _replay_dirty(glm, run, [coarse, fine, meta, indices, values], check)


@pytest.mark.parametrize("mode", ["bf16", "nvfp4", "nvfp4_split"])
def test_grouped_moe_resets_dirty_counters(glm, device, mode):
    experts, tokens, topk, n, k = 8, 4, 8, 64, 128
    count = tokens * topk
    x = (torch.arange(1, tokens + 1, device=device).float() / 8).to(torch.bfloat16)
    x = x[:, None].expand(tokens, k).contiguous()
    ids = torch.arange(experts, dtype=torch.int32, device=device).repeat(tokens)
    output = torch.empty(count, n, dtype=torch.bfloat16, device=device)
    vp, integer = ctypes.c_void_p, ctypes.c_int
    if mode == "bf16":
        weights = torch.ones(experts, n, k, dtype=torch.bfloat16, device=device)
        pointers = torch.tensor([w.data_ptr() for w in weights], dtype=torch.int64, device=device)
        size = glm.lib.glm_mma_moe_workspace_size
        size.argtypes = [integer] * 4
        size.restype = ctypes.c_size_t
        workspace = torch.empty(size(count, n, k, experts), dtype=torch.uint8, device=device)
        launch = glm.lib.glm_bf16_mul_mat_id_grouped_mma

        def run():
            launch(glm.ctx, output.data_ptr(), x.data_ptr(), pointers.data_ptr(), ids.data_ptr(),
                   topk, count, n, k, experts, workspace.data_ptr())
        scratch = [workspace]
    else:
        # E2M1 nibble 2 and E4M3 scale 0x38 both represent exactly 1.
        weights = torch.full((experts, n, k // 2), 0x22, dtype=torch.uint8, device=device)
        scales = torch.full((experts, n, k // 16), 0x38, dtype=torch.uint8, device=device)
        scale2 = torch.ones(experts, dtype=torch.float32, device=device)
        pointers = [torch.tensor([t[e].data_ptr() for e in range(experts)], dtype=torch.int64, device=device)
                    for t in (weights, scales, scale2)]
        if mode == "nvfp4":
            size = glm.lib.glm_mma_moe_coop_workspace_size
            size.argtypes, size.restype = [integer] * 4, ctypes.c_size_t
            workspace = torch.empty(size(count, n, k, experts), dtype=torch.uint8, device=device)
            launch = glm.lib.glm_nvfp4_mul_mat_id_grouped_mma_coop
            launch.argtypes, launch.restype = [vp] * 7 + [integer] * 5 + [vp], None

            def run():
                launch(glm.ctx, output.data_ptr(), x.data_ptr(), *(p.data_ptr() for p in pointers),
                       ids.data_ptr(), topk, count, n, k, experts, workspace.data_ptr())
            scratch = [workspace]
        else:
            size = glm.lib.glm_mma_moe_coop_scatter_workspace_size
            size.argtypes, size.restype = [integer] * 3, ctypes.c_size_t
            workspace = torch.empty(size(count, k, experts), dtype=torch.uint8, device=device)
            counter = torch.empty(1, dtype=torch.int32, device=device)
            scatter = glm.lib.glm_mma_moe_coop_scatter
            scatter.argtypes, scatter.restype = [vp] * 3 + [integer] * 4 + [vp], None
            launch = glm.lib.glm_mma_moe_coop_gemm
            launch.argtypes, launch.restype = [vp] * 4 + [integer] * 5 + [vp, vp, ctypes.c_bool, vp, vp], None

            def run():
                scatter(glm.ctx, x.data_ptr(), ids.data_ptr(), topk, count, k, experts, workspace.data_ptr())
                launch(glm.ctx, *(p.data_ptr() for p in pointers), experts, n, k, count, k,
                       workspace.data_ptr(), None, False, counter.data_ptr(), output.data_ptr())
            scratch = [workspace, counter]
    expected = x.float().sum(-1).repeat_interleave(topk)[:, None].expand(count, n).to(torch.bfloat16)

    def check():
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    _replay_dirty(glm, run, [*scratch, output], check)
