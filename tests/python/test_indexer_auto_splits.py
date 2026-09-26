"""Replay one auto-budget graph across growing/shrinking device-side lengths."""
import pytest
import torch

from helpers import pack_indexer_k


@pytest.mark.parametrize("topk", [1, 2048])
def test_auto_selector_replay_lengths(glm, device, topk):
    torch.manual_seed(927)
    batch, capacity = 4, 65536
    scores = torch.randint(-8, 24, (batch, capacity), device=device).to(torch.bfloat16)
    scores[:, ::7] = -torch.inf
    lengths = torch.full((batch,), capacity, dtype=torch.int32, device=device)
    idx = torch.empty(batch, topk, dtype=torch.int32, device=device)
    val = torch.empty(batch, topk, dtype=torch.bfloat16, device=device)
    hist = torch.empty(batch, 1056, dtype=torch.int32, device=device)
    meta = torch.empty(batch, 4, dtype=torch.int32, device=device)

    def call(splits):
        glm.topk_from_scores(idx, val, scores, lengths, hist, meta,
                             batch, capacity, topk, splits)

    torch.cuda.synchronize()
    call(0)
    glm.synchronize()
    glm.graph_begin_capture()
    call(0)
    graph = glm.graph_end_capture()
    executable = glm.graph_instantiate(graph)
    try:
        for live in ([65536, 32769, 2049, 0], [0, 1, 33, 2048],
                     [2049, 65536, 0, 32768], [1, 0, 2, 0]):
            lengths.copy_(torch.tensor(live, dtype=torch.int32, device=device))
            hist.fill_(0x12345678)
            meta.fill_(0x12345678)
            torch.cuda.synchronize()
            glm.graph_launch(executable)
            glm.synchronize()
            actual_idx, actual_val = idx.clone(), val.clone()
            torch.cuda.synchronize()
            call(1)
            glm.synchronize()
            assert torch.equal(idx, actual_idx), live
            assert torch.equal(val.view(torch.int16), actual_val.view(torch.int16)), live
    finally:
        glm.graph_exec_destroy(executable)
        glm.graph_destroy(graph)


@pytest.mark.parametrize("layout", ["paged", "flat", "cp"])
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_auto_scorer_replay_lengths(glm, device, monkeypatch, layout, fp8, masked):
    if fp8 and torch.cuda.get_device_capability(device)[0] < 12:
        pytest.skip("FP8 MMA requires SM120")
    monkeypatch.setenv("GLM_INDEXER_DECODE_FP8_MMA", "1" if fp8 else "0")
    torch.manual_seed(928)
    rows, heads, dim, page_size, capacity, topk = 4, 32, 128, 64, 65536, 2048
    q = torch.randn(rows, heads, dim, dtype=torch.bfloat16, device=device)
    weights = torch.randn(rows, heads, dtype=torch.bfloat16, device=device)
    k, scales = pack_indexer_k(torch.randn(capacity, dim, dtype=torch.bfloat16, device=device))
    pi = torch.arange(capacity // page_size, dtype=torch.int32, device=device)
    pip = torch.tensor([0, capacity // page_size], dtype=torch.int32, device=device)
    last = torch.tensor([page_size], dtype=torch.int32, device=device)
    qo = torch.tensor([0, rows], dtype=torch.int32, device=device)
    flat = torch.tensor([0, capacity], dtype=torch.int32, device=device)
    global_last = torch.tensor([page_size * 8], dtype=torch.int32, device=device)
    mask = torch.tensor([0x55] * 32, dtype=torch.uint8, device=device)
    mask_indptr = torch.tensor([0, 32], dtype=torch.int32, device=device)
    mask_len = torch.tensor([64], dtype=torch.int32, device=device)
    idx = torch.empty(rows, topk, dtype=torch.int32, device=device)
    val = torch.empty(rows, topk, dtype=torch.bfloat16, device=device)
    scores = torch.empty(rows, capacity, dtype=torch.bfloat16, device=device)
    row_len = torch.empty(rows, dtype=torch.int32, device=device)
    hist = torch.empty(rows, 1056, dtype=torch.int32, device=device)
    meta = torch.empty(rows, 4, dtype=torch.int32, device=device)
    world, rank = (8, 7) if layout == "cp" else (1, 0)

    def call(splits):
        glm.indexer_score_topk_v2(
            idx, val, q, k, scales, weights, pi, pip, last, qo,
            dim ** -0.5, rows, heads, dim, page_size, topk, True,
            scores, row_len, hist, meta, capacity, splits,
            cp_world_size=world, cp_rank=rank,
            global_last_page_len=global_last if layout == "cp" else None,
            kv_token_indptr=flat if layout == "flat" else None,
            custom_mask=mask if masked else None,
            mask_indptr=mask_indptr if masked else None,
            mask_kv_len=mask_len if masked else None)

    torch.cuda.synchronize()
    call(0)
    glm.synchronize()
    glm.graph_begin_capture()
    call(0)
    graph = glm.graph_end_capture()
    executable = glm.graph_instantiate(graph)
    try:
        # CP rank 7 has empty rows at global length 4; the 64-token scoring
        # tile, identity cutoff, and long-row radix path are all exercised.
        for length in [4, 65 * world, capacity * world - 1, 2049 * world,
                       8, 32769 * world, 2048 * world, 4]:
            pages = (length + page_size * world - 1) // (page_size * world)
            global_tail = length - (pages - 1) * page_size * world
            local_tail = max(0, (global_tail - 1 - rank) // world + 1)
            pip[1] = pages
            last[0] = local_tail
            global_last[0] = global_tail
            flat[1] = length
            scores.fill_(torch.nan)
            hist.fill_(0x12345678)
            meta.fill_(0x12345678)
            torch.cuda.synchronize()
            glm.graph_launch(executable)
            glm.synchronize()
            expected_lengths = [max(0, (length - rows + qi - rank) // world + 1)
                                for qi in range(rows)]
            assert row_len.cpu().tolist() == expected_lengths
            actual_idx, actual_val = idx.clone(), val.clone()
            actual_scores = scores.clone()
            torch.cuda.synchronize()
            call(32)
            glm.synchronize()
            assert torch.equal(idx, actual_idx), (layout, fp8, masked, length)
            assert torch.equal(val.view(torch.int16), actual_val.view(torch.int16))
            for qi, valid in enumerate(expected_lengths):
                assert not torch.isnan(actual_scores[qi, :valid]).any()
                assert torch.equal(scores[qi, :valid].view(torch.int16),
                                   actual_scores[qi, :valid].view(torch.int16))
    finally:
        glm.graph_exec_destroy(executable)
        glm.graph_destroy(graph)
