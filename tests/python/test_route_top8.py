"""Exact comparison against the unfused routing pipeline, including BF16 rounding."""
import pytest
import torch


def unfused(glm, logits, bias, scale, normalize):
    rows = logits.shape[0]
    sigmoid = torch.empty_like(logits)
    biased = torch.empty_like(logits)
    values = torch.empty((rows, 8), dtype=torch.bfloat16, device=logits.device)
    indices = torch.empty((rows, 8), dtype=torch.int32, device=logits.device)
    selected = torch.empty_like(values)
    weights = torch.empty_like(values)
    glm.sigmoid(sigmoid, logits, logits.numel())
    glm.add_broadcast(biased, sigmoid, bias, 256, rows)
    glm.topk(values, indices, biased, 8, 256, rows)
    glm.gather(selected, sigmoid, indices, 8, 256, rows)
    glm.row_normalize(weights, selected, scale, rows, 8, normalize)
    glm.synchronize()
    # Independent stable selection, including lower-index ties.
    expected_indices = torch.argsort(biased.float(), descending=True, stable=True)[:, :8]
    assert torch.equal(indices.long(), expected_indices)
    return weights, indices


@pytest.mark.parametrize("rows", [1, 4, 8, 32, 1024])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("pattern", ["random", "ties", "saturation", "zero_sum"])
def test_route_top8_exact(glm, device, rows, normalize, pattern):
    torch.manual_seed(123)
    logits = torch.randn(rows, 256, device=device).to(torch.bfloat16)
    bias = (torch.randn(256, device=device) * .15).to(torch.bfloat16)
    if pattern == "ties":
        logits = torch.randint(-2, 3, (rows, 256), device=device).to(torch.bfloat16)
        bias.zero_()
    elif pattern == "saturation":
        logits *= 100
        logits[:, :8] = float("inf")
        logits[:, 8:16] = -float("inf")
    elif pattern == "zero_sum":
        logits.fill_(-float("inf"))
        bias.zero_()
    torch.cuda.synchronize(device)
    ref_weights, ref_indices = unfused(glm, logits, bias, 2.5, normalize)
    weights, indices = torch.empty_like(ref_weights), torch.empty_like(ref_indices)
    for _ in range(3):
        glm.route_top8(weights, indices, logits, bias, rows, 2.5, normalize)
        glm.synchronize()
        assert torch.equal(indices, ref_indices)
        assert torch.equal(weights.view(torch.int16), ref_weights.view(torch.int16))


@pytest.mark.parametrize("scale", [0.0, 1.0, 2.5, -1.25])
def test_route_top8_bf16_range_and_graph(glm, device, scale):
    # All BF16 bit patterns, excluding NaNs (unsupported by the original router).
    logits = torch.arange(65536, device=device, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    logits.nan_to_num_(nan=0.0, posinf=float("inf"), neginf=-float("inf"))
    logits = logits.reshape(256, 256)
    bias = torch.linspace(-.5, .5, 256, device=device).to(torch.bfloat16)
    weights = torch.empty((256, 8), dtype=torch.bfloat16, device=device)
    indices = torch.empty((256, 8), dtype=torch.int32, device=device)
    torch.cuda.synchronize(device)
    glm.route_top8(weights, indices, logits, bias, 256, scale, True)
    glm.synchronize()
    glm.graph_begin_capture()
    glm.route_top8(weights, indices, logits, bias, 256, scale, True)
    graph = glm.graph_end_capture()
    executable = glm.graph_instantiate(graph)
    try:
        for _ in range(3):
            ref_weights, ref_indices = unfused(glm, logits, bias, scale, True)
            glm.graph_launch(executable)
            glm.synchronize()
            assert torch.equal(indices, ref_indices)
            assert torch.equal(weights.view(torch.int16), ref_weights.view(torch.int16))
            logits.copy_(logits.roll(13, dims=1))
            bias.neg_()
            torch.cuda.synchronize(device)
    finally:
        glm.graph_exec_destroy(executable)
        glm.graph_destroy(graph)
