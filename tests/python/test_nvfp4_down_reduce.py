"""Fused down/reduction preserves the native BF16 intermediate and sum order."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("rows", [1, 4, 8, 12, 16, 20, 24, 28, 32])
def test_nvfp4_down_reduce_graph_matches_unfused(glm, device, rows):
    torch.manual_seed(814 + rows)
    experts, n, k, topk = 17, 6144, 256, 8
    count = rows * topk
    weights = torch.randint(0, 256, (experts, n, k // 2), dtype=torch.uint8, device=device)
    scales = torch.randint(0x10, 0x50, (experts, n, k // 16), dtype=torch.uint8, device=device)
    scale2 = 0.03125 + torch.rand(experts, dtype=torch.float32, device=device) * 0.123
    ptrs = [torch.tensor([t[e].data_ptr() for e in range(experts)], dtype=torch.int64, device=device)
            for t in (weights, scales, scale2)]
    ids = torch.rand(rows, experts, device=device).argsort(dim=1)[:, :topk].contiguous().view(-1).to(torch.int32)
    activated = torch.randn(count, k, dtype=torch.bfloat16, device=device)
    routing = torch.randn(count, dtype=torch.bfloat16, device=device)
    down = torch.empty(count, n, dtype=torch.bfloat16, device=device)
    expected = torch.empty(rows, n, dtype=torch.bfloat16, device=device)
    actual = torch.empty_like(expected)
    torch.cuda.synchronize(device)

    def fused():
        glm.nvfp4_mul_mat_id_reduce(actual, activated, *(p.data_ptr() for p in ptrs), ids, routing, rows)

    fused()
    glm.synchronize()
    glm.graph_begin_capture()
    fused()
    graph = glm.graph_end_capture()
    executable = glm.graph_instantiate(graph)
    try:
        for iteration in range(3):
            # Reuse captured pointers with changed inputs, routes and coefficients.
            activated.normal_()
            routing.normal_()
            ids.copy_(torch.rand(rows, experts, device=device).argsort(dim=1)[:, :topk].contiguous().view(-1).to(torch.int32))
            torch.cuda.synchronize(device)
            glm.nvfp4_mul_mat_id(down, activated, *(p.data_ptr() for p in ptrs), ids, 1, count, n, k)
            glm.scatter_add_rows(expected, down, routing, topk, n, rows, 0)
            glm.fill(actual, float("nan"), rows * n)
            glm.graph_launch(executable)
            glm.synchronize()
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), f"rows={rows}, replay={iteration}"
    finally:
        glm.graph_exec_destroy(executable)
        glm.graph_destroy(graph)
