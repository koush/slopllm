import ctypes

import pytest
import torch


@pytest.mark.parametrize("tokens,topk,n,k,experts", [
    (65, 8, 256, 6144, 8),
    (65, 1, 6144, 256, 8),
    # Partial N tiles, preserving the grouped pipeline's 16-byte row alignment.
    (33, 3, 72, 128, 7),
    (9, 2, 136, 64, 5),
])
def test_grouped_bf16_mma(glm, device, tokens, topk, n, k, experts):
    torch.manual_seed(42)
    inputs = torch.randn(tokens, k, dtype=torch.bfloat16, device=device) * 0.1
    weights = torch.randn(experts, n, k, dtype=torch.bfloat16, device=device) * 0.1
    # Leave one expert empty; irregular routing exercises partial M tiles.
    ids = torch.randint(experts - 1, (tokens * topk,), dtype=torch.int32, device=device)
    pointers = torch.tensor([w.data_ptr() for w in weights], dtype=torch.int64, device=device)
    output = torch.empty(tokens * topk, n, dtype=torch.bfloat16, device=device)
    size = glm.lib.glm_mma_moe_workspace_size
    size.argtypes = [ctypes.c_int] * 4
    size.restype = ctypes.c_size_t
    workspace = torch.empty(size(tokens * topk, n, k, experts), dtype=torch.uint8, device=device)
    launch = glm.lib.glm_bf16_mul_mat_id_grouped_mma
    launch.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
    launch.restype = None

    def run():
        launch(glm.ctx, output.data_ptr(), inputs.data_ptr(), pointers.data_ptr(), ids.data_ptr(),
               topk, tokens * topk, n, k, experts, workspace.data_ptr())

    run()
    glm.synchronize()
    glm.graph_begin_capture()
    run()
    graph = glm.graph_end_capture()
    executable = glm.graph_instantiate(graph)
    try:
        for _ in range(2):
            output.fill_(float("nan"))
            workspace.fill_(0xA5)
            glm.graph_launch(executable)
            glm.synchronize()
            expected = torch.empty_like(output, dtype=torch.float32)
            for expert in range(experts - 1):
                routes = torch.where(ids == expert)[0]
                expected[routes] = inputs[routes // topk].float() @ weights[expert].float().T
            torch.testing.assert_close(output.float(), expected, atol=1e-2, rtol=2e-2)
            inputs.mul_(0.5)
    finally:
        glm.graph_exec_destroy(executable)
        glm.graph_destroy(graph)
