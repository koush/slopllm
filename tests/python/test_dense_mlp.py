import torch
import pytest
from helpers import ATOL, RTOL, get_model_path, has_model_cached

try:
    from safetensors import safe_open
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


def torch_rmsnorm(x, weight, eps):
    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    inv_rms = torch.rsqrt(var + eps)
    return (weight.float() * x_f * inv_rms).to(torch.bfloat16)


def torch_dense_mlp(x, gate_w, up_w, down_w, norm_w, eps=1e-5):
    h = torch_rmsnorm(x, norm_w, eps)
    gate = torch.nn.functional.linear(h, gate_w)
    up = torch.nn.functional.linear(h, up_w)
    gate_up = (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.bfloat16)
    down = torch.nn.functional.linear(gate_up, down_w)
    return down


def test_dense_mlp_random_small(glm, device):
    hidden = 256
    intermediate = 512
    batch = 2
    eps = 1e-5

    x = torch.randn(batch, hidden, dtype=torch.bfloat16, device=device)
    norm_w = torch.randn(hidden, dtype=torch.bfloat16, device=device)
    gate_w = torch.randn(intermediate, hidden, dtype=torch.bfloat16, device=device)
    up_w = torch.randn(intermediate, hidden, dtype=torch.bfloat16, device=device)
    down_w = torch.randn(hidden, intermediate, dtype=torch.bfloat16, device=device)

    normed = torch.empty_like(x)
    glm.rmsnorm(normed, x, norm_w, eps, hidden, batch)

    gate = torch.empty(batch, intermediate, dtype=torch.bfloat16, device=device)
    glm.linear(gate, normed, gate_w, batch, intermediate, hidden)

    up = torch.empty(batch, intermediate, dtype=torch.bfloat16, device=device)
    glm.linear(up, normed, up_w, batch, intermediate, hidden)

    gate_up = torch.empty(batch, intermediate, dtype=torch.bfloat16, device=device)
    glm.silu_and_mul(gate_up, gate, up, intermediate, batch)

    down = torch.empty(batch, hidden, dtype=torch.bfloat16, device=device)
    glm.linear(down, gate_up, down_w, batch, hidden, intermediate)

    ref = torch_dense_mlp(x, gate_w, up_w, down_w, norm_w, eps)
    tol = 0.5 if hidden * intermediate > 1_000_000 else 5e-2
    torch.testing.assert_close(down.cpu(), ref.cpu(), atol=tol, rtol=tol)


@pytest.mark.skipif(
    not (HAS_SAFETENSORS and has_model_cached("zai-org/GLM-5.1")),
    reason="safetensors not installed or model not cached"
)
def test_dense_mlp_layer0_weights(glm, device):
    import json
    import os

    model_dir = get_model_path("zai-org/GLM-5.1")
    index_path = os.path.join(model_dir, "model.safetensors.index.json")

    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    shard_file = weight_map.get("model.layers.0.mlp.gate_proj.weight")
    if not shard_file:
        pytest.skip("Layer 0 weights not found in index")

    shard_path = os.path.join(model_dir, shard_file)
    if not os.path.exists(shard_path):
        pytest.skip(f"Shard {shard_file} not found at {shard_path}")

    hidden = 6144
    intermediate = 12288
    batch = 1
    eps = 1e-5

    weights = {}
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith("model.layers.0.mlp.") or key.startswith("model.layers.0.input_layernorm."):
                weights[key] = f.get_tensor(key)

    if len(weights) < 4:
        pytest.skip("Not all layer 0 weights found in shard")

    norm_w = weights["model.layers.0.input_layernorm.weight"].bfloat16().to(device)
    gate_w = weights["model.layers.0.mlp.gate_proj.weight"].bfloat16().to(device)
    up_w = weights["model.layers.0.mlp.up_proj.weight"].bfloat16().to(device)
    down_w = weights["model.layers.0.mlp.down_proj.weight"].bfloat16().to(device)

    x = torch.randn(batch, hidden, dtype=torch.bfloat16, device=device)

    normed = torch.empty_like(x)
    glm.rmsnorm(normed, x, norm_w, eps, hidden, batch)

    gate = torch.empty(batch, intermediate, dtype=torch.bfloat16, device=device)
    glm.linear(gate, normed, gate_w, batch, intermediate, hidden)

    up = torch.empty(batch, intermediate, dtype=torch.bfloat16, device=device)
    glm.linear(up, normed, up_w, batch, intermediate, hidden)

    gate_up = torch.empty(batch, intermediate, dtype=torch.bfloat16, device=device)
    glm.silu_and_mul(gate_up, gate, up, intermediate, batch)

    down = torch.empty(batch, hidden, dtype=torch.bfloat16, device=device)
    glm.linear(down, gate_up, down_w, batch, hidden, intermediate)

    ref = torch_dense_mlp(x, gate_w, up_w, down_w, norm_w, eps)
    torch.testing.assert_close(down.cpu(), ref.cpu(), atol=5e-2, rtol=5e-2)
