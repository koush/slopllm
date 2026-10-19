"""Cross-check the expected-value calculations used by production-kernel tests."""
import base64
import json
from pathlib import Path
import subprocess

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]


def test_oracle_rope_cache_router_and_indexer():
    torch.manual_seed(7124)
    x = (torch.randn(1024) * 3).bfloat16().float()
    logits = torch.randn(384) * 12
    bias = torch.randn(384)
    q = torch.randn(32, 128).bfloat16().float()
    keys = torch.randn(73, 128).bfloat16().float()
    weights = torch.randn(32).bfloat16().float()
    payload = dict(x=x.tolist(), logits=logits.tolist(), bias=bias.tolist(),
                   q=q.flatten().tolist(), keys=keys.tolist(), weights=weights.tolist())
    script = """
const fs = require('node:fs');
const m = require('./tests/deepseek_v41_oracles.ts');
const p = JSON.parse(fs.readFileSync(0, 'utf8'));
const x = Float32Array.from(p.x);
const page = Buffer.alloc(128 * 584);
m.writeDsv4Cache(page, 65, x.subarray(0,512));
const q = m.dsv41IndexQuantize(Float32Array.from(p.q));
const keys = p.keys.map(k => m.dsv41IndexQuantize(Float32Array.from(k)));
const scores = m.dsv41IndexScores(q, keys, Float32Array.from(p.weights));
console.log(JSON.stringify({
  rope: [false,true].map(c => [0,37,512,16000].map(pos => Array.from(m.dsv41Rope(x,pos,c)))),
  cache: page.toString('base64'),
  bf16: m.bf16Round(Float32Array.from([1.00390625,1.01171875,-1.00390625,-1.01171875])).toString('base64'),
  route: m.dsv41Route(Float32Array.from(p.logits),Float32Array.from(p.bias)),
  emptyRoute: m.dsv41Route(new Float32Array(384).fill(-1000),new Float32Array(384)),
  scores: Array.from(scores),
  swiglu: Array.from(m.dsv41SwiGlu(Float32Array.from([-30,-11,0,11,30]),Float32Array.from([30,-30,1,20,-20])))
}));
"""
    result = subprocess.run(["node", "--require", "tsx/cjs", "-e", script],
                            cwd=ROOT, input=json.dumps(payload), capture_output=True, text=True, check=True)
    actual = json.loads(result.stdout)
    # YaRN reference: extrapolate high frequencies, interpolate low frequencies.
    for compressed in [False, True]:
        theta = 160000 if compressed else 10000
        freq = 1 / (theta ** (torch.arange(0, 64, 2).float() / 64))
        if compressed:
            import math
            low = math.floor(64 * math.log(65536 / (32 * 2 * math.pi)) / (2 * math.log(theta)))
            high = math.ceil(64 * math.log(65536 / (2 * math.pi)) / (2 * math.log(theta)))
            ramp = ((torch.arange(32).float() - low) / (high - low)).clamp(0, 1)
            freq = freq * (1 - ramp) + freq / 16 * ramp
        for i, pos in enumerate([0, 37, 512, 16000]):
            expected = x.reshape(2, 512).clone()
            rot = expected[:, -64:].reshape(2, 32, 2).clone()
            c, s = (pos * freq).cos(), (pos * freq).sin()
            expected[:, -64::2] = rot[..., 0] * c - rot[..., 1] * s
            expected[:, -63::2] = rot[..., 0] * s + rot[..., 1] * c
            torch.testing.assert_close(torch.tensor(actual["rope"][int(compressed)][i]), expected.flatten(), atol=.002, rtol=.002)
    cache = base64.b64decode(actual["cache"])
    values = x[:448].reshape(7, 64)
    exp = (values.abs().amax(1).clamp_min(1e-4) / 448).log2().ceil()
    quant = (values / 2**exp[:, None]).to(torch.float8_e4m3fn).view(torch.uint8)
    offset = 64 * 584 + 576
    assert cache[offset:offset+448] == quant.numpy().tobytes()
    assert cache[offset+448:offset+576] == x[448:512].bfloat16().view(torch.uint16).numpy().tobytes()
    assert cache[64*584+64*576+8:64*584+64*576+15] == (exp+127).byte().numpy().tobytes()
    assert not any(cache[:64*584])
    ties = torch.tensor([1.00390625, 1.01171875, -1.00390625, -1.01171875])
    assert base64.b64decode(actual["bf16"]) == ties.bfloat16().view(torch.uint16).numpy().tobytes()
    scores = torch.nn.functional.softplus(logits).sqrt()
    chosen = (scores + bias).topk(6).indices
    assert [r["expert"] for r in actual["route"]] == chosen.tolist()
    assert actual["emptyRoute"] == [{"expert": i, "weight": 0} for i in range(6)]
    np.testing.assert_allclose([r["weight"] for r in actual["route"]], (scores[chosen] / scores[chosen].sum() * 1.5).numpy(), rtol=1e-6)
    def quantize(v):
        scale = 2 ** (v.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448).log2().ceil()
        return (v / scale).to(torch.float8_e4m3fn).float() * scale
    expected_scores = ((quantize(q) @ quantize(keys).T).relu() * weights[:, None] / 64).sum(0)
    torch.testing.assert_close(torch.tensor(actual["scores"]), expected_scores, rtol=2e-5, atol=2e-5)
    gate = torch.tensor([-30., -11., 0., 11., 30.]).clamp(max=10)
    up = torch.tensor([30., -30., 1., 20., -20.]).clamp(-10, 10)
    torch.testing.assert_close(torch.tensor(actual['swiglu']), torch.nn.functional.silu(gate) * up)
