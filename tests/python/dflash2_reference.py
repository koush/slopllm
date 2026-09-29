"""Generate independent BF16 DFlash2 fixtures using real weights and synthetic inputs.

No target model, Transformers, or serving runtime is required. Attention uses an
explicit FP32 softmax and GQA head expansion; all saved inputs are deterministic.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--context-lengths", default="19,27")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--make-test-model", action="store_true", help="Create a small, well-conditioned six-layer checkpoint for strict numerical tests")
    args = parser.parse_args()
    root = Path(args.model_dir)
    if args.make_test_model:
        root.mkdir(parents=True, exist_ok=True)
        assert not (root / "model.safetensors").exists(), "Refusing to overwrite a checkpoint"
        cfg = {"architectures": ["DFlash2DraftModel"], "dtype": "bfloat16", "is_causal": False,
               "hidden_size": 256, "intermediate_size": 512, "num_hidden_layers": 6,
               "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
               "vocab_size": 128, "rms_norm_eps": 1e-5, "hidden_act": "silu",
               "sliding_window": 2048, "layer_types": ["sliding_attention"] * 6,
               "max_position_embeddings": 1048576, "rope_parameters": {"rope_theta": 1000000},
               "dflash_config": {"block_size": 8, "conv_kernel_size": 2, "conv_group_size": 16,
                                 "target_layer_ids": [5, 19, 33, 47, 61, 75], "mask_token_id": 127,
                                 "selector_rank": 32, "selector_top_k": 16}}
        torch.manual_seed(123)
        w = {}

        def matrix(name, rows, cols, scale=0.2):
            w[name] = (torch.randn(rows, cols) * (scale / cols ** 0.5)).bfloat16()

        matrix("fc.weight", 256, 1536)
        w["hidden_norm.weight"] = torch.ones(256, dtype=torch.bfloat16)
        w["norm.weight"] = torch.ones(256, dtype=torch.bfloat16)
        matrix("candidate_selector.hidden_projection.weight", 32, 256)
        matrix("candidate_selector.predecessor_codebook", 128, 32, 1.0)
        matrix("candidate_selector.successor_codebook", 128, 32, 1.0)
        for i in range(6):
            p = f"layers.{i}"
            for name in ["input_layernorm", "post_attention_layernorm"]:
                w[p + "." + name + ".weight"] = torch.ones(256, dtype=torch.bfloat16)
            for name in ["attention_conv", "mlp_conv"]:
                base = torch.ones(2, 2, 256)
                base[:, 1] = 0.15
                w[p + "." + name + ".base_kernel"] = base.bfloat16()
                matrix(p + "." + name + ".kernel_projection.weight", 64, 256, 0.05)
            for name in ["q", "k", "v"]:
                matrix(p + ".self_attn." + name + "_proj.weight", 2048 if name == "q" else 1024, 256)
            matrix(p + ".self_attn.o_proj.weight", 256, 2048)
            for name in ["q", "k"]:
                w[p + ".self_attn." + name + "_norm.weight"] = torch.ones(128, dtype=torch.bfloat16)
            for name in ["gate", "up"]:
                matrix(p + ".mlp." + name + "_proj.weight", 512, 256)
            matrix(p + ".mlp.down_proj.weight", 256, 512)
        save_file(w, str(root / "model.safetensors"))
        (root / "config.json").write_text(json.dumps(cfg))
    config = json.loads((root / "config.json").read_text())
    draft = config["dflash_config"]
    lengths = [int(v) for v in args.context_lengths.split(",")]
    assert all(n > 0 for n in lengths)
    batch, block, h = len(lengths), draft["block_size"], config["hidden_size"]
    hd, nq, nk = config["head_dim"], config["num_attention_heads"], config["num_key_value_heads"]
    offsets = [37 + 103 * i for i in range(batch)]
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    inputs = {f"hidden.{i}": torch.randn(sum(lengths), h).to(torch.bfloat16) for i in range(6)}
    # All masked positions use the same embedding, just as an actual draft block.
    embeddings = (torch.randn(h) * 0.2).repeat(batch, block, 1)
    embeddings[:, 0] = torch.randn(batch, h) * 0.2
    inputs["embeddings"] = embeddings.bfloat16().reshape(batch * block, h)
    inputs["selector.hidden"] = torch.randn(batch * (block - 1), h).bfloat16()
    inputs["selector.ids"] = torch.stack([torch.randperm(config["vocab_size"])[:draft["selector_top_k"]]
                                        for _ in range(batch * (block - 1))]).int()
    inputs["selector.logits"] = torch.randn(batch * (block - 1), draft["selector_top_k"])
    inputs["selector.anchors"] = torch.randint(config["vocab_size"], (batch,), dtype=torch.int32)
    inputs["conv.input"] = torch.randn(batch * block, h).bfloat16()
    inputs["conv.coefficients"] = (torch.randn(batch * block, 4 * h // draft["conv_group_size"]) * 0.1).bfloat16()
    weights = load_file(str(root / "model.safetensors"), device=args.device)
    tensors = {name: t.to(args.device) for name, t in inputs.items()}
    expected = {}

    def save(name, value):
        expected[name] = value.float().cpu().contiguous()

    def linear(x, name):
        return F.linear(x, weights[name + ".weight"])

    def norm(x, name, float_output=False):
        # Keep intermediates in FP32, as fused inference RMSNorm kernels do.
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + config["rms_norm_eps"])
        y = y * weights[name + ".weight"].float()
        return y if float_output else y.bfloat16()

    inv = 1.0 / (config["rope_parameters"]["rope_theta"] ** (torch.arange(0, hd, 2, device=args.device).float() / hd))
    neox = config.get("rope_is_neox_style", config.get("is_neox_style", True))

    def rope(x, positions):
        freq = positions.float()[:, None] * inv[None]
        if neox:
            phase = torch.cat((freq, freq), -1)[:, None]
            rotated = torch.cat((-x[..., hd // 2:], x[..., :hd // 2]), -1)
        else:
            phase = freq.repeat_interleave(2, -1)[:, None]
            rotated = torch.stack((-x[..., 1::2], x[..., ::2]), -1).flatten(-2)
        # Fused norm/RoPE evaluates the rotate in FP32, with BF16 cos/sin tables.
        return (x.float() * phase.cos().bfloat16().float() + rotated.float() * phase.sin().bfloat16().float()).bfloat16()

    def conv(x, coefficients, name, side):
        groups = h // draft["conv_group_size"]
        delta = coefficients.view(batch, block, 2, 2, groups)[:, :, side].float()
        base = weights[name + ".base_kernel"][side].float()
        coeff = delta.repeat_interleave(draft["conv_group_size"], -1) + base
        x = x.view(batch, block, h).float()
        shifted = F.pad(x[:, :-1], (0, 0, 1, 0))
        return (coeff[:, :, 0] * x + coeff[:, :, 1] * shifted).bfloat16().view(batch * block, h)

    context = norm(linear(torch.cat([tensors[f"hidden.{i}"] for i in range(6)], -1), "fc"), "hidden_norm")
    save("context", context)
    ctx_positions = torch.cat([torch.arange(n, device=args.device) + off for n, off in zip(lengths, offsets)])
    positions = torch.cat([torch.arange(block, device=args.device) + off + n for n, off in zip(lengths, offsets)])
    x = tensors["embeddings"]
    an = norm(x, "layers.0.input_layernorm")
    for layer in range(config["num_hidden_layers"]):
        p = f"layers.{layer}"
        inputs[p + ".residual"] = x.cpu().contiguous()
        inputs[p + ".normalized_input"] = an.cpu().contiguous()
        ac = linear(an, p + ".attention_conv.kernel_projection")
        ax = conv(an, ac, p + ".attention_conv", 0)
        save(p + ".attention_input", ax)
        q = rope(norm(linear(ax, p + ".self_attn.q_proj").view(-1, nq, hd), p + ".self_attn.q_norm", True), positions)
        k = rope(norm(linear(ax, p + ".self_attn.k_proj").view(-1, nk, hd), p + ".self_attn.k_norm", True), positions)
        v = linear(ax, p + ".self_attn.v_proj").view(-1, nk, hd)
        ck = rope(norm(linear(context, p + ".self_attn.k_proj").view(-1, nk, hd), p + ".self_attn.k_norm", True), ctx_positions)
        cv = linear(context, p + ".self_attn.v_proj").view(-1, nk, hd)
        outputs, start = [], 0
        for b, n in enumerate(lengths):
            ks = torch.cat((ck[start:start+n], k[b*block:(b+1)*block])).repeat_interleave(nq // nk, 1)
            vs = torch.cat((cv[start:start+n], v[b*block:(b+1)*block])).repeat_interleave(nq // nk, 1)
            qs = q[b*block:(b+1)*block]
            scores = torch.einsum("qhd,khd->hqk", qs.float(), ks.float()) * hd ** -0.5
            distance = (torch.arange(block, device=args.device) + n)[:, None] - torch.arange(n + block, device=args.device)[None]
            scores.masked_fill_(distance.abs()[None] >= config["sliding_window"], -torch.inf)
            out = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vs.float()).bfloat16()
            outputs.append(out.reshape(block, nq * hd))
            start += n
        ao = conv(linear(torch.cat(outputs), p + ".self_attn.o_proj"), ac, p + ".attention_conv", 1)
        x = x.float() + ao.float()
        mn = norm(x, p + ".post_attention_layernorm")
        x = x.bfloat16()
        mc = linear(mn, p + ".mlp_conv.kernel_projection")
        mx = conv(mn, mc, p + ".mlp_conv", 0)
        gate = linear(mx, p + ".mlp.gate_proj")
        up = linear(mx, p + ".mlp.up_proj")
        activated = (F.silu(gate.float()) * up.float()).bfloat16()
        mo = conv(linear(activated, p + ".mlp.down_proj"), mc, p + ".mlp_conv", 1)
        x = x.float() + mo.float()
        an = norm(x, f"layers.{layer + 1}.input_layernorm" if layer + 1 < config["num_hidden_layers"] else "norm")
        x = x.bfloat16()
        save(p + ".output", x)
        save(p + ".normalized", an)
    predictions = an.view(batch, block, h)[:, 1:].reshape(-1, h)
    save("predictions", predictions)
    for side in range(2):
        save(f"conv.{side}", conv(tensors["conv.input"], tensors["conv.coefficients"], "layers.0.attention_conv", side))
    ids = tensors["selector.ids"].long().view(batch, block - 1, -1)
    unary = tensors["selector.logits"].view_as(ids)
    gates = linear(tensors["selector.hidden"], "candidate_selector.hidden_projection").view(batch, block - 1, -1)
    previous = torch.cat((tensors["selector.anchors"][:, None, None].expand(-1, 1, ids.shape[-1]), ids[:, :-1]), 1)
    a = weights["candidate_selector.predecessor_codebook"][previous]
    b = weights["candidate_selector.successor_codebook"][ids]
    scores = unary[:, :, None] + torch.einsum("blpr,blcr->blpc", a * gates[:, :, None], b).float()
    save("selector.scores", scores.reshape(-1, ids.shape[-1], ids.shape[-1]))
    prev = torch.zeros(batch, dtype=torch.long, device=args.device)
    tokens = []
    for d in range(block - 1):
        prev = scores[torch.arange(batch, device=args.device), d, prev].argmax(-1)
        tokens.append(ids[torch.arange(batch, device=args.device), d, prev])
    save("selector.tokens", torch.stack(tokens, 1))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    save_file(inputs, str(out / "inputs.safetensors"))
    save_file(expected, str(out / "expected.safetensors"))
    (out / "case.json").write_text(json.dumps({"contextLengths": lengths, "positionOffsets": offsets, "blockSize": block,
                                            "modelHiddenSize": h}))
    print(f"Saved {len(expected)} reference checkpoints to {out}")


if __name__ == "__main__":
    main()
