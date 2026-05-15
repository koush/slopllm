#!/usr/bin/env python3
"""Generate GLM51-NVFP4-weights.md from the safetensors files in /mnt/storage/GLM-5.1-NVFP4-Fixed/."""

import glob
import json
import struct
import re
from collections import defaultdict

MODEL_DIR = "/mnt/storage/GLM-5.1-NVFP4-Fixed/"
OUTPUT = "/mnt/storage2/glm.js/GLM51-NVFP4-weights.md"

DTYPE_SIZES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}

DTYPE_READABLE = {
    "BOOL": "bool", "U8": "uint8", "I8": "int8", "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2", "I16": "int16", "U16": "uint16", "F16": "float16",
    "BF16": "bfloat16", "I32": "int32", "U32": "uint32", "F32": "float32",
    "I64": "int64", "U64": "uint64", "F64": "float64",
}


def fmt_bytes(b):
    if b >= 1024 * 1024 * 1024:
        return f"{b / 1024 / 1024 / 1024:.2f} GB"
    elif b >= 1024 * 1024:
        return f"{b / 1024 / 1024:.2f} MB"
    elif b >= 1024:
        return f"{b / 1024:.2f} KB"
    else:
        return f"{b} B"


def compute_nbytes(shape, dtype):
    n = 1
    for d in shape:
        n *= d
    return n * DTYPE_SIZES.get(dtype, 0)


def load_all_weights(model_dir):
    files = sorted(glob.glob(f"{model_dir}/model-*.safetensors"))
    all_weights = {}
    for f in files:
        with open(f, "rb") as fh:
            header_size = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(header_size))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype = info["dtype"]
            shape = info["shape"]
            nbytes = compute_nbytes(shape, dtype)
            all_weights[name] = (shape, dtype, nbytes)
    return all_weights


def classify_layers(all_weights):
    layer_indices = set()
    expert_layer_indices = set()
    mtp_layer_indices = set()
    for name in all_weights:
        m = re.match(r"model\.layers\.(\d+)\.", name)
        if m:
            layer_indices.add(int(m.group(1)))
            if ".experts." in name:
                expert_layer_indices.add(int(m.group(1)))
            if ".eh_proj." in name or ".enorm." in name or ".hnorm." in name:
                mtp_layer_indices.add(int(m.group(1)))

    dense_layers = sorted(layer_indices - expert_layer_indices - mtp_layer_indices)
    moe_layers = sorted(expert_layer_indices - mtp_layer_indices)
    mtp_layers = sorted(mtp_layer_indices)

    moe_nvfp4 = []
    moe_bf16 = []
    for l in moe_layers:
        key = f"model.layers.{l}.mlp.experts.0.gate_proj.weight"
        if key in all_weights:
            _, dtype, _ = all_weights[key]
            (moe_nvfp4 if dtype == "U8" else moe_bf16).append(l)

    experts_per_layer = defaultdict(int)
    for name in all_weights:
        m = re.match(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.weight", name)
        if m:
            experts_per_layer[int(m.group(1))] = max(
                experts_per_layer[int(m.group(1))], int(m.group(2)) + 1
            )

    return dense_layers, moe_nvfp4, moe_bf16, mtp_layers, experts_per_layer


def normalize(name, mtp_layer_indices=None):
    if mtp_layer_indices is None:
        mtp_layer_indices = set()
    m = re.match(r"model\.layers\.(\d+)\.", name)
    if m and int(m.group(1)) in mtp_layer_indices:
        n = re.sub(r"model\.layers\.\d+\.", "model.layers.MTP.", name)
    else:
        n = re.sub(r"model\.layers\.\d+\.", "model.layers.N.", name)
    n = re.sub(r"\.experts\.\d+\.", ".experts.E.", n)
    return n


def build_pattern_info(all_weights, mtp_layer_indices):
    info = defaultdict(lambda: {"shapes_dtypes": {}, "count": 0, "total_bytes": 0, "mtp_only": False})
    for name, (shape, dtype, nbytes) in sorted(all_weights.items()):
        norm = normalize(name, mtp_layer_indices)
        key = (tuple(shape), dtype)
        if key not in info[norm]["shapes_dtypes"]:
            info[norm]["shapes_dtypes"][key] = 0
        info[norm]["shapes_dtypes"][key] += 1
        info[norm]["count"] += 1
        info[norm]["total_bytes"] += nbytes
        if norm.startswith("model.layers.MTP."):
            info[norm]["mtp_only"] = True
    return info


def render_pattern_row(pattern, info):
    shapes_str = ", ".join(str(list(s)) for s, d in sorted(info["shapes_dtypes"].keys()))
    dtypes = sorted(set(d for s, d in info["shapes_dtypes"].keys()))
    dtype_str = ", ".join(DTYPE_READABLE.get(d, d) for d in dtypes)
    mtp_tag = " **MTP**" if info["mtp_only"] else ""
    return f"| `{pattern}`{mtp_tag} | {shapes_str} | {dtype_str} | {info['count']:,} | {fmt_bytes(info['total_bytes'])} |"


def main():
    all_weights = load_all_weights(MODEL_DIR)
    total_bytes = sum(nbytes for _, _, nbytes in all_weights.values())

    dense_layers, moe_nvfp4, moe_bf16, mtp_layers, experts_per_layer = classify_layers(all_weights)
    pattern_info = build_pattern_info(all_weights, set(mtp_layers))

    num_experts = max(experts_per_layer.values()) if experts_per_layer else 0

    lines = []
    lines.append("# GLM-5.1 NVFP4 Weight Reference")
    lines.append("")
    lines.append(f"Source: `{MODEL_DIR}` (80 safetensors shards)")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- **Total weights**: {len(all_weights):,}")
    lines.append(f"- **Total size**: {total_bytes:,} bytes ({total_bytes / 1024 / 1024 / 1024:.2f} GB)")
    all_layer_indices = set()
    for n in all_weights:
        m = re.match(r"model\.layers\.(\d+)\.", n)
        if m:
            all_layer_indices.add(int(m.group(1)))
    lines.append(f"- **Layers**: {len(all_layer_indices)} total")
    lines.append(f"  - Dense layers (0-2): {dense_layers}")
    lines.append(f"  - MoE NVFP4 layers (3-77): layers {moe_nvfp4[0]}-{moe_nvfp4[-1]} ({len(moe_nvfp4)} layers)")
    if moe_bf16:
        lines.append(f"  - MoE BF16 layer ({moe_bf16[0]}): experts stored in BF16 (not quantized)")
    lines.append(f"  - MTP layer ({mtp_layers[0]}): Multi-Token Prediction draft layer (MoE, BF16)")
    lines.append(f"- **Routed experts per MoE layer**: {num_experts}")
    lines.append("")
    lines.append("### MTP Layer Structure")
    lines.append("")
    lines.append(f"Layer {mtp_layers[0]} is the MTP (Multi-Token Prediction) layer used for speculative decoding.")
    lines.append("It contains a full MoE transformer decoder plus MTP-specific projection weights:")
    lines.append("- `eh_proj`: Linear projection combining embedded token + target model hidden state")
    lines.append("- `enorm`: RMSNorm on embedded token before projection")
    lines.append("- `hnorm`: RMSNorm on target model hidden state before projection")
    lines.append("- `shared_head.norm`: RMSNorm before shared LM head")
    lines.append("- Full decoder: self_attn (MLA + indexer), MoE MLP (256 routed + shared experts)")
    lines.append("")

    # All weight patterns (single table)
    lines.append("## Weight Name Patterns")
    lines.append("")
    lines.append("Weights grouped by structural pattern. `N` = layer index, `E` = expert index.")
    lines.append("`MTP` = MTP layer index (layer 78). Patterns marked **MTP** exist only in the MTP layer.")
    lines.append("Some patterns have multiple shapes/dtypes across layers (e.g., NVFP4 vs BF16 for layer 78).")
    lines.append("")
    lines.append("| Pattern | Shape | Dtype | Count | Total Size |")
    lines.append("|---|---|---|---|---|")
    for pattern in sorted(pattern_info.keys()):
        lines.append(render_pattern_row(pattern, pattern_info[pattern]))
    lines.append("")

    # Size breakdown
    lines.append("## Size Breakdown by Component")
    lines.append("")
    component_totals = defaultdict(int)
    mtp_layer_set = set(mtp_layers)
    for name, (shape, dtype, nbytes) in all_weights.items():
        if name.startswith("model.layers."):
            m = re.match(r"model\.layers\.(\d+)\.(self_attn|mlp|input_layernorm|post_attention_layernorm|eh_proj|enorm|hnorm)", name)
            layer_idx = int(re.match(r"model\.layers\.(\d+)\.", name).group(1))
            is_mtp = layer_idx in mtp_layer_set
            if m:
                comp = m.group(2)
                prefix = "MTP " if is_mtp else ""
                if comp == "self_attn":
                    component_totals[f"{prefix}Attention (self_attn)"] += nbytes
                elif comp == "mlp":
                    if ".experts." in name:
                        component_totals[f"{prefix}MoE Routed Experts"] += nbytes
                    elif ".shared_experts." in name:
                        component_totals[f"{prefix}MoE Shared Experts"] += nbytes
                    elif ".gate." in name:
                        component_totals[f"{prefix}MoE Router (gate)"] += nbytes
                    else:
                        component_totals[f"{prefix}Dense MLP"] += nbytes
                elif "layernorm" in comp:
                    component_totals[f"{prefix}LayerNorms"] += nbytes
                else:
                    component_totals[f"{prefix}Other (eh_proj/enorm/hnorm)"] += nbytes
            elif is_mtp and "shared_head" in name:
                component_totals["MTP Shared Head Norm"] += nbytes
        elif name == "model.embed_tokens.weight":
            component_totals["Embedding"] += nbytes
        elif name == "model.norm.weight":
            component_totals["Final Norm"] += nbytes
        elif name == "lm_head.weight":
            component_totals["LM Head"] += nbytes

    lines.append("| Component | Size | % of Total |")
    lines.append("|---|---|---|")
    for comp in sorted(component_totals.keys(), key=lambda x: component_totals[x], reverse=True):
        b = component_totals[comp]
        pct = 100.0 * b / total_bytes
        lines.append(f"| {comp} | {fmt_bytes(b)} | {pct:.1f}% |")
    lines.append("")

    with open(OUTPUT, "w") as f:
        f.write("\n".join(lines))

    print(f"Written {len(lines)} lines to {OUTPUT}")
    print(f"Total weights: {len(all_weights):,}")
    print(f"Total size: {total_bytes / 1024 / 1024 / 1024:.2f} GB")


if __name__ == "__main__":
    main()
