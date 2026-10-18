import { it } from "node:test";
import assert from "node:assert/strict";
import { resolveWeightType, type WeightQuantizationConfig } from "../src/weight_type";
import type { TensorMeta } from "../src/safetensors";

const meta = (dtype: string, shape: number[]): TensorMeta => ({ dtype, shape, dataOffsets: [0, 0] });
const name = "model.layers.3.mlp.experts.0.gate_proj.weight";
const nv: WeightQuantizationConfig = {
  quant_method: "modelopt", quant_algo: "NVFP4",
  config_groups: { group_0: { targets: ["Linear"], weights: { num_bits: 4, type: "float", group_size: 16 } } },
};
const mx: WeightQuantizationConfig = { quant_method: "fp8", expert_dtype: "fp4", scale_fmt: "ue8m0" };
const nvMeta = (key: string) => {
  if (key === `${name}_scale`) return meta("F8_E4M3", [128, 8]);
  assert.equal(key, `${name}_scale_2`);
  return meta("F32", [1]);
};
const mxMeta = (key: string) => {
  assert.equal(key, name.replace(/\.weight$/, ".scale"));
  return meta("F8_E8M0", [128, 4]);
};

it("resolves ModelOpt NVFP4 and mixed FP8/MXFP4 experts from config", () => {
  assert.equal(resolveWeightType(name, meta("U8", [128, 64]), nv, nvMeta), "U8_NVFP4");
  assert.equal(resolveWeightType(name, meta("I8", [128, 64]), mx, mxMeta), "U8_MXFP4");
  for (const dtype of ["BF16", "F8_E4M3", "F8_E8M0"]) {
    assert.equal(resolveWeightType(name, meta(dtype, [128, 128]), mx, () => { throw Error("unexpected metadata read"); }), dtype);
  }
  assert.throws(() => resolveWeightType("model.attn.weight", meta("I8", [128, 64]), mx, mxMeta), /ambiguous/);
});

it("honors exclusions, module targets, and rejects unknown/conflicting formats", () => {
  const packed = meta("U8", [128, 64]);
  for (const pattern of ["model.layers.3.mlp.experts.*", "re:.*experts\\.0\\.", name.slice(0, -7)]) {
    assert.throws(() => resolveWeightType(name, packed, { ...nv, ignore: [pattern] }, nvMeta), /excluded/);
  }
  assert.throws(() => resolveWeightType(name, packed, undefined, nvMeta), /missing quantization_config/);
  assert.throws(() => resolveWeightType(name, packed, { quant_algo: "INT4" }, nvMeta), /unsupported quant_algo/);
  const group = nv.config_groups!.group_0;
  assert.throws(() => resolveWeightType(name, packed, { config_groups: { a: { ...group, targets: ["lm_head"] } } }, nvMeta), /no matching/);
  assert.throws(() => resolveWeightType(name, packed, { ...nv, config_groups: {
    a: { ...group, weights: { ...group.weights, group_size: 32 } },
  } }, nvMeta), /conflicts with group size/);
  assert.throws(() => resolveWeightType(name, packed, { config_groups: {
    a: group, b: { ...group, weights: { ...group.weights, group_size: 32 } },
  } }, nvMeta), /conflicting quantization groups/);
  assert.equal(resolveWeightType(name, packed, { config_groups: {
    a: { ...group, targets: ["model.layers.*.mlp.experts.*"] },
    b: { ...group, targets: ["other.*"], weights: { ...group.weights, group_size: 32 } },
  } }, nvMeta), "U8_NVFP4");
});

it("validates scale dtype and geometry against the declared format", () => {
  assert.throws(() => resolveWeightType(name, meta("U8", [128, 64]), nv, () => meta("F8_E8M0", [128, 8])), /invalid NVFP4/);
  assert.throws(() => resolveWeightType(name, meta("I8", [128, 64]), mx, () => meta("F8_E8M0", [128, 8])), /invalid MXFP4/);
  assert.throws(() => resolveWeightType(name, meta("U8", [128, 64]), nv,
    key => key.endsWith("_2") ? meta("F32", [2]) : nvMeta(key)), /global scale/);
});
