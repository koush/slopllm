import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { after, before, describe, it } from "node:test";

import { openCheckpointSource } from "../src/checkpoint_source";
import { ExecutionWorkspace } from "../src/execution-workspace";
import { Glm51Model } from "../src/glm51_model";
import { GlmOps, f32ToBf16Bytes } from "../src/glm_ops";
import type { Tensor } from "../src/tensor";
import {
  GLM51_TINY_DIMS,
  buildGlm51ModelFixturePair,
  expectedStoredTensors,
  type ModelFixture,
  type StoredTensorExpectation,
} from "./csf_model_fixture";

// A5 GPU-0 bit-for-bit equivalence (port plan §9.7.2, M1/M2 boundary): the
// same tiny GLM-5.x-shaped model is loaded twice from identical bytes — a
// plain safetensors root and a lil-nvfp4-csf-checkpoint/1 container — through
// the real ChatModel.loadWeights path. The two loads must produce identical
// device bytes and identical A16 forward outputs (same kernels, same weights:
// only the container differs), and the retained F32 input_scale calibration
// scalars must survive in both. GPU parts run on GPU 0 only and skip cleanly
// when the native addon has not been built yet.

const GLM_NODE = path.resolve(__dirname, "../build/Release/glm.node");
const GPU_AVAILABLE = fs.existsSync(GLM_NODE);
const GPU_SKIP = GPU_AVAILABLE
  ? false
  : "build/Release/glm.node is missing (native build not finished); GPU equivalence is env-gated";

const PROMPTS = [[1, 2, 3, 4], [10, 20, 30], [5, 6, 7, 8, 9, 10, 11, 12]];

describe("csf model fixture logical equivalence (cpu)", () => {
  let scratch: string;
  let fixture: ModelFixture;

  before(() => {
    scratch = fs.mkdtempSync(path.join(os.tmpdir(), "slopllm-csf-model-fixture-"));
    fixture = buildGlm51ModelFixturePair(scratch);
  });

  after(() => {
    fs.rmSync(scratch, { recursive: true, force: true });
  });

  it("serves identical logical bytes from plain safetensors and the csf container", () => {
    const plain = openCheckpointSource(fixture.plainRoot);
    const csf = openCheckpointSource(fixture.csfRoot);
    try {
      assert.equal(plain.kind, "safetensors");
      assert.equal(csf.kind, "csf");
      const names = [...fixture.byName.keys()].sort();
      assert.deepEqual([...plain.names()].sort(), names);
      assert.deepEqual([...csf.names()].sort(), names);
      for (const name of names) {
        const expected = fixture.byName.get(name)!;
        for (const source of [plain, csf]) {
          const meta = source.meta(name);
          assert.equal(meta.dtype, expected.dtype, `${name} dtype via ${source.kind}`);
          assert.deepEqual([...meta.shape], expected.shape, `${name} shape via ${source.kind}`);
        }
        // Both readers must return the fixture's exact logical bytes: the
        // container -> buffer round trip of the compressed scales included.
        assert.deepEqual(plain.tensor(name).read(), expected.data, `${name} plain read`);
        assert.deepEqual(csf.tensor(name).read(), expected.data, `${name} csf round-trip`);
        assert.deepEqual(plain.tensor(name).read(), csf.tensor(name).read(), `${name} cross-source`);
      }
    } finally {
      plain.close();
      csf.close();
    }
  });

  it("carries the nvfp4 expert triples with compressed weight_scale planes", () => {
    const d = GLM51_TINY_DIMS;
    assert.equal(fixture.compressedScales.length, d.experts * 3, "every routed projection scale is compressed");
    for (const scaleName of fixture.compressedScales) {
      assert.ok(scaleName.endsWith(".weight_scale"));
      const weight = fixture.byName.get(scaleName.replace(/\.weight_scale$/, ".weight"))!;
      assert.equal(weight.dtype, "U8");
      const scale = fixture.byName.get(scaleName)!;
      // NVFP4 geometry: [N, K/2] U8 weight scaled by [N, K/16] E4M3.
      assert.deepEqual(scale.shape, [weight.shape[0], weight.shape[1] / 8]);
      assert.equal(fixture.byName.get(scaleName.replace(/\.weight_scale$/, ".weight_scale_2"))!.dtype, "F32");
      assert.equal(fixture.byName.get(scaleName.replace(/\.weight_scale$/, ".input_scale"))!.dtype, "F32");
    }
  });

  it("describes the loadTensor rename and conversion surface", () => {
    const expected = expectedStoredTensors(fixture);
    // Renamed NVFP4 scale buffers and the retained calibration scalars.
    assert.ok(expected.has("model.layers.1.mlp.experts.0.gate_proj.weight_weight_scale"));
    assert.ok(expected.has("model.layers.1.mlp.experts.0.gate_proj.weight_weight_scale_2"));
    assert.ok(expected.has("model.layers.1.mlp.experts.0.gate_proj.input_scale"));
    assert.equal(expected.get("model.layers.1.mlp.experts.0.gate_proj.input_scale")!.dtype, "F32");
    // General F32 tensors convert to BF16; weight_scale_2/input_scale do not.
    assert.equal(expected.get("model.layers.1.mlp.gate.e_score_correction_bias")!.dtype, "BF16");
    // MLA splits land under their absorbed names.
    for (const name of ["q_nope_proj.weight", "q_pe_proj.weight", "k_nope_proj.weight",
      "v_proj.weight", "ckv_proj.weight", "k_pe_proj.weight"]) {
      assert.ok(expected.has(`model.layers.0.self_attn.${name}`), name);
    }
    assert.ok(expected.has("lm_head.weight"), "tied lm_head alias");
    // Nothing keeps a pre-rename scale name in the stored map.
    for (const name of expected.keys()) {
      assert.ok(!name.endsWith(".weight_scale"), `${name} must be renamed on store`);
    }
  });
});

describe("a5 gpu-0 equivalence: plain safetensors vs csf container", {
  skip: GPU_SKIP,
  concurrency: false,
}, () => {
  const d = GLM51_TINY_DIMS;
  let scratch: string;
  let fixture: ModelFixture;
  let expected: Map<string, StoredTensorExpectation>;
  let ops: GlmOps;
  let plainModel: Glm51Model;
  let csfModel: Glm51Model;
  let ws: ExecutionWorkspace;

  const readBytes = (tensor: Tensor): Buffer => {
    const bytes = Buffer.alloc(tensor.bytes);
    tensor.d2h(bytes);
    ops.synchronize();
    return bytes;
  };

  before(async () => {
    scratch = fs.mkdtempSync(path.join(os.tmpdir(), "slopllm-csf-gpu-equiv-"));
    fixture = buildGlm51ModelFixturePair(scratch);
    expected = expectedStoredTensors(fixture);
    const deviceId = parseInt(process.env.SLOPLLM_GPU ?? "0", 10);
    ops = new GlmOps(deviceId);
    plainModel = await Glm51Model.fromPretrained(ops, fixture.plainRoot);
    csfModel = await Glm51Model.fromPretrained(ops, fixture.csfRoot);
    ws = new ExecutionWorkspace(ops, 2, 128);
  });

  after(() => {
    ws.free();
    plainModel.free();
    csfModel.free();
    ops.free();
    fs.rmSync(scratch, { recursive: true, force: true });
  });

  it("stores bit-identical device tensors in both loads, matching the fixture source bytes", () => {
    const plainTensors = plainModel.tensors;
    const csfTensors = csfModel.tensors;
    // Both loads must register exactly the same names (invFreq is the only
    // loader-constructed non-checkpoint tensor).
    const plainNames = [...plainTensors.keys()].sort();
    assert.deepEqual(plainNames, [...csfTensors.keys()].sort(), "stored name sets differ between loads");
    const expectedNames = [...expected.keys()].sort();
    const extras = plainNames.filter(name => !expected.has(name));
    assert.deepEqual(extras, ["invFreq"], `unexpected stored tensors: ${extras.join(", ")}`);
    assert.deepEqual(plainNames.filter(name => expected.has(name)), expectedNames, "stored names diverge from the fixture inventory");

    for (const [name, exp] of expected) {
      const a = plainTensors.get(name)!;
      const b = csfTensors.get(name)!;
      assert.ok(a && b, `${name} missing from a load`);
      for (const [label, tensor] of [["plain", a], ["csf", b]] as const) {
        assert.equal(tensor.type, exp.dtype, `${name} dtype (${label})`);
        assert.deepEqual([...tensor.shape], exp.shape, `${name} shape (${label})`);
        assert.equal(tensor.bytes, exp.bytes.length, `${name} byte size (${label})`);
      }
      const aBytes = readBytes(a);
      const bBytes = readBytes(b);
      assert.deepEqual(aBytes, bBytes, `${name} device bytes differ between plain and csf loads`);
      assert.deepEqual(aBytes, exp.bytes, `${name} device bytes differ from the fixture logical bytes`);
    }
  });

  it("retains .input_scale calibration scalars as F32 with identical bytes in both loads", () => {
    const inputScales = [...expected.keys()].filter(name => name.endsWith(".input_scale"));
    assert.equal(inputScales.length, d.experts * 3, "every routed projection keeps its input_scale");
    for (const name of inputScales) {
      for (const [label, model] of [["plain", plainModel], ["csf", csfModel]] as const) {
        const tensor = model.tensors.get(name)!;
        assert.ok(tensor, `${name} must be retained (${label})`);
        assert.equal(tensor.type, "F32", `${name} must stay F32, not join the general F32->BF16 conversion (${label})`);
        assert.equal(tensor.bytes, 4);
      }
      assert.deepEqual(readBytes(plainModel.tensors.get(name)!), readBytes(csfModel.tensors.get(name)!),
        `${name} differs between loads`);
      assert.deepEqual(readBytes(plainModel.tensors.get(name)!), expected.get(name)!.bytes,
        `${name} differs from the fixture scalar`);
    }
    // The retention is specific: ordinary F32 tensors were converted to BF16.
    assert.equal(plainModel.tensors.get("model.layers.1.mlp.gate.e_score_correction_bias")!.type, "BF16");
  });

  it("produces bit-identical per-layer linear() projections on fixed seeded inputs", () => {
    const probes: [string, number][] = [
      ["model.layers.1.mlp.experts.0.gate_proj.weight", 3],
      ["model.layers.1.mlp.experts.3.down_proj.weight", 11],
      ["model.layers.1.mlp.experts.7.up_proj.weight", 29],
      ["model.layers.1.mlp.shared_experts.gate_proj.weight", 4],
      ["model.layers.0.mlp.down_proj.weight", 5],
      ["model.layers.0.self_attn.o_proj.weight", 7],
      ["model.layers.0.self_attn.k_nope_proj.weight", 13],
    ];
    for (const [weightName, seed] of probes) {
      const weightA = plainModel.tensors.get(weightName)!;
      const weightB = csfModel.tensors.get(weightName)!;
      assert.ok(weightA && weightB, weightName);
      const k = weightA.type === "U8" ? weightA.shape[1] * 2 : weightA.shape[1];
      const values = new Float32Array(2 * k);
      for (let i = 0; i < values.length; i++) values[i] = ((i * 31 + seed * 7) % 255 - 128) / 128;
      const input = f32ToBf16Bytes(values);
      const run = (model: Glm51Model, weight: Tensor): Buffer => {
        // Unnamed: disposed named tensors are refused by Tensor.canDispose.
        const probe = ws.alloc([2, k], "BF16");
        try {
          probe.h2d(input);
          const out = probe.linear(weight);
          try {
            return readBytes(out);
          } finally {
            out[Symbol.dispose]();
          }
        } finally {
          probe[Symbol.dispose]();
        }
      };
      const a = run(plainModel, weightA);
      const b = run(csfModel, weightB);
      assert.equal(a.length, 2 * weightA.shape[0] * 2, `${weightName} output size`);
      assert.deepEqual(a, b, `${weightName} projection output differs between loads`);
    }
  });

  it("produces bit-identical a16 forward logits on identical inputs", () => {
    const forwardLogits = (model: Glm51Model, prompt: number[]): Buffer => {
      using cache = model.createChatCache(32);
      cache.reset(1);
      const state = ws.planPrefill(model, 1, [prompt.length], cache);
      state.setInput([prompt]);
      using hidden = model.forward(state);
      using logits = state.computeLogits(hidden, model);
      return readBytes(logits);
    };
    for (const prompt of PROMPTS) {
      const a = forwardLogits(plainModel, prompt);
      const b = forwardLogits(csfModel, prompt);
      assert.equal(a.length, d.vocab * 2, `logits size for prompt [${prompt}]`);
      assert.deepEqual(a, b, `a16 forward logits differ for prompt [${prompt}]`);
    }
  });
});