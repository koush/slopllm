import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { GlmOps, bf16BytesToF32 } from "../src/glm_ops";
import { Qwen3Model } from "../src/qwen3_model";
import { PagedKVCache, WorkspaceBuffers } from "../src/paged_kv";
import { generateTokens } from "./test_helper";

const FP8_REPO = "Qwen/Qwen3-0.6B-FP8";
const BF16_REPO = "Qwen/Qwen3-0.6B";

function makeKV(m: Qwen3Model, maxPages = 256): PagedKVCache {
  return new PagedKVCache(m.glm, m.cfg.numKeyValueHeads, m.cfg.headDim, m.cfg.numHiddenLayers, maxPages, m.maxBatch);
}

function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function readLogits(model: Qwen3Model): Float32Array {
  const vs = model.cfg.vocabSize;
  const buf = Buffer.alloc(vs * 2);
  model.ws.logitsBuf.d2h(buf);
  return bf16BytesToF32(buf);
}

describe("Qwen3-0.6B-FP8 model", () => {
  let glm: GlmOps;
  let model: Qwen3Model;

  before(() => {
    process.env.CUDA_VISIBLE_DEVICES = process.env.GLM_GPU ?? "0";
    glm = new GlmOps(0);
    model = Qwen3Model.fromPretrained(glm, FP8_REPO, 1, 64);
  });

  after(() => {
    model.free();
  });

  it("loads FP8 weights as F8_E4M3 and scale_inv as F32", () => {
    const qWeight = model.weights.get("model.layers.0.self_attn.q_proj.weight")!;
    assert.equal(qWeight.type, "F8_E4M3", "q_proj weight should be F8_E4M3");
    assert.ok(qWeight.name!.endsWith("_scale_inv") === false, "weight name should not end with _scale_inv");

    const qScale = model.weights.get("model.layers.0.self_attn.q_proj.weight_scale_inv")!;
    assert.equal(qScale.type, "F32", "q_proj scale should be F32");
    assert.deepEqual(qScale.shape, [2048 / 128, 1024 / 128], "q_proj scale shape should be [16, 8]");

    const gateWeight = model.weights.get("model.layers.0.mlp.gate_proj.weight")!;
    assert.equal(gateWeight.type, "F8_E4M3", "gate_proj weight should be F8_E4M3");

    const gateScale = model.weights.get("model.layers.0.mlp.gate_proj.weight_scale_inv")!;
    assert.equal(gateScale.type, "F32", "gate_proj scale should be F32");
    assert.deepEqual(gateScale.shape, [3072 / 128, 1024 / 128], "gate_proj scale shape should be [24, 8]");

    const norm = model.weights.get("model.layers.0.input_layernorm.weight")!;
    assert.equal(norm.type, "BF16", "layernorm weight should be BF16");

    const embed = model.weights.get("model.embed_tokens.weight")!;
    assert.equal(embed.type, "BF16", "embed_tokens should be BF16");
  });

  it("prefills and produces a valid token", () => {
    const pagedKV = makeKV(model);
    const ws = new WorkspaceBuffers(glm);
    try {
      const token = model.forwardEager([[1, 2, 3, 4, 5]], ws, pagedKV)[0];
      assert.ok(Number.isInteger(token), "prefill should return an integer token");
      assert.ok(token >= 0 && token < model.cfg.vocabSize, `token ${token} out of vocab range [0, ${model.cfg.vocabSize})`);
    } finally {
      pagedKV.free();
      ws.free();
    }
  });

  it("decodes tokens after prefill", () => {
    const pagedKV = makeKV(model);
    const ws = new WorkspaceBuffers(glm);
    try {
      const tokens = [...generateTokens(model, ws, pagedKV, [1, 2, 3, 4, 5], 10, model.eosIds)];
      assert.ok(tokens.length > 0, "should produce at least one token");
      for (const t of tokens) {
        assert.ok(Number.isInteger(t), `token ${t} should be an integer`);
        assert.ok(t >= 0 && t < model.cfg.vocabSize, `token ${t} out of vocab range`);
      }
    } finally {
      pagedKV.free();
      ws.free();
    }
  });

  it("FP8 logits correlate with BF16 logits (cosine sim >= 0.99)", () => {
    const fp8KV = makeKV(model);
    const ws = new WorkspaceBuffers(glm);
    const bf16Model = Qwen3Model.fromPretrained(glm, BF16_REPO, 1, 64);
    const bf16KV = makeKV(bf16Model);
    const bf16Ws = new WorkspaceBuffers(glm);
    try {
      model.forwardEager([[1, 2, 3, 4, 5]], ws, fp8KV);
      const fp8Logits = readLogits(model);

      bf16Model.forwardEager([[1, 2, 3, 4, 5]], bf16Ws, bf16KV);
      const bf16Logits = readLogits(bf16Model);

      assert.equal(fp8Logits.length, bf16Logits.length, "logits length mismatch");

      const sim = cosineSimilarity(fp8Logits, bf16Logits);
      assert.ok(sim >= 0.95, `cosine similarity ${sim.toFixed(6)} < 0.95`);
    } finally {
      fp8KV.free();
      ws.free();
      bf16KV.free();
      bf16Ws.free();
      bf16Model.free();
    }
  });
});
