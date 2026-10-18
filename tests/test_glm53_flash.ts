import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { it } from "node:test";
import { Glm53FlashModel, loadGlm53FlashConfig, GLM53_FLASH_REPO } from "../src/glm53_flash_model";
import { GlmOps, f32ToBf16Bytes } from "../src/glm_ops";
import { ParallelOps, ParallelTensor } from "../src/parallel_ops";
import { TensorParallelism } from "../src/device_ops";
import { KdaState, kdaAttention } from "../src/kda";
import { parseModelArgs, resolveModelSelection } from "../src/model_cli";
import { validateWorkerModelArgs } from "../src/run_model_loader";
import { WorkspaceBase } from "../src/workspace";
import type { Tensor } from "../src/tensor";
import type { ExecutionState } from "../src/execution-workspace";

const prefix = "model.language_model.layers.0.self_attn.";
const hidden = 64, heads = 4, width = heads * 128;

function fixture(): { dir: string; [Symbol.dispose](): void } {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "glm53-flash-"));
  const text = {
    hidden_size: hidden, intermediate_size: 128, num_hidden_layers: 2, rms_norm_eps: 1e-5,
    vocab_size: 32, num_attention_heads: 4, num_key_value_heads: 4, qk_nope_head_dim: 256,
    qk_rope_head_dim: 0, kv_lora_rank: 512, q_lora_rank: 64, v_head_dim: 256,
    layer_types: ["linear_attention", "deepseek_sparse_attention"], index_topk: 2048,
    linear_attn_config: { num_heads: heads, head_dim: 128, short_conv_kernel_size: 4, gate_lower_bound: -5 },
    hc_mult: 4, hc_eps: 1e-6, hc_sinkhorn_iters: 20, n_routed_experts: 288, n_shared_experts: 1,
    num_experts_per_tok: 8, moe_intermediate_size: 128, first_k_dense_replace: 1,
    routed_scaling_factor: 2.5, norm_topk_prob: true, n_group: 1, topk_group: 1,
    swiglu_limit: 10, index_head_dim: 128, index_n_heads: 2,
    index_kpool: 4, index_kpool_always_select_tail: true, eos_token_id: [1], max_position_embeddings: 1048576,
  };
  fs.writeFileSync(path.join(dir, "config.json"), JSON.stringify({ model_type: "glm5_next", text_config: text }));
  fs.writeFileSync(path.join(dir, "tokenizer_config.json"), JSON.stringify({ tokenizer_class: "PreTrainedTokenizer", unk_token: "[UNK]" }));
  fs.writeFileSync(path.join(dir, "tokenizer.json"), JSON.stringify({
    version: "1.0", truncation: null, padding: null, added_tokens: [], normalizer: null,
    pre_tokenizer: { type: "Whitespace" }, post_processor: null, decoder: null,
    model: { type: "WordLevel", vocab: { "[UNK]": 0, "[EOS]": 1, hello: 2 }, unk_token: "[UNK]" },
  }));
  const shards: Record<string, { dtype: string; shape: number[]; data: Buffer }>[] = [{}, {}];
  let serial = 0;
  const add = (name: string, shape: number[], dtype = "BF16", constant?: number) => {
    const count = shape.reduce((a, b) => a * b, 1);
    const values = Float32Array.from({ length: count }, (_, i) => constant ?? Math.sin(i * .131 + serial) * .1);
    const data = dtype === "BF16" ? f32ToBf16Bytes(values)
      : dtype === "F32" ? Buffer.from(values.buffer) : Buffer.alloc(count, 0x38);
    shards[serial++ % 2][name] = { dtype, shape, data };
  };
  for (const name of ["q", "k", "v"]) {
    add(`${prefix}${name}_proj.weight`, [width, hidden]);
    add(`${prefix}${name}_conv1d.weight`, [width, 1, 4]);
  }
  for (const name of ["f", "g"]) {
    add(`${prefix}${name}_a_proj.weight`, [128, hidden]);
    add(`${prefix}${name}_b_proj.weight`, [width, 128]);
  }
  add(`${prefix}b_proj.weight`, [heads, hidden]);
  add(`${prefix}o_proj.weight`, [hidden, width]);
  add(`${prefix}o_norm.weight`, [128], "BF16", 1);
  add(`${prefix}A_log`, [heads], "F32");
  add(`${prefix}dt_bias`, [width], "F32");
  add("model.language_model.layers.0.hc_attn_fn", [24, hidden * 4]);
  add("model.language_model.layers.0.hc_attn_base", [24], "F32");
  add("model.language_model.layers.0.hc_attn_scale", [3], "F32", .1);
  add("model.language_model.layers.1.mlp.gate.weight", [288, hidden], "F32");
  const expert = "model.language_model.layers.1.mlp.experts.287.gate_proj";
  add(`${expert}.weight`, [128, hidden / 2], "U8");
  add(`${expert}.weight_scale`, [128, hidden / 16], "F8_E4M3");
  add(`${expert}.weight_scale_2`, [], "F32", .1234567);
  add(`${expert}.input_scale`, [], "F32", .9876543);
  add("model.language_model.embed_tokens.weight", [32, hidden]);
  add("lm_head.weight", [32, hidden]);
  // Deliberately unsupported dtypes are safe to skip for excluded components.
  add("model.visual.weight", [16, 16], "U8");
  add("model.language_model.layers.2.mlp.experts.0.gate_proj.weight", [16, 16], "F8_E4M3");
  fs.mkdirSync(path.join(dir, "tensors"));
  const weightMap: Record<string, string> = {};
  shards.forEach((shard, i) => {
    const filename = `tensors/model-${i}.safetensors`;
    const header: Record<string, unknown> = {};
    let offset = 0;
    for (const [name, t] of Object.entries(shard)) {
      header[name] = { dtype: t.dtype, shape: t.shape, data_offsets: [offset, offset + t.data.length] };
      offset += t.data.length;
      weightMap[name] = filename;
    }
    const json = Buffer.from(JSON.stringify(header));
    const headerSize = Math.ceil(json.length / 8) * 8;
    const padded = Buffer.alloc(headerSize, 0x20);
    json.copy(padded);
    const size = Buffer.alloc(8);
    size.writeBigUInt64LE(BigInt(headerSize));
    fs.writeFileSync(path.join(dir, filename), Buffer.concat([size, padded, ...Object.values(shard).map(t => t.data)]));
  });
  fs.writeFileSync(path.join(dir, "model.safetensors.index.json"), JSON.stringify({ weight_map: weightMap }));
  return { dir, [Symbol.dispose]: () => fs.rmSync(dir, { recursive: true, force: true }) };
}

function values(t: Tensor): number[] {
  using full = t.replicate();
  const buf = Buffer.alloc(full.bytes);
  full.d2h(buf);
  if (full.type === "F32") return Array.from(new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4));
  const word = Buffer.alloc(4);
  return Array.from({ length: buf.length / 2 }, (_, i) => {
    word.writeUInt32LE(buf.readUInt16LE(i * 2) * 65536);
    return word.readFloatLE();
  });
}

function close(a: number[], b: number[], atol: number, rtol = 0): void {
  assert.equal(a.length, b.length);
  a.forEach((x, i) => assert.ok(Number.isFinite(x) && Math.abs(x - b[i]) <= atol + rtol * Math.abs(b[i]), `${i}: ${x} != ${b[i]}`));
}

it("Flash CLI selects the text checkpoint and rejects unsupported modes/loader overrides", () => {
  using f = fixture();
  const argv = ["--model-dir", f.dir, "--arena", "1"];
  const args = parseModelArgs(argv);
  assert.equal(resolveModelSelection(args).modelType, "glm5_next");
  validateWorkerModelArgs(argv, args);
  assert.throws(() => validateWorkerModelArgs(["--model", GLM53_FLASH_REPO, "--arena", "1"], args), /cannot override/);
  for (const flag of ["--mtp", "--cp"]) {
    assert.throws(() => resolveModelSelection(parseModelArgs([...argv, flag])), /does not support/);
  }
});

it("Flash loader preserves precision/scales, skips vision/MTP, and shards KDA heads", async () => {
  using f = fixture();
  const cfg = loadGlm53FlashConfig(f.dir);
  assert.equal(cfg.linearNumHeads, heads);
  assert.equal(cfg.nRoutedExperts, 288);
  using gpu0 = new GlmOps(0);
  using gpu1 = new GlmOps(1);
  using po = new ParallelOps([gpu0, gpu1]);
  using model = await Glm53FlashModel.fromPretrained(po, f.dir);
  assert.equal(model.eosIds.has(1), true);
  assert.equal(model.tokenizer.encode("hello")[0], 2);
  assert.throws(() => model.forwardModel({} as ExecutionState), /forward is not implemented/);
  assert.throws(() => model.forwardPhased({} as ExecutionState).next(), /forward is not implemented/);
  assert.throws(() => model.createChatCache(), /cache is not implemented/);
  assert.ok(![...model.tensors.keys()].some(n => n.startsWith("model.visual.") || n.includes(".layers.2.")));
  const weights = model.kdaWeights(0);
  assert.equal(weights.aLog.type, "F32");
  assert.deepEqual((weights.q as ParallelTensor).shards[0].shape, [width / 2, hidden]);
  assert.deepEqual((weights.dtBias as ParallelTensor).shards[1].shape, [width / 2]);
  assert.deepEqual((weights.qConv as ParallelTensor).shards[0].shape, [width / 2, 1, 4]);
  const expert = "model.language_model.layers.1.mlp.experts.287.gate_proj";
  assert.equal(model.tensors.get(`${expert}.weight_weight_scale`)!.type, "F8_E4M3");
  close(values(model.tensors.get(`${expert}.weight_weight_scale_2`)!), [Math.fround(.1234567)], 0);
  close(values(model.tensors.get(`${expert}.input_scale`)!), [Math.fround(.9876543)], 0);
  assert.equal(model.tensors.get("model.language_model.layers.1.mlp.gate.weight")!.type, "F32");
  po.synchronize();
});

it("complete KDA agrees across TP and resumed chunks; mHC accepts TP partial outputs", async () => {
  using f = fixture();
  using gpu0 = new GlmOps(0);
  using gpu1 = new GlmOps(1);
  using po = new ParallelOps([gpu0, gpu1]);
  using model = await Glm53FlashModel.fromPretrained(po, f.dir);
  using single = await Glm53FlashModel.fromPretrained(gpu0, f.dir);
  using ws = new WorkspaceBase(po);
  using sws = new WorkspaceBase(gpu0);
  using state = new KdaState(po, 2, heads);
  using sstate = new KdaState(gpu0, 2, heads);
  const input = Float32Array.from({ length: 5 * hidden }, (_, i) => Math.sin(i * .19));
  const execute = (workspace: WorkspaceBase, m: Glm53FlashModel, s: KdaState, indices: number[], cu: number[]) => {
    using x = workspace.alloc([indices.length, hidden], "BF16");
    x.h2d(f32ToBf16Bytes(Float32Array.from(indices.flatMap(i => Array.from(input.slice(i * hidden, (i + 1) * hidden))))));
    using boundaries = workspace.alloc([3], "I32");
    boundaries.h2d(Buffer.from(Int32Array.from(cu).buffer));
    return kdaAttention(x, m.kdaWeights(0), s, boundaries);
  };
  using out = execute(ws, model, state, [0, 1, 2, 3, 4], [0, 3, 5]);
  using ref = execute(sws, single, sstate, [0, 1, 2, 3, 4], [0, 3, 5]);
  const expected = values(ref);
  close(values(out), expected, .02, .02);
  close(values(state.recurrent), values(sstate.recurrent), 2e-5, .01);
  close(values(state.qConv), values(sstate.qConv), 0);
  state.reset();
  using first = execute(ws, model, state, [0, 1, 3], [0, 2, 3]);
  using last = execute(ws, model, state, [2, 4], [0, 1, 2]);
  const chunks = new Array(expected.length);
  for (const [tensor, indices] of [[first, [0, 1, 3]], [last, [2, 4]]] as const) {
    const data = values(tensor);
    indices.forEach((idx, row) => data.slice(row * hidden, (row + 1) * hidden).forEach((v, d) => chunks[idx * hidden + d] = v));
  }
  close(chunks, expected, .03, .03);
  close(values(state.recurrent), values(sstate.recurrent), 2e-5, .01);

  using block = execute(ws, model, state, [0, 3], [0, 1, 2]);
  assert.equal((block as ParallelTensor).parallelism, TensorParallelism.PartialSum);
  using seed = ws.alloc([2, hidden], "BF16");
  seed.fill(.5, seed.numElements);
  using streams = po.mhcStreams(seed, true);
  const get = (name: string) => model.tensors.get(`model.language_model.layers.0.hc_attn_${name}`)!;
  const pre = po.mhcPre(streams, get("fn"), get("base"), get("scale"));
  using collapsed = pre.collapsed;
  using post = pre.post;
  using comb = pre.comb;
  using mixed = po.mhcPost(streams, block, post, comb);
  using mean = po.mhcStreams(mixed, false);
  assert.ok(values(mean).every(Number.isFinite));
  assert.equal((block as ParallelTensor).parallelism, TensorParallelism.Replicated);
  assert.throws(() => po.mhcPre(seed, get("fn"), get("base"), get("scale")), /expected/);
  assert.throws(() => gpu0.mhcStreams(seed, true), /executing backend/);
  po.synchronize();
});
