import fs from "node:fs";
import path from "node:path";

import { f32ToBf16Bytes } from "../src/glm_ops";
import {
  type FixtureTensor,
  type ScaleMatrixFixture,
  makeScaleMatrix,
  sha256,
  writeSafetensorsFile,
} from "./csf_fixture";

// GPU-0 A5 equivalence fixture (port plan §9.7.2): one tiny GLM-5.x-shaped
// model written twice from identical logical bytes — once as a plain
// safetensors checkpoint directory, once as a lil-nvfp4-csf-checkpoint/1
// container holding the same logical tensors with every routed-expert
// weight_scale matrix carried as compressed fixed/exception planes. The
// inventory mirrors tests/python/test_models/glm51_small/generate_glm51_small.py
// in miniature: MLA attention, the sparse indexer, one dense MLP layer, one
// sparse MoE layer (BF16 shared expert, eight NVFP4 routed experts), and the
// loadTensor renaming surface (.weight_scale -> .weight_weight_scale,
// F32 calibration scalars, MLA weight splits). Both roots carry a
// self-contained tiny WordLevel tokenizer so Glm51Model.fromPretrained never
// escapes the fixture directories.

export interface ModelFixture {
  plainRoot: string;
  csfRoot: string;
  /** Logical source-order tensors; identical bytes in both roots. */
  tensors: FixtureTensor[];
  byName: Map<string, FixtureTensor>;
  /** Logical names replaced by compressed planes inside the container. */
  compressedScales: string[];
  configFiles: Record<string, string>;
}

export interface StoredTensorExpectation {
  /** Source logical name(s) consumed to build this stored tensor. */
  sourceNames: string[];
  dtype: string;
  shape: number[];
  bytes: Buffer;
}

export const GLM51_TINY_DIMS = {
  hidden: 1024,
  inter: 2048,
  moeInter: 512,
  heads: 8,
  qLora: 256,
  kvLora: 512,
  qkNope: 96,
  qkRope: 64,
  vHead: 128,
  // The indexer FP8 MMA path requires exactly 32 heads x 128 head dim
  // (production GLM-5.3 geometry); other shapes hit model/ops fallback gaps.
  idxHeads: 32,
  idxHead: 128,
  vocab: 256,
  experts: 8,
  layers: 2,
} as const;

export const GLM51_TINY_CONFIG: Record<string, unknown> = {
  architectures: ["GlmMoeDsaForCausalLM"],
  model_type: "glm_moe_dsa",
  hidden_act: "silu",
  dtype: "bfloat16",
  hidden_size: GLM51_TINY_DIMS.hidden,
  intermediate_size: GLM51_TINY_DIMS.inter,
  moe_intermediate_size: GLM51_TINY_DIMS.moeInter,
  num_hidden_layers: GLM51_TINY_DIMS.layers,
  num_attention_heads: GLM51_TINY_DIMS.heads,
  num_key_value_heads: GLM51_TINY_DIMS.heads,
  head_dim: 64,
  n_shared_experts: 1,
  n_routed_experts: GLM51_TINY_DIMS.experts,
  routed_scaling_factor: 2.5,
  kv_lora_rank: GLM51_TINY_DIMS.kvLora,
  q_lora_rank: GLM51_TINY_DIMS.qLora,
  qk_head_dim: GLM51_TINY_DIMS.qkNope + GLM51_TINY_DIMS.qkRope,
  qk_rope_head_dim: GLM51_TINY_DIMS.qkRope,
  qk_nope_head_dim: GLM51_TINY_DIMS.qkNope,
  v_head_dim: GLM51_TINY_DIMS.vHead,
  n_group: 1,
  topk_group: 1,
  num_experts_per_tok: 4,
  norm_topk_prob: true,
  scoring_func: "sigmoid",
  topk_method: "noaux_tc",
  max_position_embeddings: 4096,
  initializer_range: 0.02,
  rms_norm_eps: 1e-5,
  use_cache: true,
  attention_bias: false,
  attention_dropout: 0.0,
  vocab_size: GLM51_TINY_DIMS.vocab,
  tie_word_embeddings: true,
  index_topk: 2048,
  index_head_dim: GLM51_TINY_DIMS.idxHead,
  index_n_heads: GLM51_TINY_DIMS.idxHeads,
  rope_interleave: true,
  indexer_rope_interleave: true,
  ep_size: 1,
  pretraining_tp: 1,
  moe_layer_freq: 1,
  num_nextn_predict_layers: 0,
  rope_parameters: { rope_theta: 1000000, rope_type: "default" },
  eos_token_id: [3],
  pad_token_id: 0,
  mlp_layer_types: ["dense", "sparse"],
};

/** Deterministic 24-bit LCG byte stream (numerical reproducibility, not crypto). */
function lcgBytes(seed: number, length: number): Buffer {
  const out = Buffer.alloc(length);
  let state = (seed * 2654435761 + 1013904223) >>> 0;
  for (let i = 0; i < length; i++) {
    state = (state * 1664525 + 1013904223) >>> 0;
    out[i] = (state >>> 16) & 0xff;
  }
  return out;
}

/** Deterministic BF16 bytes of exact multiples of 1/128 in [-1, 1). */
function bf16Bytes(seed: number, elements: number): Buffer {
  const raw = lcgBytes(seed, elements);
  const values = new Float32Array(elements);
  for (let i = 0; i < elements; i++) values[i] = (raw[i] - 128) / 128;
  return f32ToBf16Bytes(values);
}

function f32Scalar(value: number): Buffer {
  const buf = Buffer.alloc(4);
  buf.writeFloatLE(value);
  return buf;
}

/** Sorted unique in-range exception positions with valid E4M3 code values. */
function exceptionSet(seed: number, count: number, positions: number): { exceptionPositions: number[]; exceptionValues: number[] } {
  const picks = new Set<number>();
  for (let i = 0; i < count; i++) picks.add((seed * 131 + i * 977) % positions);
  const sorted = [...picks].sort((a, b) => a - b);
  return {
    exceptionPositions: sorted,
    // Stay within finite nonnegative E4M3 codes (<= 0x7e) as the codec requires.
    exceptionValues: sorted.map((p, i) => 0x30 + ((p + i * 7) % 0x40)),
  };
}

/**
 * Logical tensor inventory of the tiny GLM-5.x model. Routed-expert
 * projections are NVFP4 triples (U8 packed weight + F8_E4M3 weight_scale +
 * F32 weight_scale_2 + F32 input_scale); attention, indexer, dense MLP and the
 * shared expert stay BF16; the router bias is F32 like the real checkpoint.
 */
function buildLogicalTensors(): { tensors: FixtureTensor[]; compressed: Map<string, ScaleMatrixFixture> } {
  const d = GLM51_TINY_DIMS;
  const tensors: FixtureTensor[] = [];
  const compressed = new Map<string, ScaleMatrixFixture>();
  let seed = 1;

  const bf16 = (name: string, shape: number[]): void => {
    tensors.push({
      name,
      dtype: "BF16",
      shape,
      data: bf16Bytes(seed++, shape.reduce((a, b) => a * b, 1)),
    });
  };

  bf16("model.embed_tokens.weight", [d.vocab, d.hidden]);
  bf16("model.norm.weight", [d.hidden]);

  for (let layer = 0; layer < d.layers; layer++) {
    const pfx = `model.layers.${layer}`;
    bf16(`${pfx}.input_layernorm.weight`, [d.hidden]);
    bf16(`${pfx}.post_attention_layernorm.weight`, [d.hidden]);

    const attn = `${pfx}.self_attn`;
    bf16(`${attn}.q_a_proj.weight`, [d.qLora, d.hidden]);
    bf16(`${attn}.q_a_layernorm.weight`, [d.qLora]);
    bf16(`${attn}.q_b_proj.weight`, [d.heads * (d.qkNope + d.qkRope), d.qLora]);
    bf16(`${attn}.kv_a_proj_with_mqa.weight`, [d.kvLora + d.qkRope, d.hidden]);
    bf16(`${attn}.kv_a_layernorm.weight`, [d.kvLora]);
    bf16(`${attn}.kv_b_proj.weight`, [d.heads * (d.qkNope + d.vHead), d.kvLora]);
    bf16(`${attn}.o_proj.weight`, [d.hidden, d.heads * d.vHead]);

    const idx = `${attn}.indexer`;
    bf16(`${idx}.wq_b.weight`, [d.idxHeads * d.idxHead, d.qLora]);
    bf16(`${idx}.wk.weight`, [d.idxHead, d.hidden]);
    bf16(`${idx}.k_norm.weight`, [d.idxHead]);
    bf16(`${idx}.k_norm.bias`, [d.idxHead]);
    bf16(`${idx}.weights_proj.weight`, [d.idxHeads, d.hidden]);

    const mlp = `${pfx}.mlp`;
    if (layer === 0) {
      // Dense MLP layer stays BF16 (like the real first_k_dense_replace layers).
      bf16(`${mlp}.gate_proj.weight`, [d.inter, d.hidden]);
      bf16(`${mlp}.up_proj.weight`, [d.inter, d.hidden]);
      bf16(`${mlp}.down_proj.weight`, [d.hidden, d.inter]);
      continue;
    }

    // Sparse MoE layer: router + BF16 shared expert + NVFP4 routed experts.
    bf16(`${mlp}.gate.weight`, [d.experts, d.hidden]);
    tensors.push({
      name: `${mlp}.gate.e_score_correction_bias`,
      dtype: "F32",
      shape: [d.experts],
      data: Buffer.concat(Array.from({ length: d.experts }, (_, i) => {
        const buf = Buffer.alloc(4);
        buf.writeFloatLE(((i * 37 + layer * 11) % 200 - 100) / 64);
        return buf;
      })),
    });
    bf16(`${mlp}.shared_experts.gate_proj.weight`, [d.moeInter, d.hidden]);
    bf16(`${mlp}.shared_experts.up_proj.weight`, [d.moeInter, d.hidden]);
    bf16(`${mlp}.shared_experts.down_proj.weight`, [d.hidden, d.moeInter]);

    for (let e = 0; e < d.experts; e++) {
      const epfx = `${mlp}.experts.${e}`;
      const projection = (proj: string, rows: number, packedK: number, columns: number, projSeed: number) => {
        tensors.push({
          name: `${epfx}.${proj}.weight`,
          dtype: "U8",
          shape: [rows, packedK],
          data: lcgBytes(projSeed, rows * packedK),
        });
        const scaleName = `${epfx}.${proj}.weight_scale`;
        const { exceptionPositions, exceptionValues } = exceptionSet(projSeed, 4, rows * columns);
        compressed.set(scaleName, makeScaleMatrix(rows, columns, { seed: projSeed, exceptionPositions, exceptionValues }));
        const matrix = compressed.get(scaleName)!;
        tensors.push(
          { name: scaleName, dtype: "F8_E4M3", shape: [rows, columns], data: matrix.logical },
          { name: `${epfx}.${proj}.weight_scale_2`, dtype: "F32", shape: [], data: f32Scalar(0.5 + (projSeed % 100) / 1000) },
          { name: `${epfx}.${proj}.input_scale`, dtype: "F32", shape: [], data: f32Scalar(0.03125 * (1 + (projSeed % 8))) },
        );
      };
      const base = 1000 + e * 16;
      projection("gate_proj", d.moeInter, d.hidden / 2, d.hidden / 16, base + 1);
      projection("up_proj", d.moeInter, d.hidden / 2, d.hidden / 16, base + 5);
      projection("down_proj", d.hidden, d.moeInter / 2, d.moeInter / 16, base + 9);
    }
  }
  return { tensors, compressed };
}

function tokenizerFiles(): Record<string, string> {
  const vocab: Record<string, number> = { "[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3 };
  for (let i = 0; i < GLM51_TINY_DIMS.vocab - 4; i++) vocab[`t${i}`] = 4 + i;
  return {
    "tokenizer_config.json": JSON.stringify({
      tokenizer_class: "PreTrainedTokenizerFast",
      model_max_length: 4096,
      bos_token: "[BOS]",
      eos_token: "[EOS]",
      pad_token: "[PAD]",
      unk_token: "[UNK]",
    }),
    "tokenizer.json": JSON.stringify({
      version: "1.0",
      truncation: null,
      padding: null,
      added_tokens: ["[PAD]", "[UNK]", "[BOS]", "[EOS]"].map((content, id) => ({
        id, content, single_word: false, lstrip: false, rstrip: false, normalized: false, special: true,
      })),
      normalizer: null,
      pre_tokenizer: { type: "Whitespace" },
      post_processor: null,
      decoder: null,
      model: { type: "WordLevel", vocab, unk_token: "[UNK]" },
    }),
  };
}

export function modelConfigFiles(): Record<string, string> {
  return {
    "config.json": JSON.stringify(GLM51_TINY_CONFIG, null, 2),
    "generation_config.json": JSON.stringify({ temperature: 1.0 }),
    ...tokenizerFiles(),
    "chat_template.jinja": "{{ messages }}",
  };
}

const CSF_IDENTITY = {
  schema: "lil-nvfp4-csf-checkpoint/1",
  codec: "byte-window4-fixed-stream-u24-exceptions/1",
  family: "glm53_744b_nvfp4",
} as const;

/** Write the plain safetensors root: one shard plus shared config files. */
function writePlainRoot(root: string, tensors: FixtureTensor[], configFiles: Record<string, string>): void {
  fs.mkdirSync(root, { recursive: true });
  writeSafetensorsFile(path.join(root, "model.safetensors"), tensors);
  for (const [name, content] of Object.entries(configFiles)) {
    fs.writeFileSync(path.join(root, name), Buffer.from(content, "utf-8"));
  }
}

/**
 * Write the CSF container root (adapted from tests/csf_fixture.ts's
 * buildCsfFixture for an arbitrary inventory): natives byte-identical to the
 * source shard, replaced scales as fixed/exception planes, full receipts,
 * manifest, build-contract, metadata and provenance files.
 */
function writeCsfRoot(
  root: string,
  tensors: FixtureTensor[],
  compressed: Map<string, ScaleMatrixFixture>,
  configFiles: Record<string, string>,
): void {
  fs.mkdirSync(path.join(root, "tensors"), { recursive: true });
  fs.mkdirSync(path.join(root, "receipts"), { recursive: true });
  fs.mkdirSync(path.join(root, "metadata"), { recursive: true });

  const file = "model-00001-of-00001.safetensors";
  const sourceBytes = writeSafetensorsFile("", tensors);
  const sourceHeader = sourceBytes.subarray(0, 8 + Number(sourceBytes.readBigUInt64LE(0)));

  const targetTensors: FixtureTensor[] = [];
  const scaleRecords: Record<string, unknown> = {};
  for (const [scaleName, matrix] of compressed) {
    const logical = tensors.find(t => t.name === scaleName)!;
    const [rows, columns] = logical.shape;
    scaleRecords[scaleName] = {
      dtype: "F8_E4M3",
      shape: [rows, columns],
      source_bytes: rows * columns,
      source_sha256: sha256(matrix.logical),
      fixed_bytes: matrix.fixed.length,
      exception_bytes: matrix.exceptions.length,
    };
    targetTensors.push(
      { name: `${scaleName}.nvfp4_csf_fixed`, dtype: "U8", shape: [matrix.fixed.length], data: matrix.fixed },
      { name: `${scaleName}.nvfp4_csf_exceptions`, dtype: "U32", shape: [matrix.exceptions.length / 4], data: matrix.exceptions },
    );
  }
  for (const tensor of tensors) {
    if (compressed.has(tensor.name)) continue;
    targetTensors.push(tensor);
  }
  const targetBytes = writeSafetensorsFile(path.join(root, "tensors", file), targetTensors);

  const weightMap: Record<string, string> = {};
  for (const tensor of tensors) weightMap[tensor.name] = file;
  const metadataFiles: Record<string, string> = {
    ...configFiles,
    "model.safetensors.index.json": JSON.stringify({ weight_map: weightMap }),
  };
  const metadataSha256: Record<string, string> = {};
  for (const [name, content] of Object.entries(metadataFiles)) {
    const data = Buffer.from(content, "utf-8");
    fs.writeFileSync(path.join(root, "metadata", name), data);
    metadataSha256[name] = sha256(data);
  }

  fs.writeFileSync(path.join(root, "receipts", `${file}.json`), JSON.stringify({
    file,
    source_file_bytes: sourceBytes.length,
    source_sha256: sha256(sourceBytes),
    target_file_bytes: targetBytes.length,
    target_sha256: sha256(targetBytes),
    source_header_base64: sourceHeader.toString("base64"),
    scales: scaleRecords,
    source_content_address_verified: true,
  }));

  const shardRecord = {
    file,
    source_file_bytes: sourceBytes.length,
    source_sha256: sha256(sourceBytes),
    target_file_bytes: targetBytes.length,
    target_sha256: sha256(targetBytes),
  };
  const identity = {
    ...CSF_IDENTITY,
    metadata_sha256: metadataSha256,
  };
  const sourceIndex = metadataFiles["model.safetensors.index.json"];
  fs.writeFileSync(path.join(root, "manifest.json"), JSON.stringify({
    ...identity,
    compressed_scale_count: compressed.size,
    compressed_scale_bytes: targetTensors
      .filter(t => t.name.endsWith(".nvfp4_csf_fixed") || t.name.endsWith(".nvfp4_csf_exceptions"))
      .reduce((sum, t) => sum + t.data.length, 0),
    original_scale_bytes: [...compressed.keys()].reduce((sum, name) => {
      return sum + tensors.find(t => t.name === name)!.data.length;
    }, 0),
    source: "/fixture/glm51-tiny/source/snapshot",
    source_model_id: "local-inference-lab/GLM-5.3-NVFP4",
    source_revision: "b472e4ee53f6a9862da5486c56c6ca21be3dab70",
    source_index_sha256: sha256(Buffer.from(sourceIndex, "utf-8")),
    source_index_size_semantics: "safetensors_files",
    shards: [shardRecord],
    status: "fixture",
  }));
  fs.writeFileSync(path.join(root, "build-contract.json"), JSON.stringify({
    ...identity,
    compressed_scale_count: compressed.size,
    source: "/fixture/glm51-tiny/source/snapshot",
    source_model_id: "local-inference-lab/GLM-5.3-NVFP4",
    source_revision: "b472e4ee53f6a9862da5486c56c6ca21be3dab70",
    source_index_sha256: sha256(Buffer.from(sourceIndex, "utf-8")),
    source_index_size_semantics: "safetensors_files",
    source_names: weightMap,
  }));
  fs.writeFileSync(path.join(root, "lil-manifest.json"), JSON.stringify({
    schema: "lil-manifest/1",
    work: { work_name: "glm51-tiny-fixture" },
  }));
  fs.writeFileSync(path.join(root, "verification.json"), JSON.stringify({
    schema: "lil-nvfp4-csf-checkpoint/1",
    passed: true,
    conditions: "fixture container; recorded claims are not verification",
    qualification: "storage and CPU reconstruction",
    scale_matrices: compressed.size,
    shards: [file],
  }));
}

/** Build both roots from identical logical bytes under `scratch`. */
export function buildGlm51ModelFixturePair(scratch: string): ModelFixture {
  const { tensors, compressed } = buildLogicalTensors();
  const configFiles = modelConfigFiles();
  const plainRoot = path.join(scratch, "plain");
  const csfRoot = path.join(scratch, "csf");
  writePlainRoot(plainRoot, tensors, configFiles);
  writeCsfRoot(csfRoot, tensors, compressed, configFiles);
  return {
    plainRoot,
    csfRoot,
    tensors,
    byName: new Map(tensors.map(t => [t.name, t])),
    compressedScales: [...compressed.keys()],
    configFiles,
  };
}

function f32BufferToBf16(data: Buffer): Buffer {
  // f32ToBf16Bytes converts the entire underlying ArrayBuffer; copy the
  // scalar bytes into an exact Float32Array instead of a pooled Buffer view.
  const values = new Float32Array(data.length / 4);
  for (let i = 0; i < values.length; i++) values[i] = data.readFloatLE(i * 4);
  return f32ToBf16Bytes(values);
}

/**
 * The stored tensors Glm51Model.loadTensor produces from the logical
 * inventory: NVFP4 scale renames (.weight_scale -> .weight_weight_scale,
 * .weight_scale_2 -> .weight_weight_scale_2), the general F32 -> BF16
 * conversion (except F32 weight_scale_2 and the retained F32 input_scale
 * calibration scalars), the loadMlaWeight q/k/v splits and transposition, and
 * the tied lm_head alias.
 */
export function expectedStoredTensors(fixture: ModelFixture): Map<string, StoredTensorExpectation> {
  const d = GLM51_TINY_DIMS;
  const out = new Map<string, StoredTensorExpectation>();
  const put = (name: string, dtype: string, shape: number[], bytes: Buffer, ...sourceNames: string[]): void => {
    out.set(name, { sourceNames, dtype, shape, bytes });
  };

  for (const tensor of fixture.tensors) {
    const { name, dtype, shape, data } = tensor;
    if (name.endsWith(".input_scale")) {
      put(name, "F32", [...shape], data, name);
    } else if (name.endsWith(".weight_scale_2")) {
      put(name.replace(/\.weight_scale_2$/, ".weight_weight_scale_2"), "F32", [...shape], data, name);
    } else if (name.endsWith(".weight_scale")) {
      put(name.replace(/\.weight_scale$/, ".weight_weight_scale"), "F8_E4M3", [...shape], data, name);
    } else if (name.endsWith(".self_attn.q_b_proj.weight")) {
      // loadMlaWeight: per-head [qkHead, qLora] blocks split into nope/pe rows.
      const pfx = name.replace(/\.q_b_proj\.weight$/, "");
      const eb = 2;
      const headPitch = (d.qkNope + d.qkRope) * d.qLora * eb;
      const heads = Array.from({ length: d.heads }, (_, h) => h);
      const nope = Buffer.concat(heads.map(h => data.subarray(h * headPitch, h * headPitch + d.qkNope * d.qLora * eb)));
      const pe = Buffer.concat(heads.map(h => data.subarray(h * headPitch + d.qkNope * d.qLora * eb, (h + 1) * headPitch)));
      put(`${pfx}.q_nope_proj.weight`, "BF16", [d.heads * d.qkNope, d.qLora], nope, name);
      put(`${pfx}.q_pe_proj.weight`, "BF16", [d.heads * d.qkRope, d.qLora], pe, name);
    } else if (name.endsWith(".self_attn.kv_b_proj.weight")) {
      const pfx = name.replace(/\.kv_b_proj\.weight$/, "");
      const eb = 2;
      const headPitch = (d.qkNope + d.vHead) * d.kvLora * eb;
      const heads = Array.from({ length: d.heads }, (_, h) => h);
      const kNope = Buffer.concat(heads.map(h => data.subarray(h * headPitch, h * headPitch + d.qkNope * d.kvLora * eb)));
      // v per head [vHead, kvLora] is transposed to [kvLora, vHead].
      const v = Buffer.alloc(d.heads * d.kvLora * d.vHead * eb);
      for (let h = 0; h < d.heads; h++) {
        const raw = h * headPitch + d.qkNope * d.kvLora * eb;
        for (let i = 0; i < d.kvLora; i++) {
          for (let j = 0; j < d.vHead; j++) {
            const src = raw + (j * d.kvLora + i) * eb;
            const dst = ((h * d.kvLora + i) * d.vHead + j) * eb;
            data.copy(v, dst, src, src + eb);
          }
        }
      }
      put(`${pfx}.k_nope_proj.weight`, "BF16", [d.heads * d.qkNope, d.kvLora], kNope, name);
      put(`${pfx}.v_proj.weight`, "BF16", [d.heads * d.kvLora, d.vHead], v, name);
    } else if (name.endsWith(".self_attn.kv_a_proj_with_mqa.weight")) {
      const pfx = name.replace(/\.kv_a_proj_with_mqa\.weight$/, "");
      const eb = 2;
      const ckvBytes = d.kvLora * d.hidden * eb;
      put(`${pfx}.ckv_proj.weight`, "BF16", [d.kvLora, d.hidden], data.subarray(0, ckvBytes), name);
      put(`${pfx}.k_pe_proj.weight`, "BF16", [d.qkRope, d.hidden], data.subarray(ckvBytes), name);
    } else if (dtype === "F32") {
      put(name, "BF16", [...shape], f32BufferToBf16(data), name);
    } else {
      put(name, dtype, [...shape], data, name);
    }
  }

  // tie_word_embeddings: lm_head aliases the embedding table bytes.
  const embed = out.get("model.embed_tokens.weight")!;
  put("lm_head.weight", embed.dtype, [...embed.shape], embed.bytes, "model.embed_tokens.weight");
  return out;
}