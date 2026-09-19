import fs from "node:fs";
import path from "node:path";
import type { ChatCache, ChatTemplateKwargs, MtpDraftBatch, MtpStepResult, PhasedPrefillPlan, TokenSelector } from "./chat_model";
import { ChatModel, CommonModelConfig, SamplingParams } from "./chat_model";
import { DeviceOps, MaskMode, TensorParallelism } from "./device_ops";
import { executionPhase, type ExecutionPlan, ExecutionState, ExecutionWorkspace } from "./execution-workspace";
import { MemcpyKind } from "./enums";
import { BF16, f32ToBf16Bytes, I32 } from "./glm_ops";
import { GlmParser } from "./glm-parser";
import { resolveModelPath } from "./model_path";
import { PagedKVCache } from "./paged_kv";
import { SafeTensorFile, type TensorMeta } from "./safetensors";
import { Tensor } from "./tensor";
import { UsingHolder } from "./using-holder";

export { ExecutionState as BatchState };
export type { SamplingParams };

const GLM51_REPO = "zai-org/GLM-5.1";
const GLM51_MODEL_DIR = "tests/python/test_models/glm51_small/glm51_small_bf16";

export interface Glm51Config extends CommonModelConfig {
  moeIntermediateSize: number;
  kvLoraRank: number;
  qLoraRank: number;
  qkNopeHeadDim: number;
  qkRopeHeadDim: number;
  qkHeadDim: number;
  vHeadDim: number;
  nRoutedExperts: number;
  nSharedExperts: number;
  numExpertsPerTok: number;
  nGroup: number;
  topkGroup: number;
  normTopkProb: boolean;
  routedScalingFactor: number;
  indexTopk: number;
  indexHeadDim: number;
  indexNHeads: number;
  indexerTypes: string[];
  indexShareForMtp: boolean;
  ropeInterleave: boolean;
  indexerRopeInterleave: boolean;
  mlpLayerTypes: string[];
  numDenseMlpLayers: number;
  firstSparseMlpLayer: number;
  eosTokenIds: number[];
}

interface MtpVerificationArtifacts {
  seed: Tensor;
  mtpHiddenStaging: Tensor;
  kvCacheLayers: Array<{
    appendCkv: Tensor;
    appendKpe: Tensor;
    cacheIdx: number;
    kvLoraRank: number;
    qkRopeDim: number;
  }>;
  indexerKvCacheLayers: Array<{
    appendIdxK: Tensor;
    cacheIdx: number;
    indexHeadDim: number;
  }>;
  sharedSlots: Tensor;
  sharedSlotsLength: Tensor;
}

function mtpTotalPaths(topks: readonly number[]): number {
  return topks.reduce((acc, topk) => acc * topk, 1);
}

export function mtpTotalTreeNodes(topks: readonly number[]): number {
  let total = 0;
  let width = 1;
  for (const topk of topks) {
    width *= topk;
    total += width;
  }
  return total;
}

function mtpDepthBoundaries(topks: readonly number[]): number[] {
  const boundaries: number[] = [];
  let total = 0;
  let width = 1;
  for (const topk of topks) {
    width *= topk;
    total += width;
    boundaries.push(total);
  }
  return boundaries;
}

function mtpTreeDepth(topks: readonly number[], nodeIndex: number): number {
  const boundaries = mtpDepthBoundaries(topks);
  let depth = 0;
  while (depth < boundaries.length && nodeIndex >= boundaries[depth]) {
    depth++;
  }
  return depth;
}

function mtpParentIndex(topks: readonly number[], nodeIndex: number, boundaries = mtpDepthBoundaries(topks)): number {
  const depth = mtpTreeDepth(topks, nodeIndex);
  if (depth === 0) {
    return -1;
  }
  const offset = nodeIndex - boundaries[depth - 1];
  const parentOffset = Math.floor(offset / topks[depth]);
  return (depth > 1 ? boundaries[depth - 2] : 0) + parentOffset;
}

function mtpChildIndex(topks: readonly number[], nodeIndex: number, digit: number, boundaries = mtpDepthBoundaries(topks)): number {
  const depth = mtpTreeDepth(topks, nodeIndex);
  const offset = nodeIndex - (depth > 0 ? boundaries[depth - 1] : 0);
  return boundaries[depth] + offset * topks[depth + 1] + digit;
}

function mtpPathDigit(topks: readonly number[], path: number, layer: number, strides: readonly number[]): number {
  return Math.floor(path / strides[layer]) % topks[layer];
}

function ensureMtpTargetMask(ws: ExecutionWorkspace, topks: readonly number[]) {
  const numTokens = mtpTotalTreeNodes(topks);
  const byteLen = Math.ceil(numTokens * numTokens / 8);
  const key = topks.join("_");
  let mask = ws.tensors.get(`glm51_mtp_target_mask_${key}`);
  let indptr = ws.tensors.get(`glm51_mtp_target_mask_indptr_${key}`);
  if (!mask || !indptr) {
    using maskH = ws.allocPinned([ws.maxBatch * byteLen], "U8");
    const boundaries = mtpDepthBoundaries(topks);
    maskH.withPinnedBuffer(buf => {
      buf.fill(0);
      for (let batch = 0; batch < ws.maxBatch; batch++) {
        const byteOffset = batch * byteLen;
        for (let query = 0; query < numTokens; query++) {
          let current = query;
          while (true) {
            const bit = query * numTokens + current;
            buf[byteOffset + (bit >> 3)] |= 1 << (bit & 7);
            current = mtpParentIndex(topks, current, boundaries);
            if (current === -1) {
              break;
            }
          }
        }
      }
    });
    using indptrH = ws.allocPinned([ws.maxBatch + 1], "I32");
    indptrH.withPinnedBuffer(buf => {
      for (let batch = 0; batch <= ws.maxBatch; batch++) {
        buf.writeInt32LE(batch * byteLen, batch * I32);
      }
    });
    mask = ws.alloc(maskH.shape, "U8", `glm51_mtp_target_mask_${key}`);
    mask.memcpy(maskH, maskH.bytes, MemcpyKind.HostToDevice);
    indptr = ws.alloc(indptrH.shape, "I32", `glm51_mtp_target_mask_indptr_${key}`);
    indptr.memcpy(indptrH, indptrH.bytes, MemcpyKind.HostToDevice);
  }
  return { mask, indptr, mode: MaskMode.CausalCustom };
}

function ensureMtpChunkMask(ws: ExecutionWorkspace, topks: readonly number[], depth: number) {
  const qoLen = mtpTotalPaths(topks.slice(0, depth));
  const boundaries = mtpDepthBoundaries(topks);
  const priorTokens = depth > 1 ? boundaries[depth - 2] : 0;
  const maskKvLenValue = priorTokens + qoLen;
  const byteLen = Math.ceil(qoLen * maskKvLenValue / 8);
  const key = `${topks.join("_")}_d${depth}`;
  let mask = ws.tensors.get(`glm51_mtp_chunk_mask_${key}`);
  let indptr = ws.tensors.get(`glm51_mtp_chunk_mask_indptr_${key}`);
  let maskKvLen = ws.tensors.get(`glm51_mtp_chunk_mask_kvlen_${key}`);
  if (!mask || !indptr || !maskKvLen) {
    using maskH = ws.allocPinned([ws.maxBatch * byteLen], "U8");
    maskH.withPinnedBuffer(buf => {
      buf.fill(0);
      for (let batch = 0; batch < ws.maxBatch; batch++) {
        const byteOffset = batch * byteLen;
        for (let query = 0; query < qoLen; query++) {
          let current = priorTokens + query;
          while (current !== -1) {
            const bit = query * maskKvLenValue + current;
            buf[byteOffset + (bit >> 3)] |= 1 << (bit & 7);
            current = mtpParentIndex(topks, current, boundaries);
          }
        }
      }
    });
    using indptrH = ws.allocPinned([ws.maxBatch + 1], "I32");
    indptrH.withPinnedBuffer(buf => {
      for (let batch = 0; batch <= ws.maxBatch; batch++) {
        buf.writeInt32LE(batch * byteLen, batch * I32);
      }
    });
    using maskKvLenH = ws.allocPinned([ws.maxBatch], "I32");
    maskKvLenH.withPinnedBuffer(buf => {
      for (let batch = 0; batch < ws.maxBatch; batch++) {
        buf.writeInt32LE(maskKvLenValue, batch * I32);
      }
    });
    mask = ws.alloc(maskH.shape, "U8", `glm51_mtp_chunk_mask_${key}`);
    mask.memcpy(maskH, maskH.bytes, MemcpyKind.HostToDevice);
    indptr = ws.alloc(indptrH.shape, "I32", `glm51_mtp_chunk_mask_indptr_${key}`);
    indptr.memcpy(indptrH, indptrH.bytes, MemcpyKind.HostToDevice);
    maskKvLen = ws.alloc(maskKvLenH.shape, "I32", `glm51_mtp_chunk_mask_kvlen_${key}`);
    maskKvLen.memcpy(maskKvLenH, maskKvLenH.bytes, MemcpyKind.HostToDevice);
  }
  return { mask, indptr, maskKvLen };
}

function ensureMtpChunkPositionIds(ws: ExecutionWorkspace, originalAllocLens: readonly number[], depth: number, numTokens: number): Tensor {
  const batchSize = originalAllocLens.length;
  const totalTokens = batchSize * numTokens;
  const key = `glm51_mtp_chunk_pos_d${depth}_n${numTokens}`;
  const positionIds = ws.ensureAlloc([ws.maxBatch * numTokens], "I32", key);
  const positionIdsH = ws.ensureAllocPinned([ws.maxBatch * numTokens], "I32", `${key}_host`);
  positionIdsH.withPinnedBuffer(buf => {
    for (let batch = 0; batch < batchSize; batch++) {
      for (let token = 0; token < numTokens; token++) {
        buf.writeInt32LE(originalAllocLens[batch] + depth, (batch * numTokens + token) * I32);
      }
    }
  });
  positionIds.memcpy(positionIdsH, totalTokens * I32, MemcpyKind.HostToDevice);
  return positionIds;
}

function ensureMtpVerificationPositionIds(ws: ExecutionWorkspace, originalAllocLens: readonly number[], topks: readonly number[]): Tensor {
  const batchSize = originalAllocLens.length;
  const numNodes = mtpTotalTreeNodes(topks);
  const key = `glm51_mtp_verify_pos_${numNodes}`;
  const positionIds = ws.ensureAlloc([ws.maxBatch * numNodes], "I32", key);
  const positionIdsH = ws.ensureAllocPinned([ws.maxBatch * numNodes], "I32", `${key}_host`);
  const boundaries = mtpDepthBoundaries(topks);
  positionIdsH.withPinnedBuffer(buf => {
    let offset = 0;
    for (const originalAllocLen of originalAllocLens) {
      for (let node = 0; node < numNodes; node++) {
        let depth = 0;
        while (depth < boundaries.length && node >= boundaries[depth]) {
          depth++;
        }
        buf.writeInt32LE(originalAllocLen + depth, offset++ * I32);
      }
    }
  });
  positionIds.memcpy(positionIdsH, batchSize * numNodes * I32, MemcpyKind.HostToDevice);
  return positionIds;
}

function loadConfig(modelDir: string): Glm51Config {
  const raw = JSON.parse(fs.readFileSync(path.join(modelDir, "config.json"), "utf-8"));
  const mlpLayerTypes = raw.mlp_layer_types ?? ["dense", "dense", "dense", ...Array(Math.max(0, raw.num_hidden_layers - 3)).fill("sparse")];
  const numDenseMlpLayers = mlpLayerTypes.filter((t: string) => t === "dense").length;
  const firstSparseMlpLayer = mlpLayerTypes.findIndex((t: string) => t === "sparse");
  const qkNopeHeadDim = raw.qk_nope_head_dim ?? 0;
  const qkRopeHeadDim = raw.qk_rope_head_dim ?? 0;
  const qkHeadDim = raw.qk_head_dim ?? (qkNopeHeadDim + qkRopeHeadDim);
  const vHeadDim = raw.v_head_dim ?? 0;
  return {
    hiddenSize: raw.hidden_size,
    intermediateSize: raw.intermediate_size,
    moeIntermediateSize: raw.moe_intermediate_size ?? raw.intermediate_size,
    numHiddenLayers: raw.num_hidden_layers,
    numNextNPredictLayers: raw.num_nextn_predict_layers,
    rmsNormEps: raw.rms_norm_eps ?? 1e-5,
    vocabSize: raw.vocab_size,
    tieWordEmbeddings: raw.tie_word_embeddings ?? false,
    numAttentionHeads: raw.num_attention_heads,
    numKeyValueHeads: raw.num_key_value_heads,
    headDim: qkHeadDim,
    ropeTheta: raw.rope_parameters?.rope_theta ?? raw.rope_theta ?? 1000000,
    numKeyValueGroups: raw.num_attention_heads / raw.num_key_value_heads,
    scaling: Math.pow(qkHeadDim, -0.5),
    kvLoraRank: raw.kv_lora_rank,
    qLoraRank: raw.q_lora_rank,
    qkNopeHeadDim,
    qkRopeHeadDim,
    qkHeadDim,
    vHeadDim,
    nRoutedExperts: raw.n_routed_experts ?? 0,
    nSharedExperts: raw.n_shared_experts ?? 0,
    numExpertsPerTok: raw.num_experts_per_tok ?? raw.topk ?? 4,
    nGroup: raw.n_group ?? 1,
    topkGroup: raw.topk_group ?? 1,
    normTopkProb: raw.norm_topk_prob ?? false,
    routedScalingFactor: raw.routed_scaling_factor ?? 1.0,
    indexTopk: raw.index_topk ?? 256,
    // set to 0 to completely disable sparse indexing and fall back to dense attention.
    // GLM_DENSE_ATTN=1 does the same from the environment.
    indexHeadDim: process.env.GLM_DENSE_ATTN === "1" ? 0 : (raw.index_head_dim ?? 64),
    indexNHeads: raw.index_n_heads ?? 4,
    indexerTypes: raw.indexer_types
      ? [...raw.indexer_types, ...(raw.num_nextn_predict_layers ? [raw.index_share_for_mtp_iteration ? "shared" : "full"] : [])]
      : Array(raw.num_hidden_layers + (raw.num_nextn_predict_layers ?? 0)).fill("full"),
    indexShareForMtp: raw.index_share_for_mtp_iteration ?? false,
    ropeInterleave: raw.rope_interleave ?? false,
    indexerRopeInterleave: raw.indexer_rope_interleave ?? raw.rope_interleave ?? false,
    mlpLayerTypes,
    numDenseMlpLayers,
    firstSparseMlpLayer: firstSparseMlpLayer >= 0 ? firstSparseMlpLayer : numDenseMlpLayers,
    eosTokenIds: Array.isArray(raw.eos_token_id) ? raw.eos_token_id : [raw.eos_token_id ?? 2],
  };
}

export class Glm51Model extends ChatModel {
  static readonly WEIGHT_PREFIX = "model.layers.";
  readonly eosIds: Set<number>;
  cfg: Glm51Config;
  invFreq: Tensor;
  readonly contextParallel: boolean;
  private readonly mtp: boolean;

  private constructor(glm: DeviceOps, config: Glm51Config, contextParallel = false, mtp = false) {
    super(glm);
    this.cfg = config;
    this.eosIds = new Set(config.eosTokenIds);
    this.invFreq = this.initInvFreq(config.qkRopeHeadDim, config.ropeTheta);
    this.contextParallel = contextParallel;
    this.mtp = mtp;
  }

  override createParser(chatTemplateKwargs: ChatTemplateKwargs = {}): GlmParser {
    return new GlmParser(this.tokenizer, {
      continue_final_message: chatTemplateKwargs.continue_final_message,
      enable_thinking: chatTemplateKwargs.enable_thinking,
    });
  }

  static async fromPretrained(glm: DeviceOps, repoIdOrDir: string = GLM51_MODEL_DIR, contextParallel = false, mtp = false): Promise<Glm51Model> {
    const modelDir = fs.existsSync(repoIdOrDir) ? repoIdOrDir : resolveModelPath(repoIdOrDir);
    const config = loadConfig(modelDir);
    const model = new Glm51Model(glm, config, contextParallel, mtp);
    await model.fromPretrained(modelDir, GLM51_REPO);
    return model;
  }

  private weightParallelism(name: string): TensorParallelism {
    if (name === "lm_head.weight") {
      return TensorParallelism.Column;
    }
    if (name === "model.embed_tokens.weight") {
      return TensorParallelism.Row;
    }
    const pfx = Glm51Model.WEIGHT_PREFIX;
    if (
      // good for decode, but bad for prefill due to gather
      // name.endsWith(".self_attn.q_a_proj.weight") ||
      // very small, output goes through kv_a_layernorm, split into replicated ckv/k_pe_proj anyway
      //name.endsWith(".self_attn.kv_a_proj_with_mqa.weight") ||
      name.endsWith(".mlp.gate_proj.weight") ||
      // moderate size weight but it is on critical path with nothing to overlap with at all.
      // the other bf16 weights (indexer, ckv, q, etc) contend with each other, so overlap works there.
      // but this linear happens in isolation.
      // name.endsWith(".mlp.gate.weight") ||
      name.endsWith(".mlp.up_proj.weight") ||
      (name.startsWith(pfx) && name.includes(".mlp.experts.") && name.endsWith(".gate_proj.weight")) ||
      (name.startsWith(pfx) && name.includes(".mlp.experts.") && name.endsWith(".up_proj.weight")) ||
      name.endsWith(".mlp.shared_experts.gate_proj.weight") ||
      name.endsWith(".mlp.shared_experts.up_proj.weight") ||
      // NVFP4 block scale tensors follow same parallelism as their weight (weight_scale, not weight_scale_2 which is scalar)
      name.endsWith(".gate_proj.weight_weight_scale") ||
      name.endsWith(".up_proj.weight_weight_scale") ||
      (name.startsWith(pfx) && name.includes(".mlp.experts.") && name.endsWith(".gate_proj.weight_weight_scale")) ||
      (name.startsWith(pfx) && name.includes(".mlp.experts.") && name.endsWith(".up_proj.weight_weight_scale")) ||
      name.endsWith(".mlp.shared_experts.gate_proj.weight_weight_scale") ||
      name.endsWith(".mlp.shared_experts.up_proj.weight_weight_scale")
      || name.endsWith(".eh_proj.weight")
      || name.endsWith(".eh_proj.weight_weight_scale")
    ) {
      return TensorParallelism.Column;
    }
    if (
      name.endsWith(".self_attn.o_proj.weight") ||
      name.endsWith(".mlp.down_proj.weight") ||
      (name.startsWith(pfx) && name.includes(".mlp.experts.") && name.endsWith(".down_proj.weight")) ||
      name.endsWith(".mlp.shared_experts.down_proj.weight") ||
      // NVFP4 block scale tensors follow same parallelism as their weight (weight_scale, not weight_scale_2 which is scalar)
      name.endsWith(".down_proj.weight_weight_scale") ||
      (name.startsWith(pfx) && name.includes(".mlp.experts.") && name.endsWith(".down_proj.weight_weight_scale")) ||
      name.endsWith(".mlp.shared_experts.down_proj.weight_weight_scale")) {
      return TensorParallelism.Row;
    }
    // Indexer weights stay Replicated.
    if (name.includes('.indexer.wq_b.weight')) {
      return TensorParallelism.Replicated;
    }
    if (name.includes(".indexer.")) {
      return TensorParallelism.Replicated;
    }
    return TensorParallelism.Replicated;
  }

  protected async loadTensor(name: string, meta: TensorMeta, st: SafeTensorFile, mmapPtr: number): Promise<void> {
    // NVFP4 scale tensors: rename to match linear() lookup convention
    // e.g. "X.gate_proj.weight_scale" → "X.gate_proj.weight_weight_scale"
    if (name.endsWith(".input_scale")) {
      return; // not used by kernel
    }
    let storeName = name;
    if (name.endsWith(".weight_scale_2")) {
      storeName = name.replace(/\.weight_scale_2$/, ".weight_weight_scale_2");
    } else if (name.endsWith(".weight_scale")) {
      storeName = name.replace(/\.weight_scale$/, ".weight_weight_scale");
    }

    if (name.endsWith(".self_attn.q_b_proj.weight") ||
      name.endsWith(".self_attn.kv_b_proj.weight") ||
      name.endsWith(".self_attn.kv_a_proj_with_mqa.weight")) {
      await this.loadMlaWeight(name, meta, st, mmapPtr);
      return;
    }

    const par = this.weightParallelism(storeName);

    if (meta.dtype === "F32" && !name.endsWith(".weight_scale_2")) {
      const numElements = meta.shape.reduce((a, b) => a * b, 1);
      const tensor = this.alloc(meta.shape, "BF16", storeName, par);
      const f32Bytes = st.readTensor(name);
      const f32Arr = new Float32Array(f32Bytes.buffer, f32Bytes.byteOffset, numElements);
      tensor.h2d(f32ToBf16Bytes(f32Arr));

      if (this.cfg.tieWordEmbeddings && name === "model.embed_tokens.weight" && !this.tensors.has("lm_head.weight")) {
        const lmHead = this.alloc(meta.shape, "BF16", "lm_head.weight", TensorParallelism.Column);
        const embedOffset = st.dataStart + meta.dataOffsets[0];
        await lmHead.mmapLoad(mmapPtr, embedOffset, lmHead.bytes);
      }
    } else {
      const dtype = meta.dtype;
      const tensor = this.alloc(meta.shape, dtype, storeName, par);
      const offset = st.dataStart + meta.dataOffsets[0];
      await tensor.mmapLoad(mmapPtr, offset, tensor.bytes);

      if (this.cfg.tieWordEmbeddings && name === "model.embed_tokens.weight" && !this.tensors.has("lm_head.weight")) {
        const lmHead = this.alloc(meta.shape, dtype, "lm_head.weight", TensorParallelism.Column);
        await lmHead.mmapLoad(mmapPtr, offset, lmHead.bytes);
      }
    }
  }

  private tryComputeAbsorbed(layerPfx: string, nHeads: number, kvLoraRank: number, qLoraRank: number, qkNopeDim: number): void {
    const kNopeProj = this.tensors.get(`${layerPfx}.k_nope_proj.weight`);
    const qNopeProj = this.tensors.get(`${layerPfx}.q_nope_proj.weight`);
    if (!kNopeProj || !qNopeProj) {
      return;
    }
    using wAbsorbedTmp = kNopeProj.bmm(qNopeProj, nHeads, kvLoraRank, qLoraRank, qkNopeDim, true, false);
    const wAbsorbed = this.alloc(wAbsorbedTmp.shape, wAbsorbedTmp.type, `${layerPfx}.absorbed.weight`, wAbsorbedTmp.parallelism);
    if (process.env.GLM_MODEL_LOAD_REPLAY !== "1") {
      wAbsorbed.memcpy(wAbsorbedTmp);
    }
  }

  private async loadMlaWeight(name: string, meta: TensorMeta, st: SafeTensorFile, mmapPtr: number): Promise<void> {
    const cfg = this.cfg;
    const nHeads = cfg.numAttentionHeads;
    const qkNopeDim = cfg.qkNopeHeadDim;
    const qkRopeDim = cfg.qkRopeHeadDim;
    const vHeadDim = cfg.vHeadDim;
    const kvLoraRank = cfg.kvLoraRank;
    const qLoraRank = cfg.qLoraRank;
    const qkHeadDim = cfg.qkHeadDim;
    const inDim = meta.shape[1];
    const offset = st.dataStart + meta.dataOffsets[0];
    const layerPfx = name.replace(/\.(q_b_proj|kv_b_proj|kv_a_proj_with_mqa)\.weight$/, "");

    if (meta.dtype !== "BF16") {
      throw new Error(`loadMlaWeight: expected BF16, got ${meta.dtype}`);
    }

    // using column parallelism means there's a gather on the absorbed
    const nopeParallelism = TensorParallelism.Column;

    if (name.endsWith(".q_b_proj.weight")) {
      const eb = 2;
      const srcPitch = qkHeadDim * inDim * eb;
      const tQNope = this.alloc([nHeads * qkNopeDim, qLoraRank], "BF16", `${layerPfx}.q_nope_proj.weight`, nopeParallelism);
      const peName = name.replace(".q_b_proj.weight", ".q_pe_proj.weight");
      const tPe = this.alloc([nHeads * qkRopeDim, qLoraRank], "BF16", peName, nopeParallelism);
      await Promise.all([
        tQNope.mmapLoad(mmapPtr, offset, tQNope.bytes, { srcOffset: 0, dstOffset: 0, srcPitch, dstPitch: qkNopeDim * inDim * eb, width: qkNopeDim * inDim * eb, height: nHeads }),
        tPe.mmapLoad(mmapPtr, offset, tPe.bytes, { srcOffset: qkNopeDim * inDim * eb, dstOffset: 0, srcPitch, dstPitch: qkRopeDim * inDim * eb, width: qkRopeDim * inDim * eb, height: nHeads }),
      ]);
    } else if (name.endsWith(".kv_b_proj.weight")) {
      const eb = 2;
      const srcPitch = (qkNopeDim + vHeadDim) * inDim * eb;
      const tKNope = this.alloc([nHeads * qkNopeDim, kvLoraRank], "BF16", `${layerPfx}.k_nope_proj.weight`, nopeParallelism);
      const vName = name.replace(".kv_b_proj.weight", ".v_proj.weight");
      const vPar = TensorParallelism.Replicated;
      using tVRaw = this.alloc([nHeads * vHeadDim, kvLoraRank], "BF16", undefined, vPar);
      await Promise.all([
        tKNope.mmapLoad(mmapPtr, offset, tKNope.bytes, { srcOffset: 0, dstOffset: 0, srcPitch, dstPitch: qkNopeDim * inDim * eb, width: qkNopeDim * inDim * eb, height: nHeads }),
        tVRaw.mmapLoad(mmapPtr, offset, tVRaw.bytes, { srcOffset: qkNopeDim * inDim * eb, dstOffset: 0, srcPitch, dstPitch: vHeadDim * inDim * eb, width: vHeadDim * inDim * eb, height: nHeads }),
      ]);
      // Transpose v_proj: [nHeads, vHeadDim, kvLoraRank] -> [nHeads, kvLoraRank, vHeadDim]
      // The transposed layout enables coalesced reads in mla_v_expand_kernel.
      using tVT = tVRaw.transpose4d(1, nHeads, vHeadDim, kvLoraRank, 0, 1, 3, 2);
      const tV = this.alloc([nHeads * kvLoraRank, vHeadDim], "BF16", vName, vPar);
      if (process.env.GLM_MODEL_LOAD_REPLAY !== "1") {
        tV.memcpy(tVT);
      }
    } else if (name.endsWith(".kv_a_proj_with_mqa.weight")) {
      const ckvName = name.replace(".kv_a_proj_with_mqa.weight", ".ckv_proj.weight");
      const kpeName = name.replace(".kv_a_proj_with_mqa.weight", ".k_pe_proj.weight");
      // good for decode, but bad for prefill due to gather
      const ckv = this.alloc([kvLoraRank, inDim], "BF16", ckvName, TensorParallelism.Replicated);
      const kpe = this.alloc([qkRopeDim, inDim], "BF16", kpeName, TensorParallelism.Replicated);
      const eb = 2;
      await Promise.all([
        ckv.mmapLoad(mmapPtr, offset, ckv.bytes),
        kpe.mmapLoad(mmapPtr, offset + kvLoraRank * inDim * eb, kpe.bytes),
      ]);
    }

    if (name.endsWith(".q_b_proj.weight") || name.endsWith(".kv_b_proj.weight")) {
      this.tryComputeAbsorbed(layerPfx, nHeads, kvLoraRank, qLoraRank, qkNopeDim);
    }
  }

  protected async loadWeights(modelDir: string): Promise<void> {
    await super.loadWeights(modelDir);
    if (this.cfg.tieWordEmbeddings && !this.tensors.has("lm_head.weight")) {
      const embedTensor = this.tensors.get("model.embed_tokens.weight");
      if (embedTensor) {
        this.tensors.set("lm_head.weight", embedTensor);
      }
    }
  }

  createChatCache(maxPages = 256, maxBatch = 1, _maxSeqLen = 4096, pageSize = 64): ChatCache {
    if (pageSize !== 64) {
      throw new Error(`createChatCache: pageSize must be 64, got ${pageSize}`);
    }
    const cfg = this.cfg;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const nLayers = cfg.numHiddenLayers + (this.mtp ? cfg.numNextNPredictLayers ?? 0 : 0);
    const sharedLayers = cfg.indexerTypes.map(t => t === "shared");
    return new PagedKVCache(this.glm, nKv, hd, nLayers, maxPages, maxBatch, pageSize, cfg.kvLoraRank, cfg.qkRopeHeadDim, this.contextParallel, cfg.indexHeadDim, sharedLayers);
  }

  prepareMtpInput(cache: ChatCache, inputIdsList: number[][]): number[][] {
    const input = super.prepareMtpInput(cache, inputIdsList);
    if (!this.mtp) {
      return input;
    }

    const sequences = cache.getPagedKV().sequences;
    if (sequences.length !== input.length) {
      throw new Error(`prepareMtpInput: cache has ${sequences.length} sequences, received ${input.length} inputs`);
    }

    for (let i = 0; i < input.length; i++) {
      const sequence = sequences[i];
      const tokenIds = sequence.getTokenIds();
      const reportedLen = tokenIds.length;
      if (reportedLen === 0) {
        continue;
      }

      let previousToken: number;
      if (reportedLen === sequence.allocLen + 1) {
        // The sampled token is reported but has not passed through the target
        // model yet. It is already the overlap token for this prefill.
        previousToken = tokenIds[sequence.allocLen];
      } else if (reportedLen === sequence.allocLen && sequence.allocLen > 0) {
        previousToken = tokenIds[sequence.allocLen - 1];
        sequence.truncate(sequence.allocLen - 1);
      } else {
        throw new Error(`prepareMtpInput: sequence ${i} has allocLen ${sequence.allocLen} but ${reportedLen} reported tokens`);
      }
      input[i].unshift(previousToken);
    }
    return input;
  }

  private mlpDense(normed: Tensor, pfx: string, BS: number): Tensor {
    return normed.swiGluMlp(this.swiGluMlpWeights(`${pfx}.mlp`));
  }

  private getExpertWeights(pfx: string, proj: string): Tensor[] {
    const numExperts = this.cfg.nRoutedExperts;
    const weights: Tensor[] = [];
    for (let e = 0; e < numExperts; e++) {
      weights.push(this.tensors.get(`${pfx}.mlp.experts.${e}.${proj}.weight`)!);
    }
    return weights;
  }

  private mlpSparse(normed: Tensor, pfx: string, BS: number): Tensor {
    const cfg = this.cfg;
    const topK = cfg.numExpertsPerTok;
    const nGroup = cfg.nGroup;
    const moeIntermediate = cfg.moeIntermediateSize;
    const hs = cfg.hiddenSize;

    if (nGroup > 1) {
      // removed untested dead code that supported this, just guard
      throw new Error(`mlpSparse: nGroup > 1 is not supported (got nGroup=${nGroup})`);
    }

    // low occupancy during decode, start this first so it can run in parallel with the rest of the code and hopefully be done by the time we need it
    using sharedMlpStream = this.glm.withStream(() => {
      const sharedWeights = this.swiGluMlpWeights(`${pfx}.mlp.shared_experts`);
      return normed.swiGluMlp(sharedWeights);
    });

    using gateLogitsBuf = normed.linear(this.tensors.get(`${pfx}.mlp.gate.weight`)!);
    const routed = gateLogitsBuf.moeRoute({
      numExpertsPerToken: topK,
      correctionBias: this.tensors.get(`${pfx}.mlp.gate.e_score_correction_bias`),
      scalingFactor: cfg.routedScalingFactor,
      normalize: cfg.normTopkProb,
    });
    using topkIndices = routed.indices;
    using normalizedWeightsStream = routed.normalizedWeightsStream;
    using _normalizedWeights = normalizedWeightsStream.result;

    const count = BS * topK;
    using topkIndicesFlat = topkIndices.reshape([count]);

    const gateWeights = this.getExpertWeights(pfx, "gate_proj");
    const upWeights = this.getExpertWeights(pfx, "up_proj");
    const downWeights = this.getExpertWeights(pfx, "down_proj");

    using routedOut = normed.swiGluMlpMoeReduce({
      gate: gateWeights, up: upWeights, down: downWeights,
      normalizedWeightsStream,
    }, topkIndicesFlat, topK, count, moeIntermediate, hs, pfx);

    sharedMlpStream.streamWaitEvent();
    using sharedDownBuf = sharedMlpStream.result;

    using result = routedOut.add(sharedDownBuf, BS * hs);
    return result.reshape([BS, hs]);
  }

  private *mlaLayerPhased(cos: Tensor, sin: Tensor, normedHolder: UsingHolder<Tensor>, residualHolder: UsingHolder<Tensor>, layerIdx: number, state: ExecutionState, sharedSlots?: UsingHolder<Tensor>, sharedSlotsLength?: UsingHolder<Tensor>): Generator<void, { normed: Tensor, residual: Tensor }, void> {
    const cfg = this.cfg;
    const normed = normedHolder.value;
    const residual = residualHolder.value;
    const nHeads = cfg.numAttentionHeads;
    const kvLoraRank = cfg.kvLoraRank;
    const qkRopeDim = cfg.qkRopeHeadDim;
    const pfx = `${Glm51Model.WEIGHT_PREFIX}${layerIdx}.self_attn`;
    const batchSize = state.batchSize;
    const BS = state.totalTokens;
    const B = state.isDecode ? batchSize : 1;
    const S = state.isDecode ? 1 : state.totalTokens;
    const dense = cfg.indexHeadDim === 0;

    using kvcache = this.glm.withStream(() => {
      using kPeRopeStream = this.glm.withStream(() => {
        using kPeRaw = normed.linear(this.tensors.get(`${pfx}.k_pe_proj.weight`)!);
        return kPeRaw.applyRotaryPosEmb(cos, sin, qkRopeDim, qkRopeDim, 1, S, B, 1, cfg.ropeInterleave)
      });
      using kPeRope = kPeRopeStream.result;

      using ckv = normed.linear(this.tensors.get(`${pfx}.ckv_proj.weight`)!);
      using ckvNormed = ckv.rmsnorm(this.tensors.get(`${pfx}.kv_a_layernorm.weight`)!, cfg.rmsNormEps);

      kPeRopeStream.streamWaitEvent();
      const cache = state.mlaKvCacheAppend(ckvNormed, kPeRope, layerIdx, kvLoraRank, qkRopeDim);

      if (dense) {
        return cache;
      }

      using appendedCkv = cache.ckv;
      using _appendedKpe = cache.kpe;

      // The physical slots are derived per-layer from sharedTopk at attention
      // time below; the topk arg here is vestigial.
      return {
        ckv: state.sparseMlaPrepareCache(sharedSlots!.value, appendedCkv, ckvNormed, kPeRope, undefined, layerIdx, kvLoraRank, qkRopeDim),
      };
    });

    const shared = cfg.indexerTypes[layerIdx] === "shared";
    const skipIndexer = dense || shared;

    // Indexer K: wk(normed) → layernorm → partial RoPE → append to kData
    // Only 'full' layers have indexer weights; 'shared' layers reuse previous topk.
    using kvcacheIndex = skipIndexer
      ? undefined
      : this.glm.withStream(() => {
        const idxRopeDim = qkRopeDim;
        using idxKRaw = normed.linear(this.tensors.get(`${pfx}.indexer.wk.weight`)!);
        using idxKNormed = idxKRaw.layernorm(
          this.tensors.get(`${pfx}.indexer.k_norm.weight`)!,
          this.tensors.get(`${pfx}.indexer.k_norm.bias`)!,
          1e-6,
        );
        using idxKOut = idxKNormed.applyRotaryPosEmb(
          cos, sin, idxRopeDim, cfg.indexHeadDim, 1, S, B, 1, cfg.indexerRopeInterleave,
        );

        return state.indexerKvCacheAppend(idxKOut, layerIdx, cfg.indexHeadDim);
      });

    using idxWeightsStream = skipIndexer ? undefined : this.glm.withStream(() => {
      const idxNHeads = cfg.indexNHeads;
      const idxWeights = normed.linear(this.tensors.get(`${pfx}.indexer.weights_proj.weight`)!);
      idxWeights.scaleInPlace(Math.sqrt(1.0 / idxNHeads), BS * idxNHeads);
      return idxWeights;
    });

    using qResidBuf = normed.linear(this.tensors.get(`${pfx}.q_a_proj.weight`)!);
    yield;
    using qNormed = qResidBuf.rmsnorm(this.tensors.get(`${pfx}.q_a_layernorm.weight`)!, cfg.rmsNormEps);
    yield;

    // Indexer q: wq_b(qNormed) → ropeTranspose → [BS, indexNHeads, indexHeadDim]
    // Only 'full' layers compute indexer Q; 'shared' layers reuse previous topk.
    using idxQStream = skipIndexer
      ? undefined
      : this.glm.withStream(() => {
        const idxHeadDim = cfg.indexHeadDim;
        const idxTopk = cfg.indexTopk;

        // | `model.layers.N.self_attn.indexer.weights_proj.weight` | [32, 6144] | bfloat16 | 78 | 29.25 MB |
        // | `model.layers.N.self_attn.indexer.wq_b.weight` | [4096, 2048] | bfloat16 | 78 | 1.22 GB |

        using idxQLin = qNormed.linear(this.tensors.get(`${pfx}.indexer.wq_b.weight`)!);
        using idxQ = idxQLin.ropeTranspose(cos, sin, qkRopeDim, cfg.indexHeadDim, cfg.indexNHeads, S, B, cfg.indexHeadDim, cfg.indexerRopeInterleave);

        kvcacheIndex?.streamWaitEvent();
        using kData = kvcacheIndex!.result.kData;
        using kScaleData = kvcacheIndex!.result.kScaleData;

        idxWeightsStream!.streamWaitEvent();
        using idxWeights = idxWeightsStream!.result;

        // Store the raw indexer top-k (token positions); slots are derived
        // per-layer/per-mode below and in the gather (slotsReady).
        return state.indexerTopk(
          idxQ, kData, kScaleData, idxWeights,
          Math.pow(idxHeadDim, -0.5), idxTopk,
        );
      });

    const cache = kvcache.result;
    using qStream = this.glm.withStream(() => {
      return this.glm.projectMlaQuery(
        state, cache.ckv!, qNormed,
        this.tensors.get(`${pfx}.q_pe_proj.weight`)!,
        this.tensors.get(`${pfx}.q_nope_proj.weight`)!,
        this.tensors.get(`${pfx}.k_nope_proj.weight`)!,
        this.tensors.get(`${pfx}.absorbed.weight`)!,
        cos, sin,
        qkRopeDim, kvLoraRank, nHeads, S, B, cfg.ropeInterleave,
      );
    });

    using ckv = cache.ckv;
    using kpe = cache.kpe;
    using qAbsorbedR = qStream.result.qAbsorbed;
    using qAbsorbedScales = qStream.result.qAbsorbedScales;
    using qPeR = qStream.result.qPe;

    idxQStream?.streamWaitEvent();
    const topkResult = idxQStream?.result;
    using _topkValues = topkResult?.values;
    using topkIndices = topkResult?.indices;

    let sparseSlots: {
      slots: Tensor,
      length: Tensor,
      stream?: Disposable & { streamWaitEvent(): void, synchronize(): void },
    } | undefined;
    if (cfg.indexHeadDim !== 0) {
      if (!sharedSlots || !sharedSlotsLength) {
        throw new Error('Shared slot holders must be installed before sparse MLA attention.');
      }

      if (!topkIndices) {
        if (!shared) {
          throw new Error('Full layers must receive topk indices; they cannot reuse the group cache.');
        }
        if (!sharedSlots.value || !sharedSlotsLength.value) {
          throw new Error(`Shared layer ${layerIdx} has no group cache; the preceding full layer did not populate one.`);
        }
        sparseSlots = {
          slots: sharedSlots.value.viewClone(),
          length: sharedSlotsLength.value.viewClone(),
        };
      }
      else {
        if (shared) {
          throw new Error('Shared layers should not receive topk indices; they reuse the group cache.');
        }

        // Release the previous group before allocating the next one so the
        // workspace reuses the same addresses on every graph replay.
        sharedSlots.release();
        sharedSlotsLength.release();

        const pagedKV = state.cache.getPagedKV();
        const kData = pagedKV.kData[layerIdx];
        const maxKv = kData.shape[0] * kData.shape[1];
        const { layer, group, stream } = this.glm.topkToSlots(
          state,
          topkIndices, state.kvTokenIndptrD,
          state.indices, state.indptrD, state.lastPageLen, state.mlaBatchIndices,
          pagedKV.pageSize, maxKv,
          layerIdx, pagedKV.contextParallel,
        );
        sharedSlots.replace(group.slots);
        sharedSlotsLength.replace(group.length);
        sparseSlots = { slots: layer.slots, length: layer.length, stream };
      }
    }

    using slots = sparseSlots?.slots;
    using slotsLength = sparseSlots?.length;
    using slotsStream = sparseSlots?.stream;

    kvcache.streamWaitEvent();
    qStream.streamWaitEvent();

    using oProjBuf = new UsingHolder<Tensor>(undefined!);
    using prefetchL2 = new UsingHolder<ReturnType<DeviceOps["withStream"]>>(undefined!);
    {
      let attnOut: Tensor;
      let lseBuf: Tensor;

      let tokenMajor = false;

      if (!dense) {
        // Sparse MLA path: SM120 kernel on packed FP8 KV cache
        // SM120 outputs [BS, nHeads, kvLoraRank] (token-major).
        // mlaVExpand reads attn_out as [batch * seqLen, heads, kv_lr] when
        // seqLen=1, batch=BS — which matches token-major layout.
        const sparseResult = state.sparseMla(
          qAbsorbedR, qPeR, ckv!, slots!, slotsLength!,
          cfg.indexTopk, cfg.scaling, qAbsorbedScales,
        );
        tokenMajor = !state.isDecode;

        attnOut = sparseResult.o;
        lseBuf = sparseResult.lse;
      } else {
        // Dense MLA path (FlashInfer plan/run)
        if (!kpe) {
          throw new Error("Dense MLA requires a separate KPE cache");
        }
        const mlaResult = state.denseMla(qAbsorbedR, qPeR, ckv, kpe, cfg.scaling);
        attnOut = mlaResult.o;
        lseBuf = mlaResult.lse;
      }

      using _attnOut = attnOut;
      using _lseBuf = lseBuf;

      const vProj = this.tensors.get(`${pfx}.v_proj.weight`)!;
      const oProj = this.tensors.get(`${pfx}.o_proj.weight`)!;
      if (BS <= 32 && process.env.GLM_L2_PREFETCH !== "0") {
        prefetchL2.replace(this.glm.withStream(() => this.glm.prefetchL2([oProj])));
      }
      using vExpanded = attnOut.mlaVExpand(vProj, S, B, lseBuf, undefined, undefined, undefined, tokenMajor);
      oProjBuf.replace(vExpanded.outputProj(oProj));
    }

    yield;
    const attnResult = residual.fusedAddRmsnorm(oProjBuf.value, this.tensors.get(`${Glm51Model.WEIGHT_PREFIX}${layerIdx}.post_attention_layernorm.weight`)!, cfg.rmsNormEps);
    using attnNormed = attnResult.normed;
    using attnResidual = attnResult.residual;
    yield;

    const mlpPfx = `${Glm51Model.WEIGHT_PREFIX}${layerIdx}`;
    using downBuf = layerIdx >= cfg.firstSparseMlpLayer
      ? this.mlpSparse(attnNormed, mlpPfx, BS)
      : this.mlpDense(attnNormed, mlpPfx, BS);

    let nextWeight: Tensor;
    if (layerIdx < cfg.numHiddenLayers - 1) {
      nextWeight = this.tensors.get(`${Glm51Model.WEIGHT_PREFIX}${layerIdx + 1}.input_layernorm.weight`)!;
    } else if (layerIdx === cfg.numHiddenLayers - 1) {
      nextWeight = this.tensors.get("model.norm.weight")!;
    } else if (layerIdx < cfg.numHiddenLayers + (cfg.numNextNPredictLayers ?? 0) - 1) {
      nextWeight = this.tensors.get(`${Glm51Model.WEIGHT_PREFIX}${layerIdx + 1}.input_layernorm.weight`)!;
    } else {
      nextWeight = this.tensors.get(`${Glm51Model.WEIGHT_PREFIX}${layerIdx}.shared_head.norm.weight`)!;
    }
    normedHolder.release();
    residualHolder.release();
    yield;
    const mlpResult = attnResidual.fusedAddRmsnorm(downBuf, nextWeight, cfg.rmsNormEps);

    // Only close the prefetch branch at layer end: outputProj reads immutable
    // weights and can use cache hits without waiting for the hint kernel.
    prefetchL2.value?.streamWaitEvent();
    slotsStream?.streamWaitEvent();
    yield;
    return { normed: mlpResult.normed, residual: mlpResult.residual };
  }

  *forwardPhased(state: ExecutionState, sharedSlots?: UsingHolder<Tensor>, sharedSlotsLength?: UsingHolder<Tensor>): Generator<void, Tensor, void> {
    const cfg = this.cfg;

    using rotaryEmbedding = this.glm.withStream(() => state.rotaryEmbedding(this.invFreq));

    const embedTable = this.tensors.get("model.embed_tokens.weight")!;

    using residual = new UsingHolder(state.embedding(embedTable));
    using normed = new UsingHolder(residual.value.rmsnorm(this.tensors.get(`${Glm51Model.WEIGHT_PREFIX}0.input_layernorm.weight`)!, cfg.rmsNormEps));

    rotaryEmbedding.streamWaitEvent();
    using cos = rotaryEmbedding.result.cos;
    using sin = rotaryEmbedding.result.sin;

    // Callers spanning multiple forwards provide persistent slot holders;
    // ordinary forwards use holders local to this invocation.
    using _localSlots = sharedSlots ? undefined : new UsingHolder<Tensor>(undefined!);
    using _localLength = sharedSlotsLength ? undefined : new UsingHolder<Tensor>(undefined!);
    sharedSlots ??= _localSlots!;
    sharedSlotsLength ??= _localLength!;

    for (let i = 0; i < cfg.numHiddenLayers; i++) {
      const result = yield* this.mlaLayerPhased(cos, sin, normed, residual, i, state, sharedSlots, sharedSlotsLength);
      normed.replace(result.normed);
      residual.replace(result.residual);
    }

    return normed.detach();
  }

  override forwardModel(state: ExecutionState, sharedSlots?: UsingHolder<Tensor>, sharedSlotsLength?: UsingHolder<Tensor>): Tensor {
    return this.runPhased(this.forwardPhased(state, sharedSlots, sharedSlotsLength));
  }

  *forwardMtpPhased(state: ExecutionState, previousHiddenState: Tensor, sharedSlots?: UsingHolder<Tensor>, sharedSlotsLength?: UsingHolder<Tensor>): Generator<void, Tensor, void> {
    const cfg = this.cfg;
    const hs = cfg.hiddenSize;
    const ws = state.ws;

    if (!this.mtp || !cfg.numNextNPredictLayers) {
      throw new Error("forwardMtp called but model is not configured for MTP or has no next-n predict layers");
    }

    using rotaryEmbedding = this.glm.withStream(() => state.rotaryEmbedding(this.invFreq));

    const embedTable = this.tensors.get("model.embed_tokens.weight")!;
    using hnormStream = ws.glm.withStream(() => {
      return previousHiddenState.rmsnorm(this.tensors.get(`${Glm51Model.WEIGHT_PREFIX}${cfg.numHiddenLayers}.hnorm.weight`)!, cfg.rmsNormEps);
    });
    using embedding = state.embedding(embedTable);

    // TODO: THIS IS NOT GRAPH CAPTURABLE
    // position 0 is supposed to be masked
    // there seems to be no adverse affect in NOT doing it, as it only affects short/initial sequence.
    // let row = 0;
    // const sequences = state.cache.getPagedKV().sequences;
    // for (let i = 0; i < state.batchSize; i++) {
    //   const seqLen = state.seqLens[i];
    //   const startPos = sequences[i].allocLen - seqLen;
    //   if (seqLen > 0 && startPos === 0) {
    //     using firstRow = embedding.narrow(row, 1);
    //     firstRow.fill(0, hs);
    //   }
    //   row += seqLen;
    // }

    using enorm = embedding.rmsnorm(this.tensors.get(`${Glm51Model.WEIGHT_PREFIX}${cfg.numHiddenLayers}.enorm.weight`)!, cfg.rmsNormEps);
    hnormStream.streamWaitEvent();
    using hnorm = hnormStream.result;
    using cat = enorm.cat([hnorm], 1);

    using residual = new UsingHolder(cat.linear(this.tensors.get(`${Glm51Model.WEIGHT_PREFIX}${cfg.numHiddenLayers}.eh_proj.weight`)!));
    using normed = new UsingHolder(residual.value.rmsnorm(this.tensors.get(`${Glm51Model.WEIGHT_PREFIX}${cfg.numHiddenLayers}.input_layernorm.weight`)!, cfg.rmsNormEps));

    rotaryEmbedding.streamWaitEvent();
    using cos = rotaryEmbedding.result.cos;
    using sin = rotaryEmbedding.result.sin;

    const layerIdx = cfg.numHiddenLayers;
    using _localSlots = sharedSlots ? undefined : new UsingHolder<Tensor>(undefined!);
    using _localLength = sharedSlotsLength ? undefined : new UsingHolder<Tensor>(undefined!);
    sharedSlots ??= _localSlots!;
    sharedSlotsLength ??= _localLength!;
    const result = yield* this.mlaLayerPhased(cos, sin, normed, residual, layerIdx, state, sharedSlots, sharedSlotsLength);
    using _residual = result.residual;

    // Return shared_head.norm(residual) so the recycled seed for the next MTP
    // step is already normed — matches sglang Glm4MoeModelNextN and vLLM v1
    // deepseek_mtp which both recycle the post-shared_head-norm state.
    return result.normed;
  }

  forwardMtp(state: ExecutionState, previousHiddenState: Tensor, sharedSlots?: UsingHolder<Tensor>, sharedSlotsLength?: UsingHolder<Tensor>) {
    return this.runPhased(this.forwardMtpPhased(state, previousHiddenState, sharedSlots, sharedSlotsLength));
  }

  private planMtpDraftDepths(ws: ExecutionWorkspace, cache: ChatCache, originalAllocLens: readonly number[], topks: readonly number[]) {
    const batchSize = originalAllocLens.length;
    const states: ExecutionState[] = [];
    const metadata: Array<{ qoLen: number; previousWidth: number; expandK: number; depth: number }> = [];
    for (let depth = 1; depth < topks.length; depth++) {
      const qoLen = mtpTotalPaths(topks.slice(0, depth));
      const previousWidth = depth > 1 ? mtpTotalPaths(topks.slice(0, depth - 1)) : 1;
      const mask = ensureMtpChunkMask(ws, topks, depth);
      const positionIds = ensureMtpChunkPositionIds(ws, originalAllocLens, depth, qoLen);
      const state = ws.planPrefill(this, batchSize, new Array(batchSize).fill(qoLen), cache, {
        ...mask,
        mode: MaskMode.CausalCustom,
        positionIds,
      });
      states.push(state);
      metadata.push({ qoLen, previousWidth, expandK: topks[depth - 1], depth });
    }
    return { states, metadata };
  }

  private runMtpDraft(
    ws: ExecutionWorkspace,
    topks: readonly number[],
    batchSize: number,
    seed: Tensor,
    sharedSlots: Tensor,
    sharedSlotsLength: Tensor,
    states: readonly ExecutionState[],
    metadata: readonly { qoLen: number; previousWidth: number; expandK: number; depth: number }[],
    treeHost: Tensor,
    samplingPolicy: TokenSelector,
  ): void {
    const hiddenSize = this.cfg.hiddenSize;
    const rowBytes = hiddenSize * BF16;
    const numTreeNodes = mtpTotalTreeNodes(topks);
    const boundaries = mtpDepthBoundaries(topks);
    const lmHead = this.tensors.get("lm_head.weight")!;

    using seedRows = seed.narrow(0, batchSize);
    using rootLogits = seedRows.linear(lmHead);
    const rootTopk = samplingPolicy.mtpEnabled ? undefined : rootLogits.topk(topks[0], this.cfg.vocabSize);
    using _rootValues = rootTopk?.values;
    using rootIndices = samplingPolicy.mtpEnabled ? samplingPolicy.sampleDraft!(rootLogits, 0) : rootTopk!.indices;
    for (let batch = 0; batch < batchSize; batch++) {
      treeHost.memcpy2d(
        batch * numTreeNodes * I32, topks[0] * I32,
        rootIndices, batch * topks[0] * I32, topks[0] * I32,
        topks[0] * I32, 1,
        MemcpyKind.DeviceToHost,
      );
    }
    if (states.length > 0) {
      states[0].inputIdsBuf.memcpy(rootIndices, rootIndices.bytes, MemcpyKind.DeviceToDevice);
    }

    for (let index = 0; index < states.length; index++) {
      const state = states[index];
      const { qoLen, previousWidth, expandK, depth } = metadata[index];
      const next = states[index + 1];
      using stateSharedSlots = new UsingHolder(sharedSlots.viewClone());
      using stateSharedSlotsLength = new UsingHolder(sharedSlotsLength.viewClone());
      state.setInput(state.inputIdsBuf);

      const previousRows = batchSize * previousWidth;
      using previousHidden = seed.narrow(0, previousRows);
      using expanded = expandK > 1 ? ws.alloc([batchSize * qoLen, hiddenSize], "BF16") : undefined;
      const inputHidden = expanded ?? previousHidden;
      if (expanded) {
        for (let child = 0; child < expandK; child++) {
          expanded.memcpy2d(
            child * rowBytes, expandK * rowBytes,
            previousHidden, 0, rowBytes,
            rowBytes, previousRows,
            MemcpyKind.DeviceToDevice,
          );
        }
      }

      using hiddenStates = this.forwardMtp(state, inputHidden, stateSharedSlots, stateSharedSlotsLength);
      using logits = hiddenStates.linear(lmHead);
      const topk = samplingPolicy.mtpEnabled ? undefined : logits.topk(topks[depth], this.cfg.vocabSize);
      using _values = topk?.values;
      using indices = samplingPolicy.mtpEnabled ? samplingPolicy.sampleDraft!(logits, depth) : topk!.indices;
      const depthOffset = boundaries[depth - 1];
      const tokensAtDepth = qoLen * topks[depth];
      for (let batch = 0; batch < batchSize; batch++) {
        treeHost.memcpy2d(
          (batch * numTreeNodes + depthOffset) * I32, tokensAtDepth * I32,
          indices, batch * tokensAtDepth * I32, tokensAtDepth * I32,
          tokensAtDepth * I32, 1,
          MemcpyKind.DeviceToHost,
        );
      }
      if (next) {
        next.inputIdsBuf.memcpy(indices, indices.bytes, MemcpyKind.DeviceToDevice);
        seed.memcpy(hiddenStates, hiddenStates.bytes, MemcpyKind.DeviceToDevice);
      }
    }
  }

  *planPrefillMtpChunk(ws: ExecutionWorkspace, cache: ChatCache, inputIds: number[][], nextTokens: number[]): ExecutionPlan<void> {
    if (!this.forwardMtp || inputIds.length !== nextTokens.length || inputIds.some(ids => ids.length === 0)) {
      throw new Error("MTP chunk prefill requires non-empty inputs and one next token per sequence");
    }
    const batchSize = inputIds.length;
    const originalAllocLens = cache.getPagedKV().sequences.map(sequence => sequence.allocLen);
    let completed = false;
    using _rollback = {
      [Symbol.dispose]: () => {
        if (!completed) {
          for (let batch = 0; batch < batchSize; batch++) {
            cache.getPagedKV().sequences[batch].truncate(originalAllocLens[batch]);
          }
        }
      },
    };
    const state = ws.planPrefill(this, batchSize, inputIds.map(ids => ids.length), cache);
    state.setInput(inputIds);

    yield* executionPhase({
      states: [state],
      inputs: {},
      captureKey: [],
      timingName: "prefill_chunk",
      run: () => {
        using sharedSlots = new UsingHolder<Tensor>(undefined!);
        using sharedSlotsLength = new UsingHolder<Tensor>(undefined!);
        using hiddenStates = this.forwardModel(state, sharedSlots, sharedSlotsLength);
        using nextDevice = ws.alloc([batchSize], "I32");
        const nextBuffer = Buffer.alloc(batchSize * I32);
        for (let batch = 0; batch < batchSize; batch++) {
          nextBuffer.writeInt32LE(nextTokens[batch], batch * I32);
        }
        nextDevice.h2d(nextBuffer);
        using rotatedInput = state.input!.rotateInputIds(state.qoIndptrD, nextDevice, batchSize);
        state.setInput(rotatedInput);
        using _mtpHidden = this.forwardMtp(state, hiddenStates, sharedSlots, sharedSlotsLength);
        return undefined;
      },
    });
    completed = true;
  }

  planPrefillMtpChunkPhased(ws: ExecutionWorkspace, cache: ChatCache, inputIds: number[][], nextTokens: number[]): PhasedPrefillPlan {
    if (!this.mtp || inputIds.length !== nextTokens.length || inputIds.some(ids => ids.length === 0)) {
      throw new Error("Phased MTP chunk prefill requires non-empty inputs and one next token per sequence");
    }
    const batchSize = inputIds.length;
    const sequences = cache.getPagedKV().sequences.slice();
    const originalAllocLens = sequences.map(sequence => sequence.allocLen);
    let completed = false;
    const rollback = () => {
      if (completed) return;
      for (let batch = 0; batch < batchSize; batch++) {
        sequences[batch].truncate(originalAllocLens[batch]);
      }
      completed = true;
    };
    const state = ws.planPrefill(this, batchSize, inputIds.map(ids => ids.length), cache);
    state.setInput(inputIds);
    const model = this;

    function* forward(): Generator<void, Tensor, void> {
      try {
        using sharedSlots = new UsingHolder<Tensor>(undefined!);
        using sharedSlotsLength = new UsingHolder<Tensor>(undefined!);
        using hiddenStates = yield* model.forwardPhased(state, sharedSlots, sharedSlotsLength);
        using nextDevice = ws.alloc([batchSize], "I32");
        const nextBuffer = Buffer.alloc(batchSize * I32);
        for (let batch = 0; batch < batchSize; batch++) {
          nextBuffer.writeInt32LE(nextTokens[batch], batch * I32);
        }
        nextDevice.h2d(nextBuffer);
        using rotatedInput = state.input!.rotateInputIds(state.qoIndptrD, nextDevice, batchSize);
        state.setInput(rotatedInput);
        const mtpHidden = yield* model.forwardMtpPhased(state, hiddenStates, sharedSlots, sharedSlotsLength);
        completed = true;
        return mtpHidden;
      } finally {
        if (!completed) rollback();
      }
    }

    return { state, generator: forward(), [Symbol.dispose]: rollback };
  }

  *planPrefillMtpDraftExtend(ws: ExecutionWorkspace, cache: ChatCache, inputIds: number[][], topks: readonly number[], samplingPolicy: TokenSelector = { selectTarget: logits => logits.argmax() }): ExecutionPlan<MtpDraftBatch> {
    const mtpEnabled = samplingPolicy.mtpEnabled === true;
    if (mtpEnabled && [samplingPolicy.prepareDraft, samplingPolicy.sampleDraft, samplingPolicy.finishDraft,
    samplingPolicy.prepareVerification, samplingPolicy.verify].some(method => typeof method !== "function")) {
      throw new Error("MTP-enabled sampling policy requires all MTP methods");
    }
    if (!this.forwardMtp || topks.length === 0) {
      throw new Error("MTP draft extend requires an MTP-enabled model and non-empty topks");
    }
    if (mtpEnabled && topks.some(topk => topk !== 1)) {
      throw new Error("Linear MTP sampling requires all draft widths to be 1");
    }
    const batchSize = inputIds.length;
    if (mtpEnabled) samplingPolicy.prepareDraft!(batchSize, topks.length);
    let completed = false;
    const prefillState = ws.planPrefill(this, batchSize, inputIds.map(ids => ids.length), cache);
    prefillState.setInput(inputIds);
    const committedAllocLens = cache.getPagedKV().sequences.map(sequence => sequence.allocLen);
    using _rollback = {
      [Symbol.dispose]: () => {
        if (!completed) {
          for (let batch = 0; batch < batchSize; batch++) {
            cache.getPagedKV().sequences[batch].truncate(committedAllocLens[batch]);
          }
        }
      },
    };

    const maxWidth = Math.max(1, ...topks.slice(0, -1).map((_, depth) => mtpTotalPaths(topks.slice(0, depth + 1))));
    const numTreeNodes = mtpTotalTreeNodes(topks);

    const prefillArtifacts = yield* executionPhase({
      states: [prefillState],
      inputs: {},
      captureKey: [],
      timingName: "draft_extend",
      run: () => {
        using sharedSlots = new UsingHolder<Tensor>(undefined!);
        using sharedSlotsLength = new UsingHolder<Tensor>(undefined!);
        const seed = ws.alloc([batchSize * maxWidth, this.cfg.hiddenSize], "BF16");
        const targetHost = ws.allocPinned([batchSize], "I32");
        using targetDevice = ws.alloc([batchSize], "I32");
        using hiddenStates = this.forwardModel(prefillState, sharedSlots, sharedSlotsLength);
        using logits = prefillState.computeLogits(hiddenStates, this);
        using target = samplingPolicy.selectTarget(logits);
        targetDevice.memcpy(target, batchSize * I32, MemcpyKind.DeviceToDevice);
        using rotatedInput = prefillState.input!.rotateInputIds(prefillState.qoIndptrD, targetDevice, batchSize);
        prefillState.setInput(rotatedInput);
        using mtpHidden = this.forwardMtp(prefillState, hiddenStates, sharedSlots, sharedSlotsLength);
        using lastIdx = prefillState.lastIdx;
        using lastHidden = mtpHidden.indexSelect(lastIdx, -1);
        // The draft seed comes from each sequence's final packed row, so retain
        // the sparse-attention slots from those same rows rather than row zero.
        const slotWidth = sharedSlots.value.shape[1];
        seed.memcpy(lastHidden, lastHidden.bytes, MemcpyKind.DeviceToDevice);
        targetHost.memcpy(targetDevice, batchSize * I32, MemcpyKind.DeviceToHost);

        if (inputIds.every(ids => ids.length === 1)) {
          // Common decode/reconditioning case: the packed rows already are the
          // final rows, so preserve the original zero-copy path.
          return {
            seed,
            targetHost,
            sharedSlots: sharedSlots.detach(),
            sharedSlotsLength: sharedSlotsLength.detach(),
          };
        }

        if (inputIds.slice(1).every(ids => ids.length === 1)) {
          // The first sequence may have several rows, but all selected final
          // rows are still contiguous and can be retained as views.
          const start = inputIds[0].length - 1;
          return {
            seed,
            targetHost,
            sharedSlots: sharedSlots.value.narrow(start, batchSize),
            sharedSlotsLength: sharedSlotsLength.value.narrow(start, batchSize),
          };
        }

        const slotRowBytes = slotWidth * I32;
        const lastSharedSlots = ws.alloc([batchSize, slotWidth], "I32");
        const lastSharedSlotsLength = ws.alloc([batchSize], "I32");
        let rowEnd = 0;
        for (let batch = 0; batch < batchSize; batch++) {
          rowEnd += inputIds[batch].length;
          lastSharedSlots.memcpy2d(
            batch * slotRowBytes, slotRowBytes,
            sharedSlots.value, (rowEnd - 1) * slotRowBytes, slotRowBytes,
            slotRowBytes, 1,
            MemcpyKind.DeviceToDevice,
          );
          lastSharedSlotsLength.memcpy2d(
            batch * I32, I32,
            sharedSlotsLength.value, (rowEnd - 1) * I32, I32,
            I32, 1,
            MemcpyKind.DeviceToDevice,
          );
        }

        return {
          seed,
          targetHost,
          sharedSlots: lastSharedSlots,
          sharedSlotsLength: lastSharedSlotsLength,
        };
      },
    });

    let targetTokens: number[];
    {
      using targetHost = prefillArtifacts.targetHost;
      const targetBuf = targetHost.readPinnedBuffer();
      targetTokens = Array.from({ length: batchSize }, (_, batch) => targetBuf.readInt32LE(batch * I32));
    }

    using seed = prefillArtifacts.seed;
    using sharedSlots = prefillArtifacts.sharedSlots;
    using sharedSlotsLength = prefillArtifacts.sharedSlotsLength;
    const draftPlan = this.planMtpDraftDepths(ws, cache, committedAllocLens, topks);
    const draftArtifacts = yield* executionPhase({
      states: draftPlan.states,
      inputs: {
        seed,
        sharedSlots: sharedSlots,
        sharedSlotsLength: sharedSlotsLength,
      },
      captureKey: [],
      timingName: "draft",
      run: (inputs) => {
        const treeHost = ws.allocPinned([batchSize * numTreeNodes], "I32");
        this.runMtpDraft(ws, topks, batchSize, inputs.seed, inputs.sharedSlots, inputs.sharedSlotsLength, draftPlan.states, draftPlan.metadata, treeHost, samplingPolicy);
        return { treeHost };
      },
    });
    const proposal = mtpEnabled ? samplingPolicy.finishDraft!() : undefined;

    for (let batch = 0; batch < batchSize; batch++) {
      cache.getPagedKV().sequences[batch].truncate(committedAllocLens[batch]);
    }
    let treeTokens: number[][];
    {
      using treeHost = draftArtifacts.treeHost;
      const treeBuf = treeHost.readPinnedBuffer();
      treeTokens = Array.from({ length: batchSize }, (_, batch) =>
        Array.from({ length: numTreeNodes }, (_, node) => treeBuf.readInt32LE((batch * numTreeNodes + node) * I32)));
    }
    completed = true;
    return {
      targetTokens,
      treeTokens,
      topks,
      ...(proposal ? { proposal } : {}),
    };
  }

  *planTargetVerification(ws: ExecutionWorkspace, cache: ChatCache, draft: MtpDraftBatch, samplingPolicy: TokenSelector = { selectTarget: logits => logits.argmax() }): ExecutionPlan<MtpStepResult> {
    const mtpEnabled = samplingPolicy.mtpEnabled === true;
    if (mtpEnabled && [samplingPolicy.prepareDraft, samplingPolicy.sampleDraft, samplingPolicy.finishDraft,
    samplingPolicy.prepareVerification, samplingPolicy.verify].some(method => typeof method !== "function")) {
      throw new Error("MTP-enabled sampling policy requires all MTP methods");
    }
    const topks = draft.topks;
    const batchSize = draft.targetTokens.length;
    if (draft.treeTokens.length !== batchSize) {
      throw new Error("MTP draft batch does not match target token batch");
    }
    if (draft.proposal && !mtpEnabled) {
      throw new Error("MTP proposal verification requires a linear sampler");
    }
    if (mtpEnabled && (topks.length === 0 || topks.some(topk => topk !== 1))) {
      throw new Error("Linear MTP sampling requires non-empty draft widths all equal to 1");
    }
    if (mtpEnabled && !draft.proposal) {
      throw new Error("Linear MTP verification requires draft proposal probabilities");
    }
    if (mtpEnabled) samplingPolicy.prepareVerification!(draft);
    const originalAllocLens = cache.getPagedKV().sequences.map(sequence => sequence.allocLen);
    const targetTopks = [1, ...topks];
    const linearDraft = topks.every(topk => topk === 1);
    const numTreeNodes = mtpTotalTreeNodes(topks);
    const numVerificationTokens = numTreeNodes + 1;
    const verificationTokens = draft.treeTokens.map((tokens, batch) => [draft.targetTokens[batch], ...tokens]);
    let completed = false;
    using _rollback = {
      [Symbol.dispose]: () => {
        if (!completed) {
          for (let batch = 0; batch < batchSize; batch++) {
            cache.getPagedKV().sequences[batch].truncate(originalAllocLens[batch]);
          }
        }
      },
    };

    const targetMask = ensureMtpTargetMask(ws, targetTopks);
    const state = ws.planPrefill(this, batchSize, new Array(batchSize).fill(numVerificationTokens), cache, {
      ...targetMask,
      positionIds: ensureMtpVerificationPositionIds(ws, originalAllocLens, targetTopks),
    });
    state.setInput(verificationTokens);
    const maxWidth = Math.max(1, ...topks.slice(0, -1).map((_, depth) => mtpTotalPaths(topks.slice(0, depth + 1))));
    const argmaxHost = ws.ensureAllocPinned([ws.maxBatch * numVerificationTokens], "I32", `glm51_mtp_verify_argmax_host_${numVerificationTokens}`);
    const acceptedHost = mtpEnabled ? ws.ensureAllocPinned([ws.maxBatch], "I32", "glm51_mtp_verify_accepted_host") : undefined;

    const artifacts = yield* executionPhase({
      states: [state],
      inputs: {},
      captureKey: ["glm51-mtp-verify", topks.join(","), samplingPolicy.captureKey ?? "greedy", ...(mtpEnabled ? ["linear", samplingPolicy.mtpCaptureKey ?? 0] : [])],
      timingName: "verification",
      run: () => {
        const seed = ws.alloc([batchSize * maxWidth, this.cfg.hiddenSize], "BF16");
        const mtpHiddenStaging = ws.alloc([batchSize * numVerificationTokens, this.cfg.hiddenSize], "BF16");
        using slots = new UsingHolder<Tensor>(undefined!);
        using slotsLength = new UsingHolder<Tensor>(undefined!);
        const kvCacheLayers: MtpVerificationArtifacts["kvCacheLayers"] = [];
        const indexerKvCacheLayers: MtpVerificationArtifacts["indexerKvCacheLayers"] = [];
        const appendMla = state.mlaKvCacheAppend.bind(state);
        state.mlaKvCacheAppend = (appendCkv, appendKpe, cacheIdx, kvLoraRank, qkRopeDim) => {
          if (!linearDraft) {
            kvCacheLayers.push({ appendCkv: appendCkv.viewClone(), appendKpe: appendKpe.viewClone(), cacheIdx, kvLoraRank, qkRopeDim });
          }
          return appendMla(appendCkv, appendKpe, cacheIdx, kvLoraRank, qkRopeDim);
        };
        const appendIndexer = state.indexerKvCacheAppend.bind(state);
        state.indexerKvCacheAppend = (appendIdxK, cacheIdx, indexHeadDim) => {
          if (!linearDraft) {
            indexerKvCacheLayers.push({ appendIdxK: appendIdxK.viewClone(), cacheIdx, indexHeadDim });
          }
          return appendIndexer(appendIdxK, cacheIdx, indexHeadDim);
        };
        using hiddenStates = this.forwardModel(state, slots, slotsLength);
        using logits = state.computeLogits(hiddenStates, this, true);
        const verified = mtpEnabled ? samplingPolicy.verify!(logits) : undefined;
        using counts = verified?.numAccepted;
        // The accepted prefix and correction must also condition the retained MTP rows.
        using selected = verified ? verified.tokens : samplingPolicy.selectTarget(logits);
        if (counts) acceptedHost!.memcpy(counts, batchSize * I32, MemcpyKind.DeviceToHost);
        argmaxHost.memcpy(selected, batchSize * numVerificationTokens * I32, MemcpyKind.DeviceToHost);
        using selectedInput = selected.narrow(0, batchSize * numVerificationTokens);
        state.setInput(selectedInput);
        using mtpHidden = this.forwardMtp(state, hiddenStates, slots, slotsLength);
        mtpHiddenStaging.memcpy(mtpHidden, mtpHidden.bytes, MemcpyKind.DeviceToDevice);
        return {
          seed,
          mtpHiddenStaging,
          kvCacheLayers,
          indexerKvCacheLayers,
          sharedSlots: slots.detach(),
          sharedSlotsLength: slotsLength.detach(),
        };
      },
    });

    using seed = artifacts.seed;
    using sharedSlots = artifacts.sharedSlots;
    using sharedSlotsLength = artifacts.sharedSlotsLength;

    const numAccepted: number[] = [];
    const acceptedTokens: number[][] = [];
    const replacements: number[] = [];
    const acceptedNodes: number[][] = [];
    {
      const argmaxBuf = argmaxHost.readPinnedBuffer();
      const acceptedBuf = acceptedHost?.readPinnedBuffer();
      const targetBoundaries = mtpDepthBoundaries(targetTopks);
      const strides = topks.map((_, index) => mtpTotalPaths(topks.slice(index + 1)));
      const numPaths = mtpTotalPaths(topks);
      for (let batch = 0; batch < batchSize; batch++) {
        if (acceptedBuf) {
          const accepted = acceptedBuf.readInt32LE(batch * I32);
          if (accepted < 0 || accepted > topks.length) {
            throw new Error(`Invalid linear MTP acceptance count ${accepted}`);
          }
          const offset = batch * numVerificationTokens;
          numAccepted.push(accepted);
          acceptedTokens.push(Array.from({ length: accepted }, (_, index) => argmaxBuf.readInt32LE((offset + index) * I32)));
          replacements.push(argmaxBuf.readInt32LE((offset + accepted) * I32));
          acceptedNodes.push(Array.from({ length: accepted + 1 }, (_, index) => index));
          continue;
        }
        let bestPath = 0;
        let bestAccepted = -1;
        let replacement = -1;
        const argmaxOffset = batch * numVerificationTokens;
        for (let path = 0; path < numPaths; path++) {
          let accepted = 0;
          let node = 0;
          for (let layer = 0; layer < topks.length; layer++) {
            const digit = mtpPathDigit(topks, path, layer, strides);
            const child = mtpChildIndex(targetTopks, node, digit, targetBoundaries);
            if (verificationTokens[batch][child] !== argmaxBuf.readInt32LE((argmaxOffset + node) * I32)) {
              break;
            }
            accepted++;
            node = child;
          }
          if (accepted > bestAccepted) {
            bestAccepted = accepted;
            bestPath = path;
            replacement = argmaxBuf.readInt32LE((argmaxOffset + node) * I32);
          }
        }
        const tokens: number[] = [];
        const nodes = [0];
        let node = 0;
        for (let layer = 0; layer < bestAccepted; layer++) {
          node = mtpChildIndex(targetTopks, node, mtpPathDigit(topks, bestPath, layer, strides), targetBoundaries);
          tokens.push(verificationTokens[batch][node]);
          nodes.push(node);
        }
        numAccepted.push(bestAccepted);
        acceptedTokens.push(tokens);
        replacements.push(replacement);
        acceptedNodes.push(nodes);
      }
    }

    const finishCounts = numAccepted.map(value => value + 1);
    let committedAllocLens: number[];
    if (linearDraft) {
      for (let batch = 0; batch < batchSize; batch++) {
        cache.getPagedKV().sequences[batch].truncate(originalAllocLens[batch] + finishCounts[batch]);
      }
      committedAllocLens = cache.getPagedKV().sequences.map(sequence => sequence.allocLen);
    } else {
      for (let batch = 0; batch < batchSize; batch++) {
        cache.getPagedKV().sequences[batch].truncate(originalAllocLens[batch]);
      }
      const commitState = ws.planPrefill(this, batchSize, finishCounts, cache);
      commitState.setInput(acceptedTokens.map((tokens, batch) => [...tokens, replacements[batch]]));
      committedAllocLens = cache.getPagedKV().sequences.map(sequence => sequence.allocLen);
      const selectedSourceRows = acceptedNodes.flatMap((nodes, batch) => nodes.slice(0, finishCounts[batch]).map(node => batch * numVerificationTokens + node));
      const selectedSourceBuf = Buffer.alloc(selectedSourceRows.length * I32);
      for (let index = 0; index < selectedSourceRows.length; index++) {
        selectedSourceBuf.writeInt32LE(selectedSourceRows[index], index * I32);
      }
      using selectedSources = ws.alloc([selectedSourceRows.length], "I32");
      selectedSources.h2d(selectedSourceBuf);

      const pagedKV = cache.getPagedKV();
      const mlaSrcCkv: Tensor[] = [];
      const mlaSrcKpe: Tensor[] = [];
      const mlaDstCkv: Tensor[] = [];
      const mlaDstKpe: Tensor[] = [];
      for (const layer of artifacts.kvCacheLayers) {
        if (layer.kvLoraRank !== this.cfg.kvLoraRank || layer.qkRopeDim !== this.cfg.qkRopeHeadDim) {
          throw new Error(`MTP cache dimensions differ at layer ${layer.cacheIdx}`);
        }
        mlaSrcCkv.push(layer.appendCkv.resumeTracking());
        mlaSrcKpe.push(layer.appendKpe.resumeTracking());
        mlaDstCkv.push(pagedKV.ckvData[layer.cacheIdx]);
        if (!pagedKV.sparseMode) mlaDstKpe.push(pagedKV.kpeData[layer.cacheIdx]);
      }

      const indexerSrc: Tensor[] = [];
      const indexerDst: Tensor[] = [];
      const indexerDstScale: Tensor[] = [];
      for (const layer of artifacts.indexerKvCacheLayers) {
        if (layer.indexHeadDim !== this.cfg.indexHeadDim) {
          throw new Error(`MTP indexer dimension differs at layer ${layer.cacheIdx}`);
        }
        const destination = pagedKV.kData[layer.cacheIdx];
        const scaleDestination = pagedKV.kScaleData[layer.cacheIdx];
        if (!destination || !scaleDestination) throw new Error(`Missing indexer cache for layer ${layer.cacheIdx}`);
        indexerSrc.push(layer.appendIdxK.resumeTracking());
        indexerDst.push(destination);
        indexerDstScale.push(scaleDestination);
      }

      using mlaSrcCkvPtrs = ws.alloc([mlaSrcCkv.length], "I64");
      using mlaSrcKpePtrs = ws.alloc([mlaSrcKpe.length], "I64");
      using mlaDstCkvPtrs = ws.alloc([mlaDstCkv.length], "I64");
      using mlaDstKpePtrs = pagedKV.sparseMode ? undefined : ws.alloc([mlaDstKpe.length], "I64");
      mlaSrcCkvPtrs.writePointers(mlaSrcCkv);
      mlaSrcKpePtrs.writePointers(mlaSrcKpe);
      mlaDstCkvPtrs.writePointers(mlaDstCkv);
      mlaDstKpePtrs?.writePointers(mlaDstKpe);

      using indexerSrcPtrs = indexerSrc.length ? ws.alloc([indexerSrc.length], "I64") : undefined;
      using indexerDstPtrs = indexerDst.length ? ws.alloc([indexerDst.length], "I64") : undefined;
      using indexerDstScalePtrs = indexerDstScale.length ? ws.alloc([indexerDstScale.length], "I64") : undefined;
      indexerSrcPtrs?.writePointers(indexerSrc);
      indexerDstPtrs?.writePointers(indexerDst);
      indexerDstScalePtrs?.writePointers(indexerDstScale);

      ws.glm.appendSelectedMtpCaches(
        mlaSrcCkvPtrs, mlaSrcKpePtrs, mlaDstCkvPtrs, mlaDstKpePtrs,
        indexerSrcPtrs, indexerDstPtrs, indexerDstScalePtrs,
        selectedSources, commitState.indices, commitState.indptrD, commitState.mlaBatchIndices, commitState.positionIds,
        pagedKV.pageSize, this.cfg.kvLoraRank, this.cfg.qkRopeHeadDim, this.cfg.indexHeadDim,
        pagedKV.sparseMode, pagedKV.contextParallel ? ws.glm.worldSize : 0,
      );

      for (const layer of artifacts.kvCacheLayers) {
        layer.appendCkv[Symbol.dispose]();
        layer.appendKpe[Symbol.dispose]();
      }
      for (const layer of artifacts.indexerKvCacheLayers) {
        layer.appendIdxK[Symbol.dispose]();
      }
    }

    const rowBytes = this.cfg.hiddenSize * BF16;
    {
      using mtpHiddenStaging = artifacts.mtpHiddenStaging;
      mtpHiddenStaging.resumeTracking();
      for (let batch = 0; batch < batchSize; batch++) {
        const finalNode = acceptedNodes[batch][acceptedNodes[batch].length - 1];
        const source = batch * numVerificationTokens + finalNode;
        seed.memcpy2d(batch * rowBytes, rowBytes, mtpHiddenStaging, source * rowBytes, rowBytes, rowBytes, 1, MemcpyKind.DeviceToDevice);
      }
    }

    const draftPlan = this.planMtpDraftDepths(ws, cache, committedAllocLens, topks);
    const treeHost = ws.ensureAllocPinned([ws.maxBatch * numTreeNodes], "I32", `glm51_mtp_draft_host_${numTreeNodes}`);
    const draftInputs: { [name: string]: Tensor } = {
      seed,
      sharedSlots,
      sharedSlotsLength,
    };

    if (mtpEnabled) samplingPolicy.prepareDraft!(batchSize, topks.length);
    yield* executionPhase({
      states: draftPlan.states,
      inputs: draftInputs,
      captureKey: ["glm51-mtp-draft", topks.join(","), `batchSize:${batchSize}`, ...(mtpEnabled ? ["linear", samplingPolicy.mtpCaptureKey ?? 0] : [])],
      timingName: "draft",
      run: (inputs) => {
        this.runMtpDraft(ws, topks, batchSize, inputs.seed, inputs.sharedSlots, inputs.sharedSlotsLength, draftPlan.states, draftPlan.metadata, treeHost, samplingPolicy);
      },
    });
    const proposal = mtpEnabled ? samplingPolicy.finishDraft!() : undefined;

    for (let batch = 0; batch < batchSize; batch++) {
      cache.getPagedKV().sequences[batch].truncate(committedAllocLens[batch]);
    }
    const treeBuf = treeHost.readPinnedBuffer();
    const treeTokens = Array.from({ length: batchSize }, (_, batch) =>
      Array.from({ length: numTreeNodes }, (_, node) => treeBuf.readInt32LE((batch * numTreeNodes + node) * I32)));
    completed = true;
    return {
      draft: {
        targetTokens: replacements,
        treeTokens,
        topks,
        ...(proposal ? { proposal } : {}),
      },
      tokens: acceptedTokens.map((tokens, batch) => [...tokens, replacements[batch]]),
      numAccepted,
      numDraftTokens: topks.length,
    };
  }
}
