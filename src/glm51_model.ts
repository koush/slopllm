import fs from "node:fs";
import path from "node:path";
import type { ChatCache } from "./chat_model";
import { ChatModel, CommonModelConfig, SamplingParams } from "./chat_model";
import { DeviceOps, StridedMmap, TensorParallelism } from "./device_ops";
import { f32ToBf16Bytes } from "./glm_ops";
import { resolveModelPath } from "./model_path";
import { ExecutionState, PagedKVCache } from "./paged_kv";
import { SafeTensorFile, type TensorMeta } from "./safetensors";
import { Tensor } from "./tensor";
import { UsingHolder } from "./using-holder";
import { WorkspaceBase } from "./workspace";

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
  ropeInterleave: boolean;
  mlpLayerTypes: string[];
  numDenseMlpLayers: number;
  firstSparseMlpLayer: number;
  eosTokenIds: number[];
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
    indexHeadDim: raw.index_head_dim ?? 64,
    indexNHeads: raw.index_n_heads ?? 4,
    ropeInterleave: raw.rope_interleave ?? false,
    mlpLayerTypes,
    numDenseMlpLayers,
    firstSparseMlpLayer: firstSparseMlpLayer >= 0 ? firstSparseMlpLayer : numDenseMlpLayers,
    eosTokenIds: Array.isArray(raw.eos_token_id) ? raw.eos_token_id : [raw.eos_token_id ?? 2],
  };
}

class Glm51ChatCache implements ChatCache {
  constructor(
    public readonly pagedKV: PagedKVCache,
  ) { }

  getPagedKV(): PagedKVCache { return this.pagedKV; }

  reset(batchSize: number): void {
    this.pagedKV.reset(batchSize);
  }

  free(): void {
    this.pagedKV.free();
  }

  [Symbol.dispose](): void {
    this.free();
  }

  prefixMatch(seqIdx: number, inputIds: number[]): number[] {
    const batchSize = Math.max(this.pagedKV.seqPages.length, 1);
    this.pagedKV.reset(batchSize);
    return inputIds.slice();
  }

  appendTokens(seqIdx: number, tokens: number[]): void {
    this.pagedKV.appendTokens(seqIdx, tokens);
  }
}

interface MlaDeferred {
  mmapPtr: number;
  offset: number;
  nHeads: number;
  headDim: number;
  inDim: number;
  rowsPerHead: number;
  outDim: number;
  par: TensorParallelism;
}

export class Glm51Model extends ChatModel {
  static readonly WEIGHT_PREFIX = "model.layers.";
  readonly eosIds: Set<number>;
  cfg: Glm51Config;
  maxBatch: number;
  maxSeqLen: number;
  invFreq: Tensor;
  private readonly pendingKNope = new Map<string, MlaDeferred>();
  private readonly pendingQNope = new Map<string, MlaDeferred>();

  private constructor(glm: DeviceOps, config: Glm51Config, maxBatch: number, maxSeqLen: number) {
    super(glm);
    this.cfg = config;
    this.maxBatch = maxBatch;
    this.maxSeqLen = maxSeqLen;
    this.eosIds = new Set(config.eosTokenIds);
    this.invFreq = this.initInvFreq(config.qkRopeHeadDim, config.ropeTheta);
  }

  static async fromPretrained(glm: DeviceOps, repoIdOrDir: string = GLM51_MODEL_DIR, maxBatch = 1, maxSeqLen = 4096): Promise<Glm51Model> {
    const modelDir = fs.existsSync(repoIdOrDir) ? repoIdOrDir : resolveModelPath(repoIdOrDir);
    const config = loadConfig(modelDir);
    const model = new Glm51Model(glm, config, maxBatch, maxSeqLen);
    await model.fromPretrained(modelDir);
    return model;
  }

  private weightParallelism(name: string): TensorParallelism {
    if (name === "lm_head.weight") return TensorParallelism.Column;
    if (name === "model.embed_tokens.weight") return TensorParallelism.Row;
    const pfx = Glm51Model.WEIGHT_PREFIX;
    if (
      // very small and immediately rmsnorm
      //name.endsWith(".self_attn.q_a_proj.weight") ||
      // very small, output goes through kv_a_layernorm, split into replicated ckv/k_pe_proj anyway
      //name.endsWith(".self_attn.kv_a_proj_with_mqa.weight") ||
      name.endsWith(".mlp.gate_proj.weight") ||
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
      name.endsWith(".mlp.shared_experts.up_proj.weight_weight_scale")) return TensorParallelism.Column;
    if (name.endsWith(".self_attn.o_proj.weight") ||
      name.endsWith(".mlp.down_proj.weight") ||
      (name.startsWith(pfx) && name.includes(".mlp.experts.") && name.endsWith(".down_proj.weight")) ||
      name.endsWith(".mlp.shared_experts.down_proj.weight") ||
      // NVFP4 block scale tensors follow same parallelism as their weight (weight_scale, not weight_scale_2 which is scalar)
      name.endsWith(".down_proj.weight_weight_scale") ||
      (name.startsWith(pfx) && name.includes(".mlp.experts.") && name.endsWith(".down_proj.weight_weight_scale")) ||
      name.endsWith(".mlp.shared_experts.down_proj.weight_weight_scale")) return TensorParallelism.Row;
    return TensorParallelism.Replicated;
  }

  protected async loadTensor(name: string, meta: TensorMeta, st: SafeTensorFile, mmapPtr: number): Promise<void> {
    if (name.includes(".indexer.")) return;

    // NVFP4 scale tensors: rename to match linear() lookup convention
    // e.g. "X.gate_proj.weight_scale" → "X.gate_proj.weight_weight_scale"
    if (name.endsWith(".input_scale")) return; // not used by kernel
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

  private async splitMlaWeightMmap(
    mmapPtr: number, offset: number,
    nHeads: number, headDim: number, inDim: number,
    name0: string, rowsPerHead0: number, outDim0: number,
    name1: string, rowsPerHead1: number, outDim1: number,
    par: TensorParallelism,
  ): Promise<[Tensor, Tensor]> {
    const eb = 2;
    const srcPitch = headDim * inDim * eb;
    const t0 = this.alloc([nHeads * rowsPerHead0, outDim0], "BF16", name0, par);
    const t1 = this.alloc([nHeads * rowsPerHead1, outDim1], "BF16", name1, par);
    const strided: StridedMmap = { srcOffset: 0, dstOffset: 0, srcPitch, dstPitch: rowsPerHead0 * inDim * eb, width: rowsPerHead0 * inDim * eb, height: nHeads };
    await t0.mmapLoad(mmapPtr, offset, t0.bytes, strided);
    await t1.mmapLoad(mmapPtr, offset, t1.bytes, { srcOffset: rowsPerHead0 * inDim * eb, dstOffset: 0, srcPitch, dstPitch: rowsPerHead1 * inDim * eb, width: rowsPerHead1 * inDim * eb, height: nHeads });
    return [t0, t1];
  }

  private async loadDeferredMlaWeight(name: string, par: TensorParallelism): Promise<Tensor> {
    const deferred = this.pendingKNope.get(name) ?? this.pendingQNope.get(name);
    if (!deferred) throw new Error(`No deferred MLA info for ${name}`);
    const eb = 2;
    const srcPitch = deferred.headDim * deferred.inDim * eb;
    const t = this.alloc([deferred.nHeads * deferred.rowsPerHead, deferred.outDim], "BF16", undefined, par);
    await t.mmapLoad(deferred.mmapPtr, deferred.offset, t.bytes, { srcOffset: 0, dstOffset: 0, srcPitch, dstPitch: deferred.rowsPerHead * deferred.inDim * eb, width: deferred.rowsPerHead * deferred.inDim * eb, height: deferred.nHeads });
    return t;
  }

  private async tryComputeAbsorbed(layerPfx: string, nHeads: number, kvLoraRank: number, qLoraRank: number, qkNopeDim: number): Promise<void> {
    const kNopeKey = `${layerPfx}.k_nope_proj.weight`;
    const qNopeKey = `${layerPfx}.q_nope_proj.weight`;
    if (!this.pendingKNope.has(kNopeKey) || !this.pendingQNope.has(qNopeKey)) return;
    const colPar = TensorParallelism.Column;
    const kNopeProj = await this.loadDeferredMlaWeight(kNopeKey, colPar);
    const qNopeProj = await this.loadDeferredMlaWeight(qNopeKey, colPar);
    const wAbsorbed = kNopeProj.bmm(qNopeProj, nHeads, kvLoraRank, qLoraRank, qkNopeDim, true, false);
    wAbsorbed.setName(`${layerPfx}.absorbed.weight`);
    kNopeProj[Symbol.dispose]();
    qNopeProj[Symbol.dispose]();
    this.pendingKNope.delete(kNopeKey);
    this.pendingQNope.delete(qNopeKey);
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
    const colPar = TensorParallelism.Column;
    const inDim = meta.shape[1];
    const offset = st.dataStart + meta.dataOffsets[0];
    const layerPfx = name.replace(/\.(q_b_proj|kv_b_proj|kv_a_proj_with_mqa)\.weight$/, "");

    if (meta.dtype !== "BF16") {
      throw new Error(`loadMlaWeight: expected BF16, got ${meta.dtype}`);
    }

    if (name.endsWith(".q_b_proj.weight")) {
      const eb = 2;
      const srcPitch = qkHeadDim * inDim * eb;
      this.pendingQNope.set(name.replace(".q_b_proj.weight", ".q_nope_proj.weight"), {
        mmapPtr, offset, nHeads, headDim: qkHeadDim, inDim, rowsPerHead: qkNopeDim, outDim: qLoraRank, par: colPar,
      });
      const peName = name.replace(".q_b_proj.weight", ".q_pe_proj.weight");
      const tPe = this.alloc([nHeads * qkRopeDim, qLoraRank], "BF16", peName, colPar);
      await tPe.mmapLoad(mmapPtr, offset, tPe.bytes, { srcOffset: qkNopeDim * inDim * eb, dstOffset: 0, srcPitch, dstPitch: qkRopeDim * inDim * eb, width: qkRopeDim * inDim * eb, height: nHeads });
    } else if (name.endsWith(".kv_b_proj.weight")) {
      const eb = 2;
      const srcPitch = (qkNopeDim + vHeadDim) * inDim * eb;
      this.pendingKNope.set(name.replace(".kv_b_proj.weight", ".k_nope_proj.weight"), {
        mmapPtr, offset, nHeads, headDim: qkNopeDim + vHeadDim, inDim, rowsPerHead: qkNopeDim, outDim: kvLoraRank, par: colPar,
      });
      const vName = name.replace(".kv_b_proj.weight", ".v_proj.weight");
      const tV = this.alloc([nHeads * vHeadDim, kvLoraRank], "BF16", vName, colPar);
      await tV.mmapLoad(mmapPtr, offset, tV.bytes, { srcOffset: qkNopeDim * inDim * eb, dstOffset: 0, srcPitch, dstPitch: vHeadDim * inDim * eb, width: vHeadDim * inDim * eb, height: nHeads });
    } else if (name.endsWith(".kv_a_proj_with_mqa.weight")) {
      await this.splitMlaWeightMmap(mmapPtr, offset,
        1, kvLoraRank + qkRopeDim, inDim,
        name.replace(".kv_a_proj_with_mqa.weight", ".ckv_proj.weight"), kvLoraRank, inDim,
        name.replace(".kv_a_proj_with_mqa.weight", ".k_pe_proj.weight"), qkRopeDim, inDim,
        TensorParallelism.Replicated);
    }

    if (name.endsWith(".q_b_proj.weight") || name.endsWith(".kv_b_proj.weight")) {
      await this.tryComputeAbsorbed(layerPfx, nHeads, kvLoraRank, qLoraRank, qkNopeDim);
    }
  }

  protected async loadWeights(modelDir: string): Promise<void> {
    await super.loadWeights(modelDir);

    this.tieEmbeddingToLmHead("model.embed_tokens.weight");
  }

  createChatCache(maxPages = 256): ChatCache {
    const cfg = this.cfg;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    return new PagedKVCache(this.glm, nKv, hd, cfg.numHiddenLayers, maxPages, this.maxBatch, 16, cfg.kvLoraRank, cfg.qkRopeHeadDim);
  }

  private mlpDense(normed: Tensor, pfx: string, BS: number): Tensor {
    return this.swiGluMlp(normed, pfx, this.cfg.intermediateSize, BS);
  }

  private getBatchIds(ws: WorkspaceBase, count: number, topK: number): Tensor {
    const key = `__moe_batch_ids_top${topK}`;
    let batchIds = ws.tensors.get(key);
    if (!batchIds || batchIds.shape[0] < count) {
      batchIds = ws.alloc([count], "I32", key);
      const arr = new Int32Array(count);
      for (let i = 0; i < count; i++) arr[i] = Math.floor(i / topK);
      batchIds.h2d(Buffer.from(arr.buffer));
    }
    return batchIds;
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
    const numExperts = cfg.nRoutedExperts;
    const topK = cfg.numExpertsPerTok;
    const nGroup = cfg.nGroup;
    const topkGroup = cfg.topkGroup;
    const moeIntermediate = cfg.moeIntermediateSize;
    const hs = cfg.hiddenSize;
    const expertsPerGroup = numExperts / nGroup;
    const ws = normed.workspace;

    using gateLogitsBuf = normed.linear(this.tensors.get(`${pfx}.mlp.gate.weight`)!, BS);
    using gateSigmoid = gateLogitsBuf.sigmoid();

    let topkInput: Tensor;
    using topkInputHolder = new UsingHolder<Tensor>(undefined!);
    const eScoreBias = this.tensors.get(`${pfx}.mlp.gate.e_score_correction_bias`);
    if (eScoreBias) {
      topkInput = gateSigmoid.add(eScoreBias);
      topkInputHolder.replace(topkInput);
    } else if (nGroup > 1) {
      using zeros = ws.alloc([BS, numExperts], "BF16");
      zeros.fill(0, BS * numExperts);
      topkInput = gateSigmoid.add(zeros, BS * numExperts);
      topkInputHolder.replace(topkInput);
    } else {
      topkInput = gateSigmoid;
    }

    if (nGroup > 1) {
      const groupTopk = topkInput.reshape([BS * nGroup, expertsPerGroup]).topk(2, expertsPerGroup);
      using _groupTopkValues = groupTopk.values;
      using groupSums = groupTopk.values.reduceSum(2, BS * nGroup);
      using groupSums2d = groupSums.reshape([BS, nGroup]);
      const groupIdxTopk = groupSums2d.topk(topkGroup, nGroup);
      using _groupIdxValues = groupIdxTopk.values;
      using groupIdx = groupIdxTopk.indices;
      using groupMask = ws.alloc([BS, nGroup], "BF16");
      groupMask.fill(0, BS * nGroup);
      groupMask.scatterScalar(groupIdx, 1.0, topkGroup, nGroup, BS);
      topkInput.groupMaskMul(groupMask, numExperts, expertsPerGroup, nGroup, BS);
    }

    const topkResult = topkInput.topk(topK, numExperts);
    using _topkValues = topkResult.values;
    using topkIndices = topkResult.indices;

    using selectedScores = gateSigmoid.gather(topkIndices, topK, numExperts, BS);
    using normalizedWeights = selectedScores.rowNormalize(cfg.routedScalingFactor, topK, BS, cfg.normTopkProb);

    const count = BS * topK;
    const topkIndicesFlat = topkIndices.reshape([count]);

    using routedOut = ws.alloc([BS, hs], "BF16");
    routedOut.fill(0, BS * hs);

    const batchIds = this.getBatchIds(ws, count, topK);
    const downBatchIds = this.getBatchIds(ws, count, 1);

    const gateWeights = this.getExpertWeights(pfx, "gate_proj");
    const upWeights = this.getExpertWeights(pfx, "up_proj");
    const downWeights = this.getExpertWeights(pfx, "down_proj");

    using gateOutStream = this.glm.withStream(() => normed.mulMatId(gateWeights, topkIndicesFlat, batchIds, count, moeIntermediate, hs, `${pfx}.gate_proj`));
    using gateOut = gateOutStream.result;
    using upOut = normed.mulMatId(upWeights, topkIndicesFlat, batchIds, count, moeIntermediate, hs, `${pfx}.up_proj`);
    gateOutStream.streamWaitEvent();
    using siluOut = gateOut.siluAndMul(upOut, moeIntermediate, count);

    using downOut = siluOut.mulMatId(downWeights, topkIndicesFlat, downBatchIds, count, hs, moeIntermediate, `${pfx}.down_proj`);

    using normalizedWeightsFlat = normalizedWeights.reshape([count]);
    routedOut.scatterAddRows(downOut, normalizedWeightsFlat, batchIds, hs, count, BS);

    using sharedGateBufStream = this.glm.withStream(() => normed.linear(this.tensors.get(`${pfx}.mlp.shared_experts.gate_proj.weight`)!, BS));
    using sharedGateBuf = sharedGateBufStream.result;
    using sharedUpBuf = normed.linear(this.tensors.get(`${pfx}.mlp.shared_experts.up_proj.weight`)!, BS);
    sharedGateBufStream.streamWaitEvent();
    using sharedSiluBuf = sharedGateBuf.siluAndMul(sharedUpBuf, moeIntermediate, BS);
    using sharedDownBuf = sharedSiluBuf.linear(this.tensors.get(`${pfx}.mlp.shared_experts.down_proj.weight`)!, BS);

    const result = routedOut.add(sharedDownBuf, BS * hs);
    return result.reshape([BS, hs]);
  }

  private mlaLayer(normed: Tensor, residual: Tensor, layerIdx: number, state: ExecutionState): { normed: Tensor, residual: Tensor } {
    const cfg = this.cfg;
    const ws = state.ws;
    const hs = cfg.hiddenSize;
    const nHeads = cfg.numAttentionHeads;
    const kvLoraRank = cfg.kvLoraRank;
    const qkRopeDim = cfg.qkRopeHeadDim;
    const vHeadDim = cfg.vHeadDim;
    const pfx = `${Glm51Model.WEIGHT_PREFIX}${layerIdx}.self_attn`;
    const batchSize = state.batchSize;
    const totalTokens = state.totalTokens;
    const BS = totalTokens;
    const B = state.isDecode ? batchSize : 1;
    const S = state.isDecode ? 1 : totalTokens;

    using rotaryEmbedding = this.glm.withStream(() => this.invFreq.rotaryEmbedding(ws.positionIds, qkRopeDim / 2, B, S));
    using cos = rotaryEmbedding.result.cos;
    using sin = rotaryEmbedding.result.sin;

    using q = this.glm.withStream(() => {
      using qResidBuf = normed.linear(this.tensors.get(`${pfx}.q_a_proj.weight`)!, BS);
      using qNormed = qResidBuf.rmsnorm(this.tensors.get(`${pfx}.q_a_layernorm.weight`)!, cfg.rmsNormEps, cfg.qLoraRank, BS);

      using qAbsorbedLin = qNormed.linear(this.tensors.get(`${pfx}.absorbed.weight`)!, BS);
      using qPeLin = qNormed.linear(this.tensors.get(`${pfx}.q_pe_proj.weight`)!, BS);

      rotaryEmbedding.streamWaitEvent();
      using qPeR = this.glm.withStream(() => qPeLin.ropeTranspose(cos, sin, qkRopeDim, qkRopeDim, nHeads, S, B, qkRopeDim, cfg.ropeInterleave));

      const qAbsorbedR = state.isDecode
        ? qAbsorbedLin.ropeTranspose(undefined!, undefined!, 0, kvLoraRank, nHeads, S, B, kvLoraRank)
        : qAbsorbedLin.ropeTranspose(cos, sin, 0, kvLoraRank, nHeads, S, B, kvLoraRank);

      qPeR.streamWaitEvent();
      return {
        qAbsorbedR,
        qPeR: qPeR.result,
      }
    });
    
    using kPeRopeStream = this.glm.withStream(() => {
      using kPeRaw = normed.linear(this.tensors.get(`${pfx}.k_pe_proj.weight`)!, BS);
      rotaryEmbedding.streamWaitEvent();
      return kPeRaw.applyRotaryPosEmb(cos, sin, qkRopeDim, 1, S, B, 1, cfg.ropeInterleave)
    });
    using kPeRope = kPeRopeStream.result;

    using qAbsorbedR = q.result.qAbsorbedR;
    using qPeR = q.result.qPeR;

    using ckv = normed.linear(this.tensors.get(`${pfx}.ckv_proj.weight`)!, BS);
    using ckvNormed = ckv.rmsnorm(this.tensors.get(`${pfx}.kv_a_layernorm.weight`)!, cfg.rmsNormEps, kvLoraRank, BS);

    const pagedKV = state.cache.getPagedKV();
    const useMla = pagedKV.ckvData.length > 0;

    using attnOut = new UsingHolder<Tensor>(undefined!);

    if (useMla && state.isDecode) {
      kPeRopeStream.streamWaitEvent();
      state.mlaKvCacheAppend(ckvNormed, kPeRope, layerIdx, kvLoraRank, qkRopeDim);
      q.streamWaitEvent();
      attnOut.replace(ws.mlaDecodePaged(qAbsorbedR, qPeR, pagedKV, layerIdx, batchSize, nHeads, kvLoraRank, qkRopeDim, cfg.scaling));
    } else if (useMla && !state.isDecode) {
      this.glm.mlaPrefillPlan(
        ws.floatWs, 128 * 1024 * 1024,
        ws.intWs, ws.pinnedIntWs, 8 * 1024 * 1024,
        ws.mlaPrefillPlanInfo,
        ws.qoIndptrH, ws.indptrH,
        ws.kvLenH,
        batchSize, nHeads, kvLoraRank, true
      );

      kPeRopeStream.streamWaitEvent();
      state.mlaKvCacheAppend(ckvNormed, kPeRope, layerIdx, kvLoraRank, qkRopeDim);
      q.streamWaitEvent();
      attnOut.replace(ws.mlaPrefillPaged(qAbsorbedR, qPeR, pagedKV, layerIdx, totalTokens, batchSize, nHeads, kvLoraRank, qkRopeDim, cfg.scaling));
    } else {
      throw new Error("GLM-5.1 requires MLA KV cache");
    }

    const vProj = this.tensors.get(`${pfx}.v_proj.weight`)!;
    using vExpanded = attnOut.value.mlaVExpand(vProj, kvLoraRank, vHeadDim, nHeads, S, B);
    using oProjBuf = vExpanded.linear(this.tensors.get(`${pfx}.o_proj.weight`)!, BS);

    const attnResult = residual.fusedAddRmsnorm(oProjBuf, this.tensors.get(`${Glm51Model.WEIGHT_PREFIX}${layerIdx}.post_attention_layernorm.weight`)!, cfg.rmsNormEps, hs, BS);
    using attnNormed = attnResult.normed;
    using attnResidual = attnResult.residual;

    const mlpPfx = `${Glm51Model.WEIGHT_PREFIX}${layerIdx}`;
    using downBuf = layerIdx >= cfg.firstSparseMlpLayer
      ? this.mlpSparse(attnNormed, mlpPfx, BS)
      : this.mlpDense(attnNormed, mlpPfx, BS);
    const nextWeight = layerIdx < cfg.numHiddenLayers - 1
      ? this.tensors.get(`${Glm51Model.WEIGHT_PREFIX}${layerIdx + 1}.input_layernorm.weight`)!
      : this.tensors.get("model.norm.weight")!;
    const mlpResult = attnResidual.fusedAddRmsnorm(downBuf, nextWeight, cfg.rmsNormEps, hs, BS);
    return { normed: mlpResult.normed, residual: mlpResult.residual };
  }

  forward(state: ExecutionState): Tensor {
    const ws = state.ws;
    using _tracker = ws.startTracking();
    const cfg = this.cfg;
    const hs = cfg.hiddenSize;
    const batchSize = state.batchSize;
    const totalTokens = state.totalTokens;
    const BS = totalTokens;
    const B = state.isDecode ? batchSize : 1;
    const S = state.isDecode ? 1 : totalTokens;

    const embedTable = this.tensors.get("model.embed_tokens.weight")!;
    using residual = new UsingHolder(embedTable.embedding(ws.inputIdsBuf, hs, BS));

    using normed = new UsingHolder(residual.value.rmsnorm(this.tensors.get(`${Glm51Model.WEIGHT_PREFIX}0.input_layernorm.weight`)!, cfg.rmsNormEps, hs, BS));

    for (let i = 0; i < cfg.numHiddenLayers; i++) {
      const result = this.mlaLayer(normed.value, residual.value, i, state);
      normed.replace(result.normed);
      residual.replace(result.residual);
    }

    const result = this.computeLogits(normed.value, state);
    return result;
  }
}
