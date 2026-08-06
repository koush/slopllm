import fs from "node:fs";
import path from "node:path";
import type { ChatCache } from "./chat_model";
import { ChatModel, CommonModelConfig, SamplingParams } from "./chat_model";
import { DeviceOps, MaskMode, StridedMmap, TensorParallelism } from "./device_ops";
import { ExecutionState } from "./execution-workspace";
import { f32ToBf16Bytes } from "./glm_ops";
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
  private readonly pendingKNope = new Map<string, Tensor>();
  private readonly pendingQNope = new Map<string, Tensor>();
  private readonly mtp: boolean;

  private constructor(glm: DeviceOps, config: Glm51Config, contextParallel = false, mtp = false) {
    super(glm);
    this.cfg = config;
    this.eosIds = new Set(config.eosTokenIds);
    this.invFreq = this.initInvFreq(config.qkRopeHeadDim, config.ropeTheta);
    this.contextParallel = contextParallel;
    this.mtp = mtp;
  }

  static async fromPretrained(glm: DeviceOps, repoIdOrDir: string = GLM51_MODEL_DIR, contextParallel = false, mtp = false): Promise<Glm51Model> {
    const modelDir = fs.existsSync(repoIdOrDir) ? repoIdOrDir : resolveModelPath(repoIdOrDir);
    const config = loadConfig(modelDir);
    const model = new Glm51Model(glm, config, contextParallel, mtp);
    await model.fromPretrained(modelDir);
    return model;
  }

  private weightParallelism(name: string): TensorParallelism {
    if (name === "lm_head.weight") return TensorParallelism.Column;
    if (name === "model.embed_tokens.weight") return TensorParallelism.Row;
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
    )
      return TensorParallelism.Column;
    if (
      name.endsWith(".self_attn.o_proj.weight") ||
      name.endsWith(".mlp.down_proj.weight") ||
      (name.startsWith(pfx) && name.includes(".mlp.experts.") && name.endsWith(".down_proj.weight")) ||
      name.endsWith(".mlp.shared_experts.down_proj.weight") ||
      // NVFP4 block scale tensors follow same parallelism as their weight (weight_scale, not weight_scale_2 which is scalar)
      name.endsWith(".down_proj.weight_weight_scale") ||
      (name.startsWith(pfx) && name.includes(".mlp.experts.") && name.endsWith(".down_proj.weight_weight_scale")) ||
      name.endsWith(".mlp.shared_experts.down_proj.weight_weight_scale")) return TensorParallelism.Row;
    // Indexer weights are always Replicated — the indexer is a small module
    // that must run identically on every GPU to produce the same topk indices.
    if (name.includes('.indexer.wq_b.weight')) return TensorParallelism.Replicated;
    if (name.includes(".indexer.")) return TensorParallelism.Replicated;
    return TensorParallelism.Replicated;
  }

  protected async loadTensor(name: string, meta: TensorMeta, st: SafeTensorFile, mmapPtr: number): Promise<void> {
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

  private tryComputeAbsorbed(layerPfx: string, nHeads: number, kvLoraRank: number, qLoraRank: number, qkNopeDim: number): void {
    const kNopeKey = `${layerPfx}.k_nope_proj.weight`;
    const qNopeKey = `${layerPfx}.q_nope_proj.weight`;
    if (!this.pendingKNope.has(kNopeKey) || !this.pendingQNope.has(qNopeKey)) return;
    using kNopeProj = this.pendingKNope.get(kNopeKey)!;
    using qNopeProj = this.pendingQNope.get(qNopeKey)!;
    this.pendingKNope.delete(kNopeKey);
    this.pendingQNope.delete(qNopeKey);
    using wAbsorbedTmp = kNopeProj.bmm(qNopeProj, nHeads, kvLoraRank, qLoraRank, qkNopeDim, true, false);
    const wAbsorbed = this.alloc(wAbsorbedTmp.shape, wAbsorbedTmp.type, `${layerPfx}.absorbed.weight`, wAbsorbedTmp.parallelism);
    wAbsorbed.memcpy(wAbsorbedTmp);
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
      const tQNope = this.alloc([nHeads * qkNopeDim, qLoraRank], "BF16", undefined, nopeParallelism);
      const peName = name.replace(".q_b_proj.weight", ".q_pe_proj.weight");
      const tPe = this.alloc([nHeads * qkRopeDim, qLoraRank], "BF16", peName, nopeParallelism);
      await Promise.all([
        tQNope.mmapLoad(mmapPtr, offset, tQNope.bytes, { srcOffset: 0, dstOffset: 0, srcPitch, dstPitch: qkNopeDim * inDim * eb, width: qkNopeDim * inDim * eb, height: nHeads }),
        tPe.mmapLoad(mmapPtr, offset, tPe.bytes, { srcOffset: qkNopeDim * inDim * eb, dstOffset: 0, srcPitch, dstPitch: qkRopeDim * inDim * eb, width: qkRopeDim * inDim * eb, height: nHeads }),
      ]);
      this.pendingQNope.set(name.replace(".q_b_proj.weight", ".q_nope_proj.weight"), tQNope);
    } else if (name.endsWith(".kv_b_proj.weight")) {
      const eb = 2;
      const srcPitch = (qkNopeDim + vHeadDim) * inDim * eb;
      const tKNope = this.alloc([nHeads * qkNopeDim, kvLoraRank], "BF16", undefined, nopeParallelism);
      const vName = name.replace(".kv_b_proj.weight", ".v_proj.weight");
      const vPar = this.contextParallel ? TensorParallelism.Replicated : TensorParallelism.Column;
      using tVRaw = this.alloc([nHeads * vHeadDim, kvLoraRank], "BF16", undefined, vPar);
      await Promise.all([
        tKNope.mmapLoad(mmapPtr, offset, tKNope.bytes, { srcOffset: 0, dstOffset: 0, srcPitch, dstPitch: qkNopeDim * inDim * eb, width: qkNopeDim * inDim * eb, height: nHeads }),
        tVRaw.mmapLoad(mmapPtr, offset, tVRaw.bytes, { srcOffset: qkNopeDim * inDim * eb, dstOffset: 0, srcPitch, dstPitch: vHeadDim * inDim * eb, width: vHeadDim * inDim * eb, height: nHeads }),
      ]);
      // Transpose v_proj: [nHeads, vHeadDim, kvLoraRank] -> [nHeads, kvLoraRank, vHeadDim]
      // The transposed layout enables coalesced reads in mla_v_expand_kernel.
      using tVT = tVRaw.transpose4d(1, nHeads, vHeadDim, kvLoraRank, 0, 1, 3, 2);
      const tV = this.alloc([nHeads * kvLoraRank, vHeadDim], "BF16", vName, vPar);
      tV.memcpy(tVT);
      this.pendingKNope.set(name.replace(".kv_b_proj.weight", ".k_nope_proj.weight"), tKNope);
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
      if (embedTensor) this.tensors.set("lm_head.weight", embedTensor);
    }
  }

  createChatCache(maxPages = 256, maxBatch = 1, _maxSeqLen = 4096, pageSize = 64): ChatCache {
    if (pageSize !== 64)
      throw new Error(`createChatCache: pageSize must be 64, got ${pageSize}`);
    const cfg = this.cfg;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const nLayers = cfg.numHiddenLayers + (this.mtp ? cfg.numNextNPredictLayers ?? 0 : 0);
    const sharedLayers = cfg.indexerTypes.map(t => t === "shared");
    return new PagedKVCache(this.glm, nKv, hd, nLayers, maxPages, maxBatch, pageSize, cfg.kvLoraRank, cfg.qkRopeHeadDim, this.contextParallel, cfg.indexHeadDim, sharedLayers);
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
    const numExperts = cfg.nRoutedExperts;
    const topK = cfg.numExpertsPerTok;
    const nGroup = cfg.nGroup;
    const moeIntermediate = cfg.moeIntermediateSize;
    const hs = cfg.hiddenSize;

    // low occupancy during decode, start this first so it can run in parallel with the rest of the code and hopefully be done by the time we need it
    using sharedDownBufStream = this.glm.withStream(() => {
      const sharedWeights = this.swiGluMlpWeights(`${pfx}.mlp.shared_experts`);
      return normed.swiGluMlp(sharedWeights);
    });

    using gateLogitsBuf = normed.linear(this.tensors.get(`${pfx}.mlp.gate.weight`)!);
    using gateSigmoid = gateLogitsBuf.sigmoid();

    using topkInputHolder = new UsingHolder<Tensor>(undefined!);
    const eScoreBias = this.tensors.get(`${pfx}.mlp.gate.e_score_correction_bias`);
    if (eScoreBias) {
      topkInputHolder.replace(gateSigmoid.add(eScoreBias));
    } else {
      topkInputHolder.replace(gateSigmoid);
    }

    if (nGroup > 1) {
      // removed untested dead code that supported this, just guard
      throw new Error(`mlpSparse: nGroup > 1 is not supported (got nGroup=${nGroup})`);
    }

    const topkResult = topkInputHolder.value.topk(topK, numExperts);
    using _topkValues = topkResult.values;
    using topkIndices = topkResult.indices;

    using normalizedWeightsStream = this.glm.withStream(() => {
      using selectedScores = gateSigmoid.gather(topkIndices, topK, numExperts, BS);
      return selectedScores.rowNormalize(cfg.routedScalingFactor, cfg.normTopkProb);
    });

    const count = BS * topK;
    using topkIndicesFlat = topkIndices.reshape([count]);

    const gateWeights = this.getExpertWeights(pfx, "gate_proj");
    const upWeights = this.getExpertWeights(pfx, "up_proj");
    const downWeights = this.getExpertWeights(pfx, "down_proj");

    using downOut = normed.swiGluMlpMoe({ gate: gateWeights, up: upWeights, down: downWeights }, topkIndicesFlat, topK, count, moeIntermediate, hs, pfx);

    normalizedWeightsStream.streamWaitEvent();
    using normalizedWeights = normalizedWeightsStream.result;
    using normalizedWeightsFlat = normalizedWeights.reshape([count]);
    using routedOut = downOut.scatterAddRows(normalizedWeightsFlat, topK, BS);

    sharedDownBufStream.streamWaitEvent();
    using sharedDownBuf = sharedDownBufStream.result;

    using result = routedOut.add(sharedDownBuf, BS * hs);
    return result.reshape([BS, hs]);
  }

  private mlaLayer(cos: Tensor, sin: Tensor, normed: Tensor, residual: Tensor, layerIdx: number, state: ExecutionState): { normed: Tensor, residual: Tensor } {
    const cfg = this.cfg;
    const nHeads = cfg.numAttentionHeads;
    const kvLoraRank = cfg.kvLoraRank;
    const qkRopeDim = cfg.qkRopeHeadDim;
    const pfx = `${Glm51Model.WEIGHT_PREFIX}${layerIdx}.self_attn`;
    const batchSize = state.batchSize;
    const BS = state.totalTokens;
    const B = state.isDecode ? batchSize : 1;
    const S = state.isDecode ? 1 : state.totalTokens;

    using kvcache = this.glm.withStream(() => {
      using kPeRopeStream = this.glm.withStream(() => {
        using kPeRaw = normed.linear(this.tensors.get(`${pfx}.k_pe_proj.weight`)!);
        return kPeRaw.applyRotaryPosEmb(cos, sin, qkRopeDim, 1, S, B, 1, cfg.ropeInterleave)
      });
      using kPeRope = kPeRopeStream.result;

      using ckv = normed.linear(this.tensors.get(`${pfx}.ckv_proj.weight`)!);
      using ckvNormed = ckv.rmsnorm(this.tensors.get(`${pfx}.kv_a_layernorm.weight`)!, cfg.rmsNormEps);

      kPeRopeStream.streamWaitEvent();
      state.mlaKvCacheAppend(ckvNormed, kPeRope, layerIdx, kvLoraRank, qkRopeDim);

      // The physical slots are derived per-layer from sharedTopk at attention
      // time (see topkSlots below); the topk arg here is vestigial.
      return state.sparseMlaPrepareCache(ckvNormed, kPeRope, undefined, layerIdx, kvLoraRank, qkRopeDim);
    });

    const dense = cfg.indexHeadDim === 0;
    const shared = cfg.indexerTypes[layerIdx] === "shared";
    const skipIndexer = dense || shared;

    // Indexer K: wk(normed) → layernorm → split → RoPE → concat → append to kData
    // Only 'full' layers have indexer weights; 'shared' layers reuse previous topk.
    using kvcacheIndex = skipIndexer
      ? undefined
      : this.glm.withStream(() => {
        const idxRopeDim = qkRopeDim;
        const idxNopeDim = cfg.indexHeadDim - idxRopeDim;
        using idxKRaw = normed.linear(this.tensors.get(`${pfx}.indexer.wk.weight`)!);
        using idxKNormed = idxKRaw.layernorm(
          this.tensors.get(`${pfx}.indexer.k_norm.weight`)!,
          this.tensors.get(`${pfx}.indexer.k_norm.bias`)!,
          1e-6,
        );
        using idxKOut = idxNopeDim > 0
          ? (() => {
            using idxKPe = idxKNormed.slice(1, 0, idxRopeDim);
            using idxKNope = idxKNormed.slice(1, idxRopeDim, idxNopeDim);
            using idxKPeRope = idxKPe.applyRotaryPosEmb(cos, sin, idxRopeDim, 1, S, B, 1, cfg.indexerRopeInterleave);
            return idxKPeRope.cat([idxKNope], 1);
          })()
          : idxKNormed.applyRotaryPosEmb(cos, sin, idxRopeDim, 1, S, B, 1, cfg.indexerRopeInterleave);

        state.indexerKvCacheAppend(idxKOut, layerIdx, cfg.indexHeadDim);
      });

    using qResidBuf = normed.linear(this.tensors.get(`${pfx}.q_a_proj.weight`)!);
    using qNormed = qResidBuf.rmsnorm(this.tensors.get(`${pfx}.q_a_layernorm.weight`)!, cfg.rmsNormEps);

    // Indexer q: wq_b(qNormed) → ropeTranspose → [BS, indexNHeads, indexHeadDim]
    // Only 'full' layers compute indexer Q; 'shared' layers reuse previous topk.
    using idxQStream = skipIndexer
      ? undefined
      : this.glm.withStream(() => {
        const idxNHeads = cfg.indexNHeads;
        const idxHeadDim = cfg.indexHeadDim;
        const idxTopk = cfg.indexTopk;

        using idxWeights = normed.linear(this.tensors.get(`${pfx}.indexer.weights_proj.weight`)!);
        idxWeights.scaleInPlace(Math.sqrt(1.0 / idxNHeads), BS * idxNHeads);

        using idxQLin = qNormed.linear(this.tensors.get(`${pfx}.indexer.wq_b.weight`)!);
        using idxQ = idxQLin.ropeTranspose(cos, sin, qkRopeDim, cfg.indexHeadDim, cfg.indexNHeads, S, B, cfg.indexHeadDim, cfg.indexerRopeInterleave);

        kvcacheIndex?.streamWaitEvent();

        // Store the raw indexer top-k (token positions); slots are derived
        // per-layer/per-mode below and in the gather (slotsReady).
        return state.indexerTopk(
          idxQ, layerIdx, idxWeights,
          Math.pow(idxHeadDim, -0.5), idxTopk,
        );
      });

    using qPeRStream = this.glm.withStream(() => {
      using qPeLin = qNormed.linear(this.tensors.get(`${pfx}.q_pe_proj.weight`)!);
      return qPeLin.ropeTranspose(cos, sin, qkRopeDim, qkRopeDim, nHeads, S, B, qkRopeDim, cfg.ropeInterleave);
    });

    using qAbsorbedRStream = this.glm.withStream(() => {
      using qAbsorbedLin = qNormed.linear(this.tensors.get(`${pfx}.absorbed.weight`)!);
      return state.isDecode
        ? qAbsorbedLin.ropeTranspose(undefined!, undefined!, 0, kvLoraRank, nHeads, S, B, kvLoraRank)
        : qAbsorbedLin.ropeTranspose(cos, sin, 0, kvLoraRank, nHeads, S, B, kvLoraRank);
    });

    idxQStream?.streamWaitEvent();
    const topkResult = idxQStream?.result;
    using topkValues = topkResult?.values;
    using topkIndices = topkResult?.indices;
    const sparseSlots = cfg.indexHeadDim === 0
      ? undefined
      : state.sparseMlaSlots(layerIdx, topkIndices);

    using slots = sparseSlots?.slots;
    using slotsLength = sparseSlots?.length;
    using slotsStream = sparseSlots?.stream;

    qAbsorbedRStream.streamWaitEvent();
    qPeRStream.streamWaitEvent();
    kvcache.streamWaitEvent();

    using qAbsorbedR = qAbsorbedRStream.result;
    using qPeR = qPeRStream.result;
    using ckv = kvcache.result;

    using oProjBuf = new UsingHolder<Tensor>(undefined!);
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
          cfg.indexTopk, cfg.scaling,
        );
        tokenMajor = !state.isDecode;

        attnOut = sparseResult.o;
        lseBuf = sparseResult.lse;
      } else {
        // Dense MLA path (FlashInfer plan/run)
        const mlaResult = state.denseMla(qAbsorbedR, qPeR, layerIdx, cfg.scaling);
        attnOut = mlaResult.o;
        lseBuf = mlaResult.lse;
      }

      using _attnOut = attnOut;
      using _lseBuf = lseBuf;

      const vProj = this.tensors.get(`${pfx}.v_proj.weight`)!;
      using vExpanded = attnOut.mlaVExpand(vProj, S, B, lseBuf, undefined, undefined, undefined, tokenMajor);
      oProjBuf.replace(vExpanded.outputProj(this.tensors.get(`${pfx}.o_proj.weight`)!));
    }

    const attnResult = residual.fusedAddRmsnorm(oProjBuf.value, this.tensors.get(`${Glm51Model.WEIGHT_PREFIX}${layerIdx}.post_attention_layernorm.weight`)!, cfg.rmsNormEps);
    using attnNormed = attnResult.normed;
    using attnResidual = attnResult.residual;

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
    const mlpResult = attnResidual.fusedAddRmsnorm(downBuf, nextWeight, cfg.rmsNormEps);

    slotsStream?.streamWaitEvent();
    return { normed: mlpResult.normed, residual: mlpResult.residual };
  }

  forwardModel(state: ExecutionState): Tensor {
    const cfg = this.cfg;

    using rotaryEmbedding = this.glm.withStream(() => state.rotaryEmbedding(this.invFreq));

    const embedTable = this.tensors.get("model.embed_tokens.weight")!;

    using residual = new UsingHolder(state.embedding(embedTable));
    using normed = new UsingHolder(residual.value.rmsnorm(this.tensors.get(`${Glm51Model.WEIGHT_PREFIX}0.input_layernorm.weight`)!, cfg.rmsNormEps));

    rotaryEmbedding.streamWaitEvent();
    using cos = rotaryEmbedding.result.cos;
    using sin = rotaryEmbedding.result.sin;

    // The group slot cache lives on state.sharedSlots / sharedSlotsLength.
    // Callers that span multiple forwards (run loop, MTP) provide persistent
    // holders; otherwise fall back to forward-local ones.
    using _localSlots = state.sharedSlots ? undefined : new UsingHolder<Tensor>(undefined!);
    using _localLength = state.sharedSlotsLength ? undefined : new UsingHolder<Tensor>(undefined!);
    state.sharedSlots ??= _localSlots!;
    state.sharedSlotsLength ??= _localLength!;

    for (let i = 0; i < cfg.numHiddenLayers; i++) {
      const result = this.mlaLayer(cos, sin, normed.value, residual.value, i, state);
      normed.replace(result.normed);
      residual.replace(result.residual);
    }

    // Export the cache so it survives a caller's MTP tracking region (see mtp.ts).
    state.sharedSlots?.value?.removeTracking();
    state.sharedSlotsLength?.value?.removeTracking();
    return normed.detach().removeTracking();
  }

  forwardMtp(state: ExecutionState, previousHiddenState: Tensor, maskPos0?: boolean) {
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
    if (maskPos0) {
      using firstRow = embedding.narrow(0, 1);
      firstRow.fill(0, hs);
    }
    using enorm = embedding.rmsnorm(this.tensors.get(`${Glm51Model.WEIGHT_PREFIX}${cfg.numHiddenLayers}.enorm.weight`)!, cfg.rmsNormEps);
    hnormStream.streamWaitEvent();
    using hnorm = hnormStream.result;
    using cat = enorm.cat([hnorm], 1);

    using residual = cat.linear(this.tensors.get(`${Glm51Model.WEIGHT_PREFIX}${cfg.numHiddenLayers}.eh_proj.weight`)!);
    using normed = residual.rmsnorm(this.tensors.get(`${Glm51Model.WEIGHT_PREFIX}${cfg.numHiddenLayers}.input_layernorm.weight`)!, cfg.rmsNormEps);

    rotaryEmbedding.streamWaitEvent();
    using cos = rotaryEmbedding.result.cos;
    using sin = rotaryEmbedding.result.sin;

    const layerIdx = cfg.numHiddenLayers;
    using _localSlots = state.sharedSlots ? undefined : new UsingHolder<Tensor>(undefined!);
    using _localLength = state.sharedSlotsLength ? undefined : new UsingHolder<Tensor>(undefined!);
    state.sharedSlots ??= _localSlots!;
    state.sharedSlotsLength ??= _localLength!;
    const result = this.mlaLayer(cos, sin, normed, residual, layerIdx, state);
    using _residual = result.residual;

    // Export the cache (slots + length) so it survives the caller's MTP tracking
    // region — otherwise startTracking's dispose frees the tensors the holders
    // still point to, causing a use-after-free on the next pass.
    state.sharedSlots?.value?.removeTracking();
    state.sharedSlotsLength?.value?.removeTracking();

    // Return shared_head.norm(residual) so the recycled seed for the next MTP
    // step is already normed — matches sglang Glm4MoeModelNextN and vLLM v1
    // deepseek_mtp which both recycle the post-shared_head-norm state.
    return result.normed.removeTracking();
  }
}
