import fs from "node:fs";
import path from "node:path";
import type { ChatCache } from "./chat_model";
import { ChatModel, SamplingParams } from "./chat_model";
import { DeviceOps, TensorParallelism } from "./device_ops";
import { bf16BytesToF32, f32ToBf16Bytes, GlmOps } from "./glm_ops";
import { resolveModelPath } from "./model_path";
import { ExecutionState, ExecutionWorkspace, PagedKVCache } from "./paged_kv";
import { SafeTensorFile, type TensorMeta } from "./safetensors";
import { Tensor } from "./tensor";
import { UsingHolder } from "./using-holder";

export type { SamplingParams };
export { ExecutionState as BatchState };

const GLM51_REPO = "zai-org/GLM-5.1";

export interface Glm51Config {
  hiddenSize: number;
  intermediateSize: number;
  moeIntermediateSize: number;
  numHiddenLayers: number;
  rmsNormEps: number;
  vocabSize: number;
  tieWordEmbeddings: boolean;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  headDim: number;
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
  ropeTheta: number;
  scaling: number;
  mlpLayerTypes: string[];
  numDenseMlpLayers: number;
  firstSparseMlpLayer: number;
  numKeyValueGroups: number;
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
    ropeTheta: raw.rope_parameters?.rope_theta ?? raw.rope_theta ?? 1000000,
    scaling: Math.pow(qkHeadDim, -0.5),
    mlpLayerTypes,
    numDenseMlpLayers,
    firstSparseMlpLayer: firstSparseMlpLayer >= 0 ? firstSparseMlpLayer : numDenseMlpLayers,
    numKeyValueGroups: raw.num_attention_heads / raw.num_key_value_heads,
  };
}

class Glm51ChatCache implements ChatCache {
  constructor(
    public readonly pagedKV: PagedKVCache,
  ) {}

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

export class Glm51Model extends ChatModel {
  static readonly WEIGHT_PREFIX = "model.layers.";
  readonly eosIds: Set<number>;
  cfg: Glm51Config;
  maxBatch: number;
  maxSeqLen: number;
  invFreq: Tensor;

  private constructor(glm: DeviceOps, config: Glm51Config, maxBatch: number, maxSeqLen: number) {
    super(glm);
    this.cfg = config;
    this.maxBatch = maxBatch;
    this.maxSeqLen = maxSeqLen;
    this.eosIds = new Set(config.vocabSize > 200000 ? [154820, 154827, 154829] : [151645, 151643]);

    const ropeDim = config.qkRopeHeadDim;
    const halfRopeDim = ropeDim / 2;
    const invFreqF32 = new Float32Array(halfRopeDim);
    for (let i = 0; i < halfRopeDim; i++) {
      invFreqF32[i] = 1.0 / Math.pow(config.ropeTheta, (2 * i) / ropeDim);
    }
    this.invFreq = this.alloc([halfRopeDim], "BF16", "invFreq");
    this.invFreq.h2d(f32ToBf16Bytes(invFreqF32));
  }

  static fromPretrained(glm: DeviceOps, repoId: string = GLM51_REPO, maxBatch = 1, maxSeqLen = 4096): Glm51Model {
    const modelDir = resolveModelPath(repoId);
    const config = loadConfig(modelDir);
    const model = new Glm51Model(glm, config, maxBatch, maxSeqLen);
    model.loadWeights(modelDir);
    return model;
  }

  private weightParallelism(name: string): TensorParallelism {
    if (name === "lm_head.weight") return TensorParallelism.Column;
    if (name === "model.embed_tokens.weight") return TensorParallelism.Row;
    const pfx = Glm51Model.WEIGHT_PREFIX;
    if (name.endsWith(".self_attn.q_a_proj.weight") ||
        name.endsWith(".self_attn.kv_a_proj_with_mqa.weight") ||
        name.endsWith(".mlp.gate_proj.weight") ||
        name.endsWith(".mlp.up_proj.weight") ||
        name.endsWith(".mlp.gate.weight") ||
        (name.startsWith(pfx) && name.includes(".mlp.experts.") && name.endsWith(".gate_proj.weight")) ||
        (name.startsWith(pfx) && name.includes(".mlp.experts.") && name.endsWith(".up_proj.weight")) ||
        name.endsWith(".mlp.shared_experts.gate_proj.weight") ||
        name.endsWith(".mlp.shared_experts.up_proj.weight")) return TensorParallelism.Column;
    if (name.endsWith(".self_attn.o_proj.weight") ||
        name.endsWith(".mlp.down_proj.weight") ||
        (name.startsWith(pfx) && name.includes(".mlp.experts.") && name.endsWith(".down_proj.weight")) ||
        name.endsWith(".mlp.shared_experts.down_proj.weight")) return TensorParallelism.Row;
    return TensorParallelism.Replicated;
  }

  protected loadTensor(name: string, meta: TensorMeta, st: SafeTensorFile, mmapPtr: number): void {
    if (name.includes(".indexer.")) return;
    if (name.endsWith(".self_attn.q_b_proj.weight") ||
        name.endsWith(".self_attn.kv_b_proj.weight") ||
        name.endsWith(".self_attn.kv_a_proj_with_mqa.weight")) {
      this.loadMlaWeight(name, meta, st, mmapPtr);
      return;
    }

    const par = this.weightParallelism(name);
    const gemmaNormSuffixes = [
      "input_layernorm.weight",
      "post_attention_layernorm.weight",
      "q_a_layernorm.weight",
      "kv_a_layernorm.weight",
    ];
    const isGemmaNorm = name === "model.norm.weight" ||
      gemmaNormSuffixes.some(s => name.endsWith(s));

    if (meta.dtype === "F32") {
      const numElements = meta.shape.reduce((a, b) => a * b, 1);
      const tensor = this.alloc(meta.shape, "BF16", name, par);
      const f32Bytes = st.readTensor(name);
      const f32Arr = new Float32Array(f32Bytes.buffer, f32Bytes.byteOffset, numElements);
      if (isGemmaNorm) {
        for (let i = 0; i < numElements; i++) f32Arr[i] += 1.0;
      }
      tensor.h2d(f32ToBf16Bytes(f32Arr));
    } else if (isGemmaNorm) {
      const numElements = meta.shape.reduce((a, b) => a * b, 1);
      const tensor = this.alloc(meta.shape, "BF16", name, par);
      const rawBytes = st.readTensor(name);
      const f32Arr = new Float32Array(numElements);
      for (let i = 0; i < numElements; i++) {
        const u16 = rawBytes.readUInt16LE(i * 2);
        const u32 = u16 << 16;
        f32Arr[i] = (new Float32Array(new Uint32Array([u32]).buffer)[0]) + 1.0;
      }
      tensor.h2d(f32ToBf16Bytes(f32Arr));
    } else {
      const dtype = meta.dtype === "F32" ? "F32" : meta.dtype;
      const tensor = this.alloc(meta.shape, dtype, name, par);
      const offset = st.dataStart + meta.dataOffsets[0];
      tensor.mmapLoad(mmapPtr, offset, tensor.bytes);

      if (this.cfg.tieWordEmbeddings && name === "model.embed_tokens.weight" && !this.tensors.has("lm_head.weight")) {
        const lmHead = this.alloc(meta.shape, dtype, "lm_head.weight", TensorParallelism.Column);
        lmHead.mmapLoad(mmapPtr, offset, lmHead.bytes);
      }
    }
  }

  private loadMlaWeight(name: string, meta: TensorMeta, st: SafeTensorFile, mmapPtr: number): void {
    const cfg = this.cfg;
    const nHeads = cfg.numAttentionHeads;
    const qkNopeDim = cfg.qkNopeHeadDim;
    const qkRopeDim = cfg.qkRopeHeadDim;
    const vHeadDim = cfg.vHeadDim;
    const kvLoraRank = cfg.kvLoraRank;
    const qLoraRank = cfg.qLoraRank;
    const qkHeadDim = cfg.qkHeadDim;
    const colPar = TensorParallelism.Column;

    const rawBytes = st.readTensor(name);
    const numElements = meta.shape.reduce((a, b) => a * b, 1);
    const f32Arr = new Float32Array(numElements);
    if (meta.dtype === "BF16") {
      for (let i = 0; i < numElements; i++) {
        const u16 = rawBytes.readUInt16LE(i * 2);
        const u32 = u16 << 16;
        f32Arr[i] = new Float32Array(new Uint32Array([u32]).buffer)[0];
      }
    } else {
      f32Arr.set(new Float32Array(rawBytes.buffer, rawBytes.byteOffset, numElements));
    }

    const inDim = meta.shape[1];

    if (name.endsWith(".q_b_proj.weight")) {
      const qNopeProjName = name.replace(".q_b_proj.weight", ".q_nope_proj.weight");
      const qPeProjName = name.replace(".q_b_proj.weight", ".q_pe_proj.weight");
      const qNopeProjF32 = new Float32Array(nHeads * qkNopeDim * qLoraRank);
      const qPeProjF32 = new Float32Array(nHeads * qkRopeDim * qLoraRank);
      for (let h = 0; h < nHeads; h++) {
        for (let i = 0; i < qkNopeDim; i++) {
          for (let j = 0; j < qLoraRank; j++) {
            qNopeProjF32[(h * qkNopeDim + i) * qLoraRank + j] = f32Arr[(h * qkHeadDim + i) * inDim + j];
          }
        }
        for (let i = 0; i < qkRopeDim; i++) {
          for (let j = 0; j < qLoraRank; j++) {
            qPeProjF32[(h * qkRopeDim + i) * qLoraRank + j] = f32Arr[(h * qkHeadDim + qkNopeDim + i) * inDim + j];
          }
        }
      }
      const qNopeProj = this.alloc([nHeads * qkNopeDim, qLoraRank], "BF16", qNopeProjName, colPar);
      qNopeProj.h2d(f32ToBf16Bytes(qNopeProjF32));
      const qPeProj = this.alloc([nHeads * qkRopeDim, qLoraRank], "BF16", qPeProjName, colPar);
      qPeProj.h2d(f32ToBf16Bytes(qPeProjF32));
    } else if (name.endsWith(".kv_b_proj.weight")) {
      const kNopeProjName = name.replace(".kv_b_proj.weight", ".k_nope_proj.weight");
      const vProjName = name.replace(".kv_b_proj.weight", ".v_proj.weight");
      const kvExpandedDim = qkNopeDim + vHeadDim;
      const kNopeProjF32 = new Float32Array(nHeads * qkNopeDim * kvLoraRank);
      const vProjF32 = new Float32Array(nHeads * vHeadDim * kvLoraRank);
      for (let h = 0; h < nHeads; h++) {
        for (let i = 0; i < qkNopeDim; i++) {
          for (let j = 0; j < kvLoraRank; j++) {
            kNopeProjF32[(h * qkNopeDim + i) * kvLoraRank + j] = f32Arr[(h * kvExpandedDim + i) * inDim + j];
          }
        }
        for (let i = 0; i < vHeadDim; i++) {
          for (let j = 0; j < kvLoraRank; j++) {
            vProjF32[(h * vHeadDim + i) * kvLoraRank + j] = f32Arr[(h * kvExpandedDim + qkNopeDim + i) * inDim + j];
          }
        }
      }
      const kNopeProj = this.alloc([nHeads * qkNopeDim, kvLoraRank], "BF16", kNopeProjName, colPar);
      kNopeProj.h2d(f32ToBf16Bytes(kNopeProjF32));
      const vProj = this.alloc([nHeads * vHeadDim, kvLoraRank], "BF16", vProjName, colPar);
      vProj.h2d(f32ToBf16Bytes(vProjF32));
    } else if (name.endsWith(".kv_a_proj_with_mqa.weight")) {
      const ckvProjName = name.replace(".kv_a_proj_with_mqa.weight", ".ckv_proj.weight");
      const kPeProjName = name.replace(".kv_a_proj_with_mqa.weight", ".k_pe_proj.weight");
      const ckvProjF32 = new Float32Array(kvLoraRank * inDim);
      const kPeProjF32 = new Float32Array(qkRopeDim * inDim);
      for (let i = 0; i < kvLoraRank; i++) {
        for (let j = 0; j < inDim; j++) {
          ckvProjF32[i * inDim + j] = f32Arr[i * inDim + j];
        }
      }
      for (let i = 0; i < qkRopeDim; i++) {
        for (let j = 0; j < inDim; j++) {
          kPeProjF32[i * inDim + j] = f32Arr[(kvLoraRank + i) * inDim + j];
        }
      }
      const ckvProj = this.alloc([kvLoraRank, inDim], "BF16", ckvProjName, colPar);
      ckvProj.h2d(f32ToBf16Bytes(ckvProjF32));
      const kPeProj = this.alloc([qkRopeDim, inDim], "BF16", kPeProjName, colPar);
      kPeProj.h2d(f32ToBf16Bytes(kPeProjF32));
    }
  }

  protected tieWeights(): void {
    if (this.cfg.tieWordEmbeddings && !this.tensors.has("lm_head.weight")) {
      const embedTensor = this.tensors.get("model.embed_tokens.weight")!;
      this.tensors.set("lm_head.weight", embedTensor);
    }

    const cfg = this.cfg;
    const nHeads = cfg.numAttentionHeads;
    const qkNopeDim = cfg.qkNopeHeadDim;
    const kvLoraRank = cfg.kvLoraRank;
    const qLoraRank = cfg.qLoraRank;

    for (let i = 0; i < cfg.numHiddenLayers; i++) {
      const pfx = `${Glm51Model.WEIGHT_PREFIX}${i}.self_attn`;
      const kNopeProj = this.tensors.get(`${pfx}.k_nope_proj.weight`);
      const qNopeProj = this.tensors.get(`${pfx}.q_nope_proj.weight`);
      if (!kNopeProj || !qNopeProj) continue;

      const absorbedName = `${pfx}.absorbed.weight`;
      const wAbsorbed = this.alloc([nHeads * kvLoraRank, qLoraRank], "BF16", absorbedName, TensorParallelism.Column);
      this.glm.bmm(wAbsorbed.data, kNopeProj.data, qNopeProj.data, 1.0, 0.0, nHeads, kvLoraRank, qLoraRank, qkNopeDim, 1, 0);
    }
  }

  createChatCache(maxPages = 256): ChatCache {
    const cfg = this.cfg;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    return new PagedKVCache(this.glm, nKv, hd, cfg.numHiddenLayers, maxPages, this.maxBatch, 16, cfg.kvLoraRank, cfg.qkRopeHeadDim);
  }

  private mlpDense(normed: Tensor, pfx: string, BS: number): Tensor {
    using gateBuf = normed.linear(this.tensors.get(`${pfx}.mlp.gate_proj.weight`)!, BS);
    using upBuf = normed.linear(this.tensors.get(`${pfx}.mlp.up_proj.weight`)!, BS);
    using siluBuf = gateBuf.siluAndMul(gateBuf, upBuf, this.cfg.intermediateSize, BS);
    return siluBuf.linear(this.tensors.get(`${pfx}.mlp.down_proj.weight`)!, BS);
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

    using gateLogitsBuf = normed.linear(this.tensors.get(`${pfx}.mlp.gate.weight`)!, BS);
    const gateLogitsBytes = Buffer.alloc(BS * numExperts * 2);
    gateLogitsBuf.d2h(gateLogitsBytes);
    const gateLogitsF32 = bf16BytesToF32(gateLogitsBytes);

    const eScoreBias = this.tensors.get(`${pfx}.mlp.gate.e_score_correction_bias`);
    let biasF32: Float32Array | null = null;
    if (eScoreBias) {
      const biasBytes = Buffer.alloc(numExperts * 2);
      eScoreBias.d2h(biasBytes);
      biasF32 = bf16BytesToF32(biasBytes);
    }

    const expertIndices: number[][] = [];
    const expertWeights: number[][] = [];
    for (let b = 0; b < BS; b++) {
      const logits = new Float32Array(numExperts);
      for (let e = 0; e < numExperts; e++) {
        logits[e] = 1.0 / (1.0 + Math.exp(-gateLogitsF32[b * numExperts + e]));
      }

      const logitsCorrected = new Float32Array(numExperts);
      for (let e = 0; e < numExperts; e++) {
        logitsCorrected[e] = logits[e] + (biasF32 ? biasF32[e] : 0);
      }

      if (nGroup > 1) {
        const groupScores = new Float32Array(nGroup);
        for (let g = 0; g < nGroup; g++) {
          const top2: number[] = [];
          for (let j = 0; j < expertsPerGroup; j++) {
            const score = logitsCorrected[g * expertsPerGroup + j];
            if (top2.length < 2) { top2.push(score); top2.sort((a, b) => b - a); }
            else if (score > top2[1]) { top2[1] = score; top2.sort((a, b) => b - a); }
          }
          groupScores[g] = top2[0] + top2[1];
        }

        const groupIdx: number[] = [];
        for (let g = 0; g < nGroup; g++) groupIdx.push(g);
        groupIdx.sort((a, b) => groupScores[b] - groupScores[a]);
        const selectedGroups = new Set(groupIdx.slice(0, topkGroup));

        for (let e = 0; e < numExperts; e++) {
          const g = Math.floor(e / expertsPerGroup);
          if (!selectedGroups.has(g)) logitsCorrected[e] = 0;
        }
      }

      const scored: { val: number; idx: number }[] = [];
      for (let e = 0; e < numExperts; e++) {
        scored.push({ val: logitsCorrected[e], idx: e });
      }
      scored.sort((a, b) => b.val - a.val);

      const selected: number[] = [];
      const weights: number[] = [];
      let weightSum = 0;
      for (let k = 0; k < topK; k++) {
        selected.push(scored[k].idx);
        weights.push(logits[scored[k].idx]);
        weightSum += logits[scored[k].idx];
      }
      if (cfg.normTopkProb && weightSum > 0) {
        for (let k = 0; k < topK; k++) weights[k] /= weightSum;
      }
      for (let k = 0; k < topK; k++) weights[k] *= cfg.routedScalingFactor;
      expertIndices.push(selected);
      expertWeights.push(weights);
    }

    using routedOut = this.alloc([BS, hs], "BF16");
    routedOut.fill(0, BS * hs);

    const scaleF32 = new Float32Array(BS);
    const scaleBuf = this.alloc([BS], "BF16");

    for (let e = 0; e < numExperts; e++) {
      let anySelected = false;
      for (let b = 0; b < BS; b++) {
        if (expertIndices[b].includes(e)) { anySelected = true; break; }
      }
      if (!anySelected) continue;

      using expertGateBuf = normed.linear(this.tensors.get(`${pfx}.mlp.experts.${e}.gate_proj.weight`)!, BS);
      using expertUpBuf = normed.linear(this.tensors.get(`${pfx}.mlp.experts.${e}.up_proj.weight`)!, BS);
      using expertSiluBuf = expertGateBuf.siluAndMul(expertGateBuf, expertUpBuf, moeIntermediate, BS);
      using expertDownBuf = expertSiluBuf.linear(this.tensors.get(`${pfx}.mlp.experts.${e}.down_proj.weight`)!, BS);

      for (let b = 0; b < BS; b++) {
        const kIdx = expertIndices[b].indexOf(e);
        scaleF32[b] = kIdx !== -1 ? expertWeights[b][kIdx] : 0;
      }
      scaleBuf.h2d(f32ToBf16Bytes(scaleF32));

      routedOut.rowScaleAdd(expertDownBuf, scaleBuf, BS, hs);
    }
    scaleBuf[Symbol.dispose]();

    using sharedGateBuf = normed.linear(this.tensors.get(`${pfx}.mlp.shared_experts.gate_proj.weight`)!, BS);
    using sharedUpBuf = normed.linear(this.tensors.get(`${pfx}.mlp.shared_experts.up_proj.weight`)!, BS);
    using sharedSiluBuf = sharedGateBuf.siluAndMul(sharedGateBuf, sharedUpBuf, moeIntermediate, BS);
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

    using qResidBuf = normed.linear(this.tensors.get(`${pfx}.q_a_proj.weight`)!, BS);
    using qNormed = qResidBuf.rmsnorm(this.tensors.get(`${pfx}.q_a_layernorm.weight`)!, cfg.rmsNormEps, cfg.qLoraRank, BS);

    using qAbsorbedLin = qNormed.linear(this.tensors.get(`${pfx}.absorbed.weight`)!, BS);
    using qPeLin = qNormed.linear(this.tensors.get(`${pfx}.q_pe_proj.weight`)!, BS);

    using ckv = normed.linear(this.tensors.get(`${pfx}.ckv_proj.weight`)!, BS);
    using kPeRaw = normed.linear(this.tensors.get(`${pfx}.k_pe_proj.weight`)!, BS);
    using ckvNormed = ckv.rmsnorm(this.tensors.get(`${pfx}.kv_a_layernorm.weight`)!, cfg.rmsNormEps, kvLoraRank, BS);

    const pagedKV = state.cache.getPagedKV();
    const useMla = pagedKV.ckvData.length > 0;

    using attnOut = new UsingHolder<Tensor>(undefined!);

    if (useMla && state.isDecode) {
      const qAbsorbedR = qAbsorbedLin.ropeTranspose(undefined!, undefined!, 0, kvLoraRank, nHeads, S, B, kvLoraRank);
      using _qAbsorbedR = qAbsorbedR;
      const qPeR = qPeLin.ropeTranspose(undefined!, undefined!, 0, qkRopeDim, nHeads, S, B, qkRopeDim);
      using _qPeR = qPeR;
      ws.mlaKvCacheAppend(ckvNormed, kPeRaw, pagedKV, layerIdx, batchSize, kvLoraRank, qkRopeDim);
      this.glm.mlaDecodePlan(
        ws.floatWs, 128 * 1024 * 1024,
        ws.intWs, ws.pinnedIntWs, 8 * 1024 * 1024,
        ws.mlaDecodePlanInfo,
        ws.indptrH,
        batchSize, nHeads, pagedKV.pageSize, false
      );
      attnOut.replace(ws.mlaDecodePaged(qAbsorbedR, qPeR, pagedKV, layerIdx, batchSize, nHeads, kvLoraRank, qkRopeDim, cfg.scaling));
    } else if (useMla && !state.isDecode) {
      using rotaryEmbedding = this.glm.withStream(() => this.invFreq.rotaryEmbedding(ws.positionIds, qkRopeDim / 2, B, S));
      using cos = rotaryEmbedding.result.cos;
      using sin = rotaryEmbedding.result.sin;

      const qAbsorbedR = qAbsorbedLin.ropeTranspose(cos, sin, 0, kvLoraRank, nHeads, S, B, kvLoraRank);
      using _qAbsorbedR = qAbsorbedR;
      const qPeFinal = qPeLin.ropeTranspose(cos, sin, qkRopeDim, qkRopeDim, nHeads, S, B, qkRopeDim);
      using _qPeFinal = qPeFinal;
      const kPeRope = kPeRaw.applyRotaryPosEmb(cos, sin, qkRopeDim, 1, S, B, 1);
      using _kPeRope = kPeRope;

      this.glm.mlaPrefillPlan(
        ws.floatWs, 128 * 1024 * 1024,
        ws.intWs, ws.pinnedIntWs, 8 * 1024 * 1024,
        ws.mlaPrefillPlanInfo,
        ws.qoIndptrD, ws.indptrH,
        ws.kvLenH,
        batchSize, nHeads, kvLoraRank, 1
      );
      ws.mlaKvCacheAppend(ckvNormed, kPeRope, pagedKV, layerIdx, batchSize, kvLoraRank, qkRopeDim);
      rotaryEmbedding.streamWaitEvent();
      attnOut.replace(ws.mlaPrefillPaged(qAbsorbedR, qPeFinal, pagedKV, layerIdx, totalTokens, batchSize, nHeads, kvLoraRank, qkRopeDim, cfg.scaling));
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

    let logitsBuf: Tensor;
    if (state.isDecode) {
      logitsBuf = normed.value.linear(this.tensors.get("lm_head.weight")!, batchSize);
    } else {
      using hiddenLast = normed.value.indexSelect(ws.lastIdx, hs, batchSize);
      logitsBuf = hiddenLast.linear(this.tensors.get("lm_head.weight")!, batchSize);
    }

    return logitsBuf.removeTracking();
  }
}
