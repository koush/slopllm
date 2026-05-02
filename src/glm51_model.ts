import fs from "node:fs";
import path from "node:path";
import type { ChatCache } from "./chat_model";
import { ChatModel, CommonModelConfig, SamplingParams } from "./chat_model";
import { DeviceOps, TensorParallelism } from "./device_ops";
import { bf16BytesToF32, f32ToBf16Bytes, getNativeAddon, GlmOps } from "./glm_ops";
import { resolveModelPath } from "./model_path";
import { ExecutionState, ExecutionWorkspace, PagedKVCache } from "./paged_kv";
import { SafeTensorFile, type TensorMeta } from "./safetensors";
import { Tensor } from "./tensor";
import { UsingHolder } from "./using-holder";

export type { SamplingParams };
export { ExecutionState as BatchState };

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
  mlpLayerTypes: string[];
  numDenseMlpLayers: number;
  firstSparseMlpLayer: number;
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
    mlpLayerTypes,
    numDenseMlpLayers,
    firstSparseMlpLayer: firstSparseMlpLayer >= 0 ? firstSparseMlpLayer : numDenseMlpLayers,
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
  private moeAuxReady = false;

  private constructor(glm: DeviceOps, config: Glm51Config, maxBatch: number, maxSeqLen: number) {
    super(glm);
    this.cfg = config;
    this.maxBatch = maxBatch;
    this.maxSeqLen = maxSeqLen;
    this.eosIds = new Set(config.vocabSize > 200000 ? [154820, 154827, 154829] : [151645, 151643]);
    this.invFreq = this.initInvFreq(config.qkRopeHeadDim, config.ropeTheta);
  }

  static fromPretrained(glm: DeviceOps, repoIdOrDir: string = GLM51_MODEL_DIR, maxBatch = 1, maxSeqLen = 4096): Glm51Model {
    const modelDir = fs.existsSync(repoIdOrDir) ? repoIdOrDir : resolveModelPath(repoIdOrDir);
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
        name.endsWith(".mlp.shared_experts.up_proj.weight") ||
        // NVFP4 scale tensors follow same parallelism as their weight
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
        // NVFP4 scale tensors follow same parallelism as their weight
        name.endsWith(".down_proj.weight_weight_scale") ||
        (name.startsWith(pfx) && name.includes(".mlp.experts.") && name.endsWith(".down_proj.weight_weight_scale")) ||
        name.endsWith(".mlp.shared_experts.down_proj.weight_weight_scale")) return TensorParallelism.Row;
    return TensorParallelism.Replicated;
  }

  protected loadTensor(name: string, meta: TensorMeta, st: SafeTensorFile, mmapPtr: number): void {
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
      this.loadMlaWeight(name, meta, st, mmapPtr);
      return;
    }

    const par = this.weightParallelism(name);

    if (meta.dtype === "F32" && !name.endsWith(".weight_scale_2")) {
      const numElements = meta.shape.reduce((a, b) => a * b, 1);
      const tensor = this.alloc(meta.shape, "BF16", storeName, par);
      const f32Bytes = st.readTensor(name);
      const f32Arr = new Float32Array(f32Bytes.buffer, f32Bytes.byteOffset, numElements);
      tensor.h2d(f32ToBf16Bytes(f32Arr));

      if (this.cfg.tieWordEmbeddings && name === "model.embed_tokens.weight" && !this.tensors.has("lm_head.weight")) {
        const lmHead = this.alloc(meta.shape, "BF16", "lm_head.weight", TensorParallelism.Column);
        const embedOffset = st.dataStart + meta.dataOffsets[0];
        lmHead.mmapLoad(mmapPtr, embedOffset, lmHead.bytes);
      }
    } else {
      const dtype = meta.dtype;
      const tensor = this.alloc(meta.shape, dtype, storeName, par);
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
    this.tieEmbeddingToLmHead("model.embed_tokens.weight");

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

    this.initMoeAuxBuffers();
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

  private initMoeAuxBuffers(): void {
    if (this.moeAuxReady) return;
    this.moeAuxReady = true;
    const cfg = this.cfg;
    const numExperts = cfg.nRoutedExperts;
    const topK = cfg.numExpertsPerTok;

    for (let i = 0; i < cfg.numHiddenLayers; i++) {
      const pfx = `${Glm51Model.WEIGHT_PREFIX}${i}`;
      if (!this.tensors.has(`${pfx}.mlp.experts.0.gate_proj.weight`)) continue;

      const isNvfp4 = this.tensors.get(`${pfx}.mlp.experts.0.gate_proj.weight`)?.type === "U8";

      for (const proj of ["gate_proj", "up_proj", "down_proj"]) {
        const ptrs = new BigInt64Array(numExperts);
        for (let e = 0; e < numExperts; e++) {
          ptrs[e] = BigInt(this.tensors.get(`${pfx}.mlp.experts.${e}.${proj}.weight`)!.data);
        }
        const name = `__moe_ptrs.${pfx}.${proj}`;
        const buf = this.alloc([numExperts], "I64", name);
        buf.h2d(Buffer.from(ptrs.buffer));

        if (isNvfp4) {
          const scalePtrs = new BigInt64Array(numExperts);
          const scale2Ptrs = new BigInt64Array(numExperts);
          for (let e = 0; e < numExperts; e++) {
            scalePtrs[e] = BigInt(this.tensors.get(`${pfx}.mlp.experts.${e}.${proj}.weight_weight_scale`)!.data);
            scale2Ptrs[e] = BigInt(this.tensors.get(`${pfx}.mlp.experts.${e}.${proj}.weight_weight_scale_2`)!.data);
          }
          const scaleName = `__moe_nvfp4_ptrs.${pfx}.${proj}.weight_weight_scale`;
          const scale2Name = `__moe_nvfp4_ptrs.${pfx}.${proj}.weight_weight_scale_2`;
          const scaleBuf = this.alloc([numExperts], "I64", scaleName);
          scaleBuf.h2d(Buffer.from(scalePtrs.buffer));
          const scale2Buf = this.alloc([numExperts], "I64", scale2Name);
          scale2Buf.h2d(Buffer.from(scale2Ptrs.buffer));
        }
      }
    }

    const count = this.maxBatch * topK;
    const batchIdsArr = new Int32Array(count);
    for (let i = 0; i < count; i++) batchIdsArr[i] = Math.floor(i / topK);
    const batchIdsBuf = this.alloc([count], "I32", "__moe_batch_ids");
    batchIdsBuf.h2d(Buffer.from(batchIdsArr.buffer));

    const downBatchIdsArr = new Int32Array(count);
    for (let i = 0; i < count; i++) downBatchIdsArr[i] = i;
    const downBatchIdsBuf = this.alloc([count], "I32", "__moe_down_batch_ids");
    downBatchIdsBuf.h2d(Buffer.from(downBatchIdsArr.buffer));

    const allZeroIndices = new Int32Array(this.maxBatch).fill(0);
    const allZeroIdxBuf = this.alloc([this.maxBatch], "I32", "__moe_all_zero_idx");
    allZeroIdxBuf.h2d(Buffer.from(allZeroIndices.buffer));
  }

  private createExpertWeightPtrs(pfx: string, projection: string): Tensor {
    const name = `__moe_ptrs.${pfx}.${projection}`;
    const existing = this.tensors.get(name);
    if (existing) return existing;
    const numExperts = this.cfg.nRoutedExperts;
    const ptrs = new BigInt64Array(numExperts);
    for (let e = 0; e < numExperts; e++) {
      ptrs[e] = BigInt(this.tensors.get(`${pfx}.mlp.experts.${e}.${projection}.weight`)!.data);
    }
    const buf = this.alloc([numExperts], "I64", name);
    buf.h2d(Buffer.from(ptrs.buffer));
    return buf;
  }

  private createNvfp4ExpertPtrs(pfx: string, projection: string, suffix: string): Tensor {
    const name = `__moe_nvfp4_ptrs.${pfx}.${projection}.${suffix}`;
    const existing = this.tensors.get(name);
    if (existing) return existing;
    const numExperts = this.cfg.nRoutedExperts;
    const ptrs = new BigInt64Array(numExperts);
    for (let e = 0; e < numExperts; e++) {
      ptrs[e] = BigInt(this.tensors.get(`${pfx}.mlp.experts.${e}.${projection}.${suffix}`)!.data);
    }
    const buf = this.alloc([numExperts], "I64", name);
    buf.h2d(Buffer.from(ptrs.buffer));
    return buf;
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

    const isNvfp4 = this.tensors.get(`${pfx}.mlp.experts.0.gate_proj.weight`)?.type === "U8";

    using gateLogitsBuf = normed.linear(this.tensors.get(`${pfx}.mlp.gate.weight`)!, BS);
    using gateSigmoid = gateLogitsBuf.sigmoid();

    let topkInput: Tensor;
    using topkInputHolder = new UsingHolder<Tensor>(undefined!);
    const eScoreBias = this.tensors.get(`${pfx}.mlp.gate.e_score_correction_bias`);
    if (eScoreBias) {
      using biasBuf = ws.alloc([BS, numExperts], "BF16");
      const allZeroIdxBuf = this.tensors.get("__moe_all_zero_idx")!;
      getNativeAddon().indexSelect((this.glm as GlmOps).ctx, biasBuf.data, eScoreBias.data, allZeroIdxBuf.data, numExperts, BS);
      topkInput = gateSigmoid.add(biasBuf, BS * numExperts);
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

    if (isNvfp4) {
      const batchIdsBuf = this.tensors.get("__moe_batch_ids")!;
      const downBatchIdsBuf = this.tensors.get("__moe_down_batch_ids")!;

      const gateWeightPtrs = this.createExpertWeightPtrs(pfx, "gate_proj");
      const gateScalePtrs = this.createNvfp4ExpertPtrs(pfx, "gate_proj", "weight_weight_scale");
      const gateScale2Ptrs = this.createNvfp4ExpertPtrs(pfx, "gate_proj", "weight_weight_scale_2");
      const upWeightPtrs = this.createExpertWeightPtrs(pfx, "up_proj");
      const upScalePtrs = this.createNvfp4ExpertPtrs(pfx, "up_proj", "weight_weight_scale");
      const upScale2Ptrs = this.createNvfp4ExpertPtrs(pfx, "up_proj", "weight_weight_scale_2");
      const downWeightPtrs = this.createExpertWeightPtrs(pfx, "down_proj");
      const downScalePtrs = this.createNvfp4ExpertPtrs(pfx, "down_proj", "weight_weight_scale");
      const downScale2Ptrs = this.createNvfp4ExpertPtrs(pfx, "down_proj", "weight_weight_scale_2");

      using gateOut = normed.nvfp4MulMatId(normed, gateWeightPtrs, gateScalePtrs, gateScale2Ptrs, topkIndicesFlat, batchIdsBuf, count, moeIntermediate, hs);
      using upOut = normed.nvfp4MulMatId(normed, upWeightPtrs, upScalePtrs, upScale2Ptrs, topkIndicesFlat, batchIdsBuf, count, moeIntermediate, hs);
      using siluOut = gateOut.siluAndMul(gateOut, upOut, moeIntermediate, count);

      using downOut = siluOut.nvfp4MulMatId(siluOut, downWeightPtrs, downScalePtrs, downScale2Ptrs, topkIndicesFlat, downBatchIdsBuf, count, hs, moeIntermediate);

      using normalizedWeightsFlat = normalizedWeights.reshape([count]);
      routedOut.scatterAddRows(downOut, normalizedWeightsFlat, batchIdsBuf, hs, count, BS);
    } else {
      const batchIdsBuf = this.tensors.get("__moe_batch_ids")!;
      const downBatchIdsBuf = this.tensors.get("__moe_down_batch_ids")!;

      const gateWeightPtrs = this.createExpertWeightPtrs(pfx, "gate_proj");
      const upWeightPtrs = this.createExpertWeightPtrs(pfx, "up_proj");
      const downWeightPtrs = this.createExpertWeightPtrs(pfx, "down_proj");

      using gateOut = normed.mulMatId(normed, gateWeightPtrs, topkIndicesFlat, batchIdsBuf, count, moeIntermediate, hs);
      using upOut = normed.mulMatId(normed, upWeightPtrs, topkIndicesFlat, batchIdsBuf, count, moeIntermediate, hs);
      using siluOut = gateOut.siluAndMul(gateOut, upOut, moeIntermediate, count);

      using downOut = siluOut.mulMatId(siluOut, downWeightPtrs, topkIndicesFlat, downBatchIdsBuf, count, hs, moeIntermediate);

      using normalizedWeightsFlat = normalizedWeights.reshape([count]);
      routedOut.scatterAddRows(downOut, normalizedWeightsFlat, batchIdsBuf, hs, count, BS);
    }

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
      ws.mlaKvCacheAppend(ckvNormed, kPeRaw, pagedKV, layerIdx, batchSize, kvLoraRank, qkRopeDim, state.isDecode);
      attnOut.replace(ws.mlaDecodePaged(qAbsorbedR, qPeR, pagedKV, layerIdx, batchSize, nHeads, kvLoraRank, qkRopeDim, cfg.scaling));
    } else if (useMla && !state.isDecode) {
      using rotaryEmbedding = this.glm.withStream(() => this.invFreq.rotaryEmbedding(ws.positionIds, qkRopeDim / 2, B, S));
      using cos = rotaryEmbedding.result.cos;
      using sin = rotaryEmbedding.result.sin;

      this.glm.mlaPrefillPlan(
        ws.floatWs, 128 * 1024 * 1024,
        ws.intWs, ws.pinnedIntWs, 8 * 1024 * 1024,
        ws.mlaPrefillPlanInfo,
        ws.qoIndptrH, ws.indptrH,
        ws.kvLenH,
        batchSize, nHeads, kvLoraRank, true
      );

      rotaryEmbedding.streamWaitEvent();

      const qAbsorbedR = qAbsorbedLin.ropeTranspose(cos, sin, 0, kvLoraRank, nHeads, S, B, kvLoraRank);
      using _qAbsorbedR = qAbsorbedR;
      const qPeFinal = qPeLin.ropeTranspose(cos, sin, qkRopeDim, qkRopeDim, nHeads, S, B, qkRopeDim);
      using _qPeFinal = qPeFinal;
      const kPeRope = kPeRaw.applyRotaryPosEmb(cos, sin, qkRopeDim, 1, S, B, 1);
      using _kPeRope = kPeRope;

      ws.mlaKvCacheAppend(ckvNormed, kPeRope, pagedKV, layerIdx, batchSize, kvLoraRank, qkRopeDim, false);
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

    const result = this.computeLogits(normed.value, state);
    return result;
  }
}
