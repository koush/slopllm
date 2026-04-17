import fs from "node:fs";
import path from "node:path";
import { GlmOps, f32ToBf16Bytes, bf16BytesToF32, I32, SAMPLING_MAX_TOPK } from "./glm_ops";
import { SafeTensorFile } from "./safetensors";
import { resolveModelPath } from "./model_path";
import { PagedKVCache, WorkspaceBuffers } from "./paged_kv";
import { Tensor } from "./tensor";
import type { ChatCache, DecodeState, PrefillState } from "./chat_model";
import { ChatModelBase, SamplingParams } from "./chat_model";

export type { SamplingParams };
export type { DecodeState, PrefillState };

export interface Qwen3Config {
  hiddenSize: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  intermediateSize: number;
  numHiddenLayers: number;
  rmsNormEps: number;
  ropeTheta: number;
  vocabSize: number;
  tieWordEmbeddings: boolean;
  attentionBias: boolean;
  numKeyValueGroups: number;
  scaling: number;
}

function loadConfig(modelDir: string): Qwen3Config {
  const raw = JSON.parse(fs.readFileSync(path.join(modelDir, "config.json"), "utf-8"));
  return {
    hiddenSize: raw.hidden_size,
    numAttentionHeads: raw.num_attention_heads,
    numKeyValueHeads: raw.num_key_value_heads,
    headDim: raw.head_dim,
    intermediateSize: raw.intermediate_size,
    numHiddenLayers: raw.num_hidden_layers,
    rmsNormEps: raw.rms_norm_eps,
    ropeTheta: raw.rope_theta,
    vocabSize: raw.vocab_size,
    tieWordEmbeddings: raw.tie_word_embeddings ?? false,
    attentionBias: raw.attention_bias ?? false,
    numKeyValueGroups: raw.num_attention_heads / raw.num_key_value_heads,
    scaling: Math.pow(raw.head_dim, -0.5),
  };
}

class Qwen3Workspace {
  hiddenA: Tensor;
  hiddenB: Tensor;
  normed: Tensor;
  qBuf: Tensor;
  kBuf: Tensor;
  vBuf: Tensor;
  qNormed: Tensor;
  kNormed: Tensor;
  qT: Tensor;
  kT: Tensor;
  vT: Tensor;
  qRope: Tensor;
  kRope: Tensor;
  oProjBuf: Tensor;
  gateBuf: Tensor;
  upBuf: Tensor;
  siluBuf: Tensor;
  downBuf: Tensor;
  cos: Tensor;
  sin: Tensor;
  positionIds: Tensor;
  logitsBuf: Tensor;
  hiddenLast: Tensor;
  lastIdx: Tensor;
  argmaxIdx: Tensor;
  flashOut: Tensor;
  inputIdsBuf: Tensor;
  qoIndptrD: Tensor;
  prefillSlotMapping: Tensor;
  sampleOutToken: Tensor;
  sampleTopkVals: Tensor;
  sampleTopkIdxs: Tensor;
  sampleWorkspace: Tensor;
  samplePenaltyTokens: Tensor;
  tensors = new Map<string, Tensor>();

  constructor(glm: GlmOps, B: number, S: number, cfg: Qwen3Config) {
    const hs = cfg.hiddenSize;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const inter = cfg.intermediateSize;
    const vs = cfg.vocabSize;
    const BS = B * S;

    this.hiddenA = Tensor.alloc(glm, [B, S, hs], "BF16");
    this.hiddenB = Tensor.alloc(glm, [B, S, hs], "BF16");
    this.normed = Tensor.alloc(glm, [B, S, hs], "BF16");
    this.qBuf = Tensor.alloc(glm, [B, S, nHeads * hd], "BF16");
    this.kBuf = Tensor.alloc(glm, [B, S, nKv * hd], "BF16");
    this.vBuf = Tensor.alloc(glm, [B, S, nKv * hd], "BF16");
    this.qNormed = Tensor.alloc(glm, [B, S, nHeads * hd], "BF16");
    this.kNormed = Tensor.alloc(glm, [B, S, nKv * hd], "BF16");
    this.qT = Tensor.alloc(glm, [B, nHeads, S, hd], "BF16");
    this.kT = Tensor.alloc(glm, [B, nKv, S, hd], "BF16");
    this.vT = Tensor.alloc(glm, [B, nKv, S, hd], "BF16");
    this.qRope = Tensor.alloc(glm, [B, nHeads, S, hd], "BF16");
    this.kRope = Tensor.alloc(glm, [B, nKv, S, hd], "BF16");
    this.oProjBuf = Tensor.alloc(glm, [B, S, hs], "BF16");
    this.gateBuf = Tensor.alloc(glm, [B, S, inter], "BF16");
    this.upBuf = Tensor.alloc(glm, [B, S, inter], "BF16");
    this.siluBuf = Tensor.alloc(glm, [B, S, inter], "BF16");
    this.downBuf = Tensor.alloc(glm, [B, S, hs], "BF16");
    this.cos = Tensor.alloc(glm, [B, S, hd], "BF16");
    this.sin = Tensor.alloc(glm, [B, S, hd], "BF16");
    this.positionIds = Tensor.alloc(glm, [B * S], "I32");
    this.logitsBuf = Tensor.alloc(glm, [B, vs], "BF16");
    this.hiddenLast = Tensor.alloc(glm, [B, hs], "BF16");
    this.lastIdx = Tensor.alloc(glm, [B], "I32");
    this.argmaxIdx = Tensor.alloc(glm, [B], "I32");
    this.flashOut = Tensor.alloc(glm, [B, nHeads, S, hd], "BF16");
    this.inputIdsBuf = Tensor.alloc(glm, [B * S], "I32");
    this.qoIndptrD = Tensor.alloc(glm, [B + 1], "I32");
    this.prefillSlotMapping = Tensor.alloc(glm, [B * S], "I32");

    this.sampleOutToken = Tensor.alloc(glm, [1], "I32");
    this.sampleTopkVals = Tensor.alloc(glm, [SAMPLING_MAX_TOPK * 256], "F32");
    this.sampleTopkIdxs = Tensor.alloc(glm, [SAMPLING_MAX_TOPK * 256], "I32");
    this.sampleWorkspace = Tensor.alloc(glm, [vs], "F32");
    this.samplePenaltyTokens = Tensor.alloc(glm, [1024], "I32");

    for (const key of Object.keys(this) as (keyof this)[]) {
      const value = this[key];
      if (value instanceof Tensor && typeof key === 'string') {
        this.tensors.set(key, value);
      }
    }
  }

  free(): void {
    for (const tensor of this.tensors.values()) {
      tensor.free();
    }
    this.tensors.clear();
  }
}

export class Qwen3Model extends ChatModelBase {
  readonly eosIds = new Set([151645, 151643]);
  declare glm: GlmOps;
  cfg: Qwen3Config;
  declare weights: Map<string, Tensor>;
  maxBatch: number;
  maxSeqLen: number;
  invFreq: Tensor;
  declare ws: Qwen3Workspace;

  private constructor(glm: GlmOps, config: Qwen3Config, weights: Map<string, Tensor>, maxBatch: number, maxSeqLen: number) {
    super();
    this.glm = glm;
    this.cfg = config;
    this.weights = weights;
    this.maxBatch = maxBatch;
    this.maxSeqLen = maxSeqLen;

    const halfDim = config.headDim / 2;
    const invFreqF32 = new Float32Array(halfDim);
    for (let i = 0; i < halfDim; i++) {
      invFreqF32[i] = 1.0 / Math.pow(config.ropeTheta, (2 * i) / config.headDim);
    }
    this.invFreq = Tensor.alloc(glm, [halfDim], "BF16");
    this.invFreq.h2d(f32ToBf16Bytes(invFreqF32));

    this.ws = new Qwen3Workspace(glm, maxBatch, maxSeqLen, config);
  }

  static fromPretrained(glm: GlmOps, repoId: string, maxBatch = 1, maxSeqLen = 4096): Qwen3Model {
    const modelDir = resolveModelPath(repoId);
    const config = loadConfig(modelDir);

    const stPath = path.join(modelDir, "model.safetensors");
    const st = SafeTensorFile.open(stPath);
    const mmapPtr = glm.mmapOpen(stPath);
    const fileSize = fs.statSync(stPath).size;

    const weights = new Map<string, Tensor>();
    for (const name of st.tensorNames()) {
      const meta = st.meta(name);
      if (name.endsWith("_scale_inv")) {
        const bf16Bytes = st.readTensor(name);
        const f32Array = bf16BytesToF32(bf16Bytes);
        const f32Buffer = Buffer.from(f32Array.buffer, f32Array.byteOffset, f32Array.byteLength);
        const tensor = Tensor.alloc(glm, meta.shape, "F32", name);
        tensor.h2d(f32Buffer);
        weights.set(name, tensor);
      } else {
        const tensor = Tensor.alloc(glm, meta.shape, meta.dtype, name);
        const offset = st.dataStart + meta.dataOffsets[0];
        glm.mmapLoad(tensor.data, mmapPtr, offset, tensor.bytes);
        weights.set(name, tensor);
      }
    }

    glm.synchronize();
    st.close();
    glm.mmapClose(mmapPtr, fileSize);

    if (config.tieWordEmbeddings && !weights.has("lm_head.weight")) {
      const embedTensor = weights.get("model.embed_tokens.weight")!;
      weights.set("lm_head.weight", embedTensor);
    }

    return new Qwen3Model(glm, config, weights, maxBatch, maxSeqLen);
  }

  free(): void {
    this.ws.free();
    this.invFreq.free();
    for (const tensor of this.weights.values()) {
      tensor.free();
    }
    this.ws = null!;
    this.weights = new Map();
  }

  createChatCache(maxPages = 256): ChatCache {
    return new PagedKVCache(this.glm, this.cfg.numKeyValueHeads, this.cfg.headDim, this.cfg.numHiddenLayers, maxPages, this.maxBatch);
  }

  protected getPagedKV(cache: ChatCache): PagedKVCache {
    if (!(cache instanceof PagedKVCache)) throw new Error("Expected PagedKVCache");
    return cache;
  }

  private mlp(BS: number, pfx: string): void {
    const cfg = this.cfg;
    const inter = cfg.intermediateSize;

    this.ws.gateBuf.linear(this.ws.normed, this.weights.get(`${pfx}.mlp.gate_proj.weight`)!, BS, inter, cfg.hiddenSize, this);
    this.ws.upBuf.linear(this.ws.normed, this.weights.get(`${pfx}.mlp.up_proj.weight`)!, BS, inter, cfg.hiddenSize, this);
    this.ws.siluBuf.siluAndMul(this.ws.gateBuf, this.ws.upBuf, inter, BS);
    this.ws.downBuf.linear(this.ws.siluBuf, this.weights.get(`${pfx}.mlp.down_proj.weight`)!, BS, cfg.hiddenSize, inter, this);
  }

  private computeQkv(pfx: string, BS: number, B: number, S: number): void {
    const cfg = this.cfg;
    const hs = cfg.hiddenSize;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;

    this.ws.qBuf.linear(this.ws.normed, this.weights.get(`${pfx}.self_attn.q_proj.weight`)!, BS, nHeads * hd, hs, this);
    this.ws.kBuf.linear(this.ws.normed, this.weights.get(`${pfx}.self_attn.k_proj.weight`)!, BS, nKv * hd, hs, this);
    this.ws.vBuf.linear(this.ws.normed, this.weights.get(`${pfx}.self_attn.v_proj.weight`)!, BS, nKv * hd, hs, this);

    this.ws.qRope.fusedNormRope(this.ws.qBuf, this.weights.get(`${pfx}.self_attn.q_norm.weight`)!, this.ws.cos, this.ws.sin, cfg.rmsNormEps, hd, hd, nHeads, S, B);
    this.ws.kRope.fusedNormRope(this.ws.kBuf, this.weights.get(`${pfx}.self_attn.k_norm.weight`)!, this.ws.cos, this.ws.sin, cfg.rmsNormEps, hd, hd, nKv, S, B);
    if (S > 1) {
      this.ws.vT.transpose4d(this.ws.vBuf, B, S, nKv, hd, 0, 2, 1, 3);
    }
  }

  private vData(S: number): number {
    return S === 1 ? this.ws.vBuf.data : this.ws.vT.data;
  }

  prefillBatchForward(state: PrefillState, ws: WorkspaceBuffers, cache: ChatCache): void {
    const pagedKV = this.getPagedKV(cache);
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const pageSize = pagedKV.pageSize;
    const batchSize = state.batchSize;
    const totalTokens = state.totalTokens;

    this.ws.hiddenA.embedding(this.weights.get("model.embed_tokens.weight")!, this.ws.inputIdsBuf, hs, totalTokens);

    glm.rotaryEmbedding(this.ws.cos.data, this.ws.sin.data, this.invFreq.data, this.ws.positionIds.data, hd / 2, 1, totalTokens);

    this.ws.normed.rmsnorm(this.ws.hiddenA, this.weights.get(`model.layers.0.input_layernorm.weight`)!, cfg.rmsNormEps, hs, totalTokens);

    for (let i = 0; i < cfg.numHiddenLayers; i++) {
      const pfx = `model.layers.${i}`;

      this.computeQkv(pfx, totalTokens, 1, totalTokens);

      glm.kvCacheWrite(
        this.ws.kRope.data, this.vData(totalTokens),
        pagedKV.kData[i], pagedKV.vData[i],
        this.ws.prefillSlotMapping.data,
        totalTokens, nKv, hd, pagedKV.pageSize,
        hd, totalTokens * hd
      );

      const qStrideN = hd;
      const qStrideH = totalTokens * hd;

      glm.batchPrefillPagedRun(
        this.ws.qRope.data, this.ws.flashOut.data,
        pagedKV.kData[i], pagedKV.vData[i],
        pagedKV.indices, pagedKV.indptrD, pagedKV.lastPageLen,
        ws.floatWs, ws.intWs,
        this.ws.qoIndptrD.data,
        ws.prefillPlanInfo,
        totalTokens, batchSize,
        nHeads, nKv, hd,
        pageSize,
        qStrideN, qStrideH,
        1, cfg.scaling
      );

      this.ws.oProjBuf.linear(this.ws.flashOut, this.weights.get(`${pfx}.self_attn.o_proj.weight`)!, totalTokens, hs, nHeads * hd, this);

      this.ws.normed.fusedAddRmsnorm(this.ws.hiddenB, this.ws.hiddenA, this.ws.oProjBuf, this.weights.get(`${pfx}.post_attention_layernorm.weight`)!, cfg.rmsNormEps, hs, totalTokens);

      this.mlp(totalTokens, pfx);

      if (i < cfg.numHiddenLayers - 1) {
        const nextPfx = `model.layers.${i + 1}`;
        this.ws.normed.fusedAddRmsnorm(this.ws.hiddenA, this.ws.hiddenB, this.ws.downBuf, this.weights.get(`${nextPfx}.input_layernorm.weight`)!, cfg.rmsNormEps, hs, totalTokens);
      } else {
        this.ws.normed.fusedAddRmsnorm(this.ws.hiddenA, this.ws.hiddenB, this.ws.downBuf, this.weights.get("model.norm.weight")!, cfg.rmsNormEps, hs, totalTokens);
        this.ws.hiddenLast.indexSelect(this.ws.normed, this.ws.lastIdx, hs, batchSize);
        this.ws.logitsBuf.linear(this.ws.hiddenLast, this.weights.get("lm_head.weight")!, batchSize, cfg.vocabSize, hs, this);
      }
    }

    this.ws.argmaxIdx.argmax(this.ws.logitsBuf, cfg.vocabSize, batchSize);
  }

  decodeBatchForward(state: DecodeState, ws: WorkspaceBuffers, cache: ChatCache): void {
    const pagedKV = this.getPagedKV(cache);
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const pageSize = pagedKV.pageSize;
    const batchSize = state.batchSize;

    this.ws.hiddenA.embedding(this.weights.get("model.embed_tokens.weight")!, this.ws.inputIdsBuf, hs, batchSize);

    glm.rotaryEmbedding(this.ws.cos.data, this.ws.sin.data, this.invFreq.data, this.ws.positionIds.data, hd / 2, batchSize, 1);

    this.ws.normed.rmsnorm(this.ws.hiddenA, this.weights.get(`model.layers.0.input_layernorm.weight`)!, cfg.rmsNormEps, hs, batchSize);

    for (let i = 0; i < cfg.numHiddenLayers; i++) {
      const pfx = `model.layers.${i}`;

      this.computeQkv(pfx, batchSize, batchSize, 1);

      glm.kvCacheWrite(
        this.ws.kRope.data, this.vData(1),
        pagedKV.kData[i], pagedKV.vData[i],
        pagedKV.slotMapping,
        batchSize, nKv, hd, pageSize,
        nKv * hd, hd
      );

      glm.batchDecodeRun(
        this.ws.qRope.data, this.ws.flashOut.data,
        pagedKV.kData[i], pagedKV.vData[i],
        pagedKV.indices, pagedKV.indptrD, pagedKV.lastPageLen,
        ws.floatWs, ws.intWs,
        ws.decodePlanInfo,
        batchSize,
        nHeads, nKv, hd, pageSize, cfg.scaling
      );

      this.ws.oProjBuf.linear(this.ws.flashOut, this.weights.get(`${pfx}.self_attn.o_proj.weight`)!, batchSize, hs, nHeads * hd, this);

      this.ws.normed.fusedAddRmsnorm(this.ws.hiddenB, this.ws.hiddenA, this.ws.oProjBuf, this.weights.get(`${pfx}.post_attention_layernorm.weight`)!, cfg.rmsNormEps, hs, batchSize);

      this.mlp(batchSize, pfx);

      if (i < cfg.numHiddenLayers - 1) {
        const nextPfx = `model.layers.${i + 1}`;
        this.ws.normed.fusedAddRmsnorm(this.ws.hiddenA, this.ws.hiddenB, this.ws.downBuf, this.weights.get(`${nextPfx}.input_layernorm.weight`)!, cfg.rmsNormEps, hs, batchSize);
      } else {
        this.ws.normed.fusedAddRmsnorm(this.ws.hiddenA, this.ws.hiddenB, this.ws.downBuf, this.weights.get("model.norm.weight")!, cfg.rmsNormEps, hs, batchSize);
        this.ws.logitsBuf.linear(this.ws.normed, this.weights.get("lm_head.weight")!, batchSize, cfg.vocabSize, hs, this);
      }
    }

    this.ws.argmaxIdx.argmax(this.ws.logitsBuf, cfg.vocabSize, batchSize);
  }
}
