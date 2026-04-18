import fs from "node:fs";
import path from "node:path";
import { GlmOps, f32ToBf16Bytes, bf16BytesToF32, I32 } from "./glm_ops";
import { SafeTensorFile } from "./safetensors";
import { resolveModelPath } from "./model_path";
import { PagedKVCache } from "./paged_kv";
import { Tensor } from "./tensor";
import type { ChatCache } from "./chat_model";
import { ChatModelBase, type BatchState, SamplingParams, SamplingWorkspaceBase } from "./chat_model";

export type { SamplingParams };
export type { BatchState };

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

class Qwen3Workspace extends SamplingWorkspaceBase {
  hiddenA: Tensor;
  hiddenB: Tensor;
  normed: Tensor;
  cos: Tensor;
  sin: Tensor;
  positionIds: Tensor;
  hiddenLast: Tensor;
  lastIdx: Tensor;
  argmaxIdx: Tensor;
  flashOut: Tensor;
  inputIdsBuf: Tensor;
  qoIndptrD: Tensor;
  prefillSlotMapping: Tensor;

  constructor(glm: GlmOps, B: number, S: number, cfg: Qwen3Config) {
    super(glm, B, cfg.vocabSize);
    const hs = cfg.hiddenSize;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;

    this.hiddenA = this.alloc([B, S, hs], "BF16", "hiddenA");
    this.hiddenB = this.alloc([B, S, hs], "BF16", "hiddenB");
    this.normed = this.alloc([B, S, hs], "BF16", "normed");
    this.cos = this.alloc([B, S, hd], "BF16", "cos");
    this.sin = this.alloc([B, S, hd], "BF16", "sin");
    this.positionIds = this.alloc([B * S], "I32", "positionIds");
    this.hiddenLast = this.alloc([B, hs], "BF16", "hiddenLast");
    this.lastIdx = this.alloc([B], "I32", "lastIdx");
    this.argmaxIdx = this.alloc([B], "I32", "argmaxIdx");
    this.flashOut = this.alloc([B, nHeads, S, hd], "BF16", "flashOut");
    this.inputIdsBuf = this.alloc([B * S], "I32", "inputIdsBuf");
    this.qoIndptrD = this.alloc([B + 1], "I32", "qoIndptrD");
    this.prefillSlotMapping = this.alloc([B * S], "I32", "prefillSlotMapping");
  }
}

export class Qwen3Model extends ChatModelBase {
  readonly eosIds = new Set([151645, 151643]);
  cfg: Qwen3Config;
  maxBatch: number;
  maxSeqLen: number;
  invFreq: Tensor;
  declare ws: Qwen3Workspace;

  private constructor(glm: GlmOps, config: Qwen3Config, maxBatch: number, maxSeqLen: number) {
    super(glm);
    this.cfg = config;
    this.maxBatch = maxBatch;
    this.maxSeqLen = maxSeqLen;

    const halfDim = config.headDim / 2;
    const invFreqF32 = new Float32Array(halfDim);
    for (let i = 0; i < halfDim; i++) {
      invFreqF32[i] = 1.0 / Math.pow(config.ropeTheta, (2 * i) / config.headDim);
    }
    this.invFreq = this.alloc([halfDim], "BF16", "invFreq");
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

    const model = new Qwen3Model(glm, config, maxBatch, maxSeqLen);

    for (const name of st.tensorNames()) {
      const meta = st.meta(name);
      if (name.endsWith("_scale_inv")) {
        const bf16Bytes = st.readTensor(name);
        const f32Array = bf16BytesToF32(bf16Bytes);
        const f32Buffer = Buffer.from(f32Array.buffer, f32Array.byteOffset, f32Array.byteLength);
        const tensor = model.alloc(meta.shape, "F32", name);
        tensor.h2d(f32Buffer);
      } else {
        const tensor = model.alloc(meta.shape, meta.dtype, name);
        const offset = st.dataStart + meta.dataOffsets[0];
        glm.mmapLoad(tensor.data, mmapPtr, offset, tensor.bytes);
      }
    }

    glm.synchronize();
    st.close();
    glm.mmapClose(mmapPtr, fileSize);

    if (config.tieWordEmbeddings && !model.tensors.has("lm_head.weight")) {
      const embedTensor = model.tensors.get("model.embed_tokens.weight")!;
      model.tensors.set("lm_head.weight", embedTensor);
    }

    return model;
  }

  free(): void {
    this.ws.free();
    super.free();
  }

  createChatCache(maxPages = 256): ChatCache {
    return new PagedKVCache(this.glm, this.cfg.numKeyValueHeads, this.cfg.headDim, this.cfg.numHiddenLayers, maxPages, this.maxBatch);
  }

  protected getPagedKV(cache: ChatCache): PagedKVCache {
    if (!(cache instanceof PagedKVCache)) throw new Error("Expected PagedKVCache");
    return cache;
  }

  private mlp(BS: number, pfx: string): Tensor {
    using gateBuf = this.ws.normed.linear(this.tensors.get(`${pfx}.mlp.gate_proj.weight`)!, BS);
    using upBuf = this.ws.normed.linear(this.tensors.get(`${pfx}.mlp.up_proj.weight`)!, BS);
    using siluBuf = gateBuf.siluAndMul(gateBuf, upBuf, this.cfg.intermediateSize, BS);
    return siluBuf.linear(this.tensors.get(`${pfx}.mlp.down_proj.weight`)!, BS);
  }

  private computeQkv(pfx: string, BS: number, B: number, S: number): { qRope: Tensor, kRope: Tensor, vBuf: Tensor, vT: Tensor | null } {
    const cfg = this.cfg;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;

    using qBuf = this.ws.normed.linear(this.tensors.get(`${pfx}.self_attn.q_proj.weight`)!, BS);
    using kBuf = this.ws.normed.linear(this.tensors.get(`${pfx}.self_attn.k_proj.weight`)!, BS);
    const vBuf = this.ws.normed.linear(this.tensors.get(`${pfx}.self_attn.v_proj.weight`)!, BS);

    const qRope = qBuf.fusedNormRope(this.tensors.get(`${pfx}.self_attn.q_norm.weight`)!, this.ws.cos, this.ws.sin, cfg.rmsNormEps, hd, hd, nHeads, S, B);
    const kRope = kBuf.fusedNormRope(this.tensors.get(`${pfx}.self_attn.k_norm.weight`)!, this.ws.cos, this.ws.sin, cfg.rmsNormEps, hd, hd, nKv, S, B);
    const vT = S > 1 ? vBuf.transpose4d(B, S, nKv, hd, 0, 2, 1, 3) : null;
    return { qRope, kRope, vBuf, vT };
  }

  forward(state: BatchState, cache: ChatCache): Tensor {
    using _tracker = this.ws.startTracking();
    const pagedKV = this.getPagedKV(cache);
    const cfg = this.cfg;
    const glm = this.glm;
    const ws = this.ws;
    const hs = cfg.hiddenSize;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const pageSize = pagedKV.pageSize;
    const batchSize = state.batchSize;
    const totalTokens = state.totalTokens;
    const BS = totalTokens;
    const B = state.isDecode ? batchSize : 1;
    const S = state.isDecode ? 1 : totalTokens;

    this.ws.hiddenA.embedding(this.tensors.get("model.embed_tokens.weight")!, this.ws.inputIdsBuf, hs, BS);

    glm.rotaryEmbedding(this.ws.cos.data, this.ws.sin.data, this.invFreq.data, this.ws.positionIds.data, hd / 2, B, S);

    this.ws.normed.rmsnorm(this.ws.hiddenA, this.tensors.get(`model.layers.0.input_layernorm.weight`)!, cfg.rmsNormEps, hs, BS);

    for (let i = 0; i < cfg.numHiddenLayers; i++) {
      const pfx = `model.layers.${i}`;

      const qkv = this.computeQkv(pfx, BS, B, S);
      using _qRope = qkv.qRope;
      using _kRope = qkv.kRope;
      using _vBuf = qkv.vBuf;
      if (qkv.vT) { using _ = qkv.vT; }

      const slotMapping = state.isDecode ? pagedKV.slotMapping.data : this.ws.prefillSlotMapping.data;
      const kStride = state.isDecode ? nKv * hd : hd;
      const vStride = state.isDecode ? hd : totalTokens * hd;
      glm.kvCacheWrite(
        qkv.kRope.data, qkv.vT ? qkv.vT.data : qkv.vBuf.data,
        pagedKV.kData[i].data, pagedKV.vData[i].data,
        slotMapping,
        BS, nKv, hd, pageSize,
        kStride, vStride
      );

      if (state.isDecode) {
        glm.batchDecodeRun(
          qkv.qRope.data, this.ws.flashOut.data,
          pagedKV.kData[i].data, pagedKV.vData[i].data,
          pagedKV.indices.data, pagedKV.indptrD.data, pagedKV.lastPageLen.data,
          ws.floatWs.data, ws.intWs.data,
          ws.decodePlanInfo.data,
          batchSize,
          nHeads, nKv, hd, pageSize, cfg.scaling
        );
      } else {
        const qStrideN = hd;
        const qStrideH = totalTokens * hd;
        glm.batchPrefillPagedRun(
          qkv.qRope.data, this.ws.flashOut.data,
          pagedKV.kData[i].data, pagedKV.vData[i].data,
          pagedKV.indices.data, pagedKV.indptrD.data, pagedKV.lastPageLen.data,
          ws.floatWs.data, ws.intWs.data,
          this.ws.qoIndptrD.data,
          ws.prefillPlanInfo.data,
          totalTokens, batchSize,
          nHeads, nKv, hd,
          pageSize,
          qStrideN, qStrideH,
          1, cfg.scaling
        );
      }

      using oProjBuf = this.ws.flashOut.linear(this.tensors.get(`${pfx}.self_attn.o_proj.weight`)!, BS);

      this.ws.normed.fusedAddRmsnorm(this.ws.hiddenB, this.ws.hiddenA, oProjBuf, this.tensors.get(`${pfx}.post_attention_layernorm.weight`)!, cfg.rmsNormEps, hs, BS);

      using downBuf = this.mlp(BS, pfx);

      if (i < cfg.numHiddenLayers - 1) {
        const nextPfx = `model.layers.${i + 1}`;
        this.ws.normed.fusedAddRmsnorm(this.ws.hiddenA, this.ws.hiddenB, downBuf, this.tensors.get(`${nextPfx}.input_layernorm.weight`)!, cfg.rmsNormEps, hs, BS);
      } else {
        this.ws.normed.fusedAddRmsnorm(this.ws.hiddenA, this.ws.hiddenB, downBuf, this.tensors.get("model.norm.weight")!, cfg.rmsNormEps, hs, BS);
      }
    }

    let logitsBuf: Tensor;
    if (state.isDecode) {
      logitsBuf = this.ws.normed.linear(this.tensors.get("lm_head.weight")!, batchSize);
    } else {
      this.ws.hiddenLast.indexSelect(this.ws.normed, this.ws.lastIdx, hs, batchSize);
      logitsBuf = this.ws.hiddenLast.linear(this.tensors.get("lm_head.weight")!, batchSize);
    }
    this.ws.argmaxIdx.argmax(logitsBuf, cfg.vocabSize, batchSize);
    return logitsBuf.removeTracking();
  }
}
