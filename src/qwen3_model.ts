import fs from "node:fs";
import path from "node:path";
import { GlmOps, f32ToBf16Bytes, bf16BytesToF32, I32 } from "./glm_ops";
import { SafeTensorFile } from "./safetensors";
import { resolveModelPath } from "./model_path";
import { PagedKVCache } from "./paged_kv";
import { Tensor } from "./tensor";
import type { ChatCache } from "./chat_model";
import { ChatModelBase, type BatchState, SamplingParams } from "./chat_model";
import { UsingHolder } from "./using-holder";

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

export class Qwen3Model extends ChatModelBase {
  readonly eosIds = new Set([151645, 151643]);
  cfg: Qwen3Config;
  maxBatch: number;
  maxSeqLen: number;
  invFreq: Tensor;

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

    model.freeze();
    return model;
  }

  free(): void {
    super.free();
  }

  createChatCache(maxPages = 256): ChatCache {
    return new PagedKVCache(this.glm, this.cfg.numKeyValueHeads, this.cfg.headDim, this.cfg.numHiddenLayers, maxPages, this.maxBatch);
  }

  protected getPagedKV(cache: ChatCache): PagedKVCache {
    if (!(cache instanceof PagedKVCache)) throw new Error("Expected PagedKVCache");
    return cache;
  }

  private mlp(normed: Tensor, BS: number, pfx: string): Tensor {
    using gateBuf = normed.linear(this.tensors.get(`${pfx}.mlp.gate_proj.weight`)!, BS);
    using upBuf = normed.linear(this.tensors.get(`${pfx}.mlp.up_proj.weight`)!, BS);
    using siluBuf = gateBuf.siluAndMul(gateBuf, upBuf, this.cfg.intermediateSize, BS);
    return siluBuf.linear(this.tensors.get(`${pfx}.mlp.down_proj.weight`)!, BS);
  }

  private computeQkv(normed: Tensor, pfx: string, BS: number, B: number, S: number, cos: Tensor, sin: Tensor): { qRope: Tensor, kRope: Tensor, vBuf: Tensor } {
    const cfg = this.cfg;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;

    using qBuf = normed.linear(this.tensors.get(`${pfx}.self_attn.q_proj.weight`)!, BS);
    using kBuf = normed.linear(this.tensors.get(`${pfx}.self_attn.k_proj.weight`)!, BS);
    const vBuf = normed.linear(this.tensors.get(`${pfx}.self_attn.v_proj.weight`)!, BS);

    const qRope = qBuf.fusedNormRope(this.tensors.get(`${pfx}.self_attn.q_norm.weight`)!, cos, sin, cfg.rmsNormEps, hd, hd, nHeads, S, B);
    const kRope = kBuf.fusedNormRope(this.tensors.get(`${pfx}.self_attn.k_norm.weight`)!, cos, sin, cfg.rmsNormEps, hd, hd, nKv, S, B);
    return { qRope, kRope, vBuf };
  }

  forward(state: BatchState): Tensor {
    const ws = state.ws;
    using _tracker = ws.startTracking();
    const pagedKV = this.getPagedKV(state.cache);
    const cfg = this.cfg;
    const glm = this.glm;
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

    const embedTable = this.tensors.get("model.embed_tokens.weight")!;
    using residual = new UsingHolder(embedTable.embedding(ws.inputIdsBuf, hs, BS));

    const rotaryEmbedding = this.invFreq.rotaryEmbedding(ws.positionIds, hd / 2, B, S);
    using cos = rotaryEmbedding.cos;
    using sin = rotaryEmbedding.sin;

    using normed = new UsingHolder(residual.value.rmsnorm(this.tensors.get(`model.layers.0.input_layernorm.weight`)!, cfg.rmsNormEps, hs, BS));

    for (let i = 0; i < cfg.numHiddenLayers; i++) {
      const pfx = `model.layers.${i}`;

      const _qkv = this.computeQkv(normed.value, pfx, BS, B, S, cos, sin);
      using qRope = _qkv.qRope;
      using kRope = _qkv.kRope;
      using vBuf = _qkv.vBuf;

      const slotMapping = ws.slotMapping.data;
      const kTokenStride = state.isDecode ? nKv * hd : hd;
      const kHeadStride = state.isDecode ? hd : BS * hd;
      const vTokenStride = nKv * hd;
      const vHeadStride = hd;
      glm.kvCacheWrite(
        kRope.data, vBuf.data,
        pagedKV.kData[i].data, pagedKV.vData[i].data,
        slotMapping,
        BS, nKv, hd, pageSize,
        kTokenStride, kHeadStride, vTokenStride, vHeadStride
      );

      using flashOut = new UsingHolder<Tensor>(undefined!);
      if (state.isDecode) {
        flashOut.replace(ws.flashDecode(qRope, pagedKV, i, batchSize, nHeads, nKv, hd, cfg.scaling));
      } else {
        const qStrideN = hd;
        const qStrideH = totalTokens * hd;
        flashOut.replace(ws.flashPrefillPaged(qRope, pagedKV, i, totalTokens, batchSize, nHeads, nKv, hd, qStrideN, qStrideH, 1, cfg.scaling));
      }

      using oProjBuf = flashOut.value.linear(this.tensors.get(`${pfx}.self_attn.o_proj.weight`)!, BS);
      const attnResult = residual.value.fusedAddRmsnorm(oProjBuf, this.tensors.get(`${pfx}.post_attention_layernorm.weight`)!, cfg.rmsNormEps, hs, BS);
      using attnNormed = attnResult.normed;
      residual.replace(attnResult.residual);

      using downBuf = this.mlp(attnNormed, BS, pfx);
      const nextWeight = i < cfg.numHiddenLayers - 1
        ? this.tensors.get(`model.layers.${i + 1}.input_layernorm.weight`)!
        : this.tensors.get("model.norm.weight")!;
      const mlpResult = residual.value.fusedAddRmsnorm(downBuf, nextWeight, cfg.rmsNormEps, hs, BS);
      normed.replace(mlpResult.normed);
      residual.replace(mlpResult.residual);
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
