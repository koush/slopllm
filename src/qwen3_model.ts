import fs from "node:fs";
import path from "node:path";
import type { ChatCache } from "./chat_model";
import { ChatModel, SamplingParams } from "./chat_model";
import { DeviceOps, TensorParallelism } from "./device_ops";
import { bf16BytesToF32, f32ToBf16Bytes } from "./glm_ops";
import { resolveModelPath } from "./model_path";
import { ExecutionState } from "./paged_kv";
import { PagedKVCache } from "./paged_kv";
import { SafeTensorFile, type TensorMeta } from "./safetensors";
import { Tensor } from "./tensor";
import { UsingHolder } from "./using-holder";

export { ExecutionState as BatchState };
export type { SamplingParams };

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

export class Qwen3Model extends ChatModel {
  readonly eosIds = new Set([151645, 151643]);
  cfg: Qwen3Config;
  maxBatch: number;
  maxSeqLen: number;
  invFreq: Tensor;

  private constructor(glm: DeviceOps, config: Qwen3Config, maxBatch: number, maxSeqLen: number) {
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

  static fromPretrained(glm: DeviceOps, repoId: string, maxBatch = 1, maxSeqLen = 4096): Qwen3Model {
    const modelDir = resolveModelPath(repoId);
    const config = loadConfig(modelDir);
    const model = new Qwen3Model(glm, config, maxBatch, maxSeqLen);
    model.loadWeights(modelDir);
    return model;
  }

  private weightParallelism(name: string): TensorParallelism {
    if (name === "lm_head.weight") return TensorParallelism.Column;
    if (name === "model.embed_tokens.weight") return TensorParallelism.Row;
    if (name.endsWith(".self_attn.q_proj.weight") ||
      name.endsWith(".self_attn.k_proj.weight") ||
      name.endsWith(".self_attn.v_proj.weight") ||
      name.endsWith(".mlp.gate_proj.weight") ||
      name.endsWith(".mlp.up_proj.weight")) return TensorParallelism.Column;
    if (name.endsWith(".self_attn.q_proj.weight_scale_inv") ||
      name.endsWith(".self_attn.k_proj.weight_scale_inv") ||
      name.endsWith(".self_attn.v_proj.weight_scale_inv") ||
      name.endsWith(".mlp.gate_proj.weight_scale_inv") ||
      name.endsWith(".mlp.up_proj.weight_scale_inv")) return TensorParallelism.Column;
    if (name.endsWith(".self_attn.o_proj.weight") ||
      name.endsWith(".mlp.down_proj.weight")) return TensorParallelism.Row;
    if (name.endsWith(".self_attn.o_proj.weight_scale_inv") ||
      name.endsWith(".mlp.down_proj.weight_scale_inv")) return TensorParallelism.Row;
    return TensorParallelism.Replicated;
  }

  protected loadTensor(name: string, meta: TensorMeta, st: SafeTensorFile, mmapPtr: number): void {
    const par = this.weightParallelism(name);

    if (name.endsWith("_scale_inv")) {
      const bf16Bytes = st.readTensor(name);
      const f32Array = bf16BytesToF32(bf16Bytes);
      const f32Buffer = Buffer.from(f32Array.buffer, f32Array.byteOffset, f32Array.byteLength);
      const tensor = this.alloc(meta.shape, "F32", name, par);
      tensor.h2d(f32Buffer);
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

  protected tieWeights(): void {
    if (this.cfg.tieWordEmbeddings && !this.tensors.has("lm_head.weight")) {
      const embedTensor = this.tensors.get("model.embed_tokens.weight")!;
      this.tensors.set("lm_head.weight", embedTensor);
    }
  }

  createChatCache(maxPages = 256): ChatCache {
    return new PagedKVCache(this.glm, this.cfg.numKeyValueHeads, this.cfg.headDim, this.cfg.numHiddenLayers, maxPages, this.maxBatch);
  }

  private mlp(normed: Tensor, BS: number, pfx: string): Tensor {
    using upStream = this.glm.withStream(() => normed.linear(this.tensors.get(`${pfx}.mlp.up_proj.weight`)!, BS));
    using upBuf = upStream.result;
    using gateBuf = normed.linear(this.tensors.get(`${pfx}.mlp.gate_proj.weight`)!, BS);
    upStream.streamWaitEvent();
    using siluBuf = gateBuf.siluAndMul(gateBuf, upBuf, this.cfg.intermediateSize, BS);
    return siluBuf.linear(this.tensors.get(`${pfx}.mlp.down_proj.weight`)!, BS);
  }

  forward(state: ExecutionState): Tensor {
    const ws = state.ws;
    using _tracker = ws.startTracking();
    const pagedKV = state.cache.getPagedKV();
    const cfg = this.cfg;
    const hs = cfg.hiddenSize;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const batchSize = state.batchSize;
    const totalTokens = state.totalTokens;
    const BS = totalTokens;
    const B = state.isDecode ? batchSize : 1;
    const S = state.isDecode ? 1 : totalTokens;

    const embedTable = this.tensors.get("model.embed_tokens.weight")!;
    using residual = new UsingHolder(embedTable.embedding(ws.inputIdsBuf, hs, BS));

    using rotaryEmbedding = this.glm.withStream(() => this.invFreq.rotaryEmbedding(ws.positionIds, hd / 2, B, S));
    using cos = rotaryEmbedding.result.cos;
    using sin = rotaryEmbedding.result.sin;

    using normed = new UsingHolder(residual.value.rmsnorm(this.tensors.get(`model.layers.0.input_layernorm.weight`)!, cfg.rmsNormEps, hs, BS));

    for (let i = 0; i < cfg.numHiddenLayers; i++) {
      const pfx = `model.layers.${i}`;

      using vStream = this.glm.withStream(() => normed.value.linear(this.tensors.get(`${pfx}.self_attn.v_proj.weight`)!, BS));
      using vBuf = vStream.result;

      using kStream = this.glm.withStream(() => {
        const kBuf = normed.value.linear(this.tensors.get(`${pfx}.self_attn.k_proj.weight`)!, BS);

        if (!i) {
          rotaryEmbedding.streamWaitEvent();
        }

        const kRope = kBuf.fusedNormRope(this.tensors.get(`${pfx}.self_attn.k_norm.weight`)!, cos, sin, cfg.rmsNormEps, hd, hd, nKv, S, B);
        vStream.streamWaitEvent();
        state.kvCacheWrite(kRope, vBuf, i, nKv, hd);
        return { kBuf, kRope };
      });
      using _kBuf = kStream.result.kBuf;
      using _kRope = kStream.result.kRope;

      using qBuf = normed.value.linear(this.tensors.get(`${pfx}.self_attn.q_proj.weight`)!, BS);
      if (!i) {
        rotaryEmbedding.streamWaitEvent();
      }
      using qRope = qBuf.fusedNormRope(this.tensors.get(`${pfx}.self_attn.q_norm.weight`)!, cos, sin, cfg.rmsNormEps, hd, hd, nHeads, S, B);

      kStream.streamWaitEvent();

      using flashOut = new UsingHolder<Tensor>(undefined!);
      if (state.isDecode) {
        flashOut.replace(ws.flashDecode(qRope, pagedKV, i, batchSize, nHeads, nKv, hd, cfg.scaling));
      } else {
        const qStrideN = hd;
        const qStrideH = totalTokens * hd;
        flashOut.replace(ws.flashPrefillPaged(qRope, pagedKV, i, totalTokens, batchSize, nHeads, nKv, hd, qStrideN, qStrideH, 1, cfg.scaling));
      }

      using reshapedFlashOut = flashOut.value.reshape([BS, nHeads * hd]);
      using oProjBuf = reshapedFlashOut.linear(this.tensors.get(`${pfx}.self_attn.o_proj.weight`)!, BS);
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
