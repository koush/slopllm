import fs from "node:fs";
import path from "node:path";
import { PagedKVCache } from "../src/paged_kv";
import type { ChatCache } from "./chat_model";
import { ChatModel, CommonModelConfig } from "./chat_model";
import { DeviceOps, TensorParallelism } from "./device_ops";
import { resolveModelPath } from "./model_path";
import { SafeTensorFile, type TensorMeta } from "./safetensors";
import { Tensor } from "./tensor";
import { UsingHolder } from "./using-holder";
import { ExecutionState } from "./execution-workspace";

export interface Qwen3Config extends CommonModelConfig {
  attentionBias: boolean;
  eosTokenIds: number[];
}

function loadConfig(modelDir: string): Qwen3Config {
  const raw = JSON.parse(fs.readFileSync(path.join(modelDir, "config.json"), "utf-8"));
  return {
    hiddenSize: raw.hidden_size,
    intermediateSize: raw.intermediate_size,
    numHiddenLayers: raw.num_hidden_layers,
    rmsNormEps: raw.rms_norm_eps,
    vocabSize: raw.vocab_size,
    tieWordEmbeddings: raw.tie_word_embeddings ?? false,
    numAttentionHeads: raw.num_attention_heads,
    numKeyValueHeads: raw.num_key_value_heads,
    headDim: raw.head_dim,
    ropeTheta: raw.rope_theta,
    numKeyValueGroups: raw.num_attention_heads / raw.num_key_value_heads,
    scaling: Math.pow(raw.head_dim, -0.5),
    attentionBias: raw.attention_bias ?? false,
    eosTokenIds: Array.isArray(raw.eos_token_id) ? raw.eos_token_id : [raw.eos_token_id ?? 151645],
  };
}

export class Qwen3Model extends ChatModel {
  readonly eosIds: Set<number>;
  cfg: Qwen3Config;
  invFreq: Tensor;

  private constructor(glm: DeviceOps, config: Qwen3Config) {
    super(glm);
    this.cfg = config;
    this.eosIds = new Set(config.eosTokenIds);
    this.invFreq = this.initInvFreq(config.headDim, config.ropeTheta);
  }

  static async fromPretrained(glm: DeviceOps, repoIdOrDir: string): Promise<Qwen3Model> {
    const modelDir = fs.existsSync(repoIdOrDir) ? repoIdOrDir : resolveModelPath(repoIdOrDir);
    const config = loadConfig(modelDir);
    const model = new Qwen3Model(glm, config);
    await model.fromPretrained(modelDir);
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

  protected async loadTensor(name: string, meta: TensorMeta, st: SafeTensorFile, mmapPtr: number): Promise<void> {
    const par = this.weightParallelism(name);

    if (name.endsWith("_scale_inv")) {
      const tensor = this.alloc(meta.shape, "BF16", name, par);
      const offset = st.dataStart + meta.dataOffsets[0];
      await tensor.mmapLoad(mmapPtr, offset, tensor.bytes);
    } else {
      const dtype = meta.dtype === "F32" ? "F32" : meta.dtype;
      const tensor = this.alloc(meta.shape, dtype, name, par);
      const offset = st.dataStart + meta.dataOffsets[0];
      await tensor.mmapLoad(mmapPtr, offset, tensor.bytes);

      if (this.cfg.tieWordEmbeddings && name === "model.embed_tokens.weight" && !this.tensors.has("lm_head.weight")) {
        const lmHead = this.alloc(meta.shape, dtype, "lm_head.weight", TensorParallelism.Column);
        await lmHead.mmapLoad(mmapPtr, offset, lmHead.bytes);
      }
    }
  }

  createChatCache(maxPages = 256, maxBatch = 1, _maxSeqLen = 4096, _pageSize = 16): ChatCache {
    return new PagedKVCache(this.glm, this.cfg.numKeyValueHeads, this.cfg.headDim, this.cfg.numHiddenLayers, maxPages, maxBatch);
  }

  private mlp(normed: Tensor, BS: number, pfx: string): Tensor {
    return this.swiGluMlp(normed, `${pfx}.mlp`, this.cfg.intermediateSize, BS);
  }

  forwardModel(state: ExecutionState): Tensor {
    const ws = state.ws;
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
    using residual = new UsingHolder(embedTable.embedding(state.input!, hs, BS));

    using rotaryEmbedding = this.glm.withStream(() => this.invFreq.rotaryEmbedding(ws.positionIds, hd / 2, B, S));
    using cos = rotaryEmbedding.result.cos;
    using sin = rotaryEmbedding.result.sin;

    using normed = new UsingHolder(residual.value.rmsnorm(this.tensors.get(`model.layers.0.input_layernorm.weight`)!, cfg.rmsNormEps, hs, BS));

    for (let i = 0; i < cfg.numHiddenLayers; i++) {
      const pfx = `model.layers.${i}`;

      using vStream = this.glm.withStream(() => normed.value.linear(this.tensors.get(`${pfx}.self_attn.v_proj.weight`)!, BS));
      using vBuf = vStream.result;

      using kStream = this.glm.withStream(() => {
        using kBuf = normed.value.linear(this.tensors.get(`${pfx}.self_attn.k_proj.weight`)!, BS);

        if (!i) {
          rotaryEmbedding.streamWaitEvent();
        }

        using kRope = kBuf.fusedNormRope(this.tensors.get(`${pfx}.self_attn.k_norm.weight`)!, cos, sin, cfg.rmsNormEps, hd, hd, nKv, S, B);
        vStream.streamWaitEvent();
        state.kvCacheWrite(kRope, vBuf, i, nKv, hd);
      });

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

    return normed.detach().removeTracking();
  }
}
