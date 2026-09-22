import fs from "node:fs";
import path from "node:path";
import type { ChatCache } from "./chat_model";
import { ChatModel, CommonModelConfig, SamplingParams } from "./chat_model";
import { DeviceOps, StridedMmap, TensorParallelism } from "./device_ops";
import { ExecutionState, ExecutionWorkspace } from "./execution-workspace";
import { f32ToBf16Bytes } from "./glm_ops";
import { resolveModelPath } from "./model_path";
import { PagedKVCache } from "./paged_kv";
import { Qwen35GdnState } from "./qwen35_gdn_state";
import { SafeTensorFile, type TensorMeta } from "./safetensors";
import { Tensor } from "./tensor";
import { UsingHolder } from "./using-holder";

class Qwen35ChatCache implements ChatCache {
  constructor(
    public readonly pagedKV: PagedKVCache,
    public readonly gdnState: Qwen35GdnState,
    public maxBatch = 1,
    public maxSeqLen = 4096,
  ) {}

  getPagedKV(): PagedKVCache { return this.pagedKV; }

  reset(batchSize: number): void {
    this.pagedKV.reset(batchSize);
    this.gdnState.reset();
  }

  free(): void {
    this.gdnState.free();
    this.pagedKV.free();
  }

  [Symbol.dispose](): void {
    this.free();
  }

  prefixMatch(seqIdx: number, inputIds: number[]): number[] {
    const batchSize = Math.max(this.pagedKV.sequences.length, 1);
    this.pagedKV.reset(batchSize);
    this.gdnState.reset();
    return inputIds.slice();
  }

  reportTokens(seqIdx: number, tokens: number[], targetToken?: number): void {
    this.pagedKV.reportTokens(seqIdx, tokens, targetToken);
  }

  prefillBatchPlanHook(
    _batchSize: number, _seqLens: number[], totalTokens: number,
  ): void {
    if (totalTokens > this.maxBatch * this.maxSeqLen) {
      throw new Error(`Total tokens ${totalTokens} exceeds max (B=${this.maxBatch}, S=${this.maxSeqLen})`);
    }
  }
}

export type { SamplingParams };

const QWEN35_REPO = "Qwen/Qwen3.5-0.8B";

export interface Qwen35Config extends CommonModelConfig {
  partialRotaryFactor: number;
  mropeSection: number[];
  mropeInterleaved: boolean;
  attnOutputGate: boolean;
  linearNumKeyHeads: number;
  linearKeyHeadDim: number;
  linearNumValueHeads: number;
  linearValueHeadDim: number;
  linearConvKernelDim: number;
  layerTypes: string[];
  numFullAttnLayers: number;
  numGdnLayers: number;
  fullAttnLayerIndices: number[];
  eosTokenIds: number[];
}

function loadConfig(modelDir: string): Qwen35Config {
  const raw = JSON.parse(fs.readFileSync(path.join(modelDir, "config.json"), "utf-8"));
  const tc = raw.text_config ?? raw;
  const layerTypes: string[] = tc.layer_types ?? [];
  const numFullAttnLayers = layerTypes.filter((t: string) => t === "full_attention").length;
  const numGdnLayers = layerTypes.filter((t: string) => t === "linear_attention").length;
  const fullAttnLayerIndices = layerTypes.map((t: string, i: number) => t === "full_attention" ? i : -1).filter((i: number) => i >= 0);
  const ropeParams = tc.rope_parameters ?? {};
  const eosTokenIds = Array.isArray(raw.eos_token_id) ? raw.eos_token_id
    : Array.isArray(tc.eos_token_id) ? tc.eos_token_id
    : [raw.eos_token_id ?? tc.eos_token_id ?? 248044];
  return {
    hiddenSize: tc.hidden_size,
    intermediateSize: tc.intermediate_size,
    numHiddenLayers: tc.num_hidden_layers,
    rmsNormEps: tc.rms_norm_eps ?? 1e-6,
    vocabSize: tc.vocab_size,
    tieWordEmbeddings: tc.tie_word_embeddings ?? false,
    numAttentionHeads: tc.num_attention_heads,
    numKeyValueHeads: tc.num_key_value_heads,
    headDim: tc.head_dim,
    ropeTheta: ropeParams.rope_theta ?? 1000000,
    numKeyValueGroups: tc.num_attention_heads / tc.num_key_value_heads,
    scaling: Math.pow(tc.head_dim, -0.5),
    partialRotaryFactor: ropeParams.partial_rotary_factor ?? 1.0,
    mropeSection: ropeParams.mrope_section ?? [],
    mropeInterleaved: ropeParams.mrope_interleaved ?? false,
    attnOutputGate: tc.attn_output_gate ?? false,
    linearNumKeyHeads: tc.linear_num_key_heads ?? 16,
    linearKeyHeadDim: tc.linear_key_head_dim ?? 128,
    linearNumValueHeads: tc.linear_num_value_heads ?? 16,
    linearValueHeadDim: tc.linear_value_head_dim ?? 128,
    linearConvKernelDim: tc.linear_conv_kernel_dim ?? 4,
    layerTypes,
    numFullAttnLayers,
    numGdnLayers,
    fullAttnLayerIndices,
    eosTokenIds,
  };
}

export class Qwen35Model extends ChatModel {
  static readonly WEIGHT_PREFIX = "model.language_model.";
  readonly eosIds: Set<number>;
  cfg: Qwen35Config;
  invFreq: Tensor;

  private constructor(glm: DeviceOps, config: Qwen35Config) {
    super(glm);
    this.cfg = config;
    this.eosIds = new Set(config.eosTokenIds);
    const ropeDim = Math.floor(config.headDim * config.partialRotaryFactor);
    this.invFreq = this.initInvFreq(ropeDim, config.ropeTheta);
  }

  static async fromPretrained(glm: DeviceOps, repoIdOrDir: string = QWEN35_REPO): Promise<Qwen35Model> {
    const modelDir = fs.existsSync(repoIdOrDir) ? repoIdOrDir : resolveModelPath(repoIdOrDir);
    const config = loadConfig(modelDir);
    const model = new Qwen35Model(glm, config);
    await model.fromPretrained(modelDir, QWEN35_REPO);
    return model;
  }

  private weightParallelism(name: string): TensorParallelism {
    if (name === "lm_head.weight") return TensorParallelism.Column;
    if (name === `${Qwen35Model.WEIGHT_PREFIX}embed_tokens.weight`) return TensorParallelism.Row;
    if (name.endsWith(".self_attn.q_proj.weight") ||
        name.endsWith(".self_attn.k_proj.weight") ||
        name.endsWith(".self_attn.v_proj.weight") ||
        name.endsWith(".mlp.gate_proj.weight") ||
        name.endsWith(".mlp.up_proj.weight") ||
        name.endsWith(".linear_attn.in_proj_qkv.weight") ||
        name.endsWith(".linear_attn.in_proj_a.weight") ||
        name.endsWith(".linear_attn.in_proj_b.weight") ||
        name.endsWith(".linear_attn.in_proj_z.weight") ||
        name.endsWith(".linear_attn.conv1d.weight") ||
        name.endsWith(".linear_attn.A_log") ||
        name.endsWith(".linear_attn.dt_bias")) return TensorParallelism.Column;
    if (name.endsWith(".self_attn.o_proj.weight") ||
        name.endsWith(".mlp.down_proj.weight") ||
        name.endsWith(".linear_attn.out_proj.weight")) return TensorParallelism.Row;
    return TensorParallelism.Replicated;
  }

  protected async loadTensor(name: string, meta: TensorMeta, st: SafeTensorFile, mmapPtr: number): Promise<void> {
    const prefix = Qwen35Model.WEIGHT_PREFIX;
    const par = this.weightParallelism(name);
    const gemmaNormSuffixes = [
      "input_layernorm.weight",
      "post_attention_layernorm.weight",
      "q_norm.weight",
      "k_norm.weight",
    ];
    const isGemmaNorm = name === `${prefix}norm.weight` ||
      gemmaNormSuffixes.some(s => name.endsWith(s));

    const isGdnQkv = name.endsWith(".linear_attn.in_proj_qkv.weight");
    const isGdnConv1d = name.endsWith(".linear_attn.conv1d.weight");

    if (name.includes("A_log") || name.includes("dt_bias")) {
      const numElements = meta.shape.reduce((a, b) => a * b, 1);
      const tensor = this.alloc(meta.shape, "F32", name, par);
      if (meta.dtype === "F32") {
        const offset = st.dataStart + meta.dataOffsets[0];
        await tensor.mmapLoad(mmapPtr, offset, tensor.bytes);
      } else {
        const rawBytes = st.readTensor(name);
        const f32Arr = new Float32Array(numElements);
        for (let i = 0; i < numElements; i++) {
          const u16 = rawBytes.readUInt16LE(i * 2);
          const u32 = u16 << 16;
          f32Arr[i] = new Float32Array(new Uint32Array([u32]).buffer)[0];
        }
        tensor.h2d(Buffer.from(f32Arr.buffer));
      }
    } else if (isGdnQkv || isGdnConv1d) {
      const dtype = meta.dtype === "F32" ? "F32" : meta.dtype;
      const tensor = this.alloc(meta.shape, dtype, name, par);
      const offset = st.dataStart + meta.dataOffsets[0];
      const numHeads = this.cfg.linearNumKeyHeads;
      const dK = this.cfg.linearKeyHeadDim;
      const dV = this.cfg.linearValueHeadDim;
      const inner = meta.shape.slice(1).reduce((a, b) => a * b, 1);
      const bpr = inner * (dtype === "F32" ? 4 : 2);
      const qRows = numHeads * dK;
      const strided: StridedMmap = { srcOffset: 0, dstOffset: 0, srcPitch: bpr, dstPitch: bpr, width: bpr, height: qRows };
      await tensor.mmapLoad(mmapPtr, offset, tensor.bytes, strided);
      await tensor.mmapLoad(mmapPtr, offset, tensor.bytes, { ...strided, srcOffset: qRows * bpr, dstOffset: qRows * bpr, height: numHeads * dK });
      await tensor.mmapLoad(mmapPtr, offset, tensor.bytes, { ...strided, srcOffset: 2 * qRows * bpr, dstOffset: 2 * qRows * bpr, height: numHeads * dV });
    } else if (meta.dtype === "F32") {
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
      await tensor.mmapLoad(mmapPtr, offset, tensor.bytes);

      if (this.cfg.tieWordEmbeddings && name === `${prefix}embed_tokens.weight` && !this.tensors.has("lm_head.weight")) {
        const lmHead = this.alloc(meta.shape, dtype, "lm_head.weight", TensorParallelism.Column);
        await lmHead.mmapLoad(mmapPtr, offset, lmHead.bytes);
      }
    }
  }

  createChatCache(maxPages = 256, maxBatch = 1, maxSeqLen = 4096, _pageSize = 16): ChatCache {
    const pagedKV = new PagedKVCache(this.glm, this.cfg.numKeyValueHeads, this.cfg.headDim, this.cfg.numFullAttnLayers, maxPages, maxBatch);
    const gdnState = new Qwen35GdnState(this.glm, this.cfg, maxBatch);
    return new Qwen35ChatCache(pagedKV, gdnState, maxBatch, maxSeqLen);
  }

  private fullAttnCacheIdx(layerIdx: number): number {
    let idx = 0;
    for (let i = 0; i < layerIdx; i++) {
      if (this.cfg.layerTypes[i] === "full_attention") idx++;
    }
    return idx;
  }

  private mlp(normed: Tensor, pfx: string, BS: number): Tensor {
    return normed.swiGluMlp(this.swiGluMlpWeights(`${pfx}.mlp`));
  }

  private *gdnLayerPrefillPhased(state: ExecutionState, normed: Tensor, residual: Tensor, layerIdx: number, gdnState: Qwen35GdnState): Generator<void, { normed: Tensor, residual: Tensor }, void> {
    const cfg = this.cfg;
    const ws = state.ws;
    const hs = cfg.hiddenSize;
    const linKDim = cfg.linearKeyHeadDim;
    const linVDim = cfg.linearValueHeadDim;
    const fullLinHeads = cfg.linearNumKeyHeads;
    const fullConvDim = fullLinHeads * (linKDim * 2 + linVDim);
    const fullZDim = fullLinHeads * linVDim;
    const fullConvStateStride = fullConvDim * (cfg.linearConvKernelDim - 1);
    const fullRecurrentStateStride = fullLinHeads * linKDim * linVDim;
    const pfx = `${Qwen35Model.WEIGHT_PREFIX}layers.${layerIdx}.linear_attn`;
    const S = state.totalTokens;
    const BS = S;
    const batchSize = gdnState.batchSize;

    using qkvLinear = normed.linear(this.tensors.get(`${pfx}.in_proj_qkv.weight`)!);
    using aBuf = normed.linear(this.tensors.get(`${pfx}.in_proj_a.weight`)!);
    using bBuf = normed.linear(this.tensors.get(`${pfx}.in_proj_b.weight`)!);
    using zBuf = normed.linear(this.tensors.get(`${pfx}.in_proj_z.weight`)!);

    const convState = gdnState.convState[layerIdx];
    const recurrentState = gdnState.recurrentState[layerIdx];
    const kernelSize = cfg.linearConvKernelDim;

    using convOut = ws.alloc([S, fullConvDim], "BF16", undefined, TensorParallelism.Row);
    convOut.causalConv1d(convState, qkvLinear, this.tensors.get(`${pfx}.conv1d.weight`)!, state.qoIndptrD, fullConvDim, S, kernelSize, batchSize, fullConvStateStride, 1, fullConvDim);

    using gdnOut = ws.alloc([S, fullZDim], "BF16", undefined, TensorParallelism.Row);

    ws.glm.gdnPrefill(
      state, gdnOut, recurrentState, convOut,
      aBuf, bBuf,
      this.tensors.get(`${pfx}.A_log`)!, this.tensors.get(`${pfx}.dt_bias`)!,
      state.qoIndptrD, fullLinHeads, linKDim, linVDim,
      fullRecurrentStateStride, 1, fullConvDim,
    );

    using gatedOut = ws.alloc([S, fullZDim], "BF16", undefined, TensorParallelism.Row);
    using reshapedGdnOut = gdnOut.reshape([S * fullLinHeads, linVDim]);
    using reshapedZBuf = zBuf.reshape([S * fullLinHeads, linVDim]);
    gatedOut.rmsnormGated(reshapedGdnOut, reshapedZBuf, this.tensors.get(`${pfx}.norm.weight`)!, cfg.rmsNormEps);

    using oProjBuf = gatedOut.linear(this.tensors.get(`${pfx}.out_proj.weight`)!);

    yield;
    const attnResult = residual.fusedAddRmsnorm(oProjBuf, this.tensors.get(`${Qwen35Model.WEIGHT_PREFIX}layers.${layerIdx}.post_attention_layernorm.weight`)!, cfg.rmsNormEps);
    using attnNormed = attnResult.normed;
    using attnResidual = attnResult.residual;
    yield;

    using downBuf = this.mlp(attnNormed, `${Qwen35Model.WEIGHT_PREFIX}layers.${layerIdx}`, BS);
    const nextWeight = layerIdx < cfg.numHiddenLayers - 1
      ? this.tensors.get(`${Qwen35Model.WEIGHT_PREFIX}layers.${layerIdx + 1}.input_layernorm.weight`)!
      : this.tensors.get(`${Qwen35Model.WEIGHT_PREFIX}norm.weight`)!;
    yield;
    const mlpResult = attnResidual.fusedAddRmsnorm(downBuf, nextWeight, cfg.rmsNormEps);
    yield;
    return { normed: mlpResult.normed, residual: mlpResult.residual };
  }

  private *gdnLayerDecodePhased(state: ExecutionState, normed: Tensor, residual: Tensor, layerIdx: number, gdnState: Qwen35GdnState): Generator<void, { normed: Tensor, residual: Tensor }, void> {
    const cfg = this.cfg;
    const ws = state.ws;
    const hs = cfg.hiddenSize;
    const linKDim = cfg.linearKeyHeadDim;
    const linVDim = cfg.linearValueHeadDim;
    const fullLinHeads = cfg.linearNumKeyHeads;
    const fullConvDim = fullLinHeads * (linKDim * 2 + linVDim);
    const fullZDim = fullLinHeads * linVDim;
    const fullConvStateStride = fullConvDim * (cfg.linearConvKernelDim - 1);
    const fullRecurrentStateStride = fullLinHeads * linKDim * linVDim;
    const pfx = `${Qwen35Model.WEIGHT_PREFIX}layers.${layerIdx}.linear_attn`;
    const BS = state.batchSize;

    using qkvBuf = normed.linear(this.tensors.get(`${pfx}.in_proj_qkv.weight`)!);
    using aBuf = normed.linear(this.tensors.get(`${pfx}.in_proj_a.weight`)!);
    using bBuf = normed.linear(this.tensors.get(`${pfx}.in_proj_b.weight`)!);
    using zBuf = normed.linear(this.tensors.get(`${pfx}.in_proj_z.weight`)!);

    const convState = gdnState.convState[layerIdx];
    const recurrentState = gdnState.recurrentState[layerIdx];
    const kernelSize = cfg.linearConvKernelDim;

    using convOut = qkvBuf.causalConv1dUpdate(convState, qkvBuf, this.tensors.get(`${pfx}.conv1d.weight`)!, fullConvDim, kernelSize, BS, fullConvStateStride);

    using gdnOut = ws.alloc([BS, fullZDim], "BF16", undefined, TensorParallelism.Row);

    ws.glm.gdnRecurrentStep(
      state, gdnOut, recurrentState, convOut,
      aBuf, bBuf,
      this.tensors.get(`${pfx}.A_log`)!, this.tensors.get(`${pfx}.dt_bias`)!,
      fullLinHeads, linKDim, linVDim,
      fullRecurrentStateStride, 1, fullConvDim,
    );

    using gatedOut = ws.alloc([BS, fullZDim], "BF16", undefined, TensorParallelism.Row);
    using reshapedGdnOut = gdnOut.reshape([BS * fullLinHeads, linVDim]);
    using reshapedZBuf = zBuf.reshape([BS * fullLinHeads, linVDim]);
    gatedOut.rmsnormGated(reshapedGdnOut, reshapedZBuf, this.tensors.get(`${pfx}.norm.weight`)!, cfg.rmsNormEps);

    using oProjBuf = gatedOut.linear(this.tensors.get(`${pfx}.out_proj.weight`)!);

    yield;
    const attnResult = residual.fusedAddRmsnorm(oProjBuf, this.tensors.get(`${Qwen35Model.WEIGHT_PREFIX}layers.${layerIdx}.post_attention_layernorm.weight`)!, cfg.rmsNormEps);
    using attnNormed = attnResult.normed;
    using attnResidual = attnResult.residual;
    yield;

    using downBuf = this.mlp(attnNormed, `${Qwen35Model.WEIGHT_PREFIX}layers.${layerIdx}`, BS);
    const nextWeight = layerIdx < cfg.numHiddenLayers - 1
      ? this.tensors.get(`${Qwen35Model.WEIGHT_PREFIX}layers.${layerIdx + 1}.input_layernorm.weight`)!
      : this.tensors.get(`${Qwen35Model.WEIGHT_PREFIX}norm.weight`)!;
    yield;
    const mlpResult = attnResidual.fusedAddRmsnorm(downBuf, nextWeight, cfg.rmsNormEps);
    yield;
    return { normed: mlpResult.normed, residual: mlpResult.residual };
  }

  private *fullAttnLayerPhased(normed: Tensor, residual: Tensor, layerIdx: number, state: ExecutionState, cos: Tensor, sin: Tensor): Generator<void, { normed: Tensor, residual: Tensor }, void> {
    const cfg = this.cfg;
    const ws = state.ws;
    const hs = cfg.hiddenSize;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const cacheIdx = this.fullAttnCacheIdx(layerIdx);
    const pfx = `${Qwen35Model.WEIGHT_PREFIX}layers.${layerIdx}.self_attn`;
    const batchSize = state.batchSize;
    const totalTokens = state.totalTokens;
    const BS = totalTokens;
    const B = state.isDecode ? batchSize : 1;
    const S = state.isDecode ? 1 : totalTokens;

    using qBuf = normed.linear(this.tensors.get(`${pfx}.q_proj.weight`)!);
    using kBuf = normed.linear(this.tensors.get(`${pfx}.k_proj.weight`)!);
    using vBuf = normed.linear(this.tensors.get(`${pfx}.v_proj.weight`)!);

    const ropeDim = Math.floor(hd * cfg.partialRotaryFactor);
    using qRope = qBuf.fusedNormRope(this.tensors.get(`${pfx}.q_norm.weight`)!, cos, sin, cfg.rmsNormEps, ropeDim, S, B, hd * 2);
    using kRope = kBuf.fusedNormRope(this.tensors.get(`${pfx}.k_norm.weight`)!, cos, sin, cfg.rmsNormEps, ropeDim, S, B);

    state.kvCacheWrite(kRope, vBuf, cacheIdx, nKv, hd);

    using flashOut = new UsingHolder<Tensor>(undefined!);
    if (state.isDecode) {
      flashOut.replace(ws.flashDecode(state, qRope, cacheIdx, nHeads, nKv, hd, cfg.scaling));
    } else {
      const qStrideN = hd;
      const qStrideH = BS * hd;
      flashOut.replace(ws.flashPrefillPaged(state, qRope, cacheIdx, nHeads, nKv, hd, qStrideN, qStrideH, 1, cfg.scaling));
    }

    if (cfg.attnOutputGate) {
      flashOut.value.gateSigmoidMul(qBuf, nHeads, hd);
    }

    using reshapedFlashOut = flashOut.value.reshape([BS, nHeads * hd]);
    using oProjBuf = reshapedFlashOut.outputProj(this.tensors.get(`${pfx}.o_proj.weight`)!);
    yield;
    const attnResult = residual.fusedAddRmsnorm(oProjBuf, this.tensors.get(`${Qwen35Model.WEIGHT_PREFIX}layers.${layerIdx}.post_attention_layernorm.weight`)!, cfg.rmsNormEps);
    using attnNormed = attnResult.normed;
    using attnResidual = attnResult.residual;
    yield;

    using downBuf = this.mlp(attnNormed, `${Qwen35Model.WEIGHT_PREFIX}layers.${layerIdx}`, BS);
    const nextWeight = layerIdx < cfg.numHiddenLayers - 1
      ? this.tensors.get(`${Qwen35Model.WEIGHT_PREFIX}layers.${layerIdx + 1}.input_layernorm.weight`)!
      : this.tensors.get(`${Qwen35Model.WEIGHT_PREFIX}norm.weight`)!;
    yield;
    const mlpResult = attnResidual.fusedAddRmsnorm(downBuf, nextWeight, cfg.rmsNormEps);
    yield;
    return { normed: mlpResult.normed, residual: mlpResult.residual };
  }

  *forwardPhased(state: ExecutionState): Generator<void, Tensor, void> {
    const ws = state.ws;
    const cache = state.cache as Qwen35ChatCache;
    const { gdnState } = cache;
    const cfg = this.cfg;
    const hs = cfg.hiddenSize;
    const hd = cfg.headDim;
    const batchSize = state.batchSize;
    const totalTokens = state.totalTokens;
    const BS = totalTokens;
    const B = state.isDecode ? batchSize : 1;
    const S = state.isDecode ? 1 : totalTokens;

    const embedTable = this.tensors.get(`${Qwen35Model.WEIGHT_PREFIX}embed_tokens.weight`)!;
    using residual = new UsingHolder(state.embedding(embedTable));

    const ropeDim = Math.floor(hd * cfg.partialRotaryFactor);
    const rotaryEmbedding = state.rotaryEmbedding(this.invFreq);
    using cos = rotaryEmbedding.cos;
    using sin = rotaryEmbedding.sin;

    using normed = new UsingHolder(residual.value.rmsnorm(this.tensors.get(`${Qwen35Model.WEIGHT_PREFIX}layers.0.input_layernorm.weight`)!, cfg.rmsNormEps));

    for (let i = 0; i < cfg.numHiddenLayers; i++) {
      let result: { normed: Tensor, residual: Tensor };
      if (cfg.layerTypes[i] === "linear_attention") {
        if (state.isDecode) {
          result = yield* this.gdnLayerDecodePhased(state, normed.value, residual.value, i, gdnState);
        } else {
          result = yield* this.gdnLayerPrefillPhased(state, normed.value, residual.value, i, gdnState);
        }
      } else {
        result = yield* this.fullAttnLayerPhased(normed.value, residual.value, i, state, cos, sin);
      }
      normed.replace(result.normed);
      residual.replace(result.residual);
    }

    return normed.detach();
  }
}
