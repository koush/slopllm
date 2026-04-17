import fs from "node:fs";
import path from "node:path";
import { GlmOps, f32ToBf16Bytes, bf16BytesToF32, BF16, I32, F32, SAMPLING_MAX_TOPK, BATCH_FLOAT_WS_SIZE, BATCH_INT_WS_SIZE, BATCH_PINNED_INT_WS_SIZE, PAGE_SIZE } from "./glm_ops";
import { SafeTensorFile } from "./safetensors";
import { resolveModelPath } from "./model_path";
import { PagedKVCache, WorkspaceBuffers } from "./paged_kv";
import { Tensor, OpContext } from "./tensor";
import { Qwen35GdnState } from "./qwen35_gdn_state";
import type { ChatModel, ChatCache, DecodeState, PrefillState } from "./chat_model";
import { SamplingParams } from "./chat_model";

class Qwen35ChatCache implements ChatCache {
  constructor(
    public readonly pagedKV: PagedKVCache,
    public readonly gdnState: Qwen35GdnState,
  ) {}

  reset(batchSize: number): void {
    this.pagedKV.reset(batchSize);
    this.gdnState.reset();
  }

  free(): void {
    this.gdnState.free();
    this.pagedKV.free();
  }

  prefixMatch(seqIdx: number, inputIds: number[]): number[] {
    const batchSize = Math.max(this.pagedKV.seqPages.length, 1);
    this.pagedKV.reset(batchSize);
    this.gdnState.reset();
    return inputIds.slice();
  }

  appendTokens(seqIdx: number, tokens: number[]): void {
    this.pagedKV.appendTokens(seqIdx, tokens);
  }
}

export type { SamplingParams };

const QWEN35_REPO = "Qwen/Qwen3.5-0.8B";
const EOS_TOKEN_IDS = new Set([248044]);

export interface Qwen35Config {
  hiddenSize: number;
  intermediateSize: number;
  numHiddenLayers: number;
  rmsNormEps: number;
  vocabSize: number;
  tieWordEmbeddings: boolean;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  partialRotaryFactor: number;
  mropeSection: number[];
  mropeInterleaved: boolean;
  ropeTheta: number;
  attnOutputGate: boolean;
  linearNumKeyHeads: number;
  linearKeyHeadDim: number;
  linearNumValueHeads: number;
  linearValueHeadDim: number;
  linearConvKernelDim: number;
  layerTypes: string[];
  numKeyValueGroups: number;
  scaling: number;
  numFullAttnLayers: number;
  numGdnLayers: number;
  fullAttnLayerIndices: number[];
}

function loadConfig(modelDir: string): Qwen35Config {
  const raw = JSON.parse(fs.readFileSync(path.join(modelDir, "config.json"), "utf-8"));
  const tc = raw.text_config ?? raw;
  const layerTypes: string[] = tc.layer_types ?? [];
  const numFullAttnLayers = layerTypes.filter((t: string) => t === "full_attention").length;
  const numGdnLayers = layerTypes.filter((t: string) => t === "linear_attention").length;
  const fullAttnLayerIndices = layerTypes.map((t: string, i: number) => t === "full_attention" ? i : -1).filter((i: number) => i >= 0);
  const ropeParams = tc.rope_parameters ?? {};
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
    partialRotaryFactor: ropeParams.partial_rotary_factor ?? 1.0,
    mropeSection: ropeParams.mrope_section ?? [],
    mropeInterleaved: ropeParams.mrope_interleaved ?? false,
    ropeTheta: ropeParams.rope_theta ?? 1000000,
    attnOutputGate: tc.attn_output_gate ?? false,
    linearNumKeyHeads: tc.linear_num_key_heads ?? 16,
    linearKeyHeadDim: tc.linear_key_head_dim ?? 128,
    linearNumValueHeads: tc.linear_num_value_heads ?? 16,
    linearValueHeadDim: tc.linear_value_head_dim ?? 128,
    linearConvKernelDim: tc.linear_conv_kernel_dim ?? 4,
    layerTypes,
    numKeyValueGroups: tc.num_attention_heads / tc.num_key_value_heads,
    scaling: Math.pow(tc.head_dim, -0.5),
    numFullAttnLayers,
    numGdnLayers,
    fullAttnLayerIndices,
  };
}

class Qwen35Workspace {
  hiddenA: Tensor;
  hiddenB: Tensor;
  normed: Tensor;
  gateBuf: Tensor;
  upBuf: Tensor;
  siluBuf: Tensor;
  downBuf: Tensor;
  logitsBuf: Tensor;
  hiddenLast: Tensor;
  lastIdx: Tensor;
  argmaxIdx: Tensor;
  positionIds: Tensor;
  cos: Tensor;
  sin: Tensor;
  inputIdsBuf: Tensor;
  flashOut: Tensor;
  flashTmp: Tensor;
  oProjBuf: Tensor;
  decodeId: Tensor;
  gdnQkvBuf: Tensor;
  gdnABuf: Tensor;
  gdnBBuf: Tensor;
  gdnZBuf: Tensor;
  gdnQBuf: Tensor;
  gdnKBuf: Tensor;
  gdnVBuf: Tensor;
  gdnOut: Tensor;
  gdnGatedOut: Tensor;
  gdnPrefillQkvLinear: Tensor;
  gdnPrefillQkvBuf: Tensor;
  gdnPrefillABuf: Tensor;
  gdnPrefillBBuf: Tensor;
  gdnPrefillZBuf: Tensor;
  gdnPrefillConvOut: Tensor;
  gdnPrefillQBuf: Tensor;
  gdnPrefillKBuf: Tensor;
  gdnPrefillVBuf: Tensor;
  gdnPrefillOut: Tensor;
  gdnPrefillGatedOut: Tensor;
  attnQBuf: Tensor;
  attnQOnly: Tensor;
  attnKBuf: Tensor;
  attnVBuf: Tensor;
  attnGateBuf: Tensor;
  attnQNormed: Tensor;
  attnKNormed: Tensor;
  attnQT: Tensor;
  attnKT: Tensor;
  attnVT: Tensor;
  attnQRope: Tensor;
  attnKRope: Tensor;
  attnSigBuf: Tensor;
  sampleOutToken: Tensor;
  sampleTopkVals: Tensor;
  sampleTopkIdxs: Tensor;
  sampleWorkspace: Tensor;
  samplePenaltyTokens: Tensor;
  qoIndptrD: Tensor;
  prefillSlotMapping: Tensor;
  tensors = new Map<string, Tensor>();

  constructor(glm: GlmOps, B: number, S: number, cfg: Qwen35Config) {
    const hs = cfg.hiddenSize;
    const inter = cfg.intermediateSize;
    const vs = cfg.vocabSize;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const linHeads = cfg.linearNumKeyHeads;
    const linKDim = cfg.linearKeyHeadDim;
    const linVDim = cfg.linearValueHeadDim;
    const convDim = linHeads * (linKDim * 2 + linVDim);
    const zDim = linHeads * linVDim;
    const BS = B * S;

    this.hiddenA = Tensor.alloc(glm, [B, S, hs], "BF16");
    this.hiddenB = Tensor.alloc(glm, [B, S, hs], "BF16");
    this.normed = Tensor.alloc(glm, [B, S, hs], "BF16");
    this.gateBuf = Tensor.alloc(glm, [B, S, inter], "BF16");
    this.upBuf = Tensor.alloc(glm, [B, S, inter], "BF16");
    this.siluBuf = Tensor.alloc(glm, [B, S, inter], "BF16");
    this.downBuf = Tensor.alloc(glm, [B, S, hs], "BF16");
    this.logitsBuf = Tensor.alloc(glm, [B, vs], "BF16");
    this.hiddenLast = Tensor.alloc(glm, [B, hs], "BF16");
    this.lastIdx = Tensor.alloc(glm, [B], "I32");
    this.argmaxIdx = Tensor.alloc(glm, [B], "I32");
    this.positionIds = Tensor.alloc(glm, [B * S], "I32");
    this.cos = Tensor.alloc(glm, [B, S, hd], "BF16");
    this.sin = Tensor.alloc(glm, [B, S, hd], "BF16");
    this.inputIdsBuf = Tensor.alloc(glm, [B * S], "I32");
    this.flashOut = Tensor.alloc(glm, [B, nHeads, S, hd], "BF16");
    this.flashTmp = Tensor.alloc(glm, [32 * 1024 * 1024], "U8");
    this.oProjBuf = Tensor.alloc(glm, [B, S, hs], "BF16");
    this.decodeId = Tensor.alloc(glm, [1], "I32");

    this.gdnQkvBuf = Tensor.alloc(glm, [B * convDim], "BF16");
    this.gdnABuf = Tensor.alloc(glm, [B * linHeads], "BF16");
    this.gdnBBuf = Tensor.alloc(glm, [B * linHeads], "BF16");
    this.gdnZBuf = Tensor.alloc(glm, [B * zDim], "BF16");
    this.gdnQBuf = Tensor.alloc(glm, [B * linHeads * linKDim], "BF16");
    this.gdnKBuf = Tensor.alloc(glm, [B * linHeads * linKDim], "BF16");
    this.gdnVBuf = Tensor.alloc(glm, [B * linHeads * linVDim], "BF16");
    this.gdnOut = Tensor.alloc(glm, [B * linHeads * linVDim], "BF16");
    this.gdnGatedOut = Tensor.alloc(glm, [B * linHeads * linVDim], "BF16");

    this.gdnPrefillQkvLinear = Tensor.alloc(glm, [BS, convDim], "BF16");
    this.gdnPrefillQkvBuf = Tensor.alloc(glm, [convDim, S], "BF16");
    this.gdnPrefillABuf = Tensor.alloc(glm, [BS * linHeads], "BF16");
    this.gdnPrefillBBuf = Tensor.alloc(glm, [BS * linHeads], "BF16");
    this.gdnPrefillZBuf = Tensor.alloc(glm, [BS, zDim], "BF16");
    this.gdnPrefillConvOut = Tensor.alloc(glm, [convDim, S], "BF16");
    this.gdnPrefillQBuf = Tensor.alloc(glm, [S * linHeads * linKDim], "BF16");
    this.gdnPrefillKBuf = Tensor.alloc(glm, [S * linHeads * linKDim], "BF16");
    this.gdnPrefillVBuf = Tensor.alloc(glm, [S * linHeads * linVDim], "BF16");
    this.gdnPrefillOut = Tensor.alloc(glm, [S * linHeads, linVDim], "BF16");
    this.gdnPrefillGatedOut = Tensor.alloc(glm, [S * linHeads, linVDim], "BF16");

    const qTotalDim = nHeads * hd;
    this.attnQBuf = Tensor.alloc(glm, [BS, qTotalDim * 2], "BF16");
    this.attnQOnly = Tensor.alloc(glm, [BS, qTotalDim], "BF16");
    this.attnKBuf = Tensor.alloc(glm, [BS, nKv * hd], "BF16");
    this.attnVBuf = Tensor.alloc(glm, [BS, nKv * hd], "BF16");
    this.attnGateBuf = Tensor.alloc(glm, [BS, qTotalDim], "BF16");
    this.attnQNormed = Tensor.alloc(glm, [BS, qTotalDim], "BF16");
    this.attnKNormed = Tensor.alloc(glm, [BS, nKv * hd], "BF16");
    this.attnQT = Tensor.alloc(glm, [B, nHeads, S, hd], "BF16");
    this.attnKT = Tensor.alloc(glm, [B, nKv, S, hd], "BF16");
    this.attnVT = Tensor.alloc(glm, [B, nKv, S, hd], "BF16");
    this.attnQRope = Tensor.alloc(glm, [B, nHeads, S, hd], "BF16");
    this.attnKRope = Tensor.alloc(glm, [B, nKv, S, hd], "BF16");
    this.attnSigBuf = Tensor.alloc(glm, [BS * nHeads * hd], "BF16");

    this.sampleOutToken = Tensor.alloc(glm, [1], "I32");
    this.sampleTopkVals = Tensor.alloc(glm, [SAMPLING_MAX_TOPK * 256], "F32");
    this.sampleTopkIdxs = Tensor.alloc(glm, [SAMPLING_MAX_TOPK * 256], "I32");
    this.sampleWorkspace = Tensor.alloc(glm, [vs], "F32");
    this.samplePenaltyTokens = Tensor.alloc(glm, [1024], "I32");

    this.qoIndptrD = Tensor.alloc(glm, [B + 1], "I32");
    this.prefillSlotMapping = Tensor.alloc(glm, [B * S], "I32");

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


export class Qwen35Model implements OpContext, ChatModel {
  readonly eosIds = new Set([248044]);
  glm: GlmOps;
  cfg: Qwen35Config;
  weights: Map<string, Tensor>;
  maxBatch: number;
  maxSeqLen: number;
  invFreq: Tensor;
  ws: Qwen35Workspace;

  private constructor(glm: GlmOps, config: Qwen35Config, weights: Map<string, Tensor>, maxBatch: number, maxSeqLen: number) {
    this.glm = glm;
    this.cfg = config;
    this.weights = weights;
    this.maxBatch = maxBatch;
    this.maxSeqLen = maxSeqLen;

    const ropeDim = Math.floor(config.headDim * config.partialRotaryFactor);
    const halfRopeDim = ropeDim / 2;
    const invFreqF32 = new Float32Array(halfRopeDim);
    for (let i = 0; i < halfRopeDim; i++) {
      invFreqF32[i] = 1.0 / Math.pow(config.ropeTheta, (2 * i) / ropeDim);
    }
    this.invFreq = Tensor.alloc(glm, [halfRopeDim], "BF16");
    this.invFreq.h2d(f32ToBf16Bytes(invFreqF32));

    this.ws = new Qwen35Workspace(glm, maxBatch, maxSeqLen, config);
  }

  static fromPretrained(glm: GlmOps, repoId: string = QWEN35_REPO, maxBatch = 1, maxSeqLen = 4096): Qwen35Model {
    const modelDir = resolveModelPath(repoId);
    const config = loadConfig(modelDir);

    const stFiles = fs.readdirSync(modelDir).filter(f => f.endsWith('.safetensors') || f.endsWith('.safetensors.json'));
    const shards: string[] = [];
    if (stFiles.some(f => f === 'model.safetensors')) {
      shards.push(path.join(modelDir, 'model.safetensors'));
    } else {
      const indexFile = stFiles.find(f => f.endsWith('.json'));
      if (indexFile) {
        const idx = JSON.parse(fs.readFileSync(path.join(modelDir, indexFile), 'utf-8'));
        for (const f of Object.keys(idx.weight_map ?? idx)) {
          if (f.endsWith('.safetensors') && !shards.includes(path.join(modelDir, f))) {
            shards.push(path.join(modelDir, f));
          }
        }
      } else {
        shards.push(...stFiles.filter(f => f.endsWith('.safetensors')).map(f => path.join(modelDir, f)));
      }
    }

    const weights = new Map<string, Tensor>();
    const prefix = "model.language_model.";
    const gemmaNormSuffixes = [
      "input_layernorm.weight",
      "post_attention_layernorm.weight",
      "q_norm.weight",
      "k_norm.weight",
    ];

    for (const stPath of shards) {
      const st = SafeTensorFile.open(stPath);
      const mmapPtr = glm.mmapOpen(stPath);
      const fileSize = fs.statSync(stPath).size;

      for (const name of st.tensorNames()) {
        let weightName = name.startsWith(prefix) ? name.slice(prefix.length) : name;
        const meta = st.meta(name);
        const isGemmaNorm = weightName === "norm.weight" ||
          gemmaNormSuffixes.some(s => weightName.endsWith(s));

        if (weightName.includes("A_log") || weightName.includes("dt_bias")) {
          const numElements = meta.shape.reduce((a, b) => a * b, 1);
          const tensor = Tensor.alloc(glm, meta.shape, "F32", weightName);
          if (meta.dtype === "F32") {
            const offset = st.dataStart + meta.dataOffsets[0];
            glm.mmapLoad(tensor.data, mmapPtr, offset, tensor.bytes);
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
          weights.set(weightName, tensor);
        } else if (meta.dtype === "F32") {
          const numElements = meta.shape.reduce((a, b) => a * b, 1);
          const tensor = Tensor.alloc(glm, meta.shape, "BF16", weightName);
          const f32Bytes = st.readTensor(name);
          const f32Arr = new Float32Array(f32Bytes.buffer, f32Bytes.byteOffset, numElements);
          if (isGemmaNorm) {
            for (let i = 0; i < numElements; i++) f32Arr[i] += 1.0;
          }
          tensor.h2d(f32ToBf16Bytes(f32Arr));
          weights.set(weightName, tensor);
        } else if (isGemmaNorm) {
          const numElements = meta.shape.reduce((a, b) => a * b, 1);
          const tensor = Tensor.alloc(glm, meta.shape, "BF16", weightName);
          const rawBytes = st.readTensor(name);
          const f32Arr = new Float32Array(numElements);
          for (let i = 0; i < numElements; i++) {
            const u16 = rawBytes.readUInt16LE(i * 2);
            const u32 = u16 << 16;
            f32Arr[i] = (new Float32Array(new Uint32Array([u32]).buffer)[0]) + 1.0;
          }
          tensor.h2d(f32ToBf16Bytes(f32Arr));
          weights.set(weightName, tensor);
        } else {
          const dtype = meta.dtype === "F32" ? "F32" : meta.dtype;
          const tensor = Tensor.alloc(glm, meta.shape, dtype, weightName);
          const offset = st.dataStart + meta.dataOffsets[0];
          glm.mmapLoad(tensor.data, mmapPtr, offset, tensor.bytes);
          weights.set(weightName, tensor);
        }
      }

      glm.synchronize();
      st.close();
      glm.mmapClose(mmapPtr, fileSize);
    }

    if (config.tieWordEmbeddings && !weights.has("lm_head.weight")) {
      const embedTensor = weights.get("embed_tokens.weight")!;
      weights.set("lm_head.weight", embedTensor);
    }

    return new Qwen35Model(glm, config, weights, maxBatch, maxSeqLen);
  }

  free(): void {
    this.ws.free();
    this.invFreq.free();
    for (const tensor of this.weights.values()) {
      if (tensor !== this.weights.get("lm_head.weight") || !this.cfg.tieWordEmbeddings) {
        tensor.free();
      }
    }
    this.weights = new Map();
  }

  createGdnState(batchSize = 1): Qwen35GdnState {
    return new Qwen35GdnState(this.glm, this.cfg, batchSize);
  }

  createChatCache(maxPages = 256): ChatCache {
    const pagedKV = new PagedKVCache(this.glm, this.cfg.numKeyValueHeads, this.cfg.headDim, this.cfg.numFullAttnLayers, maxPages, this.maxBatch);
    const gdnState = this.createGdnState(this.maxBatch);
    return new Qwen35ChatCache(pagedKV, gdnState);
  }

  *chatStream(inputIds: number[][], cache: ChatCache, ws: WorkspaceBuffers, maxNewTokens = 100, eosIds?: Set<number>, sampling?: SamplingParams): Generator<number> {
    if (!(cache instanceof Qwen35ChatCache)) throw new Error("Expected Qwen35ChatCache");
    yield* this.streamTokens(inputIds, ws, cache, maxNewTokens, eosIds ?? this.eosIds, sampling);
  }

  private fullAttnCacheIdx(layerIdx: number): number {
    let idx = 0;
    for (let i = 0; i < layerIdx; i++) {
      if (this.cfg.layerTypes[i] === "full_attention") idx++;
    }
    return idx;
  }

  private readArgmax(ptr: Tensor | number, count: number): number {
    this.ws.argmaxIdx.argmax(ptr, count);
    const buf = Buffer.alloc(I32);
    this.ws.argmaxIdx.d2h(buf);
    return buf.readInt32LE(0);
  }

  private readArgmaxBatch(batchSize: number): number[] {
    const vs = this.cfg.vocabSize;
    this.ws.argmaxIdx.argmax(this.ws.logitsBuf, vs, batchSize);
    const buf = Buffer.alloc(batchSize * I32);
    this.ws.argmaxIdx.d2h(buf);
    const result: number[] = [];
    for (let i = 0; i < batchSize; i++) {
      result.push(buf.readInt32LE(i * I32));
    }
    return result;
  }

  readLogits(): Float32Array {
    const vs = this.cfg.vocabSize;
    const buf = Buffer.alloc(vs * BF16);
    this.glm.d2h(buf, this.ws.logitsBuf.data, vs * BF16);
    return bf16BytesToF32(buf);
  }

  private sampleTokenGPU(params: SamplingParams, tokenHistory: number[]): number {
    const vs = this.cfg.vocabSize;
    const glm = this.glm;

    const hasRepPenalty = params.repetitionPenalty !== 1.0;
    const hasPresPenalty = params.presencePenalty !== 0;

    let numPenaltyTokens = 0;
    if (hasRepPenalty || hasPresPenalty) {
      const seen = new Set<number>();
      const start = Math.max(0, tokenHistory.length - params.repetitionPenaltyWindow);
      for (let i = start; i < tokenHistory.length; i++) seen.add(tokenHistory[i]);
      const penaltyBuf = Buffer.alloc(seen.size * I32);
      let offset = 0;
      for (const tid of seen) {
        if (tid < vs) {
          penaltyBuf.writeInt32LE(tid, offset);
          offset += I32;
          numPenaltyTokens++;
        }
      }
      if (numPenaltyTokens > 0) {
        this.ws.samplePenaltyTokens.h2d(penaltyBuf, numPenaltyTokens * I32);
      }
    }

    const randomVal = Math.random();
    const topK = params.topK > 0 ? params.topK : 0;
    const temperature = params.temperature > 0 ? params.temperature : 0;

    glm.sample(
      this.ws.sampleOutToken.data,
      this.ws.sampleTopkVals.data,
      this.ws.sampleTopkIdxs.data,
      this.ws.sampleWorkspace.data,
      this.ws.logitsBuf.data,
      this.ws.samplePenaltyTokens.data,
      vs,
      numPenaltyTokens,
      temperature,
      params.repetitionPenalty,
      params.presencePenalty,
      topK,
      params.topP,
      randomVal,
    );

    const buf = Buffer.alloc(I32);
    this.ws.sampleOutToken.d2h(buf);
    return buf.readInt32LE(0);
  }

  private finalNormAndLogits(count: number, src?: Tensor): void {
    const cfg = this.cfg;
    const hs = cfg.hiddenSize;
    const vs = cfg.vocabSize;
    this.ws.normed.rmsnorm(src ?? this.ws.hiddenA, this.weights.get("norm.weight")!, cfg.rmsNormEps, hs, count);
    this.ws.logitsBuf.linear(this.ws.normed, this.weights.get("lm_head.weight")!, count, vs, hs, this);
  }

  private extractLastLogits(count: number, lastIndicesBuf: Buffer): void {
    this.ws.lastIdx.h2d(lastIndicesBuf);
    this.ws.hiddenLast.indexSelect(this.ws.hiddenA, this.ws.lastIdx, this.cfg.hiddenSize, count);
    this.finalNormAndLogits(count, this.ws.hiddenLast);
  }

  private mlp(pfx: string, BS: number): void {
    const cfg = this.cfg;
    const hs = cfg.hiddenSize;
    const inter = cfg.intermediateSize;
    this.ws.gateBuf.linear(this.ws.normed, this.weights.get(`${pfx}.mlp.gate_proj.weight`)!, BS, inter, hs, this);
    this.ws.upBuf.linear(this.ws.normed, this.weights.get(`${pfx}.mlp.up_proj.weight`)!, BS, inter, hs, this);
    this.ws.siluBuf.siluAndMul(this.ws.gateBuf, this.ws.upBuf, inter, BS);
    this.ws.downBuf.linear(this.ws.siluBuf, this.weights.get(`${pfx}.mlp.down_proj.weight`)!, BS, hs, inter, this);
  }

  private gdnLayerPrefill(layerIdx: number, S: number, gdnState: Qwen35GdnState): void {
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;
    const linHeads = cfg.linearNumKeyHeads;
    const linKDim = cfg.linearKeyHeadDim;
    const linVDim = cfg.linearValueHeadDim;
    const convDim = linHeads * (linKDim * 2 + linVDim);
    const zDim = linHeads * linVDim;
    const pfx = `layers.${layerIdx}.linear_attn`;
    const BS = S;
    const batchSize = gdnState.batchSize;

    this.ws.normed.rmsnorm(this.ws.hiddenA, this.weights.get(`layers.${layerIdx}.input_layernorm.weight`)!, cfg.rmsNormEps, hs, BS);

    const qkvLinear = this.ws.gdnPrefillQkvLinear;
    const qkvBuf = this.ws.gdnPrefillQkvBuf;
    const aBuf = this.ws.gdnPrefillABuf;
    const bBuf = this.ws.gdnPrefillBBuf;
    const zBuf = this.ws.gdnPrefillZBuf;

    qkvLinear.linear(this.ws.normed, this.weights.get(`${pfx}.in_proj_qkv.weight`)!, BS, convDim, hs, this);
    aBuf.linear(this.ws.normed, this.weights.get(`${pfx}.in_proj_a.weight`)!, BS, linHeads, hs, this);
    bBuf.linear(this.ws.normed, this.weights.get(`${pfx}.in_proj_b.weight`)!, BS, linHeads, hs, this);
    zBuf.linear(this.ws.normed, this.weights.get(`${pfx}.in_proj_z.weight`)!, BS, zDim, hs, this);

    qkvBuf.transpose4d(qkvLinear, 1, S, convDim, 1, 0, 2, 1, 3);

    const convState = gdnState.convState[layerIdx];
    const recurrentState = gdnState.recurrentState[layerIdx];
    const kernelSize = cfg.linearConvKernelDim;

    const convOut = this.ws.gdnPrefillConvOut;
    glm.causalConv1d(convOut.data, convState.data, qkvBuf.data, this.weights.get(`${pfx}.conv1d.weight`)!.data, gdnState.cuSeqlens.data, convDim, S, kernelSize, batchSize, gdnState.convStateStride);

    const qBuf = this.ws.gdnPrefillQBuf;
    const kBuf = this.ws.gdnPrefillKBuf;
    const vBuf = this.ws.gdnPrefillVBuf;

    qBuf.qkvSplit(kBuf, vBuf, convOut, S, linHeads, linKDim, linVDim);

    const gdnOut = this.ws.gdnPrefillOut;

    glm.gdnPrefill(
      gdnOut.data, recurrentState.data,
      qBuf.data, kBuf.data, vBuf.data,
      aBuf.data, bBuf.data,
      this.weights.get(`${pfx}.A_log`)!.data,
      this.weights.get(`${pfx}.dt_bias`)!.data,
      gdnState.cuSeqlens.data, S, linHeads, linKDim, linVDim,
      batchSize, gdnState.recurrentStateStride,
    );

    const gatedOut = this.ws.gdnPrefillGatedOut;
    gatedOut.rmsnormGated(gdnOut, zBuf, this.weights.get(`${pfx}.norm.weight`)!, cfg.rmsNormEps, linVDim, S * linHeads);

    this.ws.oProjBuf.linear(gatedOut, this.weights.get(`${pfx}.out_proj.weight`)!, BS, hs, zDim, this);

    this.ws.hiddenB.add(this.ws.hiddenA, this.ws.oProjBuf, BS * hs);
    this.ws.normed.rmsnorm(this.ws.hiddenB, this.weights.get(`layers.${layerIdx}.post_attention_layernorm.weight`)!, cfg.rmsNormEps, hs, BS);
    this.mlp(`layers.${layerIdx}`, BS);
    this.ws.hiddenA.add(this.ws.hiddenB, this.ws.downBuf, BS * hs);
  }

  private gdnLayerDecode(layerIdx: number, gdnState: Qwen35GdnState): void {
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;
    const linHeads = cfg.linearNumKeyHeads;
    const linKDim = cfg.linearKeyHeadDim;
    const linVDim = cfg.linearValueHeadDim;
    const convDim = linHeads * (linKDim * 2 + linVDim);
    const zDim = linHeads * linVDim;
    const pfx = `layers.${layerIdx}.linear_attn`;
    const BS = gdnState.batchSize;

    this.ws.normed.rmsnorm(this.ws.hiddenA, this.weights.get(`layers.${layerIdx}.input_layernorm.weight`)!, cfg.rmsNormEps, hs, BS);

    const qkvBuf = this.ws.gdnQkvBuf;
    const aBuf = this.ws.gdnABuf;
    const bBuf = this.ws.gdnBBuf;
    const zBuf = this.ws.gdnZBuf;

    qkvBuf.linear(this.ws.normed, this.weights.get(`${pfx}.in_proj_qkv.weight`)!, BS, convDim, hs, this);
    aBuf.linear(this.ws.normed, this.weights.get(`${pfx}.in_proj_a.weight`)!, BS, linHeads, hs, this);
    bBuf.linear(this.ws.normed, this.weights.get(`${pfx}.in_proj_b.weight`)!, BS, linHeads, hs, this);
    zBuf.linear(this.ws.normed, this.weights.get(`${pfx}.in_proj_z.weight`)!, BS, zDim, hs, this);

    const convState = gdnState.convState[layerIdx];
    const recurrentState = gdnState.recurrentState[layerIdx];
    const kernelSize = cfg.linearConvKernelDim;

    glm.causalConv1dUpdate(qkvBuf.data, convState.data, qkvBuf.data, this.weights.get(`${pfx}.conv1d.weight`)!.data, convDim, kernelSize, BS, gdnState.convStateStride);

    const qkvT = this.ws.gdnPrefillQkvBuf;
    qkvT.transpose4d(qkvBuf, 1, BS, convDim, 1, 0, 2, 1, 3);

    const qBuf = this.ws.gdnQBuf;
    const kBuf = this.ws.gdnKBuf;
    const vBuf = this.ws.gdnVBuf;

    qBuf.qkvSplit(kBuf, vBuf, qkvT, BS, linHeads, linKDim, linVDim);

    const gdnOut = this.ws.gdnOut;

    glm.gdnRecurrentStep(
      gdnOut.data, recurrentState.data,
      qBuf.data, kBuf.data, vBuf.data,
      aBuf.data, bBuf.data,
      this.weights.get(`${pfx}.A_log`)!.data,
      this.weights.get(`${pfx}.dt_bias`)!.data,
      linHeads, linKDim, linVDim,
      BS, gdnState.recurrentStateStride,
    );

    const gatedOut = this.ws.gdnGatedOut;
    gatedOut.rmsnormGated(gdnOut, zBuf, this.weights.get(`${pfx}.norm.weight`)!, cfg.rmsNormEps, linVDim, BS * linHeads);

    this.ws.oProjBuf.linear(gatedOut, this.weights.get(`${pfx}.out_proj.weight`)!, BS, hs, zDim, this);

    this.ws.hiddenB.add(this.ws.hiddenA, this.ws.oProjBuf, BS * hs);
    this.ws.normed.rmsnorm(this.ws.hiddenB, this.weights.get(`layers.${layerIdx}.post_attention_layernorm.weight`)!, cfg.rmsNormEps, hs, BS);
    this.mlp(`layers.${layerIdx}`, BS);
    this.ws.hiddenA.add(this.ws.hiddenB, this.ws.downBuf, BS * hs);
  }

  private fullAttnLayerPrefillPaged(layerIdx: number, totalTokens: number, batchSize: number, pagedKV: PagedKVCache, ws: WorkspaceBuffers, gdnState: Qwen35GdnState): void {
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const pageSize = pagedKV.pageSize;
    const cacheIdx = this.fullAttnCacheIdx(layerIdx);
    const pfx = `layers.${layerIdx}.self_attn`;
    const qTotalDim = nHeads * hd;

    this.ws.normed.rmsnorm(this.ws.hiddenA, this.weights.get(`layers.${layerIdx}.input_layernorm.weight`)!, cfg.rmsNormEps, hs, totalTokens);

    const qBuf = this.ws.attnQBuf;
    const qOnly = this.ws.attnQOnly;
    const kBuf = this.ws.attnKBuf;
    const vBuf = this.ws.attnVBuf;
    const gateBuf = this.ws.attnGateBuf;

    qBuf.linear(this.ws.normed, this.weights.get(`${pfx}.q_proj.weight`)!, totalTokens, qTotalDim * 2, hs, this);
    glm.interleavedSplit(qOnly.data, gateBuf.data, qBuf.data, totalTokens, nHeads, hd);
    kBuf.linear(this.ws.normed, this.weights.get(`${pfx}.k_proj.weight`)!, totalTokens, nKv * hd, hs, this);
    vBuf.linear(this.ws.normed, this.weights.get(`${pfx}.v_proj.weight`)!, totalTokens, nKv * hd, hs, this);

    const qNormed = this.ws.attnQNormed;
    const kNormed = this.ws.attnKNormed;
    qNormed.rmsnorm(qOnly, this.weights.get(`${pfx}.q_norm.weight`)!, cfg.rmsNormEps, hd, totalTokens * nHeads);
    kNormed.rmsnorm(kBuf, this.weights.get(`${pfx}.k_norm.weight`)!, cfg.rmsNormEps, hd, totalTokens * nKv);

    const qT = this.ws.attnQT;
    const kT = this.ws.attnKT;
    const vT = this.ws.attnVT;

    qT.transpose4d(qNormed, 1, totalTokens, nHeads, hd, 0, 2, 1, 3);
    kT.transpose4d(kNormed, 1, totalTokens, nKv, hd, 0, 2, 1, 3);
    vT.transpose4d(vBuf, 1, totalTokens, nKv, hd, 0, 2, 1, 3);

    const ropeDim = Math.floor(hd * cfg.partialRotaryFactor);
    const qRope = this.ws.attnQRope;
    const kRope = this.ws.attnKRope;

    qRope.applyRotaryPosEmbPartial(qT, this.ws.cos, this.ws.sin, ropeDim, hd, nHeads, totalTokens, 1, 1);
    kRope.applyRotaryPosEmbPartial(kT, this.ws.cos, this.ws.sin, ropeDim, hd, nKv, totalTokens, 1, 1);

    glm.kvCacheWrite(
      kRope.data, vT.data,
      pagedKV.kData[cacheIdx], pagedKV.vData[cacheIdx],
      this.ws.prefillSlotMapping.data,
      totalTokens, nKv, hd, pageSize,
      hd, totalTokens * hd
    );

    const qStrideN = hd;
    const qStrideH = totalTokens * hd;

    glm.batchPrefillPagedRun(
      qRope.data, this.ws.flashOut.data,
      pagedKV.kData[cacheIdx], pagedKV.vData[cacheIdx],
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

    if (cfg.attnOutputGate) {
      const sigBuf = this.ws.attnSigBuf;
      sigBuf.sigmoid(gateBuf, totalTokens * nHeads * hd);
      this.ws.flashOut.mul(this.ws.flashOut, sigBuf, totalTokens * nHeads * hd);
    }

    this.ws.oProjBuf.linear(this.ws.flashOut, this.weights.get(`${pfx}.o_proj.weight`)!, totalTokens, hs, nHeads * hd, this);

    this.ws.hiddenB.add(this.ws.hiddenA, this.ws.oProjBuf, totalTokens * hs);
    this.ws.normed.rmsnorm(this.ws.hiddenB, this.weights.get(`layers.${layerIdx}.post_attention_layernorm.weight`)!, cfg.rmsNormEps, hs, totalTokens);
    this.mlp(`layers.${layerIdx}`, totalTokens);
    this.ws.hiddenA.add(this.ws.hiddenB, this.ws.downBuf, totalTokens * hs);
  }

  private fullAttnLayerDecodePaged(layerIdx: number, batchSize: number, pagedKV: PagedKVCache, ws: WorkspaceBuffers, gdnState: Qwen35GdnState): void {
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const pageSize = pagedKV.pageSize;
    const cacheIdx = this.fullAttnCacheIdx(layerIdx);
    const pfx = `layers.${layerIdx}.self_attn`;
    const BS = batchSize;

    this.ws.normed.rmsnorm(this.ws.hiddenA, this.weights.get(`layers.${layerIdx}.input_layernorm.weight`)!, cfg.rmsNormEps, hs, BS);

    const qTotalDim = nHeads * hd;
    const qBuf = this.ws.attnQBuf;
    const qOnly = this.ws.attnQOnly;
    const kBuf = this.ws.attnKBuf;
    const vBuf = this.ws.attnVBuf;
    const gateBuf = this.ws.attnGateBuf;

    qBuf.linear(this.ws.normed, this.weights.get(`${pfx}.q_proj.weight`)!, BS, qTotalDim * 2, hs, this);
    glm.interleavedSplit(qOnly.data, gateBuf.data, qBuf.data, BS, nHeads, hd);
    kBuf.linear(this.ws.normed, this.weights.get(`${pfx}.k_proj.weight`)!, BS, nKv * hd, hs, this);
    vBuf.linear(this.ws.normed, this.weights.get(`${pfx}.v_proj.weight`)!, BS, nKv * hd, hs, this);

    const qNormed = this.ws.attnQNormed;
    const kNormed = this.ws.attnKNormed;
    qNormed.rmsnorm(qOnly, this.weights.get(`${pfx}.q_norm.weight`)!, cfg.rmsNormEps, hd, BS * nHeads);
    kNormed.rmsnorm(kBuf, this.weights.get(`${pfx}.k_norm.weight`)!, cfg.rmsNormEps, hd, BS * nKv);

    const qT = this.ws.attnQT;
    const kT = this.ws.attnKT;
    const vT = this.ws.attnVT;

    qT.transpose4d(qNormed, BS, 1, nHeads, hd, 0, 2, 1, 3);
    kT.transpose4d(kNormed, BS, 1, nKv, hd, 0, 2, 1, 3);
    vT.transpose4d(vBuf, BS, 1, nKv, hd, 0, 2, 1, 3);

    const ropeDim = Math.floor(hd * cfg.partialRotaryFactor);
    const qRope = this.ws.attnQRope;
    const kRope = this.ws.attnKRope;

    qRope.applyRotaryPosEmbPartial(qT, this.ws.cos, this.ws.sin, ropeDim, hd, nHeads, 1, BS, 1);
    kRope.applyRotaryPosEmbPartial(kT, this.ws.cos, this.ws.sin, ropeDim, hd, nKv, 1, BS, 1);

    glm.kvCacheWrite(
      kRope.data, vT.data,
      pagedKV.kData[cacheIdx], pagedKV.vData[cacheIdx],
      pagedKV.slotMapping,
      BS, nKv, hd, pageSize,
      nKv * hd, hd
    );

    glm.batchDecodeRun(
      qRope.data, this.ws.flashOut.data,
      pagedKV.kData[cacheIdx], pagedKV.vData[cacheIdx],
      pagedKV.indices, pagedKV.indptrD, pagedKV.lastPageLen,
      ws.floatWs, ws.intWs,
      ws.decodePlanInfo,
      BS,
      nHeads, nKv, hd, pageSize,
      cfg.scaling
    );

    if (cfg.attnOutputGate) {
      const sigBuf = this.ws.attnSigBuf;
      sigBuf.sigmoid(gateBuf, BS * nHeads * hd);
      this.ws.flashOut.mul(this.ws.flashOut, sigBuf, BS * nHeads * hd);
    }

    this.ws.oProjBuf.linear(this.ws.flashOut, this.weights.get(`${pfx}.o_proj.weight`)!, BS, hs, nHeads * hd, this);

    this.ws.hiddenB.add(this.ws.hiddenA, this.ws.oProjBuf, BS * hs);
    this.ws.normed.rmsnorm(this.ws.hiddenB, this.weights.get(`layers.${layerIdx}.post_attention_layernorm.weight`)!, cfg.rmsNormEps, hs, BS);
    this.mlp(`layers.${layerIdx}`, BS);
    this.ws.hiddenA.add(this.ws.hiddenB, this.ws.downBuf, BS * hs);
  }

  prefillBatchPlan(inputIdsList: number[][], ws: WorkspaceBuffers, cache: ChatCache): PrefillState {
    if (!(cache instanceof Qwen35ChatCache)) throw new Error("Expected Qwen35ChatCache");
    const { pagedKV, gdnState } = cache;
    const cfg = this.cfg;
    const glm = this.glm;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const pageSize = pagedKV.pageSize;
    const batchSize = inputIdsList.length;

    if (pagedKV.seqPages.length !== batchSize) {
      throw new Error(`prefillBatchPlan: pagedKV has ${pagedKV.seqPages.length} sequences, expected ${batchSize}`);
    }

    const seqLens = inputIdsList.map(ids => ids.length);
    const totalTokens = seqLens.reduce((a, b) => a + b, 0);
    const startPos = pagedKV.seqKvLens.slice();

    gdnState.uploadCuSeqlens(seqLens);

    if (totalTokens > this.maxBatch * this.maxSeqLen) {
      throw new Error(`Total tokens ${totalTokens} exceeds max (B=${this.maxBatch}, S=${this.maxSeqLen})`);
    }

    const pageAllocs: [number, number][] = [];
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      pageAllocs.push(pagedKV.allocAppendPages(seqIdx, seqLens[seqIdx]));
    }

    const allIds: number[] = [];
    for (const ids of inputIdsList) allIds.push(...ids);
    const idsBuf = Int32Array.from(allIds);
    this.ws.inputIdsBuf.h2d(Buffer.from(idsBuf.buffer, idsBuf.byteOffset, idsBuf.byteLength));

    const qoIndptr = [0];
    for (const s of seqLens) {
      qoIndptr.push(qoIndptr[qoIndptr.length - 1] + s);
    }
    const qoIndptrBuf = Int32Array.from(qoIndptr);

    pagedKV.updateIndptr();

    const posIds: number[] = [];
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      for (let p = 0; p < seqLens[seqIdx]; p++) {
        posIds.push(startPos[seqIdx] + p);
      }
    }
    const posIdsBuf = Int32Array.from(posIds);
    this.ws.positionIds.h2d(Buffer.from(posIdsBuf.buffer, posIdsBuf.byteOffset, posIdsBuf.byteLength));

    const lastIndices: number[] = [];
    let offset = 0;
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      lastIndices.push(offset + seqLens[seqIdx] - 1);
      offset += seqLens[seqIdx];
    }
    const lastIdxBuf = Int32Array.from(lastIndices);
    this.ws.lastIdx.h2d(Buffer.from(lastIdxBuf.buffer, lastIdxBuf.byteOffset, lastIdxBuf.byteLength));

    const qoIndptrHostPtr = glm.allocPinned((batchSize + 1) * I32);
    glm.writePinned(qoIndptrHostPtr, Buffer.from(qoIndptrBuf.buffer, qoIndptrBuf.byteOffset, qoIndptrBuf.byteLength));

    glm.batchPrefillPagedPlan(
      ws.floatWs, BATCH_FLOAT_WS_SIZE,
      ws.intWs, ws.pinnedIntWs, BATCH_INT_WS_SIZE,
      ws.prefillPlanInfo,
      qoIndptrHostPtr, pagedKV.indptrH,
      totalTokens, batchSize,
      nHeads, nKv, hd,
      pageSize,
      1
    );

    glm.freePinned(qoIndptrHostPtr);

    this.ws.qoIndptrD.h2d(Buffer.from(qoIndptrBuf.buffer, qoIndptrBuf.byteOffset, qoIndptrBuf.byteLength));

    const slotMapping: number[] = [];
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      const pages = pagedKV.seqPages[seqIdx];
      for (let pos = 0; pos < seqLens[seqIdx]; pos++) {
        const kvPos = startPos[seqIdx] + pos;
        const pageIdxInSeq = Math.floor(kvPos / pagedKV.pageSize);
        const offsetInPage = kvPos % pagedKV.pageSize;
        const absPage = pages[pageIdxInSeq];
        slotMapping.push(absPage * pagedKV.pageSize + offsetInPage);
      }
    }
    const slotMappingBuf = Int32Array.from(slotMapping);
    this.ws.prefillSlotMapping.h2d(Buffer.from(slotMappingBuf.buffer, slotMappingBuf.byteOffset, slotMappingBuf.byteLength));

    return { batchSize, totalTokens, seqLens, pageAllocs };
  }

  prefillBatchForward(state: PrefillState, ws: WorkspaceBuffers, cache: ChatCache): void {
    if (!(cache instanceof Qwen35ChatCache)) throw new Error("Expected Qwen35ChatCache");
    const { pagedKV, gdnState } = cache;
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;
    const hd = cfg.headDim;
    const batchSize = state.batchSize;
    const totalTokens = state.totalTokens;

    this.ws.hiddenA.embedding(this.weights.get("embed_tokens.weight")!, this.ws.inputIdsBuf, hs, totalTokens);

    const ropeDim = Math.floor(hd * cfg.partialRotaryFactor);
    glm.rotaryEmbedding(this.ws.cos.data, this.ws.sin.data, this.invFreq.data, this.ws.positionIds.data, ropeDim / 2, 1, totalTokens);

    for (let i = 0; i < cfg.numHiddenLayers; i++) {
      if (cfg.layerTypes[i] === "linear_attention") {
        this.gdnLayerPrefill(i, totalTokens, gdnState);
      } else {
        this.fullAttnLayerPrefillPaged(i, totalTokens, batchSize, pagedKV, ws, gdnState);
      }
    }

    this.ws.hiddenLast.indexSelect(this.ws.hiddenA, this.ws.lastIdx, hs, batchSize);
    this.finalNormAndLogits(batchSize, this.ws.hiddenLast);

    this.ws.argmaxIdx.argmax(this.ws.logitsBuf, cfg.vocabSize, batchSize);
  }

  prefillBatchRead(state: PrefillState): number[] {
    const batchSize = state.batchSize;
    const buf = Buffer.alloc(batchSize * I32);
    this.ws.argmaxIdx.d2h(buf);
    const result: number[] = [];
    for (let i = 0; i < batchSize; i++) {
      result.push(buf.readInt32LE(i * I32));
    }
    return result;
  }

  prefillBatch(inputIdsList: number[][], ws: WorkspaceBuffers, cache: ChatCache): number[] {
    if (!(cache instanceof Qwen35ChatCache)) throw new Error("Expected Qwen35ChatCache");
    const state = this.prefillBatchPlan(inputIdsList, ws, cache);
    this.prefillBatchForward(state, ws, cache);
    return this.prefillBatchRead(state);
  }

  decodeBatchPlan(tokenIdsList: number[], ws: WorkspaceBuffers, cache: ChatCache, enableCudaGraph = false): DecodeState {
    if (!(cache instanceof Qwen35ChatCache)) throw new Error("Expected Qwen35ChatCache");
    const { pagedKV, gdnState } = cache;
    const cfg = this.cfg;
    const glm = this.glm;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const pageSize = pagedKV.pageSize;
    const batchSize = tokenIdsList.length;

    const writeLocations: [number, number][] = [];
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      writeLocations.push(pagedKV.allocDecodeToken(seqIdx));
    }

    pagedKV.updateIndptr();
    pagedKV.updateSlotMapping(writeLocations, pageSize);

    const idsBuf = Int32Array.from(tokenIdsList);
    this.ws.inputIdsBuf.h2d(Buffer.from(idsBuf.buffer, idsBuf.byteOffset, idsBuf.byteLength));

    const posIds = new Array(batchSize);
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      posIds[seqIdx] = pagedKV.seqKvLens[seqIdx] - 1;
    }
    const posIdsBuf = Int32Array.from(posIds);
    this.ws.positionIds.h2d(Buffer.from(posIdsBuf.buffer, posIdsBuf.byteOffset, posIdsBuf.byteLength));

    glm.batchDecodePlan(
      ws.floatWs, BATCH_FLOAT_WS_SIZE,
      ws.intWs, ws.pinnedIntWs, BATCH_INT_WS_SIZE,
      ws.decodePlanInfo,
      pagedKV.indptrH,
      batchSize,
      nHeads, nKv, hd, pageSize,
      enableCudaGraph
    );

    return { batchSize };
  }

  decodeBatchForward(state: DecodeState, ws: WorkspaceBuffers, cache: ChatCache): void {
    if (!(cache instanceof Qwen35ChatCache)) throw new Error("Expected Qwen35ChatCache");
    const { pagedKV, gdnState } = cache;
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;
    const hd = cfg.headDim;
    const batchSize = state.batchSize;

    this.ws.hiddenA.embedding(this.weights.get("embed_tokens.weight")!, this.ws.inputIdsBuf, hs, batchSize);

    const ropeDim = Math.floor(hd * cfg.partialRotaryFactor);
    glm.rotaryEmbedding(this.ws.cos.data, this.ws.sin.data, this.invFreq.data, this.ws.positionIds.data, ropeDim / 2, batchSize, 1);

    for (let i = 0; i < cfg.numHiddenLayers; i++) {
      if (cfg.layerTypes[i] === "linear_attention") {
        this.gdnLayerDecode(i, gdnState);
      } else {
        this.fullAttnLayerDecodePaged(i, batchSize, pagedKV, ws, gdnState);
      }
    }

    this.finalNormAndLogits(batchSize);

    this.ws.argmaxIdx.argmax(this.ws.logitsBuf, cfg.vocabSize, batchSize);
  }

  decodeBatchRead(state: DecodeState): number[] {
    const batchSize = state.batchSize;
    const buf = Buffer.alloc(batchSize * I32);
    this.ws.argmaxIdx.d2h(buf);
    const result: number[] = [];
    for (let i = 0; i < batchSize; i++) {
      result.push(buf.readInt32LE(i * I32));
    }
    return result;
  }

  decodeBatch(tokenIdsList: number[], ws: WorkspaceBuffers, cache: ChatCache): number[] {
    if (!(cache instanceof Qwen35ChatCache)) throw new Error("Expected Qwen35ChatCache");
    const state = this.decodeBatchPlan(tokenIdsList, ws, cache);
    this.decodeBatchForward(state, ws, cache);
    return this.decodeBatchRead(state);
  }

  prefill(inputIds: number[][], ws: WorkspaceBuffers, cache: ChatCache): number {
    if (!(cache instanceof Qwen35ChatCache)) throw new Error("Expected Qwen35ChatCache");
    return this.prefillBatch(inputIds, ws, cache)[0];
  }

  decode(tokenId: number, ws: WorkspaceBuffers, cache: ChatCache): number {
    return this.decodeBatch([tokenId], ws, cache)[0];
  }

  generateTokens(inputIds: number[][], ws: WorkspaceBuffers, cache: ChatCache, maxNewTokens = 100, eosTokenIds = EOS_TOKEN_IDS, sampling?: SamplingParams): number[] {
    return [...this.streamTokens(inputIds, ws, cache, maxNewTokens, eosTokenIds, sampling)];
  }

  *streamTokens(inputIds: number[][], ws: WorkspaceBuffers, cache: ChatCache, maxNewTokens = 100, eosTokenIds = EOS_TOKEN_IDS, sampling?: SamplingParams): Generator<number> {
    if (!(cache instanceof Qwen35ChatCache)) throw new Error("Expected Qwen35ChatCache");
    const { pagedKV, gdnState } = cache;
    const vs = this.cfg.vocabSize;
    if (inputIds.length !== 1) throw new Error("streamTokens only supports batch=1");

    pagedKV.reset(1);
    gdnState.reset();
    const firstTokens = this.prefillBatch(inputIds, ws, cache);
    let nextToken = firstTokens[0];
    yield nextToken;

    const tokenHistory = [...inputIds[0], nextToken];

    for (let i = 0; i < maxNewTokens - 1; i++) {
      if (eosTokenIds.has(nextToken)) break;

      const decodeTokens = this.decodeBatch([nextToken], ws, cache);
      nextToken = decodeTokens[0];

      if (sampling && (sampling.temperature > 0 || sampling.repetitionPenalty !== 1.0 || sampling.presencePenalty !== 0 || sampling.topK > 0 || sampling.topP < 1.0)) {
        nextToken = this.sampleTokenGPU(sampling, tokenHistory);
      }

      tokenHistory.push(nextToken);
      yield nextToken;
    }
  }

  generateBatch(inputIdsList: number[][], ws: WorkspaceBuffers, cache: ChatCache, maxNewTokens = 100, eosTokenIds = EOS_TOKEN_IDS): number[][] {
    if (!(cache instanceof Qwen35ChatCache)) throw new Error("Expected Qwen35ChatCache");
    const { pagedKV, gdnState } = cache;
    const batchSize = inputIdsList.length;
    pagedKV.reset(batchSize);
    gdnState.reset();
    const firstTokens = this.prefillBatch(inputIdsList, ws, cache);
    const results: number[][] = firstTokens.map(t => [t]);

    let current = firstTokens.slice();
    for (let step = 0; step < maxNewTokens - 1; step++) {
      const done = current.every((t, i) => eosTokenIds.has(t) && results[i].length > 1);
      if (done) break;
      current = this.decodeBatch(current, ws, cache);
      for (let i = 0; i < batchSize; i++) {
        if (!eosTokenIds.has(results[i][results[i].length - 1]) || results[i].length === 1) {
          results[i].push(current[i]);
        }
      }
    }
    return results;
  }
}
