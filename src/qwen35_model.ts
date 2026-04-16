import fs from "node:fs";
import path from "node:path";
import { GlmOps, f32ToBf16Bytes, bf16BytesToF32, BF16, I32 } from "./glm_ops";
import { SafeTensorFile } from "./safetensors";
import { resolveModelPath } from "./model_path";
import { PagedKVCache, WorkspaceBuffers } from "./paged_kv";
import { FlatKVCache } from "./flat_kv";
import { Tensor, OpContext } from "./tensor";

const QWEN35_REPO = "Qwen/Qwen3.5-0.8B";
const EOS_TOKEN_IDS = new Set([248044]);

export interface SamplingParams {
  temperature: number;
  topP: number;
  topK: number;
  repetitionPenalty: number;
  presencePenalty: number;
  repetitionPenaltyWindow: number;
}

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

    this.gdnQkvBuf = Tensor.alloc(glm, [convDim], "BF16");
    this.gdnABuf = Tensor.alloc(glm, [linHeads], "BF16");
    this.gdnBBuf = Tensor.alloc(glm, [linHeads], "BF16");
    this.gdnZBuf = Tensor.alloc(glm, [1, zDim], "BF16");
    this.gdnQBuf = Tensor.alloc(glm, [linHeads, linKDim], "BF16");
    this.gdnKBuf = Tensor.alloc(glm, [linHeads, linKDim], "BF16");
    this.gdnVBuf = Tensor.alloc(glm, [linHeads, linVDim], "BF16");
    this.gdnOut = Tensor.alloc(glm, [linHeads, linVDim], "BF16");
    this.gdnGatedOut = Tensor.alloc(glm, [linHeads, linVDim], "BF16");

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

export class Qwen35GdnState {
  convState: Tensor[];
  recurrentState: Tensor[];
  private glm: GlmOps;
  private cfg: Qwen35Config;

  constructor(glm: GlmOps, cfg: Qwen35Config) {
    this.glm = glm;
    this.cfg = cfg;
    const linHeads = cfg.linearNumKeyHeads;
    const linKDim = cfg.linearKeyHeadDim;
    const linVDim = cfg.linearValueHeadDim;
    const convDim = linHeads * (linKDim * 2 + linVDim);
    const kernelSize = cfg.linearConvKernelDim;
    this.convState = [];
    this.recurrentState = [];
    for (let i = 0; i < cfg.numHiddenLayers; i++) {
      if (cfg.layerTypes[i] === "linear_attention") {
        this.convState.push(Tensor.alloc(glm, [convDim * (kernelSize - 1)], "BF16"));
        this.recurrentState.push(Tensor.alloc(glm, [linHeads * linKDim * linVDim], "F32"));
      } else {
        this.convState.push(null!);
        this.recurrentState.push(null!);
      }
    }
    this.zeroStates();
  }

  private zeroStates(): void {
    const linHeads = this.cfg.linearNumKeyHeads;
    const linKDim = this.cfg.linearKeyHeadDim;
    const linVDim = this.cfg.linearValueHeadDim;
    const convDim = linHeads * (linKDim * 2 + linVDim);
    const kernelSize = this.cfg.linearConvKernelDim;
    for (let i = 0; i < this.cfg.numHiddenLayers; i++) {
      if (this.cfg.layerTypes[i] === "linear_attention") {
        this.glm.fill(this.recurrentState[i].data, 0, 2 * linHeads * linKDim * linVDim);
        this.glm.fill(this.convState[i].data, 0, convDim * (kernelSize - 1));
      }
    }
  }

  reset(): void {
    this.zeroStates();
  }

  free(): void {
    for (const t of this.convState) { if (t) t.free(); }
    for (const t of this.recurrentState) { if (t) t.free(); }
    this.convState = [];
    this.recurrentState = [];
  }
}

export interface Qwen35DecodeState {
  batchSize: number;
}

export interface Qwen35PrefillState {
  batchSize: number;
  totalTokens: number;
  seqLens: number[];
  pageAllocs: [number, number][];
}

export class Qwen35Model implements OpContext {
  glm: GlmOps;
  cfg: Qwen35Config;
  weights: Map<string, Tensor>;
  maxBatch: number;
  maxSeqLen: number;
  invFreq: Tensor;
  ws: Qwen35Workspace;
  gdnState: Qwen35GdnState;

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
    this.gdnState = new Qwen35GdnState(glm, config);
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
          // A_log and dt_bias are used as float32 by GDN kernels
          // A_log is stored as F32; dt_bias is stored as BF16 but must be uploaded as F32
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
    this.gdnState.free();
    this.invFreq.free();
    for (const tensor of this.weights.values()) {
      if (tensor !== this.weights.get("lm_head.weight") || !this.cfg.tieWordEmbeddings) {
        tensor.free();
      }
    }
    this.weights = new Map();
  }

  createFlatKVCache(): FlatKVCache {
    return new FlatKVCache(this.glm, this.cfg.numKeyValueHeads, this.cfg.headDim, this.cfg.numFullAttnLayers, this.maxBatch, this.maxSeqLen);
  }

  createPagedKVCache(maxPages = 256): PagedKVCache {
    return new PagedKVCache(this.glm, this.cfg.numKeyValueHeads, this.cfg.headDim, this.cfg.numFullAttnLayers, maxPages, this.maxBatch);
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

  sampleToken(logits: Float32Array, params: SamplingParams, tokenHistory: number[]): number {
    const vs = logits.length;

    // Repetition penalty: divide positive logits, multiply negative logits
    if (params.repetitionPenalty !== 1.0) {
      const seen = new Set<number>();
      const start = Math.max(0, tokenHistory.length - params.repetitionPenaltyWindow);
      for (let i = start; i < tokenHistory.length; i++) seen.add(tokenHistory[i]);
      for (const tid of seen) {
        if (tid < vs) {
          logits[tid] = logits[tid] > 0 ? logits[tid] / params.repetitionPenalty : logits[tid] * params.repetitionPenalty;
        }
      }
    }

    // Presence penalty: flat subtraction for tokens that appeared
    if (params.presencePenalty !== 0) {
      const seen = new Set<number>();
      const start = Math.max(0, tokenHistory.length - params.repetitionPenaltyWindow);
      for (let i = start; i < tokenHistory.length; i++) seen.add(tokenHistory[i]);
      for (const tid of seen) {
        if (tid < vs) logits[tid] -= params.presencePenalty;
      }
    }

    // Greedy: temperature <= 0 with no topK
    if (params.temperature <= 0 && params.topK <= 0) {
      let best = 0;
      for (let i = 1; i < vs; i++) {
        if (logits[i] > logits[best]) best = i;
      }
      return best;
    }

    // Temperature scaling
    const invTemp = params.temperature > 0 ? 1.0 / params.temperature : 1.0;
    let maxLogit = -Infinity;
    for (let i = 0; i < vs; i++) {
      logits[i] *= invTemp;
      if (logits[i] > maxLogit) maxLogit = logits[i];
    }

    // Top-K: zero out tokens beyond the k highest
    if (params.topK > 0 && params.topK < vs) {
      const indices = Array.from({ length: vs }, (_, i) => i);
      indices.sort((a, b) => logits[b] - logits[a]);
      const threshold = logits[indices[params.topK]];
      for (let i = 0; i < vs; i++) {
        if (logits[i] < threshold) logits[i] = -Infinity;
      }
      // Recompute maxLogit
      maxLogit = logits[indices[0]];
    }

    // Softmax
    let sumExp = 0;
    for (let i = 0; i < vs; i++) {
      const v = logits[i] - maxLogit;
      logits[i] = v > -30 ? Math.exp(v) : 0;
      sumExp += logits[i];
    }

    // Top-P: zero out tokens beyond cumulative probability threshold
    if (params.topP < 1.0 && params.topK <= 0) {
      const sorted = Array.from({ length: vs }, (_, i) => i).sort((a, b) => logits[b] - logits[a]);
      let cumSum = 0;
      let cutoff = vs;
      for (let i = 0; i < sorted.length; i++) {
        cumSum += logits[sorted[i]] / sumExp;
        if (cumSum > params.topP) {
          cutoff = i + 1;
          break;
        }
      }
      const allowed = new Set(sorted.slice(0, cutoff));
      let renorm = 0;
      for (let i = 0; i < vs; i++) {
        if (!allowed.has(i)) logits[i] = 0;
        else renorm += logits[i];
      }
      if (renorm > 0) {
        for (let i = 0; i < vs; i++) logits[i] /= renorm;
      }
    } else {
      for (let i = 0; i < vs; i++) logits[i] /= sumExp;
    }

    // Sample
    let r = Math.random();
    let cumSum = 0;
    for (let i = 0; i < vs; i++) {
      cumSum += logits[i];
      if (r <= cumSum) return i;
    }
    return vs - 1;
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

  private gdnLayerPrefill(layerIdx: number, S: number): void {
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

    const convState = this.gdnState.convState[layerIdx];
    const recurrentState = this.gdnState.recurrentState[layerIdx];
    const kernelSize = cfg.linearConvKernelDim;

    const convOut = this.ws.gdnPrefillConvOut;
    glm.causalConv1d(convOut.data, convState.data, qkvBuf.data, this.weights.get(`${pfx}.conv1d.weight`)!.data, convDim, S, kernelSize);

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
      S, linHeads, linKDim, linVDim,
    );

    const gatedOut = this.ws.gdnPrefillGatedOut;
    gatedOut.rmsnormGated(gdnOut, zBuf, this.weights.get(`${pfx}.norm.weight`)!, cfg.rmsNormEps, linVDim, S * linHeads);

    this.ws.oProjBuf.linear(gatedOut, this.weights.get(`${pfx}.out_proj.weight`)!, BS, hs, zDim, this);

    this.ws.hiddenB.add(this.ws.hiddenA, this.ws.oProjBuf, BS * hs);
    this.ws.normed.rmsnorm(this.ws.hiddenB, this.weights.get(`layers.${layerIdx}.post_attention_layernorm.weight`)!, cfg.rmsNormEps, hs, BS);
    this.mlp(`layers.${layerIdx}`, BS);
    this.ws.hiddenA.add(this.ws.hiddenB, this.ws.downBuf, BS * hs);
  }

  private gdnLayerDecode(layerIdx: number): void {
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;
    const linHeads = cfg.linearNumKeyHeads;
    const linKDim = cfg.linearKeyHeadDim;
    const linVDim = cfg.linearValueHeadDim;
    const convDim = linHeads * (linKDim * 2 + linVDim);
    const zDim = linHeads * linVDim;
    const pfx = `layers.${layerIdx}.linear_attn`;
    const BS = 1;

    this.ws.normed.rmsnorm(this.ws.hiddenA, this.weights.get(`layers.${layerIdx}.input_layernorm.weight`)!, cfg.rmsNormEps, hs, BS);

    const qkvBuf = this.ws.gdnQkvBuf;
    const aBuf = this.ws.gdnABuf;
    const bBuf = this.ws.gdnBBuf;
    const zBuf = this.ws.gdnZBuf;

    qkvBuf.linear(this.ws.normed, this.weights.get(`${pfx}.in_proj_qkv.weight`)!, BS, convDim, hs, this);
    aBuf.linear(this.ws.normed, this.weights.get(`${pfx}.in_proj_a.weight`)!, BS, linHeads, hs, this);
    bBuf.linear(this.ws.normed, this.weights.get(`${pfx}.in_proj_b.weight`)!, BS, linHeads, hs, this);
    zBuf.linear(this.ws.normed, this.weights.get(`${pfx}.in_proj_z.weight`)!, BS, zDim, hs, this);

    const convState = this.gdnState.convState[layerIdx];
    const recurrentState = this.gdnState.recurrentState[layerIdx];
    const kernelSize = cfg.linearConvKernelDim;

    glm.causalConv1dUpdate(qkvBuf.data, convState.data, qkvBuf.data, this.weights.get(`${pfx}.conv1d.weight`)!.data, convDim, kernelSize);

    const qBuf = this.ws.gdnQBuf;
    const kBuf = this.ws.gdnKBuf;
    const vBuf = this.ws.gdnVBuf;

    const keyDim = linHeads * linKDim;
    const valueDim = linHeads * linVDim;
    glm.memcpy(qBuf.data, qkvBuf.data, keyDim * BF16);
    glm.memcpy(kBuf.data, qkvBuf.data + keyDim * BF16, keyDim * BF16);
    glm.memcpy(vBuf.data, qkvBuf.data + keyDim * 2 * BF16, valueDim * BF16);

    const gdnOut = this.ws.gdnOut;

    glm.gdnRecurrentStep(
      gdnOut.data, recurrentState.data,
      qBuf.data, kBuf.data, vBuf.data,
      aBuf.data, bBuf.data,
      this.weights.get(`${pfx}.A_log`)!.data,
      this.weights.get(`${pfx}.dt_bias`)!.data,
      linHeads, linKDim, linVDim,
    );

    const gatedOut = this.ws.gdnGatedOut;
    gatedOut.rmsnormGated(gdnOut, zBuf, this.weights.get(`${pfx}.norm.weight`)!, cfg.rmsNormEps, linVDim, linHeads);

    this.ws.oProjBuf.linear(gatedOut, this.weights.get(`${pfx}.out_proj.weight`)!, BS, hs, zDim, this);

    this.ws.hiddenB.add(this.ws.hiddenA, this.ws.oProjBuf, BS * hs);
    this.ws.normed.rmsnorm(this.ws.hiddenB, this.weights.get(`layers.${layerIdx}.post_attention_layernorm.weight`)!, cfg.rmsNormEps, hs, BS);
    this.mlp(`layers.${layerIdx}`, BS);
    this.ws.hiddenA.add(this.ws.hiddenB, this.ws.downBuf, BS * hs);
  }

  private fullAttnLayerPrefillFlash(layerIdx: number, B: number, S: number, cache: FlatKVCache): void {
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const maxS = this.maxSeqLen;
    const BS = B * S;
    const cacheIdx = this.fullAttnCacheIdx(layerIdx);
    const pfx = `layers.${layerIdx}.self_attn`;
    const qTotalDim = nHeads * hd;

    this.ws.normed.rmsnorm(this.ws.hiddenA, this.weights.get(`layers.${layerIdx}.input_layernorm.weight`)!, cfg.rmsNormEps, hs, BS);

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

    qT.transpose4d(qNormed, B, S, nHeads, hd, 0, 2, 1, 3);
    kT.transpose4d(kNormed, B, S, nKv, hd, 0, 2, 1, 3);
    vT.transpose4d(vBuf, B, S, nKv, hd, 0, 2, 1, 3);

    const ropeDim = Math.floor(hd * cfg.partialRotaryFactor);
    const qRope = this.ws.attnQRope;
    const kRope = this.ws.attnKRope;

    qRope.applyRotaryPosEmbPartial(qT, this.ws.cos, this.ws.sin, ropeDim, hd, nHeads, S, B, 1);
    kRope.applyRotaryPosEmbPartial(kT, this.ws.cos, this.ws.sin, ropeDim, hd, nKv, S, B, 1);

    const kvStrideH = maxS * hd;
    const kvStrideN = hd;
    for (let h = 0; h < nKv; h++) {
      const srcOff = h * S * hd * BF16;
      const dstOff = h * maxS * hd * BF16;
      glm.memcpy(cache.kData[cacheIdx] + dstOff, kRope.data + srcOff, S * hd * BF16);
      glm.memcpy(cache.vData[cacheIdx] + dstOff, vT.data + srcOff, S * hd * BF16);
    }

    glm.flashPrefill(
      qRope.data, cache.kData[cacheIdx], cache.vData[cacheIdx],
      this.ws.flashOut.data, this.ws.flashTmp.data,
      S, S, nHeads, nKv, hd,
      hd, S * hd,
      kvStrideN, kvStrideH,
      kvStrideN, kvStrideH,
      1, 1, cfg.scaling,
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

  private fullAttnLayerDecodeFlash(layerIdx: number, cachedLen: number, cache: FlatKVCache): void {
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const maxS = this.maxSeqLen;
    const BS = 1;
    const S = 1;
    const totalLen = cachedLen + 1;
    const cacheIdx = this.fullAttnCacheIdx(layerIdx);
    const pfx = `layers.${layerIdx}.self_attn`;

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

    qT.transpose4d(qNormed, BS, S, nHeads, hd, 0, 2, 1, 3);
    kT.transpose4d(kNormed, BS, S, nKv, hd, 0, 2, 1, 3);
    vT.transpose4d(vBuf, BS, S, nKv, hd, 0, 2, 1, 3);

    const ropeDim = Math.floor(hd * cfg.partialRotaryFactor);
    const qRope = this.ws.attnQRope;
    const kRope = this.ws.attnKRope;

    qRope.applyRotaryPosEmbPartial(qT, this.ws.cos, this.ws.sin, ropeDim, hd, nHeads, S, BS, 1);
    kRope.applyRotaryPosEmbPartial(kT, this.ws.cos, this.ws.sin, ropeDim, hd, nKv, S, BS, 1);

    const kvStrideH = maxS * hd;
    const kvStrideN = hd;
    for (let h = 0; h < nKv; h++) {
      const srcOff = h * hd * BF16;
      const dstOff = (h * maxS * hd + cachedLen * hd) * BF16;
      glm.memcpy(cache.kData[cacheIdx] + dstOff, kRope.data + srcOff, hd * BF16);
      glm.memcpy(cache.vData[cacheIdx] + dstOff, vT.data + srcOff, hd * BF16);
    }

    glm.flashDecode(
      qRope.data, cache.kData[cacheIdx], cache.vData[cacheIdx],
      this.ws.flashOut.data, this.ws.flashTmp.data,
      totalLen, nHeads, nKv, hd,
      hd, S * hd,
      kvStrideN, kvStrideH,
      cfg.scaling,
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

  prefill(inputIds: number[][], cache: FlatKVCache): number {
    const B = inputIds.length;
    const S = inputIds[0].length;
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;
    const hd = cfg.headDim;
    const vs = cfg.vocabSize;
    const BS = B * S;

    if (B > this.maxBatch || S > this.maxSeqLen) {
      throw new Error(`input (B=${B}, S=${S}) exceeds max (B=${this.maxBatch}, S=${this.maxSeqLen})`);
    }
    if (cache.cachePos !== 0) {
      throw new Error("Cache must be reset before prefill");
    }

    this.gdnState.reset();

    const flat = new Int32Array(B * S);
    for (let b = 0; b < B; b++) {
      for (let s = 0; s < S; s++) {
        flat[b * S + s] = inputIds[b][s];
      }
    }
    this.ws.inputIdsBuf.h2d(Buffer.from(flat.buffer, flat.byteOffset, flat.byteLength));
    this.ws.hiddenA.embedding(this.weights.get("embed_tokens.weight")!, this.ws.inputIdsBuf, hs, BS);

    const ropeDim = Math.floor(hd * cfg.partialRotaryFactor);
    this.ws.positionIds.arange(0, 1, S);
    glm.rotaryEmbedding(this.ws.cos.data, this.ws.sin.data, this.invFreq.data, this.ws.positionIds.data, ropeDim / 2, B, S);

    for (let i = 0; i < cfg.numHiddenLayers; i++) {
      if (cfg.layerTypes[i] === "linear_attention") {
        this.gdnLayerPrefill(i, S);
      } else {
        this.fullAttnLayerPrefillFlash(i, B, S, cache);
      }
    }

    const lastIdxBuf = Int32Array.from([S - 1]);
    this.extractLastLogits(1, Buffer.from(lastIdxBuf.buffer, lastIdxBuf.byteOffset, lastIdxBuf.byteLength));

    cache.cachePos = S;

    return this.readArgmax(this.ws.logitsBuf, vs);
  }

  private decodeToken(tokenId: number, cache: FlatKVCache, cachedLen: number): void {
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;
    const hd = cfg.headDim;
    const vs = cfg.vocabSize;
    const BS = 1;

    const idsBuf = Buffer.alloc(I32);
    idsBuf.writeInt32LE(tokenId, 0);
    this.ws.decodeId.h2d(idsBuf);
    this.ws.hiddenA.embedding(this.weights.get("embed_tokens.weight")!, this.ws.decodeId, hs, BS);

    const ropeDim = Math.floor(hd * cfg.partialRotaryFactor);
    this.ws.positionIds.arange(cachedLen, 0, 1);
    glm.rotaryEmbedding(this.ws.cos.data, this.ws.sin.data, this.invFreq.data, this.ws.positionIds.data, ropeDim / 2, 1, 1);

    for (let i = 0; i < cfg.numHiddenLayers; i++) {
      if (cfg.layerTypes[i] === "linear_attention") {
        this.gdnLayerDecode(i);
      } else {
        this.fullAttnLayerDecodeFlash(i, cachedLen, cache);
      }
    }

    this.finalNormAndLogits(BS);
  }

  decode(tokenId: number, cache: FlatKVCache): number {
    const cachedLen = cache.cachePos;
    this.decodeToken(tokenId, cache, cachedLen);
    cache.cachePos = cachedLen + 1;
    return this.readArgmax(this.ws.logitsBuf, this.cfg.vocabSize);
  }

  generateTokens(inputIds: number[][], cache: FlatKVCache, maxNewTokens = 100, eosTokenIds = EOS_TOKEN_IDS, sampling?: SamplingParams): number[] {
    return [...this.streamTokens(inputIds, cache, maxNewTokens, eosTokenIds, sampling)];
  }

  *streamTokens(inputIds: number[][], cache: FlatKVCache, maxNewTokens = 100, eosTokenIds = EOS_TOKEN_IDS, sampling?: SamplingParams): Generator<number> {
    const vs = this.cfg.vocabSize;
    if (inputIds.length !== 1) throw new Error("streamTokens only supports batch=1");

    cache.reset();
    let nextToken = this.prefill(inputIds, cache);
    yield nextToken;

    const tokenHistory = [...inputIds[0], nextToken];

    for (let i = 0; i < maxNewTokens - 1; i++) {
      if (eosTokenIds.has(nextToken)) break;
      const cachedLen = cache.cachePos;
      this.decodeToken(nextToken, cache, cachedLen);
      cache.cachePos = cachedLen + 1;

      if (sampling && (sampling.temperature > 0 || sampling.repetitionPenalty !== 1.0)) {
        const logits = this.readLogits();
        nextToken = this.sampleToken(logits, sampling, tokenHistory);
      } else {
        nextToken = this.readArgmax(this.ws.logitsBuf, vs);
      }
      tokenHistory.push(nextToken);
      yield nextToken;
    }
  }

  prefillBatch(inputIdsList: number[][], ws: WorkspaceBuffers, pagedKV: PagedKVCache): number[] {
    throw new Error("Qwen35Model.prefillBatch not yet implemented");
  }

  prefillBatchPlan(inputIdsList: number[][], ws: WorkspaceBuffers, pagedKV: PagedKVCache): Qwen35PrefillState {
    throw new Error("Qwen35Model.prefillBatchPlan not yet implemented");
  }

  prefillBatchForward(state: Qwen35PrefillState, ws: WorkspaceBuffers, pagedKV: PagedKVCache): void {
    throw new Error("Qwen35Model.prefillBatchForward not yet implemented");
  }

  prefillBatchRead(state: Qwen35PrefillState): number[] {
    throw new Error("Qwen35Model.prefillBatchRead not yet implemented");
  }

  decodeBatchPlan(tokenIdsList: number[], ws: WorkspaceBuffers, pagedKV: PagedKVCache, enableCudaGraph = false): Qwen35DecodeState {
    throw new Error("Qwen35Model.decodeBatchPlan not yet implemented");
  }

  decodeBatchForward(state: Qwen35DecodeState, ws: WorkspaceBuffers, pagedKV: PagedKVCache): void {
    throw new Error("Qwen35Model.decodeBatchForward not yet implemented");
  }

  decodeBatchRead(state: Qwen35DecodeState): number[] {
    throw new Error("Qwen35Model.decodeBatchRead not yet implemented");
  }

  decodeBatch(tokenIdsList: number[], ws: WorkspaceBuffers, pagedKV: PagedKVCache): number[] {
    throw new Error("Qwen35Model.decodeBatch not yet implemented");
  }

  generateBatch(inputIdsList: number[][], ws: WorkspaceBuffers, pagedKV: PagedKVCache, maxNewTokens = 100, eosTokenIds = EOS_TOKEN_IDS): number[][] {
    throw new Error("Qwen35Model.generateBatch not yet implemented");
  }
}
