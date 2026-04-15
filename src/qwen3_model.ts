import fs from "node:fs";
import path from "node:path";
import { GlmOps, f32ToBf16Bytes, bf16BytesToF32, BF16, I32, FLASH_TMP_SIZE, BATCH_FLOAT_WS_SIZE, BATCH_INT_WS_SIZE, BATCH_PINNED_INT_WS_SIZE } from "./glm_ops";
import { SafeTensorFile } from "./safetensors";
import { resolveModelPath } from "./model_path";
import { PagedKVCache, WorkspaceBuffers } from "./paged_kv";
import { FlatKVCache } from "./flat_kv";
import { Tensor, OpContext } from "./tensor";

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
  decodeId: Tensor;
  flashOut: Tensor;
  flashTmp: Tensor;
  inputIdsBuf: Tensor;
  qoIndptrD: Tensor;
  prefillSlotMapping: Tensor;
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
    this.decodeId = Tensor.alloc(glm, [1], "I32");
    this.flashOut = Tensor.alloc(glm, [B, nHeads, S, hd], "BF16");
    this.flashTmp = Tensor.alloc(glm, [FLASH_TMP_SIZE], "U8");
    this.inputIdsBuf = Tensor.alloc(glm, [B * S], "I32");
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

export interface DecodeState {
  batchSize: number;
}

export interface PrefillState {
  batchSize: number;
  totalTokens: number;
  seqLens: number[];
  pageAllocs: [number, number][];
}

export class Qwen3Model implements OpContext {
  glm: GlmOps;
  cfg: Qwen3Config;
  weights: Map<string, Tensor>;
  maxBatch: number;
  maxSeqLen: number;
  invFreq: Tensor;
  ws: Qwen3Workspace;

  private constructor(glm: GlmOps, config: Qwen3Config, weights: Map<string, Tensor>, maxBatch: number, maxSeqLen: number) {
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

  createFlatKVCache(): FlatKVCache {
    return new FlatKVCache(this.glm, this.cfg.numKeyValueHeads, this.cfg.headDim, this.cfg.numHiddenLayers, this.maxBatch, this.maxSeqLen);
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

  prefill(inputIds: number[][], cache: FlatKVCache): number {
    const B = inputIds.length;
    const S = inputIds[0].length;
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const vs = cfg.vocabSize;
    const BS = B * S;

    if (B > this.maxBatch || S > this.maxSeqLen) {
      throw new Error(`input (B=${B}, S=${S}) exceeds max (B=${this.maxBatch}, S=${this.maxSeqLen})`);
    }
    if (cache.cachePos !== 0) {
      throw new Error("Cache must be reset before prefill");
    }

    const flat = new Int32Array(B * S);
    for (let b = 0; b < B; b++) {
      for (let s = 0; s < S; s++) {
        flat[b * S + s] = inputIds[b][s];
      }
    }
    this.ws.inputIdsBuf.h2d(Buffer.from(flat.buffer, flat.byteOffset, flat.byteLength));
    this.ws.hiddenA.embedding(this.weights.get("model.embed_tokens.weight")!, this.ws.inputIdsBuf, hs, BS);

    this.ws.positionIds.arange(0, 1, S);
    glm.rotaryEmbedding(this.ws.cos.data, this.ws.sin.data, this.invFreq.data, this.ws.positionIds.data, hd / 2, B, S);
    for (let i = 0; i < cfg.numHiddenLayers; i++) {
      this.decoderLayerPrefillFlash(B, S, i, `model.layers.${i}`, cache);
    }

    const lastIdxBuf = Int32Array.from([S - 1]);
    this.extractLastLogits(1, Buffer.from(lastIdxBuf.buffer, lastIdxBuf.byteOffset, lastIdxBuf.byteLength));

    cache.cachePos = S;

    return this.readArgmax(this.ws.logitsBuf, vs);
  }

  private decodeToken(tokenId: number, cache: FlatKVCache): void {
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;
    const hd = cfg.headDim;
    const vs = cfg.vocabSize;
    const B = 1;
    const S = 1;
    const BS = 1;
    const cachedLen = cache.cachePos;

    if (cachedLen === 0) throw new Error("Must prefill before decode");
    if (cachedLen + S > this.maxSeqLen) throw new Error("Cache overflow");

    const idsBuf = Buffer.alloc(I32);
    idsBuf.writeInt32LE(tokenId, 0);
    this.ws.decodeId.h2d(idsBuf);

    this.ws.hiddenA.embedding(this.weights.get("model.embed_tokens.weight")!, this.ws.decodeId, hs, BS);
    this.ws.positionIds.arange(cachedLen, 0, 1);
    glm.rotaryEmbedding(this.ws.cos.data, this.ws.sin.data, this.invFreq.data, this.ws.positionIds.data, hd / 2, B, S);

    for (let i = 0; i < cfg.numHiddenLayers; i++) {
      this.decoderLayerDecodeFlash(B, S, i, `model.layers.${i}`, cachedLen, cache);
    }

    this.finalNormAndLogits(BS);

    cache.cachePos = cachedLen + S;
  }

  decode(tokenId: number, cache: FlatKVCache): number {
    this.decodeToken(tokenId, cache);
    return this.readArgmax(this.ws.logitsBuf, this.cfg.vocabSize);
  }

  generateTokens(inputIds: number[][], cache: FlatKVCache, maxNewTokens = 100, eosTokenIds = new Set([151645, 151643])): number[] {
    return [...this.streamTokens(inputIds, cache, maxNewTokens, eosTokenIds)];
  }

  *streamTokens(inputIds: number[][], cache: FlatKVCache, maxNewTokens = 100, eosTokenIds = new Set([151645, 151643])): Generator<number> {
    const vs = this.cfg.vocabSize;
    if (inputIds.length !== 1) throw new Error("streamTokens only supports batch=1");

    cache.reset();
    let nextToken = this.prefill(inputIds, cache);
    yield nextToken;

    for (let i = 0; i < maxNewTokens - 1; i++) {
      if (eosTokenIds.has(nextToken)) break;
      this.decodeToken(nextToken, cache);
      nextToken = this.readArgmax(this.ws.logitsBuf, vs);
      yield nextToken;
    }
  }

  private decoderLayerPrefillFlash(B: number, S: number, layerIdx: number, pfx: string, cache: FlatKVCache): void {
    const cfg = this.cfg;
    const BS = B * S;

    this.ws.normed.rmsnorm(this.ws.hiddenA, this.weights.get(`${pfx}.input_layernorm.weight`)!, cfg.rmsNormEps, cfg.hiddenSize, BS);
    this.attentionPrefillFlash(B, S, layerIdx, pfx, cache);
    this.residualAndMlp(pfx, BS);
  }

  private decoderLayerDecodeFlash(B: number, S: number, layerIdx: number, pfx: string, cachedLen: number, cache: FlatKVCache): void {
    const cfg = this.cfg;
    const BS = B * S;

    this.ws.normed.rmsnorm(this.ws.hiddenA, this.weights.get(`${pfx}.input_layernorm.weight`)!, cfg.rmsNormEps, cfg.hiddenSize, BS);
    this.attentionDecodeFlash(B, S, layerIdx, pfx, cachedLen, cache);
    this.residualAndMlp(pfx, BS);
  }

  private mlp(BS: number, pfx: string): void {
    const cfg = this.cfg;
    const hs = cfg.hiddenSize;
    const inter = cfg.intermediateSize;

    this.ws.gateBuf.linear(this.ws.normed, this.weights.get(`${pfx}.mlp.gate_proj.weight`)!, BS, inter, hs, this);
    this.ws.upBuf.linear(this.ws.normed, this.weights.get(`${pfx}.mlp.up_proj.weight`)!, BS, inter, hs, this);
    this.ws.siluBuf.siluAndMul(this.ws.gateBuf, this.ws.upBuf, inter, BS);
    this.ws.downBuf.linear(this.ws.siluBuf, this.weights.get(`${pfx}.mlp.down_proj.weight`)!, BS, hs, inter, this);
    this.ws.hiddenA.add(this.ws.hiddenB, this.ws.downBuf, BS * hs);
  }

  private residualAndMlp(pfx: string, BS: number): void {
    const cfg = this.cfg;
    const hs = cfg.hiddenSize;

    this.ws.hiddenB.add(this.ws.hiddenA, this.ws.oProjBuf, BS * hs);
    this.ws.normed.rmsnorm(this.ws.hiddenB, this.weights.get(`${pfx}.post_attention_layernorm.weight`)!, cfg.rmsNormEps, hs, BS);
    this.mlp(BS, pfx);
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

    this.ws.qNormed.rmsnorm(this.ws.qBuf, this.weights.get(`${pfx}.self_attn.q_norm.weight`)!, cfg.rmsNormEps, hd, BS * nHeads);
    this.ws.kNormed.rmsnorm(this.ws.kBuf, this.weights.get(`${pfx}.self_attn.k_norm.weight`)!, cfg.rmsNormEps, hd, BS * nKv);

    this.ws.qT.transpose4d(this.ws.qNormed, B, S, nHeads, hd, 0, 2, 1, 3);
    this.ws.kT.transpose4d(this.ws.kNormed, B, S, nKv, hd, 0, 2, 1, 3);
    this.ws.vT.transpose4d(this.ws.vBuf, B, S, nKv, hd, 0, 2, 1, 3);

    this.ws.qRope.applyRotaryPosEmb(this.ws.qT, this.ws.cos, this.ws.sin, hd, nHeads, S, B, 1);
    this.ws.kRope.applyRotaryPosEmb(this.ws.kT, this.ws.cos, this.ws.sin, hd, nKv, S, B, 1);
  }

  private writeKvFlat(layerIdx: number, S: number, cache: FlatKVCache, offset = 0): void {
    const cfg = this.cfg;
    const glm = this.glm;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const maxS = this.maxSeqLen;

    for (let h = 0; h < nKv; h++) {
      const srcOff = h * S * hd * BF16;
      const dstOff = (h * maxS * hd + offset * hd) * BF16;
      glm.memcpy(cache.kData[layerIdx] + dstOff, this.ws.kRope.data + srcOff, S * hd * BF16);
      glm.memcpy(cache.vData[layerIdx] + dstOff, this.ws.vT.data + srcOff, S * hd * BF16);
    }
  }

  private finalNormAndLogits(count: number, src?: Tensor): void {
    const cfg = this.cfg;
    const hs = cfg.hiddenSize;
    const vs = cfg.vocabSize;

    this.ws.normed.rmsnorm(src ?? this.ws.hiddenA, this.weights.get("model.norm.weight")!, cfg.rmsNormEps, hs, count);
    this.ws.logitsBuf.linear(this.ws.normed, this.weights.get("lm_head.weight")!, count, vs, hs, this);
  }

  private extractLastLogits(count: number, lastIndicesBuf: Buffer): void {
    this.ws.lastIdx.h2d(lastIndicesBuf);
    this.ws.hiddenLast.indexSelect(this.ws.hiddenA, this.ws.lastIdx, this.cfg.hiddenSize, count);
    this.finalNormAndLogits(count, this.ws.hiddenLast);
  }

  private attentionPrefillFlash(B: number, S: number, layerIdx: number, pfx: string, cache: FlatKVCache): void {
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const maxS = this.maxSeqLen;
    const BS = B * S;

    this.computeQkv(pfx, BS, B, S);
    this.writeKvFlat(layerIdx, S, cache);

    const kvStrideH = maxS * hd;
    const kvStrideN = hd;

    glm.flashPrefill(
      this.ws.qRope.data,
      cache.kData[layerIdx],
      cache.vData[layerIdx],
      this.ws.flashOut.data,
      this.ws.flashTmp.data,
      S, S,
      nHeads, nKv, hd,
      hd, S * hd,
      kvStrideN, kvStrideH,
      kvStrideN, kvStrideH,
      1, 1,
      cfg.scaling,
    );

    this.ws.oProjBuf.linear(this.ws.flashOut, this.weights.get(`${pfx}.self_attn.o_proj.weight`)!, BS, hs, nHeads * hd, this);
  }

  private attentionDecodeFlash(B: number, S: number, layerIdx: number, pfx: string, cachedLen: number, cache: FlatKVCache): void {
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const maxS = this.maxSeqLen;
    const BS = B * S;
    const totalLen = cachedLen + S;

    this.computeQkv(pfx, BS, B, S);
    this.writeKvFlat(layerIdx, S, cache, cachedLen);

    const kvStrideH = maxS * hd;
    const kvStrideN = hd;

    glm.flashDecode(
      this.ws.qRope.data,
      cache.kData[layerIdx],
      cache.vData[layerIdx],
      this.ws.flashOut.data,
      this.ws.flashTmp.data,
      totalLen,
      nHeads, nKv, hd,
      hd, S * hd,
      kvStrideN, kvStrideH,
      cfg.scaling,
    );

    this.ws.oProjBuf.linear(this.ws.flashOut, this.weights.get(`${pfx}.self_attn.o_proj.weight`)!, BS, hs, nHeads * hd, this);
  }

  prefillBatchPlan(inputIdsList: number[][], ws: WorkspaceBuffers, pagedKV: PagedKVCache): PrefillState {
    const cfg = this.cfg;
    const glm = this.glm;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const pageSize = pagedKV.pageSize;
    const batchSize = inputIdsList.length;

    pagedKV.reset(batchSize);

    const seqLens = inputIdsList.map(ids => ids.length);
    const totalTokens = seqLens.reduce((a, b) => a + b, 0);

    const pageAllocs: [number, number][] = [];
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      pageAllocs.push(pagedKV.allocPrefillPages(seqIdx, seqLens[seqIdx]));
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
    for (const s of seqLens) {
      for (let p = 0; p < s; p++) posIds.push(p);
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
        const pageIdxInSeq = Math.floor(pos / pagedKV.pageSize);
        const offsetInPage = pos % pagedKV.pageSize;
        const absPage = pages[pageIdxInSeq];
        slotMapping.push(absPage * pagedKV.pageSize + offsetInPage);
      }
    }
    const slotMappingBuf = Int32Array.from(slotMapping);
    this.ws.prefillSlotMapping.h2d(Buffer.from(slotMappingBuf.buffer, slotMappingBuf.byteOffset, slotMappingBuf.byteLength));

    return { batchSize, totalTokens, seqLens, pageAllocs };
  }

  prefillBatchForward(state: PrefillState, ws: WorkspaceBuffers, pagedKV: PagedKVCache): void {
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

    for (let i = 0; i < cfg.numHiddenLayers; i++) {
      const pfx = `model.layers.${i}`;

      this.ws.normed.rmsnorm(this.ws.hiddenA, this.weights.get(`${pfx}.input_layernorm.weight`)!, cfg.rmsNormEps, hs, totalTokens);

      this.computeQkv(pfx, totalTokens, 1, totalTokens);

      glm.kvCacheWrite(
        this.ws.kRope.data, this.ws.vT.data,
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
      this.residualAndMlp(pfx, totalTokens);
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

  prefillBatch(inputIdsList: number[][], ws: WorkspaceBuffers, pagedKV: PagedKVCache): number[] {
    const state = this.prefillBatchPlan(inputIdsList, ws, pagedKV);
    this.prefillBatchForward(state, ws, pagedKV);
    return this.prefillBatchRead(state);
  }

  prefillBatchAppendPlan(inputIdsList: number[][], ws: WorkspaceBuffers, pagedKV: PagedKVCache): PrefillState {
    const cfg = this.cfg;
    const glm = this.glm;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const pageSize = pagedKV.pageSize;
    const batchSize = inputIdsList.length;

    if (pagedKV.seqPages.length !== batchSize) {
      throw new Error(`prefillBatchAppend: pagedKV has ${pagedKV.seqPages.length} sequences, expected ${batchSize}`);
    }

    const seqLens = inputIdsList.map(ids => ids.length);
    const totalTokens = seqLens.reduce((a, b) => a + b, 0);
    const startPos = pagedKV.seqKvLens.slice();

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

  prefillBatchAppend(inputIdsList: number[][], ws: WorkspaceBuffers, pagedKV: PagedKVCache): number[] {
    const state = this.prefillBatchAppendPlan(inputIdsList, ws, pagedKV);
    this.prefillBatchForward(state, ws, pagedKV);
    return this.prefillBatchRead(state);
  }

  decodeBatchPlan(tokenIdsList: number[], ws: WorkspaceBuffers, pagedKV: PagedKVCache, enableCudaGraph = false): DecodeState {
    const cfg = this.cfg;
    const glm = this.glm;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
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
      nHeads, nKv, pageSize,
      enableCudaGraph
    );

    return { batchSize };
  }

  decodeBatchForward(state: DecodeState, ws: WorkspaceBuffers, pagedKV: PagedKVCache): void {
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

    for (let i = 0; i < cfg.numHiddenLayers; i++) {
      const pfx = `model.layers.${i}`;

      this.ws.normed.rmsnorm(this.ws.hiddenA, this.weights.get(`${pfx}.input_layernorm.weight`)!, cfg.rmsNormEps, hs, batchSize);

      this.computeQkv(pfx, batchSize, batchSize, 1);

      glm.kvCacheWrite(
        this.ws.kRope.data, this.ws.vT.data,
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
      this.residualAndMlp(pfx, batchSize);
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

  decodeBatch(tokenIdsList: number[], ws: WorkspaceBuffers, pagedKV: PagedKVCache): number[] {
    const state = this.decodeBatchPlan(tokenIdsList, ws, pagedKV);
    this.decodeBatchForward(state, ws, pagedKV);
    return this.decodeBatchRead(state);
  }

  generateBatch(inputIdsList: number[][], ws: WorkspaceBuffers, pagedKV: PagedKVCache, maxNewTokens = 100, eosTokenIds = new Set([151645, 151643])): number[][] {
    const batchSize = inputIdsList.length;
    const vs = this.cfg.vocabSize;

    const firstTokens = this.prefillBatch(inputIdsList, ws, pagedKV);

    const nextTokens = [...firstTokens];
    const generated: number[][] = nextTokens.map(t => [t]);
    const finished = nextTokens.map(t => eosTokenIds.has(t));

    for (let step = 0; step < maxNewTokens - 1; step++) {
      if (finished.every(f => f)) break;

      const newTokens = this.decodeBatch(nextTokens, ws, pagedKV);

      for (let i = 0; i < batchSize; i++) {
        nextTokens[i] = newTokens[i];
        if (!finished[i]) {
          if (eosTokenIds.has(newTokens[i])) {
            finished[i] = true;
          } else {
            generated[i].push(newTokens[i]);
          }
        }
      }
    }

    return generated;
  }
}
