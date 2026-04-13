import fs from "node:fs";
import path from "node:path";
import { GlmOps, f32ToBf16Bytes, bf16BytesToF32, BF16, I32, FLASH_TMP_SIZE } from "./glm_ops.js";
import { SafeTensorFile } from "./safetensors.js";
import { resolveModelPath } from "./model_path.js";

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

interface Workspace {
  [key: string]: number;
}

export class Qwen3Model {
  glm: GlmOps;
  cfg: Qwen3Config;
  weights: Map<string, number>;
  maxBatch: number;
  maxSeqLen: number;
  invFreq: number;
  ws: Workspace;
  kCache: number[];
  vCache: number[];
  cachePos: number;

  private constructor(glm: GlmOps, config: Qwen3Config, weights: Map<string, number>, maxBatch: number, maxSeqLen: number) {
    this.glm = glm;
    this.cfg = config;
    this.weights = weights;
    this.maxBatch = maxBatch;
    this.maxSeqLen = maxSeqLen;
    this.cachePos = 0;

    const halfDim = config.headDim / 2;
    const invFreqF32 = new Float32Array(halfDim);
    for (let i = 0; i < halfDim; i++) {
      invFreqF32[i] = 1.0 / Math.pow(config.ropeTheta, (2 * i) / config.headDim);
    }
    this.invFreq = glm.alloc(halfDim * BF16);
    glm.h2d(this.invFreq, f32ToBf16Bytes(invFreqF32));

    this.ws = this.allocWorkspace(maxBatch, maxSeqLen);
    const cache = this.allocCache(maxBatch, maxSeqLen);
    this.kCache = cache.k;
    this.vCache = cache.v;
  }

  static fromPretrained(glm: GlmOps, repoId: string, maxBatch = 1, maxSeqLen = 4096): Qwen3Model {
    const modelDir = resolveModelPath(repoId);
    const config = loadConfig(modelDir);

    const stPath = path.join(modelDir, "model.safetensors");
    const st = SafeTensorFile.open(stPath);
    const mmapPtr = glm.mmapOpen(stPath);
    const fileSize = fs.statSync(stPath).size;

    const weights = new Map<string, number>();
    for (const name of st.tensorNames()) {
      const meta = st.meta(name);
      const size = meta.dataOffsets[1] - meta.dataOffsets[0];
      const gpuPtr = glm.alloc(size);
      const offset = st.dataStart + meta.dataOffsets[0];
      glm.mmapLoad(gpuPtr, mmapPtr, offset, size);
      weights.set(name, gpuPtr);
    }

    glm.synchronize();
    st.close();
    glm.mmapClose(mmapPtr, fileSize);

    if (config.tieWordEmbeddings && !weights.has("lm_head.weight")) {
      const embedPtr = weights.get("model.embed_tokens.weight")!;
      weights.set("lm_head.weight", embedPtr);
    }

    return new Qwen3Model(glm, config, weights, maxBatch, maxSeqLen);
  }

  private allocWorkspace(B: number, S: number): Workspace {
    const cfg = this.cfg;
    const hs = cfg.hiddenSize;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const inter = cfg.intermediateSize;
    const vs = cfg.vocabSize;
    const BS = B * S;
    const glm = this.glm;

    return {
      hiddenA: glm.alloc(BS * hs * BF16),
      hiddenB: glm.alloc(BS * hs * BF16),
      normed: glm.alloc(BS * hs * BF16),
      qBuf: glm.alloc(BS * nHeads * hd * BF16),
      kBuf: glm.alloc(BS * nKv * hd * BF16),
      vBuf: glm.alloc(BS * nKv * hd * BF16),
      qNormed: glm.alloc(BS * nHeads * hd * BF16),
      kNormed: glm.alloc(BS * nKv * hd * BF16),
      qT: glm.alloc(B * nHeads * S * hd * BF16),
      kT: glm.alloc(B * nKv * S * hd * BF16),
      vT: glm.alloc(B * nKv * S * hd * BF16),
      qRope: glm.alloc(B * nHeads * S * hd * BF16),
      kRope: glm.alloc(B * nKv * S * hd * BF16),
      kExpanded: glm.alloc(B * nHeads * S * hd * BF16),
      vExpanded: glm.alloc(B * nHeads * S * hd * BF16),
      attnScores: glm.alloc(B * nHeads * S * S * BF16),
      attnOut: glm.alloc(B * nHeads * S * hd * BF16),
      attnOutT: glm.alloc(B * nHeads * S * hd * BF16),
      oProjBuf: glm.alloc(BS * hs * BF16),
      gateBuf: glm.alloc(BS * inter * BF16),
      upBuf: glm.alloc(BS * inter * BF16),
      siluBuf: glm.alloc(BS * inter * BF16),
      downBuf: glm.alloc(BS * hs * BF16),
      cos: glm.alloc(B * S * hd * BF16),
      sin: glm.alloc(B * S * hd * BF16),
      causalMask: glm.alloc(S * S * BF16),
      maskExpanded: glm.alloc(B * nHeads * S * S * BF16),
      positionIds: glm.alloc(B * S * I32),
      logitsBuf: glm.alloc(BS * vs * BF16),
      argmaxIdx: glm.alloc(I32),
      decodeId: glm.alloc(I32),
      flashOut: glm.alloc(B * nHeads * S * hd * BF16),
      flashTmp: glm.alloc(FLASH_TMP_SIZE),
    };
  }

  private allocCache(B: number, S: number): { k: number[]; v: number[] } {
    const cfg = this.cfg;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const nLayers = cfg.numHiddenLayers;
    const glm = this.glm;

    const k: number[] = [];
    const v: number[] = [];
    for (let i = 0; i < nLayers; i++) {
      k.push(glm.alloc(B * nKv * S * hd * BF16));
      v.push(glm.alloc(B * nKv * S * hd * BF16));
    }
    return { k, v };
  }

  free(): void {
    const glm = this.glm;
    for (const ptr of Object.values(this.ws)) {
      glm.freeBuf(ptr);
    }
    for (const ptr of this.kCache) {
      glm.freeBuf(ptr);
    }
    for (const ptr of this.vCache) {
      glm.freeBuf(ptr);
    }
    glm.freeBuf(this.invFreq);
    const seen = new Set<number>();
    for (const ptr of this.weights.values()) {
      if (!seen.has(ptr)) {
        seen.add(ptr);
        glm.freeBuf(ptr);
      }
    }
    this.ws = {};
    this.kCache = [];
    this.vCache = [];
    this.weights = new Map();
  }

  resetCache(): void {
    this.cachePos = 0;
  }

  private uploadIds(inputIds: number[][]): number {
    const B = inputIds.length;
    const S = inputIds[0].length;
    const flat = new Int32Array(B * S);
    for (let b = 0; b < B; b++) {
      for (let s = 0; s < S; s++) {
        flat[b * S + s] = inputIds[b][s];
      }
    }
    const nbytes = flat.byteLength;
    const ptr = this.glm.alloc(nbytes);
    this.glm.h2d(ptr, Buffer.from(flat.buffer));
    return ptr;
  }

  private readArgmax(ptr: number, count: number): number {
    this.glm.argmax(this.ws.argmaxIdx, ptr, count);
    const buf = Buffer.alloc(I32);
    this.glm.d2h(buf, this.ws.argmaxIdx);
    return buf.readInt32LE(0);
  }

  private readLogits(ptr: number, count: number): Float32Array {
    const nbytes = count * BF16;
    const buf = Buffer.alloc(nbytes);
    this.glm.d2h(buf, ptr);
    return bf16BytesToF32(buf);
  }

  prefill(inputIds: number[][]): number {
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
    if (this.cachePos !== 0) {
      throw new Error("Cache must be reset before prefill");
    }

    const idsPtr = this.uploadIds(inputIds);
    glm.embedding(this.ws.hiddenA, this.weights.get("model.embed_tokens.weight")!, idsPtr, hs, BS);
    glm.freeBuf(idsPtr);

    glm.arange(this.ws.positionIds, 0, 1, S);
    glm.rotaryEmbedding(this.ws.cos, this.ws.sin, this.invFreq, this.ws.positionIds, hd / 2, B, S);
    glm.causalMask(this.ws.causalMask, S);

    for (let i = 0; i < cfg.numHiddenLayers; i++) {
      this.decoderLayerPrefillFlash(B, S, i, `model.layers.${i}`);
    }

    glm.rmsnorm(this.ws.normed, this.ws.hiddenA, this.weights.get("model.norm.weight")!, cfg.rmsNormEps, hs, BS);
    glm.linear(this.ws.logitsBuf, this.ws.normed, this.weights.get("lm_head.weight")!, BS, vs, hs);

    this.cachePos = S;

    const logitsPtr = this.ws.logitsBuf + (S - 1) * vs * BF16;
    return this.readArgmax(logitsPtr, vs);
  }

  private decodeToken(tokenId: number): void {
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;
    const hd = cfg.headDim;
    const vs = cfg.vocabSize;
    const B = 1;
    const S = 1;
    const BS = 1;
    const cachedLen = this.cachePos;

    if (cachedLen === 0) throw new Error("Must prefill before decode");
    if (cachedLen + S > this.maxSeqLen) throw new Error("Cache overflow");

    const idsBuf = Buffer.alloc(I32);
    idsBuf.writeInt32LE(tokenId, 0);
    glm.h2d(this.ws.decodeId, idsBuf);

    glm.embedding(this.ws.hiddenA, this.weights.get("model.embed_tokens.weight")!, this.ws.decodeId, hs, BS);
    glm.arange(this.ws.positionIds, cachedLen, 0, 1);
    glm.rotaryEmbedding(this.ws.cos, this.ws.sin, this.invFreq, this.ws.positionIds, hd / 2, B, S);

    for (let i = 0; i < cfg.numHiddenLayers; i++) {
      this.decoderLayerDecodeFlash(B, S, i, `model.layers.${i}`, cachedLen);
    }

    glm.rmsnorm(this.ws.normed, this.ws.hiddenA, this.weights.get("model.norm.weight")!, cfg.rmsNormEps, hs, BS);
    glm.linear(this.ws.logitsBuf, this.ws.normed, this.weights.get("lm_head.weight")!, BS, vs, hs);

    this.cachePos = cachedLen + S;
  }

  private get hd(): number { return this.cfg.headDim; }

  generateTokens(inputIds: number[][], maxNewTokens = 100, eosTokenIds = new Set([151645, 151643])): number[] {
    const vs = this.cfg.vocabSize;
    const B = inputIds.length;
    if (B !== 1) throw new Error("generateTokens only supports batch=1");

    this.resetCache();
    let nextToken = this.prefill(inputIds);
    const generated = [nextToken];

    for (let i = 0; i < maxNewTokens - 1; i++) {
      if (eosTokenIds.has(nextToken)) break;
      this.decodeToken(nextToken);
      nextToken = this.readArgmax(this.ws.logitsBuf, vs);
      generated.push(nextToken);
    }

    return generated;
  }

  *streamTokens(inputIds: number[][], maxNewTokens = 100, eosTokenIds = new Set([151645, 151643])): Generator<number> {
    const vs = this.cfg.vocabSize;
    if (inputIds.length !== 1) throw new Error("streamTokens only supports batch=1");

    this.resetCache();
    let nextToken = this.prefill(inputIds);
    yield nextToken;

    for (let i = 0; i < maxNewTokens - 1; i++) {
      if (eosTokenIds.has(nextToken)) break;
      this.decodeToken(nextToken);
      nextToken = this.readArgmax(this.ws.logitsBuf, vs);
      yield nextToken;
    }
  }

  private decoderLayerPrefillFlash(B: number, S: number, layerIdx: number, pfx: string): void {
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;
    const inter = cfg.intermediateSize;
    const BS = B * S;

    glm.rmsnorm(this.ws.normed, this.ws.hiddenA, this.weights.get(`${pfx}.input_layernorm.weight`)!, cfg.rmsNormEps, hs, BS);
    this.attentionPrefillFlash(B, S, layerIdx, pfx);
    glm.add(this.ws.hiddenB, this.ws.hiddenA, this.ws.oProjBuf, BS * hs);
    glm.rmsnorm(this.ws.normed, this.ws.hiddenB, this.weights.get(`${pfx}.post_attention_layernorm.weight`)!, cfg.rmsNormEps, hs, BS);
    this.mlp(B, S, pfx);
  }

  private decoderLayerDecodeFlash(B: number, S: number, layerIdx: number, pfx: string, cachedLen: number): void {
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;
    const inter = cfg.intermediateSize;
    const BS = B * S;

    glm.rmsnorm(this.ws.normed, this.ws.hiddenA, this.weights.get(`${pfx}.input_layernorm.weight`)!, cfg.rmsNormEps, hs, BS);
    this.attentionDecodeFlash(B, S, layerIdx, pfx, cachedLen);
    glm.add(this.ws.hiddenB, this.ws.hiddenA, this.ws.oProjBuf, BS * hs);
    glm.rmsnorm(this.ws.normed, this.ws.hiddenB, this.weights.get(`${pfx}.post_attention_layernorm.weight`)!, cfg.rmsNormEps, hs, BS);
    this.mlp(B, S, pfx);
  }

  private mlp(B: number, S: number, pfx: string): void {
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;
    const inter = cfg.intermediateSize;
    const BS = B * S;

    glm.linear(this.ws.gateBuf, this.ws.normed, this.weights.get(`${pfx}.mlp.gate_proj.weight`)!, BS, inter, hs);
    glm.linear(this.ws.upBuf, this.ws.normed, this.weights.get(`${pfx}.mlp.up_proj.weight`)!, BS, inter, hs);
    glm.siluAndMul(this.ws.siluBuf, this.ws.gateBuf, this.ws.upBuf, inter, BS);
    glm.linear(this.ws.downBuf, this.ws.siluBuf, this.weights.get(`${pfx}.mlp.down_proj.weight`)!, BS, hs, inter);
    glm.add(this.ws.hiddenA, this.ws.hiddenB, this.ws.downBuf, BS * hs);
  }

  private attentionPrefillFlash(B: number, S: number, layerIdx: number, pfx: string): void {
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const maxS = this.maxSeqLen;
    const BS = B * S;

    glm.linear(this.ws.qBuf, this.ws.normed, this.weights.get(`${pfx}.self_attn.q_proj.weight`)!, BS, nHeads * hd, hs);
    glm.linear(this.ws.kBuf, this.ws.normed, this.weights.get(`${pfx}.self_attn.k_proj.weight`)!, BS, nKv * hd, hs);
    glm.linear(this.ws.vBuf, this.ws.normed, this.weights.get(`${pfx}.self_attn.v_proj.weight`)!, BS, nKv * hd, hs);

    glm.rmsnorm(this.ws.qNormed, this.ws.qBuf, this.weights.get(`${pfx}.self_attn.q_norm.weight`)!, cfg.rmsNormEps, hd, BS * nHeads);
    glm.rmsnorm(this.ws.kNormed, this.ws.kBuf, this.weights.get(`${pfx}.self_attn.k_norm.weight`)!, cfg.rmsNormEps, hd, BS * nKv);

    glm.transpose4d(this.ws.qT, this.ws.qNormed, B, S, nHeads, hd, 0, 2, 1, 3);
    glm.transpose4d(this.ws.kT, this.ws.kNormed, B, S, nKv, hd, 0, 2, 1, 3);
    glm.transpose4d(this.ws.vT, this.ws.vBuf, B, S, nKv, hd, 0, 2, 1, 3);

    glm.applyRotaryPosEmb(this.ws.qRope, this.ws.qT, this.ws.cos, this.ws.sin, hd, nHeads, S, B, 1);
    glm.applyRotaryPosEmb(this.ws.kRope, this.ws.kT, this.ws.cos, this.ws.sin, hd, nKv, S, B, 1);

    for (let h = 0; h < nKv; h++) {
      const srcOff = h * S * hd * BF16;
      const dstOff = h * maxS * hd * BF16;
      glm.memcpy(this.kCache[layerIdx] + dstOff, this.ws.kRope + srcOff, S * hd * BF16);
      glm.memcpy(this.vCache[layerIdx] + dstOff, this.ws.vT + srcOff, S * hd * BF16);
    }

    const kvStrideH = maxS * hd;
    const kvStrideN = hd;

    glm.flashPrefill(
      this.ws.qRope,
      this.kCache[layerIdx],
      this.vCache[layerIdx],
      this.ws.flashOut,
      this.ws.flashTmp,
      S, S,
      nHeads, nKv, hd,
      hd, S * hd,
      kvStrideN, kvStrideH,
      kvStrideN, kvStrideH,
      1, 1,
      cfg.scaling,
    );

    glm.linear(this.ws.oProjBuf, this.ws.flashOut, this.weights.get(`${pfx}.self_attn.o_proj.weight`)!, BS, hs, nHeads * hd);
  }

  private attentionDecodeFlash(B: number, S: number, layerIdx: number, pfx: string, cachedLen: number): void {
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const maxS = this.maxSeqLen;
    const BS = B * S;
    const totalLen = cachedLen + S;

    glm.linear(this.ws.qBuf, this.ws.normed, this.weights.get(`${pfx}.self_attn.q_proj.weight`)!, BS, nHeads * hd, hs);
    glm.linear(this.ws.kBuf, this.ws.normed, this.weights.get(`${pfx}.self_attn.k_proj.weight`)!, BS, nKv * hd, hs);
    glm.linear(this.ws.vBuf, this.ws.normed, this.weights.get(`${pfx}.self_attn.v_proj.weight`)!, BS, nKv * hd, hs);

    glm.rmsnorm(this.ws.qNormed, this.ws.qBuf, this.weights.get(`${pfx}.self_attn.q_norm.weight`)!, cfg.rmsNormEps, hd, BS * nHeads);
    glm.rmsnorm(this.ws.kNormed, this.ws.kBuf, this.weights.get(`${pfx}.self_attn.k_norm.weight`)!, cfg.rmsNormEps, hd, BS * nKv);

    glm.transpose4d(this.ws.qT, this.ws.qNormed, B, S, nHeads, hd, 0, 2, 1, 3);
    glm.transpose4d(this.ws.kT, this.ws.kNormed, B, S, nKv, hd, 0, 2, 1, 3);
    glm.transpose4d(this.ws.vT, this.ws.vBuf, B, S, nKv, hd, 0, 2, 1, 3);

    glm.applyRotaryPosEmb(this.ws.qRope, this.ws.qT, this.ws.cos, this.ws.sin, hd, nHeads, S, B, 1);
    glm.applyRotaryPosEmb(this.ws.kRope, this.ws.kT, this.ws.cos, this.ws.sin, hd, nKv, S, B, 1);

    for (let h = 0; h < nKv; h++) {
      const srcOff = h * S * hd * BF16;
      const dstOff = (h * maxS * hd + cachedLen * hd) * BF16;
      glm.memcpy(this.kCache[layerIdx] + dstOff, this.ws.kRope + srcOff, S * hd * BF16);
      glm.memcpy(this.vCache[layerIdx] + dstOff, this.ws.vT + srcOff, S * hd * BF16);
    }

    const kvStrideH = maxS * hd;
    const kvStrideN = hd;

    glm.flashDecode(
      this.ws.qRope,
      this.kCache[layerIdx],
      this.vCache[layerIdx],
      this.ws.flashOut,
      this.ws.flashTmp,
      totalLen,
      nHeads, nKv, hd,
      hd, S * hd,
      kvStrideN, kvStrideH,
      cfg.scaling,
    );

    glm.linear(this.ws.oProjBuf, this.ws.flashOut, this.weights.get(`${pfx}.self_attn.o_proj.weight`)!, BS, hs, nHeads * hd);
  }
}
