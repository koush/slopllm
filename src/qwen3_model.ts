import fs from "node:fs";
import path from "node:path";
import { GlmOps, f32ToBf16Bytes, BF16, I32, FLASH_TMP_SIZE, BATCH_FLOAT_WS_SIZE, BATCH_INT_WS_SIZE, BATCH_PINNED_INT_WS_SIZE } from "./glm_ops";
import { SafeTensorFile } from "./safetensors";
import { resolveModelPath } from "./model_path";
import { PagedKVCache, WorkspaceBuffers } from "./paged_kv";
import { FlatKVCache } from "./flat_kv";

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

  private constructor(glm: GlmOps, config: Qwen3Config, weights: Map<string, number>, maxBatch: number, maxSeqLen: number) {
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
    this.invFreq = glm.alloc(halfDim * BF16);
    glm.h2d(this.invFreq, f32ToBf16Bytes(invFreqF32));

    this.ws = this.allocWorkspace(maxBatch, maxSeqLen);
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
      oProjBuf: glm.alloc(BS * hs * BF16),
      gateBuf: glm.alloc(BS * inter * BF16),
      upBuf: glm.alloc(BS * inter * BF16),
      siluBuf: glm.alloc(BS * inter * BF16),
      downBuf: glm.alloc(BS * hs * BF16),
      cos: glm.alloc(B * S * hd * BF16),
      sin: glm.alloc(B * S * hd * BF16),
      positionIds: glm.alloc(B * S * I32),
      logitsBuf: glm.alloc(B * vs * BF16),
      hiddenLast: glm.alloc(B * hs * BF16),
      lastIdx: glm.alloc(B * I32),
      argmaxIdx: glm.alloc(I32),
      decodeId: glm.alloc(I32),
      flashOut: glm.alloc(B * nHeads * S * hd * BF16),
      flashTmp: glm.alloc(FLASH_TMP_SIZE),
      inputIdsBuf: glm.alloc(BS * I32),
    };
  }

  free(): void {
    const glm = this.glm;
    for (const ptr of Object.values(this.ws)) {
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
    this.weights = new Map();
  }

  createFlatKVCache(): FlatKVCache {
    return new FlatKVCache(this.glm, this.cfg.numKeyValueHeads, this.cfg.headDim, this.cfg.numHiddenLayers, this.maxBatch, this.maxSeqLen);
  }

  private readArgmax(ptr: number, count: number): number {
    this.glm.argmax(this.ws.argmaxIdx, ptr, count);
    const buf = Buffer.alloc(I32);
    this.glm.d2h(buf, this.ws.argmaxIdx);
    return buf.readInt32LE(0);
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
    glm.h2d(this.ws.inputIdsBuf, Buffer.from(flat.buffer, flat.byteOffset, flat.byteLength));
    glm.embedding(this.ws.hiddenA, this.weights.get("model.embed_tokens.weight")!, this.ws.inputIdsBuf, hs, BS);

    glm.arange(this.ws.positionIds, 0, 1, S);
    glm.rotaryEmbedding(this.ws.cos, this.ws.sin, this.invFreq, this.ws.positionIds, hd / 2, B, S);
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
    glm.h2d(this.ws.decodeId, idsBuf);

    glm.embedding(this.ws.hiddenA, this.weights.get("model.embed_tokens.weight")!, this.ws.decodeId, hs, BS);
    glm.arange(this.ws.positionIds, cachedLen, 0, 1);
    glm.rotaryEmbedding(this.ws.cos, this.ws.sin, this.invFreq, this.ws.positionIds, hd / 2, B, S);

    for (let i = 0; i < cfg.numHiddenLayers; i++) {
      this.decoderLayerDecodeFlash(B, S, i, `model.layers.${i}`, cachedLen, cache);
    }

    this.finalNormAndLogits(BS);

    cache.cachePos = cachedLen + S;
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
    const glm = this.glm;
    const BS = B * S;

    glm.rmsnorm(this.ws.normed, this.ws.hiddenA, this.weights.get(`${pfx}.input_layernorm.weight`)!, cfg.rmsNormEps, cfg.hiddenSize, BS);
    this.attentionPrefillFlash(B, S, layerIdx, pfx, cache);
    this.residualAndMlp(pfx, BS);
  }

  private decoderLayerDecodeFlash(B: number, S: number, layerIdx: number, pfx: string, cachedLen: number, cache: FlatKVCache): void {
    const cfg = this.cfg;
    const glm = this.glm;
    const BS = B * S;

    glm.rmsnorm(this.ws.normed, this.ws.hiddenA, this.weights.get(`${pfx}.input_layernorm.weight`)!, cfg.rmsNormEps, cfg.hiddenSize, BS);
    this.attentionDecodeFlash(B, S, layerIdx, pfx, cachedLen, cache);
    this.residualAndMlp(pfx, BS);
  }

  private mlp(BS: number, pfx: string): void {
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;
    const inter = cfg.intermediateSize;

    glm.linear(this.ws.gateBuf, this.ws.normed, this.weights.get(`${pfx}.mlp.gate_proj.weight`)!, BS, inter, hs);
    glm.linear(this.ws.upBuf, this.ws.normed, this.weights.get(`${pfx}.mlp.up_proj.weight`)!, BS, inter, hs);
    glm.siluAndMul(this.ws.siluBuf, this.ws.gateBuf, this.ws.upBuf, inter, BS);
    glm.linear(this.ws.downBuf, this.ws.siluBuf, this.weights.get(`${pfx}.mlp.down_proj.weight`)!, BS, hs, inter);
    glm.add(this.ws.hiddenA, this.ws.hiddenB, this.ws.downBuf, BS * hs);
  }

  private residualAndMlp(pfx: string, BS: number): void {
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;

    glm.add(this.ws.hiddenB, this.ws.hiddenA, this.ws.oProjBuf, BS * hs);
    glm.rmsnorm(this.ws.normed, this.ws.hiddenB, this.weights.get(`${pfx}.post_attention_layernorm.weight`)!, cfg.rmsNormEps, hs, BS);
    this.mlp(BS, pfx);
  }

  private computeQkv(pfx: string, BS: number, B: number, S: number): void {
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;

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
      glm.memcpy(cache.kData[layerIdx] + dstOff, this.ws.kRope + srcOff, S * hd * BF16);
      glm.memcpy(cache.vData[layerIdx] + dstOff, this.ws.vT + srcOff, S * hd * BF16);
    }
  }

  private finalNormAndLogits(count: number, src?: number): void {
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;
    const vs = cfg.vocabSize;

    if (src === undefined) src = this.ws.hiddenA;

    glm.rmsnorm(this.ws.normed, src, this.weights.get("model.norm.weight")!, cfg.rmsNormEps, hs, count);
    glm.linear(this.ws.logitsBuf, this.ws.normed, this.weights.get("lm_head.weight")!, count, vs, hs);
  }

  private extractLastLogits(count: number, lastIndicesBuf: Buffer): void {
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;

    glm.h2d(this.ws.lastIdx, lastIndicesBuf);
    glm.indexSelect(this.ws.hiddenLast, this.ws.hiddenA, this.ws.lastIdx, hs, count);
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
      this.ws.qRope,
      cache.kData[layerIdx],
      cache.vData[layerIdx],
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
      this.ws.qRope,
      cache.kData[layerIdx],
      cache.vData[layerIdx],
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

  prefillBatch(inputIdsList: number[][], ws: WorkspaceBuffers, pagedKV: PagedKVCache): number[] {
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const vs = cfg.vocabSize;
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
    glm.h2d(this.ws.inputIdsBuf, Buffer.from(idsBuf.buffer, idsBuf.byteOffset, idsBuf.byteLength));

    glm.embedding(this.ws.hiddenA, this.weights.get("model.embed_tokens.weight")!, this.ws.inputIdsBuf, hs, totalTokens);

    const indptr = [0];
    for (const s of seqLens) {
      indptr.push(indptr[indptr.length - 1] + s);
    }
    const indptrBuf = Int32Array.from(indptr);

    const posIds: number[] = [];
    for (const s of seqLens) {
      for (let p = 0; p < s; p++) posIds.push(p);
    }
    const posIdsBuf = Int32Array.from(posIds);
    glm.h2d(this.ws.positionIds, Buffer.from(posIdsBuf.buffer, posIdsBuf.byteOffset, posIdsBuf.byteLength));

    glm.rotaryEmbedding(this.ws.cos, this.ws.sin, this.invFreq, this.ws.positionIds, hd / 2, 1, totalTokens);

    // Write indptr to host memory for plan
    const qoIndptrHostPtr = glm.allocPinned((batchSize + 1) * I32);
    const kvIndptrHostPtr = glm.allocPinned((batchSize + 1) * I32);
    glm.writePinned(qoIndptrHostPtr, Buffer.from(indptrBuf.buffer, indptrBuf.byteOffset, indptrBuf.byteLength));
    glm.writePinned(kvIndptrHostPtr, Buffer.from(indptrBuf.buffer, indptrBuf.byteOffset, indptrBuf.byteLength));

    glm.batchPrefillRaggedPlan(
      ws.floatWs, BATCH_FLOAT_WS_SIZE,
      ws.intWs, ws.pinnedIntWs, BATCH_INT_WS_SIZE,
      ws.prefillPlanInfo,
      qoIndptrHostPtr, kvIndptrHostPtr,
      totalTokens, batchSize,
      nHeads, nKv, hd,
      1
    );

    glm.freePinned(qoIndptrHostPtr);
    glm.freePinned(kvIndptrHostPtr);

    for (let i = 0; i < cfg.numHiddenLayers; i++) {
      const pfx = `model.layers.${i}`;

      glm.rmsnorm(this.ws.normed, this.ws.hiddenA, this.weights.get(`${pfx}.input_layernorm.weight`)!, cfg.rmsNormEps, hs, totalTokens);

      this.computeQkv(pfx, totalTokens, 1, totalTokens);

      // Copy KV to paged cache
      const pageSize = pagedKV.pageSize;
      for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
        const [startPage, numPages] = pageAllocs[seqIdx];
        const seqStart = seqLens.slice(0, seqIdx).reduce((a, b) => a + b, 0);
        const s = seqLens[seqIdx];
        for (let h = 0; h < nKv; h++) {
          for (let p = 0; p < numPages; p++) {
            const pageOffset = (startPage + p) * nKv * pageSize * hd;
            const kvHeadOffset = pageOffset + h * pageSize * hd;
            const tokenStart = p * pageSize;
            const tokenCount = Math.min(pageSize, s - p * pageSize);
            const srcOff = (h * totalTokens + seqStart + tokenStart) * hd * BF16;
            const dstOff = kvHeadOffset * BF16;
            const copyBytes = tokenCount * hd * BF16;
            glm.memcpy(pagedKV.kData[i] + dstOff, this.ws.kRope + srcOff, copyBytes);
            glm.memcpy(pagedKV.vData[i] + dstOff, this.ws.vT + srcOff, copyBytes);
          }
        }
      }

      const qStrideN = hd;
      const qStrideH = totalTokens * hd;
      const kvStrideN = hd;
      const kvStrideH = totalTokens * hd;

      const qoIndptrD = glm.alloc((batchSize + 1) * I32);
      const kvIndptrD = glm.alloc((batchSize + 1) * I32);
      glm.h2d(qoIndptrD, Buffer.from(indptrBuf.buffer, indptrBuf.byteOffset, indptrBuf.byteLength));
      glm.h2d(kvIndptrD, Buffer.from(indptrBuf.buffer, indptrBuf.byteOffset, indptrBuf.byteLength));

      glm.batchPrefillRaggedRun(
        this.ws.qRope, this.ws.kRope, this.ws.vT, this.ws.flashOut,
        ws.floatWs, ws.intWs,
        qoIndptrD, kvIndptrD,
        ws.prefillPlanInfo,
        totalTokens, batchSize,
        nHeads, nKv, hd,
        qStrideN, qStrideH,
        kvStrideN, kvStrideH,
        1, cfg.scaling
      );

      glm.freeBuf(qoIndptrD);
      glm.freeBuf(kvIndptrD);

      glm.linear(this.ws.oProjBuf, this.ws.flashOut, this.weights.get(`${pfx}.self_attn.o_proj.weight`)!, totalTokens, hs, nHeads * hd);
      this.residualAndMlp(pfx, totalTokens);
    }

    const lastIndices: number[] = [];
    let offset = 0;
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      lastIndices.push(offset + seqLens[seqIdx] - 1);
      offset += seqLens[seqIdx];
    }
    const lastIdxBuf = Int32Array.from(lastIndices);
    this.extractLastLogits(batchSize, Buffer.from(lastIdxBuf.buffer, lastIdxBuf.byteOffset, lastIdxBuf.byteLength));

    pagedKV.updateIndptr();

    const nextTokens: number[] = [];
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      const logitsPtr = this.ws.logitsBuf + seqIdx * vs * BF16;
      nextTokens.push(this.readArgmax(logitsPtr, vs));
    }

    return nextTokens;
  }

  decodeBatch(tokenIdsList: number[], ws: WorkspaceBuffers, pagedKV: PagedKVCache): number[] {
    const cfg = this.cfg;
    const glm = this.glm;
    const hs = cfg.hiddenSize;
    const nHeads = cfg.numAttentionHeads;
    const nKv = cfg.numKeyValueHeads;
    const hd = cfg.headDim;
    const vs = cfg.vocabSize;
    const pageSize = pagedKV.pageSize;
    const batchSize = tokenIdsList.length;

    const writeLocations: [number, number][] = [];
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      writeLocations.push(pagedKV.allocDecodeToken(seqIdx));
    }

    pagedKV.updateIndptr();

    const idsBuf = Int32Array.from(tokenIdsList);
    glm.h2d(this.ws.inputIdsBuf, Buffer.from(idsBuf.buffer, idsBuf.byteOffset, idsBuf.byteLength));

    glm.embedding(this.ws.hiddenA, this.weights.get("model.embed_tokens.weight")!, this.ws.inputIdsBuf, hs, batchSize);

    const posIds = new Array(batchSize);
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      posIds[seqIdx] = pagedKV.seqKvLens[seqIdx] - 1;
    }
    const posIdsBuf = Int32Array.from(posIds);
    glm.h2d(this.ws.positionIds, Buffer.from(posIdsBuf.buffer, posIdsBuf.byteOffset, posIdsBuf.byteLength));

    glm.rotaryEmbedding(this.ws.cos, this.ws.sin, this.invFreq, this.ws.positionIds, hd / 2, batchSize, 1);

    glm.batchDecodePlan(
      ws.floatWs, BATCH_FLOAT_WS_SIZE,
      ws.intWs, ws.pinnedIntWs, BATCH_INT_WS_SIZE,
      ws.decodePlanInfo,
      pagedKV.indptrH,
      batchSize,
      nHeads, nKv, pageSize
    );

    for (let i = 0; i < cfg.numHiddenLayers; i++) {
      const pfx = `model.layers.${i}`;

      glm.rmsnorm(this.ws.normed, this.ws.hiddenA, this.weights.get(`${pfx}.input_layernorm.weight`)!, cfg.rmsNormEps, hs, batchSize);

      this.computeQkv(pfx, batchSize, batchSize, 1);

      // Write KV to paged cache
      for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
        const [absPage, slotInPage] = writeLocations[seqIdx];
        for (let h = 0; h < nKv; h++) {
          const pageOffset = absPage * nKv * pageSize * hd;
          const kvHeadOffset = pageOffset + h * pageSize * hd;
          const tokenOffset = kvHeadOffset + slotInPage * hd;
          const srcOff = (seqIdx * nKv + h) * hd * BF16;
          glm.memcpy(pagedKV.kData[i] + tokenOffset * BF16, this.ws.kRope + srcOff, hd * BF16);
          glm.memcpy(pagedKV.vData[i] + tokenOffset * BF16, this.ws.vT + srcOff, hd * BF16);
        }
      }

      glm.batchDecodeRun(
        this.ws.qRope, this.ws.flashOut,
        pagedKV.kData[i], pagedKV.vData[i],
        pagedKV.indices, pagedKV.indptrD, pagedKV.lastPageLen,
        ws.floatWs, ws.intWs,
        ws.decodePlanInfo,
        batchSize,
        nHeads, nKv, hd, pageSize, cfg.scaling
      );

      glm.linear(this.ws.oProjBuf, this.ws.flashOut, this.weights.get(`${pfx}.self_attn.o_proj.weight`)!, batchSize, hs, nHeads * hd);
      this.residualAndMlp(pfx, batchSize);
    }

    this.finalNormAndLogits(batchSize);

    const nextTokens: number[] = [];
    for (let seqIdx = 0; seqIdx < batchSize; seqIdx++) {
      const logitsPtr = this.ws.logitsBuf + seqIdx * vs * BF16;
      nextTokens.push(this.readArgmax(logitsPtr, vs));
    }

    return nextTokens;
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