import path from "node:path";
import { after, before, describe, it } from "node:test";
import { ExecutionState, ExecutionWorkspace } from "../src/execution-workspace";
import { Glm51Model } from "../src/glm51_model";
import { GlmOps, bf16BytesToF32 } from "../src/glm_ops";
import { PagedKVCache } from "../src/paged_kv";
import { ParallelOps, ParallelTensor } from "../src/parallel_ops";
import { Tensor } from "../src/tensor";
import { UsingHolder } from "../src/using-holder";

const SMALL_MODEL_DIR = path.resolve(
  __dirname,
  "../tests/python/test_models/glm51_small/glm51_small_bf16",
);

const MAX_BATCH = 1;
const MAX_SEQ_LEN = 128;
const MAX_PAGES = 32;
const INPUT_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];

function readTensorF32(tensor: Tensor, totalElems: number): Float32Array {
  const buf = Buffer.alloc(totalElems * 2);
  tensor.d2h(buf);
  return bf16BytesToF32(buf);
}

function captureTensor(tensor: Tensor, totalElems: number): Float32Array | null {
  if (tensor instanceof ParallelTensor) {
    if (tensor.parallelism === "partial_softmax") {
      return null;
    }
    if (tensor.parallelism === "partial_sum") {
      return null;
    }
    const buf = Buffer.alloc(totalElems * 2);
    tensor.d2h(buf);
    return bf16BytesToF32(buf);
  }
  return readTensorF32(tensor, totalElems);
}

function extractRefKvAtPosition(
  refData: Float32Array, pos: number, pageSize: number, dim: number,
): Float32Array {
  const page = Math.floor(pos / pageSize);
  const offset = pos % pageSize;
  const start = (page * pageSize + offset) * dim;
  return refData.slice(start, start + dim);
}

function extractCpKvAtPosition(
  shards: Float32Array[], pos: number, pageSize: number, worldSize: number, dim: number,
): Float32Array {
  const rank = pos % worldSize;
  const localPos = Math.floor(pos / worldSize);
  const vPS = pageSize / worldSize;
  const localPage = Math.floor(localPos / vPS);
  const localOffset = localPos % vPS;
  const shardData = shards[rank];
  const start = (localPage * vPS + localOffset) * dim;
  return shardData.slice(start, start + dim);
}

function compareArrays(ref: Float32Array, cp: Float32Array, label: string, absTol = 0.005, relTol = 0.01): void {
  let maxAbsErr = 0, maxRelErr = 0, maxRelErrSig = 0, errorCount = 0;
  const n = ref.length;
  for (let i = 0; i < n; i++) {
    const absErr = Math.abs(ref[i] - cp[i]);
    const relErr = absErr / Math.max(Math.abs(ref[i]), 1e-6);
    maxAbsErr = Math.max(maxAbsErr, absErr);
    maxRelErr = Math.max(maxRelErr, relErr);
    if (Math.abs(ref[i]) > 0.01) maxRelErrSig = Math.max(maxRelErrSig, relErr);
    if (absErr > absTol + relTol * Math.abs(ref[i])) errorCount++;
  }
  console.log(`  ${label}: maxAbs=${maxAbsErr.toFixed(4)} maxRel=${maxRelErr.toFixed(4)} maxRelSig=${maxRelErrSig.toFixed(4)} errors=${errorCount}/${n}`);
}

describe("CP vs non-CP model prefill", () => {
  let glm0: GlmOps;
  let glm1: GlmOps;
  let glm2: GlmOps;
  let po: ParallelOps;
  let modelRef: Glm51Model;
  let modelCp: Glm51Model;
  let wsRef: ExecutionWorkspace;
  let wsCp: ExecutionWorkspace;

  before(async () => {
    glm0 = new GlmOps(0);
    glm1 = new GlmOps(1);
    glm2 = new GlmOps(2);
    po = new ParallelOps([glm1, glm2]);
    modelRef = await Glm51Model.fromPretrained(glm0, SMALL_MODEL_DIR, MAX_BATCH, MAX_SEQ_LEN, false);
    modelCp = await Glm51Model.fromPretrained(po, SMALL_MODEL_DIR, MAX_BATCH, MAX_SEQ_LEN, true);
    wsRef = new ExecutionWorkspace(glm0, MAX_BATCH, MAX_SEQ_LEN);
    wsCp = new ExecutionWorkspace(po, MAX_BATCH, MAX_SEQ_LEN);
  });

  after(() => {
    wsRef.free(); wsCp.free(); modelRef.free(); modelCp.free();
    po.free(); glm0.free(); glm1.free(); glm2.free();
  });

  it("KV cache after prefill", () => {
    const cfg = modelRef.cfg;
    const kvLoraRank = cfg.kvLoraRank!;
    const qkRopeDim = cfg.qkRopeHeadDim!;
    const pageSize = 16;
    const worldSize = 2;
    const seqLen = INPUT_IDS.length;

    const cacheRef = modelRef.createChatCache(MAX_PAGES);
    const cacheCp = modelCp.createChatCache(MAX_PAGES);
    cacheRef.reset(1); cacheCp.reset(1);
    const suffixRef = cacheRef.prefixMatch(0, INPUT_IDS);
    const suffixCp = cacheCp.prefixMatch(0, INPUT_IDS);

    wsRef.forwardEagerPrefill(modelRef, [suffixRef], cacheRef);
    wsCp.forwardEagerPrefill(modelCp, [suffixCp], cacheCp);
    cacheRef.appendTokens(0, suffixRef);
    cacheCp.appendTokens(0, suffixCp);

    const pagedKVRef = cacheRef.getPagedKV() as PagedKVCache;
    const pagedKVCp = cacheCp.getPagedKV() as PagedKVCache;
    const maxPages = pagedKVRef.maxPages;
    const refCkvElems = maxPages * pageSize * kvLoraRank;
    const refKpeElems = maxPages * pageSize * qkRopeDim;
    const cpVPS = pageSize / worldSize;

    for (let layer = 0; layer < cfg.numHiddenLayers; layer++) {
      const refCkv = readTensorF32(pagedKVRef.ckvData[layer], refCkvElems);
      const refKpe = readTensorF32(pagedKVRef.kpeData[layer], refKpeElems);
      const cpCkvShards: Float32Array[] = [], cpKpeShards: Float32Array[] = [];
      const cpCkvPt = pagedKVCp.ckvData[layer] as ParallelTensor;
      const cpKpePt = pagedKVCp.kpeData[layer] as ParallelTensor;
      for (let r = 0; r < worldSize; r++) {
        cpCkvShards.push(readTensorF32(cpCkvPt.shards[r], maxPages * cpVPS * kvLoraRank));
        cpKpeShards.push(readTensorF32(cpKpePt.shards[r], maxPages * cpVPS * qkRopeDim));
      }
      let ckvMax = 0, kpeMax = 0;
      for (let pos = 0; pos < seqLen; pos++) {
        const rc = extractRefKvAtPosition(refCkv, pos, pageSize, kvLoraRank);
        const cc = extractCpKvAtPosition(cpCkvShards, pos, pageSize, worldSize, kvLoraRank);
        const rk = extractRefKvAtPosition(refKpe, pos, pageSize, qkRopeDim);
        const ck = extractCpKvAtPosition(cpKpeShards, pos, pageSize, worldSize, qkRopeDim);
        for (let d = 0; d < kvLoraRank; d++) ckvMax = Math.max(ckvMax, Math.abs(rc[d] - cc[d]));
        for (let d = 0; d < qkRopeDim; d++) kpeMax = Math.max(kpeMax, Math.abs(rk[d] - ck[d]));
      }
      console.log(`Layer ${layer} KV: ckv_maxAbs=${ckvMax.toFixed(6)} kpe_maxAbs=${kpeMax.toFixed(6)}`);
    }
    cacheRef.free(); cacheCp.free();
  });

  it("layer 0 attention pipeline breakdown", () => {
    const cfg = modelRef.cfg;
    const hs = cfg.hiddenSize;
    const nHeads = cfg.numAttentionHeads;
    const kvLoraRank = cfg.kvLoraRank!;
    const qkRopeDim = cfg.qkRopeHeadDim!;
    const vHeadDim = cfg.vHeadDim;
    const seqLen = INPUT_IDS.length;
    const BS = seqLen;
    const PFX = "model.layers.0.self_attn";

    const snapshots: { label: string; ref: Float32Array | null; cp: Float32Array | null }[] = [];
    const snap2 = (label: string, tensor: Tensor, elems: number, isCp: boolean) => {
      const data = captureTensor(tensor, elems);
      if (data === null) return;
      const existing = snapshots.find(s => s.label === label);
      if (existing) { if (isCp) existing.cp = data; else existing.ref = data; }
      else snapshots.push({ label, ref: isCp ? null : data, cp: isCp ? data : null });
    };

    const patchModel = (model: Glm51Model, isCp: boolean) => {
      const origForward = model.forward.bind(model);
      const origMlaLayer = (model as any).mlaLayer.bind(model);
      let layer0Done = false;

      (model as any).mlaLayer = function(normed: Tensor, residual: Tensor, layerIdx: number, state: ExecutionState) {
        const result = origMlaLayer(normed, residual, layerIdx, state);
        if (layerIdx === 0 && !layer0Done) {
          layer0Done = true;
          snap2("L0_residual_after_layer", result.residual, BS * hs, isCp);
          snap2("L0_normed_after_layer", result.normed, BS * hs, isCp);
        }
        return result;
      };

      (model as any).forward = function(state: ExecutionState): Tensor {
        const embedTable = (model as any).tensors.get("model.embed_tokens.weight");
        using residual = new UsingHolder(embedTable.embedding(state.ws.inputIdsBuf, hs, BS));
        using normed = new UsingHolder(residual.value.rmsnorm(
          (model as any).tensors.get(`model.layers.0.input_layernorm.weight`), cfg.rmsNormEps, hs, BS));

        snap2("L0_input_normed", normed.value, BS * hs, isCp);
        snap2("L0_input_residual", residual.value, BS * hs, isCp);

        for (let i = 0; i < cfg.numHiddenLayers; i++) {
          const r = (model as any).mlaLayer(normed.value, residual.value, i, state);
          normed.replace(r.normed);
          residual.replace(r.residual);
        }
        return state.computeLogits(normed.value, model as any);
      };

      return { origForward, origMlaLayer };
    };

    const refRestore = patchModel(modelRef, false);
    const cpRestore = patchModel(modelCp, true);

    const cacheRef = modelRef.createChatCache(MAX_PAGES);
    const cacheCp = modelCp.createChatCache(MAX_PAGES);
    cacheRef.reset(1); cacheCp.reset(1);
    const suffixRefPipe = cacheRef.prefixMatch(0, INPUT_IDS);
    const suffixCpPipe = cacheCp.prefixMatch(0, INPUT_IDS);

    wsRef.forwardEagerPrefill(modelRef, [suffixRefPipe], cacheRef);
    wsCp.forwardEagerPrefill(modelCp, [suffixCpPipe], cacheCp);
    cacheRef.appendTokens(0, suffixRefPipe);
    cacheCp.appendTokens(0, suffixCpPipe);

    (modelRef as any).forward = refRestore.origForward;
    (modelRef as any).mlaLayer = refRestore.origMlaLayer;
    (modelCp as any).forward = cpRestore.origForward;
    (modelCp as any).mlaLayer = cpRestore.origMlaLayer;

    console.log("\nLayer 0 pipeline breakdown:");
    for (const s of snapshots) {
      if (s.ref && s.cp) compareArrays(s.ref, s.cp, s.label);
      else console.log(`  ${s.label}: MISSING ${s.ref ? "cp" : "ref"}`);
    }

    cacheRef.free(); cacheCp.free();
  });

  it("layer 0 MLA attention intermediate values", () => {
    const cfg = modelRef.cfg;
    const hs = cfg.hiddenSize;
    const nHeads = cfg.numAttentionHeads;
    const kvLoraRank = cfg.kvLoraRank!;
    const qkRopeDim = cfg.qkRopeHeadDim!;
    const vHeadDim = cfg.vHeadDim;
    const seqLen = INPUT_IDS.length;
    const BS = seqLen;
    const B = 1;
    const S = BS;

    const snapshots: { label: string; ref: Float32Array | null; cp: Float32Array | null }[] = [];
    const snap3 = (label: string, tensor: Tensor, elems: number, isCp: boolean) => {
      const data = captureTensor(tensor, elems);
      if (data === null) return;
      const existing = snapshots.find(s => s.label === label);
      if (existing) { if (isCp) existing.cp = data; else existing.ref = data; }
      else snapshots.push({ label, ref: isCp ? null : data, cp: isCp ? data : null });
    };

    const patchModel = (model: Glm51Model, isCp: boolean) => {
      const origMlaLayer = (model as any).mlaLayer.bind(model);
      const origForward = model.forward.bind(model);
      let captured = false;

      (model as any).mlaLayer = function(normed: Tensor, residual: Tensor, layerIdx: number, state: ExecutionState) {
        if (layerIdx === 0 && !captured) {
          captured = true;
          const cfg2 = (model as any).cfg;
          const ws = state.ws;
          const pfx = `model.layers.${layerIdx}.self_attn`;
          const nH = cfg2.numAttentionHeads;
          const kvLR = cfg2.kvLoraRank;
          const qkRD = cfg2.qkRopeHeadDim;
          const vHD = cfg2.vHeadDim;
          const scaling = cfg2.scaling;
          const contextParallel = (model as any).contextParallel;
          const pagedKV = state.cache.getPagedKV();
          const bs = state.batchSize;
          const totalTokens = state.totalTokens;

          snap3("L0_normed_input", normed, BS * hs, isCp);

          using rotaryEmbedding = (model as any).glm.withStream(() =>
            (model as any).invFreq.rotaryEmbedding(ws.positionIds, qkRD / 2, B, S));
          using cos = rotaryEmbedding.result.cos;
          using sin = rotaryEmbedding.result.sin;

          using kvcache = (model as any).glm.withStream(() => {
            using kPeRopeStream = (model as any).glm.withStream(() => {
              using kPeRaw = normed.linear((model as any).tensors.get(`${pfx}.k_pe_proj.weight`), BS);
              rotaryEmbedding.streamWaitEvent();
              return kPeRaw.applyRotaryPosEmb(cos, sin, qkRD, 1, S, B, 1, cfg2.ropeInterleave);
            });
            using kPeRope = kPeRopeStream.result;
            using ckv = normed.linear((model as any).tensors.get(`${pfx}.ckv_proj.weight`), BS);
            using ckvNormed = ckv.rmsnorm((model as any).tensors.get(`${pfx}.kv_a_layernorm.weight`), cfg2.rmsNormEps, kvLR, BS);
            kPeRopeStream.streamWaitEvent();
            state.mlaKvCacheAppend(ckvNormed, kPeRope, layerIdx, kvLR, qkRD);
          });

          using q = (model as any).glm.withStream(() => {
            using qResidBuf = normed.linear((model as any).tensors.get(`${pfx}.q_a_proj.weight`), BS);
            using qNormed = qResidBuf.rmsnorm((model as any).tensors.get(`${pfx}.q_a_layernorm.weight`), cfg2.rmsNormEps, cfg2.qLoraRank, BS);
            using qAbsorbedLin = qNormed.linear((model as any).tensors.get(`${pfx}.absorbed.weight`), BS);
            using qPeLin = qNormed.linear((model as any).tensors.get(`${pfx}.q_pe_proj.weight`), BS);
            rotaryEmbedding.streamWaitEvent();
            using qPeR = (model as any).glm.withStream(() => qPeLin.ropeTranspose(cos, sin, qkRD, qkRD, nH, S, B, qkRD, cfg2.ropeInterleave));
            const qAbsorbedR = qAbsorbedLin.ropeTranspose(cos, sin, 0, kvLR, nH, S, B, kvLR);
            snap3("L0_qAbsorbedR", qAbsorbedR, BS * nH * kvLR, isCp);
            snap3("L0_qPeR", qPeR.result, BS * nH * qkRD, isCp);
            return { qAbsorbedR, qPeR: qPeR.result };
          });

          using qAbsorbedR = q.result.qAbsorbedR;
          using qPeR = q.result.qPeR;
          kvcache.streamWaitEvent();
          q.streamWaitEvent();

          const mlaResult = ws.mlaPrefillPaged(qAbsorbedR, qPeR, pagedKV, layerIdx, totalTokens, bs, nH, kvLR, qkRD, scaling, contextParallel);

          // attnOut is PartialSoftmax in CP mode — can't d2h, skip comparison
          // snap("L0_attnOut_before_vexpand", mlaResult.o, BS * nH * kvLR, isCp);

          using attnOut = mlaResult.o;          using lseBuf = mlaResult.lse;
          const vProj = (model as any).tensors.get(`${pfx}.v_proj.weight`);
          using vExpanded = attnOut.mlaVExpand(vProj, kvLR, vHD, nH, S, B, lseBuf);

          snap3("L0_vExpanded_after_merge", vExpanded, BS * nH * vHD, isCp);

          using oProjBuf = vExpanded.linear((model as any).tensors.get(`${pfx}.o_proj.weight`), BS);

          snap3("L0_oProjBuf", oProjBuf, BS * hs, isCp);

          const attnResult = residual.fusedAddRmsnorm(
            oProjBuf,
            (model as any).tensors.get(`model.layers.${layerIdx}.post_attention_layernorm.weight`),
            cfg2.rmsNormEps, hs, BS);

          snap3("L0_attn_residual", attnResult.residual, BS * hs, isCp);
          snap3("L0_attn_normed", attnResult.normed, BS * hs, isCp);

          using attnNormed = attnResult.normed;
          using attnResidual = attnResult.residual;

          const mlpPfx = `model.layers.${layerIdx}`;
          using downBuf = layerIdx >= cfg2.firstSparseMlpLayer
            ? (model as any).mlpSparse(attnNormed, mlpPfx, BS)
            : (model as any).mlpDense(attnNormed, mlpPfx, BS);

          const nextWeight = layerIdx < cfg2.numHiddenLayers - 1
            ? (model as any).tensors.get(`model.layers.${layerIdx + 1}.input_layernorm.weight`)
            : (model as any).tensors.get("model.norm.weight");
          const mlpResult = attnResidual.fusedAddRmsnorm(downBuf, nextWeight, cfg2.rmsNormEps, hs, BS);

          snap3("L0_final_residual", mlpResult.residual, BS * hs, isCp);
          snap3("L0_final_normed", mlpResult.normed, BS * hs, isCp);

          return { normed: mlpResult.normed, residual: mlpResult.residual };
        }
        return origMlaLayer(normed, residual, layerIdx, state);
      };

      (model as any).forward = function(state: ExecutionState): Tensor {
        const embedTable = (model as any).tensors.get("model.embed_tokens.weight");
        using residual = new UsingHolder(embedTable.embedding(state.ws.inputIdsBuf, hs, BS));
        using normed = new UsingHolder(residual.value.rmsnorm(
          (model as any).tensors.get(`model.layers.0.input_layernorm.weight`), cfg.rmsNormEps, hs, BS));
        for (let i = 0; i < cfg.numHiddenLayers; i++) {
          const r = (model as any).mlaLayer(normed.value, residual.value, i, state);
          normed.replace(r.normed);
          residual.replace(r.residual);
        }
        return state.computeLogits(normed.value, model as any);
      };

      return { origForward, origMlaLayer };
    };

    const refRestore = patchModel(modelRef, false);
    const cpRestore = patchModel(modelCp, true);

    const cacheRef = modelRef.createChatCache(MAX_PAGES);
    const cacheCp = modelCp.createChatCache(MAX_PAGES);
    cacheRef.reset(1); cacheCp.reset(1);
    const suffixRefLayer = cacheRef.prefixMatch(0, INPUT_IDS);
    const suffixCpLayer = cacheCp.prefixMatch(0, INPUT_IDS);

    wsRef.forwardEagerPrefill(modelRef, [suffixRefLayer], cacheRef);
    wsCp.forwardEagerPrefill(modelCp, [suffixCpLayer], cacheCp);
    cacheRef.appendTokens(0, suffixRefLayer);
    cacheCp.appendTokens(0, suffixCpLayer);

    (modelRef as any).forward = refRestore.origForward;
    (modelRef as any).mlaLayer = refRestore.origMlaLayer;
    (modelCp as any).forward = cpRestore.origForward;
    (modelCp as any).mlaLayer = cpRestore.origMlaLayer;

    console.log("\nLayer 0 MLA attention intermediate values:");
    for (const s of snapshots) {
      if (s.ref && s.cp) compareArrays(s.ref, s.cp, s.label);
      else console.log(`  ${s.label}: MISSING ${s.ref ? "cp" : "ref"} data`);
    }

    cacheRef.free(); cacheCp.free();
  });
});
