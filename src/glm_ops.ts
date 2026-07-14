import fs from "node:fs";
import path from "node:path";
import { DeviceOps, MaskMode, StridedMmap, TensorParallelism } from "./device_ops";
import type { SamplingParams } from "./chat_model";
import { MemcpyKind, Tensor } from "./tensor";
import { SafeTensorFile } from "./safetensors";
import type { WorkspaceBase } from "./workspace";
import type { PagedKVCache } from "./paged_kv";
import type { ExecutionState } from "./execution-workspace";
import { Allocator, ArenaAllocator } from "./allocator";

// Above `count == topK` (single-token decode), mulMatId can either:
//  - run the direct per-(token,expert) GEMV kernel (mulMatId/nvfp4MulMatId), which
//    re-reads each expert's weight once per routed token (redundant when several
//    tokens share an expert, but a single dependency-free kernel launch), or
//  - run the grouped/sorted pipeline (mulMatIdGrouped/nvfp4MulMatIdGrouped), which
//    reads each expert's weight once regardless of how many tokens route to it, at
//    the cost of a 6-stage histogram/scatter/gemv/unscatter dependency chain.
// The grouped pipeline's per-stage overhead dominates at small counts (e.g. MTP
// verification batches), so the direct path wins there despite redundant reads;
// the grouped path only pays off once `count` is large enough to amortize its
// dispatch overhead against avoided redundant weight reads (true prefill territory).
const MUL_MAT_ID_GROUPED_THRESHOLD = 512;

// Below this query-token count, sparse MLA prefill is routed to the split-K
// decode kernel for better GPU occupancy (e.g. MTP tree verify). Above it, the
// prefill kernel's per-token CTAs already fill the GPU and amortize KV loads.
// Tunable — the crossover is roughly the SM count divided by heads/HPB.
const SPARSE_MLA_DECODE_DISPATCH_MAX = Number(process.env.GLM_SPARSE_DECODE_DISPATCH_MAX ?? 64);

// At or below this query-token count the indexer scores via the "direct" v2 path
// (simple per-position score kernel + a single 65536-bucket histogram): lowest
// per-launch overhead and best occupancy for small Q, since the tensor-core
// prefill score kernel underutilizes its TM=64 query tile when Q is tiny (decode,
// MTP tree verify). The v2 histogram is 256 KB/query so it can't scale — above the
// threshold the memory-scalable two-level tensor-core prefill path is used. Both
// paths support custom masks and query-sharding, so this is a pure occupancy/
// memory tradeoff. Tunable to align with SPARSE_MLA_DECODE_DISPATCH_MAX.
const INDEXER_DIRECT_DISPATCH_MAX = Number(process.env.GLM_INDEXER_DIRECT_DISPATCH_MAX ?? 64);

function findProjectRoot(dir: string): string {
  let d = dir;
  while (d !== path.dirname(d)) {
    if (fs.existsSync(path.join(d, "package.json"))) return d;
    d = path.dirname(d);
  }
  return dir;
}

function ptr(t: Tensor): number {
  return t.data;
}

let nativeAddon: NativeAddon | null = null;

export function getNativeAddon(): NativeAddon {
  if (!nativeAddon) {
    const p = path.join(findProjectRoot(__dirname), "build", "Release", "glm.node");
    nativeAddon = require(p) as NativeAddon;
  }
  return nativeAddon;
}

export function mmapOpen(filePath: string): number {
  const p = getNativeAddon().mmapOpen(filePath);
  if (!p) throw new Error(`glm_mmap_open failed for ${filePath}`);
  return p;
}

export function mmapClose(mmapPtr: number, size: number): void {
  getNativeAddon().mmapClose(mmapPtr, size);
}

interface NativeAddon {
  init(deviceId: number): number;
  free(ctx: number): void;
  alloc(ctx: number, size: number): number;
  freeBuf(ctx: number, ptr: number): void;
  h2d(ctx: number, dst: number, src: Buffer, size: number): void;
  writePointers(ctx: number, dst: number, p0: number, p1: number, p2: number, p3: number, p4: number, p5: number, p6: number, p7: number, n: number): void;
  d2h(ctx: number, dst: Buffer, src: number, size: number): void;
  rmsnorm(ctx: number, out: number, input: number, weight: number, eps: number, dim: number, batch: number): void;
  layernorm(ctx: number, out: number, input: number, weight: number, bias: number, eps: number, dim: number, batch: number): void;
  fusedAddRmsnorm(ctx: number, out: number, residual: number, inputA: number, inputB: number, weight: number, eps: number, dim: number, batch: number): void;
  fusedNormRope(ctx: number, out: number, input: number, weight: number, cos: number, sin: number, eps: number, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, inStride: number, interleaved?: boolean): void;
  siluAndMul(ctx: number, out: number, gate: number, up: number, intermediate: number, batch: number): void;
  linear(ctx: number, out: number, input: number, weight: number, batch: number, n: number, k: number): void;
  fill(ctx: number, out: number, value: number, n: number): void;
  causalMask(ctx: number, out: number, seqLen: number): void;
  indexerScore(ctx: number, out: number, q: number, kData: number, weights: number, pageIndices: number, pageIndptr: number, lastPageLen: number, qoIndptr: number, scale: number, totalQ: number, idxNHeads: number, idxHeadDim: number, pageSize: number, maxKvLen: number, causal: number): void;
  indexerScoreTopk(ctx: number, outIdx: number, q: number, kData: number, weights: number, pageIndices: number, pageIndptr: number, lastPageLen: number, qoIndptr: number, scale: number, totalQ: number, idxNHeads: number, idxHeadDim: number, pageSize: number, topk: number, causal: number, customMask?: number, maskIndptr?: number, maskKvLen?: number): void;
  indexerScoreTopkPrefill(ctx: number, outIdx: number, q: number, kData: number, weights: number, pageIndices: number, pageIndptr: number, lastPageLen: number, qoIndptr: number, scale: number, totalQ: number, idxNHeads: number, idxHeadDim: number, pageSize: number, topk: number, causal: number, customMask: number, maskIndptr: number, maskKvLen: number, scores: number, rowLen: number, maxKv: number, coarseHist: number, fineHist: number, meta: number, numSplits: number, qGlobalStart: number): void;
  indexerScoreTopkV2(ctx: number, outIdx: number, q: number, kData: number, weights: number, pageIndices: number, pageIndptr: number, lastPageLen: number, qoIndptr: number, scale: number, totalQ: number, idxNHeads: number, idxHeadDim: number, pageSize: number, topk: number, causal: number, customMask: number, maskIndptr: number, maskKvLen: number, scores: number, rowLen: number, hist: number, meta: number, maxKv: number, numSplits: number, qGlobalStart: number): void;
  topkToSlots(ctx: number, slots: number, topkLength: number, topkIdx: number, pageIndices: number, pageIndptr: number, lastPageLen: number, batchIndices: number, numTokens: number, topk: number, pageSize: number, cpWorldSize: number, cpRank: number, kvTokenIndptr?: number): void;
  rotaryEmbedding(ctx: number, cosOut: number, sinOut: number, invFreq: number, positionIds: number, dimHalf: number, batch: number, seqLen: number): void;
  applyRotaryPosEmb(ctx: number, out: number, input: number, cos: number, sin: number, ropeDim: number, nHeads: number, seqLen: number, batch: number, unsqueezeDim: number, interleaved?: boolean): void;
  indexSelect(ctx: number, out: number, src: number, indices: number, dim: number, k: number, offset: number): void;
  gather(ctx: number, out: number, input: number, indices: number, k: number, inDim: number, batch: number, elemSize: number): void;
  deinterleave(ctx: number, out: number, input: number, worldSize: number, maxTotalLen: number, pageIndptr: number, kvTokenIndptr: number, batchSize: number, pageSize: number, D: number): void;
  gatherPages(ctx: number, out: number, input: number, pageIndices: number, pageIndptr: number, lastPageLen: number, maxPages: number, batchSize: number, pageSize: number, D: number): void;
  arange(ctx: number, out: number, start: number, step: number, count: number): void;
  max(ctx: number, outValues: number, outIndices: number, input: number, dim: number, batch: number, offset: number): void;
  memcpy(ctx: number, dst: number, src: number, bytes: number, kind: number): void;
  memcpyMulti(ctx: number, src: number, dst0: number, dst1: number, dst2: number, dst3: number, dst4: number, dst5: number, dst6: number, dst7: number, N: number, numel: number, dtype: number): void;
  sumPointersDirect(ctx: number, p0: number, p1: number, p2: number, p3: number, p4: number, p5: number, p6: number, p7: number, output: number, N: number, numel: number, dtype: number): void;
  kvCacheWrite(ctx: number, srcK: number, srcV: number, dstK: number, dstV: number, slotMapping: number, batchSize: number, nKv: number, hd: number, pageSize: number, srcKTokenStride: number, srcKHeadStride: number, srcVTokenStride: number, srcVHeadStride: number): void;
  positionStep(ctx: number, positionIds: number, lastPageLen: number, slotMapping: number, indptr: number, indices: number, pageSize: number, batchSize: number, steps: number): void;
  mlaPositionStep(ctx: number, positionIds: number, lastPageLen: number, indptr: number, pageSize: number, batchSize: number, cpWorldSize: number, cpRank: number, steps: number, globalLastPageLen: number): void;
  synchronize(ctx: number): void;
  synchronizeStream(ctx: number, streamIdx: number): void;
  setStream(ctx: number, streamIdx: number): void;
  eventRecord(ctx: number, eventIdx: number, streamIdx: number): void;
  streamWaitEvent(ctx: number, streamIdx: number, eventIdx: number): void;
  allocPinned(bytes: number): number;
  freePinned(ptr: number): void;
  hostPointerToBuffer(ptr: number, size: number): Buffer;
  batchDecodePlan(ctx: number, floatWs: number, floatWsSize: number, intWs: number, pinnedIntWs: number, intWsSize: number, planInfo: number, indptrH: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, enableCudaGraph: boolean): void;
  batchDecodeRun(ctx: number, q: number, o: number, kData: number, vData: number, indices: number, indptrD: number, lastPageLen: number, floatWs: number, intWs: number, planInfo: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, smScale: number): void;
  batchPrefillPagedPlan(ctx: number, floatWs: number, floatWsSize: number, intWs: number, pinnedIntWs: number, intWsSize: number, planInfo: number, qoIndptrH: number, pagedKvIndptrH: number, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, maskMode: number): void;
  batchPrefillPagedRun(ctx: number, q: number, o: number, kData: number, vData: number, indices: number, indptrD: number, lastPageLen: number, floatWs: number, intWs: number, qIndptrD: number, planInfo: number, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, qStrideN: number, qStrideH: number, maskMode: number, smScale: number): void;
  batchPrefillRaggedPlan(ctx: number, floatWs: number, floatWsSize: number, intWs: number, pinnedIntWs: number, intWsSize: number, planInfo: number, qoIndptrH: number, kvIndptrH: number, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, maskMode: number): void;
  batchPrefillRaggedRun(ctx: number, q: number, k: number, v: number, o: number, floatWs: number, intWs: number, qIndptrD: number, kvIndptrD: number, planInfo: number, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, qStrideN: number, qStrideH: number, kvStrideN: number, kvStrideH: number, vStrideN: number, vStrideH: number, maskMode: number, smScale: number): void;
  graphBeginCapture(ctx: number): void;
  graphEndCapture(ctx: number): number;
  graphInstantiate(graph: number): number;
  graphLaunch(graphExec: number, ctx: number): void;
  graphDestroy(graph: number): void;
  graphExecDestroy(graphExec: number): void;
  mmapOpen(path: string): number;
  mmapLoadAsync(ctx: number, gpuDst: number, mmapPtr: number, offset: number, nbytes: number): Promise<void>;
  memcpyHostToDeviceAsync(ctx: number, dst: number, src: number, nbytes: number): Promise<void>;
  memcpy2dHostToDeviceAsync(ctx: number, dst: number, dpitch: number, src: number, spitch: number, width: number, height: number): Promise<void>;
  mmapClose(mmapPtr: number, size: number): void;
  fp8LinearDecode(ctx: number, bf16Out: number, bf16Input: number, fp8Weight: number, weightScale: number, m: number, n: number, k: number): void;
  nvfp4LinearDecode(ctx: number, bf16Out: number, bf16Input: number, fp4Weight: number, weightScale: number, weightScale2: number, m: number, n: number, k: number, bf16Workspace?: number): void;
  gdnRecurrentStep(ctx: number, output: number, state: number, qkv: number, aRaw: number, bRaw: number, aLog: number, dtBias: number, numHeads: number, dK: number, dV: number, batchSize: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void;
  gdnPrefill(ctx: number, output: number, state: number, qkv: number, aRaw: number, bRaw: number, aLog: number, dtBias: number, cuSeqlens: number, totalSeqLen: number, numHeads: number, dK: number, dV: number, batchSize: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void;
  mlaPrefillPlan(ctx: number, floatWs: number, floatWsSize: number, intWs: number, pinnedIntWs: number, intWsSize: number, planInfo: number, qoIndptrH: number, kvIndptrH: number, kvLenH: number, batchSize: number, numHeads: number, headDimO: number, causal: boolean, cpWorldSize?: number, cpRank?: number): void;
  mlaPrefillRun(ctx: number, qNope: number, qPe: number, ckvData: number, kpeData: number, kvIndices: number, o: number, floatWs: number, intWs: number, planInfo: number, numHeads: number, pageSize: number, maskMode: number, smScale: number, qNopeStrideN: number, qNopeStrideH: number, qPeStrideN: number, qPeStrideH: number, ckvStridePage: number, ckvStrideN: number, kpeStridePage: number, kpeStrideN: number, oStrideN: number, oStrideH: number, headDimCkv: number, headDimKpe: number, lse: number, cpWorldSize: number, cpRank: number, customMask?: number, maskIndptr?: number, maskKvLen?: number): void;
  mlaDecodePlan(ctx: number, floatWs: number, floatWsSize: number, intWs: number, pinnedIntWs: number, intWsSize: number, planInfo: number, indptrH: number, batchSize: number, numQoHeads: number, pageSize: number, enableCudaGraph: boolean, headDimCkv: number, headDimKpe: number): void;
  mlaDecodeRun(ctx: number, qNope: number, qPe: number, ckvData: number, kpeData: number, indices: number, indptrD: number, lastPageLen: number, o: number, floatWs: number, intWs: number, planInfo: number, batchSize: number, numQoHeads: number, pageSize: number, smScale: number, headDimCkv: number, headDimKpe: number, lse: number): void;
  mlaKvCacheAppend(ctx: number, ckvData: number, kpeData: number, indices: number, indptr: number, lastPageLen: number, appendCkv: number, appendKpe: number, batchIndices: number, positions: number, nnz: number, pageSize: number, headDimCkv: number, headDimKpe: number, appendCkvStrideN: number, appendKpeStrideN: number, cpWorldSize: number, cpRank: number): void;
  concatAndCacheDsMla(ctx: number, kvCache: number, appendCkv: number, appendKpe: number, indices: number, indptr: number, batchIndices: number, positions: number, nnz: number, pageSize: number, kvLoraRank: number, peDim: number, appendCkvStrideN: number, appendKpeStrideN: number, cpWorldSize: number, cpRank: number): void;
  sparseMlaPrefill(ctx: number, q: number, kvCache: number, indices: number, output: number, outLse: number, numTokens: number, numHeads: number, topk: number, pageBlockSize: number, smScale: number, strideKvBlock: number, topkLength?: number): void;
  sparseMlaDecode(ctx: number, q: number, kvCache: number, indices: number, midOut: number, midLse: number, output: number, outLse: number, numTokens: number, numHeads: number, topk: number, numSplits: number, smScale: number, strideKvBlock: number, chunksPerBlock: number, topkLength?: number): void;
  causalConv1d(ctx: number, output: number, convState: number, input: number, weight: number, cuSeqlens: number, convDim: number, totalSeqLen: number, kernelSize: number, batchSize: number, convStateStride: number, chStride: number, seqStride: number): void;
  causalConv1dUpdate(ctx: number, output: number, convState: number, input: number, weight: number, convDim: number, kernelSize: number, batchSize: number, convStateStride: number): void;
  rmsnormGated(ctx: number, output: number, input: number, gate: number, weight: number, eps: number, dim: number, batch: number): void;
  gateSigmoidMul(ctx: number, attnOut: number, gateInterleaved: number, batchSeq: number, numHeads: number, headDim: number): void;
  sampleBatch(ctx: number, outTokens: number, topkVals: number, topkIdxs: number, workspace: number, logits: number, penaltyTokens: number, penaltyCount: number, maxWindow: number, vocabSize: number, batchSize: number, temperatures: number, repPenalties: number, presPenalties: number, topKs: number, topPs: number, stepCounter: number, maxEffectiveK: number): void;
  memcpy2d(ctx: number, dst: number, dpitch: number, src: number, spitch: number, width: number, height: number, kind: number): void;
  memcpyPeer(ctx: number, dst: number, dstDevice: number, src: number, srcDevice: number, bytes: number): void;
  memcpy3dPeer(ctx: number, dstPtr: number, dstPitch: number, dstXSize: number, dstYSize: number, dstDevice: number, dstPosX: number, dstPosY: number, dstPosZ: number, srcPtr: number, srcPitch: number, srcXSize: number, srcYSize: number, srcDevice: number, srcPosX: number, srcPosY: number, srcPosZ: number, width: number, height: number, depth: number): void;
  bmm(ctx: number, C: number, A: number, B: number, alpha: number, beta: number, batch: number, M: number, N: number, K: number, transA: number, transB: number): void;
  ropeTranspose(ctx: number, out: number, input: number, cos: number, sin: number, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, inStride: number, interleaved?: boolean): void;
  mlaVExpand(ctx: number, result: number, attnOut: number, vProj: number, kvLoraRank: number, vHeadDim: number, nHeads: number, seqLen: number, batch: number, attnNHeads: number, headOffset: number, vProjHeadOffset: number): void;
  transpose4d(ctx: number, out: number, input: number, d0: number, d1: number, d2: number, d3: number, p0: number, p1: number, p2: number, p3: number): void;
  ncclUniqueId(outId: Buffer): void;
  ncclGroupStart(): void;
  ncclGroupEnd(): void;
  ncclCommInitRank(deviceId: number, rank: number, worldSize: number, uniqueId: number): number;
  ncclCommInitAll(deviceIds: number[]): number[];
  ncclCommDestroy(comm: number): void;
  ncclAllReduce(comm: number, ctx: number, sendbuff: number, recvbuff: number, count: number, datatype: number, op: number): void;
  ncclAllGather(comm: number, ctx: number, sendbuff: number, recvbuff: number, count: number, datatype: number): void;
  ncclSend(comm: number, ctx: number, sendbuff: number, count: number, datatype: number, peer: number): void;
  ncclRecv(comm: number, ctx: number, recvbuff: number, count: number, datatype: number, peer: number): void;
  ncclReduceScatter(comm: number, ctx: number, sendbuff: number, recvbuff: number, recvcount: number, datatype: number, op: number): void;
  p2pEnablePeerAccess(ctx: number, peerDevice: number): number;
  p2pCreateInstance(ctx: number, myRank: number, deviceIds: number[]): number;
  p2pDestroyInstance(instance: number): void;
  p2pGetFlagPtr(instance: number): number;
  p2pSetPeers(ctx: number, instance: number, flagPtrs: number[]): void;
  p2pAllGatherSmem(ctx: number, p0: number, p1: number, p2: number, p3: number, p4: number, p5: number, p6: number, p7: number, output: number, N: number, shardBytes: number, rank: number): void;
  p2pAllGatherRowSmem(ctx: number, p0: number, p1: number, p2: number, p3: number, p4: number, p5: number, p6: number, p7: number, output: number, N: number, shardDim1Bytes: number, fullDim1Bytes: number, outer: number, rank: number): void;
  p2pBarrier(ctx: number, instance: number, peerRank?: number): void;
  cpMergeTree(ctx: number, v0: number, v1: number, v2: number, v3: number, v4: number, v5: number, v6: number, v7: number, lse0: number, lse1: number, lse2: number, lse3: number, lse4: number, lse5: number, lse6: number, lse7: number, numShards: number, outputV: number, outputLse: number, numel: number, batchSize: number, numHeads: number, vHeadDim: number, shardNHeads: number, headOffset: number, inputNHeads: number): void;
  cpCorrectAttnOut(ctx: number, vOut: number, lses: number, globalLse: number, batchSize: number, numHeads: number, vHeadDim: number, worldSize: number, rank: number): void;
  sigmoid(ctx: number, out: number, input: number, n: number): void;
  relu(ctx: number, out: number, input: number, n: number): void;
  topk(ctx: number, outValues: number, outIndices: number, input: number, k: number, dim: number, batch: number, offset: number): void;
  indexAdd(ctx: number, out: number, indices: number, values: number, nIndices: number, dim: number): void;
  add(ctx: number, out: number, a: number, b: number, n: number): void;
  addBroadcast(ctx: number, out: number, a: number, b: number, dim: number, rows: number): void;
  scale(ctx: number, out: number, input: number, scale: number, n: number): void;
  sumPointers(ctx: number, p0: number, p1: number, p2: number, p3: number, p4: number, p5: number, p6: number, p7: number, out: number, n: number, numel: number, dtype: number, writeback?: boolean): void;
  rmsNormPointers(ctx: number, p0: number, p1: number, p2: number, p3: number, p4: number, p5: number, p6: number, p7: number, inputA: number, weight: number, out: number, residual: number, n: number, numel: number, dim: number, eps: number, dtype: number): void;
  mul(ctx: number, out: number, a: number, b: number, n: number): void;
  mulBroadcast(ctx: number, out: number, a: number, b: number, dim: number, rows: number): void;
  scatterScalar(ctx: number, out: number, indices: number, value: number, k: number, outDim: number, batch: number): void;
  maskedFill(ctx: number, out: number, input: number, mask: number, value: number, n: number): void;
  applyRotaryPosEmbPartial(ctx: number, out: number, input: number, cos: number, sin: number, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, unsqueezeDim: number, interleaved?: boolean): void;
  rowScaleAdd(ctx: number, out: number, input: number, scales: number, rows: number, dim: number): void;
  reduceSum(ctx: number, out: number, input: number, rows: number, cols: number): void;
  rowNormalize(ctx: number, out: number, input: number, scale: number, rows: number, cols: number, normalize: boolean): void;
  groupMaskMul(ctx: number, scores: number, groupMask: number, numExperts: number, expertsPerGroup: number, nGroup: number, batch: number): void;
  expertScale(ctx: number, out: number, weights: number, indices: number, expertId: number, topK: number, batch: number): void;
  mulMatId(ctx: number, output: number, input: number, weightPtrs: number, expertIds: number, topK: number, count: number, N: number, K: number): void;
  mulMatIdGrouped(ctx: number, output: number, input: number, weightPtrs: number, expertIds: number, topK: number, count: number, N: number, K: number, numExperts: number, workspace: number): void;
  groupedMoeWorkspaceSize(count: number, N: number, K: number, numExperts: number): number;
  nvfp4MulMatId(ctx: number, output: number, input: number, weightPtrs: number, scalePtrs: number, scale2Ptrs: number, expertIds: number, topK: number, count: number, N: number, K: number): void;
  nvfp4MulMatIdGrouped(ctx: number, output: number, input: number, weightPtrs: number, scalePtrs: number, scale2Ptrs: number, expertIds: number, topK: number, count: number, N: number, K: number, numExperts: number, workspace: number): void;
  mmaMoeWorkspaceSize(count: number, N: number, K: number, numExperts: number): number;
  nvfp4MulMatIdGroupedMma(ctx: number, output: number, input: number, weightPtrs: number, scalePtrs: number, scale2Ptrs: number, expertIds: number, topK: number, count: number, N: number, K: number, numExperts: number, workspace: number): void;
  mmaMoePcWorkspaceSize(count: number, N: number, K: number, numExperts: number): number;
  nvfp4MulMatIdGroupedMmaPc(ctx: number, output: number, input: number, weightPtrs: number, scalePtrs: number, scale2Ptrs: number, expertIds: number, topK: number, count: number, N: number, K: number, numExperts: number, workspace: number): void;
  mmaMoeCoopWorkspaceSize(count: number, N: number, K: number, numExperts: number): number;
  nvfp4MulMatIdGroupedMmaCoop(ctx: number, output: number, input: number, weightPtrs: number, scalePtrs: number, scale2Ptrs: number, expertIds: number, topK: number, count: number, N: number, K: number, numExperts: number, workspace: number): void;
  mmaMoeCoopScatterWorkspaceSize(count: number, K: number, numExperts: number): number;
  mmaMoeCoopGemmWorkspaceSize(count: number, N: number): number;
  mmaMoeCoopScatter(ctx: number, input: number, expertIds: number, topK: number, count: number, K: number, numExperts: number, workspace: number): void;
  mmaMoeCoopGemm(ctx: number, weightPtrs: number, scalePtrs: number, scale2Ptrs: number, numExperts: number, N: number, K: number, count: number, scatterWorkspace: number, gemmWorkspace: number, output: number): void;
  mmaMoeCoopUnscatter(ctx: number, output: number, count: number, N: number, K: number, numExperts: number, scatterWorkspace: number, gemmWorkspace: number): void;
  bf16MulMatIdGroupedMma(ctx: number, output: number, input: number, weightPtrs: number, expertIds: number, topK: number, count: number, N: number, K: number, numExperts: number, workspace: number): void;
  scatterAddRows(ctx: number, out: number, input: number, scales: number, topK: number, dim: number, numRows: number, workspace: number): void;
  rotateInputIds(ctx: number, outputIds: number, inputIds: number, qoIndptr: number, newTokens: number, batchSize: number): void;
}

export class GlmTensor extends Tensor {
  constructor(workspace: WorkspaceBase, public readonly glm: GlmOps, data: number, allocSize: number, shape: number[], type: string, name: string | undefined, pinned: boolean, view: GlmTensor | undefined) {
    super(workspace, data, allocSize, shape, type, name, pinned, view);
  }

  [Symbol.dispose](): void {
    if (!this.canDispose()) {
      return;
    }
    // if stream is active, defer disposal until stream switch
    if (this.glm.currentStream) {
      this.glm.streamTensors.get(this.glm.currentStream)!.add(this);
    }
    else {
      super[Symbol.dispose]();
    }
  }

  free(): void {
    if (this.data !== 0) {
      if (this.pinned) {
        getNativeAddon().freePinned(this.data);
      } else {
        this.glm.allocator.free(this.data);
      }
      this.detachData();
    }
  }

  h2d(data: Buffer, size?: number): void {
    getNativeAddon().h2d(this.glm.ctx, this.data, data, size ?? data.length);
  }

  d2h(buf: Buffer, size?: number): void {
    getNativeAddon().d2h(this.glm.ctx, buf, this.data, size ?? buf.length);
  }

  linear(weight: Tensor, batch: number): Tensor {
    super.linear(weight, batch);
    const n = weight.shape[0];
    let k = weight.shape[1];
    const outShape = [batch, n];
    const out = this.workspace.alloc(outShape, this.type);
    if (weight.type === "F8_E4M3") {
      const scale = weight.workspace.tensors.get(weight.name! + "_scale_inv")!;
      getNativeAddon().fp8LinearDecode(this.glm.ctx, out.data, this.data, weight.data, scale.data, batch, n, k);
    } else if (weight.type === "U8") {
      k = k * 2; // NVFP4: weight is [N, K/2] packed, kernel expects K
      const scale = weight.workspace.tensors.get(weight.name! + "_weight_scale")!;
      const scale2 = weight.workspace.tensors.get(weight.name! + "_weight_scale_2")!;
      getNativeAddon().nvfp4LinearDecode(this.glm.ctx, out.data, this.data, weight.data, scale.data, scale2.data, batch, n, k, 0);
    } else {
      getNativeAddon().linear(this.glm.ctx, out.data, this.data, weight.data, batch, n, k);
    }
    return out;
  }

  bmm(B: Tensor, batch: number, M: number, N: number, K: number, transA: boolean = false, transB: boolean = false): Tensor {
    const out = this.workspace.alloc([batch * M, N], this.type);
    getNativeAddon().bmm(this.glm.ctx, out.data, this.data, B.data, 1.0, 0.0, batch, M, N, K, transA ? 1 : 0, transB ? 1 : 0);
    return out;
  }

  transpose4d(d0: number, d1: number, d2: number, d3: number, p0: number, p1: number, p2: number, p3: number): Tensor {
    const out = this.workspace.alloc([d0 * d1 * d2 * d3], this.type);
    getNativeAddon().transpose4d(this.glm.ctx, out.data, this.data, d0, d1, d2, d3, p0, p1, p2, p3);
    return out;
  }

  writePointers(tensors: Tensor[]): void {
    super.writePointers(tensors);
    const n = tensors.length;
    if (n > 8) {
      const ptrs = new BigInt64Array(n);
      for (let i = 0; i < n; i++) {
         ptrs[i] = BigInt(tensors[i].data);
      }
      this.h2d(Buffer.from(ptrs.buffer));
      return;
    }
    const ptrs = new Array(8).fill(0);
    for (let i = 0; i < n; i++) {
      ptrs[i] = tensors[i].data;
    }
    getNativeAddon().writePointers(this.glm.ctx, this.data,
      ptrs[0], ptrs[1], ptrs[2], ptrs[3],
      ptrs[4], ptrs[5], ptrs[6], ptrs[7], n);
  }

  rmsnorm(weight: Tensor, eps: number, dim: number, batch: number): Tensor {
    super.rmsnorm(weight, eps, dim, batch);
    const out = this.workspace.alloc([batch, dim], this.type);
    getNativeAddon().rmsnorm(this.glm.ctx, out.data, this.data, weight.data, eps, dim, batch);
    return out;
  }

  layernorm(weight: Tensor, bias: Tensor, eps: number, dim: number, batch: number): Tensor {
    super.layernorm(weight, bias, eps, dim, batch);
    const out = this.workspace.alloc([batch, dim], this.type);
    getNativeAddon().layernorm(this.glm.ctx, out.data, this.data, weight.data, bias.data, eps, dim, batch);
    return out;
  }

  fusedAddRmsnorm(input: Tensor, weight: Tensor, eps: number, dim: number, batch: number): { normed: Tensor, residual: Tensor } {
    super.fusedAddRmsnorm(input, weight, eps, dim, batch);
    const normed = this.workspace.alloc([batch, dim], this.type);
    const residual = this.workspace.alloc([batch, dim], this.type);
    getNativeAddon().fusedAddRmsnorm(this.glm.ctx, normed.data, residual.data, this.data, input.data, weight.data, eps, dim, batch);
    return { normed, residual };
  }

  fusedNormRope(weight: Tensor, cos: Tensor, sin: Tensor, eps: number, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, inStride?: number, interleaved?: boolean): Tensor {
    super.fusedNormRope(weight, cos, sin, eps, ropeDim, headDim, nHeads, seqLen, batch, inStride);
    const out = this.workspace.alloc([batch, nHeads, seqLen, headDim], this.type);
    getNativeAddon().fusedNormRope(this.glm.ctx, out.data, this.data, weight.data, cos.data, sin.data, eps, ropeDim, headDim, nHeads, seqLen, batch, inStride ?? headDim, interleaved ?? false);
    return out;
  }

  embedding(ids: Tensor, hidden: number, seqLen: number): Tensor {
    super.embedding(ids, hidden, seqLen);
    const out = ids.workspace.alloc([seqLen, hidden], this.type);
    getNativeAddon().indexSelect(this.glm.ctx, out.data, this.data, ids.data, hidden, seqLen, 0);
    return out;
  }

  siluAndMul(up: Tensor, intermediate: number, batch: number): Tensor {
    super.siluAndMul(up, intermediate, batch);
    const out = this.workspace.alloc([batch, intermediate], this.type);
    getNativeAddon().siluAndMul(this.glm.ctx, out.data, this.data, up.data, intermediate, batch);
    return out;
  }

  arange(start: number, step: number, count: number): void {
    super.arange(start, step, count);
    getNativeAddon().arange(this.glm.ctx, this.data, start, step, count);
  }

  argmax(): Tensor {
    super.argmax();
    const { indices, values } = this.max();
    values[Symbol.dispose]();
    return indices;
  }

  max(offset: number = 0): { values: Tensor, indices: Tensor } {
    super.max(offset);
    const dim = this.shape[1];
    const batch = this.shape[0];
    const values = this.workspace.alloc([batch], this.type);
    const indices = this.workspace.alloc([batch], "I32");
    getNativeAddon().max(this.glm.ctx, values.data, indices.data, this.data, dim, batch, offset);
    return { values, indices };
  }

  indexSelect(indices: Tensor, batch: number, offset: number = 0): Tensor {
    super.indexSelect(indices, batch, offset);
    const dim = this.shape[1];
    const out = this.workspace.alloc([batch, dim], this.type);
    getNativeAddon().indexSelect(this.glm.ctx, out.data, this.data, indices.data, dim, batch, offset);
    return out;
  }

  gather(indices: Tensor, k: number, inDim: number, batch: number): Tensor {
    super.gather(indices, k, inDim, batch);
    const out = this.workspace.alloc([batch, k], this.type);
    const elemSize = SafeTensorFile.dtypeBytes(this.type);
    getNativeAddon().gather(this.glm.ctx, out.data, this.data, indices.data, k, inDim, batch, elemSize);
    return out;
  }

  rotateInputIds(qoIndptr: Tensor, newTokens: Tensor, batchSize: number): Tensor {
    super.rotateInputIds(qoIndptr, newTokens, batchSize);
    const out = this.workspace.alloc(this.shape, this.type);
    getNativeAddon().rotateInputIds(this.glm.ctx, out.data, this.data, qoIndptr.data, newTokens.data, batchSize);
    return out;
  }

  causalConv1d(convState: Tensor, input: Tensor, weight: Tensor, cuSeqlens: Tensor, convDim: number, totalSeqLen: number, kernelSize: number, batchSize: number, convStateStride: number, chStride: number, seqStride: number): void {
    super.causalConv1d(convState, input, weight, cuSeqlens, convDim, totalSeqLen, kernelSize, batchSize, convStateStride, chStride, seqStride);
    getNativeAddon().causalConv1d(this.glm.ctx, this.data, convState.data, input.data, weight.data, cuSeqlens.data, convDim, totalSeqLen, kernelSize, batchSize, convStateStride, chStride, seqStride);
  }

  causalConv1dUpdate(convState: Tensor, input: Tensor, weight: Tensor, convDim: number, kernelSize: number, batchSize: number, convStateStride: number): Tensor {
    super.causalConv1dUpdate(convState, input, weight, convDim, kernelSize, batchSize, convStateStride);
    const out = this.workspace.alloc([batchSize * convDim], this.type);
    getNativeAddon().causalConv1dUpdate(this.glm.ctx, out.data, convState.data, input.data, weight.data, convDim, kernelSize, batchSize, convStateStride);
    return out;
  }

  rmsnormGated(input: Tensor, gate: Tensor, weight: Tensor, eps: number, dim: number, batch: number): void {
    super.rmsnormGated(input, gate, weight, eps, dim, batch);
    getNativeAddon().rmsnormGated(this.glm.ctx, this.data, input.data, gate.data, weight.data, eps, dim, batch);
  }

  gateSigmoidMul(gate: Tensor, batchSeq: number, numHeads: number, headDim: number): void {
    super.gateSigmoidMul(gate, batchSeq, numHeads, headDim);
    getNativeAddon().gateSigmoidMul(this.glm.ctx, this.data, gate.data, batchSeq, numHeads, headDim);
  }

  fill(value: number, n: number): void {
    getNativeAddon().fill(this.glm.ctx, this.data, value, n);
  }

  async mmapLoad(mmapPtr: number, offset: number, nbytes: number, strided?: StridedMmap): Promise<void> {
    if (process.env.GLM_SKIP_MMAP_LOAD) {
      // for testing: skip actual load
      return;
    }
    if (strided) {
      return this.memcpy2dHostToDeviceAsync(strided.dstOffset, strided.dstPitch, mmapPtr + offset + strided.srcOffset, strided.srcPitch, strided.width, strided.height);
    } else {
      return this.mmapLoadAsync(mmapPtr, offset, nbytes);
    }
  }

  async mmapLoadAsync(mmapPtr: number, offset: number, nbytes: number): Promise<void> {
    if (process.env.GLM_SKIP_MMAP_LOAD) {
      // for testing: skip actual load
      return;
    }
    return getNativeAddon().mmapLoadAsync(this.glm.ctx, this.data, mmapPtr, offset, nbytes);
  }

  memcpy2dHostToDeviceAsync(dstOffset: number, dpitch: number, src: number, spitch: number, width: number, height: number): Promise<void> {
    return getNativeAddon().memcpy2dHostToDeviceAsync(this.glm.ctx, this.data + dstOffset, dpitch, src, spitch, width, height);
  }

  memcpy(src: Tensor, size?: number, kind?: MemcpyKind): void {
    super.memcpy(src, size, kind);
    if (!(src instanceof GlmTensor)) {
      throw new Error("GlmTensor.memcpy requires GlmTensor source");
    }
    const bytes = size ?? Math.min(this.allocSize, src.allocSize);
    const copyKind = kind ?? (src.pinned ? MemcpyKind.HostToDevice : MemcpyKind.DeviceToDevice);
    // if (copyKind === MemcpyKind.DeviceToDevice && this.glm !== src.glm) {
    //   getNativeAddon().memcpyPeer(src.glm.ctx, this.data, this.glm.device, src.data, src.glm.device, bytes);
    // } else {
      getNativeAddon().memcpy(src.glm.ctx, this.data, src.data, bytes, memcpyKindToNative(copyKind));
    // }
  }

  memcpy2d(dstOffset: number, dpitch: number, src: Tensor, srcOffset: number, spitch: number, width: number, height: number, kind: MemcpyKind): void {
    super.memcpy2d(dstOffset, dpitch, src, srcOffset, spitch, width, height, kind);
    if (!(src instanceof GlmTensor)) {
      throw new Error("GlmTensor.memcpy requires GlmTensor source");
    }
    // if (kind === MemcpyKind.DeviceToDevice && this.glm !== (src as GlmTensor).glm) {
    //   const s = src as GlmTensor;
    //   getNativeAddon().memcpy3dPeer(
    //     src.glm.ctx,
    //     this.data + dstOffset, dpitch, width, height, this.glm.device,
    //     0, 0, 0,
    //     s.data + srcOffset, spitch, width, height, s.glm.device,
    //     0, 0, 0,
    //     width, height, 1,
    //   );
    // } else {
      getNativeAddon().memcpy2d(src.glm.ctx, this.data + dstOffset, dpitch, (src as GlmTensor).data + srcOffset, spitch, width, height, memcpyKindToNative(kind));
    // }
  }

  rotaryEmbedding(positionIds: Tensor, dimHalf: number, batch: number, seqLen: number): { cos: Tensor, sin: Tensor } {
    super.rotaryEmbedding(positionIds, dimHalf, batch, seqLen);
    const hd = dimHalf * 2;
    const cos = positionIds.workspace.alloc([batch, seqLen, hd], this.type);
    const sin = positionIds.workspace.alloc([batch, seqLen, hd], this.type);
    getNativeAddon().rotaryEmbedding(this.glm.ctx, cos.data, sin.data, this.data, positionIds.data, dimHalf, batch, seqLen);
    return { cos, sin };
  }

  ropeTranspose(cos: Tensor, sin: Tensor, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, inStride?: number, interleaved?: boolean): Tensor {
    super.ropeTranspose(cos, sin, ropeDim, headDim, nHeads, seqLen, batch, inStride);
    const out = this.workspace.alloc([batch * seqLen, nHeads, headDim], this.type);
    getNativeAddon().ropeTranspose(this.glm.ctx, out.data, this.data, ropeDim > 0 ? cos.data : 0, ropeDim > 0 ? sin.data : 0, ropeDim, headDim, nHeads, seqLen, batch, inStride ?? headDim, interleaved ?? false);
    return out;
  }

  applyRotaryPosEmb(cos: Tensor, sin: Tensor, ropeDim: number, nHeads: number, seqLen: number, batch: number, unsqueezeDim: number, interleaved?: boolean): Tensor {
    const out = this.workspace.alloc(this.shape, this.type);
    getNativeAddon().applyRotaryPosEmb(this.glm.ctx, out.data, this.data, cos.data, sin.data, ropeDim, nHeads, seqLen, batch, unsqueezeDim, interleaved ?? false);
    return out;
  }

  mlaVExpand(vProj: Tensor, kvLoraRank: number, vHeadDim: number, nHeads: number, seqLen: number, batch: number, _lse?: Tensor, headOffset: number = 0, attnNHeads: number = nHeads, vProjHeadOffset: number = 0, tokenMajor: boolean = false): Tensor {
    super.mlaVExpand(vProj, kvLoraRank, vHeadDim, nHeads, seqLen, batch);
    const BS = batch * seqLen;
    const out = this.workspace.alloc([BS, nHeads * vHeadDim], this.type);
    const effSeqLen = tokenMajor ? 1 : seqLen;
    const effBatch = tokenMajor ? BS : batch;
    getNativeAddon().mlaVExpand(this.glm.ctx, out.data, this.data, vProj.data, kvLoraRank, vHeadDim, nHeads, effSeqLen, effBatch, attnNHeads, headOffset, vProjHeadOffset);
    return out;
  }

  sigmoid(): Tensor {
    const n = this.shape.reduce((a, b) => a * b, 1);
    const out = this.workspace.alloc(this.shape, this.type);
    getNativeAddon().sigmoid(this.glm.ctx, out.data, this.data, n);
    return out;
  }

  topk(k: number, dim: number, offset = 0): { values: Tensor, indices: Tensor } {
    const batch = this.shape.reduce((a, b) => a * b, 1) / dim;
    const values = this.workspace.alloc([batch, k], this.type);
    const indices = this.workspace.alloc([batch, k], "I32");
    getNativeAddon().topk(this.glm.ctx, values.data, indices.data, this.data, k, dim, batch, offset);
    return { values, indices };
  }

  indexAdd(indices: Tensor, values: Tensor, nIndices: number, dim: number): void {
    getNativeAddon().indexAdd(this.glm.ctx, this.data, indices.data, values.data, nIndices, dim);
  }

  add(other: Tensor, n?: number): Tensor {
    if (this.shape.length === 2 && other.shape.length === 1 && this.shape[1] === other.shape[0]) {
      const out = this.workspace.alloc(this.shape, this.type);
      getNativeAddon().addBroadcast(this.glm.ctx, out.data, this.data, other.data, this.shape[1], this.shape[0]);
      return out;
    }
    const count = n ?? this.shape.reduce((a, b) => a * b, 1);
    const out = this.workspace.alloc(this.shape, this.type);
    getNativeAddon().add(this.glm.ctx, out.data, this.data, other.data, count);
    return out;
  }

  private runSumPointers(output: Tensor, inputs: Tensor[], writeback?: boolean): void {
    const N = inputs.length;
    const numel = output.shape.reduce((a, b) => a * b, 1);
    const dtype = output.type === "F32" ? 7 : 9;
    const ptrs = new Array<number>(8).fill(0);
    for (let i = 0; i < N; i++) ptrs[i] = inputs[i].data;
    getNativeAddon().sumPointers(
      this.glm.ctx,
      ptrs[0], ptrs[1], ptrs[2], ptrs[3],
      ptrs[4], ptrs[5], ptrs[6], ptrs[7],
      output.data, N, numel, dtype, writeback,
    );
  }

  /**
   * Fused P2P AllReduce + Add + RMSNorm over N peer partial-sum tensors.
   * Computes, per row of `dim` elements:
   *   s = inputA + sum(peers);  residual = s;  out = weight * s * rsqrt(mean(s^2)+eps)
   * `out` and `residual` receive the results; `peers` are read only.
   * Requires dim % 512 == 0 and 8192 % dim == 0 (row-aligned tiles).
   */
  rmsNormPointers(inputA: Tensor, peers: Tensor[], weight: Tensor, out: Tensor, residual: Tensor, dim: number, eps: number): void {
    const N = peers.length;
    const numel = out.shape.reduce((a, b) => a * b, 1);
    const dtype = out.type === "F32" ? 7 : 9;
    const ptrs = new Array<number>(8).fill(0);
    for (let i = 0; i < N; i++) ptrs[i] = peers[i].data;
    getNativeAddon().rmsNormPointers(
      this.glm.ctx,
      ptrs[0], ptrs[1], ptrs[2], ptrs[3],
      ptrs[4], ptrs[5], ptrs[6], ptrs[7],
      inputA.data, weight.data, out.data, residual.data,
      N, numel, dim, eps, dtype,
    );
  }

  sumInPlace(tensors: Tensor[], writeback?: boolean): void {
    super.sumInPlace(tensors);
    this.runSumPointers(this, tensors, writeback);
  }

  sum(tensors: Tensor[]): Tensor {
    super.sum(tensors);
    const out = this.workspace.alloc(this.shape, this.type);
    this.runSumPointers(out, [this as Tensor, ...tensors]);
    return out;
  }

  scaleInPlace(scale: number, n: number): void {
    getNativeAddon().scale(this.glm.ctx, this.data, this.data, scale, n);
  }

  mul(other: Tensor, n?: number): Tensor {
    if (this.shape.length === 2 && other.shape.length === 1 && this.shape[1] === other.shape[0]) {
      const out = this.workspace.alloc(this.shape, this.type);
      getNativeAddon().mulBroadcast(this.glm.ctx, out.data, this.data, other.data, this.shape[1], this.shape[0]);
      return out;
    }
    const count = n ?? this.shape.reduce((a, b) => a * b, 1);
    const out = this.workspace.alloc(this.shape, this.type);
    getNativeAddon().mul(this.glm.ctx, out.data, this.data, other.data, count);
    return out;
  }

  cat(tensors: Tensor[], dim: number): Tensor {
    super.cat(tensors, dim);
    const all = [this as Tensor, ...tensors];
    const outShape = [...this.shape];
    for (const t of tensors) outShape[dim] += t.shape[dim];
    const out = this.workspace.alloc(outShape, this.type);
    const elemBytes = SafeTensorFile.dtypeBytes(this.type);
    const outerStrides = this.shape.slice(0, dim).reduce((a, b) => a * b, 1);
    const innerStride = this.shape.slice(dim + 1).reduce((a, b) => a * b, 1);
    const dstRowBytes = outShape[dim] * innerStride * elemBytes;
    let offset = 0;
    for (const t of all) {
      const srcRowBytes = t.shape[dim] * innerStride * elemBytes;
      out.memcpy2d(
        offset,
        dstRowBytes,
        t,
        0,
        srcRowBytes,
        srcRowBytes,
        outerStrides,
        MemcpyKind.DeviceToDevice,
      );
      offset += srcRowBytes;
    }
    return out;
  }

  slice(dim: number, start: number, length: number): Tensor {
    super.slice(dim, start, length);
    if (start < 0) {
      start = this.shape[dim] + start;
    }
    const outShape = [...this.shape];
    outShape[dim] = length;
    const out = this.workspace.alloc(outShape, this.type);
    const elemBytes = SafeTensorFile.dtypeBytes(this.type);
    const outerStrides = this.shape.slice(0, dim).reduce((a, b) => a * b, 1);
    const innerStride = this.shape.slice(dim + 1).reduce((a, b) => a * b, 1);
    const srcPitch = this.shape[dim] * innerStride * elemBytes;
    const dstPitch = length * innerStride * elemBytes;
    const srcOffset = start * innerStride * elemBytes;
    out.memcpy2d(0, dstPitch, this, srcOffset, srcPitch, dstPitch, outerStrides, MemcpyKind.DeviceToDevice);
    return out;
  }

  narrow(start: number, length: number): Tensor {
    super.narrow(start, length);
    if (start < 0) {
      start = this.shape[0] + start;
    }
    const innerElements = this.shape.slice(1).reduce((a, b) => a * b, 1);
    const elemBytes = SafeTensorFile.dtypeBytes(this.type);
    const byteOffset = start * innerElements * elemBytes;
    const newShape = [length, ...this.shape.slice(1)];
    const newAllocSize = this.allocSize - byteOffset;
    return this.workspace.glm.wrapTensor(this.workspace, this.data + byteOffset, newAllocSize, newShape, this.type, this.pinned, this);
  }

  scatterScalar(indices: Tensor, value: number, k: number, outDim: number, batch: number): void {
    getNativeAddon().scatterScalar(this.glm.ctx, this.data, indices.data, value, k, outDim, batch);
  }

  maskedFill(mask: Tensor, value: number, n: number): void {
    getNativeAddon().maskedFill(this.glm.ctx, this.data, this.data, mask.data, value, n);
  }

  applyRotaryPosEmbPartial(cos: Tensor, sin: Tensor, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, unsqueezeDim: number, interleaved?: boolean): Tensor {
    const out = this.workspace.alloc(this.shape, this.type);
    getNativeAddon().applyRotaryPosEmbPartial(this.glm.ctx, out.data, this.data, cos.data, sin.data, ropeDim, headDim, nHeads, seqLen, batch, unsqueezeDim, interleaved ?? false);
    return out;
  }

  rowScaleAdd(input: Tensor, scales: Tensor, rows: number, dim: number): void {
    getNativeAddon().rowScaleAdd(this.glm.ctx, this.data, input.data, scales.data, rows, dim);
  }

  reduceSum(dim: number, batch: number): Tensor {
    const out = this.workspace.alloc([batch], this.type);
    getNativeAddon().reduceSum(this.glm.ctx, out.data, this.data, batch, dim);
    return out;
  }

  rowNormalize(scale: number, dim: number, batch: number, normalize: boolean = true): Tensor {
    const out = this.workspace.alloc([batch, dim], this.type);
    getNativeAddon().rowNormalize(this.glm.ctx, out.data, this.data, scale, batch, dim, normalize);
    return out;
  }

  groupMaskMul(groupMask: Tensor, numExperts: number, expertsPerGroup: number, nGroup: number, batch: number): void {
    getNativeAddon().groupMaskMul(this.glm.ctx, this.data, groupMask.data, numExperts, expertsPerGroup, nGroup, batch);
  }

  expertScale(weights: Tensor, indices: Tensor, expertId: number, topK: number, batch: number): void {
    getNativeAddon().expertScale(this.glm.ctx, this.data, weights.data, indices.data, expertId, topK, batch);
  }

  mulMatId(weights: Tensor[], expertIds: Tensor, topK: number, count: number, N: number, K: number, name: string): Tensor {
    const out = this.workspace.alloc([count, N], this.type);
    const ptrName = `__moe_ptrs.${name}`;
    let weightPtrs = this.workspace.tensors.get(ptrName);
    if (!weightPtrs) {
      weightPtrs = this.workspace.alloc([weights.length], "I64", ptrName);
      weightPtrs.writePointers(weights);
    }
    if (weights[0].type === "U8") {
      const scalePtrName = ptrName + "_weight_scale";
      const scale2PtrName = ptrName + "_weight_scale_2";
      let scalePtrs = this.workspace.tensors.get(scalePtrName);
      if (!scalePtrs) {
        const scaleTensors = weights.map(w => w.workspace.tensors.get(w.name! + "_weight_scale")!);
        scalePtrs = this.workspace.alloc([weights.length], "I64", scalePtrName);
        scalePtrs.writePointers(scaleTensors);
      }
      let scale2Ptrs = this.workspace.tensors.get(scale2PtrName);
      if (!scale2Ptrs) {
        const scale2Tensors = weights.map(w => w.workspace.tensors.get(w.name! + "_weight_scale_2")!);
        scale2Ptrs = this.workspace.alloc([weights.length], "I64", scale2PtrName);
        scale2Ptrs.writePointers(scale2Tensors);
      }
      if (count > MUL_MAT_ID_GROUPED_THRESHOLD) {
        const numExperts = weights.length;
        const wsSize = getNativeAddon().mmaMoeCoopWorkspaceSize(count, N, K, numExperts);
        using wsTensor = this.workspace.allocRaw(wsSize);
        getNativeAddon().nvfp4MulMatIdGroupedMmaCoop(this.glm.ctx, out.data, this.data, weightPtrs.data, scalePtrs.data, scale2Ptrs.data, expertIds.data, topK, count, N, K, numExperts, wsTensor.data);
      } else {
        getNativeAddon().nvfp4MulMatId(this.glm.ctx, out.data, this.data, weightPtrs.data, scalePtrs.data, scale2Ptrs.data, expertIds.data, topK, count, N, K);
      }
    } else if (count > MUL_MAT_ID_GROUPED_THRESHOLD) {
      const numExperts = weights.length;
      const wsSize = getNativeAddon().mmaMoeWorkspaceSize(count, N, K, numExperts);
      using wsTensor = this.workspace.allocRaw(wsSize);
      getNativeAddon().bf16MulMatIdGroupedMma(this.glm.ctx, out.data, this.data, weightPtrs.data, expertIds.data, topK, count, N, K, numExperts, wsTensor.data);
    } else {
      getNativeAddon().mulMatId(this.glm.ctx, out.data, this.data, weightPtrs.data, expertIds.data, topK, count, N, K);
    }
    return out;
  }

  private getMoeNvfp4Ptrs(weights: Tensor[], name: string): { weightPtrs: Tensor, scalePtrs: Tensor, scale2Ptrs: Tensor } {
    const ptrName = `__moe_ptrs.${name}`;
    let weightPtrs = this.workspace.tensors.get(ptrName);
    if (!weightPtrs) {
      weightPtrs = this.workspace.alloc([weights.length], "I64", ptrName);
      weightPtrs.writePointers(weights);
    }
    const scalePtrName = ptrName + "_weight_scale";
    let scalePtrs = this.workspace.tensors.get(scalePtrName);
    if (!scalePtrs) {
      const scaleTensors = weights.map(w => w.workspace.tensors.get(w.name! + "_weight_scale")!);
      scalePtrs = this.workspace.alloc([weights.length], "I64", scalePtrName);
      scalePtrs.writePointers(scaleTensors);
    }
    const scale2PtrName = ptrName + "_weight_scale_2";
    let scale2Ptrs = this.workspace.tensors.get(scale2PtrName);
    if (!scale2Ptrs) {
      const scale2Tensors = weights.map(w => w.workspace.tensors.get(w.name! + "_weight_scale_2")!);
      scale2Ptrs = this.workspace.alloc([weights.length], "I64", scale2PtrName);
      scale2Ptrs.writePointers(scale2Tensors);
    }
    return { weightPtrs, scalePtrs, scale2Ptrs };
  }

  swiGluMlpMoe(
    weights: { gate: Tensor[], up: Tensor[], down: Tensor[] },
    topkIndicesFlat: Tensor,
    topK: number, count: number,
    moeIntermediate: number, hs: number,
    pfx: string,
  ): Tensor {
    super.swiGluMlpMoe(weights, topkIndicesFlat, topK, count, moeIntermediate, hs, pfx);

    if (count <= MUL_MAT_ID_GROUPED_THRESHOLD || weights.gate[0].type !== "U8") {
      using gateOutStream = this.workspace.glm.withStream(() => this.mulMatId(weights.gate, topkIndicesFlat, topK, count, moeIntermediate, hs, `${pfx}.gate_proj`));
      using upOut = this.mulMatId(weights.up, topkIndicesFlat, topK, count, moeIntermediate, hs, `${pfx}.up_proj`);
      gateOutStream.streamWaitEvent();
      using gateOut = gateOutStream.result;
      using siluOut = gateOut.siluAndMul(upOut, moeIntermediate, count);
      return siluOut.mulMatId(weights.down, topkIndicesFlat, 1, count, hs, moeIntermediate, `${pfx}.down_proj`);
    }

    const numExperts = weights.gate.length;
    const ctx = this.glm.ctx;
    const gatePtrs = this.getMoeNvfp4Ptrs(weights.gate, `${pfx}.gate_proj`);
    const upPtrs = this.getMoeNvfp4Ptrs(weights.up, `${pfx}.up_proj`);

    const scatterWsSize = getNativeAddon().mmaMoeCoopScatterWorkspaceSize(count, hs, numExperts);
    using scatterWs = this.workspace.allocRaw(scatterWsSize);
    getNativeAddon().mmaMoeCoopScatter(ctx, this.data, topkIndicesFlat.data, topK, count, hs, numExperts, scatterWs.data);

    const gemmWsSize = getNativeAddon().mmaMoeCoopGemmWorkspaceSize(count, moeIntermediate);

    using gateStream = this.workspace.glm.withStream(() => {
      using gemmWs = this.workspace.allocRaw(gemmWsSize);
      const gateOut = this.workspace.alloc([count, moeIntermediate], this.type);
      getNativeAddon().mmaMoeCoopGemm(ctx, gatePtrs.weightPtrs.data, gatePtrs.scalePtrs.data, gatePtrs.scale2Ptrs.data,
                                       numExperts, moeIntermediate, hs, count, scatterWs.data, gemmWs.data, gateOut.data);
      return gateOut;
    });

    using upGemmWs = this.workspace.allocRaw(gemmWsSize);
    using upOut = this.workspace.alloc([count, moeIntermediate], this.type);
    getNativeAddon().mmaMoeCoopGemm(ctx, upPtrs.weightPtrs.data, upPtrs.scalePtrs.data, upPtrs.scale2Ptrs.data,
                                     numExperts, moeIntermediate, hs, count, scatterWs.data, upGemmWs.data, upOut.data);

    gateStream.streamWaitEvent();
    using gateOut = gateStream.result;
    using siluOut = gateOut.siluAndMul(upOut, moeIntermediate, count);
    return siluOut.mulMatId(weights.down, topkIndicesFlat, 1, count, hs, moeIntermediate, `${pfx}.down_proj`);
  }

  scatterAddRows(scales: Tensor, topK: number, dim: number, numRows: number): Tensor {
    const out = this.workspace.alloc([numRows, dim], this.type);
    getNativeAddon().scatterAddRows(this.glm.ctx, out.data, this.data, scales.data, topK, dim, numRows, 0);
    return out;
  }

  sampleBatch(outTokens: Tensor, topkVals: Tensor, topkIdxs: Tensor, workspace: Tensor, logits: Tensor, penaltyTokens: Tensor, penaltyCount: Tensor, maxWindow: number, vocabSize: number, batchSize: number, temperatures: Tensor, repPenalties: Tensor, presPenalties: Tensor, topKs: Tensor, topPs: Tensor, stepCounter: Tensor, maxEffectiveK: number): void {
    getNativeAddon().sampleBatch(this.glm.ctx, outTokens.data, topkVals.data, topkIdxs.data, workspace.data, logits.data, penaltyTokens.data, penaltyCount.data, maxWindow, vocabSize, batchSize, temperatures.data, repPenalties.data, presPenalties.data, topKs.data, topPs.data, stepCounter.data, maxEffectiveK);
  }
}

export class GlmOps implements DeviceOps {
  readonly worldSize = 1;
  ctx: number;
  device: number;
  allocator: Allocator;
  capturing = false;

  constructor(deviceId: number = 0, libPath?: string, arenaGb?: number) {
    if (libPath) {
      nativeAddon = require(libPath) as NativeAddon;
    }
    const native = getNativeAddon();
    this.ctx = native.init(deviceId);
    if (!this.ctx) {
      throw new Error(`glm_init failed on device ${deviceId}`);
    }
    this.device = deviceId;

    if (arenaGb) {
      const size = arenaGb * 1024 * 1024 * 1024;
      const base = getNativeAddon().alloc(this.ctx, size);
      this.allocator = new ArenaAllocator(base, size);
    }
    else {
      this.allocator = {
        alloc: (size: number) => {
          if (this.capturing) {
            console.warn("Warning: allocating during capture will fail.");
          }
          return getNativeAddon().alloc(this.ctx, size)
        },
        free: (ptr: number) => getNativeAddon().freeBuf(this.ctx, ptr),
      }
    }
  }

  free(): void {
    getNativeAddon().free(this.ctx);
  }

  allocPinned(bytes: number): number {
    const p = getNativeAddon().allocPinned(bytes);
    if (!p) throw new Error(`allocPinned failed for size ${bytes}`);
    return p;
  }

  newTensor(workspace: WorkspaceBase, shape: number[], type: string, pinned: boolean, name?: string, _parallelism?: TensorParallelism): GlmTensor {
    const size = Tensor.byteCount(shape, type);
    const data = pinned ? this.allocPinned(size) : this.allocator.alloc(size);
    return new GlmTensor(workspace, this, data, size, shape, type, name, pinned, undefined);
  }

  wrapTensor(workspace: WorkspaceBase, data: number, allocSize: number, shape: number[], type: string, pinned: boolean, view: GlmTensor | undefined): Tensor {
    return new GlmTensor(workspace, this, data, allocSize, shape, type, undefined, pinned, view);
  }

  synchronize(): void {
    getNativeAddon().synchronize(this.ctx);
  }

  synchronizeStream(streamIdx: number): void {
    getNativeAddon().synchronizeStream(this.ctx, streamIdx);
  }

  streamTensors = new Map<number, Set<GlmTensor>>();
  setStream(streamIdx: number): void {
    getNativeAddon().setStream(this.ctx, streamIdx);
    this.currentStream = streamIdx;
    if (streamIdx) {
      if (!this.streamTensors.has(streamIdx))
        this.streamTensors.set(streamIdx, new Set());
    }
  }

  currentStream = 0;
  availableStreams = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
  disposeStream(stream: number) {
    if (this.availableStreams.includes(stream))
      throw new Error(`Stream ${stream} already disposed`);
    this.availableStreams.push(stream);
    const tensors = this.streamTensors.get(stream);
    this.streamTensors.delete(stream);
    for (const tensor of tensors!) {
      tensor[Symbol.dispose]();
    }
  }

  withStream<T>(fn: () => T) {
    const stream = this.availableStreams.pop();
    if (stream === undefined)
      throw new Error("No available streams");
    const currentStream = this.currentStream;
    // Record event on stream 0 so the alternate stream can wait for
    // all prior work (e.g. rmsnorm output that K/V will read).
    getNativeAddon().eventRecord(this.ctx, currentStream, currentStream);
    if (this.streamTensors.has(stream)) {
      throw new Error(`Stream ${stream} already in use`);
    }
    this.setStream(stream);
    getNativeAddon().streamWaitEvent(this.ctx, stream, currentStream);
    const result = fn();
    // Record event on the alternate stream so stream 0 can wait at dispose.
    getNativeAddon().eventRecord(this.ctx, stream, stream);
    this.setStream(currentStream);
    return {
      [Symbol.dispose]: () => {
        this.disposeStream(stream);
      },
      result,
      synchronize: () => {
        getNativeAddon().synchronizeStream(this.ctx, stream);
      },
      streamWaitEvent: () => {
        getNativeAddon().streamWaitEvent(this.ctx, this.currentStream, stream);
      }
    }
  }

  eventRecord(eventIdx: number, streamIdx: number): void {
    getNativeAddon().eventRecord(this.ctx, eventIdx, streamIdx);
  }

  streamWaitEvent(streamIdx: number, eventIdx: number): void {
    getNativeAddon().streamWaitEvent(this.ctx, streamIdx, eventIdx);
  }

  kvCacheWrite(srcK: Tensor, srcV: Tensor, dstK: Tensor, dstV: Tensor, slotMapping: Tensor, batchSize: number, nKv: number, hd: number, srcKTokenStride: number, srcKHeadStride: number, srcVTokenStride: number, srcVHeadStride: number): void {
    const pageSize = dstK.shape[1] / (nKv * hd);
    getNativeAddon().kvCacheWrite(this.ctx, ptr(srcK), ptr(srcV), ptr(dstK), ptr(dstV), ptr(slotMapping), batchSize, nKv, hd, pageSize, srcKTokenStride, srcKHeadStride, srcVTokenStride, srcVHeadStride);
  }

  gatherPages(srcData: Tensor, pageIndices: Tensor, pageIndptrD: Tensor, lastPageLen: Tensor, batchSize: number, paddedKvLen: number, _kvTokenIndptrD: Tensor, _contextParallel: boolean): Tensor {
    const pageSize = srcData.shape[1];
    const D = srcData.shape[2];
    const out = pageIndptrD.workspace.alloc([paddedKvLen / pageSize, pageSize, D], srcData.type);
    const elemBytes = srcData.type === "U8" ? 1 : 2;
    const maxPages = srcData.shape[0];
    getNativeAddon().gatherPages(this.ctx, out.data, srcData.data, pageIndices.data, pageIndptrD.data, lastPageLen.data, maxPages, batchSize, pageSize, D * elemBytes);
    return out;
  }

  indexerScore(out: Tensor, q: Tensor, kData: Tensor, weights: Tensor, pageIndices: Tensor, pageIndptr: Tensor, lastPageLen: Tensor, qoIndptr: Tensor, scale: number, totalQ: number, idxNHeads: number, idxHeadDim: number, pageSize: number, maxKvLen: number, causal: boolean): void {
    getNativeAddon().indexerScore(this.ctx, out.data, q.data, kData.data, weights.data, pageIndices.data, pageIndptr.data, lastPageLen.data, qoIndptr.data, scale, totalQ, idxNHeads, idxHeadDim, pageSize, maxKvLen, causal ? 1 : 0);
  }

  // Indexer top-k selection -> physical KV slots. Runs the full pipeline
  // (score+topk, then map token positions to slots) on this device. Decode uses
  // the multi-block score+histogram kernel (v2): totalQ is small (batch), so
  // per-query scratch is tiny. Prefill uses the fused two-level kernel: multi-
  // block grid (numSplits × totalQ) parallelizes across both KV and query dims,
  // with a coarse/fine histogram top-K that avoids materializing scores.
  // Scratch: ~17 MB (coarseHist + fineHist + meta) vs 2.1 GB for v2 on prefill.
  // Writes the compacted valid-slot count per query into `topkLength` (a stable
  // caller buffer), which feeds the sparse kernel's topk_length so it only walks
  // ceil(count/BI) candidate tiles instead of the full topk.
  indexerTopkSlots(idxQ: Tensor, kData: Tensor, weights: Tensor, pageIndices: Tensor, indptr: Tensor, lastPageLen: Tensor, qoIndptr: Tensor, batchIndices: Tensor, topkLength: Tensor, scale: number, topk: number, decode: boolean, _contextParallel?: boolean, cpWorldSize: number = 1, cpRank: number = 0, globalLastPageLen?: Tensor, customMask?: Tensor, maskIndptr?: Tensor, maskKvLen?: Tensor, qGlobalStart: number = 0, kvTokenIndptrD?: Tensor): Tensor {
    const totalQ = idxQ.shape[0];
    const idxNHeads = idxQ.shape[1];
    const idxHeadDim = idxQ.shape[2];
    const pageSize = kData.shape[1];
    const maxKv = kData.shape[0] * kData.shape[1];
    const scoreLastPageLen = globalLastPageLen ?? lastPageLen;
    // Kernel selection is a q-len heuristic, not the caller's decode flag: the
    // direct v2 path (per-position score + single histogram) wins for small Q but
    // its 256 KB/query histogram doesn't scale, so large Q uses the two-level
    // tensor-core prefill path. Both paths support custom masks and query-sharding
    // (qGlobalStart). `decode` now only supplies causal semantics — a lone decode
    // query attends to all past KV (causal limit kvLen-1), i.e. causal for a
    // single-query row, so pass causal=0 there and causal=1 for a multi-query
    // (prefill / MTP tree-verify) row routed here at small Q.
    const useDirect = totalQ <= INDEXER_DIRECT_DISPATCH_MAX;
    using topkIdx = useDirect
      ? this.indexerScoreTopkV2(idxQ, kData, weights, pageIndices, indptr, scoreLastPageLen, qoIndptr, scale, totalQ, idxNHeads, idxHeadDim, pageSize, topk, maxKv, decode ? 0 : 1, customMask, maskIndptr, maskKvLen, qGlobalStart)
      : this.indexerScoreTopkPrefill(idxQ, kData, weights, pageIndices, indptr, scoreLastPageLen, qoIndptr, scale, totalQ, idxNHeads, idxHeadDim, pageSize, topk, maxKv, customMask, maskIndptr, maskKvLen, qGlobalStart);
    // const slots = idxQ.workspace.alloc([totalQ, topk], "I32");
    const slots = idxQ.workspace.ensureAlloc([maxKv, topk], "I32", "idxslots-shared").narrow(0, totalQ);
    getNativeAddon().topkToSlots(this.ctx, ptr(slots), ptr(topkLength), ptr(topkIdx), ptr(pageIndices), ptr(indptr), ptr(lastPageLen), ptr(batchIndices), totalQ, topk, pageSize, cpWorldSize, cpRank, kvTokenIndptrD ? ptr(kvTokenIndptrD) : 0);
    return slots;
  }

  private indexerScoreTopkPrefill(q: Tensor, kData: Tensor, weights: Tensor, pageIndices: Tensor, pageIndptr: Tensor, lastPageLen: Tensor, qoIndptr: Tensor, scale: number, totalQ: number, idxNHeads: number, idxHeadDim: number, pageSize: number, topk: number, maxKv: number, customMask?: Tensor, maskIndptr?: Tensor, maskKvLen?: Tensor, qGlobalStart: number = 0): Tensor {
    const numSplits = Math.min(256, Math.max(1, Math.ceil(maxKv / 256)));
    const indices = q.workspace.alloc([totalQ, topk], "I32");
    using scores = q.workspace.alloc([totalQ, maxKv], "BF16");
    using rowLen = q.workspace.alloc([totalQ], "I32");
    using coarseHist = q.workspace.alloc([totalQ, 1024], "I32");
    using fineHist = q.workspace.alloc([totalQ, 64], "I32");
    using meta = q.workspace.alloc([totalQ, 4], "I32");
    getNativeAddon().indexerScoreTopkPrefill(this.ctx, indices.data, q.data, kData.data, weights.data, pageIndices.data, pageIndptr.data, lastPageLen.data, qoIndptr.data, scale, totalQ, idxNHeads, idxHeadDim, pageSize, topk, 1 /*causal*/, customMask ? customMask.data : 0, maskIndptr ? maskIndptr.data : 0, maskKvLen ? maskKvLen.data : 0, scores.data, rowLen.data, maxKv, coarseHist.data, fineHist.data, meta.data, numSplits, qGlobalStart);
    return indices;
  }

  private indexerScoreTopkV2(q: Tensor, kData: Tensor, weights: Tensor, pageIndices: Tensor, pageIndptr: Tensor, lastPageLen: Tensor, qoIndptr: Tensor, scale: number, totalQ: number, idxNHeads: number, idxHeadDim: number, pageSize: number, topk: number, maxKv: number, causal = 0, customMask?: Tensor, maskIndptr?: Tensor, maskKvLen?: Tensor, qGlobalStart = 0): Tensor {
    const numSplits = Math.min(256, Math.max(1, Math.ceil(maxKv / 256)));
    const indices = q.workspace.alloc([totalQ, topk], "I32");
    using scores = q.workspace.alloc([totalQ, maxKv], "BF16");
    using rowLen = q.workspace.alloc([totalQ], "I32");
    using hist = q.workspace.alloc([totalQ, 65536], "I32");
    using meta = q.workspace.alloc([totalQ, 4], "I32");
    getNativeAddon().indexerScoreTopkV2(this.ctx, indices.data, q.data, kData.data, weights.data, pageIndices.data, pageIndptr.data, lastPageLen.data, qoIndptr.data, scale, totalQ, idxNHeads, idxHeadDim, pageSize, topk, causal, customMask ? customMask.data : 0, maskIndptr ? maskIndptr.data : 0, maskKvLen ? maskKvLen.data : 0, scores.data, rowLen.data, hist.data, meta.data, maxKv, numSplits, qGlobalStart);
    return indices;
  }

  positionStep(positionIds: Tensor, lastPageLen: Tensor, slotMapping: Tensor, indptr: Tensor, indices: Tensor, pageSize: number, batchSize: number, steps = 1): void {
    getNativeAddon().positionStep(this.ctx, ptr(positionIds), ptr(lastPageLen), ptr(slotMapping), ptr(indptr), ptr(indices), pageSize, batchSize, steps);
  }

  mlaPositionStep(positionIds: Tensor, lastPageLen: Tensor, indptr: Tensor, pageSize: number, batchSize: number, _contextParallel?: boolean, cpWorldSize = 1, cpRank = 0, steps = 1, globalLastPageLen?: Tensor): void {
    getNativeAddon().mlaPositionStep(this.ctx, ptr(positionIds), ptr(lastPageLen), ptr(indptr), pageSize, batchSize, cpWorldSize, cpRank, steps, globalLastPageLen ? ptr(globalLastPageLen) : 0);
  }

  hostPointerToBuffer(ptr: number, size: number): Buffer {
    return getNativeAddon().hostPointerToBuffer(ptr, size);
  }

  batchDecodePlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, indptrH: Tensor, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, enableCudaGraph: boolean): void {
    getNativeAddon().batchDecodePlan(this.ctx, ptr(floatWs), floatWsSize, ptr(intWs), ptr(pinnedIntWs), intWsSize, ptr(planInfo), ptr(indptrH), batchSize, numQoHeads, numKvHeads, headDim, pageSize, enableCudaGraph);
  }

  batchDecodeRun(state: ExecutionState, q: Tensor, o: Tensor, kData: Tensor, vData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, numQoHeads: number, numKvHeads: number, headDim: number, smScale: number): void {
    const pageSize = kData.shape[1] / (numKvHeads * headDim);
    getNativeAddon().batchDecodeRun(this.ctx, ptr(q), ptr(o), ptr(kData), ptr(vData), ptr(indices), ptr(indptrD), ptr(lastPageLen), ptr(floatWs), ptr(intWs), ptr(planInfo), state.batchSize, numQoHeads, numKvHeads, headDim, pageSize, smScale);
  }

  batchPrefillPagedPlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, qoIndptrH: Tensor, pagedKvIndptrH: Tensor, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, maskMode: MaskMode): void {
    getNativeAddon().batchPrefillPagedPlan(this.ctx, ptr(floatWs), floatWsSize, ptr(intWs), ptr(pinnedIntWs), intWsSize, ptr(planInfo), ptr(qoIndptrH), ptr(pagedKvIndptrH), totalQoRows, batchSize, numQoHeads, numKvHeads, headDim, pageSize, maskMode);
  }

  batchPrefillPagedRun(state: ExecutionState, q: Tensor, o: Tensor, kData: Tensor, vData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, floatWs: Tensor, intWs: Tensor, qIndptrD: Tensor, planInfo: Tensor, numQoHeads: number, numKvHeads: number, headDim: number, qStrideN: number, qStrideH: number, maskMode: MaskMode, smScale: number): void {
    const pageSize = kData.shape[1] / (numKvHeads * headDim);
    getNativeAddon().batchPrefillPagedRun(this.ctx, ptr(q), ptr(o), ptr(kData), ptr(vData), ptr(indices), ptr(indptrD), ptr(lastPageLen), ptr(floatWs), ptr(intWs), ptr(qIndptrD), ptr(planInfo), state.totalTokens, state.batchSize, numQoHeads, numKvHeads, headDim, pageSize, qStrideN, qStrideH, maskMode, smScale);
  }

  batchPrefillRaggedPlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, qoIndptrH: Tensor, kvIndptrH: Tensor, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, maskMode: MaskMode): void {
    getNativeAddon().batchPrefillRaggedPlan(this.ctx, ptr(floatWs), floatWsSize, ptr(intWs), ptr(pinnedIntWs), intWsSize, ptr(planInfo), ptr(qoIndptrH), ptr(kvIndptrH), totalQoRows, batchSize, numQoHeads, numKvHeads, headDim, maskMode);
  }

  batchPrefillRaggedRun(state: ExecutionState, q: Tensor, k: Tensor, v: Tensor, o: Tensor, floatWs: Tensor, intWs: Tensor, qIndptrD: Tensor, kvIndptrD: Tensor, planInfo: Tensor, numQoHeads: number, numKvHeads: number, headDim: number, qStrideN: number, qStrideH: number, kvStrideN: number, kvStrideH: number, vStrideN: number, vStrideH: number, maskMode: MaskMode, smScale: number): void {
    getNativeAddon().batchPrefillRaggedRun(this.ctx, ptr(q), ptr(k), ptr(v), ptr(o), ptr(floatWs), ptr(intWs), ptr(qIndptrD), ptr(kvIndptrD), ptr(planInfo), state.totalTokens, state.batchSize, numQoHeads, numKvHeads, headDim, qStrideN, qStrideH, kvStrideN, kvStrideH, vStrideN, vStrideH, maskMode, smScale);
  }

  graphBeginCapture(): void {
    getNativeAddon().graphBeginCapture(this.ctx);
    this.capturing = true;
  }

  graphEndCapture(): number {
    const graph = getNativeAddon().graphEndCapture(this.ctx);
    if (!graph) throw new Error("CUDA graph capture failed");
    this.capturing = false;
    return graph;
  }

  graphInstantiate(graph: number): number {
    const exec = getNativeAddon().graphInstantiate(graph);
    if (!exec) throw new Error("CUDA graph instantiation failed");
    return exec;
  }

  graphLaunch(graphExec: number): void {
    getNativeAddon().graphLaunch(graphExec, this.ctx);
  }

  graphDestroy(graph: number): void {
    getNativeAddon().graphDestroy(graph);
  }

  graphExecDestroy(graphExec: number): void {
    getNativeAddon().graphExecDestroy(graphExec);
  }

  mlaPrefillPlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, qoIndptrH: Tensor, kvIndptrH: Tensor, kvLenH: Tensor, lastPageLenH: Tensor, batchSize: number, numHeads: number, headDimO: number, causal: boolean, pageSize: number, seqKvLens: number[], contextParallel?: boolean, cpWorldSize: number = 0, cpRank: number = 0): void {
    getNativeAddon().mlaPrefillPlan(this.ctx, ptr(floatWs), floatWsSize, ptr(intWs), ptr(pinnedIntWs), intWsSize, ptr(planInfo), ptr(qoIndptrH), ptr(kvIndptrH), ptr(kvLenH), batchSize, numHeads, headDimO, causal, cpWorldSize, cpRank);
  }

  mlaPrefillRun(state: ExecutionState, qNope: Tensor, qPe: Tensor, ckvData: Tensor, kpeData: Tensor, kvIndices: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, smScale: number, maskMode: MaskMode, cpWorldSize: number = 0, cpRank: number = 0, customMask?: Tensor, maskIndptr?: Tensor, maskKvLen?: Tensor): { o: Tensor, lse: Tensor } {
    const numHeads = qNope.shape[1];
    const headDimCkv = ckvData.shape[2];
    const headDimKpe = kpeData.shape[2];
    const pageSize = ckvData.shape[1];
    const ckvStridePage = pageSize * headDimCkv;
    const ckvStrideN = headDimCkv;
    const kpeStridePage = kpeData.shape[1] * headDimKpe;
    const kpeStrideN = headDimKpe;
    const qNopeStrideN = qNope.shape[1] * qNope.shape[2];
    const qNopeStrideH = qNope.shape[2];
    const qPeStrideN = qPe.shape[1] * qPe.shape[2];
    const qPeStrideH = qPe.shape[2];
    const totalTokens = state.totalTokens;
    const oStrideN = headDimCkv;
    const oStrideH = totalTokens * headDimCkv;
    const o = qNope.workspace.alloc([1, numHeads, totalTokens, headDimCkv], qNope.type);
    const lse = qNope.workspace.alloc([totalTokens, numHeads], "F32");
    getNativeAddon().mlaPrefillRun(this.ctx, ptr(qNope), ptr(qPe), ptr(ckvData), ptr(kpeData), ptr(kvIndices), ptr(o), ptr(floatWs), ptr(intWs), ptr(planInfo), numHeads, pageSize, maskMode, smScale, qNopeStrideN, qNopeStrideH, qPeStrideN, qPeStrideH, ckvStridePage, ckvStrideN, kpeStridePage, kpeStrideN, oStrideN, oStrideH, headDimCkv, headDimKpe, ptr(lse), cpWorldSize, cpRank, customMask ? ptr(customMask) : 0, maskIndptr ? ptr(maskIndptr) : 0, maskKvLen ? ptr(maskKvLen) : 0);
    return { o, lse };
  }

  mlaDecodePlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, indptrH: Tensor, lastPageLenH: Tensor, batchSize: number, numQoHeads: number, pageSize: number, enableCudaGraph: boolean, headDimCkv: number, headDimKpe: number, contextParallel?: boolean, cpWorldSize?: number, cpRank?: number, _seqKvLens?: number[]): void {
    getNativeAddon().mlaDecodePlan(this.ctx, ptr(floatWs), floatWsSize, ptr(intWs), ptr(pinnedIntWs), intWsSize, ptr(planInfo), ptr(indptrH), batchSize, numQoHeads, pageSize, enableCudaGraph, headDimCkv, headDimKpe);
  }

  mlaDecodeRun(state: ExecutionState, qNope: Tensor, qPe: Tensor, ckvData: Tensor, kpeData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, smScale: number): { o: Tensor, lse: Tensor } {
    const numQoHeads = qNope.shape[1];
    const headDimCkv = ckvData.shape[2];
    const headDimKpe = kpeData.shape[2];
    const pageSize = ckvData.shape[1];
    const o = qNope.workspace.alloc([state.batchSize, numQoHeads, 1, headDimCkv], qNope.type);
    const lse = qNope.workspace.alloc([state.batchSize, numQoHeads], "F32");
    getNativeAddon().mlaDecodeRun(this.ctx, ptr(qNope), ptr(qPe), ptr(ckvData), ptr(kpeData), ptr(indices), ptr(indptrD), ptr(lastPageLen), ptr(o), ptr(floatWs), ptr(intWs), ptr(planInfo), state.batchSize, numQoHeads, pageSize, smScale, headDimCkv, headDimKpe, ptr(lse));
    return { o, lse };
  }

  mlaKvCacheAppend(ckvData: Tensor, kpeData: Tensor | null, indices: Tensor, indptr: Tensor, lastPageLen: Tensor, appendCkv: Tensor, appendKpe: Tensor | null, batchIndices: Tensor, positions: Tensor, nnz: number, headDimCkv: number, headDimKpe: number, appendCkvStrideN: number, appendKpeStrideN: number, pageSize: number = ckvData.shape[1], cpWorldSize: number = 0, cpRank: number = 0): void {
    getNativeAddon().mlaKvCacheAppend(this.ctx, ptr(ckvData), kpeData ? ptr(kpeData) : 0, ptr(indices), ptr(indptr), ptr(lastPageLen), ptr(appendCkv), appendKpe ? ptr(appendKpe) : 0, ptr(batchIndices), ptr(positions), nnz, pageSize, headDimCkv, headDimKpe, appendCkvStrideN, appendKpeStrideN, cpWorldSize, cpRank);
  }

  concatAndCacheDsMla(kvCache: Tensor, appendCkv: Tensor, appendKpe: Tensor, indices: Tensor, indptr: Tensor, batchIndices: Tensor, positions: Tensor, nnz: number, kvLoraRank: number, peDim: number, appendCkvStrideN: number, appendKpeStrideN: number, pageSize: number = kvCache.shape[1], cpWorldSize: number = 0, cpRank: number = 0): void {
    getNativeAddon().concatAndCacheDsMla(this.ctx, ptr(kvCache), ptr(appendCkv), ptr(appendKpe), ptr(indices), ptr(indptr), ptr(batchIndices), ptr(positions), nnz, pageSize, kvLoraRank, peDim, appendCkvStrideN, appendKpeStrideN, cpWorldSize, cpRank);
  }

  gdnRecurrentStep(state: ExecutionState, output: Tensor, recurrentState: Tensor, qkv: Tensor, aRaw: Tensor, bRaw: Tensor, aLog: Tensor, dtBias: Tensor, numHeads: number, dK: number, dV: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void {
    if (recurrentState.type !== "F32") throw new Error(`gdnRecurrentStep: state must be F32, got ${recurrentState.type}`);
    if (aLog.type !== "F32") throw new Error(`gdnRecurrentStep: aLog must be F32, got ${aLog.type}`);
    if (dtBias.type !== "F32") throw new Error(`gdnRecurrentStep: dtBias must be F32, got ${dtBias.type}`);
    getNativeAddon().gdnRecurrentStep(this.ctx, output.data, recurrentState.data, qkv.data, aRaw.data, bRaw.data, aLog.data, dtBias.data, numHeads, dK, dV, state.batchSize, stateStride, qkvChStride, qkvSeqStride);
  }

  gdnPrefill(state: ExecutionState, output: Tensor, recurrentState: Tensor, qkv: Tensor, aRaw: Tensor, bRaw: Tensor, aLog: Tensor, dtBias: Tensor, cuSeqlens: Tensor, numHeads: number, dK: number, dV: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void {
    if (recurrentState.type !== "F32") throw new Error(`gdnPrefill: state must be F32, got ${recurrentState.type}`);
    if (aLog.type !== "F32") throw new Error(`gdnPrefill: aLog must be F32, got ${aLog.type}`);
    if (dtBias.type !== "F32") throw new Error(`gdnPrefill: dtBias must be F32, got ${dtBias.type}`);
    if (cuSeqlens.type !== "I32") throw new Error(`gdnPrefill: cuSeqlens must be I32, got ${cuSeqlens.type}`);
    getNativeAddon().gdnPrefill(this.ctx, output.data, recurrentState.data, qkv.data, aRaw.data, bRaw.data, aLog.data, dtBias.data, cuSeqlens.data, state.totalTokens, numHeads, dK, dV, state.batchSize, stateStride, qkvChStride, qkvSeqStride);
  }

  sparseMlaPrefill(state: ExecutionState, qAbsorbed: Tensor, qPe: Tensor, kvCache: Tensor, indices: Tensor, topk: number, smScale: number, topkLength: Tensor, _pageIndptrD: Tensor, _lastPageLen: Tensor, _kvTokenIndptrD: Tensor): { o: Tensor, lse: Tensor } {
    const numTokens = state.totalTokens;
    const numHeads = qAbsorbed.shape[1];
    const headDim = qAbsorbed.shape[2];
    const elemBytes = SafeTensorFile.dtypeBytes(kvCache.type);
    const pageBlockSize = kvCache.shape[1];
    const effectiveStrideKvBlock = pageBlockSize * kvCache.shape[2] * elemBytes;
    if (pageBlockSize !== 64) throw new Error(`sparseMlaPrefill: SM120 kernel requires pageBlockSize=64, got ${pageBlockSize} ${kvCache.shape}`);
    using q = qAbsorbed.cat([qPe], 2);
    const o = q.workspace.alloc([numTokens, numHeads, headDim], "BF16");
    const lse = q.workspace.alloc([numTokens, numHeads], "F32");
    // Small query counts (e.g. MTP tree verify) starve the prefill kernel: its
    // grid is only numTokens × ceil(NUM_HEADS/HPB) CTAs, leaving the GPU idle.
    // Route to the split-K decode kernel — same mask-free slot attention, but
    // numTokens × ceil(topk/64) CTAs — which fills the SMs. Correctness is
    // identical (causality lives in the slots, not the kernel).
    if (numTokens <= SPARSE_MLA_DECODE_DISPATCH_MAX) {
      const numSplits = Math.ceil(topk / 64);
      using midOut = q.workspace.alloc([numTokens, numHeads, numSplits, headDim], "BF16");
      using midLse = q.workspace.alloc([numTokens, numHeads, numSplits], "F32");
      getNativeAddon().sparseMlaDecode(this.ctx, ptr(q), ptr(kvCache), ptr(indices), ptr(midOut), ptr(midLse), ptr(o), ptr(lse), numTokens, numHeads, topk, numSplits, smScale, effectiveStrideKvBlock, 0, ptr(topkLength));
      return { o, lse };
    }
    getNativeAddon().sparseMlaPrefill(this.ctx, ptr(q), ptr(kvCache), ptr(indices), ptr(o), ptr(lse), numTokens, numHeads, topk, pageBlockSize, smScale, effectiveStrideKvBlock, ptr(topkLength));
    return { o, lse };
  }

  sparseMlaDecode(state: ExecutionState, qAbsorbed: Tensor, qPe: Tensor, kvCache: Tensor, indices: Tensor, topk: number, numSplits: number, smScale: number, chunksPerBlock: number, topkLength?: Tensor): { o: Tensor, lse: Tensor } {
    const numTokens = state.batchSize;
    const numHeads = qAbsorbed.shape[1];
    const headDim = qAbsorbed.shape[2];
    const elemBytes = SafeTensorFile.dtypeBytes(kvCache.type);
    const pageBlockSize = kvCache.shape[1];
    const effectiveStrideKvBlock = pageBlockSize * kvCache.shape[2] * elemBytes;
    if (pageBlockSize !== 64) throw new Error(`sparseMlaDecode: SM120 kernel requires pageBlockSize=64, got ${pageBlockSize}`);
    using q = qAbsorbed.cat([qPe], 2);
    const o = q.workspace.alloc([numTokens, numHeads, headDim], "BF16");
    const lse = q.workspace.alloc([numTokens, numHeads], "F32");
    using midOut = q.workspace.alloc([numTokens, numHeads, numSplits, headDim], "BF16");
    using midLse = q.workspace.alloc([numTokens, numHeads, numSplits], "F32");
    getNativeAddon().sparseMlaDecode(this.ctx, ptr(q), ptr(kvCache), ptr(indices), ptr(midOut), ptr(midLse), ptr(o), ptr(lse), numTokens, numHeads, topk, numSplits, smScale, effectiveStrideKvBlock, chunksPerBlock, topkLength ? ptr(topkLength) : 0);
    return { o, lse };
  }

  cpMergeTree(vPtrs: number[], lsePtrs: number[], numShards: number, outputV: Tensor, outputLse: Tensor | null, numel: number, batchSize: number, numHeads: number, vHeadDim: number, shardNHeads?: number, headOffset?: number, inputNHeads?: number): void {
    const snh = shardNHeads ?? numHeads;
    const ho = headOffset ?? 0;
    const inh = inputNHeads ?? numHeads;
    const v = new Array<number>(8).fill(0);
    for (let i = 0; i < numShards; i++) v[i] = vPtrs[i];
    const lse = new Array<number>(8).fill(0);
    for (let i = 0; i < numShards; i++) lse[i] = lsePtrs[i];
    getNativeAddon().cpMergeTree(
      this.ctx,
      v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
      lse[0], lse[1], lse[2], lse[3], lse[4], lse[5], lse[6], lse[7],
      numShards, ptr(outputV), outputLse ? ptr(outputLse) : 0, numel, batchSize, numHeads, vHeadDim,
      snh, ho, inh,
    );
  }

  cpCorrectAttnOut(vOut: Tensor, lses: Tensor, globalLse: Tensor | null, batchSize: number, numHeads: number, vHeadDim: number, worldSize: number, rank: number): void {
    getNativeAddon().cpCorrectAttnOut(
      this.ctx, ptr(vOut), ptr(lses), globalLse ? ptr(globalLse) : 0,
      batchSize, numHeads, vHeadDim, worldSize, rank,
    );
  }

  ncclReduceScatter(comm: number, sendbuff: Tensor, recvbuff: Tensor, recvcount: number, datatype: number, op: number): void {
    getNativeAddon().ncclReduceScatter(comm, this.ctx, ptr(sendbuff), ptr(recvbuff), recvcount, datatype, op);
  }

  p2pBarrier(instance: number, peerRank: number = -1): void {
    getNativeAddon().p2pBarrier(this.ctx, instance, peerRank);
  }
}

export function f32ToBf16Bytes(arr: Float32Array): Buffer {
  const u32 = new Uint32Array(arr.buffer);
  const u16 = new Uint16Array(u32.length);
  for (let i = 0; i < u32.length; i++) {
    u16[i] = u32[i] >>> 16;
  }
  return Buffer.from(u16.buffer);
}

export function bf16BytesToF32(buf: Buffer): Float32Array {
  const u16 = new Uint16Array(buf.buffer, buf.byteOffset, buf.length / 2);
  const u32 = new Uint32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    u32[i] = u16[i] << 16;
  }
  return new Float32Array(u32.buffer);
}

export const BF16 = 2;
export const I32 = 4;
export const F32 = 4;

const MEMCPY_H2H = 0;
const MEMCPY_H2D = 1;
const MEMCPY_D2H = 2;
const MEMCPY_D2D = 3;

function memcpyKindToNative(kind: MemcpyKind): number {
  switch (kind) {
    case MemcpyKind.HostToHost: return MEMCPY_H2H;
    case MemcpyKind.HostToDevice: return MEMCPY_H2D;
    case MemcpyKind.DeviceToHost: return MEMCPY_D2H;
    case MemcpyKind.DeviceToDevice: return MEMCPY_D2D;
    case MemcpyKind.Default: return 4;
  }
}

export const NCCL_UNIQUE_ID_BYTES = 128;
export const NCCL_INT8 = 0;
export const NCCL_UINT8 = 1;
export const NCCL_INT32 = 2;
export const NCCL_UINT32 = 3;
export const NCCL_INT64 = 4;
export const NCCL_UINT64 = 5;
export const NCCL_FLOAT16 = 6;
export const NCCL_FLOAT32 = 7;
export const NCCL_FLOAT64 = 8;
export const NCCL_BFLOAT16 = 9;
export const NCCL_SUM = 0;
export const NCCL_PROD = 1;
export const NCCL_MAX = 2;
export const NCCL_MIN = 3;
