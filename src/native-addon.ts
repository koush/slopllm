import fs from "node:fs";
import path from "node:path";

let nativeAddon: NativeAddon | null = null;

function findProjectRoot(dir: string): string {
  let d = dir;
  while (d !== path.dirname(d)) {
    if (fs.existsSync(path.join(d, "package.json"))) return d;
    d = path.dirname(d);
  }
  return dir;
}

export function getNativeAddon(libPath?: string): NativeAddon {
  if (libPath) {
    nativeAddon = require(libPath) as NativeAddon;
  } else if (!nativeAddon) {
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

export interface NativeAddon {
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
  linear(ctx: number, out: number, input: number, weight: number, batch: number, n: number, k: number, workspace: number, workspaceSize: number): void;
  fill(ctx: number, out: number, value: number, n: number): void;
  indexerScore(ctx: number, out: number, q: number, kData: number, weights: number, pageIndices: number, pageIndptr: number, lastPageLen: number, qoIndptr: number, scale: number, totalQ: number, idxNHeads: number, idxHeadDim: number, pageSize: number, maxKvLen: number, causal: number, kvTokenIndptr?: number): void;
  indexerScoreTopkPrefill(ctx: number, outIdx: number, outScores: number, q: number, kData: number, weights: number, pageIndices: number, pageIndptr: number, lastPageLen: number, qoIndptr: number, scale: number, totalQ: number, idxNHeads: number, idxHeadDim: number, pageSize: number, topk: number, causal: number, qGlobalStart: number, customMask: number, maskIndptr: number, maskKvLen: number, scores: number, rowLen: number, maxKv: number, coarseHist: number, fineHist: number, meta: number, queryTiles: number, cpWorldSize: number, cpRank: number, globalLastPageLen: number, kvTokenIndptr?: number): void;
  indexerScoreTopkV2(ctx: number, outIdx: number, outScores: number, q: number, kData: number, weights: number, pageIndices: number, pageIndptr: number, lastPageLen: number, qoIndptr: number, scale: number, totalQ: number, idxNHeads: number, idxHeadDim: number, pageSize: number, topk: number, causal: number, qGlobalStart: number, customMask: number, maskIndptr: number, maskKvLen: number, scores: number, rowLen: number, hist: number, meta: number, maxKv: number, numSplits: number, cpWorldSize: number, cpRank: number, globalLastPageLen: number, kvTokenIndptr?: number): void;
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
  kvCacheWrite(ctx: number, srcK: number, srcV: number, dstK: number, dstV: number, slotMapping: number, batchSize: number, nKv: number, hd: number, pageSize: number, srcKTokenStride: number, srcKHeadStride: number, srcVTokenStride: number, srcVHeadStride: number): void;
  synchronize(ctx: number): void;
  synchronizeAsync(ctx: number): Promise<void>;
  synchronizeStream(ctx: number, streamIdx: number): void;
  synchronizeStreamAsync(ctx: number, streamIdx: number): Promise<void>;
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
  graphInstantiate(ctx: number, graph: number): number;
  graphLaunch(ctx: number, graphExec: number): void;
  graphDestroy(ctx: number, graph: number): void;
  graphExecDestroy(ctx: number, graphExec: number): void;
  mmapOpen(path: string): number;
  mmapLoadAsync(ctx: number, gpuDst: number, mmapPtr: number, offset: number, nbytes: number): Promise<void>;
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
  indexerKvCacheAppendFlat(ctx: number, kData: number, appendK: number, kvTokenIndptr: number, batchIndices: number, positions: number, nnz: number, headDim: number, appendStrideN: number): void;
  concatAndCacheDsMla(ctx: number, kvCache: number, appendCkv: number, appendKpe: number, indices: number, indptr: number, batchIndices: number, positions: number, nnz: number, pageSize: number, kvLoraRank: number, peDim: number, appendCkvStrideN: number, appendKpeStrideN: number, cpWorldSize: number, cpRank: number): void;
  sparseMlaPrefill(ctx: number, qNope: number, qRope: number, kvCache: number, indices: number, output: number, outLse: number, numTokens: number, numHeads: number, topk: number, smScale: number, strideKvBlock: number, topkLength: number): void;
  sparseMlaDecode(ctx: number, qNope: number, qRope: number, kvCache: number, indices: number, midOut: number, midLse: number, output: number, outLse: number, numTokens: number, numHeads: number, topk: number, numSplits: number, smScale: number, strideKvBlock: number, chunksPerBlock: number, topkLength?: number): void;
  gatherTopkCkv(ctx: number, flatP0: number, flatP1: number, flatP2: number, flatP3: number, flatP4: number, flatP5: number, flatP6: number, flatP7: number, localKvCache: number, topkIdx: number, batchIndices: number, pageIndices: number, pageIndptr: number, kvTokenIndptr: number, N: number, cpWorldSize: number, cpRank: number, effPageSize: number, bptBytes: number, numTokens: number, topk: number, paddedKvLen: number, scratchBitmap: number, scratchUnique: number, scratchCounter: number): void;
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
  p2pAllGatherRowWrite(ctx: number, localShard: number, p0: number, p1: number, p2: number, p3: number, p4: number, p5: number, p6: number, p7: number, output: number, N: number, shardDim1Bytes: number, fullDim1Bytes: number, outer: number, rank: number): void;
  p2pReduceScatterWrite(ctx: number, localShard: number, p0: number, p1: number, p2: number, p3: number, p4: number, p5: number, p6: number, p7: number, N: number, chunkBytes: number, rank: number): void;
  p2pReduceGatherWrite(ctx: number, staging: number, p0: number, p1: number, p2: number, p3: number, p4: number, p5: number, p6: number, p7: number, N: number, chunkLen: number, rank: number, dtype: number): void;
  p2pBarrier(ctx: number, instance: number, peerRank?: number): void;
  p2pArrive(ctx: number, instance: number, peerRank?: number): void;
  p2pWait(ctx: number, instance: number, peerRank?: number): void;
  cpMergeTree(ctx: number, v0: number, v1: number, v2: number, v3: number, v4: number, v5: number, v6: number, v7: number, lse0: number, lse1: number, lse2: number, lse3: number, lse4: number, lse5: number, lse6: number, lse7: number, numShards: number, outputV: number, outputLse: number, numel: number, batchSize: number, numHeads: number, vHeadDim: number, shardNHeads: number, headOffset: number, inputNHeads: number): void;
  cpMergeScatter(ctx: number, localV: number, localLse: number, dv0: number, dv1: number, dv2: number, dv3: number, dv4: number, dv5: number, dv6: number, dv7: number, dl0: number, dl1: number, dl2: number, dl3: number, dl4: number, dl5: number, dl6: number, dl7: number, worldSize: number, batchSize: number, shardNHeads: number, vHeadDim: number, inputNHeads: number, numHeads: number, rank: number): void;
  cpMergeLocal(ctx: number, stageV: number, stageLse: number, outputV: number, outputLse: number, worldSize: number, batchSize: number, shardNHeads: number, vHeadDim: number): void;
  cpCorrectAttnOut(ctx: number, vOut: number, lses: number, globalLse: number, batchSize: number, numHeads: number, vHeadDim: number, worldSize: number, rank: number): void;
  sigmoid(ctx: number, out: number, input: number, n: number): void;
  relu(ctx: number, out: number, input: number, n: number): void;
  topk(ctx: number, outValues: number, outIndices: number, input: number, k: number, dim: number, batch: number, offset: number): void;
  topkFromScores(ctx: number, outValues: number, outIndices: number, scores: number, rowLen: number, hist: number, meta: number, batch: number, stride: number, topk: number, numSplits: number, cpWorldSize?: number, cpRank?: number): void;
  sortTopkByIndex(ctx: number, outIdx: number, outScores: number, batch: number, topk: number): void;
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
  reduceSum(ctx: number, out: number, input: number, rows: number, cols: number): void;
  rowNormalize(ctx: number, out: number, input: number, scale: number, rows: number, cols: number, normalize: boolean): void;
  groupMaskMul(ctx: number, scores: number, groupMask: number, numExperts: number, expertsPerGroup: number, nGroup: number, batch: number): void;
  mulMatId(ctx: number, output: number, input: number, weightPtrs: number, expertIds: number, topK: number, count: number, N: number, K: number): void;
  nvfp4MulMatId(ctx: number, output: number, input: number, weightPtrs: number, scalePtrs: number, scale2Ptrs: number, expertIds: number, topK: number, count: number, N: number, K: number): void;
  mmaMoeWorkspaceSize(count: number, N: number, K: number, numExperts: number): number;
  mmaMoeCoopWorkspaceSize(count: number, N: number, K: number, numExperts: number): number;
  nvfp4MulMatIdGroupedMmaCoop(ctx: number, output: number, input: number, weightPtrs: number, scalePtrs: number, scale2Ptrs: number, expertIds: number, topK: number, count: number, N: number, K: number, numExperts: number, workspace: number): void;
  mmaMoeCoopScatterWorkspaceSize(count: number, K: number, numExperts: number): number;
  mmaMoeCoopGemmWorkspaceSize(count: number, N: number): number;
  mmaMoeCoopScatter(ctx: number, input: number, expertIds: number, topK: number, count: number, K: number, numExperts: number, workspace: number): void;
  mmaMoeCoopGemm(ctx: number, weightPtrs: number, scalePtrs: number, scale2Ptrs: number, numExperts: number, N: number, K: number, count: number, scatterK: number, scatterWorkspace: number, sortedInput: number, outputSorted: boolean, gemmWorkspace: number, output: number): void;
  bf16MulMatIdGroupedMma(ctx: number, output: number, input: number, weightPtrs: number, expertIds: number, topK: number, count: number, N: number, K: number, numExperts: number, workspace: number): void;
  scatterAddRows(ctx: number, out: number, input: number, scales: number, topK: number, dim: number, numRows: number, workspace: number): void;
  rotateInputIds(ctx: number, outputIds: number, inputIds: number, qoIndptr: number, newTokens: number, batchSize: number): void;
}
