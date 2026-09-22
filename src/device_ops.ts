import type { ExecutionState } from "./execution-workspace";
import type { Tensor } from "./tensor";
import type { WorkspaceBase } from "./workspace";
import type { HeapKey } from "./heap";

export enum MaskMode {
  None = 0,
  Causal = 1,
  Custom = 2,
  CausalCustom = 4,
}

export enum TensorParallelism {
  Replicated = "replicated",
  Column = "column",
  Row = "row",
  PartialSum = "partial_sum",
  PartialSoftmax = "partial_softmax",
}

export interface StridedMmap {
  srcOffset: number;
  dstOffset: number;
  srcPitch: number;
  dstPitch: number;
  width: number;
  height: number;
}

// A slots tensor and its paired per-query valid count. Always travel together —
// a slots buffer is only meaningful alongside the length that bounds it.
export type SlotSet = { slots: Tensor, length: Tensor };

export type MlaQuery = { qAbsorbed: Tensor, qPe: Tensor, qAbsorbedScales?: Tensor };

/** E4M3 values and FP32 power-of-two dequantization scales along the last dimension. */
export type Fp8Quantized = { values: Tensor, scales: Tensor };

export interface StreamResult<T> extends Disposable {
  streamId: number;
  result: T;
  streamWaitEvent(): void;
  synchronize(): void;
}

export function fp8ScaleShape(input: Tensor, blockSize: number): number[] {
  const width = input.shape.at(-1);
  if (input.type !== 'BF16' || !width || !Number.isInteger(blockSize) || blockSize <= 0 || width % blockSize !== 0) {
    throw new Error('quantizeFp8 requires BF16 input and a positive block size dividing the last dimension');
  }
  return [...input.shape.slice(0, -1), width / blockSize];
}

export function notifySynchronizedWorkspaces(workspaces: WeakRef<WorkspaceBase>[]): void {
  for (let index = workspaces.length - 1; index >= 0; index--) {
    const workspace = workspaces[index].deref();
    if (workspace) {
      workspace.synchronizeComplete();
    } else {
      workspaces.splice(index, 1);
    }
  }
}

export function notifyHostWorldSynchronization(workspaces: WeakRef<WorkspaceBase>[]): void {
  for (let index = workspaces.length - 1; index >= 0; index--) {
    const workspace = workspaces[index].deref();
    if (workspace) {
      workspace.drainHeap(0, undefined);
    } else {
      workspaces.splice(index, 1);
    }
  }
}

export interface DeviceOps extends Disposable {
  readonly worldSize: number;
  synchronizeListeners: WeakRef<WorkspaceBase>[];
  currentStream: number;
  readonly activeStreams: readonly number[];
  availableStreams: number[];

  newTensor(workspace: WorkspaceBase, shape: number[], type: string, pinned: boolean, name?: string, parallelism?: TensorParallelism, recycleKey?: HeapKey | null): Tensor;
  wrapTensor(workspace: WorkspaceBase, data: number, allocSize: number, shape: number[], type: string, pinned: boolean, view: Tensor | undefined, recycleKey?: HeapKey | null): Tensor;
  synchronize(streamIdx?: number): void;
  synchronizeAsync(streamIdx?: number): Promise<void>;
  /** Host-only stream-0 heap promotion; does not wait for GPU completion or release pinned buffers. */
  hostSynchronizeWorld(): void;
  setStream(streamIdx: number): void;
  eventRecord(eventIdx: number, streamIdx: number): void;
  streamWaitEvent(streamIdx: number, eventIdx: number): void;
  withStream<T>(fn: () => T): StreamResult<T>;
  /** Best-effort L2 warming of up to eight local tensor ranges in one grid. */
  prefetchL2(tensors: readonly Tensor[]): void;

  kvCacheWrite(srcK: Tensor, srcV: Tensor, dstK: Tensor, dstV: Tensor, slotMapping: Tensor, batchSize: number, nKv: number, hd: number, srcKTokenStride: number, srcKHeadStride: number, srcVTokenStride: number, srcVHeadStride: number): void;

  batchDecodePlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, indptrH: Tensor, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, enableCudaGraph: boolean): void;
  batchDecodeRun(state: ExecutionState, q: Tensor, o: Tensor, kData: Tensor, vData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, numQoHeads: number, numKvHeads: number, headDim: number, smScale: number): void;
  batchPrefillPagedPlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, qoIndptrH: Tensor, pagedKvIndptrH: Tensor, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, maskMode: MaskMode): void;
  batchPrefillPagedRun(state: ExecutionState, q: Tensor, o: Tensor, kData: Tensor, vData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, floatWs: Tensor, intWs: Tensor, qIndptrD: Tensor, planInfo: Tensor, numQoHeads: number, numKvHeads: number, headDim: number, qStrideN: number, qStrideH: number, maskMode: MaskMode, smScale: number): void;
  batchPrefillRaggedPlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, qoIndptrH: Tensor, kvIndptrH: Tensor, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, maskMode: MaskMode): void;
  batchPrefillRaggedRun(state: ExecutionState, q: Tensor, k: Tensor, v: Tensor, o: Tensor, floatWs: Tensor, intWs: Tensor, qIndptrD: Tensor, kvIndptrD: Tensor, planInfo: Tensor, numQoHeads: number, numKvHeads: number, headDim: number, qStrideN: number, qStrideH: number, kvStrideN: number, kvStrideH: number, vStrideN: number, vStrideH: number, maskMode: MaskMode, smScale: number): void;

  mlaPrefillPlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, qoIndptrH: Tensor, kvIndptrH: Tensor, kvLenH: Tensor, lastPageLenH: Tensor, batchSize: number, numHeads: number, headDimO: number, causal: boolean, pageSize: number, seqKvLens: number[], contextParallel?: boolean, cpWorldSize?: number, cpRank?: number): void;
  mlaPrefillRun(state: ExecutionState, qNope: Tensor, qPe: Tensor, ckvData: Tensor, kpeData: Tensor, kvIndices: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, smScale: number, maskMode: MaskMode, cpWorldSize?: number, cpRank?: number, customMask?: Tensor, maskIndptr?: Tensor, maskKvLen?: Tensor): { o: Tensor, lse: Tensor };
  mlaDecodePlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, indptrH: Tensor, lastPageLenH: Tensor, batchSize: number, numQoHeads: number, pageSize: number, enableCudaGraph: boolean, headDimCkv: number, headDimKpe: number, seqKvLens: number[], contextParallel?: boolean, cpWorldSize?: number, cpRank?: number): void;
  sparseMlaDecodePlan(lastPageLenH: Tensor, batchSize: number, seqKvLens: number[], pageSize: number, contextParallel: boolean): void;
  mlaDecodeRun(state: ExecutionState, qNope: Tensor, qPe: Tensor, ckvData: Tensor, kpeData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, smScale: number): { o: Tensor, lse: Tensor };
  projectMlaQuery(state: ExecutionState, kvCache: Tensor, qNormed: Tensor, qPeWeight: Tensor, qNopeWeight: Tensor, kNopeWeight: Tensor, absorbedWeight: Tensor, cos: Tensor, sin: Tensor, qkRopeDim: number, kvLoraRank: number, nHeads: number, seqLen: number, batch: number, ropeInterleave: boolean): MlaQuery;
  // Undefined indices selects flat indexer addressing; indptr then holds token prefix sums.
  mlaKvCacheAppend(state: ExecutionState, cacheIdx: number, ckvData: Tensor, kpeData: Tensor | null, indices: Tensor | undefined, indptr: Tensor, lastPageLen: Tensor, appendCkv: Tensor, appendKpe: Tensor | null, batchIndices: Tensor, positions: Tensor, nnz: number, headDimCkv: number, headDimKpe: number, appendCkvStrideN: number, appendKpeStrideN: number, pageSize?: number, cpWorldSize?: number, cpRank?: number): { ckv: Tensor; kpe?: Tensor };
  concatAndCacheDsMla(state: ExecutionState, cacheIdx: number, kvCache: Tensor, appendCkv: Tensor, appendKpe: Tensor, indices: Tensor | undefined, indptr: Tensor, batchIndices: Tensor, positions: Tensor, nnz: number, kvLoraRank: number, peDim: number, appendCkvStrideN: number, appendKpeStrideN: number): Tensor;
  appendSelectedMtpCaches(mlaSrcCkvPtrs: Tensor, mlaSrcKpePtrs: Tensor, mlaDstCkvPtrs: Tensor, mlaDstKpePtrs: Tensor | undefined,
    indexerSrcPtrs: Tensor | undefined, indexerDstPtrs: Tensor | undefined, indexerDstScalePtrs: Tensor | undefined,
    sourceRows: Tensor, indices: Tensor, indptr: Tensor, batchIndices: Tensor, positions: Tensor,
    pageSize: number, kvLoraRank: number, peDim: number, indexHeadDim: number, sparseMode: boolean,
    cpWorldSize?: number, cpRank?: number): void;
  quantizeFp8(input: Tensor, blockSize: number): Fp8Quantized;

  // Decode chunking hint: 0 selects automatic planning; the actual prefill kernel ignores it.
  sparseMlaPrefill(state: ExecutionState, qAbsorbed: Tensor, qPe: Tensor, kvCache: Tensor, indices: Tensor, topk: number, smScale: number, topkLength: Tensor, pageIndptrD: Tensor, lastPageLen: Tensor, kvTokenIndptrD: Tensor, qAbsorbedScales?: Tensor, chunksPerBlock?: number): { o: Tensor, lse: Tensor };
  sparseMlaDecode(state: ExecutionState, qAbsorbed: Tensor, qPe: Tensor, kvCache: Tensor, indices: Tensor, topk: number, numSplits: number, smScale: number, topkLength?: Tensor, qAbsorbedScales?: Tensor, chunksPerBlock?: number): { o: Tensor, lse: Tensor };
  gdnRecurrentStep(state: ExecutionState, output: Tensor, recurrentState: Tensor, qkv: Tensor, aRaw: Tensor, bRaw: Tensor, aLog: Tensor, dtBias: Tensor, numHeads: number, dK: number, dV: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void;
  gdnPrefill(state: ExecutionState, output: Tensor, recurrentState: Tensor, qkv: Tensor, aRaw: Tensor, bRaw: Tensor, aLog: Tensor, dtBias: Tensor, cuSeqlens: Tensor, numHeads: number, dK: number, dV: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void;

  gatherPages(srcData: Tensor, pageIndices: Tensor, pageIndptrD: Tensor, lastPageLen: Tensor, batchSize: number, paddedKvLen: number, kvTokenIndptrD: Tensor, contextParallel: boolean): Tensor;

  indexerScore(out: Tensor, q: Tensor, kData: Tensor, kScaleData: Tensor, weights: Tensor, pageIndices: Tensor, pageIndptr: Tensor, lastPageLen: Tensor, qoIndptr: Tensor, scale: number, totalQ: number, idxNHeads: number, idxHeadDim: number, pageSize: number, maxKvLen: number, causal: boolean, kvTokenIndptr?: Tensor): void;
  // Indexer top-k scoring: returns { values: [totalQ, topk] BF16 scores, indices: [totalQ, topk] I32 positions }.
  indexerTopk(state: ExecutionState, idxQ: Tensor, kData: Tensor, kScaleData: Tensor, weights: Tensor, pageIndices: Tensor, indptr: Tensor, lastPageLen: Tensor, qoIndptr: Tensor, scale: number, topk: number, decode: boolean, qGlobalStart?: number, customMask?: Tensor, maskIndptr?: Tensor, maskKvLen?: Tensor, cpWorldSize?: number, cpRank?: number, globalLastPageLen?: Tensor, kvTokenIndptr?: Tensor): { values: Tensor, indices: Tensor };
  // Sort each top-k row ascending by index (-1 padding last), in place.
  sortTopkByIndex(indices: Tensor, values: Tensor, batch: number, topk: number): void;
  // Convert top-k indices to physical KV slots. Layer and group share memory
  // through independent views, allowing the caller to retain the group slots.
  topkToSlots(state: ExecutionState, topkIdx: Tensor, kvTokenIndptrD: Tensor, pageIndices: Tensor, indptr: Tensor, lastPageLen: Tensor, batchIndices: Tensor, pageSize: number, maxKv: number, cacheIdx: number, contextParallel?: boolean, cpWorldSize?: number, cpRank?: number, providedLength?: Tensor): {
    layer: SlotSet, group: SlotSet,
  };

  graphBeginCapture(): void;
  graphEndCapture(): number;
  graphInstantiate(graph: number): number;
  graphLaunch(graphExec: number): void;
  graphDestroy(graph: number): void;
  graphExecDestroy(graphExec: number): void;

  sampleBatch(outTokens: Tensor, topkVals: Tensor, topkIdxs: Tensor, workspace: Tensor, logits: Tensor, penaltyTokens: Tensor, penaltyCount: Tensor, maxWindow: number, vocabSize: number, batchSize: number, temperatures: Tensor, repPenalties: Tensor, presPenalties: Tensor, topKs: Tensor, topPs: Tensor, stepCounter: Tensor, maxEffectiveK: number, outProbs?: Tensor, outIds?: Tensor, supportCapacity?: number): void;
  sampleCandidates(outTokens: Tensor, outProbs: Tensor, outIds: Tensor, candidateValues: Tensor, candidateIds: Tensor, temperatures: Tensor, topKs: Tensor, topPs: Tensor, stepCounter: Tensor, batchSize: number, candidateCount: number, supportCapacity: number): void;
  specRejectLinear(outTokens: Tensor, outAccepted: Tensor, draftTokens: Tensor, qProbs: Tensor, qIds: Tensor, pProbs: Tensor, pIds: Tensor, stepCounter: Tensor, batchSize: number, depth: number, capacity: number): void;
}
