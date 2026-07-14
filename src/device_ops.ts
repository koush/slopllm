import type { Tensor } from "./tensor";
import type { WorkspaceBase } from "./workspace";
import type { PagedKVCache } from "./paged_kv";
import type { ExecutionState } from "./execution-workspace";

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

export interface DeviceOps {
  readonly worldSize: number;
  newTensor(workspace: WorkspaceBase, shape: number[], type: string, pinned: boolean, name?: string, parallelism?: TensorParallelism): Tensor;
  wrapTensor(workspace: WorkspaceBase, data: number, allocSize: number, shape: number[], type: string, pinned: boolean, view: Tensor | undefined): Tensor;
  synchronize(): void;
  synchronizeStream(streamIdx: number): void;
  setStream(streamIdx: number): void;
  eventRecord(eventIdx: number, streamIdx: number): void;
  streamWaitEvent(streamIdx: number, eventIdx: number): void;
  currentStream: number;
  availableStreams: number[];
  withStream<T>(fn: () => T): Disposable  & { result: T, streamWaitEvent(): void, synchronize(): void };

  kvCacheWrite(srcK: Tensor, srcV: Tensor, dstK: Tensor, dstV: Tensor, slotMapping: Tensor, batchSize: number, nKv: number, hd: number, srcKTokenStride: number, srcKHeadStride: number, srcVTokenStride: number, srcVHeadStride: number): void;

  positionStep(positionIds: Tensor, lastPageLen: Tensor, slotMapping: Tensor, indptr: Tensor, indices: Tensor, pageSize: number, batchSize: number, steps?: number): void;
  mlaPositionStep(positionIds: Tensor, lastPageLen: Tensor, indptr: Tensor, pageSize: number, batchSize: number, contextParallel?: boolean, cpWorldSize?: number, cpRank?: number, steps?: number, globalLastPageLen?: Tensor): void;

  batchDecodePlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, indptrH: Tensor, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, enableCudaGraph: boolean): void;
  batchDecodeRun(state: ExecutionState, q: Tensor, o: Tensor, kData: Tensor, vData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, numQoHeads: number, numKvHeads: number, headDim: number, smScale: number): void;
  batchPrefillPagedPlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, qoIndptrH: Tensor, pagedKvIndptrH: Tensor, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, maskMode: MaskMode): void;
  batchPrefillPagedRun(state: ExecutionState, q: Tensor, o: Tensor, kData: Tensor, vData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, floatWs: Tensor, intWs: Tensor, qIndptrD: Tensor, planInfo: Tensor, numQoHeads: number, numKvHeads: number, headDim: number, qStrideN: number, qStrideH: number, maskMode: MaskMode, smScale: number): void;
  batchPrefillRaggedPlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, qoIndptrH: Tensor, kvIndptrH: Tensor, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, maskMode: MaskMode): void;
  batchPrefillRaggedRun(state: ExecutionState, q: Tensor, k: Tensor, v: Tensor, o: Tensor, floatWs: Tensor, intWs: Tensor, qIndptrD: Tensor, kvIndptrD: Tensor, planInfo: Tensor, numQoHeads: number, numKvHeads: number, headDim: number, qStrideN: number, qStrideH: number, kvStrideN: number, kvStrideH: number, vStrideN: number, vStrideH: number, maskMode: MaskMode, smScale: number): void;

  mlaPrefillPlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, qoIndptrH: Tensor, kvIndptrH: Tensor, kvLenH: Tensor, lastPageLenH: Tensor, batchSize: number, numHeads: number, headDimO: number, causal: boolean, pageSize: number, seqKvLens: number[], contextParallel?: boolean, cpWorldSize?: number, cpRank?: number): void;
  mlaPrefillRun(state: ExecutionState, qNope: Tensor, qPe: Tensor, ckvData: Tensor, kpeData: Tensor, kvIndices: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, smScale: number, maskMode: MaskMode, cpWorldSize?: number, cpRank?: number, customMask?: Tensor, maskIndptr?: Tensor, maskKvLen?: Tensor): { o: Tensor, lse: Tensor };
  mlaDecodePlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, indptrH: Tensor, lastPageLenH: Tensor, batchSize: number, numQoHeads: number, pageSize: number, enableCudaGraph: boolean, headDimCkv: number, headDimKpe: number, contextParallel?: boolean, cpWorldSize?: number, cpRank?: number, seqKvLens?: number[]): void;
  mlaDecodeRun(state: ExecutionState, qNope: Tensor, qPe: Tensor, ckvData: Tensor, kpeData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, smScale: number): { o: Tensor, lse: Tensor };
  mlaKvCacheAppend(ckvData: Tensor, kpeData: Tensor | null, indices: Tensor, indptr: Tensor, lastPageLen: Tensor, appendCkv: Tensor, appendKpe: Tensor | null, batchIndices: Tensor, positions: Tensor, nnz: number, headDimCkv: number, headDimKpe: number, appendCkvStrideN: number, appendKpeStrideN: number, pageSize?: number, cpWorldSize?: number, cpRank?: number): void;
  concatAndCacheDsMla(kvCache: Tensor, appendCkv: Tensor, appendKpe: Tensor, indices: Tensor, indptr: Tensor, batchIndices: Tensor, positions: Tensor, nnz: number, kvLoraRank: number, peDim: number, appendCkvStrideN: number, appendKpeStrideN: number, pageSize?: number, cpWorldSize?: number, cpRank?: number): void;

  sparseMlaPrefill(state: ExecutionState, qAbsorbed: Tensor, qPe: Tensor, kvCache: Tensor, indices: Tensor, topk: number, smScale: number, topkLength: Tensor, pageIndptrD: Tensor, lastPageLen: Tensor, kvTokenIndptrD: Tensor): { o: Tensor, lse: Tensor };
  sparseMlaDecode(state: ExecutionState, qAbsorbed: Tensor, qPe: Tensor, kvCache: Tensor, indices: Tensor, topk: number, numSplits: number, smScale: number, chunksPerBlock: number, topkLength?: Tensor): { o: Tensor, lse: Tensor };

  gatherPages(srcData: Tensor, pageIndices: Tensor, pageIndptrD: Tensor, lastPageLen: Tensor, batchSize: number, paddedKvLen: number, kvTokenIndptrD: Tensor, contextParallel: boolean): Tensor;

  gdnRecurrentStep(state: ExecutionState, output: Tensor, recurrentState: Tensor, qkv: Tensor, aRaw: Tensor, bRaw: Tensor, aLog: Tensor, dtBias: Tensor, numHeads: number, dK: number, dV: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void;
  gdnPrefill(state: ExecutionState, output: Tensor, recurrentState: Tensor, qkv: Tensor, aRaw: Tensor, bRaw: Tensor, aLog: Tensor, dtBias: Tensor, cuSeqlens: Tensor, numHeads: number, dK: number, dV: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void;

  indexerScore(out: Tensor, q: Tensor, kData: Tensor, weights: Tensor, pageIndices: Tensor, pageIndptr: Tensor, lastPageLen: Tensor, qoIndptr: Tensor, scale: number, totalQ: number, idxNHeads: number, idxHeadDim: number, pageSize: number, maxKvLen: number, causal: boolean): void;
  // Indexer top-k -> physical KV slots (score+topk then slot mapping). Returns
  // the [totalQ, topk] slots tensor and writes the compacted valid count per
  // query into `topkLength` (a stable caller buffer, for the sparse kernel's
  // early-out). `decode` selects the multi-block v2 kernel; `maxKv` bounds the
  // score scratch (max seq len).
  indexerTopkSlots(idxQ: Tensor, kData: Tensor, weights: Tensor, pageIndices: Tensor, indptr: Tensor, lastPageLen: Tensor, qoIndptr: Tensor, batchIndices: Tensor, topkLength: Tensor, scale: number, topk: number, decode: boolean, contextParallel?: boolean, cpWorldSize?: number, cpRank?: number, globalLastPageLen?: Tensor, customMask?: Tensor, maskIndptr?: Tensor, maskKvLen?: Tensor, qGlobalStart?: number, kvTokenIndptrD?: Tensor): Tensor;

  graphBeginCapture(): void;
  graphEndCapture(): number;
  graphInstantiate(graph: number): number;
  graphLaunch(graphExec: number): void;
  graphDestroy(graph: number): void;
  graphExecDestroy(graphExec: number): void;
}
