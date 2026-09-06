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

export interface WorkspaceMemoryStats {
  regions: number;
  freeBytes: number;
}

// A slots tensor and its paired per-query valid count. Always travel together —
// a slots buffer is only meaningful alongside the length that bounds it.
export type SlotSet = { slots: Tensor, length: Tensor };

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

export interface DeviceOps extends Disposable {
  readonly worldSize: number;
  synchronizeListeners: WeakRef<WorkspaceBase>[];
  newTensor(workspace: WorkspaceBase, shape: number[], type: string, pinned: boolean, name?: string, parallelism?: TensorParallelism, recycleKey?: HeapKey | null): Tensor;
  wrapTensor(workspace: WorkspaceBase, data: number, allocSize: number, shape: number[], type: string, pinned: boolean, view: Tensor | undefined, recycleKey?: HeapKey | null): Tensor;
  workspaceMemoryStats(workspace: WorkspaceBase): WorkspaceMemoryStats[];
  reclaimWorkspaceMemory(workspace: WorkspaceBase): void;
  deviceHeapStats(): WorkspaceMemoryStats[];
  synchronize(): void;
  synchronizeAsync(): Promise<void>;
  synchronizeStream(streamIdx: number): void;
  synchronizeStreamAsync(streamIdx: number): Promise<void>;
  setStream(streamIdx: number): void;
  eventRecord(eventIdx: number, streamIdx: number): void;
  streamWaitEvent(streamIdx: number, eventIdx: number): void;
  currentStream: number;
  readonly activeStreams: readonly number[];
  availableStreams: number[];
  withStream<T>(fn: () => T): Disposable & { result: T, streamWaitEvent(): void, synchronize(): void };

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
  projectMlaQuery(state: ExecutionState, kvCache: Tensor, qNormed: Tensor, qPeWeight: Tensor, absorbedWeight: Tensor, cos: Tensor, sin: Tensor, qkRopeDim: number, kvLoraRank: number, nHeads: number, seqLen: number, batch: number, ropeInterleave: boolean): { qAbsorbed: Tensor, qPe: Tensor };
  mlaKvCacheAppend(state: ExecutionState, cacheIdx: number, ckvData: Tensor, kpeData: Tensor | null, indices: Tensor, indptr: Tensor, lastPageLen: Tensor, appendCkv: Tensor, appendKpe: Tensor | null, batchIndices: Tensor, positions: Tensor, nnz: number, headDimCkv: number, headDimKpe: number, appendCkvStrideN: number, appendKpeStrideN: number, pageSize?: number, cpWorldSize?: number, cpRank?: number): { ckv: Tensor; kpe?: Tensor };
  concatAndCacheDsMla(state: ExecutionState, cacheIdx: number, kvCache: Tensor, appendCkv: Tensor, appendKpe: Tensor, indices: Tensor | undefined, indptr: Tensor, batchIndices: Tensor, positions: Tensor, nnz: number, kvLoraRank: number, peDim: number, appendCkvStrideN: number, appendKpeStrideN: number): Tensor;
  appendSelectedMtpCaches(mlaSrcCkvPtrs: Tensor, mlaSrcKpePtrs: Tensor, mlaDstCkvPtrs: Tensor, mlaDstKpePtrs: Tensor | undefined,
    indexerSrcPtrs: Tensor | undefined, indexerDstPtrs: Tensor | undefined, indexerDstScalePtrs: Tensor | undefined,
    sourceRows: Tensor, indices: Tensor, indptr: Tensor, batchIndices: Tensor, positions: Tensor,
    pageSize: number, kvLoraRank: number, peDim: number, indexHeadDim: number, sparseMode: boolean,
    cpWorldSize?: number, cpRank?: number): void;
  sparseMlaPrepareCache(state: ExecutionState, groupSlots: Tensor, cacheIdx: number, kvCache: Tensor, appendCkv: Tensor, appendKpe: Tensor, topk: Tensor | undefined, indices: Tensor | null, indptr: Tensor, batchIndices: Tensor, positions: Tensor, nnz: number, kvLoraRank: number, peDim: number, appendCkvStrideN: number, appendKpeStrideN: number): Tensor;

  sparseMlaPrefill(state: ExecutionState, qAbsorbed: Tensor, qPe: Tensor, kvCache: Tensor, indices: Tensor, topk: number, smScale: number, topkLength: Tensor, pageIndptrD: Tensor, lastPageLen: Tensor, kvTokenIndptrD: Tensor): { o: Tensor, lse: Tensor };
  sparseMlaDecode(state: ExecutionState, qAbsorbed: Tensor, qPe: Tensor, kvCache: Tensor, indices: Tensor, topk: number, numSplits: number, smScale: number, chunksPerBlock: number, topkLength?: Tensor): { o: Tensor, lse: Tensor };
  gatherPages(srcData: Tensor, pageIndices: Tensor, pageIndptrD: Tensor, lastPageLen: Tensor, batchSize: number, paddedKvLen: number, kvTokenIndptrD: Tensor, contextParallel: boolean): Tensor;
  // Sparse topk-driven P2P gather of packed CKV tokens into the caller-provided
  // output flat buffers (same [paddedKvLen/pageSize, pageSize, BPT] U8 layout
  // gatherPages produces). The implementation does NOT allocate outputs —
  // callers pre-allocate and pass them in. Returns void.
  //
  // Each rank reads each (query, k) topk entry from its local paged cache shard
  // once and fan-out writes those BPT bytes to all N peer output flat buffers at
  // flat_slot = kvTokenIndptr[seq] + token_pos. CP filter drops positions whose
  // global pos % cp_world_size != this rank's cp_rank. After the call every peer
  // buffer has the same union of topk tokens; non-topk flat slots are NOT touched
  // (preserving stale data — the consumer only reads slots in the
  // topk_length-bounded prefix).
  //
  // outputs: up to 8 caller-pre-allocated flat output tensors. For GlmOps this
  //          is the literal peer-buffer table (N = outputs.length, 1..8). For
  //          ParallelOps the callsite passes a single ParallelTensor in
  //          outputs[0]; ParallelOps internally extracts its N shards and uses
  //          them as the per-rank peer table.
  // kvCache: per-rank paged CKV cache (Row under CP, Replicated otherwise).
  // topkIdx:        [num_tokens, topk] I32 Replicated — global KV positions in seq.
  // pageIndices:    per-rank page-id table.
  // pageIndptr:     [B+1] I32 per-rank page range per seq.
  // kvTokenIndptr:  [B+1] I32 Replicated — global per-seq token prefix sum
  //                 (= flat-slot base for the output buffers).
  // batchIndices:   [num_tokens] I32 Replicated — seq index per query.
  // topk:           number of topk entries per query.
  // paddedKvLen:    output flat buffer size in tokens (page-aligned).
  // CP signaling: cpWorldSize=0 (default) indicates no CP at all (filter
  //               skipped); CP callers pass cpWorldSize=world_size and
  //               cpRank=i. cpWorldSize=1 is degenerate CP — filter runs but
  //               is a no-op (pos % 1 == 0). effPageSize defaults to
  //               kvCache.shape[1] (full page size) when omitted and MUST be
  //               pageSize/worldSize for CP callers.
  gatherTopkCkv(state: ExecutionState, kvCache: Tensor, outputs: readonly Tensor[], topkIdx: Tensor, pageIndices: Tensor, pageIndptr: Tensor, kvTokenIndptr: Tensor, batchIndices: Tensor, topk: number, paddedKvLen: number, cpWorldSize?: number, cpRank?: number, effPageSize?: number): void;
  gdnRecurrentStep(state: ExecutionState, output: Tensor, recurrentState: Tensor, qkv: Tensor, aRaw: Tensor, bRaw: Tensor, aLog: Tensor, dtBias: Tensor, numHeads: number, dK: number, dV: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void;
  gdnPrefill(state: ExecutionState, output: Tensor, recurrentState: Tensor, qkv: Tensor, aRaw: Tensor, bRaw: Tensor, aLog: Tensor, dtBias: Tensor, cuSeqlens: Tensor, numHeads: number, dK: number, dV: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void;

  indexerScore(out: Tensor, q: Tensor, kData: Tensor, kScaleData: Tensor, weights: Tensor, pageIndices: Tensor, pageIndptr: Tensor, lastPageLen: Tensor, qoIndptr: Tensor, scale: number, totalQ: number, idxNHeads: number, idxHeadDim: number, pageSize: number, maxKvLen: number, causal: boolean, kvTokenIndptr?: Tensor): void;
  // Indexer top-k scoring: returns { values: [totalQ, topk] BF16 scores, indices: [totalQ, topk] I32 positions }.
  indexerTopk(state: ExecutionState, idxQ: Tensor, kData: Tensor, kScaleData: Tensor, weights: Tensor, pageIndices: Tensor, indptr: Tensor, lastPageLen: Tensor, qoIndptr: Tensor, scale: number, topk: number, decode: boolean, qGlobalStart?: number, customMask?: Tensor, maskIndptr?: Tensor, maskKvLen?: Tensor, cpWorldSize?: number, cpRank?: number, globalLastPageLen?: Tensor, kvTokenIndptr?: Tensor): { values: Tensor, indices: Tensor };
  // Sort each top-k row ascending by index (-1 padding last), in place.
  sortTopkByIndex(indices: Tensor, values: Tensor, batch: number, topk: number): void;
  // Convert top-k indices to physical KV slots for layer `cacheIdx` AND for the
  // shared group that follows it. The flat/paged addressing of each is decided
  // internally from `cacheIdx` (see ParallelOps.topkSlotMode) so callers stay
  // mode-agnostic; the device level ignores `cacheIdx` and uses
  // cpWorldSize/cpRank directly.
  //
  // `group` is always present. When both need the same addressing — every mode
  // except decode sparse-gather — or when no shared group follows, it is a
  // viewClone of `layer`: same memory, independent handle, so the caller can
  // `using` one and park the other in a holder without a double dispose.
  //
  // `stream` is the background CKV gather (decode sparse-gather only); join it
  // before the group's shared layers read their gathered buffers.
  topkToSlots(state: ExecutionState, topkIdx: Tensor, kvTokenIndptrD: Tensor, pageIndices: Tensor, indptr: Tensor, lastPageLen: Tensor, batchIndices: Tensor, pageSize: number, maxKv: number, cacheIdx: number, contextParallel?: boolean, cpWorldSize?: number, cpRank?: number, providedLength?: Tensor): {
    layer: SlotSet, group: SlotSet, stream?: Disposable & { streamWaitEvent(): void, synchronize(): void },
  };

  graphBeginCapture(): void;
  graphEndCapture(): number;
  graphInstantiate(graph: number): number;
  graphLaunch(graphExec: number): void;
  graphDestroy(graph: number): void;
  graphExecDestroy(graphExec: number): void;

  sampleBatch(outTokens: Tensor, topkVals: Tensor, topkIdxs: Tensor, workspace: Tensor, logits: Tensor, penaltyTokens: Tensor, penaltyCount: Tensor, maxWindow: number, vocabSize: number, batchSize: number, temperatures: Tensor, repPenalties: Tensor, presPenalties: Tensor, topKs: Tensor, topPs: Tensor, stepCounter: Tensor, maxEffectiveK: number): void;
}
