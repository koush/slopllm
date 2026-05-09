import type { Tensor } from "./tensor";
import type { WorkspaceBase } from "./workspace";

export enum TensorParallelism {
  Replicated = "replicated",
  Column = "column",
  Row = "row",
  PartialSum = "partial_sum",
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

  kvCacheWrite(srcK: Tensor, srcV: Tensor, dstK: Tensor, dstV: Tensor, slotMapping: Tensor, batchSize: number, nKv: number, hd: number, pageSize: number, srcKTokenStride: number, srcKHeadStride: number, srcVTokenStride: number, srcVHeadStride: number): void;

  decodeStep(positionIds: Tensor, lastPageLen: Tensor, slotMapping: Tensor, indptr: Tensor, indices: Tensor, pageSize: number, batchSize: number): void;

  batchDecodePlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, indptrH: Tensor, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, enableCudaGraph: boolean): void;
  batchDecodeRun(q: Tensor, o: Tensor, kData: Tensor, vData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, smScale: number): void;
  batchPrefillPagedPlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, qoIndptrH: Tensor, pagedKvIndptrH: Tensor, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, maskMode: number): void;
  batchPrefillPagedRun(q: Tensor, o: Tensor, kData: Tensor, vData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, floatWs: Tensor, intWs: Tensor, qIndptrD: Tensor, planInfo: Tensor, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, qStrideN: number, qStrideH: number, maskMode: number, smScale: number): void;

  mlaPrefillPlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, qoIndptrH: Tensor, kvIndptrH: Tensor, kvLenH: Tensor, batchSize: number, numHeads: number, headDimO: number, causal: boolean): void;
  mlaPrefillRun(qNope: Tensor, qPe: Tensor, ckvData: Tensor, kpeData: Tensor, kvIndices: Tensor, o: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, numHeads: number, pageSize: number, maskMode: number, smScale: number, qNopeStrideN: number, qNopeStrideH: number, qPeStrideN: number, qPeStrideH: number, ckvStridePage: number, ckvStrideN: number, kpeStridePage: number, kpeStrideN: number, oStrideN: number, oStrideH: number, headDimCkv: number, headDimKpe: number, lse: Tensor | null, cpWorldSize?: number, cpRank?: number, cpKvLen?: Tensor | null): void;
  mlaDecodePlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, indptrH: Tensor, batchSize: number, numQoHeads: number, pageSize: number, enableCudaGraph: boolean, headDimCkv: number, headDimKpe: number): void;
  mlaDecodeRun(qNope: Tensor, qPe: Tensor, ckvData: Tensor, kpeData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, o: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, batchSize: number, numQoHeads: number, pageSize: number, smScale: number, headDimCkv: number, headDimKpe: number, lse: Tensor | null): void;
  mlaKvCacheAppend(ckvData: Tensor, kpeData: Tensor, indices: Tensor, indptr: Tensor, lastPageLen: Tensor, appendCkv: Tensor, appendKpe: Tensor, batchIndices: Tensor, positions: Tensor, nnz: number, pageSize: number, headDimCkv: number, headDimKpe: number, appendCkvStrideN: number, appendKpeStrideN: number, cpRank?: number, cpWorldSize?: number): void;

  graphBeginCapture(): void;
  graphEndCapture(): number;
  graphInstantiate(graph: number): number;
  graphLaunch(graphExec: number): void;
  graphDestroy(graph: number): void;
  graphExecDestroy(graphExec: number): void;
}
