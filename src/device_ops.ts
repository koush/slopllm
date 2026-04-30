import type { Tensor } from "./tensor";
import type { WorkspaceBase } from "./workspace";

export enum TensorParallelism {
  Replicated = "replicated",
  Column = "column",
  Row = "row",
  PartialSum = "partial_sum",
}

export interface GdnQkvLayout {
  numHeads: number;
  dK: number;
  dV: number;
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

  mlaPrefillPlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, qoIndptrH: Tensor, kvIndptrH: Tensor, kvLenH: Tensor, batchSize: number, numHeads: number, headDimO: number, causal: number): void;
  mlaPrefillRun(qNope: Tensor, qPe: Tensor, ckvData: Tensor, kpeData: Tensor, kvIndices: Tensor, o: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, numHeads: number, pageSize: number, maskMode: number, smScale: number, qNopeStrideN: number, qNopeStrideH: number, qPeStrideN: number, qPeStrideH: number, ckvStridePage: number, ckvStrideN: number, kpeStridePage: number, kpeStrideN: number, oStrideN: number, oStrideH: number): void;
  mlaDecodePlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, indptrH: Tensor, batchSize: number, numQoHeads: number, pageSize: number, enableCudaGraph: boolean): void;
  mlaDecodeRun(qNope: Tensor, qPe: Tensor, ckvData: Tensor, kpeData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, o: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, batchSize: number, numQoHeads: number, pageSize: number, smScale: number): void;

  bmm(C: number, A: number, B: number, alpha: number, beta: number, batch: number, M: number, N: number, K: number, transA: number, transB: number): void;

  ropeTranspose(ctx: number, out: number, input: number, cos: number, sin: number, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, inStride: number): void;
  mlaVExpand(ctx: number, result: number, attnOut: number, vProj: number, kvLoraRank: number, vHeadDim: number, nHeads: number, seqLen: number, batch: number): void;

  sigmoid(ctx: number, out: number, input: number, n: number): void;
  topk(ctx: number, outValues: number, outIndices: number, input: number, k: number, dim: number, batch: number): void;
  indexAdd(ctx: number, out: number, indices: number, values: number, nIndices: number, dim: number): void;
  add(ctx: number, out: number, a: number, b: number, n: number): void;
  scale(ctx: number, out: number, input: number, scale: number, n: number): void;
  mul(ctx: number, out: number, a: number, b: number, n: number): void;
  scatterScalar(ctx: number, out: number, indices: number, value: number, k: number, outDim: number, batch: number): void;
  maskedFill(ctx: number, out: number, input: number, mask: number, value: number, n: number): void;
  applyRotaryPosEmbPartial(ctx: number, out: number, input: number, cos: number, sin: number, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, unsqueezeDim: number): void;
  rowScaleAdd(ctx: number, out: number, input: number, scales: number, rows: number, dim: number): void;

  graphBeginCapture(): void;
  graphEndCapture(): number;
  graphInstantiate(graph: number): number;
  graphLaunch(graphExec: number): void;
  graphDestroy(graph: number): void;
  graphExecDestroy(graphExec: number): void;
}
