import type { Tensor } from "./tensor";

export interface DeviceOps {
  alloc(size: number): number;
  allocPinned(bytes: number): number;
  freeBuf(ptr: Tensor): void;
  freePinned(ptr: Tensor): void;
  h2d(dst: Tensor, cpuData: Buffer, size?: number): void;
  d2h(cpuBuf: Buffer, src: Tensor, size?: number): void;
  synchronize(): void;

  rmsnorm(out: Tensor, input: Tensor, weight: Tensor, eps: number, dim: number, batch: number): void;
  fusedAddRmsnorm(out: Tensor, residual: Tensor, inputA: Tensor, inputB: Tensor, weight: Tensor, eps: number, dim: number, batch: number): void;
  fusedNormRope(out: Tensor, input: Tensor, weight: Tensor, cos: Tensor, sin: Tensor, eps: number, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, inStride: number): void;
  siluAndMul(out: Tensor, gate: Tensor, up: Tensor, intermediate: number, batch: number): void;
  linear(out: Tensor, input: Tensor, weight: Tensor, batch: number, n: number, k: number): void;
  embedding(out: Tensor, table: Tensor, ids: Tensor, hidden: number, seqLen: number): void;
  fill(out: Tensor, value: number, n: number): void;
  arange(out: Tensor, start: number, step: number, count: number): void;
  argmax(outIndex: Tensor, input: Tensor, dim: number, batch: number): void;
  indexSelect(out: Tensor, src: Tensor, indices: Tensor, dim: number, k: number): void;
  kvCacheWrite(srcK: Tensor, srcV: Tensor, dstK: Tensor, dstV: Tensor, slotMapping: Tensor, batchSize: number, nKv: number, hd: number, pageSize: number, srcKTokenStride: number, srcKHeadStride: number, srcVTokenStride: number, srcVHeadStride: number): void;
  rotaryEmbedding(cosOut: Tensor, sinOut: Tensor, invFreq: Tensor, positionIds: Tensor, dimHalf: number, batch: number, seqLen: number): void;

  fp8LinearDecode(bf16Out: Tensor, bf16Input: Tensor, fp8Weight: Tensor, weightScale: Tensor, m: number, n: number, k: number): void;
  gdnRecurrentStep(output: Tensor, state: Tensor, qkv: Tensor, aRaw: Tensor, bRaw: Tensor, aLog: Tensor, dtBias: Tensor, numHeads: number, dK: number, dV: number, batchSize: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void;
  gdnPrefill(output: Tensor, state: Tensor, qkv: Tensor, aRaw: Tensor, bRaw: Tensor, aLog: Tensor, dtBias: Tensor, cuSeqlens: Tensor, totalSeqLen: number, numHeads: number, dK: number, dV: number, batchSize: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void;
  causalConv1d(output: Tensor, convState: Tensor, input: Tensor, weight: Tensor, cuSeqlens: Tensor, convDim: number, totalSeqLen: number, kernelSize: number, batchSize: number, convStateStride: number, chStride: number, seqStride: number): void;
  causalConv1dUpdate(output: Tensor, convState: Tensor, input: Tensor, weight: Tensor, convDim: number, kernelSize: number, batchSize: number, convStateStride: number): void;
  rmsnormGated(output: Tensor, input: Tensor, gate: Tensor, weight: Tensor, eps: number, dim: number, batch: number): void;
  gateSigmoidMul(attnOut: Tensor, gateInterleaved: Tensor, batchSeq: number, numHeads: number, headDim: number): void;

  writePinned(dst: Tensor, src: Buffer, size?: number): void;
  batchDecodePlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, indptrH: Tensor, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, enableCudaGraph: boolean): void;
  batchDecodeRun(q: Tensor, o: Tensor, kData: Tensor, vData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, smScale: number): void;
  batchPrefillPagedPlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, qoIndptrH: Tensor, pagedKvIndptrH: Tensor, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, maskMode: number): void;
  batchPrefillPagedRun(q: Tensor, o: Tensor, kData: Tensor, vData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, floatWs: Tensor, intWs: Tensor, qIndptrD: Tensor, planInfo: Tensor, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, qStrideN: number, qStrideH: number, maskMode: number, smScale: number): void;

  sampleBatch(outTokens: Tensor, topkVals: Tensor, topkIdxs: Tensor, workspace: Tensor, logits: Tensor, penaltyTokens: Tensor, penaltyOffsets: Tensor, vocabSize: number, batchSize: number, temperatures: Tensor, repPenalties: Tensor, presPenalties: Tensor, topKs: Tensor, topPs: Tensor, randomVals: Tensor, maxEffectiveK: number): void;

  graphBeginCapture(): void;
  graphEndCapture(): number;
  graphInstantiate(graph: number): number;
  graphLaunch(graphExec: number): void;
  graphDestroy(graph: number): void;
  graphExecDestroy(graphExec: number): void;

  mmapOpen(filePath: string): number;
  mmapLoad(gpuDst: Tensor, mmapPtr: number, offset: number, nbytes: number): void;
  mmapClose(mmapPtr: number, size: number): void;
}
