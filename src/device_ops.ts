export interface DeviceOps {
//   alloc(size: number): number;
  allocPinned(bytes: number): number;
//   freeBuf(ptr: number): void;
  freePinned(ptr: number): void;
  h2d(gpuPtr: number, cpuData: Buffer, size?: number): void;
//   d2h(cpuBuf: Buffer, gpuPtr: number, size?: number): void;
//   synchronize(): void;

//   rmsnorm(out: number, input: number, weight: number, eps: number, dim: number, batch: number): void;
//   fusedAddRmsnorm(out: number, residual: number, inputA: number, inputB: number, weight: number, eps: number, dim: number, batch: number): void;
//   fusedNormRope(out: number, input: number, weight: number, cos: number, sin: number, eps: number, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, inStride: number): void;
//   siluAndMul(out: number, gate: number, up: number, intermediate: number, batch: number): void;
//   linear(out: number, input: number, weight: number, batch: number, n: number, k: number): void;
//   embedding(out: number, table: number, ids: number, hidden: number, seqLen: number): void;
//   fill(out: number, value: number, n: number): void;
//   arange(out: number, start: number, step: number, count: number): void;
//   argmax(outIndex: number, input: number, dim: number, batch: number): void;
//   indexSelect(out: number, src: number, indices: number, dim: number, k: number): void;
//   memcpy(dst: number, src: number, bytes: number): void;
  kvCacheWrite(srcK: number, srcV: number, dstK: number, dstV: number, slotMapping: number, batchSize: number, nKv: number, hd: number, pageSize: number, srcKTokenStride: number, srcKHeadStride: number, srcVTokenStride: number, srcVHeadStride: number): void;
//   rotaryEmbedding(cosOut: number, sinOut: number, invFreq: number, positionIds: number, dimHalf: number, batch: number, seqLen: number): void;

//   fp8LinearDecode(bf16Out: number, bf16Input: number, fp8Weight: number, weightScale: number, m: number, n: number, k: number): void;
//   gdnRecurrentStep(output: number, state: number, qkv: number, aRaw: number, bRaw: number, aLog: number, dtBias: number, numHeads: number, dK: number, dV: number, batchSize: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void;
//   gdnPrefill(output: number, state: number, qkv: number, aRaw: number, bRaw: number, aLog: number, dtBias: number, cuSeqlens: number, totalSeqLen: number, numHeads: number, dK: number, dV: number, batchSize: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void;
//   causalConv1d(output: number, convState: number, input: number, weight: number, cuSeqlens: number, convDim: number, totalSeqLen: number, kernelSize: number, batchSize: number, convStateStride: number, chStride: number, seqStride: number): void;
//   causalConv1dUpdate(output: number, convState: number, input: number, weight: number, convDim: number, kernelSize: number, batchSize: number, convStateStride: number): void;
//   rmsnormGated(output: number, input: number, gate: number, weight: number, eps: number, dim: number, batch: number): void;
//   gateSigmoidMul(attnOut: number, gateInterleaved: number, batchSeq: number, numHeads: number, headDim: number): void;

  writePinned(dst: number, src: Buffer, size?: number): void;
  batchDecodePlan(floatWs: number, floatWsSize: number, intWs: number, pinnedIntWs: number, intWsSize: number, planInfo: number, indptrH: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, enableCudaGraph: boolean): void;
  batchDecodeRun(q: number, o: number, kData: number, vData: number, indices: number, indptrD: number, lastPageLen: number, floatWs: number, intWs: number, planInfo: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, smScale: number): void;
  batchPrefillPagedPlan(floatWs: number, floatWsSize: number, intWs: number, pinnedIntWs: number, intWsSize: number, planInfo: number, qoIndptrH: number, pagedKvIndptrH: number, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, maskMode: number): void;
  batchPrefillPagedRun(q: number, o: number, kData: number, vData: number, indices: number, indptrD: number, lastPageLen: number, floatWs: number, intWs: number, qIndptrD: number, planInfo: number, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, qStrideN: number, qStrideH: number, maskMode: number, smScale: number): void;

//   sampleBatch(outTokens: number, topkVals: number, topkIdxs: number, workspace: number, logits: number, penaltyTokens: number, penaltyOffsets: number, vocabSize: number, batchSize: number, temperatures: number, repPenalties: number, presPenalties: number, topKs: number, topPs: number, randomVals: number, maxEffectiveK: number): void;

//   mmapOpen(filePath: string): number;
//   mmapLoad(gpuDst: number, mmapPtr: number, offset: number, nbytes: number): void;
//   mmapClose(mmapPtr: number, size: number): void;
}
