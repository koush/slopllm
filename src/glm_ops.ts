import path from "node:path";
import fs from "node:fs";
import { WorkspaceBase } from "./workspace";
import { Tensor } from "./tensor";
import { DECODE_PLAN_INFO_SIZE, PREFILL_PLAN_INFO_SIZE } from "./paged_kv";

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

interface NativeAddon {
  init(deviceId: number): number;
  free(ctx: number): void;
  alloc(ctx: number, size: number): number;
  freeBuf(ctx: number, ptr: number): void;
  h2d(ctx: number, dst: number, src: Buffer, size: number): void;
  d2h(ctx: number, dst: Buffer, src: number, size: number): void;
  rmsnorm(ctx: number, out: number, input: number, weight: number, eps: number, dim: number, batch: number): void;
  fusedAddRmsnorm(ctx: number, out: number, residual: number, inputA: number, inputB: number, weight: number, eps: number, dim: number, batch: number): void;
  fusedNormRope(ctx: number, out: number, input: number, weight: number, cos: number, sin: number, eps: number, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, inStride: number): void;
  siluAndMul(ctx: number, out: number, gate: number, up: number, intermediate: number, batch: number): void;
  linear(ctx: number, out: number, input: number, weight: number, batch: number, n: number, k: number): void;
  embedding(ctx: number, out: number, table: number, ids: number, hidden: number, seqLen: number): void;
  layernorm(ctx: number, out: number, input: number, weight: number, bias: number, eps: number, dim: number, batch: number): void;
  relu(ctx: number, out: number, input: number, n: number): void;
  sigmoid(ctx: number, out: number, input: number, n: number): void;
  softmax(ctx: number, out: number, input: number, mask: number, dim: number, batch: number): void;
  causalMask(ctx: number, out: number, seqLen: number): void;
  fill(ctx: number, out: number, value: number, n: number): void;
  gather(ctx: number, out: number, input: number, indices: number, k: number, inDim: number, batch: number): void;
  scatterScalar(ctx: number, out: number, indices: number, value: number, k: number, outDim: number, batch: number): void;
  catLastDim(ctx: number, out: number, a: number, b: number, aLastDim: number, bLastDim: number, outer: number): void;
  maskedFill(ctx: number, out: number, input: number, mask: number, value: number, n: number): void;
  indexAdd(ctx: number, out: number, indices: number, values: number, nIndices: number, dim: number): void;
  rotaryEmbedding(ctx: number, cosOut: number, sinOut: number, invFreq: number, positionIds: number, dimHalf: number, batch: number, seqLen: number): void;
  applyRotaryPosEmb(ctx: number, out: number, x: number, cos: number, sin: number, ropeDim: number, nHeads: number, seqLen: number, batch: number, unsqueezeDim: number): void;
  applyRotaryPosEmbPartial(ctx: number, out: number, x: number, cos: number, sin: number, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, unsqueezeDim: number): void;
  topk(ctx: number, outValues: number, outIndices: number, input: number, k: number, dim: number, batch: number): void;
  bmm(ctx: number, C: number, A: number, B: number, alpha: number, beta: number, batch: number, M: number, N: number, K: number, transB: number): void;
  scale(ctx: number, out: number, input: number, s: number, n: number): void;
  add(ctx: number, out: number, a: number, b: number, n: number): void;
  expandDim1(ctx: number, out: number, input: number, dim1Out: number, dim1In: number, seqLen: number, headDim: number, batch: number): void;
  expandDim1Strided(ctx: number, out: number, input: number, dim1Out: number, dim1In: number, seqLen: number, headDim: number, batch: number, headStride: number): void;
  transpose4d(ctx: number, out: number, input: number, d0: number, d1: number, d2: number, d3: number, p0: number, p1: number, p2: number, p3: number): void;
  mul(ctx: number, out: number, a: number, b: number, n: number): void;
  reduceSum(ctx: number, out: number, input: number, rows: number, cols: number): void;
  indexSelect(ctx: number, out: number, src: number, indices: number, dim: number, k: number): void;
  arange(ctx: number, out: number, start: number, step: number, count: number): void;
  argmax(ctx: number, outIndex: number, input: number, dim: number, batch: number): void;
  memcpy(ctx: number, dst: number, src: number, bytes: number): void;
  kvCacheWrite(ctx: number, srcK: number, srcV: number, dstK: number, dstV: number, slotMapping: number, batchSize: number, nKv: number, hd: number, pageSize: number, srcKTokenStride: number, srcKHeadStride: number, srcVTokenStride: number, srcVHeadStride: number): void;
  synchronize(ctx: number): void;
  flashPrefill(ctx: number, q: number, k: number, v: number, o: number, tmp: number, qoLen: number, kvLen: number, numQoHeads: number, numKvHeads: number, headDim: number, qStrideN: number, qStrideH: number, kvStrideN: number, kvStrideH: number, vStrideN: number, vStrideH: number, maskMode: number, kvLayout: number, smScale: number): void;
  flashDecode(ctx: number, q: number, k: number, v: number, o: number, tmp: number, kvLen: number, numQoHeads: number, numKvHeads: number, headDim: number, qStrideN: number, qStrideH: number, kvStrideN: number, kvStrideH: number, smScale: number): void;
  allocPinned(bytes: number): number;
  freePinned(ptr: number): void;
  writePinned(dst: number, src: Buffer, size: number): void;
  batchDecodePlan(ctx: number, floatWs: number, floatWsSize: number, intWs: number, pinnedIntWs: number, intWsSize: number, planInfo: number, indptrH: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, enableCudaGraph: boolean): void;
  batchDecodeRun(ctx: number, q: number, o: number, kData: number, vData: number, indices: number, indptrD: number, lastPageLen: number, floatWs: number, intWs: number, planInfo: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, smScale: number): void;
  batchPrefillPagedPlan(ctx: number, floatWs: number, floatWsSize: number, intWs: number, pinnedIntWs: number, intWsSize: number, planInfo: number, qoIndptrH: number, pagedKvIndptrH: number, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, maskMode: number): void;
  batchPrefillPagedRun(ctx: number, q: number, o: number, kData: number, vData: number, indices: number, indptrD: number, lastPageLen: number, floatWs: number, intWs: number, qIndptrD: number, planInfo: number, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, qStrideN: number, qStrideH: number, maskMode: number, smScale: number): void;
  graphBeginCapture(ctx: number): void;
  graphEndCapture(ctx: number): number;
  graphInstantiate(graph: number): number;
  graphLaunch(graphExec: number, ctx: number): void;
  graphExecUpdate(graphExec: number, graph: number): number;
  graphDestroy(graph: number): void;
  graphExecDestroy(graphExec: number): void;
  mmapOpen(path: string): number;
  mmapLoad(ctx: number, gpuDst: number, mmapPtr: number, offset: number, nbytes: number): void;
  mmapClose(mmapPtr: number, size: number): void;
  fp8LinearDecode(ctx: number, bf16Out: number, bf16Input: number, fp8Weight: number, weightScale: number, m: number, n: number, k: number): void;
  gdnRecurrentStep(ctx: number, output: number, state: number, qkv: number, aRaw: number, bRaw: number, aLog: number, dtBias: number, numHeads: number, dK: number, dV: number, batchSize: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void;
  gdnPrefill(ctx: number, output: number, state: number, qkv: number, aRaw: number, bRaw: number, aLog: number, dtBias: number, cuSeqlens: number, totalSeqLen: number, numHeads: number, dK: number, dV: number, batchSize: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void;
  causalConv1d(ctx: number, output: number, convState: number, input: number, weight: number, cuSeqlens: number, convDim: number, totalSeqLen: number, kernelSize: number, batchSize: number, convStateStride: number, chStride: number, seqStride: number): void;
  causalConv1dUpdate(ctx: number, output: number, convState: number, input: number, weight: number, convDim: number, kernelSize: number, batchSize: number, convStateStride: number): void;
  rmsnormGated(ctx: number, output: number, input: number, gate: number, weight: number, eps: number, dim: number, batch: number): void;
  gateSigmoidMul(ctx: number, attnOut: number, gateInterleaved: number, batchSeq: number, numHeads: number, headDim: number): void;
  sampleBatch(ctx: number, outTokens: number, topkVals: number, topkIdxs: number, workspace: number, logits: number, penaltyTokens: number, penaltyOffsets: number, vocabSize: number, batchSize: number, temperatures: number, repPenalties: number, presPenalties: number, topKs: number, topPs: number, randomVals: number, maxEffectiveK: number): void;
  memcpy2d(ctx: number, dst: number, dpitch: number, src: number, spitch: number, width: number, height: number, kind: number): void;
  ncclUniqueId(outId: Buffer): void;
  ncclCommInitRank(rank: number, worldSize: number, uniqueId: number): number;
  ncclCommDestroy(comm: number): void;
  ncclAllReduce(comm: number, ctx: number, sendbuff: number, recvbuff: number, count: number, datatype: number, op: number): void;
  ncclAllGather(comm: number, ctx: number, sendbuff: number, recvbuff: number, count: number, datatype: number): void;
}

export class GlmOps {
  native: NativeAddon;
  ctx: number;
  device: number;

  constructor(deviceId: number = 0, libPath?: string) {
    const p = libPath ?? path.join(findProjectRoot(__dirname), "build", "Release", "glm.node");
    this.native = require(p) as NativeAddon;
    this.ctx = this.native.init(deviceId);
    if (!this.ctx) {
      throw new Error(`glm_init failed on device ${deviceId}`);
    }
    this.device = deviceId;
  }

  free(): void {
    this.native.free(this.ctx);
  }

  alloc(size: number): number {
    const p = this.native.alloc(this.ctx, size);
    if (!p) throw new Error(`glm_alloc failed for size ${size}`);
    return p;
  }

  freeBuf(ptr: Tensor): void {
    this.native.freeBuf(this.ctx, ptr.data);
  }

  h2d(dst: Tensor, cpuData: Buffer, size?: number): void {
    this.native.h2d(this.ctx, dst.data, cpuData, size ?? cpuData.length);
  }

  d2h(cpuBuf: Buffer, src: Tensor, size?: number): void {
    this.native.d2h(this.ctx, cpuBuf, src.data, size ?? cpuBuf.length);
  }

  synchronize(): void {
    this.native.synchronize(this.ctx);
  }

  rmsnorm(out: Tensor, input: Tensor, weight: Tensor, eps: number, dim: number, batch: number): void {
    this.native.rmsnorm(this.ctx, ptr(out), ptr(input), ptr(weight), eps, dim, batch);
  }

  fusedAddRmsnorm(out: Tensor, residual: Tensor, inputA: Tensor, inputB: Tensor, weight: Tensor, eps: number, dim: number, batch: number): void {
    this.native.fusedAddRmsnorm(this.ctx, ptr(out), ptr(residual), ptr(inputA), ptr(inputB), ptr(weight), eps, dim, batch);
  }

  fusedNormRope(out: Tensor, input: Tensor, weight: Tensor, cos: Tensor, sin: Tensor, eps: number, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, inStride: number): void {
    this.native.fusedNormRope(this.ctx, ptr(out), ptr(input), ptr(weight), ptr(cos), ptr(sin), eps, ropeDim, headDim, nHeads, seqLen, batch, inStride);
  }

  siluAndMul(out: Tensor, gate: Tensor, up: Tensor, intermediate: number, batch: number): void {
    this.native.siluAndMul(this.ctx, ptr(out), ptr(gate), ptr(up), intermediate, batch);
  }

  linear(out: Tensor, input: Tensor, weight: Tensor, batch: number, n: number, k: number): void {
    this.native.linear(this.ctx, ptr(out), ptr(input), ptr(weight), batch, n, k);
  }

  embedding(out: Tensor, table: Tensor, ids: Tensor, hidden: number, seqLen: number): void {
    this.native.embedding(this.ctx, ptr(out), ptr(table), ptr(ids), hidden, seqLen);
  }

  layernorm(out: Tensor, input: Tensor, weight: Tensor, bias: Tensor, eps: number, dim: number, batch: number): void {
    this.native.layernorm(this.ctx, ptr(out), ptr(input), ptr(weight), ptr(bias), eps, dim, batch);
  }

  softmax(out: Tensor, input: Tensor, mask: Tensor, dim: number, batch: number): void {
    this.native.softmax(this.ctx, ptr(out), ptr(input), ptr(mask), dim, batch);
  }

  causalMask(out: Tensor, seqLen: number): void {
    this.native.causalMask(this.ctx, ptr(out), seqLen);
  }

  fill(out: Tensor, value: number, n: number): void {
    this.native.fill(this.ctx, ptr(out), value, n);
  }

  add(out: Tensor, a: Tensor, b: Tensor, n: number): void {
    this.native.add(this.ctx, ptr(out), ptr(a), ptr(b), n);
  }

  arange(out: Tensor, start: number, step: number, count: number): void {
    this.native.arange(this.ctx, ptr(out), start, step, count);
  }

  argmax(outIndex: Tensor, input: Tensor, dim: number, batch: number = 1): void {
    this.native.argmax(this.ctx, ptr(outIndex), ptr(input), dim, batch);
  }

  memcpy(dst: number, src: number, bytes: number): void {
    this.native.memcpy(this.ctx, dst, src, bytes);
  }

  kvCacheWrite(srcK: Tensor, srcV: Tensor, dstK: Tensor, dstV: Tensor, slotMapping: Tensor, batchSize: number, nKv: number, hd: number, pageSize: number, srcKTokenStride: number, srcKHeadStride: number, srcVTokenStride: number, srcVHeadStride: number): void {
    this.native.kvCacheWrite(this.ctx, ptr(srcK), ptr(srcV), ptr(dstK), ptr(dstV), ptr(slotMapping), batchSize, nKv, hd, pageSize, srcKTokenStride, srcKHeadStride, srcVTokenStride, srcVHeadStride);
  }

  rotaryEmbedding(cosOut: Tensor, sinOut: Tensor, invFreq: Tensor, positionIds: Tensor, dimHalf: number, batch: number, seqLen: number): void {
    this.native.rotaryEmbedding(this.ctx, ptr(cosOut), ptr(sinOut), ptr(invFreq), ptr(positionIds), dimHalf, batch, seqLen);
  }

  applyRotaryPosEmb(out: Tensor, x: Tensor, cos: Tensor, sin: Tensor, ropeDim: number, nHeads: number, seqLen: number, batch: number, unsqueezeDim: number): void {
    this.native.applyRotaryPosEmb(this.ctx, ptr(out), ptr(x), ptr(cos), ptr(sin), ropeDim, nHeads, seqLen, batch, unsqueezeDim);
  }

  applyRotaryPosEmbPartial(out: Tensor, x: Tensor, cos: Tensor, sin: Tensor, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, unsqueezeDim: number): void {
    this.native.applyRotaryPosEmbPartial(this.ctx, ptr(out), ptr(x), ptr(cos), ptr(sin), ropeDim, headDim, nHeads, seqLen, batch, unsqueezeDim);
  }

  expandDim1(out: Tensor, input: Tensor, dim1Out: number, dim1In: number, seqLen: number, headDim: number, batch: number): void {
    this.native.expandDim1(this.ctx, ptr(out), ptr(input), dim1Out, dim1In, seqLen, headDim, batch);
  }

  expandDim1Strided(out: Tensor, input: Tensor, dim1Out: number, dim1In: number, seqLen: number, headDim: number, batch: number, headStride: number): void {
    this.native.expandDim1Strided(this.ctx, ptr(out), ptr(input), dim1Out, dim1In, seqLen, headDim, batch, headStride);
  }

  indexSelect(out: Tensor, src: Tensor, indices: Tensor, dim: number, k: number): void {
    this.native.indexSelect(this.ctx, ptr(out), ptr(src), ptr(indices), dim, k);
  }

  transpose4d(out: Tensor, input: Tensor, d0: number, d1: number, d2: number, d3: number, p0: number, p1: number, p2: number, p3: number): void {
    this.native.transpose4d(this.ctx, ptr(out), ptr(input), d0, d1, d2, d3, p0, p1, p2, p3);
  }

  bmm(C: Tensor, A: Tensor, B: Tensor, alpha: number, beta: number, batch: number, M: number, N: number, K: number, transB: number): void {
    this.native.bmm(this.ctx, ptr(C), ptr(A), ptr(B), alpha, beta, batch, M, N, K, transB);
  }

  flashPrefill(q: Tensor, k: Tensor, v: Tensor, o: Tensor, tmp: Tensor, qoLen: number, kvLen: number, numQoHeads: number, numKvHeads: number, headDim: number, qStrideN: number, qStrideH: number, kvStrideN: number, kvStrideH: number, vStrideN: number, vStrideH: number, maskMode: number, kvLayout: number, smScale: number): void {
    this.native.flashPrefill(this.ctx, ptr(q), ptr(k), ptr(v), ptr(o), ptr(tmp), qoLen, kvLen, numQoHeads, numKvHeads, headDim, qStrideN, qStrideH, kvStrideN, kvStrideH, vStrideN, vStrideH, maskMode, kvLayout, smScale);
  }

  flashDecode(q: Tensor, k: Tensor, v: Tensor, o: Tensor, tmp: Tensor, kvLen: number, numQoHeads: number, numKvHeads: number, headDim: number, qStrideN: number, qStrideH: number, kvStrideN: number, kvStrideH: number, smScale: number): void {
    this.native.flashDecode(this.ctx, ptr(q), ptr(k), ptr(v), ptr(o), ptr(tmp), kvLen, numQoHeads, numKvHeads, headDim, qStrideN, qStrideH, kvStrideN, kvStrideH, smScale);
  }

  allocPinned(bytes: number): number {
    const p = this.native.allocPinned(bytes);
    if (!p) throw new Error(`allocPinned failed for size ${bytes}`);
    return p;
  }

  freePinned(ptr: Tensor): void {
    this.native.freePinned(ptr.data);
  }

  writePinned(dst: Tensor, src: Buffer, size?: number): void {
    this.native.writePinned(dst.data, src, size ?? src.length);
  }

  batchDecodePlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, indptrH: Tensor, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, enableCudaGraph: boolean): void {
    this.native.batchDecodePlan(this.ctx, ptr(floatWs), floatWsSize, ptr(intWs), ptr(pinnedIntWs), intWsSize, ptr(planInfo), ptr(indptrH), batchSize, numQoHeads, numKvHeads, headDim, pageSize, enableCudaGraph);
  }

  batchDecodeRun(q: Tensor, o: Tensor, kData: Tensor, vData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, smScale: number): void {
    this.native.batchDecodeRun(this.ctx, ptr(q), ptr(o), ptr(kData), ptr(vData), ptr(indices), ptr(indptrD), ptr(lastPageLen), ptr(floatWs), ptr(intWs), ptr(planInfo), batchSize, numQoHeads, numKvHeads, headDim, pageSize, smScale);
  }

  batchPrefillPagedPlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, qoIndptrH: Tensor, pagedKvIndptrH: Tensor, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, maskMode: number): void {
    this.native.batchPrefillPagedPlan(this.ctx, ptr(floatWs), floatWsSize, ptr(intWs), ptr(pinnedIntWs), intWsSize, ptr(planInfo), ptr(qoIndptrH), ptr(pagedKvIndptrH), totalQoRows, batchSize, numQoHeads, numKvHeads, headDim, pageSize, maskMode);
  }

  batchPrefillPagedRun(q: Tensor, o: Tensor, kData: Tensor, vData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, floatWs: Tensor, intWs: Tensor, qIndptrD: Tensor, planInfo: Tensor, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, qStrideN: number, qStrideH: number, maskMode: number, smScale: number): void {
    this.native.batchPrefillPagedRun(this.ctx, ptr(q), ptr(o), ptr(kData), ptr(vData), ptr(indices), ptr(indptrD), ptr(lastPageLen), ptr(floatWs), ptr(intWs), ptr(qIndptrD), ptr(planInfo), totalQoRows, batchSize, numQoHeads, numKvHeads, headDim, pageSize, qStrideN, qStrideH, maskMode, smScale);
  }

  mmapOpen(filePath: string): number {
    const p = this.native.mmapOpen(filePath);
    if (!p) throw new Error(`glm_mmap_open failed for ${filePath}`);
    return p;
  }

  mmapLoad(gpuDst: Tensor, mmapPtr: number, offset: number, nbytes: number): void {
    this.native.mmapLoad(this.ctx, ptr(gpuDst), mmapPtr, offset, nbytes);
  }

  mmapClose(mmapPtr: number, size: number): void {
    this.native.mmapClose(mmapPtr, size);
  }

  graphBeginCapture(): void {
    this.native.graphBeginCapture(this.ctx);
  }

  graphEndCapture(): number {
    return this.native.graphEndCapture(this.ctx);
  }

  graphInstantiate(graph: number): number {
    return this.native.graphInstantiate(graph);
  }

  graphLaunch(graphExec: number): void {
    this.native.graphLaunch(graphExec, this.ctx);
  }

  graphExecUpdate(graphExec: number, graph: number): number {
    return this.native.graphExecUpdate(graphExec, graph);
  }

  graphDestroy(graph: number): void {
    this.native.graphDestroy(graph);
  }

  graphExecDestroy(graphExec: number): void {
    this.native.graphExecDestroy(graphExec);
  }

  fp8LinearDecode(bf16Out: Tensor, bf16Input: Tensor, fp8Weight: Tensor, weightScale: Tensor, m: number, n: number, k: number): void {
    this.native.fp8LinearDecode(this.ctx, ptr(bf16Out), ptr(bf16Input), ptr(fp8Weight), ptr(weightScale), m, n, k);
  }

  gdnRecurrentStep(output: Tensor, state: Tensor, qkv: Tensor, aRaw: Tensor, bRaw: Tensor, aLog: Tensor, dtBias: Tensor, numHeads: number, dK: number, dV: number, batchSize: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void {
    this.native.gdnRecurrentStep(this.ctx, ptr(output), ptr(state), ptr(qkv), ptr(aRaw), ptr(bRaw), ptr(aLog), ptr(dtBias), numHeads, dK, dV, batchSize, stateStride, qkvChStride, qkvSeqStride);
  }

  gdnPrefill(output: Tensor, state: Tensor, qkv: Tensor, aRaw: Tensor, bRaw: Tensor, aLog: Tensor, dtBias: Tensor, cuSeqlens: Tensor, totalSeqLen: number, numHeads: number, dK: number, dV: number, batchSize: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void {
    this.native.gdnPrefill(this.ctx, ptr(output), ptr(state), ptr(qkv), ptr(aRaw), ptr(bRaw), ptr(aLog), ptr(dtBias), ptr(cuSeqlens), totalSeqLen, numHeads, dK, dV, batchSize, stateStride, qkvChStride, qkvSeqStride);
  }

  causalConv1d(output: Tensor, convState: Tensor, input: Tensor, weight: Tensor, cuSeqlens: Tensor, convDim: number, totalSeqLen: number, kernelSize: number, batchSize: number, convStateStride: number, chStride: number, seqStride: number): void {
    this.native.causalConv1d(this.ctx, ptr(output), ptr(convState), ptr(input), ptr(weight), ptr(cuSeqlens), convDim, totalSeqLen, kernelSize, batchSize, convStateStride, chStride, seqStride);
  }

  causalConv1dUpdate(output: Tensor, convState: Tensor, input: Tensor, weight: Tensor, convDim: number, kernelSize: number, batchSize: number, convStateStride: number): void {
    this.native.causalConv1dUpdate(this.ctx, ptr(output), ptr(convState), ptr(input), ptr(weight), convDim, kernelSize, batchSize, convStateStride);
  }

  rmsnormGated(output: Tensor, input: Tensor, gate: Tensor, weight: Tensor, eps: number, dim: number, batch: number): void {
    this.native.rmsnormGated(this.ctx, ptr(output), ptr(input), ptr(gate), ptr(weight), eps, dim, batch);
  }

  sigmoid(out: Tensor, input: Tensor, n: number): void {
    this.native.sigmoid(this.ctx, ptr(out), ptr(input), n);
  }

  mul(out: Tensor, a: Tensor, b: Tensor, n: number): void {
    this.native.mul(this.ctx, ptr(out), ptr(a), ptr(b), n);
  }

  gateSigmoidMul(attnOut: Tensor, gateInterleaved: Tensor, batchSeq: number, numHeads: number, headDim: number): void {
    this.native.gateSigmoidMul(this.ctx, ptr(attnOut), ptr(gateInterleaved), batchSeq, numHeads, headDim);
  }

  sampleBatch(outTokens: Tensor, topkVals: Tensor, topkIdxs: Tensor, workspace: Tensor, logits: Tensor, penaltyTokens: Tensor, penaltyOffsets: Tensor, vocabSize: number, batchSize: number, temperatures: Tensor, repPenalties: Tensor, presPenalties: Tensor, topKs: Tensor, topPs: Tensor, randomVals: Tensor, maxEffectiveK: number): void {
    this.native.sampleBatch(this.ctx, ptr(outTokens), ptr(topkVals), ptr(topkIdxs), ptr(workspace), ptr(logits), ptr(penaltyTokens), ptr(penaltyOffsets), vocabSize, batchSize, ptr(temperatures), ptr(repPenalties), ptr(presPenalties), ptr(topKs), ptr(topPs), ptr(randomVals), maxEffectiveK);
  }

  memcpy2d(dst: number, dpitch: number, src: number, spitch: number, width: number, height: number, kind: number): void {
    this.native.memcpy2d(this.ctx, dst, dpitch, src, spitch, width, height, kind);
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
export const SAMPLING_MAX_TOPK = 256;
export const SAMPLING_BLOCK_SIZE = 256;
export const FLASH_TMP_SIZE = 32 * 1024 * 1024;
export const BATCH_FLOAT_WS_SIZE = 128 * 1024 * 1024;
export const BATCH_INT_WS_SIZE = 8 * 1024 * 1024;
export const BATCH_PINNED_INT_WS_SIZE = 8 * 1024 * 1024;
export const PAGE_SIZE = 16;

export const MEMCPY_H2H = 0;
export const MEMCPY_H2D = 1;
export const MEMCPY_D2H = 2;
export const MEMCPY_D2D = 3;

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
