import path from "node:path";

interface NativeAddon {
  init(deviceId: number): number;
  free(ctx: number): void;
  alloc(ctx: number, size: number): number;
  freeBuf(ctx: number, ptr: number): void;
  h2d(ctx: number, dst: number, src: Buffer, size: number): void;
  d2h(ctx: number, dst: Buffer, src: number, size: number): void;
  rmsnorm(ctx: number, out: number, input: number, weight: number, eps: number, dim: number, batch: number): void;
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
  kvCacheWrite(ctx: number, srcK: number, srcV: number, dstK: number, dstV: number, slotMapping: number, batchSize: number, nKv: number, hd: number, pageSize: number): void;
  synchronize(ctx: number): void;
  flashPrefill(ctx: number, q: number, k: number, v: number, o: number, tmp: number, qoLen: number, kvLen: number, numQoHeads: number, numKvHeads: number, headDim: number, qStrideN: number, qStrideH: number, kvStrideN: number, kvStrideH: number, vStrideN: number, vStrideH: number, maskMode: number, kvLayout: number, smScale: number): void;
  flashDecode(ctx: number, q: number, k: number, v: number, o: number, tmp: number, kvLen: number, numQoHeads: number, numKvHeads: number, headDim: number, qStrideN: number, qStrideH: number, kvStrideN: number, kvStrideH: number, smScale: number): void;
  allocPinned(bytes: number): number;
  freePinned(ptr: number): void;
  writePinned(dst: number, src: Buffer, size: number): void;
  batchDecodePlan(ctx: number, floatWs: number, floatWsSize: number, intWs: number, pinnedIntWs: number, intWsSize: number, planInfo: number, indptrH: number, batchSize: number, numQoHeads: number, numKvHeads: number, pageSize: number, enableCudaGraph: boolean): void;
  batchDecodeRun(ctx: number, q: number, o: number, kData: number, vData: number, indices: number, indptrD: number, lastPageLen: number, floatWs: number, intWs: number, planInfo: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, smScale: number): void;
  batchPrefillRaggedPlan(ctx: number, floatWs: number, floatWsSize: number, intWs: number, pinnedIntWs: number, intWsSize: number, planInfo: number, qoIndptrH: number, kvIndptrH: number, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, maskMode: number): void;
  batchPrefillRaggedRun(ctx: number, q: number, k: number, v: number, o: number, floatWs: number, intWs: number, qIndptrD: number, kvIndptrD: number, planInfo: number, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, qStrideN: number, qStrideH: number, kvStrideN: number, kvStrideH: number, maskMode: number, smScale: number): void;
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
}

export class GlmOps {
  private native: NativeAddon;
  ctx: number;
  device: number;

  constructor(deviceId: number = 0, libPath?: string) {
    const p = libPath ?? path.join(__dirname, "..", "build", "Release", "glm.node");
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
    const ptr = this.native.alloc(this.ctx, size);
    if (!ptr) throw new Error(`glm_alloc failed for size ${size}`);
    return ptr;
  }

  freeBuf(ptr: number): void {
    this.native.freeBuf(this.ctx, ptr);
  }

  h2d(gpuPtr: number, cpuData: Buffer, size?: number): void {
    this.native.h2d(this.ctx, gpuPtr, cpuData, size ?? cpuData.length);
  }

  d2h(cpuBuf: Buffer, gpuPtr: number, size?: number): void {
    this.native.d2h(this.ctx, cpuBuf, gpuPtr, size ?? cpuBuf.length);
  }

  synchronize(): void {
    this.native.synchronize(this.ctx);
  }

  rmsnorm(out: number, input: number, weight: number, eps: number, dim: number, batch: number): void {
    this.native.rmsnorm(this.ctx, out, input, weight, eps, dim, batch);
  }

  siluAndMul(out: number, gate: number, up: number, intermediate: number, batch: number): void {
    this.native.siluAndMul(this.ctx, out, gate, up, intermediate, batch);
  }

  linear(out: number, input: number, weight: number, batch: number, n: number, k: number): void {
    this.native.linear(this.ctx, out, input, weight, batch, n, k);
  }

  embedding(out: number, table: number, ids: number, hidden: number, seqLen: number): void {
    this.native.embedding(this.ctx, out, table, ids, hidden, seqLen);
  }

  layernorm(out: number, input: number, weight: number, bias: number, eps: number, dim: number, batch: number): void {
    this.native.layernorm(this.ctx, out, input, weight, bias, eps, dim, batch);
  }

  softmax(out: number, input: number, mask: number, dim: number, batch: number): void {
    this.native.softmax(this.ctx, out, input, mask, dim, batch);
  }

  causalMask(out: number, seqLen: number): void {
    this.native.causalMask(this.ctx, out, seqLen);
  }

  add(out: number, a: number, b: number, n: number): void {
    this.native.add(this.ctx, out, a, b, n);
  }

  arange(out: number, start: number, step: number, count: number): void {
    this.native.arange(this.ctx, out, start, step, count);
  }

  argmax(outIndex: number, input: number, dim: number, batch: number = 1): void {
    this.native.argmax(this.ctx, outIndex, input, dim, batch);
  }

  memcpy(dst: number, src: number, bytes: number): void {
    this.native.memcpy(this.ctx, dst, src, bytes);
  }

  kvCacheWrite(srcK: number, srcV: number, dstK: number, dstV: number, slotMapping: number, batchSize: number, nKv: number, hd: number, pageSize: number): void {
    this.native.kvCacheWrite(this.ctx, srcK, srcV, dstK, dstV, slotMapping, batchSize, nKv, hd, pageSize);
  }

  rotaryEmbedding(cosOut: number, sinOut: number, invFreq: number, positionIds: number, dimHalf: number, batch: number, seqLen: number): void {
    this.native.rotaryEmbedding(this.ctx, cosOut, sinOut, invFreq, positionIds, dimHalf, batch, seqLen);
  }

  applyRotaryPosEmb(out: number, x: number, cos: number, sin: number, ropeDim: number, nHeads: number, seqLen: number, batch: number, unsqueezeDim: number): void {
    this.native.applyRotaryPosEmb(this.ctx, out, x, cos, sin, ropeDim, nHeads, seqLen, batch, unsqueezeDim);
  }

  expandDim1(out: number, input: number, dim1Out: number, dim1In: number, seqLen: number, headDim: number, batch: number): void {
    this.native.expandDim1(this.ctx, out, input, dim1Out, dim1In, seqLen, headDim, batch);
  }

  expandDim1Strided(out: number, input: number, dim1Out: number, dim1In: number, seqLen: number, headDim: number, batch: number, headStride: number): void {
    this.native.expandDim1Strided(this.ctx, out, input, dim1Out, dim1In, seqLen, headDim, batch, headStride);
  }

  indexSelect(out: number, src: number, indices: number, dim: number, k: number): void {
    this.native.indexSelect(this.ctx, out, src, indices, dim, k);
  }

  transpose4d(out: number, input: number, d0: number, d1: number, d2: number, d3: number, p0: number, p1: number, p2: number, p3: number): void {
    this.native.transpose4d(this.ctx, out, input, d0, d1, d2, d3, p0, p1, p2, p3);
  }

  bmm(C: number, A: number, B: number, alpha: number, beta: number, batch: number, M: number, N: number, K: number, transB: number): void {
    this.native.bmm(this.ctx, C, A, B, alpha, beta, batch, M, N, K, transB);
  }

  flashPrefill(q: number, k: number, v: number, o: number, tmp: number, qoLen: number, kvLen: number, numQoHeads: number, numKvHeads: number, headDim: number, qStrideN: number, qStrideH: number, kvStrideN: number, kvStrideH: number, vStrideN: number, vStrideH: number, maskMode: number, kvLayout: number, smScale: number): void {
    this.native.flashPrefill(this.ctx, q, k, v, o, tmp, qoLen, kvLen, numQoHeads, numKvHeads, headDim, qStrideN, qStrideH, kvStrideN, kvStrideH, vStrideN, vStrideH, maskMode, kvLayout, smScale);
  }

  flashDecode(q: number, k: number, v: number, o: number, tmp: number, kvLen: number, numQoHeads: number, numKvHeads: number, headDim: number, qStrideN: number, qStrideH: number, kvStrideN: number, kvStrideH: number, smScale: number): void {
    this.native.flashDecode(this.ctx, q, k, v, o, tmp, kvLen, numQoHeads, numKvHeads, headDim, qStrideN, qStrideH, kvStrideN, kvStrideH, smScale);
  }

  allocPinned(bytes: number): number {
    const ptr = this.native.allocPinned(bytes);
    if (!ptr) throw new Error(`allocPinned failed for size ${bytes}`);
    return ptr;
  }

  freePinned(ptr: number): void {
    this.native.freePinned(ptr);
  }

  writePinned(dst: number, src: Buffer, size?: number): void {
    this.native.writePinned(dst, src, size ?? src.length);
  }

  batchDecodePlan(floatWs: number, floatWsSize: number, intWs: number, pinnedIntWs: number, intWsSize: number, planInfo: number, indptrH: number, batchSize: number, numQoHeads: number, numKvHeads: number, pageSize: number, enableCudaGraph: boolean): void {
    this.native.batchDecodePlan(this.ctx, floatWs, floatWsSize, intWs, pinnedIntWs, intWsSize, planInfo, indptrH, batchSize, numQoHeads, numKvHeads, pageSize, enableCudaGraph);
  }

  batchDecodeRun(q: number, o: number, kData: number, vData: number, indices: number, indptrD: number, lastPageLen: number, floatWs: number, intWs: number, planInfo: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, smScale: number): void {
    this.native.batchDecodeRun(this.ctx, q, o, kData, vData, indices, indptrD, lastPageLen, floatWs, intWs, planInfo, batchSize, numQoHeads, numKvHeads, headDim, pageSize, smScale);
  }

  batchPrefillRaggedPlan(floatWs: number, floatWsSize: number, intWs: number, pinnedIntWs: number, intWsSize: number, planInfo: number, qoIndptrH: number, kvIndptrH: number, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, maskMode: number): void {
    this.native.batchPrefillRaggedPlan(this.ctx, floatWs, floatWsSize, intWs, pinnedIntWs, intWsSize, planInfo, qoIndptrH, kvIndptrH, totalQoRows, batchSize, numQoHeads, numKvHeads, headDim, maskMode);
  }

  batchPrefillRaggedRun(q: number, k: number, v: number, o: number, floatWs: number, intWs: number, qIndptrD: number, kvIndptrD: number, planInfo: number, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, qStrideN: number, qStrideH: number, kvStrideN: number, kvStrideH: number, maskMode: number, smScale: number): void {
    this.native.batchPrefillRaggedRun(this.ctx, q, k, v, o, floatWs, intWs, qIndptrD, kvIndptrD, planInfo, totalQoRows, batchSize, numQoHeads, numKvHeads, headDim, qStrideN, qStrideH, kvStrideN, kvStrideH, maskMode, smScale);
  }

  batchPrefillPagedPlan(floatWs: number, floatWsSize: number, intWs: number, pinnedIntWs: number, intWsSize: number, planInfo: number, qoIndptrH: number, pagedKvIndptrH: number, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, maskMode: number): void {
    this.native.batchPrefillPagedPlan(this.ctx, floatWs, floatWsSize, intWs, pinnedIntWs, intWsSize, planInfo, qoIndptrH, pagedKvIndptrH, totalQoRows, batchSize, numQoHeads, numKvHeads, headDim, pageSize, maskMode);
  }

  batchPrefillPagedRun(q: number, o: number, kData: number, vData: number, indices: number, indptrD: number, lastPageLen: number, floatWs: number, intWs: number, qIndptrD: number, planInfo: number, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, qStrideN: number, qStrideH: number, maskMode: number, smScale: number): void {
    this.native.batchPrefillPagedRun(this.ctx, q, o, kData, vData, indices, indptrD, lastPageLen, floatWs, intWs, qIndptrD, planInfo, totalQoRows, batchSize, numQoHeads, numKvHeads, headDim, pageSize, qStrideN, qStrideH, maskMode, smScale);
  }

  mmapOpen(filePath: string): number {
    const ptr = this.native.mmapOpen(filePath);
    if (!ptr) throw new Error(`glm_mmap_open failed for ${filePath}`);
    return ptr;
  }

  mmapLoad(gpuDst: number, mmapPtr: number, offset: number, nbytes: number): void {
    this.native.mmapLoad(this.ctx, gpuDst, mmapPtr, offset, nbytes);
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
export const FLASH_TMP_SIZE = 32 * 1024 * 1024;
export const BATCH_FLOAT_WS_SIZE = 128 * 1024 * 1024;
export const BATCH_INT_WS_SIZE = 8 * 1024 * 1024;
export const BATCH_PINNED_INT_WS_SIZE = 8 * 1024 * 1024;
export const PAGE_SIZE = 16;
