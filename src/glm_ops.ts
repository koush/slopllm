import fs from "node:fs";
import path from "node:path";
import { DeviceOps, TensorParallelism } from "./device_ops";
import type { SamplingParams } from "./chat_model";
import { MemcpyKind, Tensor } from "./tensor";
import { SafeTensorFile } from "./safetensors";
import type { WorkspaceBase } from "./workspace";

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

let nativeAddon: NativeAddon | null = null;

export function getNativeAddon(): NativeAddon {
  if (!nativeAddon) {
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
  fill(ctx: number, out: number, value: number, n: number): void;
  rotaryEmbedding(ctx: number, cosOut: number, sinOut: number, invFreq: number, positionIds: number, dimHalf: number, batch: number, seqLen: number): void;
  indexSelect(ctx: number, out: number, src: number, indices: number, dim: number, k: number): void;
  gather(ctx: number, out: number, input: number, indices: number, k: number, inDim: number, batch: number, elemSize: number): void;
  arange(ctx: number, out: number, start: number, step: number, count: number): void;
  max(ctx: number, outValues: number, outIndices: number, input: number, dim: number, batch: number, offset: number): void;
  memcpy(ctx: number, dst: number, src: number, bytes: number, kind: number): void;
  kvCacheWrite(ctx: number, srcK: number, srcV: number, dstK: number, dstV: number, slotMapping: number, batchSize: number, nKv: number, hd: number, pageSize: number, srcKTokenStride: number, srcKHeadStride: number, srcVTokenStride: number, srcVHeadStride: number): void;
  synchronize(ctx: number): void;
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
  ncclGroupStart(): void;
  ncclGroupEnd(): void;
  ncclCommInitRank(deviceId: number, rank: number, worldSize: number, uniqueId: number): number;
  ncclCommInitAll(deviceIds: number[]): number[];
  ncclCommDestroy(comm: number): void;
  ncclAllReduce(comm: number, ctx: number, sendbuff: number, recvbuff: number, count: number, datatype: number, op: number): void;
  ncclAllGather(comm: number, ctx: number, sendbuff: number, recvbuff: number, count: number, datatype: number): void;
}

export class GlmTensor extends Tensor {
  constructor(workspace: WorkspaceBase, public readonly glm: GlmOps, data: number, allocSize: number, shape: number[], type: string, name: string | undefined, pinned: boolean, view: GlmTensor | undefined) {
    super(workspace, data, allocSize, shape, type, name, pinned, view);
  }

  free(): void {
    if (this.data !== 0) {
      if (this.pinned) {
        getNativeAddon().freePinned(this.data);
      } else {
        getNativeAddon().freeBuf(this.glm.ctx, this.data);
      }
      (this as { data: number }).data = 0;
    }
  }

  h2d(data: Buffer, size?: number): void {
    getNativeAddon().h2d(this.glm.ctx, this.data, data, size ?? data.length);
  }

  d2h(buf: Buffer, size?: number): void {
    getNativeAddon().d2h(this.glm.ctx, buf, this.data, size ?? buf.length);
  }

  linear(weight: Tensor, batch: number): Tensor {
    super.linear(weight, batch);
    const n = weight.shape[0];
    const k = weight.shape[1];
    const outShape = [batch, n];
    const out = this.workspace.alloc(outShape, this.type);
    if (weight.type === "F8_E4M3") {
      const scale = weight.workspace.tensors.get(weight.name! + "_scale_inv")!;
      getNativeAddon().fp8LinearDecode(this.glm.ctx, out.data, this.data, weight.data, scale.data, batch, n, k);
    } else {
      getNativeAddon().linear(this.glm.ctx, out.data, this.data, weight.data, batch, n, k);
    }
    return out;
  }

  rmsnorm(weight: Tensor, eps: number, dim: number, batch: number): Tensor {
    super.rmsnorm(weight, eps, dim, batch);
    const out = this.workspace.alloc([batch, dim], this.type);
    getNativeAddon().rmsnorm(this.glm.ctx, out.data, this.data, weight.data, eps, dim, batch);
    return out;
  }

  fusedAddRmsnorm(input: Tensor, weight: Tensor, eps: number, dim: number, batch: number): { normed: Tensor, residual: Tensor } {
    super.fusedAddRmsnorm(input, weight, eps, dim, batch);
    const normed = this.workspace.alloc([batch, dim], this.type);
    const residual = this.workspace.alloc([batch, dim], this.type);
    getNativeAddon().fusedAddRmsnorm(this.glm.ctx, normed.data, residual.data, this.data, input.data, weight.data, eps, dim, batch);
    return { normed, residual };
  }

  fusedNormRope(weight: Tensor, cos: Tensor, sin: Tensor, eps: number, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, inStride?: number): Tensor {
    super.fusedNormRope(weight, cos, sin, eps, ropeDim, headDim, nHeads, seqLen, batch, inStride);
    const out = this.workspace.alloc([batch, nHeads, seqLen, headDim], this.type);
    getNativeAddon().fusedNormRope(this.glm.ctx, out.data, this.data, weight.data, cos.data, sin.data, eps, ropeDim, headDim, nHeads, seqLen, batch, inStride ?? headDim);
    return out;
  }

  embedding(ids: Tensor, hidden: number, seqLen: number): Tensor {
    super.embedding(ids, hidden, seqLen);
    const out = ids.workspace.alloc([seqLen, hidden], this.type);
    getNativeAddon().embedding(this.glm.ctx, out.data, this.data, ids.data, hidden, seqLen);
    return out;
  }

  siluAndMul(gate: Tensor, up: Tensor, intermediate: number, batch: number): Tensor {
    super.siluAndMul(gate, up, intermediate, batch);
    const out = this.workspace.alloc([batch, intermediate], this.type);
    getNativeAddon().siluAndMul(this.glm.ctx, out.data, gate.data, up.data, intermediate, batch);
    return out;
  }

  arange(start: number, step: number, count: number): void {
    super.arange(start, step, count);
    getNativeAddon().arange(this.glm.ctx, this.data, start, step, count);
  }

  argmax(): Tensor {
    super.argmax();
    const { indices, values} = this.max();
    values[Symbol.dispose]();
    return indices;
  }

  max(offset: number = 0): { values: Tensor, indices: Tensor } {
    super.max(offset);
    const batch = this.shape[0];
    const dim = this.shape[1];
    const values = this.workspace.alloc([batch], this.type);
    const indices = this.workspace.alloc([batch], "I32");
    getNativeAddon().max(this.glm.ctx, values.data, indices.data, this.data, dim, batch, offset);
    return { values, indices };
  }

  indexSelect(indices: Tensor, dim: number, batch: number): Tensor {
    super.indexSelect(indices, dim, batch);
    const out = this.workspace.alloc([batch, dim], this.type);
    getNativeAddon().indexSelect(this.glm.ctx, out.data, this.data, indices.data, dim, batch);
    return out;
  }

  gather(indices: Tensor, k: number, inDim: number, batch: number): Tensor {
    super.gather(indices, k, inDim, batch);
    const out = this.workspace.alloc([batch, k], this.type);
    const elemSize = SafeTensorFile.dtypeBytes(this.type);
    getNativeAddon().gather(this.glm.ctx, out.data, this.data, indices.data, k, inDim, batch, elemSize);
    return out;
  }

  gdnRecurrentStep(state: Tensor, qkv: Tensor, aRaw: Tensor, bRaw: Tensor, aLog: Tensor, dtBias: Tensor, numHeads: number, dK: number, dV: number, batchSize: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void {
    super.gdnRecurrentStep(state, qkv, aRaw, bRaw, aLog, dtBias, numHeads, dK, dV, batchSize, stateStride, qkvChStride, qkvSeqStride);
    getNativeAddon().gdnRecurrentStep(this.glm.ctx, this.data, state.data, qkv.data, aRaw.data, bRaw.data, aLog.data, dtBias.data, numHeads, dK, dV, batchSize, stateStride, qkvChStride, qkvSeqStride);
  }

  gdnPrefill(state: Tensor, qkv: Tensor, aRaw: Tensor, bRaw: Tensor, aLog: Tensor, dtBias: Tensor, cuSeqlens: Tensor, totalSeqLen: number, numHeads: number, dK: number, dV: number, batchSize: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void {
    super.gdnPrefill(state, qkv, aRaw, bRaw, aLog, dtBias, cuSeqlens, totalSeqLen, numHeads, dK, dV, batchSize, stateStride, qkvChStride, qkvSeqStride);
    getNativeAddon().gdnPrefill(this.glm.ctx, this.data, state.data, qkv.data, aRaw.data, bRaw.data, aLog.data, dtBias.data, cuSeqlens.data, totalSeqLen, numHeads, dK, dV, batchSize, stateStride, qkvChStride, qkvSeqStride);
  }

  causalConv1d(convState: Tensor, input: Tensor, weight: Tensor, cuSeqlens: Tensor, convDim: number, totalSeqLen: number, kernelSize: number, batchSize: number, convStateStride: number, chStride: number, seqStride: number): void {
    super.causalConv1d(convState, input, weight, cuSeqlens, convDim, totalSeqLen, kernelSize, batchSize, convStateStride, chStride, seqStride);
    getNativeAddon().causalConv1d(this.glm.ctx, this.data, convState.data, input.data, weight.data, cuSeqlens.data, convDim, totalSeqLen, kernelSize, batchSize, convStateStride, chStride, seqStride);
  }

  causalConv1dUpdate(convState: Tensor, input: Tensor, weight: Tensor, convDim: number, kernelSize: number, batchSize: number, convStateStride: number): Tensor {
    super.causalConv1dUpdate(convState, input, weight, convDim, kernelSize, batchSize, convStateStride);
    const out = this.workspace.alloc([batchSize * convDim], this.type);
    getNativeAddon().causalConv1dUpdate(this.glm.ctx, out.data, convState.data, input.data, weight.data, convDim, kernelSize, batchSize, convStateStride);
    return out;
  }

  rmsnormGated(input: Tensor, gate: Tensor, weight: Tensor, eps: number, dim: number, batch: number): void {
    super.rmsnormGated(input, gate, weight, eps, dim, batch);
    getNativeAddon().rmsnormGated(this.glm.ctx, this.data, input.data, gate.data, weight.data, eps, dim, batch);
  }

  gateSigmoidMul(gate: Tensor, batchSeq: number, numHeads: number, headDim: number): void {
    super.gateSigmoidMul(gate, batchSeq, numHeads, headDim);
    getNativeAddon().gateSigmoidMul(this.glm.ctx, this.data, gate.data, batchSeq, numHeads, headDim);
  }

  fill(value: number, n: number): void {
    getNativeAddon().fill(this.glm.ctx, this.data, value, n);
  }

  mmapLoad(mmapPtr: number, offset: number, nbytes: number, _gdnQkvLayout?: import("./device_ops").GdnQkvLayout): void {
    getNativeAddon().mmapLoad(this.glm.ctx, this.data, mmapPtr, offset, nbytes);
  }

  writePinned(src: Buffer, size?: number): void {
    getNativeAddon().writePinned(this.data, src, size ?? src.length);
  }

  memcpy(src: Tensor, size?: number, kind?: MemcpyKind): void {
    if (!(src instanceof GlmTensor)) {
      throw new Error("GlmTensor.memcpy requires GlmTensor source");
    }
    const bytes = size ?? Math.min(this.allocSize, src.allocSize);
    const copyKind = kind ?? (src.pinned ? MemcpyKind.HostToDevice : MemcpyKind.DeviceToDevice);
    getNativeAddon().memcpy(this.glm.ctx, this.data, src.data, bytes, memcpyKindToNative(copyKind));
  }

  rotaryEmbedding(positionIds: Tensor, dimHalf: number, batch: number, seqLen: number): { cos: Tensor, sin: Tensor } {
    super.rotaryEmbedding(positionIds, dimHalf, batch, seqLen);
    const hd = dimHalf * 2;
    const cos = positionIds.workspace.alloc([batch, seqLen, hd], this.type);
    const sin = positionIds.workspace.alloc([batch, seqLen, hd], this.type);
    getNativeAddon().rotaryEmbedding(this.glm.ctx, cos.data, sin.data, this.data, positionIds.data, dimHalf, batch, seqLen);
    return { cos, sin };
  }

  protected doSampleBatch(outTokens: Tensor, topkVals: Tensor, topkIdxs: Tensor, workspace: Tensor, logits: Tensor, penaltyTokens: Tensor, penaltyOffsets: Tensor, vocabSize: number, batchSize: number, temperatures: Tensor, repPenalties: Tensor, presPenalties: Tensor, topKs: Tensor, topPs: Tensor, randomVals: Tensor, maxEffectiveK: number): void {
    getNativeAddon().sampleBatch(this.glm.ctx, outTokens.data, topkVals.data, topkIdxs.data, workspace.data, logits.data, penaltyTokens.data, penaltyOffsets.data, vocabSize, batchSize, temperatures.data, repPenalties.data, presPenalties.data, topKs.data, topPs.data, randomVals.data, maxEffectiveK);
  }
}

export class GlmOps implements DeviceOps {
  ctx: number;
  device: number;

  constructor(deviceId: number = 0, libPath?: string) {
    if (libPath) {
      nativeAddon = require(libPath) as NativeAddon;
    }
    const native = getNativeAddon();
    this.ctx = native.init(deviceId);
    if (!this.ctx) {
      throw new Error(`glm_init failed on device ${deviceId}`);
    }
    this.device = deviceId;
  }

  free(): void {
    getNativeAddon().free(this.ctx);
  }

  alloc(size: number): number {
    const p = getNativeAddon().alloc(this.ctx, size);
    if (!p) throw new Error(`glm_alloc failed for size ${size}`);
    return p;
  }

  allocPinned(bytes: number): number {
    const p = getNativeAddon().allocPinned(bytes);
    if (!p) throw new Error(`allocPinned failed for size ${bytes}`);
    return p;
  }

  newTensor(workspace: WorkspaceBase, shape: number[], type: string, pinned: boolean, name?: string, _parallelism?: TensorParallelism): GlmTensor {
    const size = Tensor.byteCount(shape, type);
    const data = pinned ? this.allocPinned(size) : this.alloc(size);
    return new GlmTensor(workspace, this, data, size, shape, type, name, pinned, undefined);
  }

  wrapTensor(workspace: WorkspaceBase, data: number, allocSize: number, shape: number[], type: string, pinned: boolean, view: GlmTensor | undefined): Tensor {
    return new GlmTensor(workspace, this, data, allocSize, shape, type, undefined, pinned, view);
  }

  freeBuf(ptr: Tensor): void {
    getNativeAddon().freeBuf(this.ctx, ptr.data);
  }

  h2d(dst: Tensor, cpuData: Buffer, size?: number): void {
    getNativeAddon().h2d(this.ctx, dst.data, cpuData, size ?? cpuData.length);
  }

  d2h(cpuBuf: Buffer, src: Tensor, size?: number): void {
    getNativeAddon().d2h(this.ctx, cpuBuf, src.data, size ?? cpuBuf.length);
  }

  synchronize(): void {
    getNativeAddon().synchronize(this.ctx);
  }

  fill(out: Tensor, value: number, n: number): void {
    getNativeAddon().fill(this.ctx, ptr(out), value, n);
  }

  arange(out: Tensor, start: number, step: number, count: number): void {
    getNativeAddon().arange(this.ctx, ptr(out), start, step, count);
  }

  memcpy(dst: number, src: number, bytes: number, kind: MemcpyKind): void {
    getNativeAddon().memcpy(this.ctx, dst, src, bytes, memcpyKindToNative(kind));
  }

  kvCacheWrite(srcK: Tensor, srcV: Tensor, dstK: Tensor, dstV: Tensor, slotMapping: Tensor, batchSize: number, nKv: number, hd: number, pageSize: number, srcKTokenStride: number, srcKHeadStride: number, srcVTokenStride: number, srcVHeadStride: number): void {
    getNativeAddon().kvCacheWrite(this.ctx, ptr(srcK), ptr(srcV), ptr(dstK), ptr(dstV), ptr(slotMapping), batchSize, nKv, hd, pageSize, srcKTokenStride, srcKHeadStride, srcVTokenStride, srcVHeadStride);
  }

  freePinned(ptr: Tensor): void {
    getNativeAddon().freePinned(ptr.data);
  }

  writePinned(dst: Tensor, src: Buffer, size?: number): void {
    getNativeAddon().writePinned(dst.data, src, size ?? src.length);
  }

  batchDecodePlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, indptrH: Tensor, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, enableCudaGraph: boolean): void {
    getNativeAddon().batchDecodePlan(this.ctx, ptr(floatWs), floatWsSize, ptr(intWs), ptr(pinnedIntWs), intWsSize, ptr(planInfo), ptr(indptrH), batchSize, numQoHeads, numKvHeads, headDim, pageSize, enableCudaGraph);
  }

  batchDecodeRun(q: Tensor, o: Tensor, kData: Tensor, vData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, smScale: number): void {
    getNativeAddon().batchDecodeRun(this.ctx, ptr(q), ptr(o), ptr(kData), ptr(vData), ptr(indices), ptr(indptrD), ptr(lastPageLen), ptr(floatWs), ptr(intWs), ptr(planInfo), batchSize, numQoHeads, numKvHeads, headDim, pageSize, smScale);
  }

  batchPrefillPagedPlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, qoIndptrH: Tensor, pagedKvIndptrH: Tensor, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, maskMode: number): void {
    getNativeAddon().batchPrefillPagedPlan(this.ctx, ptr(floatWs), floatWsSize, ptr(intWs), ptr(pinnedIntWs), intWsSize, ptr(planInfo), ptr(qoIndptrH), ptr(pagedKvIndptrH), totalQoRows, batchSize, numQoHeads, numKvHeads, headDim, pageSize, maskMode);
  }

  batchPrefillPagedRun(q: Tensor, o: Tensor, kData: Tensor, vData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, floatWs: Tensor, intWs: Tensor, qIndptrD: Tensor, planInfo: Tensor, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, qStrideN: number, qStrideH: number, maskMode: number, smScale: number): void {
    getNativeAddon().batchPrefillPagedRun(this.ctx, ptr(q), ptr(o), ptr(kData), ptr(vData), ptr(indices), ptr(indptrD), ptr(lastPageLen), ptr(floatWs), ptr(intWs), ptr(qIndptrD), ptr(planInfo), totalQoRows, batchSize, numQoHeads, numKvHeads, headDim, pageSize, qStrideN, qStrideH, maskMode, smScale);
  }

  mmapLoad(gpuDst: Tensor, mmapPtr: number, offset: number, nbytes: number): void {
    getNativeAddon().mmapLoad(this.ctx, ptr(gpuDst), mmapPtr, offset, nbytes);
  }

  graphBeginCapture(): void {
    getNativeAddon().graphBeginCapture(this.ctx);
  }

  graphEndCapture(): number {
    const graph = getNativeAddon().graphEndCapture(this.ctx);
    if (!graph) throw new Error("CUDA graph capture failed");
    return graph;
  }

  graphInstantiate(graph: number): number {
    const exec = getNativeAddon().graphInstantiate(graph);
    if (!exec) throw new Error("CUDA graph instantiation failed");
    return exec;
  }

  graphLaunch(graphExec: number): void {
    getNativeAddon().graphLaunch(graphExec, this.ctx);
  }

  graphDestroy(graph: number): void {
    getNativeAddon().graphDestroy(graph);
  }

  graphExecDestroy(graphExec: number): void {
    getNativeAddon().graphExecDestroy(graphExec);
  }

  fp8LinearDecode(bf16Out: Tensor, bf16Input: Tensor, fp8Weight: Tensor, weightScale: Tensor, m: number, n: number, k: number): void {
    getNativeAddon().fp8LinearDecode(this.ctx, ptr(bf16Out), ptr(bf16Input), ptr(fp8Weight), ptr(weightScale), m, n, k);
  }

  gdnRecurrentStep(output: Tensor, state: Tensor, qkv: Tensor, aRaw: Tensor, bRaw: Tensor, aLog: Tensor, dtBias: Tensor, numHeads: number, dK: number, dV: number, batchSize: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void {
    getNativeAddon().gdnRecurrentStep(this.ctx, ptr(output), ptr(state), ptr(qkv), ptr(aRaw), ptr(bRaw), ptr(aLog), ptr(dtBias), numHeads, dK, dV, batchSize, stateStride, qkvChStride, qkvSeqStride);
  }

  gdnPrefill(output: Tensor, state: Tensor, qkv: Tensor, aRaw: Tensor, bRaw: Tensor, aLog: Tensor, dtBias: Tensor, cuSeqlens: Tensor, totalSeqLen: number, numHeads: number, dK: number, dV: number, batchSize: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void {
    getNativeAddon().gdnPrefill(this.ctx, ptr(output), ptr(state), ptr(qkv), ptr(aRaw), ptr(bRaw), ptr(aLog), ptr(dtBias), ptr(cuSeqlens), totalSeqLen, numHeads, dK, dV, batchSize, stateStride, qkvChStride, qkvSeqStride);
  }

  causalConv1d(output: Tensor, convState: Tensor, input: Tensor, weight: Tensor, cuSeqlens: Tensor, convDim: number, totalSeqLen: number, kernelSize: number, batchSize: number, convStateStride: number, chStride: number, seqStride: number): void {
    getNativeAddon().causalConv1d(this.ctx, ptr(output), ptr(convState), ptr(input), ptr(weight), ptr(cuSeqlens), convDim, totalSeqLen, kernelSize, batchSize, convStateStride, chStride, seqStride);
  }

  causalConv1dUpdate(output: Tensor, convState: Tensor, input: Tensor, weight: Tensor, convDim: number, kernelSize: number, batchSize: number, convStateStride: number): void {
    getNativeAddon().causalConv1dUpdate(this.ctx, ptr(output), ptr(convState), ptr(input), ptr(weight), convDim, kernelSize, batchSize, convStateStride);
  }

  rmsnormGated(output: Tensor, input: Tensor, gate: Tensor, weight: Tensor, eps: number, dim: number, batch: number): void {
    getNativeAddon().rmsnormGated(this.ctx, ptr(output), ptr(input), ptr(gate), ptr(weight), eps, dim, batch);
  }

  gateSigmoidMul(attnOut: Tensor, gateInterleaved: Tensor, batchSeq: number, numHeads: number, headDim: number): void {
    getNativeAddon().gateSigmoidMul(this.ctx, ptr(attnOut), ptr(gateInterleaved), batchSeq, numHeads, headDim);
  }

  sampleBatch(outTokens: Tensor, topkVals: Tensor, topkIdxs: Tensor, workspace: Tensor, logits: Tensor, penaltyTokens: Tensor, penaltyOffsets: Tensor, vocabSize: number, batchSize: number, temperatures: Tensor, repPenalties: Tensor, presPenalties: Tensor, topKs: Tensor, topPs: Tensor, randomVals: Tensor, maxEffectiveK: number): void {
    getNativeAddon().sampleBatch(this.ctx, ptr(outTokens), ptr(topkVals), ptr(topkIdxs), ptr(workspace), ptr(logits), ptr(penaltyTokens), ptr(penaltyOffsets), vocabSize, batchSize, ptr(temperatures), ptr(repPenalties), ptr(presPenalties), ptr(topKs), ptr(topPs), ptr(randomVals), maxEffectiveK);
  }

  memcpy2d(dst: number, dpitch: number, src: number, spitch: number, width: number, height: number, kind: MemcpyKind): void {
    getNativeAddon().memcpy2d(this.ctx, dst, dpitch, src, spitch, width, height, memcpyKindToNative(kind));
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

const MEMCPY_H2H = 0;
const MEMCPY_H2D = 1;
const MEMCPY_D2H = 2;
const MEMCPY_D2D = 3;

function memcpyKindToNative(kind: MemcpyKind): number {
  switch (kind) {
    case MemcpyKind.HostToHost: return MEMCPY_H2H;
    case MemcpyKind.HostToDevice: return MEMCPY_H2D;
    case MemcpyKind.DeviceToHost: return MEMCPY_D2H;
    case MemcpyKind.DeviceToDevice: return MEMCPY_D2D;
    case MemcpyKind.Default: return 4;
  }
}

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
