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
  applyRotaryPosEmb(ctx: number, out: number, input: number, cos: number, sin: number, ropeDim: number, nHeads: number, seqLen: number, batch: number, unsqueezeDim: number): void;
  indexSelect(ctx: number, out: number, src: number, indices: number, dim: number, k: number): void;
  gather(ctx: number, out: number, input: number, indices: number, k: number, inDim: number, batch: number, elemSize: number): void;
  arange(ctx: number, out: number, start: number, step: number, count: number): void;
  max(ctx: number, outValues: number, outIndices: number, input: number, dim: number, batch: number, offset: number): void;
  memcpy(ctx: number, dst: number, src: number, bytes: number, kind: number): void;
  kvCacheWrite(ctx: number, srcK: number, srcV: number, dstK: number, dstV: number, slotMapping: number, batchSize: number, nKv: number, hd: number, pageSize: number, srcKTokenStride: number, srcKHeadStride: number, srcVTokenStride: number, srcVHeadStride: number): void;
  decodeStep(ctx: number, positionIds: number, lastPageLen: number, slotMapping: number, indptr: number, indices: number, pageSize: number, batchSize: number): void;
  synchronize(ctx: number): void;
  synchronizeStream(ctx: number, streamIdx: number): void;
  setStream(ctx: number, streamIdx: number): void;
  eventRecord(ctx: number, eventIdx: number, streamIdx: number): void;
  streamWaitEvent(ctx: number, streamIdx: number, eventIdx: number): void;
  allocPinned(bytes: number): number;
  freePinned(ptr: number): void;
  hostPointerToBuffer(ptr: number, size: number): Buffer;
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
  nvfp4LinearDecode(ctx: number, bf16Out: number, bf16Input: number, fp4Weight: number, weightScale: number, weightScale2: number, m: number, n: number, k: number): void;
  gdnRecurrentStep(ctx: number, output: number, state: number, qkv: number, aRaw: number, bRaw: number, aLog: number, dtBias: number, numHeads: number, dK: number, dV: number, batchSize: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void;
  gdnPrefill(ctx: number, output: number, state: number, qkv: number, aRaw: number, bRaw: number, aLog: number, dtBias: number, cuSeqlens: number, totalSeqLen: number, numHeads: number, dK: number, dV: number, batchSize: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void;
  mlaPrefillPlan(ctx: number, floatWs: number, floatWsSize: number, intWs: number, pinnedIntWs: number, intWsSize: number, planInfo: number, qoIndptrH: number, kvIndptrH: number, kvLenH: number, batchSize: number, numHeads: number, headDimO: number, causal: boolean): void;
  mlaPrefillRun(ctx: number, qNope: number, qPe: number, ckvData: number, kpeData: number, kvIndices: number, o: number, floatWs: number, intWs: number, planInfo: number, numHeads: number, pageSize: number, maskMode: number, smScale: number, qNopeStrideN: number, qNopeStrideH: number, qPeStrideN: number, qPeStrideH: number, ckvStridePage: number, ckvStrideN: number, kpeStridePage: number, kpeStrideN: number, oStrideN: number, oStrideH: number, headDimCkv: number, headDimKpe: number): void;
  mlaDecodePlan(ctx: number, floatWs: number, floatWsSize: number, intWs: number, pinnedIntWs: number, intWsSize: number, planInfo: number, indptrH: number, batchSize: number, numQoHeads: number, pageSize: number, enableCudaGraph: boolean, headDimCkv: number, headDimKpe: number): void;
  mlaDecodeRun(ctx: number, qNope: number, qPe: number, ckvData: number, kpeData: number, indices: number, indptrD: number, lastPageLen: number, o: number, floatWs: number, intWs: number, planInfo: number, batchSize: number, numQoHeads: number, pageSize: number, smScale: number, headDimCkv: number, headDimKpe: number): void;
  mlaKvCacheAppend(ctx: number, ckvData: number, kpeData: number, indices: number, indptr: number, lastPageLen: number, appendCkv: number, appendKpe: number, batchIndices: number, positions: number, nnz: number, pageSize: number, headDimCkv: number, headDimKpe: number, appendCkvStrideN: number, appendKpeStrideN: number): void;
  causalConv1d(ctx: number, output: number, convState: number, input: number, weight: number, cuSeqlens: number, convDim: number, totalSeqLen: number, kernelSize: number, batchSize: number, convStateStride: number, chStride: number, seqStride: number): void;
  causalConv1dUpdate(ctx: number, output: number, convState: number, input: number, weight: number, convDim: number, kernelSize: number, batchSize: number, convStateStride: number): void;
  rmsnormGated(ctx: number, output: number, input: number, gate: number, weight: number, eps: number, dim: number, batch: number): void;
  gateSigmoidMul(ctx: number, attnOut: number, gateInterleaved: number, batchSeq: number, numHeads: number, headDim: number): void;
  sampleBatch(ctx: number, outTokens: number, topkVals: number, topkIdxs: number, workspace: number, logits: number, penaltyTokens: number, penaltyCount: number, maxWindow: number, vocabSize: number, batchSize: number, temperatures: number, repPenalties: number, presPenalties: number, topKs: number, topPs: number, stepCounter: number, maxEffectiveK: number): void;
  memcpy2d(ctx: number, dst: number, dpitch: number, src: number, spitch: number, width: number, height: number, kind: number): void;
  bmm(ctx: number, C: number, A: number, B: number, alpha: number, beta: number, batch: number, M: number, N: number, K: number, transA: number, transB: number): void;
  ropeTranspose(ctx: number, out: number, input: number, cos: number, sin: number, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, inStride: number): void;
  mlaVExpand(ctx: number, result: number, attnOut: number, vProj: number, kvLoraRank: number, vHeadDim: number, nHeads: number, seqLen: number, batch: number): void;
  ncclUniqueId(outId: Buffer): void;
  ncclGroupStart(): void;
  ncclGroupEnd(): void;
  ncclCommInitRank(deviceId: number, rank: number, worldSize: number, uniqueId: number): number;
  ncclCommInitAll(deviceIds: number[]): number[];
  ncclCommDestroy(comm: number): void;
  ncclAllReduce(comm: number, ctx: number, sendbuff: number, recvbuff: number, count: number, datatype: number, op: number): void;
  ncclAllGather(comm: number, ctx: number, sendbuff: number, recvbuff: number, count: number, datatype: number): void;
  p2pEnablePeerAccess(ctx: number, peerDevice: number): number;
  p2pCreateInstance(ctx: number, myRank: number, worldSize: number, maxBytes: number): number;
  p2pDestroyInstance(instance: number): void;
  p2pGetDataPtr(instance: number): number;
  p2pGetFlagPtr(instance: number): number;
  p2pSetPeers(ctx: number, instance: number, dataPtrs: number[], flagPtrs: number[]): void;
  p2pAllReduce(ctx: number, instance: number, in_: number, out: number, count: number, dtype: number): void;
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
  reduceSum(ctx: number, out: number, input: number, rows: number, cols: number): void;
  rowNormalize(ctx: number, out: number, input: number, scale: number, rows: number, cols: number, normalize: boolean): void;
  groupMaskMul(ctx: number, scores: number, groupMask: number, numExperts: number, expertsPerGroup: number, nGroup: number, batch: number): void;
  expertScale(ctx: number, out: number, weights: number, indices: number, expertId: number, topK: number, batch: number): void;
  mulMatId(ctx: number, output: number, input: number, weightPtrs: number, expertIds: number, batchIds: number, count: number, N: number, K: number): void;
  nvfp4MulMatId(ctx: number, output: number, input: number, weightPtrs: number, scalePtrs: number, scale2Ptrs: number, expertIds: number, batchIds: number, count: number, N: number, K: number): void;
  scatterAddRows(ctx: number, out: number, input: number, scales: number, batchIds: number, dim: number, count: number, numRows: number, workspace: number): void;
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
      this.detachData();
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
    let k = weight.shape[1];
    const outShape = [batch, n];
    const out = this.workspace.alloc(outShape, this.type);
    if (weight.type === "F8_E4M3") {
      const scale = weight.workspace.tensors.get(weight.name! + "_scale_inv")!;
      getNativeAddon().fp8LinearDecode(this.glm.ctx, out.data, this.data, weight.data, scale.data, batch, n, k);
    } else if (weight.type === "U8") {
      k = k * 2; // NVFP4: weight is [N, K/2] packed, kernel expects K
      const scale = weight.workspace.tensors.get(weight.name! + "_weight_scale")!;
      const scale2 = weight.workspace.tensors.get(weight.name! + "_weight_scale_2")!;
      getNativeAddon().nvfp4LinearDecode(this.glm.ctx, out.data, this.data, weight.data, scale.data, scale2.data, batch, n, k);
    } else {
      getNativeAddon().linear(this.glm.ctx, out.data, this.data, weight.data, batch, n, k);
    }
    return out;
  }

  bmm(B: Tensor, batch: number, M: number, N: number, K: number, transA: boolean = false, transB: boolean = false): Tensor {
    const out = this.workspace.alloc([batch * M, N], this.type);
    getNativeAddon().bmm(this.glm.ctx, out.data, this.data, B.data, 1.0, 0.0, batch, M, N, K, transA ? 1 : 0, transB ? 1 : 0);
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

  mlaKvCacheAppend(ckvData: Tensor, kpeData: Tensor, indices: Tensor, indptr: Tensor, lastPageLen: Tensor, appendCkv: Tensor, appendKpe: Tensor, batchIndices: Tensor, positions: Tensor, nnz: number, pageSize: number, headDimCkv: number, headDimKpe: number, appendCkvStrideN: number, appendKpeStrideN: number): void {
    getNativeAddon().mlaKvCacheAppend(this.glm.ctx, ptr(ckvData), ptr(kpeData), ptr(indices), ptr(indptr), ptr(lastPageLen), this.data, ptr(appendKpe), ptr(batchIndices), ptr(positions), nnz, pageSize, headDimCkv, headDimKpe, appendCkvStrideN, appendKpeStrideN);
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

  ropeTranspose(cos: Tensor, sin: Tensor, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, inStride?: number): Tensor {
    super.ropeTranspose(cos, sin, ropeDim, headDim, nHeads, seqLen, batch, inStride);
    const out = this.workspace.alloc([batch * nHeads, seqLen, headDim], this.type);
    getNativeAddon().ropeTranspose(this.glm.ctx, out.data, this.data, ropeDim > 0 ? cos.data : 0, ropeDim > 0 ? sin.data : 0, ropeDim, headDim, nHeads, seqLen, batch, inStride ?? headDim);
    return out;
  }

  applyRotaryPosEmb(cos: Tensor, sin: Tensor, ropeDim: number, nHeads: number, seqLen: number, batch: number, unsqueezeDim: number): Tensor {
    const out = this.workspace.alloc(this.shape, this.type);
    getNativeAddon().applyRotaryPosEmb(this.glm.ctx, out.data, this.data, cos.data, sin.data, ropeDim, nHeads, seqLen, batch, unsqueezeDim);
    return out;
  }

  mlaVExpand(vProj: Tensor, kvLoraRank: number, vHeadDim: number, nHeads: number, seqLen: number, batch: number): Tensor {
    super.mlaVExpand(vProj, kvLoraRank, vHeadDim, nHeads, seqLen, batch);
    const BS = batch * seqLen;
    const out = this.workspace.alloc([BS, nHeads * vHeadDim], this.type);
    getNativeAddon().mlaVExpand(this.glm.ctx, out.data, this.data, vProj.data, kvLoraRank, vHeadDim, nHeads, seqLen, batch);
    return out;
  }

  sigmoid(): Tensor {
    const n = this.shape.reduce((a, b) => a * b, 1);
    const out = this.workspace.alloc(this.shape, this.type);
    getNativeAddon().sigmoid(this.glm.ctx, out.data, this.data, n);
    return out;
  }

  topk(k: number, dim: number): { values: Tensor, indices: Tensor } {
    const batch = this.shape.reduce((a, b) => a * b, 1) / dim;
    const values = this.workspace.alloc([batch, k], this.type);
    const indices = this.workspace.alloc([batch, k], "I32");
    getNativeAddon().topk(this.glm.ctx, values.data, indices.data, this.data, k, dim, batch);
    return { values, indices };
  }

  indexAdd(indices: Tensor, values: Tensor, nIndices: number, dim: number): void {
    getNativeAddon().indexAdd(this.glm.ctx, this.data, indices.data, values.data, nIndices, dim);
  }

  add(other: Tensor, n?: number): Tensor {
    const count = n ?? this.shape.reduce((a, b) => a * b, 1);
    const out = this.workspace.alloc(this.shape, this.type);
    getNativeAddon().add(this.glm.ctx, out.data, this.data, other.data, count);
    return out;
  }

  scaleInPlace(scale: number, n: number): void {
    getNativeAddon().scale(this.glm.ctx, this.data, this.data, scale, n);
  }

  mul(other: Tensor, n?: number): Tensor {
    const count = n ?? this.shape.reduce((a, b) => a * b, 1);
    const out = this.workspace.alloc(this.shape, this.type);
    getNativeAddon().mul(this.glm.ctx, out.data, this.data, other.data, count);
    return out;
  }

  scatterScalar(indices: Tensor, value: number, k: number, outDim: number, batch: number): void {
    getNativeAddon().scatterScalar(this.glm.ctx, this.data, indices.data, value, k, outDim, batch);
  }

  maskedFill(mask: Tensor, value: number, n: number): void {
    getNativeAddon().maskedFill(this.glm.ctx, this.data, this.data, mask.data, value, n);
  }

  applyRotaryPosEmbPartial(cos: Tensor, sin: Tensor, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, unsqueezeDim: number): Tensor {
    const out = this.workspace.alloc(this.shape, this.type);
    getNativeAddon().applyRotaryPosEmbPartial(this.glm.ctx, out.data, this.data, cos.data, sin.data, ropeDim, headDim, nHeads, seqLen, batch, unsqueezeDim);
    return out;
  }

  rowScaleAdd(input: Tensor, scales: Tensor, rows: number, dim: number): void {
    getNativeAddon().rowScaleAdd(this.glm.ctx, this.data, input.data, scales.data, rows, dim);
  }

  reduceSum(dim: number, batch: number): Tensor {
    const out = this.workspace.alloc([batch], this.type);
    getNativeAddon().reduceSum(this.glm.ctx, out.data, this.data, batch, dim);
    return out;
  }

  rowNormalize(scale: number, dim: number, batch: number, normalize: boolean = true): Tensor {
    const out = this.workspace.alloc([batch, dim], this.type);
    getNativeAddon().rowNormalize(this.glm.ctx, out.data, this.data, scale, batch, dim, normalize);
    return out;
  }

  groupMaskMul(groupMask: Tensor, numExperts: number, expertsPerGroup: number, nGroup: number, batch: number): void {
    getNativeAddon().groupMaskMul(this.glm.ctx, this.data, groupMask.data, numExperts, expertsPerGroup, nGroup, batch);
  }

  expertScale(weights: Tensor, indices: Tensor, expertId: number, topK: number, batch: number): void {
    getNativeAddon().expertScale(this.glm.ctx, this.data, weights.data, indices.data, expertId, topK, batch);
  }

  mulMatId(input: Tensor, weightPtrs: Tensor, expertIds: Tensor, batchIds: Tensor, count: number, N: number, K: number): Tensor {
    const out = this.workspace.alloc([count, N], this.type);
    getNativeAddon().mulMatId(this.glm.ctx, out.data, input.data, weightPtrs.data, expertIds.data, batchIds.data, count, N, K);
    return out;
  }

  nvfp4MulMatId(input: Tensor, weightPtrs: Tensor, scalePtrs: Tensor, scale2Ptrs: Tensor, expertIds: Tensor, batchIds: Tensor, count: number, N: number, K: number): Tensor {
    const out = this.workspace.alloc([count, N], this.type);
    getNativeAddon().nvfp4MulMatId(this.glm.ctx, out.data, input.data, weightPtrs.data, scalePtrs.data, scale2Ptrs.data, expertIds.data, batchIds.data, count, N, K);
    return out;
  }

  scatterAddRows(input: Tensor, scales: Tensor, batchIds: Tensor, dim: number, count: number, numRows: number, _workspace?: Tensor): void {
    getNativeAddon().scatterAddRows(this.glm.ctx, this.data, input.data, scales.data, batchIds.data, dim, count, numRows, 0);
  }

  doSampleBatch(outTokens: Tensor, topkVals: Tensor, topkIdxs: Tensor, workspace: Tensor, logits: Tensor, penaltyTokens: Tensor, penaltyCount: Tensor, maxWindow: number, vocabSize: number, batchSize: number, temperatures: Tensor, repPenalties: Tensor, presPenalties: Tensor, topKs: Tensor, topPs: Tensor, stepCounter: Tensor, maxEffectiveK: number): void {
    getNativeAddon().sampleBatch(this.glm.ctx, outTokens.data, topkVals.data, topkIdxs.data, workspace.data, logits.data, penaltyTokens.data, penaltyCount.data, maxWindow, vocabSize, batchSize, temperatures.data, repPenalties.data, presPenalties.data, topKs.data, topPs.data, stepCounter.data, maxEffectiveK);
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

  synchronizeStream(streamIdx: number): void {
    getNativeAddon().synchronizeStream(this.ctx, streamIdx);
  }

  setStream(streamIdx: number): void {
    getNativeAddon().setStream(this.ctx, streamIdx);
    this.currentStream = streamIdx;
  }

  currentStream = 0;
  availableStreams = [1, 2, 3, 4, 5, 6, 7];
  withStream<T>(fn: () => T) {
    const stream = this.availableStreams.pop();
    if (stream === undefined)
      throw new Error("No available streams");
    const currentStream = this.currentStream;
    // Record event on stream 0 so the alternate stream can wait for
    // all prior work (e.g. rmsnorm output that K/V will read).
    getNativeAddon().eventRecord(this.ctx, currentStream, currentStream);
    this.setStream(stream);
    getNativeAddon().streamWaitEvent(this.ctx, stream, currentStream);
    const result = fn();
    // Record event on the alternate stream so stream 0 can wait at dispose.
    getNativeAddon().eventRecord(this.ctx, stream, stream);
    this.setStream(currentStream);
    return {
      [Symbol.dispose]: () => {
        if (this.availableStreams.includes(stream))
          throw new Error(`Stream ${stream} already disposed`);
        this.availableStreams.push(stream);
      },
      result,
      synchronize: () => {
        getNativeAddon().synchronizeStream(this.ctx, stream);
      },
      streamWaitEvent: () => {
        getNativeAddon().streamWaitEvent(this.ctx, this.currentStream, stream);
      }
    }
  }

  eventRecord(eventIdx: number, streamIdx: number): void {
    getNativeAddon().eventRecord(this.ctx, eventIdx, streamIdx);
  }

  streamWaitEvent(streamIdx: number, eventIdx: number): void {
    getNativeAddon().streamWaitEvent(this.ctx, streamIdx, eventIdx);
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

  decodeStep(positionIds: Tensor, lastPageLen: Tensor, slotMapping: Tensor, indptr: Tensor, indices: Tensor, pageSize: number, batchSize: number): void {
    getNativeAddon().decodeStep(this.ctx, ptr(positionIds), ptr(lastPageLen), ptr(slotMapping), ptr(indptr), ptr(indices), pageSize, batchSize);
  }

  freePinned(ptr: Tensor): void {
    getNativeAddon().freePinned(ptr.data);
  }

  hostPointerToBuffer(ptr: number, size: number): Buffer {
    return getNativeAddon().hostPointerToBuffer(ptr, size);
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

  nvfp4LinearDecode(bf16Out: Tensor, bf16Input: Tensor, fp4Weight: Tensor, weightScale: Tensor, weightScale2: Tensor, m: number, n: number, k: number): void {
    getNativeAddon().nvfp4LinearDecode(this.ctx, ptr(bf16Out), ptr(bf16Input), ptr(fp4Weight), ptr(weightScale), ptr(weightScale2), m, n, k);
  }

  gdnRecurrentStep(output: Tensor, state: Tensor, qkv: Tensor, aRaw: Tensor, bRaw: Tensor, aLog: Tensor, dtBias: Tensor, numHeads: number, dK: number, dV: number, batchSize: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void {
    getNativeAddon().gdnRecurrentStep(this.ctx, ptr(output), ptr(state), ptr(qkv), ptr(aRaw), ptr(bRaw), ptr(aLog), ptr(dtBias), numHeads, dK, dV, batchSize, stateStride, qkvChStride, qkvSeqStride);
  }

  gdnPrefill(output: Tensor, state: Tensor, qkv: Tensor, aRaw: Tensor, bRaw: Tensor, aLog: Tensor, dtBias: Tensor, cuSeqlens: Tensor, totalSeqLen: number, numHeads: number, dK: number, dV: number, batchSize: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void {
    getNativeAddon().gdnPrefill(this.ctx, ptr(output), ptr(state), ptr(qkv), ptr(aRaw), ptr(bRaw), ptr(aLog), ptr(dtBias), ptr(cuSeqlens), totalSeqLen, numHeads, dK, dV, batchSize, stateStride, qkvChStride, qkvSeqStride);
  }

  mlaPrefillPlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, qoIndptrH: Tensor, kvIndptrH: Tensor, kvLenH: Tensor, batchSize: number, numHeads: number, headDimO: number, causal: boolean): void {
    getNativeAddon().mlaPrefillPlan(this.ctx, ptr(floatWs), floatWsSize, ptr(intWs), ptr(pinnedIntWs), intWsSize, ptr(planInfo), ptr(qoIndptrH), ptr(kvIndptrH), ptr(kvLenH), batchSize, numHeads, headDimO, causal);
  }

  mlaPrefillRun(qNope: Tensor, qPe: Tensor, ckvData: Tensor, kpeData: Tensor, kvIndices: Tensor, o: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, numHeads: number, pageSize: number, maskMode: number, smScale: number, qNopeStrideN: number, qNopeStrideH: number, qPeStrideN: number, qPeStrideH: number, ckvStridePage: number, ckvStrideN: number, kpeStridePage: number, kpeStrideN: number, oStrideN: number, oStrideH: number, headDimCkv: number, headDimKpe: number): void {
    getNativeAddon().mlaPrefillRun(this.ctx, ptr(qNope), ptr(qPe), ptr(ckvData), ptr(kpeData), ptr(kvIndices), ptr(o), ptr(floatWs), ptr(intWs), ptr(planInfo), numHeads, pageSize, maskMode, smScale, qNopeStrideN, qNopeStrideH, qPeStrideN, qPeStrideH, ckvStridePage, ckvStrideN, kpeStridePage, kpeStrideN, oStrideN, oStrideH, headDimCkv, headDimKpe);
  }

  mlaDecodePlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, indptrH: Tensor, batchSize: number, numQoHeads: number, pageSize: number, enableCudaGraph: boolean, headDimCkv: number, headDimKpe: number): void {
    getNativeAddon().mlaDecodePlan(this.ctx, ptr(floatWs), floatWsSize, ptr(intWs), ptr(pinnedIntWs), intWsSize, ptr(planInfo), ptr(indptrH), batchSize, numQoHeads, pageSize, enableCudaGraph, headDimCkv, headDimKpe);
  }

  mlaDecodeRun(qNope: Tensor, qPe: Tensor, ckvData: Tensor, kpeData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, o: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, batchSize: number, numQoHeads: number, pageSize: number, smScale: number, headDimCkv: number, headDimKpe: number): void {
    getNativeAddon().mlaDecodeRun(this.ctx, ptr(qNope), ptr(qPe), ptr(ckvData), ptr(kpeData), ptr(indices), ptr(indptrD), ptr(lastPageLen), ptr(o), ptr(floatWs), ptr(intWs), ptr(planInfo), batchSize, numQoHeads, pageSize, smScale, headDimCkv, headDimKpe);
  }

  mlaKvCacheAppend(ckvData: Tensor, kpeData: Tensor, indices: Tensor, indptr: Tensor, lastPageLen: Tensor, appendCkv: Tensor, appendKpe: Tensor, batchIndices: Tensor, positions: Tensor, nnz: number, pageSize: number, headDimCkv: number, headDimKpe: number, appendCkvStrideN: number, appendKpeStrideN: number): void {
    getNativeAddon().mlaKvCacheAppend(this.ctx, ptr(ckvData), ptr(kpeData), ptr(indices), ptr(indptr), ptr(lastPageLen), ptr(appendCkv), ptr(appendKpe), ptr(batchIndices), ptr(positions), nnz, pageSize, headDimCkv, headDimKpe, appendCkvStrideN, appendKpeStrideN);
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

  sampleBatch(outTokens: Tensor, topkVals: Tensor, topkIdxs: Tensor, workspace: Tensor, logits: Tensor, penaltyTokens: Tensor, penaltyCount: Tensor, maxWindow: number, vocabSize: number, batchSize: number, temperatures: Tensor, repPenalties: Tensor, presPenalties: Tensor, topKs: Tensor, topPs: Tensor, stepCounter: Tensor, maxEffectiveK: number): void {
    getNativeAddon().sampleBatch(this.ctx, ptr(outTokens), ptr(topkVals), ptr(topkIdxs), ptr(workspace), ptr(logits), ptr(penaltyTokens), ptr(penaltyCount), maxWindow, vocabSize, batchSize, ptr(temperatures), ptr(repPenalties), ptr(presPenalties), ptr(topKs), ptr(topPs), ptr(stepCounter), maxEffectiveK);
  }

  memcpy2d(dst: number, dpitch: number, src: number, spitch: number, width: number, height: number, kind: MemcpyKind): void {
    getNativeAddon().memcpy2d(this.ctx, dst, dpitch, src, spitch, width, height, memcpyKindToNative(kind));
  }

  ropeTranspose(ctx: number, out: number, input: number, cos: number, sin: number, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, inStride: number): void {
    getNativeAddon().ropeTranspose(ctx, out, input, cos, sin, ropeDim, headDim, nHeads, seqLen, batch, inStride);
  }

  mlaVExpand(ctx: number, result: number, attnOut: number, vProj: number, kvLoraRank: number, vHeadDim: number, nHeads: number, seqLen: number, batch: number): void {
    getNativeAddon().mlaVExpand(ctx, result, attnOut, vProj, kvLoraRank, vHeadDim, nHeads, seqLen, batch);
  }

  sigmoid(ctx: number, out: number, input: number, n: number): void {
    getNativeAddon().sigmoid(ctx, out, input, n);
  }

  topk(ctx: number, outValues: number, outIndices: number, input: number, k: number, dim: number, batch: number): void {
    getNativeAddon().topk(ctx, outValues, outIndices, input, k, dim, batch);
  }

  indexAdd(ctx: number, out: number, indices: number, values: number, nIndices: number, dim: number): void {
    getNativeAddon().indexAdd(ctx, out, indices, values, nIndices, dim);
  }

  add(ctx: number, out: number, a: number, b: number, n: number): void {
    getNativeAddon().add(ctx, out, a, b, n);
  }

  scale(ctx: number, out: number, input: number, scale: number, n: number): void {
    getNativeAddon().scale(ctx, out, input, scale, n);
  }

  mul(ctx: number, out: number, a: number, b: number, n: number): void {
    getNativeAddon().mul(ctx, out, a, b, n);
  }

  scatterScalar(ctx: number, out: number, indices: number, value: number, k: number, outDim: number, batch: number): void {
    getNativeAddon().scatterScalar(ctx, out, indices, value, k, outDim, batch);
  }

  maskedFill(ctx: number, out: number, input: number, mask: number, value: number, n: number): void {
    getNativeAddon().maskedFill(ctx, out, input, mask, value, n);
  }

  applyRotaryPosEmbPartial(ctx: number, out: number, input: number, cos: number, sin: number, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, unsqueezeDim: number): void {
    getNativeAddon().applyRotaryPosEmbPartial(ctx, out, input, cos, sin, ropeDim, headDim, nHeads, seqLen, batch, unsqueezeDim);
  }

  rowScaleAdd(ctx: number, out: number, input: number, scales: number, rows: number, dim: number): void {
    getNativeAddon().rowScaleAdd(ctx, out, input, scales, rows, dim);
  }

  rowNormalize(ctx: number, out: number, input: number, scale: number, rows: number, cols: number, normalize: boolean): void {
    getNativeAddon().rowNormalize(ctx, out, input, scale, rows, cols, normalize);
  }

  groupMaskMul(ctx: number, scores: number, groupMask: number, numExperts: number, expertsPerGroup: number, nGroup: number, batch: number): void {
    getNativeAddon().groupMaskMul(ctx, scores, groupMask, numExperts, expertsPerGroup, nGroup, batch);
  }

  expertScale(ctx: number, out: number, weights: number, indices: number, expertId: number, topK: number, batch: number): void {
    getNativeAddon().expertScale(ctx, out, weights, indices, expertId, topK, batch);
  }

  mulMatId(ctx: number, output: number, input: number, weightPtrs: number, expertIds: number, batchIds: number, count: number, N: number, K: number): void {
    getNativeAddon().mulMatId(ctx, output, input, weightPtrs, expertIds, batchIds, count, N, K);
  }

  nvfp4MulMatId(ctx: number, output: number, input: number, weightPtrs: number, scalePtrs: number, scale2Ptrs: number, expertIds: number, batchIds: number, count: number, N: number, K: number): void {
    getNativeAddon().nvfp4MulMatId(ctx, output, input, weightPtrs, scalePtrs, scale2Ptrs, expertIds, batchIds, count, N, K);
  }

  scatterAddRows(ctx: number, out: number, input: number, scales: number, batchIds: number, dim: number, count: number, numRows: number, workspace: number): void {
    getNativeAddon().scatterAddRows(ctx, out, input, scales, batchIds, dim, count, numRows, workspace);
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
