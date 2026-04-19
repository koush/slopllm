import { GlmOps } from "./glm_ops";
import { SafeTensorFile } from "./safetensors";
import type { PagedKVCache } from "./paged_kv";
import type { CommonModelWorkspace } from "./chat_model";

function ptr(t: Tensor | number): number {
  return typeof t === "number" ? t : t.data;
}

function numElements(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

export interface TensorWorkspace {
  readonly glm: GlmOps;
  readonly tensors: Map<string, Tensor>;
  readonly tracked: Set<Tensor>;
  readonly disposed: Set<Tensor>;
  readonly exported: Set<Tensor>;
  alloc(shape: number[], type: string, name?: string): Tensor;
}

export class Tensor implements Disposable {
  data: number;
  readonly allocSize: number;
  readonly shape: number[];
  readonly type: string;
  readonly name?: string;
  private readonly workspace: TensorWorkspace;
  readonly pinned: boolean;

  constructor(workspace: TensorWorkspace, data: number, allocSize: number, shape: number[], type: string, name: string | undefined, pinned: boolean) {
    this.workspace = workspace;
    this.data = data;
    this.allocSize = allocSize;
    this.shape = shape;
    this.type = type;
    this.name = name;
    this.pinned = pinned;
  }

  static byteCount(shape: number[], type: string): number {
    return Math.ceil(numElements(shape) * SafeTensorFile.dtypeBytes(type));
  }

  get bytes(): number {
    return Math.ceil(numElements(this.shape) * SafeTensorFile.dtypeBytes(this.type));
  }

  free(): void {
    if (this.data !== 0) {
      if (this.pinned) {
        this.workspace.glm.freePinned(this.data);
      } else {
        this.workspace.glm.freeBuf(this.data);
      }
      (this as { data: number }).data = 0;
    }
  }

  [Symbol.dispose](): void {
    if (this.name !== undefined) {
      throw new Error("Cannot dispose named tensor");
    }
    if (this.data === 0) return;
    this.workspace.tracked.delete(this);
    this.workspace.disposed.add(this);
  }

  removeTracking(): this {
    if (this.name !== undefined) {
      throw new Error("Cannot removeTracking on named tensor");
    }
    this.workspace.tracked.delete(this);
    this.workspace.exported.add(this);
    return this;
  }

  h2d(data: Buffer, size?: number): void {
    this.workspace.glm.h2d(this.data, data, size);
  }

  d2h(buf: Buffer, size?: number): void {
    this.workspace.glm.d2h(buf, this.data, size);
  }

  linear(weight: Tensor, batch: number): Tensor {
    const n = weight.shape[0];
    const k = weight.shape[1];
    const outShape = [batch, n];
    const out = this.workspace.alloc(outShape, this.type);
    if (weight.type === "F8_E4M3") {
      const scale = weight.workspace.tensors.get(weight.name! + "_scale_inv")!;
      this.workspace.glm.fp8LinearDecode(out.data, this.data, weight.data, scale.data, batch, n, k);
    } else {
      this.workspace.glm.linear(out.data, this.data, weight.data, batch, n, k);
    }
    return out;
  }

  rmsnorm(weight: Tensor | number, eps: number, dim: number, batch: number): Tensor {
    const out = this.workspace.alloc([batch, dim], this.type);
    this.workspace.glm.rmsnorm(out.data, this.data, ptr(weight), eps, dim, batch);
    return out;
  }

  fusedAddRmsnorm(input: Tensor | number, weight: Tensor | number, eps: number, dim: number, batch: number): { normed: Tensor, residual: Tensor } {
    const normed = this.workspace.alloc([batch, dim], this.type);
    const residual = this.workspace.alloc([batch, dim], this.type);
    this.workspace.glm.fusedAddRmsnorm(normed.data, residual.data, this.data, ptr(input), ptr(weight), eps, dim, batch);
    return { normed, residual };
  }

  fusedNormRope(weight: Tensor | number, cos: Tensor | number, sin: Tensor | number, eps: number, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, inStride?: number): Tensor {
    const out = this.workspace.alloc([batch, nHeads, seqLen, headDim], this.type);
    this.workspace.glm.fusedNormRope(out.data, this.data, ptr(weight), ptr(cos), ptr(sin), eps, ropeDim, headDim, nHeads, seqLen, batch, inStride ?? headDim);
    return out;
  }

  embedding(ids: Tensor, hidden: number, seqLen: number): Tensor {
    const out = ids.workspace.alloc([seqLen, hidden], this.type);
    this.workspace.glm.embedding(out.data, this.data, ptr(ids), hidden, seqLen);
    return out;
  }

  siluAndMul(gate: Tensor | number, up: Tensor | number, intermediate: number, batch: number): Tensor {
    const out = this.workspace.alloc([batch, intermediate], this.type);
    this.workspace.glm.siluAndMul(out.data, ptr(gate), ptr(up), intermediate, batch);
    return out;
  }

  arange(start: number, step: number, count: number): void {
    this.workspace.glm.arange(this.data, start, step, count);
  }

  argmax(input: Tensor | number, dim: number, batch: number = 1): void {
    this.workspace.glm.argmax(this.data, ptr(input), dim, batch);
  }

  indexSelect(indices: Tensor | number, dim: number, batch: number): Tensor {
    const out = this.workspace.alloc([batch, dim], this.type);
    this.workspace.glm.indexSelect(out.data, this.data, ptr(indices), dim, batch);
    return out;
  }

  gdnRecurrentStep(state: Tensor | number, qkv: Tensor | number, aRaw: Tensor | number, bRaw: Tensor | number, aLog: Tensor | number, dtBias: Tensor | number, numHeads: number, dK: number, dV: number, batchSize: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void {
    this.workspace.glm.gdnRecurrentStep(this.data, ptr(state), ptr(qkv), ptr(aRaw), ptr(bRaw), ptr(aLog), ptr(dtBias), numHeads, dK, dV, batchSize, stateStride, qkvChStride, qkvSeqStride);
  }

  gdnPrefill(state: Tensor | number, qkv: Tensor | number, aRaw: Tensor | number, bRaw: Tensor | number, aLog: Tensor | number, dtBias: Tensor | number, cuSeqlens: Tensor | number, totalSeqLen: number, numHeads: number, dK: number, dV: number, batchSize: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void {
    this.workspace.glm.gdnPrefill(this.data, ptr(state), ptr(qkv), ptr(aRaw), ptr(bRaw), ptr(aLog), ptr(dtBias), ptr(cuSeqlens), totalSeqLen, numHeads, dK, dV, batchSize, stateStride, qkvChStride, qkvSeqStride);
  }

  causalConv1d(convState: Tensor | number, input: Tensor | number, weight: Tensor | number, cuSeqlens: Tensor | number, convDim: number, totalSeqLen: number, kernelSize: number, batchSize: number, convStateStride: number, chStride: number, seqStride: number): void {
    this.workspace.glm.causalConv1d(this.data, ptr(convState), ptr(input), ptr(weight), ptr(cuSeqlens), convDim, totalSeqLen, kernelSize, batchSize, convStateStride, chStride, seqStride);
  }


  rmsnormGated(input: Tensor | number, gate: Tensor | number, weight: Tensor | number, eps: number, dim: number, batch: number): void {
    this.workspace.glm.rmsnormGated(this.data, ptr(input), ptr(gate), ptr(weight), eps, dim, batch);
  }

  flashDecode(pagedKV: PagedKVCache, cacheIdx: number, batchSize: number, nHeads: number, nKv: number, hd: number, smScale: number): Tensor {
    const ws = this.workspace as unknown as CommonModelWorkspace;
    const out = this.workspace.alloc([batchSize, nHeads, 1, hd], this.type);
    this.workspace.glm.batchDecodeRun(
      this.data, out.data,
      pagedKV.kData[cacheIdx].data, pagedKV.vData[cacheIdx].data,
      pagedKV.indices.data, pagedKV.indptrD.data, pagedKV.lastPageLen.data,
      ws.floatWs.data, ws.intWs.data,
      ws.decodePlanInfo.data,
      batchSize, nHeads, nKv, hd, pagedKV.pageSize, smScale
    );
    return out;
  }

  flashPrefillPaged(pagedKV: PagedKVCache, cacheIdx: number, totalTokens: number, batchSize: number, nHeads: number, nKv: number, hd: number, qStrideN: number, qStrideH: number, maskMode: number, smScale: number): Tensor {
    const ws = this.workspace as unknown as CommonModelWorkspace;
    const out = this.workspace.alloc([1, nHeads, totalTokens, hd], this.type);
    this.workspace.glm.batchPrefillPagedRun(
      this.data, out.data,
      pagedKV.kData[cacheIdx].data, pagedKV.vData[cacheIdx].data,
      pagedKV.indices.data, pagedKV.indptrD.data, pagedKV.lastPageLen.data,
      ws.floatWs.data, ws.intWs.data,
      ws.qoIndptrD.data,
      ws.prefillPlanInfo.data,
      totalTokens, batchSize, nHeads, nKv, hd, pagedKV.pageSize,
      qStrideN, qStrideH, maskMode, smScale
    );
    return out;
  }

  gateSigmoidMul(gate: Tensor | number, batchSeq: number, numHeads: number, headDim: number): void {
    this.workspace.glm.gateSigmoidMul(this.data, ptr(gate), batchSeq, numHeads, headDim);
  }

  rotaryEmbedding(positionIds: Tensor, dimHalf: number, batch: number, seqLen: number): { cos: Tensor, sin: Tensor } {
    const hd = dimHalf * 2;
    const cos = positionIds.workspace.alloc([batch, seqLen, hd], this.type);
    const sin = positionIds.workspace.alloc([batch, seqLen, hd], this.type);
    this.workspace.glm.rotaryEmbedding(cos.data, sin.data, this.data, positionIds.data, dimHalf, batch, seqLen);
    return { cos, sin };
  }
}
