import { GlmOps } from "./glm_ops";
import { SafeTensorFile } from "./safetensors";

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

  fusedAddRmsnorm(residualOut: Tensor | number, input: Tensor | number, weight: Tensor | number, eps: number, dim: number, batch: number): Tensor {
    const out = this.workspace.alloc([batch, dim], this.type);
    this.workspace.glm.fusedAddRmsnorm(out.data, ptr(residualOut), this.data, ptr(input), ptr(weight), eps, dim, batch);
    return out;
  }

  fusedNormRope(weight: Tensor | number, cos: Tensor | number, sin: Tensor | number, eps: number, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, inStride?: number): Tensor {
    const out = this.workspace.alloc([batch, nHeads, seqLen, headDim], this.type);
    this.workspace.glm.fusedNormRope(out.data, this.data, ptr(weight), ptr(cos), ptr(sin), eps, ropeDim, headDim, nHeads, seqLen, batch, inStride ?? headDim);
    return out;
  }

  embedding(table: Tensor | number, ids: Tensor | number, hidden: number, seqLen: number): void {
    this.workspace.glm.embedding(this.data, ptr(table), ptr(ids), hidden, seqLen);
  }

  siluAndMul(gate: Tensor | number, up: Tensor | number, intermediate: number, batch: number): Tensor {
    const out = this.workspace.alloc([batch, intermediate], this.type);
    this.workspace.glm.siluAndMul(out.data, ptr(gate), ptr(up), intermediate, batch);
    return out;
  }

  add(a: Tensor | number, b: Tensor | number, n: number): void {
    this.workspace.glm.add(this.data, ptr(a), ptr(b), n);
  }

  transpose4d(d0: number, d1: number, d2: number, d3: number, p0: number, p1: number, p2: number, p3: number): Tensor {
    const dims = [d0, d1, d2, d3];
    const out = this.workspace.alloc([dims[p0], dims[p1], dims[p2], dims[p3]], this.type);
    this.workspace.glm.transpose4d(out.data, this.data, d0, d1, d2, d3, p0, p1, p2, p3);
    return out;
  }

  applyRotaryPosEmb(x: Tensor | number, cos: Tensor | number, sin: Tensor | number, ropeDim: number, nHeads: number, seqLen: number, batch: number, unsqueezeDim: number): void {
    this.workspace.glm.applyRotaryPosEmb(this.data, ptr(x), ptr(cos), ptr(sin), ropeDim, nHeads, seqLen, batch, unsqueezeDim);
  }

  applyRotaryPosEmbPartial(x: Tensor | number, cos: Tensor | number, sin: Tensor | number, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, unsqueezeDim: number): void {
    this.workspace.glm.applyRotaryPosEmbPartial(this.data, ptr(x), ptr(cos), ptr(sin), ropeDim, headDim, nHeads, seqLen, batch, unsqueezeDim);
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

  gdnRecurrentStep(state: Tensor | number, qkv: Tensor | number, aRaw: Tensor | number, bRaw: Tensor | number, aLog: Tensor | number, dtBias: Tensor | number, numHeads: number, dK: number, dV: number, batchSize: number, stateStride: number, qkvSeqStride: number): void {
    this.workspace.glm.gdnRecurrentStep(this.data, ptr(state), ptr(qkv), ptr(aRaw), ptr(bRaw), ptr(aLog), ptr(dtBias), numHeads, dK, dV, batchSize, stateStride, qkvSeqStride);
  }

  gdnPrefill(state: Tensor | number, qkv: Tensor | number, aRaw: Tensor | number, bRaw: Tensor | number, aLog: Tensor | number, dtBias: Tensor | number, cuSeqlens: Tensor | number, totalSeqLen: number, numHeads: number, dK: number, dV: number, batchSize: number, stateStride: number, qkvSeqStride: number): void {
    this.workspace.glm.gdnPrefill(this.data, ptr(state), ptr(qkv), ptr(aRaw), ptr(bRaw), ptr(aLog), ptr(dtBias), ptr(cuSeqlens), totalSeqLen, numHeads, dK, dV, batchSize, stateStride, qkvSeqStride);
  }

  causalConv1d(convState: Tensor | number, input: Tensor | number, weight: Tensor | number, cuSeqlens: Tensor | number, convDim: number, totalSeqLen: number, kernelSize: number, batchSize: number, convStateStride: number): void {
    this.workspace.glm.causalConv1d(this.data, ptr(convState), ptr(input), ptr(weight), ptr(cuSeqlens), convDim, totalSeqLen, kernelSize, batchSize, convStateStride);
  }

  causalConv1dUpdate(convState: Tensor | number, input: Tensor | number, weight: Tensor | number, convDim: number, kernelSize: number, batchSize: number, convStateStride: number): void {
    this.workspace.glm.causalConv1dUpdate(this.data, ptr(convState), ptr(input), ptr(weight), convDim, kernelSize, batchSize, convStateStride);
  }

  rmsnormGated(input: Tensor | number, gate: Tensor | number, weight: Tensor | number, eps: number, dim: number, batch: number): void {
    this.workspace.glm.rmsnormGated(this.data, ptr(input), ptr(gate), ptr(weight), eps, dim, batch);
  }

  sigmoid(input: Tensor | number, n: number): void {
    this.workspace.glm.sigmoid(this.data, ptr(input), n);
  }

  mul(a: Tensor | number, b: Tensor | number, n: number): void {
    this.workspace.glm.mul(this.data, ptr(a), ptr(b), n);
  }
}
