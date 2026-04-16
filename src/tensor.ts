import { GlmOps } from "./glm_ops";
import { SafeTensorFile } from "./safetensors";

function ptr(t: Tensor | number): number {
  return typeof t === "number" ? t : t.data;
}

function numElements(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

export interface OpContext {
  glm: GlmOps;
  cfg: { hiddenSize: number; intermediateSize: number };
  weights: Map<string, Tensor>;
  ws: {
    tensors: Map<string, Tensor>;
  };
}

export class Tensor {
  data: number;
  readonly type: string;
  readonly shape: number[];
  private glm: GlmOps;
  readonly name?: string;

  private constructor(glm: GlmOps, data: number, shape: number[], type: string, name?: string) {
    this.glm = glm;
    this.data = data;
    this.shape = shape;
    this.type = type;
    this.name = name;
  }

  static alloc(glm: GlmOps, shape: number[], type: string, name?: string): Tensor {
    const bytes = Math.ceil(numElements(shape) * SafeTensorFile.dtypeBytes(type));
    const data = glm.alloc(bytes);
    return new Tensor(glm, data, shape, type, name);
  }

  get bytes(): number {
    return Math.ceil(numElements(this.shape) * SafeTensorFile.dtypeBytes(this.type));
  }

  free(): void {
    if (this.data !== 0) {
      this.glm.freeBuf(this.data);
      this.data = 0;
    }
  }

  h2d(data: Buffer, size?: number): void {
    this.glm.h2d(this.data, data, size);
  }

  d2h(buf: Buffer, size?: number): void {
    this.glm.d2h(buf, this.data, size);
  }

  linear(input: Tensor | number, weight: Tensor | number, batch: number, n: number, k: number, context?: OpContext): void {
    const w = typeof weight === "number" ? undefined : weight;
    if (context && w && w.type === "F8_E4M3") {
      const scale = context.weights.get(w.name! + "_scale_inv")!;
      context.glm.fp8LinearDecode(this.data, ptr(input), w.data, scale.data, batch, n, k);
    } else {
      this.glm.linear(this.data, ptr(input), ptr(weight), batch, n, k);
    }
  }

  rmsnorm(input: Tensor | number, weight: Tensor | number, eps: number, dim: number, batch: number): void {
    this.glm.rmsnorm(this.data, ptr(input), ptr(weight), eps, dim, batch);
  }

  embedding(table: Tensor | number, ids: Tensor | number, hidden: number, seqLen: number): void {
    this.glm.embedding(this.data, ptr(table), ptr(ids), hidden, seqLen);
  }

  siluAndMul(gate: Tensor | number, up: Tensor | number, intermediate: number, batch: number): void {
    this.glm.siluAndMul(this.data, ptr(gate), ptr(up), intermediate, batch);
  }

  add(a: Tensor | number, b: Tensor | number, n: number): void {
    this.glm.add(this.data, ptr(a), ptr(b), n);
  }

  transpose4d(input: Tensor | number, d0: number, d1: number, d2: number, d3: number, p0: number, p1: number, p2: number, p3: number): void {
    this.glm.transpose4d(this.data, ptr(input), d0, d1, d2, d3, p0, p1, p2, p3);
  }

  applyRotaryPosEmb(x: Tensor | number, cos: Tensor | number, sin: Tensor | number, ropeDim: number, nHeads: number, seqLen: number, batch: number, unsqueezeDim: number): void {
    this.glm.applyRotaryPosEmb(this.data, ptr(x), ptr(cos), ptr(sin), ropeDim, nHeads, seqLen, batch, unsqueezeDim);
  }

  applyRotaryPosEmbPartial(x: Tensor | number, cos: Tensor | number, sin: Tensor | number, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, unsqueezeDim: number): void {
    this.glm.applyRotaryPosEmbPartial(this.data, ptr(x), ptr(cos), ptr(sin), ropeDim, headDim, nHeads, seqLen, batch, unsqueezeDim);
  }

  arange(start: number, step: number, count: number): void {
    this.glm.arange(this.data, start, step, count);
  }

  argmax(input: Tensor | number, dim: number, batch: number = 1): void {
    this.glm.argmax(this.data, ptr(input), dim, batch);
  }

  indexSelect(src: Tensor | number, indices: Tensor | number, dim: number, k: number): void {
    this.glm.indexSelect(this.data, ptr(src), ptr(indices), dim, k);
  }

  gdnRecurrentStep(state: Tensor | number, q: Tensor | number, k: Tensor | number, v: Tensor | number, aRaw: Tensor | number, bRaw: Tensor | number, aLog: Tensor | number, dtBias: Tensor | number, numHeads: number, dK: number, dV: number): void {
    this.glm.gdnRecurrentStep(this.data, ptr(state), ptr(q), ptr(k), ptr(v), ptr(aRaw), ptr(bRaw), ptr(aLog), ptr(dtBias), numHeads, dK, dV);
  }

  gdnPrefill(state: Tensor | number, q: Tensor | number, k: Tensor | number, v: Tensor | number, aRaw: Tensor | number, bRaw: Tensor | number, aLog: Tensor | number, dtBias: Tensor | number, seqLen: number, numHeads: number, dK: number, dV: number): void {
    this.glm.gdnPrefill(this.data, ptr(state), ptr(q), ptr(k), ptr(v), ptr(aRaw), ptr(bRaw), ptr(aLog), ptr(dtBias), seqLen, numHeads, dK, dV);
  }

  causalConv1d(convState: Tensor | number, input: Tensor | number, weight: Tensor | number, convDim: number, seqLen: number, kernelSize: number): void {
    this.glm.causalConv1d(this.data, ptr(convState), ptr(input), ptr(weight), convDim, seqLen, kernelSize);
  }

  causalConv1dUpdate(convState: Tensor | number, input: Tensor | number, weight: Tensor | number, convDim: number, kernelSize: number): void {
    this.glm.causalConv1dUpdate(this.data, ptr(convState), ptr(input), ptr(weight), convDim, kernelSize);
  }

  rmsnormGated(input: Tensor | number, gate: Tensor | number, weight: Tensor | number, eps: number, dim: number, batch: number): void {
    this.glm.rmsnormGated(this.data, ptr(input), ptr(gate), ptr(weight), eps, dim, batch);
  }

  sigmoid(input: Tensor | number, n: number): void {
    this.glm.sigmoid(this.data, ptr(input), n);
  }

  mul(a: Tensor | number, b: Tensor | number, n: number): void {
    this.glm.mul(this.data, ptr(a), ptr(b), n);
  }

  qkvSplit(kOut: Tensor | number, vOut: Tensor | number, qkvIn: Tensor | number, seqLen: number, numHeads: number, dK: number, dV: number): void {
    this.glm.qkvSplit(this.data, ptr(kOut), ptr(vOut), ptr(qkvIn), seqLen, numHeads, dK, dV);
  }
}
