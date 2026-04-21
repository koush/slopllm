import { type SamplingParams } from "./chat_model";
import { I32, SAMPLING_BLOCK_SIZE, SAMPLING_MAX_TOPK } from "./glm_ops";
import { SafeTensorFile } from "./safetensors";
import { WorkspaceBase } from "./workspace";

function ptr(t: Tensor | number): number {
  return typeof t === "number" ? t : t.data;
}

function numElements(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

export class Tensor implements Disposable {
  data: number;
  readonly allocSize: number;
  readonly shape: number[];
  readonly type: string;
  readonly name?: string;
  private readonly workspace: WorkspaceBase;
  readonly pinned: boolean;

  constructor(workspace: WorkspaceBase, data: number, allocSize: number, shape: number[], type: string, name: string | undefined, pinned: boolean) {
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

  argmax(): Tensor {
    const batch = this.shape[0];
    const dim = this.shape[1];
    const out = this.workspace.alloc([batch], "I32");
    this.workspace.glm.argmax(out.data, this.data, dim, batch);
    return out;
  }

  readInt32LE(): number[] {
    const count = numElements(this.shape);
    const buf = Buffer.alloc(count * I32);
    this.d2h(buf);
    const result: number[] = [];
    for (let i = 0; i < count; i++) {
      result.push(buf.readInt32LE(i * I32));
    }
    return result;
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

  causalConv1dUpdate(convState: Tensor | number, input: Tensor | number, weight: Tensor | number, convDim: number, kernelSize: number, batchSize: number, convStateStride: number): Tensor {
    const out = this.workspace.alloc([batchSize * convDim], this.type);
    this.workspace.glm.causalConv1dUpdate(out.data, ptr(convState), ptr(input), ptr(weight), convDim, kernelSize, batchSize, convStateStride);
    return out;
  }

  rmsnormGated(input: Tensor | number, gate: Tensor | number, weight: Tensor | number, eps: number, dim: number, batch: number): void {
    this.workspace.glm.rmsnormGated(this.data, ptr(input), ptr(gate), ptr(weight), eps, dim, batch);
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

  sampleBatchGPU(params: SamplingParams[], tokenHistories: number[][]): Tensor {
    const batchSize = params.length;
    if (batchSize !== tokenHistories.length) {
      throw new Error(`sampleBatchGPU: params length ${batchSize} != tokenHistories length ${tokenHistories.length}`);
    }
    const vs = this.shape[this.shape.length - 1];

    const penaltyBufSize = batchSize * 1024 * I32;
    const penaltyBuf = Buffer.alloc(penaltyBufSize);
    const offsetsBuf = Buffer.alloc((batchSize + 1) * I32);
    const tempBuf = Buffer.alloc(batchSize * 4);
    const repBuf = Buffer.alloc(batchSize * 4);
    const presBuf = Buffer.alloc(batchSize * 4);
    const topKBuf = Buffer.alloc(batchSize * I32);
    const topPBuf = Buffer.alloc(batchSize * 4);
    const randBuf = Buffer.alloc(batchSize * 4);

    let maxEffectiveK = 0;
    let penaltyOffset = 0;
    offsetsBuf.writeInt32LE(0, 0);

    for (let i = 0; i < batchSize; i++) {
      const p = params[i];
      const history = tokenHistories[i];

      const hasRepPenalty = p.repetitionPenalty !== 1.0;
      const hasPresPenalty = p.presencePenalty !== 0;

      let numPenaltyTokens = 0;
      if (hasRepPenalty || hasPresPenalty) {
        const seen = new Set<number>();
        const start = Math.max(0, history.length - p.repetitionPenaltyWindow);
        for (let j = start; j < history.length; j++) seen.add(history[j]);
        for (const tid of seen) {
          if (tid < vs) {
            penaltyBuf.writeInt32LE(tid, penaltyOffset * I32 + numPenaltyTokens * I32);
            numPenaltyTokens++;
          }
        }
      }
      penaltyOffset += numPenaltyTokens;
      offsetsBuf.writeInt32LE(penaltyOffset, (i + 1) * I32);

      const topK = p.topK > 0 ? p.topK : 0;
      const temperature = p.temperature > 0 ? p.temperature : 0;

      let effectiveK: number;
      if (temperature <= 0 && topK <= 0) {
        effectiveK = 1;
      } else if (topK > 0) {
        effectiveK = topK < vs ? topK : vs;
      } else {
        effectiveK = 64;
      }
      if (effectiveK > maxEffectiveK) maxEffectiveK = effectiveK;

      tempBuf.writeFloatLE(temperature, i * 4);
      repBuf.writeFloatLE(p.repetitionPenalty, i * 4);
      presBuf.writeFloatLE(p.presencePenalty, i * 4);
      topKBuf.writeInt32LE(topK, i * I32);
      topPBuf.writeFloatLE(p.topP, i * 4);
      randBuf.writeFloatLE(Math.random(), i * 4);
    }

    using topkVals = this.workspace.alloc([batchSize * SAMPLING_MAX_TOPK * SAMPLING_BLOCK_SIZE], "F32");
    using topkIdxs = this.workspace.alloc([batchSize * SAMPLING_MAX_TOPK * SAMPLING_BLOCK_SIZE], "I32");
    using sampleWorkspace = this.workspace.alloc([batchSize * vs], "F32");
    using penaltyTokens = this.workspace.alloc([batchSize * 1024], "I32");
    using penaltyOffsets = this.workspace.alloc([batchSize + 1], "I32");
    using temperatures = this.workspace.alloc([batchSize], "F32");
    using repPenalties = this.workspace.alloc([batchSize], "F32");
    using presPenalties = this.workspace.alloc([batchSize], "F32");
    using topKs = this.workspace.alloc([batchSize], "I32");
    using topPs = this.workspace.alloc([batchSize], "F32");
    using randomVals = this.workspace.alloc([batchSize], "F32");
    const outToken = this.workspace.alloc([batchSize], "I32");

    if (penaltyOffset > 0) {
      penaltyTokens.h2d(penaltyBuf, penaltyOffset * I32);
    }
    penaltyOffsets.h2d(offsetsBuf);
    temperatures.h2d(tempBuf);
    repPenalties.h2d(repBuf);
    presPenalties.h2d(presBuf);
    topKs.h2d(topKBuf);
    topPs.h2d(topPBuf);
    randomVals.h2d(randBuf);

    this.workspace.glm.sampleBatch(
      outToken.data,
      topkVals.data,
      topkIdxs.data,
      sampleWorkspace.data,
      this.data,
      penaltyTokens.data,
      penaltyOffsets.data,
      vs,
      batchSize,
      temperatures.data,
      repPenalties.data,
      presPenalties.data,
      topKs.data,
      topPs.data,
      randomVals.data,
      maxEffectiveK,
    );

    return outToken;
  }

  sampleTokenGPU(params: SamplingParams, tokenHistory: number[]): Tensor {
    return this.sampleBatchGPU([params], [tokenHistory]);
  }
}
