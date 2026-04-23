import { type SamplingParams } from "./chat_model";
import { TensorParallelism } from "./device_ops";
import { SafeTensorFile } from "./safetensors";
import { WorkspaceBase } from "./workspace";

function numElements(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

export abstract class Tensor implements Disposable {
  parallelism: TensorParallelism = TensorParallelism.Replicated;
  constructor(public readonly workspace: WorkspaceBase, public data: number, public readonly allocSize: number, public readonly shape: number[], public readonly type: string, public readonly name: string | undefined, public readonly pinned: boolean) {
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

  abstract free(): void;

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

  abstract h2d(data: Buffer, size?: number): void;
  abstract d2h(buf: Buffer, size?: number): void;

  abstract linear(weight: Tensor, batch: number): Tensor;
  abstract rmsnorm(weight: Tensor, eps: number, dim: number, batch: number): Tensor;
  abstract fusedAddRmsnorm(input: Tensor, weight: Tensor, eps: number, dim: number, batch: number): { normed: Tensor, residual: Tensor };
  abstract fusedNormRope(weight: Tensor, cos: Tensor, sin: Tensor, eps: number, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, inStride?: number): Tensor;
  abstract embedding(ids: Tensor, hidden: number, seqLen: number): Tensor;
  abstract siluAndMul(gate: Tensor, up: Tensor, intermediate: number, batch: number): Tensor;
  abstract arange(start: number, step: number, count: number): void;
  abstract argmax(): Tensor;
  abstract indexSelect(indices: Tensor, dim: number, batch: number): Tensor;

  abstract gdnRecurrentStep(state: Tensor, qkv: Tensor, aRaw: Tensor, bRaw: Tensor, aLog: Tensor, dtBias: Tensor, numHeads: number, dK: number, dV: number, batchSize: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void;
  abstract gdnPrefill(state: Tensor, qkv: Tensor, aRaw: Tensor, bRaw: Tensor, aLog: Tensor, dtBias: Tensor, cuSeqlens: Tensor, totalSeqLen: number, numHeads: number, dK: number, dV: number, batchSize: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void;
  abstract causalConv1d(convState: Tensor, input: Tensor, weight: Tensor, cuSeqlens: Tensor, convDim: number, totalSeqLen: number, kernelSize: number, batchSize: number, convStateStride: number, chStride: number, seqStride: number): void;
  abstract causalConv1dUpdate(convState: Tensor, input: Tensor, weight: Tensor, convDim: number, kernelSize: number, batchSize: number, convStateStride: number): Tensor;
  abstract rmsnormGated(input: Tensor, gate: Tensor, weight: Tensor, eps: number, dim: number, batch: number): void;
  abstract gateSigmoidMul(gate: Tensor, batchSeq: number, numHeads: number, headDim: number): void;
  abstract fill(value: number, n: number): void;
  abstract mmapLoad(mmapPtr: number, offset: number, nbytes: number, gdnQkvLayout?: import("./device_ops").GdnQkvLayout): void;
  abstract writePinned(src: Buffer, size?: number): void;
  abstract memcpy(src: Tensor, size?: number): void;
  abstract rotaryEmbedding(positionIds: Tensor, dimHalf: number, batch: number, seqLen: number): { cos: Tensor, sin: Tensor };

  readInt32LE(): number[] {
    const count = numElements(this.shape);
    const buf = Buffer.alloc(count * 4);
    this.d2h(buf);
    const result: number[] = [];
    for (let i = 0; i < count; i++) {
      result.push(buf.readInt32LE(i * 4));
    }
    return result;
  }

  protected abstract doSampleBatch(outTokens: Tensor, topkVals: Tensor, topkIdxs: Tensor, workspace: Tensor, logits: Tensor, penaltyTokens: Tensor, penaltyOffsets: Tensor, vocabSize: number, batchSize: number, temperatures: Tensor, repPenalties: Tensor, presPenalties: Tensor, topKs: Tensor, topPs: Tensor, randomVals: Tensor, maxEffectiveK: number): void;

  sampleBatchGPU(params: SamplingParams[], tokenHistories: number[][]): Tensor {
    const batchSize = params.length;
    if (batchSize !== tokenHistories.length) {
      throw new Error(`sampleBatchGPU: params length ${batchSize} != tokenHistories length ${tokenHistories.length}`);
    }
    const vs = this.shape[this.shape.length - 1];

    const I32 = 4;
    const SAMPLING_MAX_TOPK = 256;
    const SAMPLING_BLOCK_SIZE = 256;

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

    this.doSampleBatch(
      outToken,
      topkVals,
      topkIdxs,
      sampleWorkspace,
      this,
      penaltyTokens,
      penaltyOffsets,
      vs,
      batchSize,
      temperatures,
      repPenalties,
      presPenalties,
      topKs,
      topPs,
      randomVals,
      maxEffectiveK,
    );

    return outToken;
  }

  sampleTokenGPU(params: SamplingParams, tokenHistory: number[]): Tensor {
    return this.sampleBatchGPU([params], [tokenHistory]);
  }
}
