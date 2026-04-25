import { type SamplingParams } from "./chat_model";
import { TensorParallelism } from "./device_ops";
import { SafeTensorFile } from "./safetensors";
import { WorkspaceBase } from "./workspace";

export const enum MemcpyKind {
  HostToHost = 0,
  HostToDevice = 1,
  DeviceToHost = 2,
  DeviceToDevice = 3,
  Default = 4,
}

function numElements(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

export abstract class Tensor implements Disposable {
  parallelism: TensorParallelism = TensorParallelism.Replicated;
  constructor(public readonly workspace: WorkspaceBase,
    public data: number,
    public readonly allocSize: number,
    public readonly shape: number[],
    public readonly type: string,
    public readonly name: string | undefined,
    public readonly pinned: boolean,
    public readonly view: Tensor | undefined) {
    this.data = data;
    this.allocSize = allocSize;
    this.shape = shape;
    this.type = type;
    this.name = name;
    this.pinned = pinned;
    this.view = view;
  }

  static byteCount(shape: number[], type: string): number {
    return Math.ceil(numElements(shape) * SafeTensorFile.dtypeBytes(type));
  }

  get bytes(): number {
    return Math.ceil(numElements(this.shape) * SafeTensorFile.dtypeBytes(this.type));
  }

  reshape(newShape: number[]): Tensor {
    const current = numElements(this.shape);
    const target = numElements(newShape);
    if (current !== target) {
      throw new Error(`reshape: cannot reshape [${this.shape}] (${current} elements) to [${newShape}] (${target} elements)`);
    }

    const reshaped = this.workspace.glm.wrapTensor(this.workspace, this.data, this.allocSize, newShape, this.type, this.pinned, this);
    return reshaped;
  }

  abstract free(): void;

  [Symbol.dispose](): void {
    if (this.name !== undefined) {
      throw new Error("Cannot dispose named tensor");
    }
    if (this.data === 0) return;
    this.workspace.tracked.delete(this);
    if (this.view) return;
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

  linear(weight: Tensor, batch: number): Tensor {
    if (this.shape.length !== 2) throw new Error(`linear: input must be 2D, got shape [${this.shape}]`);
    if (weight.shape.length !== 2) throw new Error(`linear: weight must be 2D, got shape [${weight.shape}]`);
    if (this.shape[0] < batch) throw new Error(`linear: input batch ${this.shape[0]} < ${batch}`);
    if (this.shape[1] !== weight.shape[1]) throw new Error(`linear: input dim ${this.shape[1]} != weight dim ${weight.shape[1]}`);
    if (weight.type !== "BF16" && weight.type !== "F8_E4M3") throw new Error(`linear: weight type must be BF16 or F8_E4M3, got ${weight.type}`);
    return undefined as never;
  }

  rmsnorm(weight: Tensor, eps: number, dim: number, batch: number): Tensor {
    if (this.shape.length !== 2 || this.shape[0] < batch || this.shape[1] !== dim) {
      throw new Error(`rmsnorm: input shape [${this.shape}] incompatible with batch=${batch}, dim=${dim}`);
    }
    if (numElements(weight.shape) !== dim) throw new Error(`rmsnorm: weight has ${numElements(weight.shape)} elements, expected ${dim}`);
    return undefined as never;
  }

  fusedAddRmsnorm(input: Tensor, weight: Tensor, eps: number, dim: number, batch: number): { normed: Tensor, residual: Tensor } {
    if (this.shape.length !== 2 || this.shape[0] < batch || this.shape[1] !== dim) {
      throw new Error(`fusedAddRmsnorm: residual shape [${this.shape}] incompatible with batch=${batch}, dim=${dim}`);
    }
    if (input.shape.length !== 2 || input.shape[0] < batch || input.shape[1] !== dim) {
      throw new Error(`fusedAddRmsnorm: input shape [${input.shape}] incompatible with batch=${batch}, dim=${dim}`);
    }
    if (numElements(weight.shape) !== dim) throw new Error(`fusedAddRmsnorm: weight has ${numElements(weight.shape)} elements, expected ${dim}`);
    return undefined as never;
  }

  fusedNormRope(weight: Tensor, cos: Tensor, sin: Tensor, eps: number, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, inStride?: number): Tensor {
    if (numElements(weight.shape) !== headDim) throw new Error(`fusedNormRope: weight has ${numElements(weight.shape)} elements, expected ${headDim}`);
    if (cos.shape.length !== sin.shape.length) throw new Error(`fusedNormRope: cos ndim ${cos.shape.length} != sin ndim ${sin.shape.length}`);
    for (let i = 0; i < cos.shape.length; i++) {
      if (cos.shape[i] !== sin.shape[i]) throw new Error(`fusedNormRope: cos shape [${cos.shape}] != sin shape [${sin.shape}]`);
    }
    return undefined as never;
  }

  embedding(ids: Tensor, hidden: number, seqLen: number): Tensor {
    if (this.shape.length !== 2 || this.shape[1] !== hidden) {
      throw new Error(`embedding: table shape [${this.shape}] incompatible with hidden=${hidden}`);
    }
    if (ids.type !== "I32") throw new Error(`embedding: ids must be I32, got ${ids.type}`);
    if (ids.shape.length !== 1 || ids.shape[0] < seqLen) {
      throw new Error(`embedding: ids shape [${ids.shape}] insufficient for seqLen=${seqLen}`);
    }
    return undefined as never;
  }

  siluAndMul(gate: Tensor, up: Tensor, intermediate: number, batch: number): Tensor {
    if (gate.shape.length !== 2 || gate.shape[0] < batch || gate.shape[1] !== intermediate) {
      throw new Error(`siluAndMul: gate shape [${gate.shape}] incompatible with batch=${batch}, intermediate=${intermediate}`);
    }
    if (up.shape.length !== 2 || up.shape[0] < batch || up.shape[1] !== intermediate) {
      throw new Error(`siluAndMul: up shape [${up.shape}] incompatible with batch=${batch}, intermediate=${intermediate}`);
    }
    if (gate.type !== up.type) throw new Error(`siluAndMul: gate type ${gate.type} != up type ${up.type}`);
    return undefined as never;
  }

  arange(start: number, step: number, count: number): void {
    if (count <= 0) throw new Error(`arange: count must be positive, got ${count}`);
  }

  argmax(): Tensor {
    if (this.shape.length !== 2) throw new Error(`argmax: expected 2D tensor, got ${this.shape.length}D shape [${this.shape}]`);
    return undefined as never;
  }

  max(offset?: number): { values: Tensor, indices: Tensor } {
    if (this.shape.length !== 2) throw new Error(`max: expected 2D tensor, got ${this.shape.length}D shape [${this.shape}]`);
    return undefined as never;
  }

  gather(indices: Tensor, k: number, inDim: number, batch: number): Tensor {
    if (numElements(this.shape) < batch * inDim) throw new Error(`gather: source has ${numElements(this.shape)} elements (shape [${this.shape}]${this.view ? ', view of [' + this.view.shape + ']' : ''}), needs ${batch * inDim} (batch=${batch}, inDim=${inDim})`);
    if (indices.type !== "I32") throw new Error(`gather: indices must be I32, got ${indices.type}`);
    if (numElements(indices.shape) < batch * k) throw new Error(`gather: indices has ${numElements(indices.shape)} elements (shape [${indices.shape}]${indices.view ? ', view of [' + indices.view.shape + ']' : ''}), needs ${batch * k} (batch=${batch}, k=${k})`);
    return undefined as never;
  }

  indexSelect(indices: Tensor, dim: number, batch: number): Tensor {
    if (indices.type !== "I32") throw new Error(`indexSelect: indices must be I32, got ${indices.type}`);
    if (numElements(indices.shape) < batch) {
      throw new Error(`indexSelect: indices has ${numElements(indices.shape)} elements (shape [${indices.shape}]${indices.view ? ', view of [' + indices.view.shape + ']' : ''}), insufficient for batch=${batch}`);
    }
    return undefined as never;
  }

  gdnRecurrentStep(state: Tensor, qkv: Tensor, aRaw: Tensor, bRaw: Tensor, aLog: Tensor, dtBias: Tensor, numHeads: number, dK: number, dV: number, batchSize: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void {
    if (state.type !== "F32") throw new Error(`gdnRecurrentStep: state must be F32, got ${state.type}`);
    if (aLog.type !== "F32") throw new Error(`gdnRecurrentStep: aLog must be F32, got ${aLog.type}`);
    if (dtBias.type !== "F32") throw new Error(`gdnRecurrentStep: dtBias must be F32, got ${dtBias.type}`);
  }

  gdnPrefill(state: Tensor, qkv: Tensor, aRaw: Tensor, bRaw: Tensor, aLog: Tensor, dtBias: Tensor, cuSeqlens: Tensor, totalSeqLen: number, numHeads: number, dK: number, dV: number, batchSize: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void {
    if (state.type !== "F32") throw new Error(`gdnPrefill: state must be F32, got ${state.type}`);
    if (aLog.type !== "F32") throw new Error(`gdnPrefill: aLog must be F32, got ${aLog.type}`);
    if (dtBias.type !== "F32") throw new Error(`gdnPrefill: dtBias must be F32, got ${dtBias.type}`);
    if (cuSeqlens.type !== "I32") throw new Error(`gdnPrefill: cuSeqlens must be I32, got ${cuSeqlens.type}`);
  }

  causalConv1d(convState: Tensor, input: Tensor, weight: Tensor, cuSeqlens: Tensor, convDim: number, totalSeqLen: number, kernelSize: number, batchSize: number, convStateStride: number, chStride: number, seqStride: number): void {
    if (cuSeqlens.type !== "I32") throw new Error(`causalConv1d: cuSeqlens must be I32, got ${cuSeqlens.type}`);
  }

  causalConv1dUpdate(convState: Tensor, input: Tensor, weight: Tensor, convDim: number, kernelSize: number, batchSize: number, convStateStride: number): Tensor {
    return undefined as never;
  }

  rmsnormGated(input: Tensor, gate: Tensor, weight: Tensor, eps: number, dim: number, batch: number): void {
    if (input.shape.length !== 2 || input.shape[0] < batch || input.shape[1] !== dim) {
      throw new Error(`rmsnormGated: input shape [${input.shape}] incompatible with batch=${batch}, dim=${dim}`);
    }
    if (gate.shape.length !== 2 || gate.shape[0] < batch || gate.shape[1] !== dim) {
      throw new Error(`rmsnormGated: gate shape [${gate.shape}] incompatible with batch=${batch}, dim=${dim}`);
    }
    if (numElements(weight.shape) !== dim) throw new Error(`rmsnormGated: weight has ${numElements(weight.shape)} elements, expected ${dim}`);
  }

  gateSigmoidMul(gate: Tensor, batchSeq: number, numHeads: number, headDim: number): void {
    if (gate.type !== this.type) throw new Error(`gateSigmoidMul: gate type ${gate.type} != output type ${this.type}`);
  }

  abstract fill(value: number, n: number): void;
  abstract mmapLoad(mmapPtr: number, offset: number, nbytes: number, gdnQkvLayout?: import("./device_ops").GdnQkvLayout): void;
  abstract writePinned(src: Buffer, size?: number): void;
  abstract memcpy(src: Tensor, size?: number, kind?: MemcpyKind): void;
  rotaryEmbedding(positionIds: Tensor, dimHalf: number, batch: number, seqLen: number): { cos: Tensor, sin: Tensor } {
    if (positionIds.type !== "I32") throw new Error(`rotaryEmbedding: positionIds must be I32, got ${positionIds.type}`);
    return undefined as never;
  }

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
