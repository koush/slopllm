import { type SamplingParams } from "./chat_model";
import { DeviceOps, TensorParallelism } from "./device_ops";
import { getNativeAddon } from "./glm_ops";
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
  private pinnedBuffer?: Buffer;

  constructor(public readonly workspace: WorkspaceBase,
    public readonly data: number,
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

  withPinnedBuffer(fn: (buf: Buffer) => void) {
    if (!this.pinned) {
      throw new Error("Tensor is not pinned");
    }
    if (this.data === 0) {
      throw new Error("Tensor has no data");
    }
    this.pinnedBuffer ||= getNativeAddon().hostPointerToBuffer(this.data, this.allocSize);
    fn(this.pinnedBuffer);
  }

  readPinnedBuffer(): Buffer {
    if (!this.pinned) {
      throw new Error("Tensor is not pinned");
    }
    if (this.data === 0) {
      throw new Error("Tensor has no data");
    }
    this.pinnedBuffer ||= getNativeAddon().hostPointerToBuffer(this.data, this.allocSize);
    return this.pinnedBuffer;
  }

  setName(name: string) {
    if (this.name)
      throw new Error(`Tensor already has name ${this.name}, cannot rename to ${name}`);
    (this as { name: string }).name = name;
    this.workspace.tracked.delete(this);
    this.workspace.exported.delete(this);
    this.workspace.tensors.set(name, this);
  }

  detachData() {
    (this as { data: number }).data = 0;
    this.pinnedBuffer = undefined;
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
    const weightK = weight.type === "U8" ? weight.shape[1] * 2 : weight.shape[1]; // NVFP4: packed K/2
    if (this.shape[1] !== weightK) throw new Error(`linear: input dim ${this.shape[1]} != weight dim ${weightK} (weight type ${weight.type}, shape [${weight.shape}])`);
    if (weight.type !== "BF16" && weight.type !== "F8_E4M3" && weight.type !== "U8") throw new Error(`linear: weight type must be BF16, F8_E4M3, or U8, got ${weight.type}`);
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

  mlaKvCacheAppend(ckvData: Tensor, kpeData: Tensor, indices: Tensor, indptr: Tensor, lastPageLen: Tensor, appendCkv: Tensor, appendKpe: Tensor, batchIndices: Tensor, positions: Tensor, nnz: number, pageSize: number, headDimCkv: number, headDimKpe: number, appendCkvStrideN: number, appendKpeStrideN: number): void {
  }

  abstract fill(value: number, n: number): void;
  abstract mmapLoad(mmapPtr: number, offset: number, nbytes: number, gdnQkvLayout?: import("./device_ops").GdnQkvLayout): void;
  abstract memcpy(src: Tensor, size?: number, kind?: MemcpyKind): void;
  rotaryEmbedding(positionIds: Tensor, dimHalf: number, batch: number, seqLen: number): { cos: Tensor, sin: Tensor } {
    if (positionIds.type !== "I32") throw new Error(`rotaryEmbedding: positionIds must be I32, got ${positionIds.type}`);
    return undefined as never;
  }

  ropeTranspose(cos: Tensor, sin: Tensor, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, inStride?: number): Tensor {
    if (ropeDim > 0 && cos.shape.length !== sin.shape.length) throw new Error(`ropeTranspose: cos ndim ${cos.shape.length} != sin ndim ${sin.shape.length}`);
    return undefined as never;
  }

  applyRotaryPosEmb(cos: Tensor, sin: Tensor, ropeDim: number, nHeads: number, seqLen: number, batch: number, unsqueezeDim: number): Tensor {
    return undefined as never;
  }

  mlaVExpand(vProj: Tensor, kvLoraRank: number, vHeadDim: number, nHeads: number, seqLen: number, batch: number): Tensor {
    return undefined as never;
  }

  sigmoid(): Tensor {
    return undefined as never;
  }

  topk(k: number, dim: number): { values: Tensor, indices: Tensor } {
    return undefined as never;
  }

  indexAdd(indices: Tensor, values: Tensor, nIndices: number, dim: number): void {
  }

  add(other: Tensor, n?: number): Tensor {
    return undefined as never;
  }

  scaleInPlace(scale: number, n: number): void {
  }

  mul(other: Tensor, n?: number): Tensor {
    return undefined as never;
  }

  scatterScalar(indices: Tensor, value: number, k: number, outDim: number, batch: number): void {
  }

  maskedFill(mask: Tensor, value: number, n: number): void {
  }

  applyRotaryPosEmbPartial(cos: Tensor, sin: Tensor, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, unsqueezeDim: number): Tensor {
    return undefined as never;
  }

  rowScaleAdd(input: Tensor, scales: Tensor, rows: number, dim: number): void {
  }

  reduceSum(dim: number, batch: number): Tensor {
    return undefined as never;
  }

  rowNormalize(scale: number, dim: number, batch: number, normalize?: boolean): Tensor {
    return undefined as never;
  }

  groupMaskMul(groupMask: Tensor, numExperts: number, expertsPerGroup: number, nGroup: number, batch: number): void {
  }

  expertScale(weights: Tensor, indices: Tensor, expertId: number, topK: number, batch: number): void {
  }

  mulMatId(input: Tensor, weightPtrs: Tensor, expertIds: Tensor, batchIds: Tensor, count: number, N: number, K: number): Tensor {
    return undefined as never;
  }

  nvfp4MulMatId(input: Tensor, weightPtrs: Tensor, scalePtrs: Tensor, scale2Ptrs: Tensor, expertIds: Tensor, batchIds: Tensor, count: number, N: number, K: number): Tensor {
    return undefined as never;
  }

  scatterAddRows(input: Tensor, scales: Tensor, batchIds: Tensor, dim: number, count: number, numRows: number, _workspace?: Tensor): void {
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

  abstract doSampleBatch(outTokens: Tensor, topkVals: Tensor, topkIdxs: Tensor, workspace: Tensor, logits: Tensor, penaltyTokens: Tensor, penaltyCount: Tensor, maxWindow: number, vocabSize: number, batchSize: number, temperatures: Tensor, repPenalties: Tensor, presPenalties: Tensor, topKs: Tensor, topPs: Tensor, stepCounter: Tensor, maxEffectiveK: number): void;

  sampleTokenGPU(params: SamplingParams, tokenHistory: number[]): Tensor {
    return this.sampleBatchGPU([params], [tokenHistory]);
  }

  sampleBatchGPU(params: SamplingParams[], tokenHistories: number[][]): Tensor {
    const vs = this.shape[this.shape.length - 1];
    const maxWindow = Math.max(...params.map(p => p.repetitionPenaltyWindow));
    using ws = new SamplingWorkspace(this.workspace.glm, params, vs, maxWindow, tokenHistories);
    const outToken = this.workspace.alloc([ws.batchSize], "I32");
    return ws.sampleInto(this, outToken);
  }
}

export class SamplingWorkspace extends WorkspaceBase {
  readonly vocabSize: number;
  readonly batchSize: number;
  readonly maxWindow: number;
  params!: SamplingParams[];

  readonly penaltyTokens: Tensor;
  readonly penaltyCount: Tensor;
  readonly stepCounter: Tensor;
  readonly temperatures: Tensor;
  readonly repPenalties: Tensor;
  readonly presPenalties: Tensor;
  readonly topKs: Tensor;
  readonly topPs: Tensor;
  readonly outToken: Tensor;

  private readonly topkVals: Tensor;
  private readonly topkIdxs: Tensor;
  private readonly sampleWorkspaceBuf: Tensor;

  constructor(glm: DeviceOps, params: SamplingParams[], vocabSize: number, maxWindow: number, tokenHistories?: number[][]) {
    super(glm);
    this.vocabSize = vocabSize;
    this.batchSize = params.length;
    this.maxWindow = maxWindow;

    this.penaltyTokens = this.alloc([maxWindow > 0 ? this.batchSize * maxWindow : this.batchSize], "I32");
    this.penaltyCount = this.alloc([this.batchSize], "I32");
    this.stepCounter = this.alloc([1], "U32");
    this.temperatures = this.alloc([this.batchSize], "F32");
    this.repPenalties = this.alloc([this.batchSize], "F32");
    this.presPenalties = this.alloc([this.batchSize], "F32");
    this.topKs = this.alloc([this.batchSize], "I32");
    this.topPs = this.alloc([this.batchSize], "F32");
    this.outToken = this.alloc([this.batchSize], "I32");

    const SAMPLING_MAX_TOPK = 256;
    const SAMPLING_BLOCK_SIZE = 256;
    this.topkVals = this.alloc([this.batchSize * SAMPLING_MAX_TOPK * SAMPLING_BLOCK_SIZE], "F32");
    this.topkIdxs = this.alloc([this.batchSize * SAMPLING_MAX_TOPK * SAMPLING_BLOCK_SIZE], "I32");
    this.sampleWorkspaceBuf = this.alloc([this.batchSize * vocabSize], "F32");

    const seedBuf = Buffer.alloc(4);
    seedBuf.writeUInt32LE(Math.floor(Math.random() * 0xFFFFFFFF) >>> 0, 0);
    this.stepCounter.h2d(seedBuf);

    this.initPenaltyState(params, tokenHistories ?? []);
    this.updateSampler(params);
  }

  private initPenaltyState(params: SamplingParams[], tokenHistories: number[][]): void {
    const I32 = 4;
    const batchSize = this.batchSize;
    const vs = this.vocabSize;
    const maxWindow = this.maxWindow;

    const penaltyBufSize = maxWindow > 0 ? batchSize * maxWindow * I32 : batchSize * I32;
    const penaltyBuf = Buffer.alloc(penaltyBufSize);
    const countBuf = Buffer.alloc(batchSize * I32);

    for (let i = 0; i < batchSize; i++) {
      const p = params[i];
      const hasPenalty = p.repetitionPenalty !== 1.0 || p.presencePenalty !== 0;
      let numTokens = 0;
      if (hasPenalty && maxWindow > 0) {
        const history = tokenHistories[i];
        const seen = new Set<number>();
        const start = Math.max(0, history.length - maxWindow);
        for (let j = start; j < history.length; j++) seen.add(history[j]);
        for (const tid of seen) {
          if (tid < vs) {
            penaltyBuf.writeInt32LE(tid, (i * maxWindow + numTokens) * I32);
            numTokens++;
          }
        }
      }
      countBuf.writeInt32LE(numTokens, i * I32);
    }

    if (maxWindow > 0) {
      this.penaltyTokens.h2d(penaltyBuf, penaltyBufSize);
    }
    this.penaltyCount.h2d(countBuf);
  }

  updateSampler(params: SamplingParams[]): void {
    if (params.length !== this.batchSize) {
      throw new Error(`updateSampler: expected ${this.batchSize} params, got ${params.length}`);
    }
    this.params = params;

    const I32 = 4;
    const batchSize = this.batchSize;

    const tempBuf = Buffer.alloc(batchSize * 4);
    const repBuf = Buffer.alloc(batchSize * 4);
    const presBuf = Buffer.alloc(batchSize * 4);
    const topKBuf = Buffer.alloc(batchSize * I32);
    const topPBuf = Buffer.alloc(batchSize * 4);

    for (let i = 0; i < batchSize; i++) {
      const p = params[i];
      const topK = p.topK > 0 ? p.topK : 0;
      const temperature = p.temperature > 0 ? p.temperature : 0;
      tempBuf.writeFloatLE(temperature, i * 4);
      repBuf.writeFloatLE(p.repetitionPenalty, i * 4);
      presBuf.writeFloatLE(p.presencePenalty, i * 4);
      topKBuf.writeInt32LE(topK, i * I32);
      topPBuf.writeFloatLE(p.topP, i * 4);
    }

    this.temperatures.h2d(tempBuf);
    this.repPenalties.h2d(repBuf);
    this.presPenalties.h2d(presBuf);
    this.topKs.h2d(topKBuf);
    this.topPs.h2d(topPBuf);
  }

  sample(logits: Tensor): Tensor {
    return this.sampleInto(logits, this.outToken);
  }

  sampleInto(logits: Tensor, outToken: Tensor): Tensor {
    const batchSize = this.batchSize;
    const vs = this.vocabSize;

    let maxEffectiveK = 0;
    for (let i = 0; i < batchSize; i++) {
      const p = this.params[i];
      const topK = p.topK > 0 ? p.topK : 0;
      const temperature = p.temperature > 0 ? p.temperature : 0;
      let effectiveK: number;
      if (temperature <= 0 && topK <= 0) {
        effectiveK = 1;
      } else if (topK > 0) {
        effectiveK = topK < vs ? topK : vs;
      } else {
        effectiveK = 32;
      }
      if (effectiveK > maxEffectiveK) maxEffectiveK = effectiveK;
    }

    logits.doSampleBatch(
      outToken,
      this.topkVals,
      this.topkIdxs,
      this.sampleWorkspaceBuf,
      logits,
      this.penaltyTokens,
      this.penaltyCount,
      this.maxWindow,
      vs,
      batchSize,
      this.temperatures,
      this.repPenalties,
      this.presPenalties,
      this.topKs,
      this.topPs,
      this.stepCounter,
      maxEffectiveK,
    );

    return outToken;
  }
}
