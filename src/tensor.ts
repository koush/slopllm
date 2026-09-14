import { StridedMmap, TensorParallelism } from "./device_ops";
import { MemcpyKind } from "./enums";
import { getNativeAddon } from "./native-addon";
import { SafeTensorFile } from "./safetensors";
import { type WorkspaceBase } from "./workspace";
import type { HeapKey } from "./heap";

export interface MoeRoutingOptions {
  numExpertsPerToken: number;
  correctionBias?: Tensor;
  scalingFactor: number;
  normalize: boolean;
}

export interface MoeRoutingWeightsStream extends Disposable {
  result: Tensor;
  streamWaitEvent(): void;
  /** Composite backends retain each device's own wait operation. */
  shards?: readonly MoeRoutingWeightsStream[];
}

export interface MoeRoutingResult {
  indices: Tensor;
  normalizedWeightsStream: MoeRoutingWeightsStream;
}

function numElements(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

export abstract class Tensor implements Disposable {
  parallelism: TensorParallelism = TensorParallelism.Replicated;
  private pinnedBuffer?: Buffer;
  // stack: string;
  views = new Set<Tensor>();
  disposed = false;
  viewDisposed = false;
  id: number;
  static nextId = 1;
  captured = false;

  constructor(public workspace: WorkspaceBase,
    public readonly data: number,
    public readonly allocSize: number,
    public readonly shape: number[],
    public readonly type: string,
    public readonly name: string | undefined,
    public readonly pinned: boolean,
    public readonly view: Tensor | undefined,
    public readonly recycleKey: HeapKey | null = null) {
    this.id = Tensor.nextId++;
    while (this.view?.view) {
      this.view = this.view.view;
    }
    // this.stack = this.name ? undefined! : new Error("Tensor allocated at:").stack!;
    if (this.view) {
      this.view.views.add(this);
    }
  }

  stage() {
    if (this.name)
      throw new Error(`Cannot stage named tensor ${this.name}`);
    if (this.disposed)
      throw new Error(`Tensor has been disposed and can not be staged`);
    if (this.captured)
      return;
    if (this.view) {
      this.view.stage();
    }
    this.workspace.tracked.delete(this);
    this.workspace.staged.add(this);
  }

  unstage() {
    if (this.disposed)
      return;
    if (!this.workspace.staged.has(this))
      return;
    if (this.view) {
      this.view.unstage();
    }
    this.workspace.staged.delete(this);
    this.workspace.tracked.add(this);
  }

  same(other: Tensor): boolean {
    if (this === other) return true;
    if (!other.matches(this)) return false;
    if (this.data !== other.data) return false;
    if (this.parallelism !== other.parallelism) return false;
    return true;
  }

  matches(other: Tensor): boolean {
    if (this === other) return true;
    if (this.shape.length !== other.shape.length) return false;
    for (let i = 0; i < this.shape.length; i++) {
      if (this.shape[i] !== other.shape[i]) return false;
    }
    if (this.type !== other.type) return false;
    if (this.pinned !== other.pinned) return false;
    return true;
  }

  get numElements(): number {
    return numElements(this.shape);
  }

  static byteCount(shape: number[], type: string): number {
    return Math.ceil(numElements(shape) * SafeTensorFile.dtypeBytes(type));
  }

  get bytes(): number {
    return Math.ceil(this.numElements * SafeTensorFile.dtypeBytes(this.type));
  }

  memoryRanges(): readonly { data: number; bytes: number }[] {
    return [{ data: this.data, bytes: this.bytes }];
  }

  debugDescription(): string {
    const ptr = `0x${this.data.toString(16)}`;
    const recycleKey = this.recycleKey === null
      ? "default"
      : typeof this.recycleKey === "number"
        ? `stream:${this.recycleKey}`
        : typeof this.recycleKey === "symbol"
          ? this.recycleKey.toString()
          : this.recycleKey === undefined
            ? "synchronized"
            : `object:${this.recycleKey.constructor?.name ?? "unknown"}`;
    const heapState = this.pinned || this.data === 0
      ? "n/a"
      : this.workspace.describeDeviceRange(this.data, this.allocSize);
    return `id=${this.id} shape=[${this.shape}] type=${this.type} ptr=${ptr} bytes=${this.bytes} allocSize=${this.allocSize} parallelism=${this.parallelism} captured=${this.captured} disposed=${this.disposed} recycleKey=${recycleKey} workspace=${this.workspace.constructor.name} heaps=(${heapState})`;
  }

  withPinnedBuffer(fn: (buf: Buffer) => void) {
    if (!this.pinned) {
      throw new Error("Tensor is not pinned");
    }
    if (this.data === 0) {
      throw new Error("Tensor has no data");
    }
    // A view may have more backing capacity than its logical byte range.
    this.pinnedBuffer ||= getNativeAddon().hostPointerToBuffer(this.data, this.bytes);
    fn(this.pinnedBuffer);
  }

  readPinnedBuffer(): Buffer {
    if (!this.pinned) {
      throw new Error("Tensor is not pinned");
    }
    if (this.data === 0) {
      throw new Error("Tensor has no data");
    }
    this.pinnedBuffer ||= getNativeAddon().hostPointerToBuffer(this.data, this.bytes);
    return this.pinnedBuffer;
  }

  detachData() {
    (this as { data: number }).data = 0;
    this.pinnedBuffer = undefined;
  }

  reshape(newShape: number[], newType?: string): Tensor {
    const outType = newType ?? this.type;
    const currentBytes = Math.ceil(this.numElements * SafeTensorFile.dtypeBytes(this.type));
    const targetBytes = Math.ceil(numElements(newShape) * SafeTensorFile.dtypeBytes(outType));
    if (currentBytes !== targetBytes) {
      throw new Error(`reshape: cannot reshape [${this.shape}] (${this.type}, ${currentBytes} bytes) to [${newShape}] (${outType}, ${targetBytes} bytes)`);
    }

    const reshaped = this.workspace.glm.wrapTensor(this.workspace, this.data, this.allocSize, newShape, outType, this.pinned, this, this.recycleKey);
    return reshaped;
  }

  viewClone(): Tensor {
    return this.reshape(this.shape);
  }

  abstract free(): void;

  capture() {
    const captured = this.workspace.glm.wrapTensor(this.workspace, this.data, this.allocSize, this.shape, this.type, this.pinned, this.view?.capture(), this.recycleKey);
    (captured as { name: string | undefined }).name = this.name;
    captured.captured = true;
    return captured;
  }

  uncapture(): Tensor {
    if (!this.captured) {
      return this.viewClone();
    }

    for (const tracked of this.workspace.tracked) {
      if (this.same(tracked))
        return tracked.viewClone();
    }
    for (const exported of this.workspace.staged) {
      if (this.same(exported))
        return exported.viewClone();
    }

    for (const tensor of this.workspace.tracked) {
      if (this.same(tensor)) {
        throw new Error("Tensor found in tracked set after uncapture check");
      }
    }

    for (const tensor of this.workspace.staged) {
      if (this.same(tensor)) {
        throw new Error("Tensor found in staged set after uncapture check");
      }
    }

    if (this.view) {
      using uncapturedView = this.view.uncapture();
      return this.workspace.glm.wrapTensor(this.workspace, this.data, this.allocSize, this.shape, this.type, this.pinned, uncapturedView, this.recycleKey);
    }

    if (!this.pinned && this.workspace.claimDevice(this.data, this.allocSize, this.recycleKey)) {
      const ret = this.workspace.glm.wrapTensor(this.workspace, this.data, this.allocSize, this.shape, this.type, false, undefined, this.recycleKey);
      this.workspace.addTracked(ret);
      return ret;
    }
    if (this.pinned) {
      for (const disposed of this.workspace.getDisposedPools(true)) {
        for (const check of disposed) {
          if (this.data === check.data && this.allocSize === check.allocSize) {
            disposed.delete(check);
            check.detachData();
            const ret = this.workspace.glm.wrapTensor(this.workspace, this.data, this.allocSize, this.shape, this.type, true, undefined, this.recycleKey);
            this.workspace.addTracked(ret);
            return ret;
          }
        }
      }
    }

    return this._uncapture();
  }

  _uncapture() {
    const lineage = this.recycleKey === null ? undefined : [this.recycleKey, undefined];
    const copy = this.workspace.alloc(this.shape, this.type, undefined, this.parallelism, lineage);
    // possible to get the exact same allocation, maybe optimize for this in the future
    if (this.same(copy)) {
      return copy;
    }

    copy.memcpy(this);
    return copy;
  }

  canDispose() {
    if (this.disposed) {
      return false;
    }
    if (this.captured) {
      return false;
    }
    if (this.name !== undefined) {
      // return false?
      throw new Error(`Cannot dispose named tensor ${this.name}`);
    }
    if (this.workspace.staged.has(this)) {
      return false;
    }
    return true;
  }

  [Symbol.dispose](): void {
    if (!this.canDispose()) {
      return;
    }
    // if stream is active, defer disposal until stream switch
    if (this.views.size) {
      this.viewDisposed = true;
      return;
    }
    this.disposed = true;
    this.workspace.tracked.delete(this);
    this.workspace.staged.delete(this);
    if (this.view) {
      this.view.views.delete(this);
      if (this.view.viewDisposed) {
        this.view[Symbol.dispose]();
      }
      return;
    }
    if (this.data === 0) return;
    if (this.pinned) {
      this.workspace.synchronizingHost.add(this);
    } else {
      const recycleKey = this.recycleKey === null ? this.workspace.glm.currentStream : this.recycleKey;
      this.workspace.recycleDevice(this.data, this.allocSize, recycleKey);
      this.detachData();
    }
  }

  removeTracking(): this {
    if (this.name !== undefined) {
      throw new Error(`Cannot removeTracking on named tensor ${this.name}`);
    }
    if (this.captured)
      return this;
    if (this.workspace.tracking) {
      this.stage();
    }
    return this;
  }

  resumeTracking(): this {
    if (this.name !== undefined) {
      throw new Error(`Cannot resumeTracking on named tensor ${this.name}`);
    }
    if (this.disposed) {
      return this;
    }
    if (this.view) {
      this.view.resumeTracking();
    }
    this.workspace.tracked.add(this);
    this.workspace.staged.delete(this);
    return this;
  }

  abstract h2d(data: Buffer, size?: number): void;
  abstract d2h(buf: Buffer, size?: number): void;

  linear(weight: Tensor): Tensor {
    if (this.shape.length !== 2) throw new Error(`linear: input must be 2D, got shape [${this.shape}]`);
    if (weight.shape.length !== 2) throw new Error(`linear: weight must be 2D, got shape [${weight.shape}]`);
    const weightK = weight.type === "U8" ? weight.shape[1] * 2 : weight.shape[1]; // NVFP4: packed K/2
    if (this.shape[1] !== weightK) throw new Error(`linear: input dim ${this.shape[1]} != weight dim ${weightK} (weight type ${weight.type}, shape [${weight.shape}])`);
    if (weight.type !== "BF16" && weight.type !== "F8_E4M3" && weight.type !== "U8") throw new Error(`linear: weight type must be BF16, F8_E4M3, or U8, got ${weight.type}`);
    return undefined as never;
  }

  outputProj(weight: Tensor): Tensor {
    return this.linear(weight);
  }

  // tokenMajor: A=[M,batch,K], B=[batch,K,N], output=[M,batch,N].
  // Default mode uses contiguous batch-major matrices and returns [batch*M,N].
  bmm(B: Tensor, batch: number, M: number, N: number, K: number, transA: boolean = false, transB: boolean = false, tokenMajor: boolean = false): Tensor {
    if (tokenMajor && (transA || transB)) {
      throw new Error("bmm: tokenMajor requires non-transposed operands");
    }
    return undefined as never;
  }

  transpose4d(d0: number, d1: number, d2: number, d3: number, p0: number, p1: number, p2: number, p3: number): Tensor {
    return undefined as never;
  }

  writePointers(tensors: Tensor[]): void {
    if (this.type !== "I64") throw new Error(`writePointers: expected I64 tensor, got ${this.type}`);
    const n = tensors.length;
    if (this.numElements < n) throw new Error(`writePointers: tensor has ${this.numElements} elements, need ${n}`);
    return undefined as never;
  }

  rmsnorm(weight: Tensor, eps: number): Tensor {
    if (this.shape.length !== 2) {
      throw new Error(`rmsnorm: expected 2D input, got [${this.shape}]`);
    }
    const dim = this.shape[1];
    if (weight.numElements !== dim) throw new Error(`rmsnorm: weight has ${weight.numElements} elements, expected ${dim}`);
    return undefined as never;
  }

  layernorm(weight: Tensor, bias: Tensor, eps: number): Tensor {
    if (this.shape.length !== 2) {
      throw new Error(`layernorm: expected 2D input, got [${this.shape}]`);
    }
    const dim = this.shape[1];
    if (weight.numElements !== dim) throw new Error(`layernorm: weight has ${weight.numElements} elements, expected ${dim}`);
    if (bias.numElements !== dim) throw new Error(`layernorm: bias has ${bias.numElements} elements, expected ${dim}`);
    return undefined as never;
  }

  fusedAddRmsnorm(input: Tensor, weight: Tensor, eps: number): { normed: Tensor, residual: Tensor } {
    if (this.shape.length !== 2) {
      throw new Error(`fusedAddRmsnorm: expected 2D residual, got [${this.shape}]`);
    }
    if (input.shape.length !== 2 || input.shape[0] !== this.shape[0] || input.shape[1] !== this.shape[1]) {
      throw new Error(`fusedAddRmsnorm: input shape [${input.shape}] does not match residual shape [${this.shape}]`);
    }
    const dim = this.shape[1];
    if (weight.numElements !== dim) throw new Error(`fusedAddRmsnorm: weight has ${weight.numElements} elements, expected ${dim}`);
    return undefined as never;
  }

  fusedNormRope(weight: Tensor, cos: Tensor, sin: Tensor, eps: number, ropeDim: number, seqLen: number, batch: number, inStride?: number, interleaved?: boolean): Tensor {
    if (this.shape[0] !== batch * seqLen) throw new Error(`fusedNormRope: input shape[0]=${this.shape[0]} != batch*seqLen=${batch * seqLen}`);
    if (cos.shape.length !== sin.shape.length) throw new Error(`fusedNormRope: cos ndim ${cos.shape.length} != sin ndim ${sin.shape.length}`);
    for (let i = 0; i < cos.shape.length; i++) {
      if (cos.shape[i] !== sin.shape[i]) throw new Error(`fusedNormRope: cos shape [${cos.shape}] != sin shape [${sin.shape}]`);
    }
    return undefined as never;
  }

  embedding(ids: Tensor): Tensor {
    if (this.shape.length !== 2) {
      throw new Error(`embedding: table must be 2D, got [${this.shape}]`);
    }
    if (ids.type !== "I32") throw new Error(`embedding: ids must be I32, got ${ids.type}`);
    if (ids.shape.length !== 1) {
      throw new Error(`embedding: ids must be 1D, got [${ids.shape}]`);
    }
    return undefined as never;
  }

  siluAndMul(up: Tensor): Tensor {
    if (this.shape.length !== 2) {
      throw new Error(`siluAndMul: gate must be 2D, got [${this.shape}]`);
    }
    if (up.shape.length !== 2 || up.shape[0] !== this.shape[0] || up.shape[1] !== this.shape[1]) {
      throw new Error(`siluAndMul: up shape [${up.shape}] does not match gate shape [${this.shape}]`);
    }
    if (this.type !== up.type) throw new Error(`siluAndMul: gate type ${this.type} != up type ${up.type}`);
    return undefined as never;
  }

  swiGluMlp(weights: { gate: Tensor, up: Tensor, down: Tensor }): Tensor {
    using upStream = this.workspace.glm.withStream(() => this.linear(weights.up));
    using upBuf = upStream.result;
    using gateBuf = this.linear(weights.gate);
    upStream.streamWaitEvent();
    using siluBuf = gateBuf.siluAndMul(upBuf);
    return siluBuf.linear(weights.down);
  }

  /** SiLU-gated MoE MLP: gate_proj + up_proj → silu_and_mul → down_proj.
   *  Weights are per-expert arrays. topkIndicesFlat is [count] expert ids (count = batch * topK).
   *  Gate/up use topK routing; down uses topK=1 (each token-expert pair is independent).
   *  Implementations should overlap gate and up on separate streams. */
  swiGluMlpMoe(
    weights: { gate: Tensor[], up: Tensor[], down: Tensor[] },
    topkIndicesFlat: Tensor,
    topK: number, count: number,
    moeIntermediate: number, hs: number,
    pfx: string,
  ): Tensor {
    if (weights.gate.length !== weights.up.length || weights.gate.length !== weights.down.length) {
      throw new Error(`swiGluMlpMoe: expert count mismatch gate=${weights.gate.length} up=${weights.up.length} down=${weights.down.length}`);
    }
    if (topkIndicesFlat.shape.length !== 1 || topkIndicesFlat.shape[0] < count) {
      throw new Error(`swiGluMlpMoe: topkIndicesFlat shape [${topkIndicesFlat.shape}] insufficient for count=${count}`);
    }
    return undefined as never;
  }

  /** Routed expert MLP and weighted combine. The routing stream and result are caller-owned. */
  swiGluMlpMoeReduce(
    inputs: {
      gate: Tensor[], up: Tensor[], down: Tensor[],
      normalizedWeightsStream: MoeRoutingWeightsStream,
    },
    topkIndicesFlat: Tensor,
    topK: number, count: number,
    moeIntermediate: number, hs: number,
    pfx: string,
  ): Tensor {
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
    if (this.numElements < batch * inDim) throw new Error(`gather: source has ${this.numElements} elements (shape [${this.shape}]${this.view ? ', view of [' + this.view.shape + ']' : ''}), needs ${batch * inDim} (batch=${batch}, inDim=${inDim})`);
    if (indices.type !== "I32") throw new Error(`gather: indices must be I32, got ${indices.type}`);
    if (indices.numElements < batch * k) throw new Error(`gather: indices has ${indices.numElements} elements (shape [${indices.shape}]${indices.view ? ', view of [' + indices.view.shape + ']' : ''}), needs ${batch * k} (batch=${batch}, k=${k})`);
    return undefined as never;
  }

  indexSelect(indices: Tensor, offset: number = 0): Tensor {
    if (this.shape.length !== 2) throw new Error(`indexSelect: input must be 2D, got shape [${this.shape}]`);
    if (indices.type !== "I32") throw new Error(`indexSelect: indices must be I32, got ${indices.type}`);
    return undefined as never;
  }

  rotateInputIds(qoIndptr: Tensor, newTokens: Tensor, batchSize: number): Tensor {
    if (this.type !== "I32") throw new Error(`rotateInputIds: inputIds must be I32, got ${this.type}`);
    if (qoIndptr.type !== "I32") throw new Error(`rotateInputIds: qoIndptr must be I32, got ${qoIndptr.type}`);
    if (newTokens.type !== "I32") throw new Error(`rotateInputIds: newTokens must be I32, got ${newTokens.type}`);
    if (qoIndptr.numElements < batchSize + 1) throw new Error(`rotateInputIds: qoIndptr has ${qoIndptr.numElements} elements, need ${batchSize + 1}`);
    if (newTokens.numElements < batchSize) throw new Error(`rotateInputIds: newTokens has ${newTokens.numElements} elements, need ${batchSize}`);
    return undefined as never;
  }

  causalConv1d(convState: Tensor, input: Tensor, weight: Tensor, cuSeqlens: Tensor, convDim: number, totalSeqLen: number, kernelSize: number, batchSize: number, convStateStride: number, chStride: number, seqStride: number): void {
    if (cuSeqlens.type !== "I32") throw new Error(`causalConv1d: cuSeqlens must be I32, got ${cuSeqlens.type}`);
  }

  causalConv1dUpdate(convState: Tensor, input: Tensor, weight: Tensor, convDim: number, kernelSize: number, batchSize: number, convStateStride: number): Tensor {
    return undefined as never;
  }

  rmsnormGated(input: Tensor, gate: Tensor, weight: Tensor, eps: number): void {
    if (input.shape.length !== 2) {
      throw new Error(`rmsnormGated: input must be 2D, got [${input.shape}]`);
    }
    const dim = input.shape[1];
    const batch = input.shape[0];
    if (gate.shape.length !== 2 || gate.shape[0] !== batch || gate.shape[1] !== dim) {
      throw new Error(`rmsnormGated: gate shape [${gate.shape}] does not match input shape [${input.shape}]`);
    }
    if (weight.numElements !== dim) throw new Error(`rmsnormGated: weight has ${weight.numElements} elements, expected ${dim}`);
  }

  gateSigmoidMul(gate: Tensor, numHeads: number, headDim: number): void {
    if (gate.type !== this.type) throw new Error(`gateSigmoidMul: gate type ${gate.type} != output type ${this.type}`);
  }

  abstract fill(value: number, n: number): void;
  abstract mmapLoad(mmapPtr: number, offset: number, nbytes: number, strided?: StridedMmap): Promise<void>;
  abstract mmapLoadAsync(mmapPtr: number, offset: number, nbytes: number): Promise<void>;
  abstract memcpy2dHostToDeviceAsync(dstOffset: number, dpitch: number, src: number, spitch: number, width: number, height: number): Promise<void>;
  memcpy(src: Tensor, size?: number, kind?: MemcpyKind): void {
    if (size !== undefined) {
      if (size < 0) {
        throw new Error(`memcpy: negative size ${size}`);
      }
      if (size > this.bytes) {
        throw new Error(`memcpy: size ${size} exceeds destination ${this.bytes} bytes`);
      }
      if (size > src.bytes) {
        throw new Error(`memcpy: size ${size} exceeds source ${src.bytes} bytes`);
      }
    }
  }
  memcpy2d(dstOffset: number, dpitch: number, src: Tensor, srcOffset: number, spitch: number, width: number, height: number, kind: MemcpyKind): void {
    if (dstOffset < 0 || srcOffset < 0 || width < 0 || height < 0 || dpitch < 0 || spitch < 0) {
      throw new Error(`memcpy2d: negative parameter (dstOffset=${dstOffset}, srcOffset=${srcOffset}, width=${width}, height=${height}, dpitch=${dpitch}, spitch=${spitch})`);
    }
    if (width > dpitch) {
      throw new Error(`memcpy2d: width ${width} exceeds dpitch ${dpitch}`);
    }
    if (width > spitch) {
      throw new Error(`memcpy2d: width ${width} exceeds spitch ${spitch}`);
    }
    const dstEnd = dstOffset + (height > 0 ? (height - 1) * dpitch + width : 0);
    const srcEnd = srcOffset + (height > 0 ? (height - 1) * spitch + width : 0);
    if (dstEnd > this.bytes) {
      throw new Error(`memcpy2d: dst region end ${dstEnd} exceeds destination ${this.bytes} bytes`);
    }
    if (srcEnd > src.bytes) {
      throw new Error(`memcpy2d: src region end ${srcEnd} exceeds source ${src.bytes} bytes`);
    }
  }
  rotaryEmbedding(positionIds: Tensor, batch: number, seqLen: number): { cos: Tensor, sin: Tensor } {
    if (positionIds.type !== "I32") throw new Error(`rotaryEmbedding: positionIds must be I32, got ${positionIds.type}`);
    if (positionIds.shape[0] !== batch * seqLen) throw new Error(`rotaryEmbedding: positionIds shape[0]=${positionIds.shape[0]} != batch*seqLen=${batch * seqLen}`);
    return undefined as never;
  }

  absorbMlaQuery(kNopeWeight: Tensor, nHeads: number, kvLoraRank: number): Tensor {
    const rows = this.shape[0];
    const nopeDim = this.shape[1] / nHeads;
    if (this.shape.length !== 2 || !Number.isInteger(nopeDim) ||
      kNopeWeight.shape.length !== 2 || kNopeWeight.shape[0] !== nHeads * nopeDim || kNopeWeight.shape[1] !== kvLoraRank) {
      throw new Error("absorbMlaQuery: incompatible query and key projection shapes");
    }
    return this.bmm(kNopeWeight, nHeads, rows, kvLoraRank, nopeDim, false, false, true);
  }

  ropeTranspose(cos: Tensor, sin: Tensor, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, inStride?: number, interleaved?: boolean): Tensor {
    if (this.shape[0] !== batch * seqLen) throw new Error(`ropeTranspose: input shape[0]=${this.shape[0]} != batch*seqLen=${batch * seqLen}`);
    if (ropeDim > 0 && cos.shape.length !== sin.shape.length) throw new Error(`ropeTranspose: cos ndim ${cos.shape.length} != sin ndim ${sin.shape.length}`);
    return undefined as never;
  }

  applyRotaryPosEmb(cos: Tensor, sin: Tensor, ropeDim: number, nHeads: number, seqLen: number, batch: number, unsqueezeDim: number, interleaved?: boolean): Tensor {
    if (this.shape.length === 2) {
      if (this.shape[0] !== batch * seqLen) throw new Error(`applyRotaryPosEmb: input shape[0]=${this.shape[0]} != batch*seqLen=${batch * seqLen}`);
    } else if (this.shape.length === 4) {
      if (this.shape[0] !== batch) throw new Error(`applyRotaryPosEmb: input shape[0]=${this.shape[0]} != batch=${batch}`);
      const seqDim = unsqueezeDim === 1 ? 2 : 1;
      if (this.shape[seqDim] !== seqLen) throw new Error(`applyRotaryPosEmb: input shape[${seqDim}]=${this.shape[seqDim]} != seqLen=${seqLen}`);
    }
    return undefined as never;
  }

  mlaVExpand(vProj: Tensor, seqLen: number, batch: number, lse?: Tensor, headOffset?: number, attnNHeads?: number, vProjHeadOffset?: number, tokenMajor?: boolean): Tensor {
    const BS = batch * seqLen;
    if (this.shape.length === 4) {
      if (this.shape[0] !== batch || this.shape[2] !== seqLen) throw new Error(`mlaVExpand: input shape [${this.shape}] != [batch=${batch}, _, seqLen=${seqLen}, _]`);
    } else {
      if (this.shape[0] !== BS) throw new Error(`mlaVExpand: input shape[0]=${this.shape[0]} != batch*seqLen=${BS}`);
    }
    return undefined as never;
  }

  sigmoid(): Tensor {
    return undefined as never;
  }

  topk(k: number, dim: number, offset?: number): { values: Tensor, indices: Tensor } {
    return undefined as never;
  }

  protected validateMoeRoute(options: MoeRoutingOptions): void {
    if (this.type !== "BF16" || this.shape.length !== 2) {
      throw new Error("moeRoute: expected BF16 logits [rows, experts]");
    }
    const experts = this.shape[1];
    const topK = options.numExpertsPerToken;
    if (!Number.isInteger(topK) || topK < 1 || topK > experts) {
      throw new Error(`moeRoute: invalid experts per token ${topK} for ${experts} experts`);
    }
    const bias = options.correctionBias;
    if (bias && (bias.type !== "BF16" || bias.shape.length !== 1 || bias.shape[0] !== experts)) {
      throw new Error(`moeRoute: expected BF16 correction bias [${experts}]`);
    }
  }

  /** Sigmoid routing: biased expert selection, unbiased normalized weights. */
  moeRoute(options: MoeRoutingOptions): MoeRoutingResult {
    this.validateMoeRoute(options);
    const [rows, experts] = this.shape;
    const topK = options.numExpertsPerToken;
    using sigmoid = this.sigmoid();
    using selection = options.correctionBias
      ? sigmoid.add(options.correctionBias)
      : sigmoid.viewClone();
    const topk = selection.topk(topK, experts);
    using values = topk.values;
    const normalizedWeightsStream = this.workspace.glm.withStream(() => {
      using scoresView = sigmoid.viewClone();
      using indicesView = topk.indices.viewClone();
      using selected = scoresView.gather(indicesView, topK, experts, rows);
      return selected.rowNormalize(options.scalingFactor, options.normalize);
    });
    return { indices: topk.indices, normalizedWeightsStream };
  }

  indexAdd(indices: Tensor, values: Tensor, nIndices: number, dim: number): void {
  }

  add(other: Tensor, n?: number): Tensor {
    if (this.shape.length === 2 && other.shape.length === 1 && this.shape[1] === other.shape[0]) {
      return undefined as never;
    }
    return undefined as never;
  }

  sumInPlace(tensors: Tensor[], writeback?: boolean): void {
    if (tensors.length === 0 || tensors.length > 8) {
      throw new Error(`sumInPlace: requires 1-8 additional tensors, got ${tensors.length}`);
    }
    for (let i = 0; i < tensors.length; i++) {
      if (tensors[i].type !== this.type) {
        throw new Error(`sumInPlace: tensor ${i} type ${tensors[i].type} != ${this.type}`);
      }
    }
  }

  sum(tensors: Tensor[]): Tensor {
    if (tensors.length === 0 || tensors.length > 7) {
      throw new Error(`sum: requires 1-7 additional tensors, got ${tensors.length}`);
    }
    for (let i = 0; i < tensors.length; i++) {
      if (tensors[i].type !== this.type) {
        throw new Error(`sum: tensor ${i} type ${tensors[i].type} != ${this.type}`);
      }
    }
    return undefined as never;
  }

  scaleInPlace(scale: number, n: number): void {
  }

  mul(other: Tensor, n?: number): Tensor {
    if (this.shape.length === 2 && other.shape.length === 1 && this.shape[1] === other.shape[0]) {
      return undefined as never;
    }
    return undefined as never;
  }

  cat(tensors: Tensor[], dim: number): Tensor {
    if (tensors.length === 0) throw new Error("cat: requires at least one tensor");
    const ndim = this.shape.length;
    for (let i = 0; i < tensors.length; i++) {
      if (tensors[i].shape.length !== ndim) {
        throw new Error(`cat: tensor ${i} has ${tensors[i].shape.length}D shape, expected ${ndim}D`);
      }
      for (let d = 0; d < ndim; d++) {
        if (d !== dim && tensors[i].shape[d] !== this.shape[d]) {
          throw new Error(`cat: tensor ${i} shape [${tensors[i].shape}] mismatch on dim ${d} (expected ${this.shape[d]})`);
        }
      }
    }
    return undefined as never;
  }

  scatterScalar(indices: Tensor, value: number, k: number): void {
  }

  maskedFill(mask: Tensor, value: number, n: number): void {
  }

  applyRotaryPosEmbPartial(cos: Tensor, sin: Tensor, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, unsqueezeDim: number, interleaved?: boolean): Tensor {
    if (this.shape[0] !== batch * seqLen) throw new Error(`applyRotaryPosEmbPartial: input shape[0]=${this.shape[0]} != batch*seqLen=${batch * seqLen}`);
    return undefined as never;
  }

  reduceSum(): Tensor {
    return undefined as never;
  }

  rowNormalize(scale: number, normalize?: boolean): Tensor {
    return undefined as never;
  }

  groupMaskMul(groupMask: Tensor, expertsPerGroup: number, nGroup: number): void {
  }

  mulMatId(weights: Tensor[], expertIds: Tensor, topK: number, count: number, N: number, K: number, name: string): Tensor {
    return undefined as never;
  }

  slice(dim: number, start: number, length: number): Tensor {
    if (dim < 0 || dim >= this.shape.length) {
      throw new Error(`slice: dim ${dim} out of range for ${this.shape.length}D tensor`);
    }
    if (start < 0) {
      start = this.shape[dim] + start;
    }
    if (start < 0 || start > this.shape[dim]) {
      throw new Error(`slice: start ${start} out of range for dim ${dim} (size ${this.shape[dim]})`);
    }
    if (length <= 0) {
      throw new Error(`slice: length must be positive, got ${length}`);
    }
    if (start + length > this.shape[dim]) {
      throw new Error(`slice: start ${start} + length ${length} exceeds dim ${dim} size ${this.shape[dim]})`);
    }
    return undefined as never;
  }

  narrow(start: number, length: number): Tensor {
    if (this.shape.length === 0) {
      throw new Error(`narrow: cannot narrow a scalar tensor`);
    }
    if (start < 0) {
      start = this.shape[0] + start;
    }
    if (start < 0 || start > this.shape[0]) {
      throw new Error(`narrow: start ${start} out of range for dim 0 (size ${this.shape[0]})`);
    }
    if (length <= 0) {
      throw new Error(`narrow: length must be positive, got ${length}`);
    }
    if (start + length > this.shape[0]) {
      throw new Error(`narrow: start ${start} + length ${length} exceeds dim 0 size ${this.shape[0]})`);
    }
    return undefined as never;
  }

  readInt32LEArray(): number[] {
    const count = this.numElements;
    const buf = Buffer.alloc(count * 4);
    this.d2h(buf);
    const result: number[] = [];
    for (let i = 0; i < count; i++) {
      result.push(buf.readInt32LE(i * 4));
    }
    return result;
  }

}
