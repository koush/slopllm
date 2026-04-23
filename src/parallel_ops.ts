import { DeviceOps, TensorParallelism } from "./device_ops";
import { GlmOps, getNativeAddon, f32ToBf16Bytes, bf16BytesToF32, MEMCPY_H2D, NCCL_BFLOAT16, NCCL_FLOAT32, NCCL_INT32, NCCL_SUM } from "./glm_ops";
import { Tensor } from "./tensor";
import { WorkspaceBase } from "./workspace";

export class ParallelTensor extends Tensor {
  parallelism: TensorParallelism;
  readonly shards: readonly Tensor[];
  readonly fullShape: number[];
  private readonly devices: readonly GlmOps[];
  private readonly parallelOps: ParallelOps;

  constructor(
    workspace: WorkspaceBase,
    parallelOps: ParallelOps,
    parallelism: TensorParallelism,
    shards: readonly Tensor[],
    fullShape: number[],
    type: string,
    name: string | undefined,
    pinned: boolean,
  ) {
    super(workspace, 0, 0, fullShape, type, name, pinned);
    this.parallelOps = parallelOps;
    this.devices = parallelOps.devices;
    this.parallelism = parallelism;
    this.shards = shards;
    this.fullShape = fullShape;
  }

  free(): void {
    for (let i = 0; i < this.shards.length; i++) {
      const shard = this.shards[i];
      if (shard.data !== 0) {
        if (this.pinned) {
          this.devices[i].freePinned(shard);
        } else {
          this.devices[i].freeBuf(shard);
        }
        (shard as { data: number }).data = 0;
      }
    }
  }

  [Symbol.dispose](): void {
    if (this.name !== undefined) {
      throw new Error("Cannot dispose named tensor");
    }
    this.workspace.tracked.delete(this);
    for (const shard of this.shards) {
      if (shard.data !== 0) {
        shard[Symbol.dispose]();
      }
    }
    (this.shards as Tensor[]).length = 0;
  }

  shard(rank: number): Tensor {
    return this.shards[rank];
  }

  allReduce(): ParallelTensor {
    if (this.parallelism !== TensorParallelism.PartialSum) {
      throw new Error(`allReduce requires PartialSum tensor, got ${this.parallelism}`);
    }
    const count = this.shards[0].shape.reduce((a, b) => a * b, 1);
    const dtype = this.parallelOps.ncclDatatype(this.type);
    const comms = this.parallelOps.comms;
    getNativeAddon().ncclGroupStart();
    for (let i = 0; i < this.devices.length; i++) {
      getNativeAddon().ncclAllReduce(
        comms[i], this.devices[i].ctx,
        this.shards[i].data, this.shards[i].data,
        count, dtype, NCCL_SUM,
      );
    }
    getNativeAddon().ncclGroupEnd();
    this.parallelism = TensorParallelism.Replicated;
    return this;
  }

  allGather(workspace: WorkspaceBase): ParallelTensor {
    if (this.parallelism === TensorParallelism.Replicated) {
      return this;
    }
    if (this.parallelism === TensorParallelism.PartialSum) {
      throw new Error("allGather cannot be used on PartialSum tensors; use allReduce instead");
    }
    const output = this.parallelOps.newTensor(workspace, this.fullShape, this.type, false, undefined, TensorParallelism.Replicated);
    const count = this.shards[0].shape.reduce((a, b) => a * b, 1);
    const dtype = this.parallelOps.ncclDatatype(this.type);
    const comms = this.parallelOps.comms;

    if (this.parallelism === TensorParallelism.Column) {
      getNativeAddon().ncclGroupStart();
      for (let i = 0; i < this.devices.length; i++) {
        getNativeAddon().ncclAllGather(
          comms[i], this.devices[i].ctx,
          this.shards[i].data, output.shards[i].data,
          count, dtype,
        );
      }
      getNativeAddon().ncclGroupEnd();
      return output;
    }

    if (this.parallelism === TensorParallelism.Row) {
      const eb = this.type === "BF16" ? 2 : this.type === "F32" ? 4 : this.type === "I32" ? 4 : 1;
      const outer = this.fullShape[0];
      const inner = this.fullShape.slice(2).reduce((a, b) => a * b, 1);
      const shardDim1 = this.fullShape[1] / this.devices.length;
      const shardBytes = count * eb;
      const totalElems = this.fullShape.reduce((a, b) => a * b, 1);
      const totalBytes = totalElems * eb;

      const shardWss = this.parallelOps.getShardWorkspaces(workspace);
      const tempTensors = shardWss.map(ws => ws.alloc([totalBytes], "U8"));

      getNativeAddon().ncclGroupStart();
      for (let i = 0; i < this.devices.length; i++) {
        getNativeAddon().ncclAllGather(
          comms[i], this.devices[i].ctx,
          this.shards[i].data, tempTensors[i].data,
          count, dtype,
        );
      }
      getNativeAddon().ncclGroupEnd();

      for (let i = 0; i < this.devices.length; i++) {
        for (let r = 0; r < this.devices.length; r++) {
          this.devices[i].memcpy2d(
            output.shards[i].data + r * shardDim1 * inner * eb,
            this.fullShape[1] * inner * eb,
            tempTensors[i].data + r * shardBytes,
            shardDim1 * inner * eb,
            shardDim1 * inner * eb,
            outer,
            3,
          );
        }
        tempTensors[i][Symbol.dispose]();
      }

      return output;
    }

    throw new Error(`allGather: unsupported parallelism ${this.parallelism}`);
  }

  all(workspace: WorkspaceBase): ParallelTensor {
    switch (this.parallelism) {
      case TensorParallelism.Replicated:
        return this;
      case TensorParallelism.PartialSum:
        return this.allReduce();
      case TensorParallelism.Row:
      case TensorParallelism.Column:
        return this.allGather(workspace);
    }
  }

  h2d(data: Buffer, size?: number): void {
    this.parallelOps.h2d(this, data, size);
  }

  d2h(buf: Buffer, size?: number): void {
    this.parallelOps.d2h(buf, this, size);
  }

  override linear(weight: Tensor, batch: number): Tensor {
    const pWeight = weight as ParallelTensor;
    const n = weight.shape[0];
    const k = weight.shape[1];
    const outPar = (pWeight instanceof ParallelTensor && this instanceof ParallelTensor)
      ? ParallelOps.linearOutputParallelism(pWeight.parallelism, this.parallelism)
      : TensorParallelism.Replicated;
    const out = this.workspace.alloc([batch, n], this.type, undefined, outPar);
    if (weight.type === "F8_E4M3") {
      const scale = weight.workspace.tensors.get(weight.name! + "_scale_inv")!;
      this.parallelOps.fp8LinearDecode(out, this, weight, scale, batch, n, k);
    } else {
      this.parallelOps.linear(out, this, weight, batch, n, k);
    }
    return out;
  }

  rmsnorm(weight: Tensor, eps: number, dim: number, batch: number): Tensor {
    const out = this.workspace.alloc([batch, dim], this.type);
    this.parallelOps.rmsnorm(out, this, weight, eps, dim, batch);
    return out;
  }

  fusedAddRmsnorm(input: Tensor, weight: Tensor, eps: number, dim: number, batch: number): { normed: Tensor, residual: Tensor } {
    const normed = this.workspace.alloc([batch, dim], this.type);
    const residual = this.workspace.alloc([batch, dim], this.type);
    this.parallelOps.fusedAddRmsnorm(normed, residual, this, input, weight, eps, dim, batch);
    return { normed, residual };
  }

  override fusedNormRope(weight: Tensor, cos: Tensor, sin: Tensor, eps: number, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, inStride?: number): Tensor {
    const outPar = this instanceof ParallelTensor ? this.parallelism : TensorParallelism.Replicated;
    const out = this.workspace.alloc([batch, nHeads, seqLen, headDim], this.type, undefined, outPar);
    this.parallelOps.fusedNormRope(out, this, weight, cos, sin, eps, ropeDim, headDim, nHeads, seqLen, batch, inStride ?? headDim);
    return out;
  }

  override embedding(ids: Tensor, hidden: number, seqLen: number): Tensor {
    const pTable = this as ParallelTensor;
    const outPar = (pTable instanceof ParallelTensor && pTable.parallelism === TensorParallelism.Row)
      ? TensorParallelism.Row
      : TensorParallelism.Replicated;
    const out = ids.workspace.alloc([seqLen, hidden], this.type, undefined, outPar);
    this.parallelOps.embedding(out, this, ids, hidden, seqLen);
    return out;
  }

  override siluAndMul(gate: Tensor, up: Tensor, intermediate: number, batch: number): Tensor {
    const pGate = gate as ParallelTensor;
    const outPar = (pGate instanceof ParallelTensor) ? pGate.parallelism : TensorParallelism.Replicated;
    const out = this.workspace.alloc([batch, intermediate], this.type, undefined, outPar);
    this.parallelOps.siluAndMul(out, gate, up, intermediate, batch);
    return out;
  }

  arange(start: number, step: number, count: number): void {
    this.parallelOps.arange(this, start, step, count);
  }

  argmax(): Tensor {
    const batch = this.shape[0];
    const dim = this.shape[1];
    const out = this.workspace.alloc([batch], "I32");
    this.parallelOps.argmax(out, this, dim, batch);
    return out;
  }

  indexSelect(indices: Tensor, dim: number, batch: number): Tensor {
    const out = this.workspace.alloc([batch, dim], this.type);
    this.parallelOps.indexSelect(out, this, indices, dim, batch);
    return out;
  }

  gdnRecurrentStep(_state: Tensor, _qkv: Tensor, _aRaw: Tensor, _bRaw: Tensor, _aLog: Tensor, _dtBias: Tensor, _numHeads: number, _dK: number, _dV: number, _batchSize: number, _stateStride: number, _qkvChStride: number, _qkvSeqStride: number): void {
    throw new Error("ParallelTensor.gdnRecurrentStep not implemented");
  }

  gdnPrefill(_state: Tensor, _qkv: Tensor, _aRaw: Tensor, _bRaw: Tensor, _aLog: Tensor, _dtBias: Tensor, _cuSeqlens: Tensor, _totalSeqLen: number, _numHeads: number, _dK: number, _dV: number, _batchSize: number, _stateStride: number, _qkvChStride: number, _qkvSeqStride: number): void {
    throw new Error("ParallelTensor.gdnPrefill not implemented");
  }

  causalConv1d(_convState: Tensor, _input: Tensor, _weight: Tensor, _cuSeqlens: Tensor, _convDim: number, _totalSeqLen: number, _kernelSize: number, _batchSize: number, _convStateStride: number, _chStride: number, _seqStride: number): void {
    throw new Error("ParallelTensor.causalConv1d not implemented");
  }

  causalConv1dUpdate(_convState: Tensor, _input: Tensor, _weight: Tensor, _convDim: number, _kernelSize: number, _batchSize: number, _convStateStride: number): Tensor {
    throw new Error("ParallelTensor.causalConv1dUpdate not implemented");
  }

  rmsnormGated(input: Tensor, gate: Tensor, weight: Tensor, eps: number, dim: number, batch: number): void {
    this.parallelOps.rmsnormGated(this, input, gate, weight, eps, dim, batch);
  }

  gateSigmoidMul(gate: Tensor, batchSeq: number, numHeads: number, headDim: number): void {
    this.parallelOps.gateSigmoidMul(this, gate, batchSeq, numHeads, headDim);
  }

  fill(value: number, n: number): void {
    this.parallelOps.fill(this, value, n);
  }

  mmapLoad(mmapPtr: number, offset: number, nbytes: number): void {
    this.parallelOps.mmapLoad(this, mmapPtr, offset, nbytes);
  }

  writePinned(src: Buffer, size?: number): void {
    this.parallelOps.writePinned(this, src, size);
  }

  rotaryEmbedding(positionIds: Tensor, dimHalf: number, batch: number, seqLen: number): { cos: Tensor, sin: Tensor } {
    const hd = dimHalf * 2;
    const cos = positionIds.workspace.alloc([batch, seqLen, hd], this.type);
    const sin = positionIds.workspace.alloc([batch, seqLen, hd], this.type);
    this.parallelOps.rotaryEmbedding(cos, sin, this, positionIds, dimHalf, batch, seqLen);
    return { cos, sin };
  }

  protected doSampleBatch(outTokens: Tensor, topkVals: Tensor, topkIdxs: Tensor, workspace: Tensor, logits: Tensor, penaltyTokens: Tensor, penaltyOffsets: Tensor, vocabSize: number, batchSize: number, temperatures: Tensor, repPenalties: Tensor, presPenalties: Tensor, topKs: Tensor, topPs: Tensor, randomVals: Tensor, maxEffectiveK: number): void {
    this.parallelOps.sampleBatch(outTokens, topkVals, topkIdxs, workspace, logits, penaltyTokens, penaltyOffsets, vocabSize, batchSize, temperatures, repPenalties, presPenalties, topKs, topPs, randomVals, maxEffectiveK);
  }
}

export class ParallelOps implements DeviceOps {
  readonly devices: readonly GlmOps[];
  readonly worldSize: number;
  readonly comms: number[];
  private readonly shardWorkspaces = new WeakMap<WorkspaceBase, WorkspaceBase[]>();

  constructor(devices: GlmOps[]) {
    if (devices.length === 0) {
      throw new Error("ParallelOps requires at least one device");
    }
    if ((devices.length & (devices.length - 1)) !== 0) {
      throw new Error("ParallelOps requires power-of-2 device count");
    }
    this.devices = devices;
    this.worldSize = devices.length;
    if (devices.length > 1) {
      const deviceIds = devices.map(d => d.device);
      this.comms = getNativeAddon().ncclCommInitAll(deviceIds);
    } else {
      this.comms = [];
    }
  }

  free(): void {
    if (this.comms.length > 0) {
      for (const comm of this.comms) {
        getNativeAddon().ncclCommDestroy(comm);
      }
    }
  }

  ncclDatatype(type: string): number {
    switch (type) {
      case "BF16": return NCCL_BFLOAT16;
      case "F32": return NCCL_FLOAT32;
      case "I32": return NCCL_INT32;
      default: throw new Error(`Unsupported NCCL datatype for type ${type}`);
    }
  }

  getShardWorkspaces(workspace: WorkspaceBase): WorkspaceBase[] {
    let wss = this.shardWorkspaces.get(workspace);
    if (wss === undefined) {
      wss = this.devices.map(glm => new WorkspaceBase(glm));
      this.shardWorkspaces.set(workspace, wss);
    }
    return wss;
  }

  shardWorkspacesFor(workspace: WorkspaceBase): readonly WorkspaceBase[] {
    return this.shardWorkspaces.get(workspace) ?? [];
  }

  shardShape(fullShape: number[], parallelism: TensorParallelism): number[] {
    switch (parallelism) {
      case TensorParallelism.Column:
        if (fullShape[0] % this.worldSize !== 0) {
          throw new Error(`Column parallel: shape[0]=${fullShape[0]} not divisible by worldSize=${this.worldSize}`);
        }
        return [fullShape[0] / this.worldSize, ...fullShape.slice(1)];
      case TensorParallelism.Row:
        if (fullShape.length < 2) {
          throw new Error("Row parallel: requires at least 2D shape");
        }
        if (fullShape[1] % this.worldSize !== 0) {
          throw new Error(`Row parallel: shape[1]=${fullShape[1]} not divisible by worldSize=${this.worldSize}`);
        }
        return [fullShape[0], fullShape[1] / this.worldSize, ...fullShape.slice(2)];
      case TensorParallelism.Replicated:
      case TensorParallelism.PartialSum:
      default:
        return [...fullShape];
    }
  }

  static linearOutputParallelism(weightPar: TensorParallelism, inputPar: TensorParallelism): TensorParallelism {
    if (weightPar === TensorParallelism.Column && inputPar === TensorParallelism.Replicated) {
      return TensorParallelism.Row;
    }
    if (weightPar === TensorParallelism.Row && inputPar === TensorParallelism.Row) {
      return TensorParallelism.PartialSum;
    }
    if (weightPar === TensorParallelism.Replicated && inputPar === TensorParallelism.Replicated) {
      return TensorParallelism.Replicated;
    }
    throw new Error(`linear: unsupported parallelism combination W=${weightPar}, X=${inputPar}`);
  }

  newTensor(workspace: WorkspaceBase, shape: number[], type: string, pinned: boolean, name?: string, parallelism?: TensorParallelism): ParallelTensor {
    const par = parallelism ?? TensorParallelism.Replicated;
    const ss = this.shardShape(shape, par);
    const shardWss = this.getShardWorkspaces(workspace);
    const shards: Tensor[] = shardWss.map(ws =>
      pinned ? ws.allocPinned(ss, type) : ws.alloc(ss, type),
    );
    return new ParallelTensor(workspace, this, par, shards, shape, type, name, pinned);
  }

  wrapTensor(_workspace: WorkspaceBase, _data: number, _allocSize: number, _shape: number[], _type: string, _pinned: boolean): Tensor {
    throw new Error("ParallelOps.wrapTensor not supported; tensor recycling happens at shard level");
  }

  linear(out: Tensor, input: Tensor, weight: Tensor, batch: number, n: number, k: number): void {
    const pOut = out as ParallelTensor;
    const pInput = input as ParallelTensor;
    const pWeight = weight as ParallelTensor;

    if (!(pOut instanceof ParallelTensor) || !(pInput instanceof ParallelTensor) || !(pWeight instanceof ParallelTensor)) {
      throw new Error("ParallelOps.linear requires ParallelTensor arguments");
    }

    const expectedPar = ParallelOps.linearOutputParallelism(pWeight.parallelism, pInput.parallelism);
    if (pOut.parallelism !== expectedPar) {
      throw new Error(`linear: output parallelism ${pOut.parallelism} does not match expected ${expectedPar} for W=${pWeight.parallelism}, X=${pInput.parallelism}`);
    }

    if (pWeight.parallelism === TensorParallelism.Column && pInput.parallelism === TensorParallelism.Replicated) {
      const shardN = n / this.worldSize;
      for (let i = 0; i < this.worldSize; i++) {
        this.devices[i].linear(pOut.shards[i], pInput.shards[i], pWeight.shards[i], batch, shardN, k);
      }
    } else if (pWeight.parallelism === TensorParallelism.Row && pInput.parallelism === TensorParallelism.Row) {
      const shardK = k / this.worldSize;
      for (let i = 0; i < this.worldSize; i++) {
        this.devices[i].linear(pOut.shards[i], pInput.shards[i], pWeight.shards[i], batch, n, shardK);
      }
    } else if (pWeight.parallelism === TensorParallelism.Replicated && pInput.parallelism === TensorParallelism.Replicated) {
      for (let i = 0; i < this.worldSize; i++) {
        this.devices[i].linear(pOut.shards[i], pInput.shards[i], pWeight.shards[i], batch, n, k);
      }
    } else {
      throw new Error(`linear: unsupported parallelism combination W=${pWeight.parallelism}, X=${pInput.parallelism}`);
    }
  }

  synchronize(): void {
    for (const device of this.devices) {
      device.synchronize();
    }
  }

  private elemBytes(type: string): number {
    switch (type) {
      case "BF16": return 2;
      case "I32": return 4;
      case "F32": return 4;
      default: return 1;
    }
  }

  private shapeElems(shape: number[]): number {
    return shape.reduce((a, b) => a * b, 1);
  }

  h2d(dst: Tensor, cpuData: Buffer, size?: number): void {
    const pt = dst as ParallelTensor;
    if (!(pt instanceof ParallelTensor)) {
      throw new Error("ParallelOps.h2d requires ParallelTensor");
    }
    const eb = this.elemBytes(pt.type);

    if (pt.parallelism === TensorParallelism.Replicated || pt.parallelism === TensorParallelism.PartialSum) {
      const sz = size ?? cpuData.length;
      for (let i = 0; i < this.worldSize; i++) {
        this.devices[i].h2d(pt.shards[i], cpuData.subarray(0, sz));
      }
      return;
    }

    if (pt.parallelism === TensorParallelism.Column) {
      const totalElems = this.shapeElems(pt.fullShape);
      const shardBytes = (totalElems / this.worldSize) * eb;
      for (let i = 0; i < this.worldSize; i++) {
        this.devices[i].h2d(pt.shards[i], cpuData.subarray(i * shardBytes, (i + 1) * shardBytes));
      }
      return;
    }

    if (pt.parallelism === TensorParallelism.Row) {
      const outer = pt.fullShape[0];
      const inner = this.shapeElems(pt.fullShape.slice(2));
      const shardDim1 = pt.fullShape[1] / this.worldSize;
      const fullStride = pt.fullShape[1] * inner * eb;
      const shardStride = shardDim1 * inner * eb;
      for (let i = 0; i < this.worldSize; i++) {
        const shardBuf = Buffer.alloc(outer * shardStride);
        for (let r = 0; r < outer; r++) {
          cpuData.copy(
            shardBuf,
            r * shardStride,
            r * fullStride + i * shardStride,
            r * fullStride + i * shardStride + shardStride,
          );
        }
        this.devices[i].h2d(pt.shards[i], shardBuf);
      }
      return;
    }

    throw new Error(`ParallelOps.h2d: unsupported parallelism ${pt.parallelism}`);
  }

  d2h(cpuBuf: Buffer, src: Tensor, size?: number): void {
    const pt = src as ParallelTensor;
    if (!(pt instanceof ParallelTensor)) {
      throw new Error("ParallelOps.d2h requires ParallelTensor");
    }
    const eb = this.elemBytes(pt.type);

    if (pt.parallelism === TensorParallelism.Replicated) {
      const shardBytes = this.shapeElems(pt.shards[0].shape) * eb;
      this.devices[0].d2h(cpuBuf.subarray(0, shardBytes), pt.shards[0]);
      return;
    }

    if (pt.parallelism === TensorParallelism.Column) {
      const totalElems = this.shapeElems(pt.fullShape);
      const shardBytes = (totalElems / this.worldSize) * eb;
      for (let i = 0; i < this.worldSize; i++) {
        this.devices[i].d2h(cpuBuf.subarray(i * shardBytes, (i + 1) * shardBytes), pt.shards[i]);
      }
      return;
    }

    if (pt.parallelism === TensorParallelism.Row) {
      const outer = pt.fullShape[0];
      const inner = this.shapeElems(pt.fullShape.slice(2));
      const shardDim1 = pt.fullShape[1] / this.worldSize;
      const fullStride = pt.fullShape[1] * inner * eb;
      const shardStride = shardDim1 * inner * eb;
      for (let i = 0; i < this.worldSize; i++) {
        const shardBuf = Buffer.alloc(outer * shardStride);
        this.devices[i].d2h(shardBuf, pt.shards[i]);
        for (let r = 0; r < outer; r++) {
          shardBuf.copy(
            cpuBuf,
            r * fullStride + i * shardStride,
            r * shardStride,
            r * shardStride + shardStride,
          );
        }
      }
      return;
    }

    if (pt.parallelism === TensorParallelism.PartialSum) {
      const totalElems = this.shapeElems(pt.fullShape);
      const shardBytes = totalElems * eb;

      if (pt.type === "F32") {
        const result = new Float32Array(totalElems);
        for (let i = 0; i < this.worldSize; i++) {
          const shardBuf = Buffer.alloc(shardBytes);
          this.devices[i].d2h(shardBuf, pt.shards[i]);
          const shardArr = new Float32Array(shardBuf.buffer, shardBuf.byteOffset, totalElems);
          for (let j = 0; j < totalElems; j++) {
            result[j] += shardArr[j];
          }
        }
        Buffer.from(result.buffer, result.byteOffset, result.byteLength).copy(cpuBuf, 0);
      } else if (pt.type === "BF16") {
        const result = new Float32Array(totalElems);
        for (let i = 0; i < this.worldSize; i++) {
          const shardBuf = Buffer.alloc(shardBytes);
          this.devices[i].d2h(shardBuf, pt.shards[i]);
          const shardF32 = bf16BytesToF32(shardBuf);
          for (let j = 0; j < totalElems; j++) {
            result[j] += shardF32[j];
          }
        }
        const bf16Buf = f32ToBf16Bytes(result);
        bf16Buf.copy(cpuBuf, 0);
      } else if (pt.type === "I32") {
        const result = new Int32Array(totalElems);
        for (let i = 0; i < this.worldSize; i++) {
          const shardBuf = Buffer.alloc(shardBytes);
          this.devices[i].d2h(shardBuf, pt.shards[i]);
          const shardArr = new Int32Array(shardBuf.buffer, shardBuf.byteOffset, totalElems);
          for (let j = 0; j < totalElems; j++) {
            result[j] += shardArr[j];
          }
        }
        Buffer.from(result.buffer, result.byteOffset, result.byteLength).copy(cpuBuf, 0);
      } else {
        throw new Error(`ParallelOps.d2h with PartialSum does not support type ${pt.type}`);
      }
      return;
    }

    throw new Error(`ParallelOps.d2h: unsupported parallelism ${pt.parallelism}`);
  }

  private assertParallel(name: string, tensor: ParallelTensor, ...allowed: TensorParallelism[]): void {
    if (!allowed.includes(tensor.parallelism)) {
      throw new Error(`${name}: unsupported parallelism ${tensor.parallelism}, expected ${allowed.join(" or ")}`);
    }
  }

  private cast(tensor: Tensor): ParallelTensor {
    return tensor as ParallelTensor;
  }

  siluAndMul(out: Tensor, gate: Tensor, up: Tensor, intermediate: number, batch: number): void {
    const pOut = this.cast(out);
    const pGate = this.cast(gate);
    const pUp = this.cast(up);
    if (pGate.parallelism !== pUp.parallelism) {
      throw new Error(`siluAndMul: gate parallelism ${pGate.parallelism} != up parallelism ${pUp.parallelism}`);
    }
    const shardIntermediate = pGate.parallelism === TensorParallelism.Row || pGate.parallelism === TensorParallelism.Column
      ? intermediate / this.worldSize
      : intermediate;
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].siluAndMul(pOut.shards[i], pGate.shards[i], pUp.shards[i], shardIntermediate, batch);
    }
  }

  fill(out: Tensor, value: number, n: number): void {
    const pOut = this.cast(out);
    const shardN = pOut.parallelism === TensorParallelism.Row || pOut.parallelism === TensorParallelism.Column
      ? n / this.worldSize
      : n;
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].fill(pOut.shards[i], value, shardN);
    }
  }

  arange(out: Tensor, start: number, step: number, count: number): void {
    const pOut = this.cast(out);
    this.assertParallel("arange", pOut, TensorParallelism.Replicated, TensorParallelism.PartialSum);
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].arange(pOut.shards[i], start, step, count);
    }
  }

  rmsnorm(out: Tensor, input: Tensor, weight: Tensor, eps: number, dim: number, batch: number): void {
    const pOut = this.cast(out);
    const pInput = this.cast(input);
    const pWeight = this.cast(weight);

    if (pInput.parallelism === TensorParallelism.Row || pInput.parallelism === TensorParallelism.Column) {
      const gathered = pInput.allGather(pInput.workspace);
      this.rmsnorm(out, gathered, weight, eps, dim, batch);
      return;
    }

    if (pInput.parallelism === TensorParallelism.PartialSum) {
      pInput.allReduce();
      this.rmsnorm(out, pInput, weight, eps, dim, batch);
      return;
    }

    this.assertParallel("rmsnorm input", pInput, TensorParallelism.Replicated);
    this.assertParallel("rmsnorm weight", pWeight, TensorParallelism.Replicated);
    this.assertParallel("rmsnorm output", pOut, TensorParallelism.Replicated);

    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].rmsnorm(pOut.shards[i], pInput.shards[i], pWeight.shards[i], eps, dim, batch);
    }
  }

  fusedAddRmsnorm(out: Tensor, residual: Tensor, inputA: Tensor, inputB: Tensor, weight: Tensor, eps: number, dim: number, batch: number): void {
    const pOut = this.cast(out);
    const pResidual = this.cast(residual);
    const pInputA = this.cast(inputA);
    const pInputB = this.cast(inputB);
    const pWeight = this.cast(weight);

    if (pInputA.parallelism === TensorParallelism.PartialSum) {
      pInputA.allReduce();
      this.fusedAddRmsnorm(out, residual, pInputA, inputB, weight, eps, dim, batch);
      return;
    }

    if (pInputA.parallelism === TensorParallelism.Row || pInputA.parallelism === TensorParallelism.Column) {
      const gathered = pInputA.allGather(pInputA.workspace);
      this.fusedAddRmsnorm(out, residual, gathered, inputB, weight, eps, dim, batch);
      return;
    }

    if (pInputB.parallelism === TensorParallelism.PartialSum) {
      pInputB.allReduce();
      this.fusedAddRmsnorm(out, residual, inputA, pInputB, weight, eps, dim, batch);
      return;
    }

    if (pInputB.parallelism === TensorParallelism.Row || pInputB.parallelism === TensorParallelism.Column) {
      const gathered = pInputB.allGather(pInputB.workspace);
      this.fusedAddRmsnorm(out, residual, inputA, gathered, weight, eps, dim, batch);
      return;
    }

    this.assertParallel("fusedAddRmsnorm inputA", pInputA, TensorParallelism.Replicated);
    this.assertParallel("fusedAddRmsnorm inputB", pInputB, TensorParallelism.Replicated);
    this.assertParallel("fusedAddRmsnorm weight", pWeight, TensorParallelism.Replicated);
    this.assertParallel("fusedAddRmsnorm out", pOut, TensorParallelism.Replicated);
    this.assertParallel("fusedAddRmsnorm residual", pResidual, TensorParallelism.Replicated);

    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].fusedAddRmsnorm(pOut.shards[i], pResidual.shards[i], pInputA.shards[i], pInputB.shards[i], pWeight.shards[i], eps, dim, batch);
    }
  }

  fusedNormRope(out: Tensor, input: Tensor, weight: Tensor, cos: Tensor, sin: Tensor, eps: number, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, inStride: number): void {
    const pOut = this.cast(out);
    const pInput = this.cast(input);
    const pWeight = this.cast(weight);
    const pCos = this.cast(cos);
    const pSin = this.cast(sin);

    if (pInput.parallelism === TensorParallelism.PartialSum) {
      pInput.allReduce();
      this.fusedNormRope(out, pInput, weight, cos, sin, eps, ropeDim, headDim, nHeads, seqLen, batch, inStride);
      return;
    }

    if (pInput.parallelism === TensorParallelism.Row) {
      const shardNHeads = nHeads / this.worldSize;
      this.assertParallel("fusedNormRope output", pOut, TensorParallelism.Row);
      this.assertParallel("fusedNormRope weight", pWeight, TensorParallelism.Replicated);
      this.assertParallel("fusedNormRope cos", pCos, TensorParallelism.Replicated);
      this.assertParallel("fusedNormRope sin", pSin, TensorParallelism.Replicated);
      const shardInStride = pInput.fullShape[2] === pInput.fullShape[1]
        ? inStride / this.worldSize
        : inStride;
      for (let i = 0; i < this.worldSize; i++) {
        this.devices[i].fusedNormRope(pOut.shards[i], pInput.shards[i], pWeight.shards[i], pCos.shards[i], pSin.shards[i], eps, ropeDim, headDim, shardNHeads, seqLen, batch, shardInStride);
      }
      return;
    }

    if (pInput.parallelism === TensorParallelism.Column) {
      const gathered = pInput.allGather(pInput.workspace);
      this.fusedNormRope(out, gathered, weight, cos, sin, eps, ropeDim, headDim, nHeads, seqLen, batch, inStride);
      return;
    }

    this.assertParallel("fusedNormRope input", pInput, TensorParallelism.Replicated);
    this.assertParallel("fusedNormRope output", pOut, TensorParallelism.Replicated);
    this.assertParallel("fusedNormRope weight", pWeight, TensorParallelism.Replicated);
    this.assertParallel("fusedNormRope cos", pCos, TensorParallelism.Replicated);
    this.assertParallel("fusedNormRope sin", pSin, TensorParallelism.Replicated);

    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].fusedNormRope(pOut.shards[i], pInput.shards[i], pWeight.shards[i], pCos.shards[i], pSin.shards[i], eps, ropeDim, headDim, nHeads, seqLen, batch, inStride);
    }
  }

  embedding(out: Tensor, table: Tensor, ids: Tensor, hidden: number, seqLen: number): void {
    const pOut = this.cast(out);
    const pTable = this.cast(table);
    const pIds = this.cast(ids);

    this.assertParallel("embedding ids", pIds, TensorParallelism.Replicated, TensorParallelism.PartialSum);

    if (pTable.parallelism === TensorParallelism.Row) {
      const shardHidden = hidden / this.worldSize;
      for (let i = 0; i < this.worldSize; i++) {
        this.devices[i].embedding(pOut.shards[i], pTable.shards[i], pIds.shards[i], shardHidden, seqLen);
      }
      return;
    }

    this.assertParallel("embedding table", pTable, TensorParallelism.Replicated);
    this.assertParallel("embedding output", pOut, TensorParallelism.Replicated);
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].embedding(pOut.shards[i], pTable.shards[i], pIds.shards[i], hidden, seqLen);
    }
  }

  argmax(outIndex: Tensor, input: Tensor, dim: number, batch: number): void {
    const pOut = this.cast(outIndex);
    const pInput = this.cast(input);

    if (pInput.parallelism === TensorParallelism.PartialSum) {
      pInput.allReduce();
      this.argmax(outIndex, pInput, dim, batch);
      return;
    }

    if (pInput.parallelism === TensorParallelism.Row || pInput.parallelism === TensorParallelism.Column) {
      const gathered = pInput.allGather(pInput.workspace);
      this.argmax(outIndex, gathered, dim, batch);
      return;
    }

    this.assertParallel("argmax input", pInput, TensorParallelism.Replicated);
    this.assertParallel("argmax output", pOut, TensorParallelism.Replicated);
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].argmax(pOut.shards[i], pInput.shards[i], dim, batch);
    }
  }

  indexSelect(out: Tensor, src: Tensor, indices: Tensor, dim: number, k: number): void {
    const pOut = this.cast(out);
    const pSrc = this.cast(src);
    const pIndices = this.cast(indices);

    this.assertParallel("indexSelect src", pSrc, TensorParallelism.Replicated);
    this.assertParallel("indexSelect indices", pIndices, TensorParallelism.Replicated, TensorParallelism.PartialSum);
    this.assertParallel("indexSelect output", pOut, TensorParallelism.Replicated);

    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].indexSelect(pOut.shards[i], pSrc.shards[i], pIndices.shards[i], dim, k);
    }
  }

  gateSigmoidMul(attnOut: Tensor, gateInterleaved: Tensor, batchSeq: number, numHeads: number, headDim: number): void {
    const pOut = this.cast(attnOut);
    const pGate = this.cast(gateInterleaved);

    const shardNumHeads = pOut.parallelism === TensorParallelism.Row || pOut.parallelism === TensorParallelism.Column
      ? numHeads / this.worldSize
      : numHeads;

    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].gateSigmoidMul(pOut.shards[i], pGate.shards[i], batchSeq, shardNumHeads, headDim);
    }
  }

  rmsnormGated(output: Tensor, input: Tensor, gate: Tensor, weight: Tensor, eps: number, dim: number, batch: number): void {
    const pOutput = this.cast(output);
    const pInput = this.cast(input);
    const pGate = this.cast(gate);
    const pWeight = this.cast(weight);

    if (pInput.parallelism === TensorParallelism.Row || pInput.parallelism === TensorParallelism.Column) {
      const gathered = pInput.allGather(pInput.workspace);
      this.rmsnormGated(output, gathered, gate, weight, eps, dim, batch);
      return;
    }

    if (pInput.parallelism === TensorParallelism.PartialSum) {
      pInput.allReduce();
      this.rmsnormGated(output, pInput, gate, weight, eps, dim, batch);
      return;
    }

    this.assertParallel("rmsnormGated input", pInput, TensorParallelism.Replicated);
    this.assertParallel("rmsnormGated gate", pGate, TensorParallelism.Replicated);
    this.assertParallel("rmsnormGated weight", pWeight, TensorParallelism.Replicated);
    this.assertParallel("rmsnormGated output", pOutput, TensorParallelism.Replicated);

    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].rmsnormGated(pOutput.shards[i], pInput.shards[i], pGate.shards[i], pWeight.shards[i], eps, dim, batch);
    }
  }

  rotaryEmbedding(cosOut: Tensor, sinOut: Tensor, invFreq: Tensor, positionIds: Tensor, dimHalf: number, batch: number, seqLen: number): void {
    const pCosOut = this.cast(cosOut);
    const pSinOut = this.cast(sinOut);
    const pInvFreq = this.cast(invFreq);
    const pPositionIds = this.cast(positionIds);

    this.assertParallel("rotaryEmbedding invFreq", pInvFreq, TensorParallelism.Replicated);
    this.assertParallel("rotaryEmbedding positionIds", pPositionIds, TensorParallelism.Replicated);
    this.assertParallel("rotaryEmbedding cosOut", pCosOut, TensorParallelism.Replicated);
    this.assertParallel("rotaryEmbedding sinOut", pSinOut, TensorParallelism.Replicated);

    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].rotaryEmbedding(pCosOut.shards[i], pSinOut.shards[i], pInvFreq.shards[i], pPositionIds.shards[i], dimHalf, batch, seqLen);
    }
  }

  fp8LinearDecode(bf16Out: Tensor, bf16Input: Tensor, fp8Weight: Tensor, weightScale: Tensor, m: number, n: number, k: number): void {
    const pOut = this.cast(bf16Out);
    const pInput = this.cast(bf16Input);
    const pWeight = this.cast(fp8Weight);
    const pScale = this.cast(weightScale);

    if (pWeight.parallelism === TensorParallelism.Column && pInput.parallelism === TensorParallelism.Replicated) {
      const shardN = n / this.worldSize;
      for (let i = 0; i < this.worldSize; i++) {
        this.devices[i].fp8LinearDecode(pOut.shards[i], pInput.shards[i], pWeight.shards[i], pScale.shards[i], m, shardN, k);
      }
    } else if (pWeight.parallelism === TensorParallelism.Row && pInput.parallelism === TensorParallelism.Row) {
      const shardK = k / this.worldSize;
      for (let i = 0; i < this.worldSize; i++) {
        this.devices[i].fp8LinearDecode(pOut.shards[i], pInput.shards[i], pWeight.shards[i], pScale.shards[i], m, n, shardK);
      }
    } else if (pWeight.parallelism === TensorParallelism.Replicated && pInput.parallelism === TensorParallelism.Replicated) {
      for (let i = 0; i < this.worldSize; i++) {
        this.devices[i].fp8LinearDecode(pOut.shards[i], pInput.shards[i], pWeight.shards[i], pScale.shards[i], m, n, k);
      }
    } else {
      throw new Error(`fp8LinearDecode: unsupported parallelism W=${pWeight.parallelism}, X=${pInput.parallelism}`);
    }
  }

  kvCacheWrite(srcK: Tensor, srcV: Tensor, dstK: Tensor, dstV: Tensor, slotMapping: Tensor, batchSize: number, nKv: number, hd: number, pageSize: number, srcKTokenStride: number, srcKHeadStride: number, srcVTokenStride: number, srcVHeadStride: number): void {
    const pSrcK = this.cast(srcK);
    const pSrcV = this.cast(srcV);
    const pDstK = this.cast(dstK);
    const pDstV = this.cast(dstV);
    const pSlotMapping = this.cast(slotMapping);

    this.assertParallel("kvCacheWrite slotMapping", pSlotMapping, TensorParallelism.Replicated);

    const shardNKv = nKv / this.worldSize;
    const isRowPar = pSrcK.parallelism === TensorParallelism.Row || pSrcK.parallelism === TensorParallelism.Column;
    const shardKTokenStride = isRowPar && srcKTokenStride !== hd ? srcKTokenStride / this.worldSize : srcKTokenStride;
    const shardVTokenStride = isRowPar ? srcVTokenStride / this.worldSize : srcVTokenStride;
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].kvCacheWrite(pSrcK.shards[i], pSrcV.shards[i], pDstK.shards[i], pDstV.shards[i], pSlotMapping.shards[i], batchSize, shardNKv, hd, pageSize, shardKTokenStride, srcKHeadStride, shardVTokenStride, srcVHeadStride);
    }
  }

  writePinned(dst: Tensor, src: Buffer, size?: number): void {
    const pDst = this.cast(dst);
    this.assertParallel("writePinned", pDst, TensorParallelism.Replicated);
    this.devices[0].writePinned(pDst.shards[0], src, size);
    for (let i = 1; i < this.worldSize; i++) {
      this.devices[i].h2d(pDst.shards[i], src, size ?? src.length);
    }
  }

  gdnRecurrentStep(): void { throw new Error("ParallelOps.gdnRecurrentStep not implemented"); }
  gdnPrefill(): void { throw new Error("ParallelOps.gdnPrefill not implemented"); }
  causalConv1d(): void { throw new Error("ParallelOps.causalConv1d not implemented"); }
  causalConv1dUpdate(): void { throw new Error("ParallelOps.causalConv1dUpdate not implemented"); }
  batchDecodePlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, indptrH: Tensor, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, enableCudaGraph: boolean): void {
    const pFloatWs = this.cast(floatWs);
    const pIntWs = this.cast(intWs);
    const pPinnedIntWs = this.cast(pinnedIntWs);
    const pPlanInfo = this.cast(planInfo);
    const pIndptrH = this.cast(indptrH);
    this.assertParallel("batchDecodePlan floatWs", pFloatWs, TensorParallelism.Replicated);
    this.assertParallel("batchDecodePlan intWs", pIntWs, TensorParallelism.Replicated);
    this.assertParallel("batchDecodePlan planInfo", pPlanInfo, TensorParallelism.Replicated);
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].batchDecodePlan(pFloatWs.shards[i], floatWsSize, pIntWs.shards[i], pPinnedIntWs.shards[i], intWsSize, pPlanInfo.shards[i], pIndptrH.shards[i], batchSize, numQoHeads / this.worldSize, numKvHeads / this.worldSize, headDim, pageSize, enableCudaGraph);
    }
  }

  batchDecodeRun(q: Tensor, o: Tensor, kData: Tensor, vData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, smScale: number): void {
    const pQ = this.cast(q);
    const pO = this.cast(o);
    const pKData = this.cast(kData);
    const pVData = this.cast(vData);
    const pIndices = this.cast(indices);
    const pIndptrD = this.cast(indptrD);
    const pLastPageLen = this.cast(lastPageLen);
    const pFloatWs = this.cast(floatWs);
    const pIntWs = this.cast(intWs);
    const pPlanInfo = this.cast(planInfo);
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].batchDecodeRun(pQ.shards[i], pO.shards[i], pKData.shards[i], pVData.shards[i], pIndices.shards[i], pIndptrD.shards[i], pLastPageLen.shards[i], pFloatWs.shards[i], pIntWs.shards[i], pPlanInfo.shards[i], batchSize, numQoHeads / this.worldSize, numKvHeads / this.worldSize, headDim, pageSize, smScale);
    }
  }

  batchPrefillPagedPlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, qoIndptrH: Tensor, pagedKvIndptrH: Tensor, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, maskMode: number): void {
    const pFloatWs = this.cast(floatWs);
    const pIntWs = this.cast(intWs);
    const pPinnedIntWs = this.cast(pinnedIntWs);
    const pPlanInfo = this.cast(planInfo);
    const pQoIndptrH = this.cast(qoIndptrH);
    const pPagedKvIndptrH = this.cast(pagedKvIndptrH);
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].batchPrefillPagedPlan(pFloatWs.shards[i], floatWsSize, pIntWs.shards[i], pPinnedIntWs.shards[i], intWsSize, pPlanInfo.shards[i], pQoIndptrH.shards[i], pPagedKvIndptrH.shards[i], totalQoRows, batchSize, numQoHeads / this.worldSize, numKvHeads / this.worldSize, headDim, pageSize, maskMode);
    }
  }

  batchPrefillPagedRun(q: Tensor, o: Tensor, kData: Tensor, vData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, floatWs: Tensor, intWs: Tensor, qIndptrD: Tensor, planInfo: Tensor, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, qStrideN: number, qStrideH: number, maskMode: number, smScale: number): void {
    const pQ = this.cast(q);
    const pO = this.cast(o);
    const pKData = this.cast(kData);
    const pVData = this.cast(vData);
    const pIndices = this.cast(indices);
    const pIndptrD = this.cast(indptrD);
    const pLastPageLen = this.cast(lastPageLen);
    const pFloatWs = this.cast(floatWs);
    const pIntWs = this.cast(intWs);
    const pQIndptrD = this.cast(qIndptrD);
    const pPlanInfo = this.cast(planInfo);
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].batchPrefillPagedRun(pQ.shards[i], pO.shards[i], pKData.shards[i], pVData.shards[i], pIndices.shards[i], pIndptrD.shards[i], pLastPageLen.shards[i], pFloatWs.shards[i], pIntWs.shards[i], pQIndptrD.shards[i], pPlanInfo.shards[i], totalQoRows, batchSize, numQoHeads / this.worldSize, numKvHeads / this.worldSize, headDim, pageSize, qStrideN, qStrideH, maskMode, smScale);
    }
  }

  mmapLoad(gpuDst: Tensor, mmapPtr: number, offset: number, nbytes: number): void {
    const pDst = this.cast(gpuDst);

    if (pDst.parallelism === TensorParallelism.Replicated || pDst.parallelism === TensorParallelism.PartialSum) {
      for (let i = 0; i < this.worldSize; i++) {
        this.devices[i].mmapLoad(pDst.shards[i], mmapPtr, offset, nbytes);
      }
      return;
    }

    if (pDst.parallelism === TensorParallelism.Column) {
      const shardElems = pDst.shards[0].shape.reduce((a, b) => a * b, 1);
      const shardBytes = shardElems * this.elemBytes(pDst.type);
      for (let i = 0; i < this.worldSize; i++) {
        this.devices[i].mmapLoad(pDst.shards[i], mmapPtr, offset + i * shardBytes, shardBytes);
      }
      return;
    }

    if (pDst.parallelism === TensorParallelism.Row) {
      const outer = pDst.fullShape[0];
      const inner = pDst.fullShape.slice(2).reduce((a, b) => a * b, 1);
      const fullDim1 = pDst.fullShape[1];
      const shardDim1 = fullDim1 / this.worldSize;
      const eb = this.elemBytes(pDst.type);
      const srcPitch = fullDim1 * inner * eb;
      const dstPitch = shardDim1 * inner * eb;
      const srcBase = mmapPtr + offset;
      for (let i = 0; i < this.worldSize; i++) {
        this.devices[i].memcpy2d(
          pDst.shards[i].data, dstPitch,
          srcBase + i * dstPitch, srcPitch,
          dstPitch, outer,
          MEMCPY_H2D,
        );
      }
      return;
    }

    throw new Error(`mmapLoad: unsupported parallelism ${pDst.parallelism}`);
  }

  sampleBatch(outTokens: Tensor, topkVals: Tensor, topkIdxs: Tensor, workspace: Tensor, logits: Tensor, penaltyTokens: Tensor, penaltyOffsets: Tensor, vocabSize: number, batchSize: number, temperatures: Tensor, repPenalties: Tensor, presPenalties: Tensor, topKs: Tensor, topPs: Tensor, randomVals: Tensor, maxEffectiveK: number): void {
    const pLogits = this.cast(logits);
    if (pLogits.parallelism === TensorParallelism.Row || pLogits.parallelism === TensorParallelism.Column) {
      const gathered = pLogits.allGather(pLogits.workspace);
      this.sampleBatch(outTokens, topkVals, topkIdxs, workspace, gathered, penaltyTokens, penaltyOffsets, vocabSize, batchSize, temperatures, repPenalties, presPenalties, topKs, topPs, randomVals, maxEffectiveK);
      gathered[Symbol.dispose]();
      return;
    }
    if (pLogits.parallelism === TensorParallelism.PartialSum) {
      pLogits.allReduce();
      this.sampleBatch(outTokens, topkVals, topkIdxs, workspace, logits, penaltyTokens, penaltyOffsets, vocabSize, batchSize, temperatures, repPenalties, presPenalties, topKs, topPs, randomVals, maxEffectiveK);
      return;
    }
    const pOut = this.cast(outTokens);
    const pTopkVals = this.cast(topkVals);
    const pTopkIdxs = this.cast(topkIdxs);
    const pWorkspace = this.cast(workspace);
    const pPenaltyTokens = this.cast(penaltyTokens);
    const pPenaltyOffsets = this.cast(penaltyOffsets);
    const pTemps = this.cast(temperatures);
    const pRepPen = this.cast(repPenalties);
    const pPresPen = this.cast(presPenalties);
    const pTopKs = this.cast(topKs);
    const pTopPs = this.cast(topPs);
    const pRandomVals = this.cast(randomVals);
    this.assertParallel("sampleBatch logits", pLogits, TensorParallelism.Replicated);
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].sampleBatch(pOut.shards[i], pTopkVals.shards[i], pTopkIdxs.shards[i], pWorkspace.shards[i], pLogits.shards[i], pPenaltyTokens.shards[i], pPenaltyOffsets.shards[i], vocabSize, batchSize, pTemps.shards[i], pRepPen.shards[i], pPresPen.shards[i], pTopKs.shards[i], pTopPs.shards[i], pRandomVals.shards[i], maxEffectiveK);
    }
  }

  graphBeginCapture(): void {
    for (const device of this.devices) {
      device.graphBeginCapture();
    }
  }

  graphEndCapture(): number {
    const graphs = this.devices.map(d => d.graphEndCapture());
    return graphs[0];
  }

  graphInstantiate(graph: number): number {
    const execs = this.devices.map(d => d.graphInstantiate(graph));
    return execs[0];
  }

  graphLaunch(graphExec: number): void {
    for (const device of this.devices) {
      device.graphLaunch(graphExec);
    }
  }

  graphDestroy(graph: number): void {
    for (const device of this.devices) {
      device.graphDestroy(graph);
    }
  }

  graphExecDestroy(graphExec: number): void {
    for (const device of this.devices) {
      device.graphExecDestroy(graphExec);
    }
  }
}
