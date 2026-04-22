import { DeviceOps, TensorParallelism } from "./device_ops";
import { GlmOps, f32ToBf16Bytes, bf16BytesToF32, NCCL_BFLOAT16, NCCL_FLOAT32, NCCL_INT32, NCCL_SUM } from "./glm_ops";
import { Tensor } from "./tensor";
import { WorkspaceBase } from "./workspace";

export class ParallelTensor extends Tensor {
  parallelism: TensorParallelism;
  readonly shards: readonly Tensor[];
  readonly fullShape: number[];
  private readonly devices: readonly GlmOps[];
  private readonly parallelOps: ParallelOps;
  private _disposed = false;

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
    this._disposed = true;
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
    if (this._disposed) return;
    this._disposed = true;
    this.workspace.tracked.delete(this);
    for (const shard of this.shards) {
      if (shard.data !== 0) {
        shard[Symbol.dispose]();
      }
    }
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
    this.devices[0].native.ncclGroupStart();
    for (let i = 0; i < this.devices.length; i++) {
      this.devices[i].native.ncclAllReduce(
        comms[i], this.devices[i].ctx,
        this.shards[i].data, this.shards[i].data,
        count, dtype, NCCL_SUM,
      );
    }
    this.devices[0].native.ncclGroupEnd();
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
      this.devices[0].native.ncclGroupStart();
      for (let i = 0; i < this.devices.length; i++) {
        this.devices[i].native.ncclAllGather(
          comms[i], this.devices[i].ctx,
          this.shards[i].data, output.shards[i].data,
          count, dtype,
        );
      }
      this.devices[0].native.ncclGroupEnd();
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

      this.devices[0].native.ncclGroupStart();
      for (let i = 0; i < this.devices.length; i++) {
        this.devices[i].native.ncclAllGather(
          comms[i], this.devices[i].ctx,
          this.shards[i].data, tempTensors[i].data,
          count, dtype,
        );
      }
      this.devices[0].native.ncclGroupEnd();

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
      this.comms = devices[0].native.ncclCommInitAll(deviceIds);
    } else {
      this.comms = [];
    }
  }

  free(): void {
    if (this.comms.length > 0) {
      for (const comm of this.comms) {
        this.devices[0].native.ncclCommDestroy(comm);
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

  freeBuf(_ptr: Tensor): void { throw new Error("ParallelOps.freeBuf not implemented"); }
  freePinned(_ptr: Tensor): void { throw new Error("ParallelOps.freePinned not implemented"); }
  rmsnorm(): void { throw new Error("ParallelOps.rmsnorm not implemented"); }
  fusedAddRmsnorm(): void { throw new Error("ParallelOps.fusedAddRmsnorm not implemented"); }
  fusedNormRope(): void { throw new Error("ParallelOps.fusedNormRope not implemented"); }
  siluAndMul(): void { throw new Error("ParallelOps.siluAndMul not implemented"); }
  embedding(): void { throw new Error("ParallelOps.embedding not implemented"); }
  fill(): void { throw new Error("ParallelOps.fill not implemented"); }
  arange(): void { throw new Error("ParallelOps.arange not implemented"); }
  argmax(): void { throw new Error("ParallelOps.argmax not implemented"); }
  indexSelect(): void { throw new Error("ParallelOps.indexSelect not implemented"); }
  kvCacheWrite(): void { throw new Error("ParallelOps.kvCacheWrite not implemented"); }
  rotaryEmbedding(): void { throw new Error("ParallelOps.rotaryEmbedding not implemented"); }
  fp8LinearDecode(): void { throw new Error("ParallelOps.fp8LinearDecode not implemented"); }
  gdnRecurrentStep(): void { throw new Error("ParallelOps.gdnRecurrentStep not implemented"); }
  gdnPrefill(): void { throw new Error("ParallelOps.gdnPrefill not implemented"); }
  causalConv1d(): void { throw new Error("ParallelOps.causalConv1d not implemented"); }
  causalConv1dUpdate(): void { throw new Error("ParallelOps.causalConv1dUpdate not implemented"); }
  rmsnormGated(): void { throw new Error("ParallelOps.rmsnormGated not implemented"); }
  gateSigmoidMul(): void { throw new Error("ParallelOps.gateSigmoidMul not implemented"); }
  writePinned(): void { throw new Error("ParallelOps.writePinned not implemented"); }
  batchDecodePlan(): void { throw new Error("ParallelOps.batchDecodePlan not implemented"); }
  batchDecodeRun(): void { throw new Error("ParallelOps.batchDecodeRun not implemented"); }
  batchPrefillPagedPlan(): void { throw new Error("ParallelOps.batchPrefillPagedPlan not implemented"); }
  batchPrefillPagedRun(): void { throw new Error("ParallelOps.batchPrefillPagedRun not implemented"); }
  sampleBatch(): void { throw new Error("ParallelOps.sampleBatch not implemented"); }
  graphBeginCapture(): void { throw new Error("ParallelOps.graphBeginCapture not implemented"); }
  graphEndCapture(): number { throw new Error("ParallelOps.graphEndCapture not implemented"); }
  graphInstantiate(): number { throw new Error("ParallelOps.graphInstantiate not implemented"); }
  graphLaunch(): void { throw new Error("ParallelOps.graphLaunch not implemented"); }
  graphDestroy(): void { throw new Error("ParallelOps.graphDestroy not implemented"); }
  graphExecDestroy(): void { throw new Error("ParallelOps.graphExecDestroy not implemented"); }
  mmapOpen(): number { throw new Error("ParallelOps.mmapOpen not implemented"); }
  mmapLoad(): void { throw new Error("ParallelOps.mmapLoad not implemented"); }
  mmapClose(): void { throw new Error("ParallelOps.mmapClose not implemented"); }
}
