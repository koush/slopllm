import { DeviceOps, StridedMmap, TensorParallelism } from "./device_ops";
import { GlmOps, getNativeAddon, f32ToBf16Bytes, bf16BytesToF32, NCCL_BFLOAT16, NCCL_FLOAT32, NCCL_INT32, NCCL_SUM } from "./glm_ops";
import { MemcpyKind } from "./tensor";
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
    view: ParallelTensor | undefined,
  ) {
    super(workspace, 0, 0, fullShape, type, name, pinned, view);
    this.parallelOps = parallelOps;
    this.devices = parallelOps.devices;
    this.parallelism = parallelism;
    this.shards = shards;
    this.fullShape = fullShape;
    for (let i = 0; i < shards.length; i++) {
      shards[i].setName(name);
    }
  }

  private get worldSize(): number {
    return this.shards.length;
  }

  private static elemBytes(type: string): number {
    switch (type) {
      case "BF16": return 2;
      case "I32": return 4;
      case "I64": return 8;
      case "F32": return 4;
      case "U8": return 1;
      case "F8_E5M2": return 1;
      case "F8_E4M3": return 1;
      default:
        console.warn(`Unknown type ${type}, assuming 1 byte per element`);
        return 1;
    }
  }

  private static shapeElems(shape: number[]): number {
    return shape.reduce((a, b) => a * b, 1);
  }

  private assertParallel(name: string, tensor: ParallelTensor, ...allowed: TensorParallelism[]): void {
    if (!allowed.includes(tensor.parallelism)) {
      throw new Error(`${name}: unsupported parallelism ${tensor.parallelism}, expected ${allowed.join(" or ")}`);
    }
  }

  private shardDim(dim: number, name: string): number {
    if (dim % this.worldSize !== 0) {
      throw new Error(`${name}: dimension ${dim} not divisible by worldSize=${this.worldSize}`);
    }
    return dim / this.worldSize;
  }

  private cast(tensor: Tensor): ParallelTensor {
    return tensor as ParallelTensor;
  }

  free(): void {
    for (let i = 0; i < this.shards.length; i++) {
      const shard = this.shards[i];
      shard.free();
    }
  }

  [Symbol.dispose](): void {
    if (this.name !== undefined) {
      throw new Error("Cannot dispose named tensor");
    }
    this.workspace.tracked.delete(this);
    for (const shard of this.shards) {
      shard[Symbol.dispose]();
    }
    (this.shards as Tensor[]).length = 0;
  }

  override setName(name: string | undefined): void {
    if (name === undefined) {
      if (this.name !== undefined) {
        this.workspace.tensors.delete(this.name);
        (this as { name: string | undefined }).name = undefined;
      }
    } else {
      if (this.name) throw new Error(`Tensor already has name ${this.name}, cannot rename to ${name}`);
      (this as { name: string }).name = name;
      this.workspace.tracked.delete(this);
      this.workspace.exported.delete(this);
      this.workspace.tensors.set(name, this);
    }
    for (let i = 0; i < this.shards.length; i++) {
      this.shards[i].setName(name);
    }
  }

  shard(rank: number): Tensor {
    return this.shards[rank];
  }

  private static computeReshapeParallelism(oldShape: number[], newShape: number[], oldPar: TensorParallelism, worldSize: number): TensorParallelism {
    if (oldShape.length !== 2 || newShape.length !== 2) return oldPar;
    if (oldPar === TensorParallelism.Row) {
      if (newShape[1] === oldShape[1]) return TensorParallelism.Row;
      if (newShape[0] % worldSize === 0) return TensorParallelism.Column;
      return TensorParallelism.Row;
    }
    if (oldPar === TensorParallelism.Column) {
      if (newShape[0] === oldShape[0]) return TensorParallelism.Column;
      if (newShape[1] % worldSize === 0) return TensorParallelism.Row;
      return TensorParallelism.Column;
    }
    return oldPar;
  }

  override reshape(newShape: number[]): Tensor {
    const current = this.shape.reduce((a, b) => a * b, 1);
    const target = newShape.reduce((a, b) => a * b, 1);
    if (current !== target) {
      throw new Error(`reshape: cannot reshape [${this.shape}] (${current} elements) to [${newShape}] (${target} elements)`);
    }
    const newPar = ParallelTensor.computeReshapeParallelism(this.shape, newShape, this.parallelism, this.worldSize);
    const newShardShape = this.parallelOps.shardShape(newShape, newPar);
    const reshapedShards: Tensor[] = this.shards.map(s => s.reshape(newShardShape));
    return new ParallelTensor(
      this.workspace, this.parallelOps, newPar,
      reshapedShards, newShape, this.type, undefined, this.pinned, this,
    );
  }

  allReduce(): ParallelTensor {
    if (this.parallelism !== TensorParallelism.PartialSum) {
      throw new Error(`allReduce requires PartialSum tensor, got ${this.parallelism}`);
    }
    // console.warn(`Performing allReduce on PartialSum tensor with shape [${this.fullShape}] and type ${this.type}, this may be slow`);
    const count = this.shards[0].shape.reduce((a, b) => a * b, 1);
    const dtype = this.parallelOps.ncclDatatype(this.type);
    this.parallelOps.doAllReduce(this.shards, count, dtype);
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
    const elemBytes = ParallelTensor.elemBytes(this.type);

    if (this.parallelOps.tryP2PAllGather(this.shards, output.shards, count, elemBytes, this.parallelism, this.fullShape)) {
      return output;
    }

    // console.warn(`Falling back to NCCL allGather for parallelism ${this.parallelism}, this may be slow`);

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
          output.shards[i].memcpy2d(
            r * shardDim1 * inner * eb,
            this.fullShape[1] * inner * eb,
            tempTensors[i].data + r * shardBytes,
            shardDim1 * inner * eb,
            shardDim1 * inner * eb,
            outer,
            MemcpyKind.DeviceToDevice,
          );
        }
        tempTensors[i][Symbol.dispose]();
      }

      return output;
    }

    throw new Error(`allGather: unsupported parallelism ${this.parallelism}`);
  }

  h2d(data: Buffer, size?: number): void {
    const eb = ParallelTensor.elemBytes(this.type);

    if (this.parallelism === TensorParallelism.Replicated || this.parallelism === TensorParallelism.PartialSum) {
      const sz = size ?? data.length;
      for (let i = 0; i < this.worldSize; i++) {
        this.shards[i].h2d(data.subarray(0, sz));
      }
      return;
    }

    if (this.parallelism === TensorParallelism.Column) {
      const totalElems = ParallelTensor.shapeElems(this.fullShape);
      const shardBytes = (totalElems / this.worldSize) * eb;
      for (let i = 0; i < this.worldSize; i++) {
        this.shards[i].h2d(data.subarray(i * shardBytes, (i + 1) * shardBytes));
      }
      return;
    }

    if (this.parallelism === TensorParallelism.Row) {
      const outer = this.fullShape[0];
      const inner = ParallelTensor.shapeElems(this.fullShape.slice(2));
      const shardDim1 = this.fullShape[1] / this.worldSize;
      const fullStride = this.fullShape[1] * inner * eb;
      const shardStride = shardDim1 * inner * eb;
      for (let i = 0; i < this.worldSize; i++) {
        const shardBuf = Buffer.alloc(outer * shardStride);
        for (let r = 0; r < outer; r++) {
          data.copy(
            shardBuf,
            r * shardStride,
            r * fullStride + i * shardStride,
            r * fullStride + i * shardStride + shardStride,
          );
        }
        this.shards[i].h2d(shardBuf);
      }
      return;
    }

    throw new Error(`ParallelTensor.h2d: unsupported parallelism ${this.parallelism}`);
  }

  d2h(buf: Buffer, size?: number): void {
    const eb = ParallelTensor.elemBytes(this.type);

    if (this.parallelism === TensorParallelism.Replicated) {
      const shardBytes = ParallelTensor.shapeElems(this.shards[0].shape) * eb;
      this.shards[0].d2h(buf.subarray(0, shardBytes));
      return;
    }

    if (this.parallelism === TensorParallelism.Column) {
      const totalElems = ParallelTensor.shapeElems(this.fullShape);
      const shardBytes = (totalElems / this.worldSize) * eb;
      for (let i = 0; i < this.worldSize; i++) {
        this.shards[i].d2h(buf.subarray(i * shardBytes, (i + 1) * shardBytes));
      }
      return;
    }

    if (this.parallelism === TensorParallelism.Row) {
      const outer = this.fullShape[0];
      const inner = ParallelTensor.shapeElems(this.fullShape.slice(2));
      const shardDim1 = this.fullShape[1] / this.worldSize;
      const fullStride = this.fullShape[1] * inner * eb;
      const shardStride = shardDim1 * inner * eb;
      for (let i = 0; i < this.worldSize; i++) {
        const shardBuf = Buffer.alloc(outer * shardStride);
        this.shards[i].d2h(shardBuf);
        for (let r = 0; r < outer; r++) {
          shardBuf.copy(
            buf,
            r * fullStride + i * shardStride,
            r * shardStride,
            r * shardStride + shardStride,
          );
        }
      }
      return;
    }

    if (this.parallelism === TensorParallelism.PartialSum) {
      const totalElems = ParallelTensor.shapeElems(this.fullShape);
      const shardBytes = totalElems * eb;

      if (this.type === "F32") {
        const result = new Float32Array(totalElems);
        for (let i = 0; i < this.worldSize; i++) {
          const shardBuf = Buffer.alloc(shardBytes);
          this.shards[i].d2h(shardBuf);
          const shardArr = new Float32Array(shardBuf.buffer, shardBuf.byteOffset, totalElems);
          for (let j = 0; j < totalElems; j++) {
            result[j] += shardArr[j];
          }
        }
        Buffer.from(result.buffer, result.byteOffset, result.byteLength).copy(buf, 0);
      } else if (this.type === "BF16") {
        const result = new Float32Array(totalElems);
        for (let i = 0; i < this.worldSize; i++) {
          const shardBuf = Buffer.alloc(shardBytes);
          this.shards[i].d2h(shardBuf);
          const shardF32 = bf16BytesToF32(shardBuf);
          for (let j = 0; j < totalElems; j++) {
            result[j] += shardF32[j];
          }
        }
        const bf16Buf = f32ToBf16Bytes(result);
        bf16Buf.copy(buf, 0);
      } else if (this.type === "I32") {
        const result = new Int32Array(totalElems);
        for (let i = 0; i < this.worldSize; i++) {
          const shardBuf = Buffer.alloc(shardBytes);
          this.shards[i].d2h(shardBuf);
          const shardArr = new Int32Array(shardBuf.buffer, shardBuf.byteOffset, totalElems);
          for (let j = 0; j < totalElems; j++) {
            result[j] += shardArr[j];
          }
        }
        Buffer.from(result.buffer, result.byteOffset, result.byteLength).copy(buf, 0);
      } else {
        throw new Error(`ParallelTensor.d2h with PartialSum does not support type ${this.type}`);
      }
      return;
    }

    throw new Error(`ParallelTensor.d2h: unsupported parallelism ${this.parallelism}`);
  }

  override linear(weight: Tensor, batch: number): Tensor {
    super.linear(weight, batch);
    const pWeight = weight as ParallelTensor;
    const n = weight.shape[0];
    const WP = pWeight.parallelism;
    const XP = this.parallelism;

    // --- Direct paths (no communication) ---

    // Column weight + Replicated input → Row output
    if (WP === TensorParallelism.Column && XP === TensorParallelism.Replicated) {
      const shards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        shards.push(this.shards[i].linear(pWeight.shards[i], batch));
      }
      return this.parallelOps.wrapShards(this.workspace, shards, [batch, n], this.type, TensorParallelism.Row);
    }

    // Row weight + Row input → PartialSum output
    if (WP === TensorParallelism.Row && XP === TensorParallelism.Row) {
      const shards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        shards.push(this.shards[i].linear(pWeight.shards[i], batch));
      }
      return this.parallelOps.wrapShards(this.workspace, shards, [batch, n], this.type, TensorParallelism.PartialSum);
    }

    // Replicated weight + Replicated input → Replicated output
    if (WP === TensorParallelism.Replicated && XP === TensorParallelism.Replicated) {
      const shards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        shards.push(this.shards[i].linear(pWeight.shards[i], batch));
      }
      return this.parallelOps.wrapShards(this.workspace, shards, [batch, n], this.type, TensorParallelism.Replicated);
    }

    // Replicated weight + Column input → Column output
    if (WP === TensorParallelism.Replicated && XP === TensorParallelism.Column) {
      const shards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        shards.push(this.shards[i].linear(pWeight.shards[i], batch));
      }
      return this.parallelOps.wrapShards(this.workspace, shards, [batch, n], this.type, TensorParallelism.Column);
    }

    // Replicated weight + PartialSum input → PartialSum output (distributivity: Σ(P_i @ W) = (Σ P_i) @ W)
    if (WP === TensorParallelism.Replicated && XP === TensorParallelism.PartialSum) {
      const shards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        shards.push(this.shards[i].linear(pWeight.shards[i], batch));
      }
      return this.parallelOps.wrapShards(this.workspace, shards, [batch, n], this.type, TensorParallelism.PartialSum);
    }

    // PartialSum weight + Replicated input → PartialSum output (distributivity: X @ (Σ W_i) = Σ(X @ W_i))
    if (WP === TensorParallelism.PartialSum && XP === TensorParallelism.Replicated) {
      const shards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        shards.push(this.shards[i].linear(pWeight.shards[i], batch));
      }
      return this.parallelOps.wrapShards(this.workspace, shards, [batch, n], this.type, TensorParallelism.PartialSum);
    }

    // --- Fallback paths (need communication) ---

    // PartialSum input: allReduce to Replicated, then retry (hits Replicated+Column or Replicated+Replicated)
    if (XP === TensorParallelism.PartialSum) {
      this.allReduce();
      return this.linear(weight, batch);
    }

    // PartialSum weight with non-Replicated input: can't allReduce a persistent weight
    if (WP === TensorParallelism.PartialSum) {
      throw new Error(`linear: PartialSum weight requires Replicated input (got ${XP}); allReduce would mutate persistent weight`);
    }

    // Row weight: K dimension mismatch, allGather weight to Replicated
    if (WP === TensorParallelism.Row) {
      using gathered = pWeight.allGather(pWeight.workspace);
      return this.linear(gathered, batch);
    }

    // Row input: K dimension mismatch, allGather input to Replicated
    if (XP === TensorParallelism.Row) {
      using gathered = this.allGather(this.workspace);
      return gathered.linear(weight, batch);
    }

    // Column+Column: allGather input to Replicated, then Column+Replicated → Row
    if (XP === TensorParallelism.Column && WP === TensorParallelism.Column) {
      using gathered = this.allGather(this.workspace);
      return gathered.linear(weight, batch);
    }

    throw new Error(`linear: unsupported parallelism combination W=${WP}, X=${XP}`);
  }

  bmm(B: Tensor, batch: number, M: number, N: number, K: number, transA: boolean = false, transB: boolean = false): Tensor {
    const pB = B as ParallelTensor;

    if (this.parallelism === pB.parallelism) {
      const shardBatch = (this.parallelism === TensorParallelism.Column) ? batch / this.worldSize : batch;
      const outPar = this.parallelism === TensorParallelism.Row
        ? TensorParallelism.PartialSum
        : this.parallelism;
      const outShards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        outShards.push(this.shards[i].bmm(pB.shards[i], shardBatch, M, N, K, transA, transB));
      }
      return this.parallelOps.wrapShards(this.workspace, outShards, [batch * M, N], this.type, outPar);
    }

    if (this.parallelism === TensorParallelism.Column && pB.parallelism === TensorParallelism.Replicated) {
      const shardBatch = batch / this.worldSize;
      const outShards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        outShards.push(this.shards[i].bmm(pB.shards[i], shardBatch, M, N, K, transA, transB));
      }
      return this.parallelOps.wrapShards(this.workspace, outShards, [batch * M, N], this.type, TensorParallelism.Column);
    }

    if (this.parallelism === TensorParallelism.Replicated && pB.parallelism === TensorParallelism.Replicated) {
      const outShards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        outShards.push(this.shards[i].bmm(pB.shards[i], batch, M, N, K, transA, transB));
      }
      return this.parallelOps.wrapShards(this.workspace, outShards, [batch * M, N], this.type, TensorParallelism.Replicated);
    }

    using gatheredA = this.allGather(this.workspace);
    using gatheredB = pB.allGather(pB.workspace);
    return gatheredA.bmm(gatheredB, batch, M, N, K, transA, transB);
  }

  writePointers(tensors: Tensor[]): void {
    super.writePointers(tensors);
    const n = tensors.length;
    for (let d = 0; d < this.worldSize; d++) {
      const shardPtrs: Tensor[] = [];
      for (let i = 0; i < n; i++) {
        shardPtrs.push((tensors[i] as ParallelTensor).shards[d]);
      }
      this.shards[d].writePointers(shardPtrs);
    }
  }

  add(other: Tensor, n?: number): Tensor {
    const pOther = other as ParallelTensor;

    if (this.parallelism === pOther.parallelism) {
      const outShards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        outShards.push(this.shards[i].add(pOther.shards[i], n));
      }
      return this.parallelOps.wrapShards(this.workspace, outShards, this.shape, this.type, this.parallelism);
    }

    if (this.parallelism === TensorParallelism.PartialSum && pOther.parallelism === TensorParallelism.Replicated) {
      const outShards: Tensor[] = [];
      outShards.push(this.shards[0].add(pOther.shards[0], n));
      const shardWss = this.parallelOps.getShardWorkspaces(this.workspace);
      for (let i = 1; i < this.worldSize; i++) {
        const shard = shardWss[i].alloc(this.shards[i].shape, this.type);
        shard.memcpy(this.shards[i]);
        outShards.push(shard);
      }
      return this.parallelOps.wrapShards(this.workspace, outShards, this.shape, this.type, TensorParallelism.PartialSum);
    }
    if (pOther.parallelism === TensorParallelism.PartialSum && this.parallelism === TensorParallelism.Replicated) {
      return pOther.add(this, n);
    }

    if (this.parallelism === TensorParallelism.PartialSum) {
      using gathered = pOther.allGather(pOther.workspace);
      return this.add(gathered, n);
    }
    if (pOther.parallelism === TensorParallelism.PartialSum) {
      using gathered = this.allGather(this.workspace);
      return pOther.add(gathered, n);
    }

    if ((this.parallelism === TensorParallelism.Row || this.parallelism === TensorParallelism.Column) && pOther.parallelism === TensorParallelism.Replicated) {
      using gathered = this.allGather(this.workspace);
      return gathered.add(pOther, n);
    }
    if (this.parallelism === TensorParallelism.Replicated && (pOther.parallelism === TensorParallelism.Row || pOther.parallelism === TensorParallelism.Column)) {
      using gathered = pOther.allGather(pOther.workspace);
      return this.add(gathered, n);
    }

    using gatheredThis = this.allGather(this.workspace);
    using gatheredOther = pOther.allGather(pOther.workspace);
    return gatheredThis.add(gatheredOther, n);
  }

  mul(other: Tensor, n?: number): Tensor {
    const pOther = other as ParallelTensor;

    if (this.parallelism === pOther.parallelism) {
      const outShards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        outShards.push(this.shards[i].mul(pOther.shards[i], n));
      }
      return this.parallelOps.wrapShards(this.workspace, outShards, this.shape, this.type, this.parallelism);
    }

    if (this.parallelism === TensorParallelism.PartialSum && pOther.parallelism === TensorParallelism.Replicated) {
      const outShards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        outShards.push(this.shards[i].mul(pOther.shards[i], n));
      }
      return this.parallelOps.wrapShards(this.workspace, outShards, this.shape, this.type, TensorParallelism.PartialSum);
    }
    if (pOther.parallelism === TensorParallelism.PartialSum && this.parallelism === TensorParallelism.Replicated) {
      return pOther.mul(this, n);
    }

    if (this.parallelism === TensorParallelism.PartialSum) {
      using gathered = pOther.allGather(pOther.workspace);
      return this.mul(gathered, n);
    }
    if (pOther.parallelism === TensorParallelism.PartialSum) {
      using gathered = this.allGather(this.workspace);
      return pOther.mul(gathered, n);
    }

    if ((this.parallelism === TensorParallelism.Row || this.parallelism === TensorParallelism.Column) && pOther.parallelism === TensorParallelism.Replicated) {
      using gathered = this.allGather(this.workspace);
      return gathered.mul(pOther, n);
    }
    if (this.parallelism === TensorParallelism.Replicated && (pOther.parallelism === TensorParallelism.Row || pOther.parallelism === TensorParallelism.Column)) {
      using gathered = pOther.allGather(pOther.workspace);
      return this.mul(gathered, n);
    }

    using gatheredThis = this.allGather(this.workspace);
    using gatheredOther = pOther.allGather(pOther.workspace);
    return gatheredThis.mul(gatheredOther, n);
  }

  rmsnorm(weight: Tensor, eps: number, dim: number, batch: number): Tensor {
    super.rmsnorm(weight, eps, dim, batch);

    // not worth it because attention as it only delays the gather by a little bit and introduces more gpu-gpu comms
    // if (this.parallelism === TensorParallelism.Row) {
    //   const shardDim = dim / this.worldSize;
    //   const pWeight = weight as ParallelTensor;
    //   const output = this.parallelOps.newTensor(this.workspace, this.fullShape, this.type, false, undefined, TensorParallelism.Row);
    //   if (this.parallelOps.tryP2PRmsnorm(this.shards, pWeight.shards, output.shards, eps, shardDim, dim, batch, false)) {
    //     return output;
    //   }
    //   using gathered = this.allGather(this.workspace);
    //   const result = gathered.rmsnorm(weight, eps, dim, batch);
    //   output[Symbol.dispose]();
    //   return result;
    // }

    if (this.parallelism === TensorParallelism.Row || this.parallelism === TensorParallelism.Column) {
      using gathered = this.allGather(this.workspace);
      const result = gathered.rmsnorm(weight, eps, dim, batch);
      return result;
    }

    if (this.parallelism === TensorParallelism.PartialSum) {
      this.allReduce();
      return this.rmsnorm(weight, eps, dim, batch);
    }

    const pWeight = weight as ParallelTensor;
    this.assertParallel("rmsnorm input", this, TensorParallelism.Replicated);
    this.assertParallel("rmsnorm weight", pWeight, TensorParallelism.Replicated);

    const shards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      shards.push(this.shards[i].rmsnorm(pWeight.shards[i], eps, dim, batch));
    }
    return this.parallelOps.wrapShards(this.workspace, shards, [batch, dim], this.type, TensorParallelism.Replicated);
  }

  fusedAddRmsnorm(input: Tensor, weight: Tensor, eps: number, dim: number, batch: number): { normed: Tensor, residual: Tensor } {
    super.fusedAddRmsnorm(input, weight, eps, dim, batch);
    const pInput = input as ParallelTensor;

    if (this.parallelism === TensorParallelism.PartialSum) {
      this.allReduce();
      return this.fusedAddRmsnorm(input, weight, eps, dim, batch);
    }

    if (this.parallelism === TensorParallelism.Row || this.parallelism === TensorParallelism.Column) {
      using gathered = this.allGather(this.workspace);
      return gathered.fusedAddRmsnorm(input, weight, eps, dim, batch);
    }

    if (pInput.parallelism === TensorParallelism.PartialSum) {
      pInput.allReduce();
      return this.fusedAddRmsnorm(pInput, weight, eps, dim, batch);
    }

    if (pInput.parallelism === TensorParallelism.Row || pInput.parallelism === TensorParallelism.Column) {
      using gathered = pInput.allGather(pInput.workspace);
      return this.fusedAddRmsnorm(gathered, weight, eps, dim, batch);
    }

    const pWeight = weight as ParallelTensor;
    this.assertParallel("fusedAddRmsnorm inputA", this, TensorParallelism.Replicated);
    this.assertParallel("fusedAddRmsnorm inputB", pInput, TensorParallelism.Replicated);
    this.assertParallel("fusedAddRmsnorm weight", pWeight, TensorParallelism.Replicated);

    const normedShards: Tensor[] = [];
    const residualShards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      const result = this.shards[i].fusedAddRmsnorm(pInput.shards[i], pWeight.shards[i], eps, dim, batch);
      normedShards.push(result.normed);
      residualShards.push(result.residual);
    }
    const normed = this.parallelOps.wrapShards(this.workspace, normedShards, [batch, dim], this.type, TensorParallelism.Replicated);
    const residual = this.parallelOps.wrapShards(this.workspace, residualShards, [batch, dim], this.type, TensorParallelism.Replicated);
    return { normed, residual };
  }

  override fusedNormRope(weight: Tensor, cos: Tensor, sin: Tensor, eps: number, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, inStride?: number, interleaved?: boolean): Tensor {
    super.fusedNormRope(weight, cos, sin, eps, ropeDim, headDim, nHeads, seqLen, batch, inStride, interleaved);
    if (this.parallelism === TensorParallelism.PartialSum) {
      this.allReduce();
      return this.fusedNormRope(weight, cos, sin, eps, ropeDim, headDim, nHeads, seqLen, batch, inStride, interleaved);
    }

    if (this.parallelism === TensorParallelism.Column) {
      using gathered = this.allGather(this.workspace);
      return gathered.fusedNormRope(weight, cos, sin, eps, ropeDim, headDim, nHeads, seqLen, batch, inStride, interleaved);
    }

    const pWeight = weight as ParallelTensor;
    const pCos = cos as ParallelTensor;
    const pSin = sin as ParallelTensor;
    const stride = inStride ?? headDim;

    if (this.parallelism === TensorParallelism.Row) {
      const shardNHeads = this.shardDim(nHeads, "fusedNormRope nHeads");
      const shardInStride = this.fullShape[2] === this.fullShape[1]
        ? stride / this.worldSize
        : stride;
      this.assertParallel("fusedNormRope weight", pWeight, TensorParallelism.Replicated);
      this.assertParallel("fusedNormRope cos", pCos, TensorParallelism.Replicated);
      this.assertParallel("fusedNormRope sin", pSin, TensorParallelism.Replicated);
      const shards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        shards.push(this.shards[i].fusedNormRope(pWeight.shards[i], pCos.shards[i], pSin.shards[i], eps, ropeDim, headDim, shardNHeads, seqLen, batch, shardInStride, interleaved));
      }
      return this.parallelOps.wrapShards(this.workspace, shards, [batch, nHeads, seqLen, headDim], this.type, TensorParallelism.Row);
    }

    this.assertParallel("fusedNormRope input", this, TensorParallelism.Replicated);
    this.assertParallel("fusedNormRope weight", pWeight, TensorParallelism.Replicated);
    this.assertParallel("fusedNormRope cos", pCos, TensorParallelism.Replicated);
    this.assertParallel("fusedNormRope sin", pSin, TensorParallelism.Replicated);

    const shards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      shards.push(this.shards[i].fusedNormRope(pWeight.shards[i], pCos.shards[i], pSin.shards[i], eps, ropeDim, headDim, nHeads, seqLen, batch, stride, interleaved));
    }
    return this.parallelOps.wrapShards(this.workspace, shards, [batch, nHeads, seqLen, headDim], this.type, TensorParallelism.Replicated);
  }

  override embedding(ids: Tensor, hidden: number, seqLen: number): Tensor {
    super.embedding(ids, hidden, seqLen);
    const pIds = ids as ParallelTensor;
    this.assertParallel("embedding ids", pIds, TensorParallelism.Replicated, TensorParallelism.PartialSum);

    if (this.parallelism === TensorParallelism.Row) {
      const shardHidden = hidden / this.worldSize;
      const shards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        shards.push(this.shards[i].embedding(pIds.shards[i], shardHidden, seqLen));
      }
      return this.parallelOps.wrapShards(pIds.workspace, shards, [seqLen, hidden], this.type, TensorParallelism.Row);
    }

    this.assertParallel("embedding table", this, TensorParallelism.Replicated);
    const shards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      shards.push(this.shards[i].embedding(pIds.shards[i], hidden, seqLen));
    }
    return this.parallelOps.wrapShards(pIds.workspace, shards, [seqLen, hidden], this.type, TensorParallelism.Replicated);
  }

  override siluAndMul(up: Tensor, intermediate: number, batch: number): Tensor {
    super.siluAndMul(up, intermediate, batch);
    const pUp = up as ParallelTensor;

    if (this.parallelism === pUp.parallelism) {
      const outPar = this.parallelism;
      const shardIntermediate = this.parallelism === TensorParallelism.Row || this.parallelism === TensorParallelism.Column
        ? intermediate / this.worldSize
        : intermediate;
      const shards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        shards.push(this.shards[i].siluAndMul(pUp.shards[i], shardIntermediate, batch));
      }
      return this.parallelOps.wrapShards(this.workspace, shards, [batch, intermediate], this.type, outPar);
    }

    if (this.parallelism === TensorParallelism.PartialSum) {
      this.allReduce();
      return this.siluAndMul(up, intermediate, batch);
    }
    if (pUp.parallelism === TensorParallelism.PartialSum) {
      pUp.allReduce();
      return this.siluAndMul(pUp, intermediate, batch);
    }

    using gatheredGate = this.allGather(this.workspace);
    using gatheredUp = pUp.allGather(pUp.workspace);
    return gatheredGate.siluAndMul(gatheredUp, intermediate, batch);
  }

  arange(start: number, step: number, count: number): void {
    super.arange(start, step, count);
    this.assertParallel("arange", this, TensorParallelism.Replicated, TensorParallelism.PartialSum);
    for (let i = 0; i < this.worldSize; i++) {
      this.shards[i].arange(start, step, count);
    }
  }

  argmax(): Tensor {
    super.argmax();
    const { indices, values } = this.max();
    values[Symbol.dispose]();
    return indices;
  }

  max(offset: number = 0): { values: Tensor, indices: Tensor } {
    super.max(offset);
    if (this.parallelism === TensorParallelism.PartialSum) {
      this.allReduce();
      return this.max(offset);
    }

    if (this.parallelism === TensorParallelism.Row) {
      const batch = this.fullShape[0];
      const dim = this.fullShape[1];
      const ws = this.worldSize;
      const shardDim = dim / ws;

      const localValuesShards: Tensor[] = [];
      const localIndicesShards: Tensor[] = [];
      for (let i = 0; i < ws; i++) {
        const { values, indices } = this.shards[i].max(i * shardDim + offset);
        localValuesShards.push(values);
        localIndicesShards.push(indices);
      }

      using allValuesPar = this.parallelOps.wrapShards(this.workspace, localValuesShards, [batch, ws], this.type, TensorParallelism.Row);
      using allIndicesPar = this.parallelOps.wrapShards(this.workspace, localIndicesShards, [batch, ws], "I32", TensorParallelism.Row);

      using allValues = allValuesPar.allGather(this.workspace);
      using allIndices = allIndicesPar.allGather(this.workspace);

      const { values: rankValues, indices: rankIndices } = allValues.max(0);

      using _rankIndices = rankIndices;
      using gatheredIndices = allIndices.gather(rankIndices, 1, ws, batch);

      const finalIndices = this.parallelOps.newTensor(this.workspace, [batch], "I32", false, undefined, TensorParallelism.Replicated);
      const pGatheredIndices = gatheredIndices as ParallelTensor;
      const idxBytes = batch * 4;
      for (let i = 0; i < ws; i++) {
        finalIndices.shards[i].memcpy(pGatheredIndices.shards[i], idxBytes);
      }

      return { values: rankValues, indices: finalIndices };
    }

    if (this.parallelism === TensorParallelism.Column) {
      const batch = this.fullShape[0];

      const localValuesShards: Tensor[] = [];
      const localIndicesShards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        const { values, indices } = this.shards[i].max(offset);
        localValuesShards.push(values);
        localIndicesShards.push(indices);
      }

      const values = this.parallelOps.wrapShards(this.workspace, localValuesShards, [batch], this.type, TensorParallelism.Column);
      const indices = this.parallelOps.wrapShards(this.workspace, localIndicesShards, [batch], "I32", TensorParallelism.Column);

      return { values, indices };
    }

    this.assertParallel("max input", this, TensorParallelism.Replicated);

    const valuesShards: Tensor[] = [];
    const indicesShards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      const { values, indices } = this.shards[i].max(offset);
      valuesShards.push(values);
      indicesShards.push(indices);
    }
    const values = this.parallelOps.wrapShards(this.workspace, valuesShards, [this.shape[0]], this.type, TensorParallelism.Replicated);
    const indices = this.parallelOps.wrapShards(this.workspace, indicesShards, [this.shape[0]], "I32", TensorParallelism.Replicated);
    return { values, indices };
  }

  indexSelect(indices: Tensor, dim: number, batch: number): Tensor {
    super.indexSelect(indices, dim, batch);
    const pIndices = indices as ParallelTensor;
    this.assertParallel("indexSelect src", this, TensorParallelism.Replicated);
    this.assertParallel("indexSelect indices", pIndices, TensorParallelism.Replicated, TensorParallelism.PartialSum);

    const shards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      shards.push(this.shards[i].indexSelect(pIndices.shards[i], dim, batch));
    }
    return this.parallelOps.wrapShards(this.workspace, shards, [batch, dim], this.type, TensorParallelism.Replicated);
  }

  gather(indices: Tensor, k: number, inDim: number, batch: number): Tensor {
    super.gather(indices, k, inDim, batch);
    const pIndices = indices as ParallelTensor;
    this.assertParallel("gather src", this, TensorParallelism.Replicated);
    this.assertParallel("gather indices", pIndices, TensorParallelism.Replicated, TensorParallelism.PartialSum);

    const shards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      shards.push(this.shards[i].gather(pIndices.shards[i], k, inDim, batch));
    }
    return this.parallelOps.wrapShards(this.workspace, shards, [batch, k], this.type, TensorParallelism.Replicated);
  }

  gdnRecurrentStep(state: Tensor, qkv: Tensor, aRaw: Tensor, bRaw: Tensor, aLog: Tensor, dtBias: Tensor, numHeads: number, dK: number, dV: number, batchSize: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void {
    super.gdnRecurrentStep(state, qkv, aRaw, bRaw, aLog, dtBias, numHeads, dK, dV, batchSize, stateStride, qkvChStride, qkvSeqStride);
    const pState = this.cast(state);
    const pQkv = this.cast(qkv);
    const pARaw = this.cast(aRaw);
    const pBRaw = this.cast(bRaw);
    const pALog = this.cast(aLog);
    const pDtBias = this.cast(dtBias);
    const isRowPar = pQkv.parallelism === TensorParallelism.Row || pQkv.parallelism === TensorParallelism.Column;
    const shardHeads = isRowPar ? this.shardDim(numHeads, "gdnRecurrentStep numHeads") : numHeads;
    const shardStateStride = isRowPar ? stateStride / this.worldSize : stateStride;
    const shardSeqStride = isRowPar ? qkvSeqStride / this.worldSize : qkvSeqStride;
    for (let i = 0; i < this.worldSize; i++) {
      this.shards[i].gdnRecurrentStep(pState.shards[i], pQkv.shards[i], pARaw.shards[i], pBRaw.shards[i], pALog.shards[i], pDtBias.shards[i], shardHeads, dK, dV, batchSize, shardStateStride, qkvChStride, shardSeqStride);
    }
  }

  gdnPrefill(state: Tensor, qkv: Tensor, aRaw: Tensor, bRaw: Tensor, aLog: Tensor, dtBias: Tensor, cuSeqlens: Tensor, totalSeqLen: number, numHeads: number, dK: number, dV: number, batchSize: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void {
    super.gdnPrefill(state, qkv, aRaw, bRaw, aLog, dtBias, cuSeqlens, totalSeqLen, numHeads, dK, dV, batchSize, stateStride, qkvChStride, qkvSeqStride);
    const pState = this.cast(state);
    const pQkv = this.cast(qkv);
    const pARaw = this.cast(aRaw);
    const pBRaw = this.cast(bRaw);
    const pALog = this.cast(aLog);
    const pDtBias = this.cast(dtBias);
    const pCuSeqlens = this.cast(cuSeqlens);
    const isRowPar = pQkv.parallelism === TensorParallelism.Row || pQkv.parallelism === TensorParallelism.Column;
    const shardHeads = isRowPar ? this.shardDim(numHeads, "gdnPrefill numHeads") : numHeads;
    const shardStateStride = isRowPar ? stateStride / this.worldSize : stateStride;
    const shardSeqStride = isRowPar ? qkvSeqStride / this.worldSize : qkvSeqStride;
    for (let i = 0; i < this.worldSize; i++) {
      this.shards[i].gdnPrefill(pState.shards[i], pQkv.shards[i], pARaw.shards[i], pBRaw.shards[i], pALog.shards[i], pDtBias.shards[i], pCuSeqlens.shards[i], totalSeqLen, shardHeads, dK, dV, batchSize, shardStateStride, qkvChStride, shardSeqStride);
    }
  }

  causalConv1d(convState: Tensor, input: Tensor, weight: Tensor, cuSeqlens: Tensor, convDim: number, totalSeqLen: number, kernelSize: number, batchSize: number, convStateStride: number, chStride: number, seqStride: number): void {
    super.causalConv1d(convState, input, weight, cuSeqlens, convDim, totalSeqLen, kernelSize, batchSize, convStateStride, chStride, seqStride);
    const pConvState = this.cast(convState);
    const pInput = this.cast(input);
    const pWeight = this.cast(weight);
    const pCuSeqlens = this.cast(cuSeqlens);
    const isRowPar = pInput.parallelism === TensorParallelism.Row || pInput.parallelism === TensorParallelism.Column;
    const shardConvDim = isRowPar ? convDim / this.worldSize : convDim;
    const shardConvStateStride = isRowPar ? convStateStride / this.worldSize : convStateStride;
    const shardSeqStride = isRowPar ? seqStride / this.worldSize : seqStride;
    for (let i = 0; i < this.worldSize; i++) {
      this.shards[i].causalConv1d(pConvState.shards[i], pInput.shards[i], pWeight.shards[i], pCuSeqlens.shards[i], shardConvDim, totalSeqLen, kernelSize, batchSize, shardConvStateStride, chStride, shardSeqStride);
    }
  }

  causalConv1dUpdate(convState: Tensor, input: Tensor, weight: Tensor, convDim: number, kernelSize: number, batchSize: number, convStateStride: number): Tensor {
    super.causalConv1dUpdate(convState, input, weight, convDim, kernelSize, batchSize, convStateStride);
    const pConvState = this.cast(convState);
    const pInput = input as ParallelTensor;
    const pWeight = this.cast(weight);
    const isRowPar = pInput.parallelism === TensorParallelism.Row || pInput.parallelism === TensorParallelism.Column;
    const parallelism = isRowPar ? TensorParallelism.Row : TensorParallelism.Replicated;
    const shardConvDim = isRowPar ? convDim / this.worldSize : convDim;
    const shardConvStateStride = isRowPar ? convStateStride / this.worldSize : convStateStride;
    const shardOuts: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      shardOuts.push(this.shards[i].causalConv1dUpdate(pConvState.shards[i], pInput.shards[i], pWeight.shards[i], shardConvDim, kernelSize, batchSize, shardConvStateStride));
    }
    return this.parallelOps.wrapShards(this.workspace, shardOuts, [batchSize, convDim], this.type, parallelism);
  }

  rmsnormGated(input: Tensor, gate: Tensor, weight: Tensor, eps: number, dim: number, batch: number): void {
    super.rmsnormGated(input, gate, weight, eps, dim, batch);
    const pInput = input as ParallelTensor;
    const pGate = gate as ParallelTensor;
    const pWeight = weight as ParallelTensor;

    if (this.parallelism === TensorParallelism.Row && pGate.parallelism === TensorParallelism.Row &&
      pInput.parallelism === TensorParallelism.Row && pWeight.parallelism === TensorParallelism.Replicated) {
      const shardBatch = batch / this.worldSize;
      for (let i = 0; i < this.worldSize; i++) {
        this.shards[i].rmsnormGated(pInput.shards[i], pGate.shards[i], pWeight.shards[i], eps, dim, shardBatch);
      }
      return;
    }

    if (pInput.parallelism === TensorParallelism.Column && pGate.parallelism === TensorParallelism.Column &&
      pWeight.parallelism === TensorParallelism.Replicated &&
      (this.parallelism === TensorParallelism.Column || this.parallelism === TensorParallelism.Row)) {
      const shardBatch = batch / this.worldSize;
      for (let i = 0; i < this.worldSize; i++) {
        this.shards[i].rmsnormGated(pInput.shards[i], pGate.shards[i], pWeight.shards[i], eps, dim, shardBatch);
      }
      return;
    }

    if (pInput.parallelism === TensorParallelism.Row) {
      using gathered = pInput.allGather(pInput.workspace);
      this.rmsnormGated(gathered, gate, weight, eps, dim, batch);
      return;
    }

    if (pInput.parallelism === TensorParallelism.PartialSum) {
      pInput.allReduce();
      this.rmsnormGated(pInput, gate, weight, eps, dim, batch);
      return;
    }

    this.assertParallel("rmsnormGated input", pInput, TensorParallelism.Replicated);
    this.assertParallel("rmsnormGated gate", pGate, TensorParallelism.Replicated);
    this.assertParallel("rmsnormGated weight", pWeight, TensorParallelism.Replicated);
    this.assertParallel("rmsnormGated output", this, TensorParallelism.Replicated);

    for (let i = 0; i < this.worldSize; i++) {
      this.shards[i].rmsnormGated(pInput.shards[i], pGate.shards[i], pWeight.shards[i], eps, dim, batch);
    }
  }

  gateSigmoidMul(gate: Tensor, batchSeq: number, numHeads: number, headDim: number): void {
    super.gateSigmoidMul(gate, batchSeq, numHeads, headDim);
    const pGate = gate as ParallelTensor;

    if (this.parallelism === pGate.parallelism) {
      const shardNumHeads = this.parallelism === TensorParallelism.Row || this.parallelism === TensorParallelism.Column
        ? this.shardDim(numHeads, "gateSigmoidMul numHeads")
        : numHeads;
      for (let i = 0; i < this.worldSize; i++) {
        this.shards[i].gateSigmoidMul(pGate.shards[i], batchSeq, shardNumHeads, headDim);
      }
      return;
    }

    if ((this.parallelism === TensorParallelism.Row || this.parallelism === TensorParallelism.Column) && pGate.parallelism === TensorParallelism.Replicated) {
      using gathered = this.allGather(this.workspace);
      gathered.gateSigmoidMul(gate, batchSeq, numHeads, headDim);
      for (let i = 0; i < this.worldSize; i++) {
        this.shards[i].memcpy((gathered as ParallelTensor).shards[i]);
      }
      return;
    }

    using gatheredThis = this.allGather(this.workspace);
    using gatheredGate = pGate.allGather(pGate.workspace);
    gatheredThis.gateSigmoidMul(gatheredGate, batchSeq, numHeads, headDim);
    for (let i = 0; i < this.worldSize; i++) {
      this.shards[i].memcpy((gatheredThis as ParallelTensor).shards[i]);
    }
  }

  fill(value: number, n: number): void {
    const shardN = this.parallelism === TensorParallelism.Row || this.parallelism === TensorParallelism.Column
      ? n / this.worldSize
      : n;
    for (let i = 0; i < this.worldSize; i++) {
      this.shards[i].fill(value, shardN);
    }
  }

  async mmapLoad(mmapPtr: number, offset: number, nbytes: number, strided?: StridedMmap): Promise<void> {
    if (strided) {
      if (this.parallelism === TensorParallelism.Replicated || this.parallelism === TensorParallelism.PartialSum) {
        await Promise.all(this.shards.map(s => s.mmapLoad(mmapPtr, offset, nbytes, strided)));
        return;
      } else if (this.parallelism === TensorParallelism.Column) {
        const shardHeight = strided.height / this.shards.length;
        const shardDstOffset = Math.trunc(strided.dstOffset * shardHeight / strided.height);
        await Promise.all(this.shards.map((s, i) => s.mmapLoad(mmapPtr, offset, nbytes, {
          srcOffset: strided.srcOffset + i * shardHeight * strided.srcPitch,
          dstOffset: shardDstOffset,
          srcPitch: strided.srcPitch,
          dstPitch: strided.dstPitch,
          width: strided.width,
          height: shardHeight,
        })));
        return;
      } else if (this.parallelism === TensorParallelism.Row) {
        const shardWidth = strided.width / this.shards.length;
        const shardDstPitch = strided.dstPitch / this.shards.length;
        await Promise.all(this.shards.map((s, i) => s.mmapLoad(mmapPtr, offset, nbytes, {
          srcOffset: strided.srcOffset + i * shardWidth,
          dstOffset: strided.dstOffset,
          srcPitch: strided.srcPitch,
          dstPitch: shardDstPitch,
          width: shardWidth,
          height: strided.height,
        })));
        return;
      } else {
        throw new Error(`mmapLoad strided: unsupported parallelism ${this.parallelism}`);
      }
    }

    if (this.parallelism === TensorParallelism.Replicated || this.parallelism === TensorParallelism.PartialSum) {
      await Promise.all(this.shards.map(s => s.mmapLoad(mmapPtr, offset, nbytes)));
      return;
    }

    if (this.parallelism === TensorParallelism.Column) {
      const shardElems = this.shards[0].shape.reduce((a, b) => a * b, 1);
      const shardBytes = shardElems * ParallelTensor.elemBytes(this.type);
      await Promise.all(this.shards.map((s, i) => s.mmapLoad(mmapPtr, offset + i * shardBytes, shardBytes)));
      return;
    }

    if (this.parallelism === TensorParallelism.Row) {
      const outer = this.fullShape[0];
      const inner = this.fullShape.slice(2).reduce((a, b) => a * b, 1);
      const fullDim1 = this.fullShape[1];
      const shardDim1 = fullDim1 / this.worldSize;
      const eb = ParallelTensor.elemBytes(this.type);
      const srcPitch = fullDim1 * inner * eb;
      const dstPitch = shardDim1 * inner * eb;
      await Promise.all(this.shards.map((s, i) => s.mmapLoad(mmapPtr, offset, nbytes, {
        srcOffset: i * dstPitch,
        dstOffset: 0,
        srcPitch,
        dstPitch,
        width: dstPitch,
        height: outer,
      })));
      return;
    }

    throw new Error(`mmapLoad: unsupported parallelism ${this.parallelism}`);
  }

  mmapLoadAsync(mmapPtr: number, offset: number, nbytes: number): Promise<void> {
    throw new Error("ParallelTensor.mmapLoadAsync: use mmapLoad instead");
  }

  memcpy2dHostToDeviceAsync(dstOffset: number, dpitch: number, src: number, spitch: number, width: number, height: number): Promise<void> {
    throw new Error("ParallelTensor.memcpy2dHostToDeviceAsync: use mmapLoad instead");
  }

  withPinnedBuffer(fn: (buf: Buffer) => void): void {
    for (let i = 0; i < this.shards.length; i++) {
      this.shards[i].withPinnedBuffer(fn);
    }
  }

  readPinnedBuffer(): Buffer {
    return this.shards[0].readPinnedBuffer();
  }

  memcpy(src: Tensor, size?: number, kind?: MemcpyKind): void {
    if (!(src instanceof ParallelTensor)) {
      throw new Error("ParallelTensor.memcpy requires ParallelTensor source");
    }
    const bytes = size ?? Math.min(this.allocSize, src.allocSize);
    const copyKind = kind ?? MemcpyKind.DeviceToDevice;
    for (let i = 0; i < this.shards.length; i++) {
      this.shards[i].memcpy(src.shards[i], bytes, copyKind);
    }
  }

  memcpy2d(dstOffset: number, dpitch: number, src: number, spitch: number, width: number, height: number, kind: MemcpyKind): void {
    throw new Error("ParallelTensor.memcpy2d: use shard tensors directly");
  }

  sigmoid(): Tensor {
    if (this.parallelism === TensorParallelism.PartialSum) {
      this.allReduce();
      return this.sigmoid();
    }
    const outShards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      outShards.push(this.shards[i].sigmoid());
    }
    return this.parallelOps.wrapShards(this.workspace, outShards, this.fullShape, this.type, this.parallelism);
  }

  topk(k: number, dim: number): { values: Tensor, indices: Tensor } {
    if (this.parallelism === TensorParallelism.PartialSum) {
      this.allReduce();
      return this.topk(k, dim);
    }
    if (this.parallelism === TensorParallelism.Row || this.parallelism === TensorParallelism.Column) {
      throw new Error(`topk: unsupported parallelism ${this.parallelism} (dimension is sharded, results would be incomplete)`);
    }
    const valuesShards: Tensor[] = [];
    const indicesShards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      const result = this.shards[i].topk(k, dim);
      valuesShards.push(result.values);
      indicesShards.push(result.indices);
    }
    const values = this.parallelOps.wrapShards(this.workspace, valuesShards, [...this.fullShape.slice(0, -1), k], this.type, this.parallelism);
    const indices = this.parallelOps.wrapShards(this.workspace, indicesShards, [...this.fullShape.slice(0, -1), k], "I32", this.parallelism);
    return { values, indices };
  }

  reduceSum(dim: number, batch: number): Tensor {
    if (this.parallelism === TensorParallelism.PartialSum) {
      throw new Error("reduceSum: unsupported input parallelism PartialSum (allReduce first)");
    }
    const outShards: Tensor[] = [];
    const isSharded = this.parallelism === TensorParallelism.Row || this.parallelism === TensorParallelism.Column;
    const shardDim = isSharded ? this.parallelOps.shardDim(dim, "reduceSum dim") : dim;
    const shardBatch = isSharded ? batch : batch;
    for (let i = 0; i < this.worldSize; i++) {
      outShards.push(this.shards[i].reduceSum(shardDim, shardBatch));
    }
    const outPar = isSharded ? TensorParallelism.PartialSum : this.parallelism;
    return this.parallelOps.wrapShards(this.workspace, outShards, [batch], this.type, outPar);
  }

  rowNormalize(scale: number, dim: number, batch: number, normalize: boolean = true): Tensor {
    if (this.parallelism === TensorParallelism.PartialSum) {
      throw new Error("rowNormalize: unsupported input parallelism PartialSum (allReduce first)");
    }
    const outShards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      outShards.push(this.shards[i].rowNormalize(scale, dim, batch, normalize));
    }
    return this.parallelOps.wrapShards(this.workspace, outShards, this.fullShape, this.type, this.parallelism);
  }

  scatterScalar(indices: Tensor, value: number, k: number, outDim: number, batch: number): void {
    const pIndices = indices as ParallelTensor;
    if (this.parallelism !== pIndices.parallelism) {
      throw new Error(`scatterScalar: input parallelism ${this.parallelism} must match indices parallelism ${pIndices.parallelism}`);
    }
    for (let i = 0; i < this.worldSize; i++) {
      this.shards[i].scatterScalar(pIndices.shards[i], value, k, outDim, batch);
    }
  }

  groupMaskMul(groupMask: Tensor, numExperts: number, expertsPerGroup: number, nGroup: number, batch: number): void {
    const pGroupMask = groupMask as ParallelTensor;
    if (this.parallelism !== pGroupMask.parallelism) {
      throw new Error(`groupMaskMul: input parallelism ${this.parallelism} must match groupMask parallelism ${pGroupMask.parallelism}`);
    }
    for (let i = 0; i < this.worldSize; i++) {
      this.shards[i].groupMaskMul(pGroupMask.shards[i], numExperts, expertsPerGroup, nGroup, batch);
    }
  }

  ropeTranspose(cos: Tensor, sin: Tensor, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, inStride?: number, interleaved?: boolean): Tensor {
    super.ropeTranspose(cos, sin, ropeDim, headDim, nHeads, seqLen, batch, inStride, interleaved);
    const pCos = cos ? cos as ParallelTensor : undefined;
    const pSin = sin ? sin as ParallelTensor : undefined;
    if (pCos && pCos.parallelism !== TensorParallelism.Replicated) {
      throw new Error(`ropeTranspose: cos must be Replicated, got ${pCos.parallelism}`);
    }
    if (pSin && pSin.parallelism !== TensorParallelism.Replicated) {
      throw new Error(`ropeTranspose: sin must be Replicated, got ${pSin.parallelism}`);
    }
    const shardNHeads = (this.parallelism === TensorParallelism.Row || this.parallelism === TensorParallelism.Column)
      ? this.parallelOps.shardDim(nHeads, "ropeTranspose nHeads")
      : nHeads;
    if (this.parallelism === TensorParallelism.PartialSum) {
      throw new Error(`ropeTranspose: unsupported input parallelism ${this.parallelism}`);
    }
    const outShards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      const shardCos = pCos ? pCos.shards[i] : undefined!;
      const shardSin = pSin ? pSin.shards[i] : undefined!;
      outShards.push(this.shards[i].ropeTranspose(shardCos, shardSin, ropeDim, headDim, shardNHeads, seqLen, batch, inStride ?? headDim, interleaved));
    }
    return this.parallelOps.wrapShards(this.workspace, outShards, [batch * seqLen, nHeads, headDim], this.type, this.parallelism);
  }

  applyRotaryPosEmb(cos: Tensor, sin: Tensor, ropeDim: number, nHeads: number, seqLen: number, batch: number, unsqueezeDim: number, interleaved?: boolean): Tensor {
    super.applyRotaryPosEmb(cos, sin, ropeDim, nHeads, seqLen, batch, unsqueezeDim, interleaved);
    const pCos = cos as ParallelTensor;
    const pSin = sin as ParallelTensor;
    if (pCos.parallelism !== TensorParallelism.Replicated) {
      throw new Error(`applyRotaryPosEmb: cos must be Replicated, got ${pCos.parallelism}`);
    }
    if (pSin.parallelism !== TensorParallelism.Replicated) {
      throw new Error(`applyRotaryPosEmb: sin must be Replicated, got ${pSin.parallelism}`);
    }
    if (this.parallelism === TensorParallelism.PartialSum) {
      throw new Error(`applyRotaryPosEmb: unsupported input parallelism ${this.parallelism}`);
    }
    const shardNHeads = (this.parallelism === TensorParallelism.Row || this.parallelism === TensorParallelism.Column)
      ? this.parallelOps.shardDim(nHeads, "applyRotaryPosEmb nHeads")
      : nHeads;
    const outShards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      outShards.push(this.shards[i].applyRotaryPosEmb(pCos.shards[i], pSin.shards[i], ropeDim, shardNHeads, seqLen, batch, unsqueezeDim, interleaved));
    }
    return this.parallelOps.wrapShards(this.workspace, outShards, this.fullShape, this.type, this.parallelism);
  }

  mlaVExpand(vProj: Tensor, kvLoraRank: number, vHeadDim: number, nHeads: number, seqLen: number, batch: number, lse?: Tensor): Tensor {
    super.mlaVExpand(vProj, kvLoraRank, vHeadDim, nHeads, seqLen, batch);
    const pVProj = vProj as ParallelTensor;
    if (this.parallelism === TensorParallelism.PartialSum || pVProj.parallelism === TensorParallelism.PartialSum) {
      throw new Error(`mlaVExpand: unsupported parallelism this=${this.parallelism}, vProj=${pVProj.parallelism}`);
    }
    const shardNHeads = (pVProj.parallelism === TensorParallelism.Row || pVProj.parallelism === TensorParallelism.Column)
      ? this.parallelOps.shardDim(nHeads, "mlaVExpand nHeads")
      : nHeads;
    const isPartialSoftmax = this.parallelism === TensorParallelism.PartialSoftmax;
    const attnNHeads = isPartialSoftmax ? nHeads : shardNHeads;
    const isVProjSharded = pVProj.parallelism === TensorParallelism.Row || pVProj.parallelism === TensorParallelism.Column;
    const BS = batch * seqLen;
    const outShards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      const shardHeadOffset = (isPartialSoftmax && isVProjSharded) ? i * shardNHeads : 0;
      outShards.push(this.shards[i].mlaVExpand(pVProj.shards[i], kvLoraRank, vHeadDim, shardNHeads, seqLen, batch, undefined, shardHeadOffset, attnNHeads));
    }
    const isCp = this.parallelism === TensorParallelism.PartialSoftmax;
    const vExpandedPar = isCp ? TensorParallelism.Column : this.parallelism;
    const vExpandedFullShape = isCp ? [BS * this.parallelOps.worldSize, nHeads * vHeadDim] : [BS, nHeads * vHeadDim];
    const vExpanded = this.parallelOps.wrapShards(this.workspace, outShards, vExpandedFullShape, this.type, vExpandedPar);
    if (isCp) {
      const pLse = lse as ParallelTensor;
      const cpShardNHeads = isVProjSharded ? shardNHeads : this.parallelOps.shardDim(nHeads, "mlaVExpand cpShardNHeads");
      const cpInputNHeads = isVProjSharded ? shardNHeads : nHeads;
      return this.parallelOps.contextParallelMerge(
        vExpanded, pLse,
        BS, nHeads, vHeadDim,
        null, this.workspace,
        cpShardNHeads, cpInputNHeads,
      );
    }
    return vExpanded;
  }

  mulMatId(weights: Tensor[], expertIds: Tensor, batchIds: Tensor, count: number, N: number, K: number, name: string): Tensor {
    const pWeights = weights.map(w => w as ParallelTensor);
    const pExpertIds = expertIds as ParallelTensor;
    const pBatchIds = batchIds as ParallelTensor;
    const inputPar = this.parallelism;
    let outN = N, outK = K, outPar: TensorParallelism;
    if (inputPar === TensorParallelism.Replicated) {
      outN = this.parallelOps.shardDim(N, "mulMatId N");
      outPar = TensorParallelism.Row;
    } else if (inputPar === TensorParallelism.Row) {
      outK = this.parallelOps.shardDim(K, "mulMatId K");
      outPar = TensorParallelism.PartialSum;
    } else {
      throw new Error(`mulMatId: unsupported input parallelism ${inputPar}`);
    }
    const outShards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      const shardWeights = pWeights.map(w => w.shards[i]);
      outShards.push(this.shards[i].mulMatId(shardWeights, pExpertIds.shards[i], pBatchIds.shards[i], count, outN, outK, name));
    }
    return this.parallelOps.wrapShards(this.workspace, outShards, [count, N], this.type, outPar);
  }

  scatterAddRows(scales: Tensor, batchIds: Tensor, dim: number, count: number, numRows: number): Tensor {
    const pScales = scales as ParallelTensor;
    const pBatchIds = batchIds as ParallelTensor;
    const outShards: Tensor[] = [];
    if (this.parallelism === TensorParallelism.PartialSum) {
      for (let i = 0; i < this.worldSize; i++) {
        outShards.push(this.shards[i].scatterAddRows(pScales.shards[i], pBatchIds.shards[i], dim, count, numRows));
      }
      return this.parallelOps.wrapShards(this.workspace, outShards, [numRows, dim], this.type, TensorParallelism.PartialSum);
    } else {
      for (let i = 0; i < this.worldSize; i++) {
        outShards.push(this.shards[i].scatterAddRows(pScales.shards[i], pBatchIds.shards[i], dim, count, numRows));
      }
      return this.parallelOps.wrapShards(this.workspace, outShards, [numRows, dim], this.type, this.parallelism);
    }
  }

  rotaryEmbedding(positionIds: Tensor, dimHalf: number, batch: number, seqLen: number): { cos: Tensor, sin: Tensor } {
    super.rotaryEmbedding(positionIds, dimHalf, batch, seqLen);
    const pPositionIds = positionIds as ParallelTensor;
    this.assertParallel("rotaryEmbedding invFreq", this, TensorParallelism.Replicated);
    this.assertParallel("rotaryEmbedding positionIds", pPositionIds, TensorParallelism.Replicated);

    const cosShards: Tensor[] = [];
    const sinShards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      const result = this.shards[i].rotaryEmbedding(pPositionIds.shards[i], dimHalf, batch, seqLen);
      cosShards.push(result.cos);
      sinShards.push(result.sin);
    }
    const hd = dimHalf * 2;
    const cos = this.parallelOps.wrapShards(pPositionIds.workspace, cosShards, [batch, seqLen, hd], this.type, TensorParallelism.Replicated);
    const sin = this.parallelOps.wrapShards(pPositionIds.workspace, sinShards, [batch, seqLen, hd], this.type, TensorParallelism.Replicated);
    return { cos, sin };
  }

  sampleBatch(outTokens: Tensor, topkVals: Tensor, topkIdxs: Tensor, workspace: Tensor, logits: Tensor, penaltyTokens: Tensor, penaltyCount: Tensor, maxWindow: number, vocabSize: number, batchSize: number, temperatures: Tensor, repPenalties: Tensor, presPenalties: Tensor, topKs: Tensor, topPs: Tensor, stepCounter: Tensor, maxEffectiveK: number): void {
    const pLogits = logits as ParallelTensor;
    if (pLogits.parallelism === TensorParallelism.Row || pLogits.parallelism === TensorParallelism.Column) {
      using gathered = pLogits.allGather(pLogits.workspace);
      this.sampleBatch(outTokens, topkVals, topkIdxs, workspace, gathered, penaltyTokens, penaltyCount, maxWindow, vocabSize, batchSize, temperatures, repPenalties, presPenalties, topKs, topPs, stepCounter, maxEffectiveK);
      return;
    }
    if (pLogits.parallelism === TensorParallelism.PartialSum) {
      pLogits.allReduce();
      this.sampleBatch(outTokens, topkVals, topkIdxs, workspace, logits, penaltyTokens, penaltyCount, maxWindow, vocabSize, batchSize, temperatures, repPenalties, presPenalties, topKs, topPs, stepCounter, maxEffectiveK);
      return;
    }
    const pOut = this.cast(outTokens);
    const pTopkVals = this.cast(topkVals);
    const pTopkIdxs = this.cast(topkIdxs);
    const pWorkspace = this.cast(workspace);
    const pPenaltyTokens = this.cast(penaltyTokens);
    const pPenaltyCount = this.cast(penaltyCount);
    const pTemps = this.cast(temperatures);
    const pRepPen = this.cast(repPenalties);
    const pPresPen = this.cast(presPenalties);
    const pTopKs = this.cast(topKs);
    const pTopPs = this.cast(topPs);
    const pStepCounter = this.cast(stepCounter);
    this.assertParallel("sampleBatch logits", pLogits, TensorParallelism.Replicated);
    for (let i = 0; i < this.worldSize; i++) {
      pLogits.shards[i].sampleBatch(pOut.shards[i], pTopkVals.shards[i], pTopkIdxs.shards[i], pWorkspace.shards[i], pLogits.shards[i], pPenaltyTokens.shards[i], pPenaltyCount.shards[i], maxWindow, vocabSize, batchSize, pTemps.shards[i], pRepPen.shards[i], pPresPen.shards[i], pTopKs.shards[i], pTopPs.shards[i], pStepCounter.shards[i], maxEffectiveK);
    }
  }
}

/**
 * Custom one-shot AllReduce group using direct peer-mapped reads, intended
 * for small messages on PCIe-only (no-NVLink) topologies where NCCL ring
 * AllReduce is latency-bound. The group owns one device-side instance per
 * rank, and a kernel call from each rank participates in the same lock-step
 * AllReduce. Concurrent AllReduces from the same group on different streams
 * would race and are not supported.
 */
class P2PAllReduceGroup {
  /** Per-rank GlmP2PInstance native pointers. */
  readonly instances: number[];
  private readonly flagPtrs: number[];
  private dataBufs: (Tensor | null)[];
  private maxSlotBytes: number = 0;
  readonly worldSize: number;
  private readonly devices: readonly GlmOps[];

  constructor(devices: readonly GlmOps[]) {
    this.worldSize = devices.length;
    this.devices = devices;
    this.dataBufs = new Array(devices.length).fill(null);
    const addon = getNativeAddon();

    // 1. Create one instance per rank (metadata only, no data buffer).
    this.instances = devices.map((dev, rank) =>
      addon.p2pCreateInstance(dev.ctx, rank, devices.length));

    // 2. Cache flag pointers (never changes).
    this.flagPtrs = this.instances.map(inst => addon.p2pGetFlagPtr(inst));
  }

  /**
   * Ensure the P2P data buffer is at least `slotBytes` per slot.
   * Allocates or grows the buffer as needed. Each buffer is `2 * slotBytes`
   * for double buffering.
   */
  ensureCapacity(slotBytes: number, shardWorkspaces: WorkspaceBase[]): void {
    if (slotBytes <= this.maxSlotBytes) return;
    const bufBytes = slotBytes * 2;
    const addon = getNativeAddon();

    // Allocate new data buffer tensors on each device.
    const newBufs = this.devices.map((_, i) =>
      shardWorkspaces[i].allocRaw(bufBytes));

    // Dispose old buffers (returns memory to workspace for reuse).
    for (const buf of this.dataBufs) {
      if (buf) buf[Symbol.dispose]();
    }
    this.dataBufs = newBufs;
    this.maxSlotBytes = slotBytes;

    // Update max_bytes on all instances.
    for (let i = 0; i < this.worldSize; ++i) {
      addon.p2pSetMaxBytes(this.instances[i], slotBytes);
    }

    // Update peer pointers on all ranks.
    const dataPtrs = newBufs.map(buf => buf.data);
    for (let i = 0; i < this.worldSize; ++i) {
      addon.p2pSetPeers(this.devices[i].ctx, this.instances[i], dataPtrs, this.flagPtrs);
    }
  }

  free(): void {
    for (const buf of this.dataBufs) {
      if (buf) buf[Symbol.dispose]();
    }
    this.dataBufs = new Array(this.worldSize).fill(null);
    this.maxSlotBytes = 0;
    for (const inst of this.instances) {
      getNativeAddon().p2pDestroyInstance(inst);
    }
  }
}

export class ParallelOps implements DeviceOps {
  readonly devices: readonly GlmOps[];
  readonly worldSize: number;
  readonly comms: number[];
  private readonly shardWorkspaces = new WeakMap<WorkspaceBase, WorkspaceBase[]>();
  /** Lazy-initialized P2P groups per stream. */
  private p2pGroups = new Map<number, P2PAllReduceGroup>();
  private p2pEnabled: boolean;
  /** When true, all GPUs are context-parallel shards. MLA ops auto-inject cpWorldSize/cpRank. */

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
    // Enable P2P peer access early, before model weights are loaded,
    // to avoid VA-space fragmentation that can cause cudaDeviceEnablePeerAccess
    // to fail with cudaErrorMemoryAllocation on large models.
    this.p2pEnabled = process.env.GLM_DISABLE_P2P_ALLREDUCE !== "1" && devices.length > 1;
    if (this.p2pEnabled) {
      this.p2pEnabled = this.enablePeerAccess(devices);
    }
  }

  /** Enable P2P peer access between all device pairs. Returns true on success. */
  private enablePeerAccess(devices: readonly GlmOps[]): boolean {
    const N = devices.length;
    const addon = getNativeAddon();
    for (let i = 0; i < N; ++i) {
      for (let j = 0; j < N; ++j) {
        if (i === j) continue;
        const rc = addon.p2pEnablePeerAccess(devices[i].ctx, devices[j].device);
        if (rc !== 0) {
          console.warn(`P2P peer access dev ${devices[i].device} -> ${devices[j].device} failed; ` +
            `falling back to NCCL for small AllReduce`);
          return false;
        }
      }
    }
    return true;
  }

  /** Get (and lazily create) the P2P group for the given stream. */
  private getP2PGroup(stream: number): P2PAllReduceGroup | null {
    if (!this.p2pEnabled) return null;
    if (!this.p2pGroups.has(stream)) {
      try {
        this.p2pGroups.set(stream, new P2PAllReduceGroup(this.devices));
      } catch (e) {
        console.warn(`P2P AllReduce group creation failed (${e}); falling back to NCCL`);
        this.p2pEnabled = false;
        return null;
      }
    }
    return this.p2pGroups.get(stream) || null;
  }

  /**
   * Try to AllReduce via the custom P2P kernel. Returns true on success
   * (caller must skip the NCCL fallback). Returns false if the message is
   * too large for the P2P group, in which case the caller should NCCL.
   */
  private tryP2PAllReduce(shards: readonly Tensor[], count: number, dtype: number): boolean {
    if (!this.p2pEnabled)
      return false;
    if (dtype !== NCCL_BFLOAT16 && dtype !== NCCL_FLOAT32)
      return false;
    // Single-block kernel limit: 1024 threads * 8 vec = 8192 elements.
    if (count > 8192)
      return false;
    const group = this.getP2PGroup(shards[0].workspace.glm.currentStream);
    if (!group)
      return false;
    const elemBytes = dtype === NCCL_BFLOAT16 ? 2 : 4;
    const slotBytes = count * elemBytes;
    const shardWorkspaces = this.getShardWorkspaces(shards[0].workspace);
    group.ensureCapacity(slotBytes, shardWorkspaces);
    const addon = getNativeAddon();
    for (let i = 0; i < this.worldSize; ++i) {
      addon.p2pAllReduce(this.devices[i].ctx, group.instances[i],
        shards[i].data, shards[i].data, count, dtype);
    }
    return true;
  }

  /**
   * Try to AllGather via the custom P2P kernel. Returns true on success
   * (caller must skip the NCCL fallback). Returns false if the shard is
   * too large for the P2P group, in which case the caller should use NCCL.
   * Dtype-agnostic: copies raw bytes, supports all element types.
   */
  tryP2PAllGather(
    shards: readonly Tensor[],
    outputShards: readonly Tensor[],
    count: number,
    elemBytes: number,
    parallelism: TensorParallelism,
    fullShape: number[],
  ): boolean {
    if (!this.p2pEnabled)
      return false;
    const shardBytes = count * elemBytes;
    const group = this.getP2PGroup(shards[0].workspace.glm.currentStream);
    if (!group)
      return false;
    const shardWorkspaces = this.getShardWorkspaces(shards[0].workspace);
    group.ensureCapacity(shardBytes, shardWorkspaces);
    const addon = getNativeAddon();

    if (parallelism === TensorParallelism.Column) {
      for (let i = 0; i < this.worldSize; ++i) {
        addon.p2pAllGather(
          this.devices[i].ctx, group.instances[i],
          shards[i].data, outputShards[i].data,
          shardBytes,
        );
      }
      return true;
    }

    if (parallelism === TensorParallelism.Row) {
      const outer = fullShape[0];
      const inner = fullShape.slice(2).reduce((a, b) => a * b, 1);
      const shardDim1 = fullShape[1] / this.worldSize;
      const shardDim1Bytes = shardDim1 * inner * elemBytes;
      const fullDim1Bytes = fullShape[1] * inner * elemBytes;
      for (let i = 0; i < this.worldSize; ++i) {
        addon.p2pAllGatherRow(
          this.devices[i].ctx, group.instances[i],
          shards[i].data, outputShards[i].data,
          shardBytes, shardDim1Bytes, fullDim1Bytes, outer,
        );
      }
      return true;
    }

    return false;
  }

  tryP2PRmsnorm(
    inputShards: readonly Tensor[],
    weightShards: readonly Tensor[],
    outputShards: readonly Tensor[],
    eps: number,
    shardDim: number,
    fullDim: number,
    batch: number,
    weightIsSharded: boolean,
  ): boolean {
    if (!this.p2pEnabled)
      return false;
    const group = this.getP2PGroup(inputShards[0].workspace.glm.currentStream);
    if (!group)
      return false;
    const slotBytes = batch * 4;
    const shardWorkspaces = this.getShardWorkspaces(inputShards[0].workspace);
    group.ensureCapacity(slotBytes, shardWorkspaces);
    const addon = getNativeAddon();
    for (let i = 0; i < this.worldSize; ++i) {
      const weightPtr = weightIsSharded
        ? weightShards[i].data
        : weightShards[i].data + i * shardDim * 2;
      addon.p2pRmsnorm(
        this.devices[i].ctx, group.instances[i],
        inputShards[i].data, weightPtr, outputShards[i].data,
        eps, shardDim, fullDim, batch,
      );
    }
    return true;
  }

  /** Public wrapper used by ParallelTensor.allReduce. */
  doAllReduce(shards: readonly Tensor[], count: number, dtype: number): void {
    if (this.tryP2PAllReduce(shards, count, dtype))
      return;
    const addon = getNativeAddon();
    addon.ncclGroupStart();
    for (let i = 0; i < this.worldSize; ++i) {
      addon.ncclAllReduce(
        this.comms[i], this.devices[i].ctx,
        shards[i].data, shards[i].data,
        count, dtype, NCCL_SUM,
      );
    }
    addon.ncclGroupEnd();
  }

  /**
   * Merge partial attention outputs from context-parallel shards using
   * NCCL AllGather + local merge.
   *
   * Each GPU has partial_v_out [batch, nHeads * vHeadDim] (BF16)
   * and partial_lse [batch, nHeads] (F32).
   *
   * When shardNHeads is provided, only processes and outputs heads
   * [i * shardNHeads, (i+1) * shardNHeads) per device i, producing
   * contiguous Row-parallel output [batch, shardNHeads * vHeadDim].
   *
   * When shardNHeads is omitted, outputs all heads (Replicated).
   */
  contextParallelMerge(
    partialVOuts: ParallelTensor,
    partialLses: ParallelTensor,
    batchSize: number,
    numHeads: number,
    vHeadDim: number,
    mergedLse: Tensor | null,
    workspace: WorkspaceBase,
    shardNHeads?: number,
    inputNHeads?: number,
  ): ParallelTensor {
    if (this.p2pEnabled) {
      return this.p2pCpMerge(partialVOuts.shards, partialLses.shards, batchSize, numHeads, vHeadDim, mergedLse, workspace, shardNHeads, inputNHeads);
    }
    const snh = shardNHeads ?? numHeads;
    const inh = inputNHeads ?? snh;
    const isHeads = snh !== numHeads;
    using gatheredVOutShards = partialVOuts.allGather(partialVOuts.workspace);
    using gatheredLseShards = partialLses.allGather(partialLses.workspace);
    const shardWss = this.getShardWorkspaces(workspace);
    const shardVOutShape = isHeads ? [batchSize, snh * vHeadDim] : [batchSize, numHeads * vHeadDim];
    const mergedVOutShards = shardWss.map(ws => ws.alloc(shardVOutShape, "BF16"));
    const shardLseShape = isHeads ? [batchSize, snh] : [batchSize, numHeads];
    const mergedLseShards = mergedLse ? shardWss.map(ws => ws.alloc(shardLseShape, "F32")) : null;

    // AllGather each shard's partial_v_out and partial_lse to all GPUs.
    // After AllGather, each GPU has all shards' data and can run cp_merge locally.
    const vOutElemBytes = 2; // BF16
    const lseElemBytes = 4;  // F32
    const vOutElemsPerShard = batchSize * inh * vHeadDim;
    const lseElemsPerShard = batchSize * numHeads;
    const numShards = this.worldSize;

    // Build pointer arrays and run cp_merge on each device.
    for (let i = 0; i < this.worldSize; i++) {
      const vPtrs: number[] = [];
      const lsePtrs: number[] = [];
      for (let s = 0; s < numShards; s++) {
        vPtrs.push(gatheredVOutShards.shards[i].data + s * vOutElemsPerShard * vOutElemBytes);
        lsePtrs.push(gatheredLseShards.shards[i].data + s * lseElemsPerShard * lseElemBytes);
      }
      const headOffset = isHeads ? i * snh : 0;
      this.devices[i].contextParallelMerge(
        vPtrs, lsePtrs, numShards,
        mergedVOutShards[i], mergedLseShards ? mergedLseShards[i] : null,
        batchSize, numHeads, vHeadDim, snh, headOffset, inh,
      );
    }

    const outParallelism = isHeads ? TensorParallelism.Row : TensorParallelism.Replicated;
    const fullVOutShape = [batchSize, numHeads * vHeadDim];
    return this.wrapShards(workspace, mergedVOutShards, fullVOutShape, "BF16", outParallelism);
  }

  /**
   * Merge partial attention outputs using fused P2P + online softmax merge.
   * This avoids NCCL overhead for small payloads.
   *
   * Each GPU has partial_v_out [batch, nHeads * vHeadDim] (BF16)
   * and partial_lse [batch, nHeads] (F32).
   *
   * When shardNHeads is provided, only processes and outputs heads
   * [i * shardNHeads, (i+1) * shardNHeads) per device i, producing
   * contiguous Row-parallel output [batch, shardNHeads * vHeadDim].
   *
   * When shardNHeads is omitted, outputs all heads (Replicated).
   */
  p2pCpMerge(
    partialVOuts: readonly Tensor[],
    partialLses: readonly Tensor[],
    batchSize: number,
    numHeads: number,
    vHeadDim: number,
    mergedLse: Tensor | null,
    workspace: WorkspaceBase,
    shardNHeads?: number,
    inputNHeads?: number,
  ): ParallelTensor {
    const snh = shardNHeads ?? numHeads;
    const inh = inputNHeads ?? snh;
    const isHeads = snh !== numHeads;
    const numShards = partialVOuts.length;
    if (numShards !== this.worldSize) {
      throw new Error(`p2pCpMerge: numShards=${numShards} != worldSize=${this.worldSize}`);
    }
    if (numShards < 2) {
      throw new Error(`p2pCpMerge: requires at least 2 shards, got ${numShards}`);
    }

    // P2P buffer holds per-shard data: v_out is [B, inh, D] (input_n_heads stride),
    // lse is [B, numHeads] (full heads).
    const vOutBytes = batchSize * inh * vHeadDim * 2; // BF16
    const lseBytes = batchSize * numHeads * 4;             // F32
    const slotBytes = vOutBytes + lseBytes;

    const group = this.getP2PGroup(partialVOuts[0].workspace.glm.currentStream);
    if (!group) {
      throw new Error("p2pCpMerge: P2P not available");
    }
    const shardWorkspaces = this.getShardWorkspaces(partialVOuts[0].workspace);
    group.ensureCapacity(slotBytes, shardWorkspaces);
    const shardVOutShape = isHeads ? [batchSize, snh * vHeadDim] : [batchSize, numHeads * vHeadDim];

    const mergedVOutShards = shardWorkspaces.map(ws => ws.alloc(shardVOutShape, "BF16"));
    let mergedLseShards: Tensor[] | null = null;
    if (mergedLse) {
      const shardLseShape = isHeads ? [batchSize, snh] : [batchSize, numHeads];
      mergedLseShards = shardWorkspaces.map(ws => ws.alloc(shardLseShape, "F32"));
    }

    for (let i = 0; i < this.worldSize; i++) {
      const headOffset = isHeads ? i * snh : 0;
      this.devices[i].p2pCpMerge(
        group.instances[i],
        partialVOuts[i], partialLses[i],
        mergedVOutShards[i], mergedLseShards ? mergedLseShards[i] : null,
        numShards, batchSize, numHeads, vHeadDim, snh, headOffset, inh,
      );
    }

    const outParallelism = isHeads ? TensorParallelism.Row : TensorParallelism.Replicated;
    const fullVOutShape = [batchSize, numHeads * vHeadDim];
    return new ParallelTensor(
      workspace, this,
      outParallelism,
      mergedVOutShards, fullVOutShape,
      "BF16", undefined, false, undefined,
    );
  }

  free(): void {
    for (const group of this.p2pGroups.values()) {
      group.free();
    }
    this.p2pGroups.clear();
    if (this.comms.length > 0) {
      for (const comm of this.comms) {
        getNativeAddon().ncclCommDestroy(comm);
      }
    }
  }

  shardDim(dim: number, name: string): number {
    if (dim % this.worldSize !== 0) {
      throw new Error(`${name}: dimension ${dim} not divisible by worldSize=${this.worldSize}`);
    }
    return dim / this.worldSize;
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

  newTensor(workspace: WorkspaceBase, shape: number[], type: string, pinned: boolean, name?: string, parallelism?: TensorParallelism): ParallelTensor {
    const par = parallelism ?? TensorParallelism.Replicated;
    const ss = this.shardShape(shape, par);
    const shardWss = this.getShardWorkspaces(workspace);
    const shards: Tensor[] = shardWss.map(ws =>
      pinned ? ws.allocPinned(ss, type) : ws.alloc(ss, type),
    );
    return new ParallelTensor(workspace, this, par, shards, shape, type, name, pinned, undefined);
  }

  wrapTensor(workspace: WorkspaceBase, data: number, allocSize: number, shape: number[], type: string, pinned: boolean, view: ParallelTensor | undefined): Tensor {
    if (!view)
      throw new Error("ParallelOps.wrapTensor not supported; tensor recycling happens at shard level");
    return new ParallelTensor(workspace, this, view.parallelism, view.shards, shape, type, undefined, pinned, view);
  }

  wrapShards(workspace: WorkspaceBase, shards: Tensor[], fullShape: number[], type: string, parallelism: TensorParallelism): ParallelTensor {
    const pt = new ParallelTensor(workspace, this, parallelism, shards, fullShape, type, undefined, false, undefined);
    workspace.tracked.add(pt);
    return pt;
  }

  synchronize(): void {
    for (const device of this.devices) {
      device.synchronize();
    }
  }

  synchronizeStream(streamIdx: number): void {
    for (const device of this.devices) {
      device.synchronizeStream(streamIdx);
    }
  }

  availableStreams: number[] = [];
  currentStream = 0;
  setStream(streamIdx: number): void {
    for (const device of this.devices) {
      device.setStream(streamIdx);
    }
  }

  eventRecord(eventIdx: number, streamIdx: number): void {
    for (const device of this.devices) {
      device.eventRecord(eventIdx, streamIdx);
    }
  }

  streamWaitEvent(streamIdx: number, eventIdx: number): void {
    throw new Error("ParallelOps.streamWaitEvent should be used on the device level, not on ParallelOps");
  }

  withStream<T>(fn: () => T) {
    const currentStreams = this.devices.map(device => device.currentStream);
    const streams = this.devices.map(device => device.availableStreams.pop());
    if (streams.includes(undefined)) {
      throw new Error("Not enough available streams on devices");
    }
    for (let i = 0; i < this.devices.length; i++) {
      this.devices[i].eventRecord(currentStreams[i], currentStreams[i]);
      this.devices[i].setStream(streams[i]!);
      this.devices[i].streamWaitEvent(streams[i]!, currentStreams[i]);
    }
    const result = fn();
    for (let i = 0; i < this.devices.length; i++) {
      this.devices[i].eventRecord(streams[i]!, streams[i]!);
      this.devices[i].setStream(currentStreams[i]);
    }
    return {
      [Symbol.dispose]: () => {
        for (let i = 0; i < this.devices.length; i++) {
          this.devices[i].disposeStream(streams[i]!);
        }
      },
      streamWaitEvent: () => {
        for (let i = 0; i < this.devices.length; i++) {
          this.devices[i].streamWaitEvent(this.devices[i].currentStream, streams[i]!);
        }
      },
      synchronize: () => {
        for (let i = 0; i < this.devices.length; i++) {
          this.devices[i].synchronizeStream(this.devices[i].currentStream);
        }
      },
      result,
    };
  }

  private cast(tensor: Tensor): ParallelTensor {
    return tensor as ParallelTensor;
  }

  private assertParallel(name: string, tensor: ParallelTensor, ...allowed: TensorParallelism[]): void {
    if (!allowed.includes(tensor.parallelism)) {
      throw new Error(`${name}: unsupported parallelism ${tensor.parallelism}, expected ${allowed.join(" or ")}`);
    }
  }

  kvCacheWrite(srcK: Tensor, srcV: Tensor, dstK: Tensor, dstV: Tensor, slotMapping: Tensor, batchSize: number, nKv: number, hd: number, pageSize: number, srcKTokenStride: number, srcKHeadStride: number, srcVTokenStride: number, srcVHeadStride: number): void {
    const pSrcK = this.cast(srcK);
    const pSrcV = this.cast(srcV);
    const pDstK = this.cast(dstK);
    const pDstV = this.cast(dstV);
    const pSlotMapping = this.cast(slotMapping);

    this.assertParallel("kvCacheWrite slotMapping", pSlotMapping, TensorParallelism.Replicated);

    const shardNKv = this.shardDim(nKv, "kvCacheWrite nKv");
    const isRowPar = pSrcK.parallelism === TensorParallelism.Row || pSrcK.parallelism === TensorParallelism.Column;
    const shardKTokenStride = isRowPar && srcKTokenStride !== hd ? srcKTokenStride / this.worldSize : srcKTokenStride;
    const shardVTokenStride = isRowPar ? srcVTokenStride / this.worldSize : srcVTokenStride;
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].kvCacheWrite(pSrcK.shards[i], pSrcV.shards[i], pDstK.shards[i], pDstV.shards[i], pSlotMapping.shards[i], batchSize, shardNKv, hd, pageSize, shardKTokenStride, srcKHeadStride, shardVTokenStride, srcVHeadStride);
    }
  }

  decodeStep(positionIds: Tensor, lastPageLen: Tensor, slotMapping: Tensor, indptr: Tensor, indices: Tensor, pageSize: number, batchSize: number): void {
    const pPositionIds = this.cast(positionIds);
    const pLastPageLen = this.cast(lastPageLen);
    const pSlotMapping = this.cast(slotMapping);
    const pIndptr = this.cast(indptr);
    const pIndices = this.cast(indices);
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].decodeStep(pPositionIds.shards[i], pLastPageLen.shards[i], pSlotMapping.shards[i], pIndptr.shards[i], pIndices.shards[i], pageSize, batchSize);
    }
  }

  mlaDecodeStep(positionIds: Tensor, lastPageLen: Tensor, indptr: Tensor, pageSize: number, batchSize: number, contextParallel?: boolean, _cpWorldSize?: number, _cpRank?: number): void {
    const pPositionIds = this.cast(positionIds);
    const pLastPageLen = this.cast(lastPageLen);
    const pIndptr = this.cast(indptr);
    const cpWs = contextParallel ? this.worldSize : 1;
    for (let i = 0; i < this.worldSize; i++) {
      const cpR = contextParallel ? i : 0;
      this.devices[i].mlaDecodeStep(pPositionIds.shards[i], pLastPageLen.shards[i], pIndptr.shards[i], pageSize, batchSize, contextParallel, cpWs, cpR);
    }
  }

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
      this.devices[i].batchDecodePlan(pFloatWs.shards[i], floatWsSize, pIntWs.shards[i], pPinnedIntWs.shards[i], intWsSize, pPlanInfo.shards[i], pIndptrH.shards[i], batchSize, this.shardDim(numQoHeads, "batchDecodePlan numQoHeads"), this.shardDim(numKvHeads, "batchDecodePlan numKvHeads"), headDim, pageSize, enableCudaGraph);
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
      this.devices[i].batchDecodeRun(pQ.shards[i], pO.shards[i], pKData.shards[i], pVData.shards[i], pIndices.shards[i], pIndptrD.shards[i], pLastPageLen.shards[i], pFloatWs.shards[i], pIntWs.shards[i], pPlanInfo.shards[i], batchSize, this.shardDim(numQoHeads, "batchDecodeRun numQoHeads"), this.shardDim(numKvHeads, "batchDecodeRun numKvHeads"), headDim, pageSize, smScale);
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
      this.devices[i].batchPrefillPagedPlan(pFloatWs.shards[i], floatWsSize, pIntWs.shards[i], pPinnedIntWs.shards[i], intWsSize, pPlanInfo.shards[i], pQoIndptrH.shards[i], pPagedKvIndptrH.shards[i], totalQoRows, batchSize, this.shardDim(numQoHeads, "batchPrefillPagedPlan numQoHeads"), this.shardDim(numKvHeads, "batchPrefillPagedPlan numKvHeads"), headDim, pageSize, maskMode);
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
      this.devices[i].batchPrefillPagedRun(pQ.shards[i], pO.shards[i], pKData.shards[i], pVData.shards[i], pIndices.shards[i], pIndptrD.shards[i], pLastPageLen.shards[i], pFloatWs.shards[i], pIntWs.shards[i], pQIndptrD.shards[i], pPlanInfo.shards[i], totalQoRows, batchSize, this.shardDim(numQoHeads, "batchPrefillPagedRun numQoHeads"), this.shardDim(numKvHeads, "batchPrefillPagedRun numKvHeads"), headDim, pageSize, qStrideN, qStrideH, maskMode, smScale);
    }
  }

  private adjustCpLastPageLen(lastPageLenH: ParallelTensor, batchSize: number, seqKvLens: number[], pageSize: number): void {
    const cpWorldSize = this.worldSize;
    const effectivePageSize = pageSize / cpWorldSize;
    for (let r = 0; r < cpWorldSize; r++) {
      lastPageLenH.shards[r].withPinnedBuffer(buf => {
        for (let s = 0; s < batchSize; s++) {
          const N = seqKvLens[s];
          const localKvLen = N > r ? Math.floor((N - 1 - r) / cpWorldSize) + 1 : 0;
          const remainder = localKvLen % effectivePageSize;
          buf.writeInt32LE(localKvLen > 0 ? (remainder !== 0 ? remainder : effectivePageSize) : 0, s * 4);
        }
      });
    }
  }

  mlaPrefillPlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, qoIndptrH: Tensor, kvIndptrH: Tensor, kvLenH: Tensor, lastPageLenH: Tensor, batchSize: number, numHeads: number, headDimO: number, causal: boolean, pageSize: number, seqKvLens: number[], contextParallel?: boolean, _cpWorldSize?: number, _cpRank?: number): void {
    const pFloatWs = this.cast(floatWs);
    const pIntWs = this.cast(intWs);
    const pPinnedIntWs = this.cast(pinnedIntWs);
    const pPlanInfo = this.cast(planInfo);
    const pQoIndptrH = this.cast(qoIndptrH);
    const pKvIndptrH = this.cast(kvIndptrH);
    const pKvLenH = this.cast(kvLenH);
    const pLastPageLenH = this.cast(lastPageLenH);
    const effectiveNumHeads = contextParallel ? numHeads : this.shardDim(numHeads, "mlaPrefillPlan numHeads");
    const effectiveCpWorldSize = contextParallel ? this.worldSize : undefined;
    if (contextParallel) {
      this.adjustCpLastPageLen(pLastPageLenH, batchSize, seqKvLens, pageSize);
    }
    for (let i = 0; i < this.worldSize; i++) {
      const effectiveCpRank = contextParallel ? i : undefined;
      this.devices[i].mlaPrefillPlan(pFloatWs.shards[i], floatWsSize, pIntWs.shards[i], pPinnedIntWs.shards[i], intWsSize, pPlanInfo.shards[i], pQoIndptrH.shards[i], pKvIndptrH.shards[i], pKvLenH.shards[i], pLastPageLenH.shards[i], batchSize, effectiveNumHeads, headDimO, causal, pageSize, seqKvLens, contextParallel, effectiveCpWorldSize, effectiveCpRank);
    }
  }

  mlaPrefillRun(qNope: Tensor, qPe: Tensor, ckvData: Tensor, kpeData: Tensor, kvIndices: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, numHeads: number, pageSize: number, maskMode: number, smScale: number, qNopeStrideN: number, qNopeStrideH: number, qPeStrideN: number, qPeStrideH: number, ckvStridePage: number, ckvStrideN: number, kpeStridePage: number, kpeStrideN: number, oStrideN: number, oStrideH: number, headDimCkv: number, headDimKpe: number, contextParallel?: boolean): { o: Tensor, lse: Tensor } {
    let pQNope = this.cast(qNope);
    let pQPe = this.cast(qPe);
    const pCkvData = this.cast(ckvData);
    const pKpeData = this.cast(kpeData);
    const pKvIndices = this.cast(kvIndices);
    const pFloatWs = this.cast(floatWs);
    const pIntWs = this.cast(intWs);
    const pPlanInfo = this.cast(planInfo);
    const effectiveNumHeads = contextParallel ? numHeads : this.shardDim(numHeads, "mlaPrefillRun numHeads");
    const effectivePageSize = contextParallel ? pageSize / this.worldSize : pageSize;
    const effectiveQNopeStrideN = contextParallel ? qNopeStrideN : effectiveNumHeads * headDimCkv;
    const effectiveQPeStrideN = contextParallel ? qPeStrideN : effectiveNumHeads * headDimKpe;
    const effectiveCpWorldSize = contextParallel ? this.worldSize : undefined;
    const effectiveCkvStridePage = contextParallel ? ckvStridePage / this.worldSize : ckvStridePage;
    const effectiveKpeStridePage = contextParallel ? kpeStridePage / this.worldSize : kpeStridePage;
    const totalTokens = oStrideH / headDimCkv;
    const lsePar = contextParallel ? TensorParallelism.Column : TensorParallelism.Replicated;
    const lseFullShape = contextParallel ? [totalTokens * this.worldSize, numHeads] : [totalTokens, numHeads];
    const oPar = contextParallel ? TensorParallelism.PartialSoftmax : qNope.parallelism;
    const oFullShape = [1, numHeads, totalTokens, headDimCkv];
    // In CP mode, Q may be Row-parallel (head-sharded from Column-parallel weights).
    // AllGather to Replicated so each GPU has all heads for its KV shard.
    let gatheredQNope: ParallelTensor | undefined;
    let gatheredQPe: ParallelTensor | undefined;
    if (contextParallel) {
      if (pQNope.parallelism === TensorParallelism.Row) {
        if (pQPe.parallelism === TensorParallelism.Row) {
          using stream = this.withStream(() => {
            gatheredQPe = pQPe.allGather(pQPe.workspace);
            pQPe = gatheredQPe;
          });

          gatheredQNope = pQNope.allGather(pQNope.workspace);
          pQNope = gatheredQNope;

          stream.streamWaitEvent();
        }
        else {
            gatheredQNope = pQNope.allGather(pQNope.workspace);
            pQNope = gatheredQNope;
            gatheredQPe = pQPe;
        }
      }
    }
    const oShards: Tensor[] = [];
    const lseShards: Tensor[] = [];
    try {
      for (let i = 0; i < this.worldSize; i++) {
        const effectiveCpRank = contextParallel ? i : undefined;
        const shardResult = this.devices[i].mlaPrefillRun(pQNope.shards[i], pQPe.shards[i], pCkvData.shards[i], pKpeData.shards[i], pKvIndices.shards[i], pFloatWs.shards[i], pIntWs.shards[i], pPlanInfo.shards[i], effectiveNumHeads, effectivePageSize, maskMode, smScale, effectiveQNopeStrideN, qNopeStrideH, effectiveQPeStrideN, qPeStrideH, effectiveCkvStridePage, ckvStrideN, effectiveKpeStridePage, kpeStrideN, oStrideN, oStrideH, headDimCkv, headDimKpe, contextParallel, effectiveCpWorldSize, effectiveCpRank);
        oShards.push(shardResult.o);
        lseShards.push(shardResult.lse);
      }
    } finally {
      if (gatheredQNope) gatheredQNope[Symbol.dispose]();
      if (gatheredQPe) gatheredQPe[Symbol.dispose]();
    }
    const o = new ParallelTensor(pFloatWs.workspace, this, oPar, oShards, oFullShape, qNope.type, undefined, false, undefined);
    const lse = new ParallelTensor(pFloatWs.workspace, this, lsePar, lseShards, lseFullShape, "F32", undefined, false, undefined);
    return { o, lse };
  }

  mlaDecodePlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, indptrH: Tensor, lastPageLenH: Tensor, batchSize: number, numQoHeads: number, pageSize: number, enableCudaGraph: boolean, headDimCkv: number, headDimKpe: number, contextParallel?: boolean, _cpWorldSize?: number, _cpRank?: number, seqKvLens?: number[]): void {
    const pFloatWs = this.cast(floatWs);
    const pIntWs = this.cast(intWs);
    const pPinnedIntWs = this.cast(pinnedIntWs);
    const pPlanInfo = this.cast(planInfo);
    const pIndptrH = this.cast(indptrH);
    const pLastPageLenH = this.cast(lastPageLenH);
    const effectiveNumQoHeads = contextParallel ? numQoHeads : this.shardDim(numQoHeads, "mlaDecodePlan numQoHeads");
    const effectivePageSize = contextParallel ? pageSize / this.worldSize : pageSize;
    if (contextParallel && seqKvLens) {
      this.adjustCpLastPageLen(pLastPageLenH, batchSize, seqKvLens, pageSize);
    }
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].mlaDecodePlan(pFloatWs.shards[i], floatWsSize, pIntWs.shards[i], pPinnedIntWs.shards[i], intWsSize, pPlanInfo.shards[i], pIndptrH.shards[i], pLastPageLenH.shards[i], batchSize, effectiveNumQoHeads, effectivePageSize, enableCudaGraph, headDimCkv, headDimKpe, contextParallel);
    }
  }

  mlaDecodeRun(qNope: Tensor, qPe: Tensor, ckvData: Tensor, kpeData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, batchSize: number, numQoHeads: number, pageSize: number, smScale: number, headDimCkv: number, headDimKpe: number, contextParallel?: boolean, _cpWorldSize?: number, _cpRank?: number): { o: Tensor, lse: Tensor } {
    let pQNope = this.cast(qNope);
    let pQPe = this.cast(qPe);
    const pCkvData = this.cast(ckvData);
    const pKpeData = this.cast(kpeData);
    const pIndices = this.cast(indices);
    const pIndptrD = this.cast(indptrD);
    const pLastPageLen = this.cast(lastPageLen);
    const pFloatWs = this.cast(floatWs);
    const pIntWs = this.cast(intWs);
    const pPlanInfo = this.cast(planInfo);
    const effectiveNumQoHeads = contextParallel ? numQoHeads : this.shardDim(numQoHeads, "mlaDecodeRun numQoHeads");
    const effectivePageSize = contextParallel ? pageSize / this.worldSize : pageSize;
    const lsePar = contextParallel ? TensorParallelism.Column : TensorParallelism.Replicated;
    const lseFullShape = contextParallel ? [batchSize * this.worldSize, numQoHeads] : [batchSize, numQoHeads];
    const oPar = contextParallel ? TensorParallelism.PartialSoftmax : qNope.parallelism;
    const oFullShape = [batchSize, numQoHeads, 1, headDimCkv];
    let gatheredQNope: ParallelTensor | undefined;
    let gatheredQPe: ParallelTensor | undefined;
    if (contextParallel) {
      if (pQNope.parallelism === TensorParallelism.Row) {
        gatheredQNope = pQNope.allGather(pQNope.workspace);
        pQNope = gatheredQNope;
      }
      if (pQPe.parallelism === TensorParallelism.Row) {
        gatheredQPe = pQPe.allGather(pQPe.workspace);
        pQPe = gatheredQPe;
      }
    }
    const oShards: Tensor[] = [];
    const lseShards: Tensor[] = [];
    try {
      for (let i = 0; i < this.worldSize; i++) {
        const shardResult = this.devices[i].mlaDecodeRun(pQNope.shards[i], pQPe.shards[i], pCkvData.shards[i], pKpeData.shards[i], pIndices.shards[i], pIndptrD.shards[i], pLastPageLen.shards[i], pFloatWs.shards[i], pIntWs.shards[i], pPlanInfo.shards[i], batchSize, effectiveNumQoHeads, effectivePageSize, smScale, headDimCkv, headDimKpe);
        oShards.push(shardResult.o);
        lseShards.push(shardResult.lse);
      }
    } finally {
      if (gatheredQNope) gatheredQNope[Symbol.dispose]();
      if (gatheredQPe) gatheredQPe[Symbol.dispose]();
    }
    const o = new ParallelTensor(pFloatWs.workspace, this, oPar, oShards, oFullShape, qNope.type, undefined, false, undefined);
    const lse = new ParallelTensor(pFloatWs.workspace, this, lsePar, lseShards, lseFullShape, "F32", undefined, false, undefined);
    return { o, lse };
  }

  mlaKvCacheAppend(ckvData: Tensor, kpeData: Tensor, indices: Tensor, indptr: Tensor, lastPageLen: Tensor, appendCkv: Tensor, appendKpe: Tensor, batchIndices: Tensor, positions: Tensor, nnz: number, pageSize: number, headDimCkv: number, headDimKpe: number, appendCkvStrideN: number, appendKpeStrideN: number, contextParallel?: boolean, _cpWorldSize?: number, _cpRank?: number): void {
    const pCkvData = this.cast(ckvData);
    const pKpeData = this.cast(kpeData);
    const pIndices = this.cast(indices);
    const pIndptr = this.cast(indptr);
    const pLastPageLen = this.cast(lastPageLen);
    const pAppendCkv = this.cast(appendCkv);
    const pAppendKpe = this.cast(appendKpe);
    const pBatchIndices = this.cast(batchIndices);
    const pPositions = this.cast(positions);
    const effectiveCpWorldSize = contextParallel ? this.worldSize : undefined;
    for (let i = 0; i < this.worldSize; i++) {
      const effectiveCpRank = contextParallel ? i : undefined;
      this.devices[i].mlaKvCacheAppend(pCkvData.shards[i], pKpeData.shards[i], pIndices.shards[i], pIndptr.shards[i], pLastPageLen.shards[i], pAppendCkv.shards[i], pAppendKpe.shards[i], pBatchIndices.shards[i], pPositions.shards[i], nnz, pageSize, headDimCkv, headDimKpe, appendCkvStrideN, appendKpeStrideN, contextParallel, effectiveCpWorldSize, effectiveCpRank);
    }
  }

  private graphHandles: (number | undefined)[][] = [];
  private graphExecHandles: number[][] = [];

  graphBeginCapture(): void {
    for (const device of this.devices) {
      device.graphBeginCapture();
    }
  }

  graphEndCapture(): number {
    const graphs = this.devices.map(d => d.graphEndCapture());
    const idx = this.graphHandles.length;
    this.graphHandles.push(graphs);
    return idx;
  }

  graphInstantiate(graph: number): number {
    const handles = this.graphHandles[graph]!;
    const execs = this.devices.map((d, i) => d.graphInstantiate(handles[i]!));
    const idx = this.graphExecHandles.length;
    this.graphExecHandles.push(execs);
    return idx;
  }

  graphLaunch(graphExec: number): void {
    const execs = this.graphExecHandles[graphExec];
    for (let i = 0; i < this.devices.length; i++) {
      this.devices[i].graphLaunch(execs[i]);
    }
  }

  graphDestroy(graph: number): void {
    const handles = this.graphHandles[graph];
    if (handles !== undefined) {
      for (let i = 0; i < this.devices.length; i++) {
        this.devices[i].graphDestroy(handles[i]!);
      }
      delete this.graphHandles[graph];
    }
  }

  graphExecDestroy(graphExec: number): void {
    const execs = this.graphExecHandles[graphExec];
    for (let i = 0; i < this.devices.length; i++) {
      this.devices[i].graphExecDestroy(execs[i]);
    }
    delete this.graphExecHandles[graphExec];
  }
}
