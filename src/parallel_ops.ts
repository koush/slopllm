import { DeviceOps, GdnQkvLayout, TensorParallelism } from "./device_ops";
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
  }

  private get worldSize(): number {
    return this.shards.length;
  }

  private static elemBytes(type: string): number {
    switch (type) {
      case "BF16": return 2;
      case "I32": return 4;
      case "F32": return 4;
      default: return 1;
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
      if (this.pinned) {
        this.devices[i].freePinned(shard);
      } else {
        this.devices[i].freeBuf(shard);
      }
      shard.detachData();
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

  shard(rank: number): Tensor {
    return this.shards[rank];
  }

  override reshape(newShape: number[]): Tensor {
    const current = this.shape.reduce((a, b) => a * b, 1);
    const target = newShape.reduce((a, b) => a * b, 1);
    if (current !== target) {
      throw new Error(`reshape: cannot reshape [${this.shape}] (${current} elements) to [${newShape}] (${target} elements)`);
    }
    const newShardShape = this.parallelOps.shardShape(newShape, this.parallelism);
    const reshapedShards: Tensor[] = this.shards.map(s => s.reshape(newShardShape));
    return new ParallelTensor(
      this.workspace, this.parallelOps, this.parallelism,
      reshapedShards, newShape, this.type, undefined, this.pinned, this,
    );
  }

  allReduce(): ParallelTensor {
    if (this.parallelism !== TensorParallelism.PartialSum) {
      throw new Error(`allReduce requires PartialSum tensor, got ${this.parallelism}`);
    }
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
            MemcpyKind.DeviceToDevice,
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
    const eb = ParallelTensor.elemBytes(this.type);

    if (this.parallelism === TensorParallelism.Replicated || this.parallelism === TensorParallelism.PartialSum) {
      const sz = size ?? data.length;
      for (let i = 0; i < this.worldSize; i++) {
        this.devices[i].h2d(this.shards[i], data.subarray(0, sz));
      }
      return;
    }

    if (this.parallelism === TensorParallelism.Column) {
      const totalElems = ParallelTensor.shapeElems(this.fullShape);
      const shardBytes = (totalElems / this.worldSize) * eb;
      for (let i = 0; i < this.worldSize; i++) {
        this.devices[i].h2d(this.shards[i], data.subarray(i * shardBytes, (i + 1) * shardBytes));
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
        this.devices[i].h2d(this.shards[i], shardBuf);
      }
      return;
    }

    throw new Error(`ParallelTensor.h2d: unsupported parallelism ${this.parallelism}`);
  }

  d2h(buf: Buffer, size?: number): void {
    const eb = ParallelTensor.elemBytes(this.type);

    if (this.parallelism === TensorParallelism.Replicated) {
      const shardBytes = ParallelTensor.shapeElems(this.shards[0].shape) * eb;
      this.devices[0].d2h(buf.subarray(0, shardBytes), this.shards[0]);
      return;
    }

    if (this.parallelism === TensorParallelism.Column) {
      const totalElems = ParallelTensor.shapeElems(this.fullShape);
      const shardBytes = (totalElems / this.worldSize) * eb;
      for (let i = 0; i < this.worldSize; i++) {
        this.devices[i].d2h(buf.subarray(i * shardBytes, (i + 1) * shardBytes), this.shards[i]);
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
        this.devices[i].d2h(shardBuf, this.shards[i]);
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
          this.devices[i].d2h(shardBuf, this.shards[i]);
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
          this.devices[i].d2h(shardBuf, this.shards[i]);
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
          this.devices[i].d2h(shardBuf, this.shards[i]);
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
    let k = weight.shape[1];
    if (weight.type === "U8") k = k * 2; // NVFP4: weight is [N, K/2] packed, kernel expects K
    const outPar = ParallelOps.linearOutputParallelism(pWeight.parallelism, this.parallelism);

    if (weight.type === "F8_E4M3") {
      const scale = weight.workspace.tensors.get(weight.name! + "_scale_inv")!;
      const pScale = scale as ParallelTensor;
      const out = this.workspace.alloc([batch, n], this.type, undefined, outPar) as ParallelTensor;

      if (pWeight.parallelism === TensorParallelism.Column && this.parallelism === TensorParallelism.Replicated) {
        const shardN = n / this.worldSize;
        for (let i = 0; i < this.worldSize; i++) {
          this.devices[i].fp8LinearDecode(out.shards[i], this.shards[i], pWeight.shards[i], pScale.shards[i], batch, shardN, k);
        }
      } else if (pWeight.parallelism === TensorParallelism.Row && this.parallelism === TensorParallelism.Row) {
        const shardK = k / this.worldSize;
        for (let i = 0; i < this.worldSize; i++) {
          this.devices[i].fp8LinearDecode(out.shards[i], this.shards[i], pWeight.shards[i], pScale.shards[i], batch, n, shardK);
        }
      } else if (pWeight.parallelism === TensorParallelism.Replicated && this.parallelism === TensorParallelism.Replicated) {
        for (let i = 0; i < this.worldSize; i++) {
          this.devices[i].fp8LinearDecode(out.shards[i], this.shards[i], pWeight.shards[i], pScale.shards[i], batch, n, k);
        }
      } else {
        throw new Error(`fp8LinearDecode: unsupported parallelism W=${pWeight.parallelism}, X=${this.parallelism}`);
      }
      return out;
    }

    if (weight.type === "U8") {
      const scale = weight.workspace.tensors.get(weight.name! + "_weight_scale")!;
      const scale2 = weight.workspace.tensors.get(weight.name! + "_weight_scale_2")!;
      const pScale = scale as ParallelTensor;
      const pScale2 = scale2 as ParallelTensor;
      const out = this.workspace.alloc([batch, n], this.type, undefined, outPar) as ParallelTensor;

      if (pWeight.parallelism === TensorParallelism.Column && this.parallelism === TensorParallelism.Replicated) {
        const shardN = n / this.worldSize;
        for (let i = 0; i < this.worldSize; i++) {
          this.devices[i].nvfp4LinearDecode(out.shards[i], this.shards[i], pWeight.shards[i], pScale.shards[i], pScale2.shards[i], batch, shardN, k);
        }
      } else if (pWeight.parallelism === TensorParallelism.Row && this.parallelism === TensorParallelism.Row) {
        const shardK = k / this.worldSize;
        for (let i = 0; i < this.worldSize; i++) {
          this.devices[i].nvfp4LinearDecode(out.shards[i], this.shards[i], pWeight.shards[i], pScale.shards[i], pScale2.shards[i], batch, n, shardK);
        }
      } else if (pWeight.parallelism === TensorParallelism.Replicated && this.parallelism === TensorParallelism.Replicated) {
        for (let i = 0; i < this.worldSize; i++) {
          this.devices[i].nvfp4LinearDecode(out.shards[i], this.shards[i], pWeight.shards[i], pScale.shards[i], pScale2.shards[i], batch, n, k);
        }
      } else {
        throw new Error(`nvfp4LinearDecode: unsupported parallelism W=${pWeight.parallelism}, X=${this.parallelism}`);
      }
      return out;
    }

    const shards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      shards.push(this.shards[i].linear(pWeight.shards[i], batch));
    }
    return this.parallelOps.wrapShards(this.workspace, shards, [batch, n], this.type, outPar);
  }

  bmm(B: Tensor, batch: number, M: number, N: number, K: number, transA: boolean = false, transB: boolean = false): Tensor {
    const pB = B as ParallelTensor;
    const shardBatch = (this.parallelism === TensorParallelism.Column) ? batch / this.worldSize : batch;
    const outShards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      outShards.push(this.shards[i].bmm(pB.shards[i], shardBatch, M, N, K, transA, transB));
    }
    return this.parallelOps.wrapShards(this.workspace, outShards, [batch * M, N], this.type, this.parallelism);
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
    if (this.shape.length === 2 && other.shape.length === 1 && this.shape[1] === other.shape[0]) {
      const outShards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        outShards.push(this.shards[i].add(pOther.shards[i], n));
      }
      return this.parallelOps.wrapShards(this.workspace, outShards, this.shape, this.type, this.parallelism);
    }
    const outShards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      outShards.push(this.shards[i].add(pOther.shards[i], n));
    }
    return this.parallelOps.wrapShards(this.workspace, outShards, this.shape, this.type, this.parallelism);
  }

  mul(other: Tensor, n?: number): Tensor {
    const pOther = other as ParallelTensor;
    if (this.shape.length === 2 && other.shape.length === 1 && this.shape[1] === other.shape[0]) {
      const outShards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        outShards.push(this.shards[i].mul(pOther.shards[i], n));
      }
      return this.parallelOps.wrapShards(this.workspace, outShards, this.shape, this.type, this.parallelism);
    }
    const outShards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      outShards.push(this.shards[i].mul(pOther.shards[i], n));
    }
    return this.parallelOps.wrapShards(this.workspace, outShards, this.shape, this.type, this.parallelism);
  }

  rmsnorm(weight: Tensor, eps: number, dim: number, batch: number): Tensor {
    super.rmsnorm(weight, eps, dim, batch);
    if (this.parallelism === TensorParallelism.Row || this.parallelism === TensorParallelism.Column) {
      const gathered = this.allGather(this.workspace);
      const result = gathered.rmsnorm(weight, eps, dim, batch);
      gathered[Symbol.dispose]();
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
      const gathered = this.allGather(this.workspace);
      const result = gathered.fusedAddRmsnorm(input, weight, eps, dim, batch);
      gathered[Symbol.dispose]();
      return result;
    }

    if (pInput.parallelism === TensorParallelism.PartialSum) {
      pInput.allReduce();
      return this.fusedAddRmsnorm(pInput, weight, eps, dim, batch);
    }

    if (pInput.parallelism === TensorParallelism.Row || pInput.parallelism === TensorParallelism.Column) {
      const gathered = pInput.allGather(pInput.workspace);
      const result = this.fusedAddRmsnorm(gathered, weight, eps, dim, batch);
      gathered[Symbol.dispose]();
      return result;
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

  override fusedNormRope(weight: Tensor, cos: Tensor, sin: Tensor, eps: number, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, inStride?: number): Tensor {
    super.fusedNormRope(weight, cos, sin, eps, ropeDim, headDim, nHeads, seqLen, batch, inStride);
    if (this.parallelism === TensorParallelism.PartialSum) {
      this.allReduce();
      return this.fusedNormRope(weight, cos, sin, eps, ropeDim, headDim, nHeads, seqLen, batch, inStride);
    }

    if (this.parallelism === TensorParallelism.Column) {
      const gathered = this.allGather(this.workspace);
      const result = gathered.fusedNormRope(weight, cos, sin, eps, ropeDim, headDim, nHeads, seqLen, batch, inStride);
      gathered[Symbol.dispose]();
      return result;
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
        shards.push(this.shards[i].fusedNormRope(pWeight.shards[i], pCos.shards[i], pSin.shards[i], eps, ropeDim, headDim, shardNHeads, seqLen, batch, shardInStride));
      }
      return this.parallelOps.wrapShards(this.workspace, shards, [batch, nHeads, seqLen, headDim], this.type, TensorParallelism.Row);
    }

    this.assertParallel("fusedNormRope input", this, TensorParallelism.Replicated);
    this.assertParallel("fusedNormRope weight", pWeight, TensorParallelism.Replicated);
    this.assertParallel("fusedNormRope cos", pCos, TensorParallelism.Replicated);
    this.assertParallel("fusedNormRope sin", pSin, TensorParallelism.Replicated);

    const shards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      shards.push(this.shards[i].fusedNormRope(pWeight.shards[i], pCos.shards[i], pSin.shards[i], eps, ropeDim, headDim, nHeads, seqLen, batch, stride));
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

  override siluAndMul(gate: Tensor, up: Tensor, intermediate: number, batch: number): Tensor {
    super.siluAndMul(gate, up, intermediate, batch);
    const pGate = gate as ParallelTensor;
    const pUp = up as ParallelTensor;
    if (pGate.parallelism !== pUp.parallelism) {
      throw new Error(`siluAndMul: gate parallelism ${pGate.parallelism} != up parallelism ${pUp.parallelism}`);
    }
    const outPar = pGate.parallelism;
    const shardIntermediate = pGate.parallelism === TensorParallelism.Row || pGate.parallelism === TensorParallelism.Column
      ? intermediate / this.worldSize
      : intermediate;
    const shards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      shards.push(this.shards[i].siluAndMul(pGate.shards[i], pUp.shards[i], shardIntermediate, batch));
    }
    return this.parallelOps.wrapShards(this.workspace, shards, [batch, intermediate], this.type, outPar);
  }

  arange(start: number, step: number, count: number): void {
    super.arange(start, step, count);
    this.assertParallel("arange", this, TensorParallelism.Replicated, TensorParallelism.PartialSum);
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].arange(this.shards[i], start, step, count);
    }
  }

  argmax(): Tensor {
    super.argmax();
    const { indices, values} = this.max();
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

      const allValuesPar = this.parallelOps.wrapShards(this.workspace, localValuesShards, [batch, ws], this.type, TensorParallelism.Row);
      const allIndicesPar = this.parallelOps.wrapShards(this.workspace, localIndicesShards, [batch, ws], "I32", TensorParallelism.Row);

      const allValues = allValuesPar.allGather(this.workspace);
      const allIndices = allIndicesPar.allGather(this.workspace);
      allValuesPar[Symbol.dispose]();
      allIndicesPar[Symbol.dispose]();

      const { values: rankValues, indices: rankIndices } = allValues.max(0);
      allValues[Symbol.dispose]();

      const gatheredIndices = allIndices.gather(rankIndices, 1, ws, batch);
      allIndices[Symbol.dispose]();
      rankIndices[Symbol.dispose]();

      const finalIndices = this.parallelOps.newTensor(this.workspace, [batch], "I32", false, undefined, TensorParallelism.Replicated);
      const pGatheredIndices = gatheredIndices as ParallelTensor;
      const idxBytes = batch * 4;
      for (let i = 0; i < ws; i++) {
        finalIndices.shards[i].memcpy(pGatheredIndices.shards[i], idxBytes);
      }
      gatheredIndices[Symbol.dispose]();

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
      this.devices[i].gdnRecurrentStep(this.shards[i], pState.shards[i], pQkv.shards[i], pARaw.shards[i], pBRaw.shards[i], pALog.shards[i], pDtBias.shards[i], shardHeads, dK, dV, batchSize, shardStateStride, qkvChStride, shardSeqStride);
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
      this.devices[i].gdnPrefill(this.shards[i], pState.shards[i], pQkv.shards[i], pARaw.shards[i], pBRaw.shards[i], pALog.shards[i], pDtBias.shards[i], pCuSeqlens.shards[i], totalSeqLen, shardHeads, dK, dV, batchSize, shardStateStride, qkvChStride, shardSeqStride);
    }
  }

  mlaKvCacheAppend(ckvData: Tensor, kpeData: Tensor, indices: Tensor, indptr: Tensor, lastPageLen: Tensor, appendCkv: Tensor, appendKpe: Tensor, batchIndices: Tensor, positions: Tensor, nnz: number, pageSize: number, headDimCkv: number, headDimKpe: number, appendCkvStrideN: number, appendKpeStrideN: number): void {
    super.mlaKvCacheAppend(ckvData, kpeData, indices, indptr, lastPageLen, appendCkv, appendKpe, batchIndices, positions, nnz, pageSize, headDimCkv, headDimKpe, appendCkvStrideN, appendKpeStrideN);
    const pCkvData = this.cast(ckvData);
    const pKpeData = this.cast(kpeData);
    const pIndices = this.cast(indices);
    const pIndptr = this.cast(indptr);
    const pLastPageLen = this.cast(lastPageLen);
    const pAppendCkv = this.cast(appendCkv);
    const pAppendKpe = this.cast(appendKpe);
    const pBatchIndices = this.cast(batchIndices);
    const pPositions = this.cast(positions);
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].mlaKvCacheAppend(pCkvData.shards[i], pKpeData.shards[i], pIndices.shards[i], pIndptr.shards[i], pLastPageLen.shards[i], pAppendCkv.shards[i], pAppendKpe.shards[i], pBatchIndices.shards[i], pPositions.shards[i], nnz, pageSize, headDimCkv, headDimKpe, appendCkvStrideN, appendKpeStrideN);
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
      this.devices[i].causalConv1d(this.shards[i], pConvState.shards[i], pInput.shards[i], pWeight.shards[i], pCuSeqlens.shards[i], shardConvDim, totalSeqLen, kernelSize, batchSize, shardConvStateStride, chStride, shardSeqStride);
    }
  }

  causalConv1dUpdate(convState: Tensor, input: Tensor, weight: Tensor, convDim: number, kernelSize: number, batchSize: number, convStateStride: number): Tensor {
    super.causalConv1dUpdate(convState, input, weight, convDim, kernelSize, batchSize, convStateStride);
    const pConvState = this.cast(convState);
    const pInput = input as ParallelTensor;
    const pWeight = this.cast(weight);
    const isRowPar = pInput.parallelism === TensorParallelism.Row || pInput.parallelism === TensorParallelism.Column;
    const parallelism = isRowPar ? TensorParallelism.Row : TensorParallelism.Replicated;
    const out = this.workspace.alloc([batchSize, convDim], this.type, undefined, parallelism) as ParallelTensor;
    const shardConvDim = isRowPar ? convDim / this.worldSize : convDim;
    const shardConvStateStride = isRowPar ? convStateStride / this.worldSize : convStateStride;
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].causalConv1dUpdate(out.shards[i], pConvState.shards[i], pInput.shards[i], pWeight.shards[i], shardConvDim, kernelSize, batchSize, shardConvStateStride);
    }
    return out;
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
        this.devices[i].rmsnormGated(this.shards[i], pInput.shards[i], pGate.shards[i], pWeight.shards[i], eps, dim, shardBatch);
      }
      return;
    }

    if (pInput.parallelism === TensorParallelism.Column) {
      const shardBatch = batch / this.worldSize;
      for (let i = 0; i < this.worldSize; i++) {
        this.devices[i].rmsnormGated(this.shards[i], pInput.shards[i], pGate.shards[i], pWeight.shards[i], eps, dim, shardBatch);
      }
      return;
    }

    if (pInput.parallelism === TensorParallelism.Row) {
      const gathered = pInput.allGather(pInput.workspace);
      this.rmsnormGated(gathered, gate, weight, eps, dim, batch);
      gathered[Symbol.dispose]();
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
      this.devices[i].rmsnormGated(this.shards[i], pInput.shards[i], pGate.shards[i], pWeight.shards[i], eps, dim, batch);
    }
  }

  gateSigmoidMul(gate: Tensor, batchSeq: number, numHeads: number, headDim: number): void {
    super.gateSigmoidMul(gate, batchSeq, numHeads, headDim);
    const pGate = gate as ParallelTensor;
    const shardNumHeads = this.parallelism === TensorParallelism.Row || this.parallelism === TensorParallelism.Column
      ? this.shardDim(numHeads, "gateSigmoidMul numHeads")
      : numHeads;
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].gateSigmoidMul(this.shards[i], pGate.shards[i], batchSeq, shardNumHeads, headDim);
    }
  }

  fill(value: number, n: number): void {
    const shardN = this.parallelism === TensorParallelism.Row || this.parallelism === TensorParallelism.Column
      ? n / this.worldSize
      : n;
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].fill(this.shards[i], value, shardN);
    }
  }

  mmapLoad(mmapPtr: number, offset: number, nbytes: number, gdnQkvLayout?: GdnQkvLayout): void {
    if (gdnQkvLayout && this.parallelism === TensorParallelism.Column) {
      const { numHeads, dK, dV } = gdnQkvLayout;
      const Hlocal = this.shardDim(numHeads, "mmapLoad numHeads");
      const qRows = numHeads * dK;
      const bytesPerRow = this.fullShape.slice(1).reduce((a, b) => a * b, 1) * ParallelTensor.elemBytes(this.type);
      const srcBase = mmapPtr + offset;
      for (let i = 0; i < this.worldSize; i++) {
        const hStart = i * Hlocal;
        const shardData = this.shards[i].data;
        this.devices[i].memcpy2d(
          shardData, bytesPerRow,
          srcBase + hStart * dK * bytesPerRow, bytesPerRow,
          bytesPerRow, Hlocal * dK,
          MemcpyKind.HostToDevice,
        );
        this.devices[i].memcpy2d(
          shardData + Hlocal * dK * bytesPerRow, bytesPerRow,
          srcBase + (qRows + hStart * dK) * bytesPerRow, bytesPerRow,
          bytesPerRow, Hlocal * dK,
          MemcpyKind.HostToDevice,
        );
        this.devices[i].memcpy2d(
          shardData + 2 * Hlocal * dK * bytesPerRow, bytesPerRow,
          srcBase + (2 * qRows + hStart * dV) * bytesPerRow, bytesPerRow,
          bytesPerRow, Hlocal * dV,
          MemcpyKind.HostToDevice,
        );
      }
      return;
    }

    if (this.parallelism === TensorParallelism.Replicated || this.parallelism === TensorParallelism.PartialSum) {
      for (let i = 0; i < this.worldSize; i++) {
        this.devices[i].mmapLoad(this.shards[i], mmapPtr, offset, nbytes);
      }
      return;
    }

    if (this.parallelism === TensorParallelism.Column) {
      const shardElems = this.shards[0].shape.reduce((a, b) => a * b, 1);
      const shardBytes = shardElems * ParallelTensor.elemBytes(this.type);
      for (let i = 0; i < this.worldSize; i++) {
        this.devices[i].mmapLoad(this.shards[i], mmapPtr, offset + i * shardBytes, shardBytes);
      }
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
      const srcBase = mmapPtr + offset;
      for (let i = 0; i < this.worldSize; i++) {
        this.devices[i].memcpy2d(
          this.shards[i].data, dstPitch,
          srcBase + i * dstPitch, srcPitch,
          dstPitch, outer,
          MemcpyKind.HostToDevice,
        );
      }
      return;
    }

    throw new Error(`mmapLoad: unsupported parallelism ${this.parallelism}`);
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
      this.devices[i].memcpy(this.shards[i].data, src.shards[i].data, bytes, copyKind);
    }
  }

  sigmoid(): Tensor {
    const outShards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      outShards.push(this.shards[i].sigmoid());
    }
    return this.parallelOps.wrapShards(this.workspace, outShards, this.fullShape, this.type, this.parallelism);
  }

  topk(k: number, dim: number): { values: Tensor, indices: Tensor } {
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
    const outShards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      outShards.push(this.shards[i].rowNormalize(scale, dim, batch, normalize));
    }
    return this.parallelOps.wrapShards(this.workspace, outShards, this.fullShape, this.type, this.parallelism);
  }

  scatterScalar(indices: Tensor, value: number, k: number, outDim: number, batch: number): void {
    const pIndices = indices as ParallelTensor;
    for (let i = 0; i < this.worldSize; i++) {
      this.shards[i].scatterScalar(pIndices.shards[i], value, k, outDim, batch);
    }
  }

  groupMaskMul(groupMask: Tensor, numExperts: number, expertsPerGroup: number, nGroup: number, batch: number): void {
    const pGroupMask = groupMask as ParallelTensor;
    for (let i = 0; i < this.worldSize; i++) {
      this.shards[i].groupMaskMul(pGroupMask.shards[i], numExperts, expertsPerGroup, nGroup, batch);
    }
  }

  ropeTranspose(cos: Tensor, sin: Tensor, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, inStride?: number): Tensor {
    super.ropeTranspose(cos, sin, ropeDim, headDim, nHeads, seqLen, batch, inStride);
    const pCos = cos ? cos as ParallelTensor : undefined;
    const pSin = sin ? sin as ParallelTensor : undefined;
    const shardNHeads = (this.parallelism === TensorParallelism.Row || this.parallelism === TensorParallelism.Column)
      ? this.parallelOps.shardDim(nHeads, "ropeTranspose nHeads")
      : nHeads;
    const outShards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      const shardCos = pCos ? pCos.shards[i] : undefined!;
      const shardSin = pSin ? pSin.shards[i] : undefined!;
      outShards.push(this.shards[i].ropeTranspose(shardCos, shardSin, ropeDim, headDim, shardNHeads, seqLen, batch, inStride ?? headDim));
    }
    return this.parallelOps.wrapShards(this.workspace, outShards, [batch * nHeads, seqLen, headDim], this.type, this.parallelism);
  }

  applyRotaryPosEmb(cos: Tensor, sin: Tensor, ropeDim: number, nHeads: number, seqLen: number, batch: number, unsqueezeDim: number): Tensor {
    super.applyRotaryPosEmb(cos, sin, ropeDim, nHeads, seqLen, batch, unsqueezeDim);
    const pCos = cos as ParallelTensor;
    const pSin = sin as ParallelTensor;
    const shardNHeads = (this.parallelism === TensorParallelism.Row || this.parallelism === TensorParallelism.Column)
      ? this.parallelOps.shardDim(nHeads, "applyRotaryPosEmb nHeads")
      : nHeads;
    const outShards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      outShards.push(this.shards[i].applyRotaryPosEmb(pCos.shards[i], pSin.shards[i], ropeDim, shardNHeads, seqLen, batch, unsqueezeDim));
    }
    return this.parallelOps.wrapShards(this.workspace, outShards, this.fullShape, this.type, this.parallelism);
  }

  mlaVExpand(vProj: Tensor, kvLoraRank: number, vHeadDim: number, nHeads: number, seqLen: number, batch: number): Tensor {
    super.mlaVExpand(vProj, kvLoraRank, vHeadDim, nHeads, seqLen, batch);
    const pVProj = vProj as ParallelTensor;
    const shardNHeads = (this.parallelism === TensorParallelism.Row || this.parallelism === TensorParallelism.Column)
      ? this.parallelOps.shardDim(nHeads, "mlaVExpand nHeads")
      : nHeads;
    const BS = batch * seqLen;
    const outShards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      outShards.push(this.shards[i].mlaVExpand(pVProj.shards[i], kvLoraRank, vHeadDim, shardNHeads, seqLen, batch));
    }
    return this.parallelOps.wrapShards(this.workspace, outShards, [BS, nHeads * vHeadDim], this.type, this.parallelism);
  }

  mulMatId(input: Tensor, weightPtrs: Tensor, expertIds: Tensor, batchIds: Tensor, count: number, N: number, K: number): Tensor {
    const pInput = input as ParallelTensor;
    const pWeightPtrs = weightPtrs as ParallelTensor;
    const pExpertIds = expertIds as ParallelTensor;
    const pBatchIds = batchIds as ParallelTensor;
    const weightPar = pWeightPtrs.parallelism;
    if (weightPar !== TensorParallelism.Replicated) {
      throw new Error(`mulMatId: weight pointers must be Replicated, got ${weightPar}`);
    }
    const inputPar = pInput.parallelism;
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
      outShards.push(pInput.shards[i].mulMatId(pInput.shards[i], pWeightPtrs.shards[i], pExpertIds.shards[i], pBatchIds.shards[i], count, outN, outK));
    }
    return this.parallelOps.wrapShards(this.workspace, outShards, [count, N], this.type, outPar);
  }

  nvfp4MulMatId(input: Tensor, weightPtrs: Tensor, scalePtrs: Tensor, scale2Ptrs: Tensor, expertIds: Tensor, batchIds: Tensor, count: number, N: number, K: number): Tensor {
    const pInput = input as ParallelTensor;
    const pWeightPtrs = weightPtrs as ParallelTensor;
    const pScalePtrs = scalePtrs as ParallelTensor;
    const pScale2Ptrs = scale2Ptrs as ParallelTensor;
    const pExpertIds = expertIds as ParallelTensor;
    const pBatchIds = batchIds as ParallelTensor;
    const weightPar = pWeightPtrs.parallelism;
    if (weightPar !== TensorParallelism.Replicated) {
      throw new Error(`nvfp4MulMatId: weight pointers must be Replicated, got ${weightPar}`);
    }
    const inputPar = pInput.parallelism;
    let outN = N, outK = K, outPar: TensorParallelism;
    if (inputPar === TensorParallelism.Replicated) {
      outN = this.parallelOps.shardDim(N, "nvfp4MulMatId N");
      outPar = TensorParallelism.Row;
    } else if (inputPar === TensorParallelism.Row) {
      outK = this.parallelOps.shardDim(K, "nvfp4MulMatId K");
      outPar = TensorParallelism.PartialSum;
    } else {
      throw new Error(`nvfp4MulMatId: unsupported input parallelism ${inputPar}`);
    }
    const outShards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      outShards.push(pInput.shards[i].nvfp4MulMatId(pInput.shards[i], pWeightPtrs.shards[i], pScalePtrs.shards[i], pScale2Ptrs.shards[i], pExpertIds.shards[i], pBatchIds.shards[i], count, outN, outK));
    }
    return this.parallelOps.wrapShards(this.workspace, outShards, [count, N], this.type, outPar);
  }

  scatterAddRows(input: Tensor, scales: Tensor, batchIds: Tensor, dim: number, count: number, numRows: number, _workspace?: Tensor): void {
    const pInput = input as ParallelTensor;
    const pScales = scales as ParallelTensor;
    const pBatchIds = batchIds as ParallelTensor;
    if (pInput.parallelism === TensorParallelism.PartialSum) {
      pInput.allReduce();
    }
    for (let i = 0; i < this.worldSize; i++) {
      this.shards[i].scatterAddRows(pInput.shards[i], pScales.shards[i], pBatchIds.shards[i], dim, count, numRows);
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

  doSampleBatch(outTokens: Tensor, topkVals: Tensor, topkIdxs: Tensor, workspace: Tensor, logits: Tensor, penaltyTokens: Tensor, penaltyCount: Tensor, maxWindow: number, vocabSize: number, batchSize: number, temperatures: Tensor, repPenalties: Tensor, presPenalties: Tensor, topKs: Tensor, topPs: Tensor, stepCounter: Tensor, maxEffectiveK: number): void {
    const pLogits = logits as ParallelTensor;
    if (pLogits.parallelism === TensorParallelism.Row || pLogits.parallelism === TensorParallelism.Column) {
      const gathered = pLogits.allGather(pLogits.workspace);
      this.doSampleBatch(outTokens, topkVals, topkIdxs, workspace, gathered, penaltyTokens, penaltyCount, maxWindow, vocabSize, batchSize, temperatures, repPenalties, presPenalties, topKs, topPs, stepCounter, maxEffectiveK);
      gathered[Symbol.dispose]();
      return;
    }
    if (pLogits.parallelism === TensorParallelism.PartialSum) {
      pLogits.allReduce();
      this.doSampleBatch(outTokens, topkVals, topkIdxs, workspace, logits, penaltyTokens, penaltyCount, maxWindow, vocabSize, batchSize, temperatures, repPenalties, presPenalties, topKs, topPs, stepCounter, maxEffectiveK);
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
      this.devices[i].sampleBatch(pOut.shards[i], pTopkVals.shards[i], pTopkIdxs.shards[i], pWorkspace.shards[i], pLogits.shards[i], pPenaltyTokens.shards[i], pPenaltyCount.shards[i], maxWindow, vocabSize, batchSize, pTemps.shards[i], pRepPen.shards[i], pPresPen.shards[i], pTopKs.shards[i], pTopPs.shards[i], pStepCounter.shards[i], maxEffectiveK);
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
  readonly maxBytes: number;
  readonly worldSize: number;

  constructor(devices: readonly GlmOps[], maxBytes: number) {
    this.worldSize = devices.length;
    this.maxBytes = maxBytes;
    const N = devices.length;
    const addon = getNativeAddon();

    // 1. Enable peer access in both directions for every pair.
    for (let i = 0; i < N; ++i) {
      for (let j = 0; j < N; ++j) {
        if (i === j) continue;
        const rc = addon.p2pEnablePeerAccess(devices[i].ctx, devices[j].device);
        if (rc !== 0) {
          throw new Error(`P2P enable failed dev ${devices[i].device} -> ${devices[j].device}`);
        }
      }
    }

    // 2. Create one instance per rank.
    this.instances = devices.map((dev, rank) =>
      addon.p2pCreateInstance(dev.ctx, rank, N, maxBytes));

    // 3. Collect per-rank data + flag pointers.
    const dataPtrs = this.instances.map(inst => addon.p2pGetDataPtr(inst));
    const flagPtrs = this.instances.map(inst => addon.p2pGetFlagPtr(inst));

    // 4. Tell each rank about all peers' pointers.
    for (let i = 0; i < N; ++i) {
      addon.p2pSetPeers(devices[i].ctx, this.instances[i], dataPtrs, flagPtrs);
    }
  }

  free(): void {
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
  /** Lazy-initialized custom one-shot AllReduce group for small messages. */
  private p2pGroup: P2PAllReduceGroup | null = null;
  /** Max BF16 elements per shard for which P2P AllReduce is used. */
  private readonly p2pMaxElems: number;
  private readonly p2pEnabled: boolean;

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
      for (const device of devices) {
        device.synchronize();
      }
    } else {
      this.comms = [];
    }
    // P2P AllReduce: opt-out via env var. Block-size 1024 * vec 8 = 8192
    // BF16 elements max per call. We allocate 16 KB per rank's data buf.
    this.p2pEnabled = process.env.GLM_DISABLE_P2P_ALLREDUCE !== "1" && devices.length > 1;
    this.p2pMaxElems = 8192;  // matches kernel block_size * vec
  }

  /** Get (and lazily create) the P2P AllReduce group sized for small messages. */
  private getP2PGroup(): P2PAllReduceGroup | null {
    if (!this.p2pEnabled) return null;
    if (this.p2pGroup === null) {
      // 16 KB per rank covers BF16 [hidden=8192] or F32 [hidden=4096].
      this.p2pGroup = new P2PAllReduceGroup(this.devices, 16 * 1024);
    }
    return this.p2pGroup;
  }

  /**
   * Try to AllReduce via the custom P2P kernel. Returns true on success
   * (caller must skip the NCCL fallback). Returns false if the message is
   * too large for the P2P group, in which case the caller should NCCL.
   */
  private tryP2PAllReduce(shards: readonly Tensor[], count: number, dtype: number): boolean {
    if (!this.p2pEnabled) return false;
    if (dtype !== NCCL_BFLOAT16 && dtype !== NCCL_FLOAT32) return false;
    if (count > this.p2pMaxElems) return false;
    const group = this.getP2PGroup();
    if (!group) return false;
    const addon = getNativeAddon();
    for (let i = 0; i < this.worldSize; ++i) {
      addon.p2pAllReduce(this.devices[i].ctx, group.instances[i],
                         shards[i].data, shards[i].data, count, dtype);
    }
    return true;
  }

  /** Public wrapper used by ParallelTensor.allReduce. */
  doAllReduce(shards: readonly Tensor[], count: number, dtype: number): void {
    if (this.tryP2PAllReduce(shards, count, dtype)) return;
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

  free(): void {
    if (this.p2pGroup !== null) {
      this.p2pGroup.free();
      this.p2pGroup = null;
    }
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
          this.devices[i].availableStreams.push(streams[i]!);
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

  mlaPrefillPlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, qoIndptrH: Tensor, kvIndptrH: Tensor, kvLenH: Tensor, batchSize: number, numHeads: number, headDimO: number, causal: boolean): void {
    const pFloatWs = this.cast(floatWs);
    const pIntWs = this.cast(intWs);
    const pPinnedIntWs = this.cast(pinnedIntWs);
    const pPlanInfo = this.cast(planInfo);
    const pQoIndptrH = this.cast(qoIndptrH);
    const pKvIndptrH = this.cast(kvIndptrH);
    const pKvLenH = this.cast(kvLenH);
    const shardNumHeads = this.shardDim(numHeads, "mlaPrefillPlan numHeads");
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].mlaPrefillPlan(pFloatWs.shards[i], floatWsSize, pIntWs.shards[i], pPinnedIntWs.shards[i], intWsSize, pPlanInfo.shards[i], pQoIndptrH.shards[i], pKvIndptrH.shards[i], pKvLenH.shards[i], batchSize, shardNumHeads, headDimO, causal);
    }
  }

  mlaPrefillRun(qNope: Tensor, qPe: Tensor, ckvData: Tensor, kpeData: Tensor, kvIndices: Tensor, o: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, numHeads: number, pageSize: number, maskMode: number, smScale: number, qNopeStrideN: number, qNopeStrideH: number, qPeStrideN: number, qPeStrideH: number, ckvStridePage: number, ckvStrideN: number, kpeStridePage: number, kpeStrideN: number, oStrideN: number, oStrideH: number, headDimCkv: number, headDimKpe: number): void {
    const pQNope = this.cast(qNope);
    const pQPe = this.cast(qPe);
    const pCkvData = this.cast(ckvData);
    const pKpeData = this.cast(kpeData);
    const pKvIndices = this.cast(kvIndices);
    const pO = this.cast(o);
    const pFloatWs = this.cast(floatWs);
    const pIntWs = this.cast(intWs);
    const pPlanInfo = this.cast(planInfo);
    const shardNumHeads = this.shardDim(numHeads, "mlaPrefillRun numHeads");
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].mlaPrefillRun(pQNope.shards[i], pQPe.shards[i], pCkvData.shards[i], pKpeData.shards[i], pKvIndices.shards[i], pO.shards[i], pFloatWs.shards[i], pIntWs.shards[i], pPlanInfo.shards[i], shardNumHeads, pageSize, maskMode, smScale, qNopeStrideN, qNopeStrideH, qPeStrideN, qPeStrideH, ckvStridePage, ckvStrideN, kpeStridePage, kpeStrideN, oStrideN, oStrideH, headDimCkv, headDimKpe);
    }
  }

  mlaDecodePlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, indptrH: Tensor, batchSize: number, numQoHeads: number, pageSize: number, enableCudaGraph: boolean, headDimCkv: number, headDimKpe: number): void {
    const pFloatWs = this.cast(floatWs);
    const pIntWs = this.cast(intWs);
    const pPinnedIntWs = this.cast(pinnedIntWs);
    const pPlanInfo = this.cast(planInfo);
    const pIndptrH = this.cast(indptrH);
    const shardNumQoHeads = this.shardDim(numQoHeads, "mlaDecodePlan numQoHeads");
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].mlaDecodePlan(pFloatWs.shards[i], floatWsSize, pIntWs.shards[i], pPinnedIntWs.shards[i], intWsSize, pPlanInfo.shards[i], pIndptrH.shards[i], batchSize, shardNumQoHeads, pageSize, enableCudaGraph, headDimCkv, headDimKpe);
    }
  }

  mlaDecodeRun(qNope: Tensor, qPe: Tensor, ckvData: Tensor, kpeData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, o: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, batchSize: number, numQoHeads: number, pageSize: number, smScale: number, headDimCkv: number, headDimKpe: number): void {
    const pQNope = this.cast(qNope);
    const pQPe = this.cast(qPe);
    const pCkvData = this.cast(ckvData);
    const pKpeData = this.cast(kpeData);
    const pIndices = this.cast(indices);
    const pIndptrD = this.cast(indptrD);
    const pLastPageLen = this.cast(lastPageLen);
    const pO = this.cast(o);
    const pFloatWs = this.cast(floatWs);
    const pIntWs = this.cast(intWs);
    const pPlanInfo = this.cast(planInfo);
    const shardNumQoHeads = this.shardDim(numQoHeads, "mlaDecodeRun numQoHeads");
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].mlaDecodeRun(pQNope.shards[i], pQPe.shards[i], pCkvData.shards[i], pKpeData.shards[i], pIndices.shards[i], pIndptrD.shards[i], pLastPageLen.shards[i], pO.shards[i], pFloatWs.shards[i], pIntWs.shards[i], pPlanInfo.shards[i], batchSize, shardNumQoHeads, pageSize, smScale, headDimCkv, headDimKpe);
    }
  }

  ropeTranspose(ctx: number, out: number, input: number, cos: number, sin: number, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, inStride: number): void {
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].ropeTranspose(ctx, out, input, cos, sin, ropeDim, headDim, nHeads, seqLen, batch, inStride);
    }
  }

  mlaVExpand(ctx: number, result: number, attnOut: number, vProj: number, kvLoraRank: number, vHeadDim: number, nHeads: number, seqLen: number, batch: number): void {
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].mlaVExpand(ctx, result, attnOut, vProj, kvLoraRank, vHeadDim, nHeads, seqLen, batch);
    }
  }

  sigmoid(ctx: number, out: number, input: number, n: number): void {
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].sigmoid(ctx, out, input, n);
    }
  }

  topk(ctx: number, outValues: number, outIndices: number, input: number, k: number, dim: number, batch: number): void {
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].topk(ctx, outValues, outIndices, input, k, dim, batch);
    }
  }

  indexAdd(ctx: number, out: number, indices: number, values: number, nIndices: number, dim: number): void {
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].indexAdd(ctx, out, indices, values, nIndices, dim);
    }
  }

  add(ctx: number, out: number, a: number, b: number, n: number): void {
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].add(ctx, out, a, b, n);
    }
  }

  addBroadcast(ctx: number, out: number, a: number, b: number, dim: number, rows: number): void {
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].addBroadcast(ctx, out, a, b, dim, rows);
    }
  }

  scale(ctx: number, out: number, input: number, scale: number, n: number): void {
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].scale(ctx, out, input, scale, n);
    }
  }

  mul(ctx: number, out: number, a: number, b: number, n: number): void {
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].mul(ctx, out, a, b, n);
    }
  }

  mulBroadcast(ctx: number, out: number, a: number, b: number, dim: number, rows: number): void {
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].mulBroadcast(ctx, out, a, b, dim, rows);
    }
  }

  scatterScalar(ctx: number, out: number, indices: number, value: number, k: number, outDim: number, batch: number): void {
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].scatterScalar(ctx, out, indices, value, k, outDim, batch);
    }
  }

  maskedFill(ctx: number, out: number, input: number, mask: number, value: number, n: number): void {
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].maskedFill(ctx, out, input, mask, value, n);
    }
  }

  applyRotaryPosEmbPartial(ctx: number, out: number, input: number, cos: number, sin: number, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, unsqueezeDim: number): void {
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].applyRotaryPosEmbPartial(ctx, out, input, cos, sin, ropeDim, headDim, nHeads, seqLen, batch, unsqueezeDim);
    }
  }

  rowScaleAdd(ctx: number, out: number, input: number, scales: number, rows: number, dim: number): void {
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].rowScaleAdd(ctx, out, input, scales, rows, dim);
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
