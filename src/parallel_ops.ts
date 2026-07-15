import { DeviceOps, MaskMode, StridedMmap, TensorParallelism } from "./device_ops";
import { ExecutionState } from "./execution-workspace";
import { bf16BytesToF32, f32ToBf16Bytes, getNativeAddon, GlmOps, GlmTensor, NCCL_BFLOAT16, NCCL_FLOAT32, NCCL_INT32, NCCL_SUM, NCCL_UINT8 } from "./glm_ops";
import { MemcpyKind, Tensor } from "./tensor";
import { UsingHolder } from "./using-holder";
import { WorkspaceBase } from "./workspace";

// Master switch for the CP "gather CKV" path in sparse MLA prefill. When true,
// CP prefill gathers the CKV cache into a flat Replicated buffer and the indexer
// emits flat slots to match. When false, both are disabled: Q is AllGathered and
// the indexer/sparse kernel run against the paged cache (the pre-gather path).
// Toggle here or via GLM_CP_GATHER_KV=0. Both the gather (sparseMlaPrefill) and
// the indexer flat-slot mode (topkToSlots ignores kvTokenIndptr when off)
// read this so they never diverge.
export const CP_GATHER_KV = process.env.GLM_CP_GATHER_KV !== "0";

export class ParallelTensor extends Tensor {
  parallelism: TensorParallelism;
  readonly shards: readonly Tensor[];
  private readonly devices: readonly GlmOps[];
  private readonly parallelOps: ParallelOps;

  constructor(
    workspace: WorkspaceBase,
    parallelOps: ParallelOps,
    parallelism: TensorParallelism,
    shards: readonly Tensor[],
    shape: number[],
    type: string,
    name: string | undefined,
    pinned: boolean,
    view: ParallelTensor | undefined,
  ) {
    super(workspace, 0, 0, shape, type, name, pinned, view);
    this.parallelOps = parallelOps;
    this.devices = parallelOps.devices;
    this.parallelism = parallelism;
    this.shards = shards;
    for (let i = 0; i < shards.length; i++) {
      if (shards[i].name !== name) {
        throw new Error(`Shard ${i} has name ${shards[i].name}, expected ${name}`);
      }
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

  capture() {
    this.removeTracking();
    // Create captured shard orphans directly via wrapTensor rather than calling
    // s.capture() on each shard. Using s.capture() would call removeTracking() on
    // the original shard, moving it into the shard workspace's exported set. Shard
    // workspaces never have startTracking() called on them, so shards stuck in
    // exported can never be disposed or recycled — leaking GPU memory every call.
    // Instead, wrap the same GPU pointer in a new captured (immutable, orphan)
    // tensor without touching the original shard's tracking state. The original
    // shards stay in sw.tracked, so when this ParallelTensor is later disposed by
    // startTracking() cleaning up main ws.exported, each shard's canDispose()
    // returns true and its GPU memory enters the shard workspace's disposed pool
    // for recycling. The captured shards are pure orphans (not in any workspace
    // set) held solely by the returned captured ParallelTensor / CaptureManager.
    const capturedShards = this.shards.map(s => {
      const captured = s.workspace.glm.wrapTensor(
        s.workspace, s.data, s.allocSize, s.shape, s.type, s.pinned, undefined);
      (captured as { name: string | undefined }).name = s.name;
      captured.captured = true;
      return captured;
    });
    const captured = new ParallelTensor(this.workspace, this.parallelOps, this.parallelism, capturedShards, this.shape, this.type, undefined, this.pinned, undefined);
    (captured as { name: string | undefined }).name = this.name;
    captured.captured = true;
    return captured;
  }

  [Symbol.dispose](): void {
    if (!this.canDispose()) {
      return;
    }
    for (const shard of this.shards) {
      shard[Symbol.dispose]();
    }
    (this.shards as Tensor[]).length = 0;
    super[Symbol.dispose]();
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
    return this.parallelOps.wrapShards(
      this.workspace, reshapedShards, newShape, this.type, newPar, this,
    );
  }

  override viewClone(): Tensor {
    const clonedShards: Tensor[] = this.shards.map(s => s.viewClone());
    return this.parallelOps.wrapShards(
      this.workspace, clonedShards, this.shape, this.type, this.parallelism, this,
    );
  }

  allReduce(): void {
    if (this.parallelism !== TensorParallelism.PartialSum) {
      throw new Error(`allReduce requires PartialSum tensor, got ${this.parallelism}`);
    }
    if (!this.tryP2PAllReduce()) {
      const count = this.shards[0].shape.reduce((a, b) => a * b, 1);
      const dtype = this.parallelOps.ncclDatatype(this.type);
      const addon = getNativeAddon();
      addon.ncclGroupStart();
      for (let i = 0; i < this.worldSize; ++i) {
        addon.ncclAllReduce(
          this.parallelOps.comms[i], this.devices[i].ctx,
          this.shards[i].data, this.shards[i].data,
          count, dtype, NCCL_SUM,
        );
      }
      addon.ncclGroupEnd();
    }
    this.parallelism = TensorParallelism.Replicated;
  }

  /**
   * Try to AllReduce via the custom P2P kernel. Returns true on success
   * (caller must skip the NCCL fallback). Returns false if the message is
   * too large for the P2P group, in which case the caller should NCCL.
   */
  private tryP2PAllReduce(): boolean {
    if (!this.parallelOps.p2pEnabled)
      return false;
    if (this.type !== "BF16" && this.type !== "F32")
      return false;
    const group = this.parallelOps.getP2PGroup(this.shards[0].workspace.glm.currentStream);
    if (!group)
      return false;

    const count = this.shards[0].shape.reduce((a, b) => a * b, 1);

    // all to all reduce
    if (false) {
      if (count > 65536 * 2)
        return false;

      const gatheredShards: Tensor[] = [];
      for (const shard of this.shards) {
        const outerShape = shard.shape[0];
        const gatheredShape = [outerShape * this.worldSize, ...shard.shape.slice(1)];
        const gathered = shard.workspace.alloc(gatheredShape, shard.type);
        gatheredShards.push(gathered);
      }

      const addon = getNativeAddon();
      for (let i = 0; i < this.worldSize; i++) {
        const shard = this.shards[i];
        const outer = shard.shape[0];
        const gatheredNarrows = gatheredShards.map(g => g.narrow(i * outer, outer));
        const ptrs = new Array<number>(8).fill(0);
        for (let k = 0; k < this.worldSize; k++) {
          ptrs[k] = gatheredNarrows[(i + k) % this.worldSize].data;
        }
        addon.memcpyMulti(this.devices[i].ctx, shard.data,
          ptrs[0], ptrs[1], ptrs[2], ptrs[3],
          ptrs[4], ptrs[5], ptrs[6], ptrs[7],
          this.worldSize, shard.numElements, shard.type === "F32" ? 7 : 9);
        for (const n of gatheredNarrows) {
          n[Symbol.dispose]();
        }
      }

      this.parallelOps.p2pBarrier();
      this.parallelOps.sourceCleanup();

      for (let i = 0; i < this.worldSize; i++) {
        const gathered = gatheredShards[i];
        const outer = this.shards[i].shape[0];
        const shard = this.shards[i] as GlmTensor;
        const numel = shard.numElements;
        const dtype = shard.type === "F32" ? 7 : 9;
        const gatheredNarrows = new Array<Tensor>(this.worldSize);
        const ptrs = new Array<number>(8).fill(0);
        for (let k = 0; k < this.worldSize; k++) {
          gatheredNarrows[k] = gathered.narrow(k * outer, outer);
          ptrs[k] = (gatheredNarrows[k] as GlmTensor).data;
        }
        addon.sumPointersDirect(
          this.devices[i].ctx,
          ptrs[0], ptrs[1], ptrs[2], ptrs[3],
          ptrs[4], ptrs[5], ptrs[6], ptrs[7],
          shard.data, this.worldSize, numel, dtype,
        );
        for (const gatheredNarrow of gatheredNarrows) {
          gatheredNarrow[Symbol.dispose]();
        }
        gathered[Symbol.dispose]();
      }

      return true;
    }

    // row reduce + scatter
    if (true) {
      if (count > 65536 * 2)
        return false;

      this.parallelOps.p2pBarrier();
      this.parallelOps.sourceCleanup();

      if (count % this.worldSize !== 0)
        return false;
      const chunkLen = count / this.worldSize;
      const flatShards = this.shards.map(s => s.reshape([count]));
      this.parallelOps.p2pSources.push(...flatShards);
      for (let i = 0; i < this.worldSize; i++) {
        const rowShards: Tensor[] = new Array(this.worldSize);
        // rank i starts with its own shard (j = i) and walks outward,
        // instead of every rank hitting flatShards[0] first
        for (let k = 0; k < this.worldSize; k++) {
          const j = (i + k) % this.worldSize;
          const v = flatShards[j].narrow(i * chunkLen, chunkLen);
          rowShards[k] = v;
          this.parallelOps.p2pSources.push(v);
        }
        rowShards[0].sumInPlace(rowShards, true);
      }
      this.parallelOps.p2pBarrier();

      return true;
    }

    // butterfly reduce
    if (true) {
      if (count > 65536 * 2)
        return false;

      let current: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        const copy = this.shards[i].workspace.alloc(this.shards[i].shape, this.shards[i].type);
        copy.memcpy(this.shards[i]);
        current.push(copy);
      }

      for (let reduceHalf = this.worldSize / 2; reduceHalf >= 1; reduceHalf /= 2) {
        const peerRanks = Array.from({ length: this.worldSize }, (_, i) => i ^ reduceHalf);
        const isLast = reduceHalf === 1;

        this.parallelOps.p2pBarrier(peerRanks);
        if (reduceHalf === this.worldSize / 2) {
          this.parallelOps.sourceCleanup();
        }
        this.parallelOps.p2pSources.push(...current);

        if (isLast) {
          for (let i = 0; i < this.worldSize; i++) {
            const peer = i ^ reduceHalf;
            this.shards[i].sumInPlace([current[i], current[peer]]);
          }
        } else {
          const output: Tensor[] = new Array(this.worldSize);
          for (let i = 0; i < this.worldSize; i++) {
            const peer = i ^ reduceHalf;
            output[i] = current[i].sum([current[peer]]);
          }
          current = output;
        }
      }

      return true;
    }

    return false;
  }

  allGather(workspace: WorkspaceBase): ParallelTensor {
    if (this.parallelism === TensorParallelism.Replicated) {
      return this;
    }
    if (this.parallelism === TensorParallelism.PartialSum) {
      throw new Error("allGather cannot be used on PartialSum tensors; use allReduce instead");
    }
    const output = this.workspace.alloc(this.shape, this.type, undefined, TensorParallelism.Replicated) as ParallelTensor;

    if (this.tryP2PAllGather(output)) {
      return output;
    }

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
      const outer = this.shape[0];
      const inner = this.shape.slice(2).reduce((a, b) => a * b, 1);
      const shardDim1 = this.shape[1] / this.devices.length;
      const shardBytes = count * eb;
      const totalElems = this.shape.reduce((a, b) => a * b, 1);
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
            this.shape[1] * inner * eb,
            tempTensors[i], r * shardBytes,
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

  /**
   * Try to AllGather via the custom P2P kernel. Returns true on success
   * (caller must skip the NCCL fallback). Returns false if the shard is
   * too large for the P2P group, in which case the caller should use NCCL.
   * Dtype-agnostic: copies raw bytes, supports all element types.
   */
  private tryP2PAllGather(output: ParallelTensor): boolean {
    if (!this.parallelOps.p2pEnabled)
      return false;
    const count = this.shards[0].shape.reduce((a, b) => a * b, 1);
    if (count > 65536 * 2)
      return false;
    const elemBytes = ParallelTensor.elemBytes(this.type);
    const shardBytes = count * elemBytes;
    const group = this.parallelOps.getP2PGroup(this.shards[0].workspace.glm.currentStream);
    if (!group)
      return false;

    const addon = getNativeAddon();
    const ptrs = new Array<number>(8).fill(0);
    for (let i = 0; i < this.worldSize; i++) ptrs[i] = this.shards[i].data;

    if (this.parallelism === TensorParallelism.Column) {
      // Smem-staged AllGather: barrier, then single kernel reads from all peers via TMA.
      this.parallelOps.p2pBarrier();
      this.parallelOps.sourceCleanup();
      this.parallelOps.p2pSources.push(...this.shards.map(s => s.viewClone()));
      for (let i = 0; i < this.worldSize; ++i) {
        addon.p2pAllGatherSmem(
          this.devices[i].ctx,
          ptrs[0], ptrs[1], ptrs[2], ptrs[3],
          ptrs[4], ptrs[5], ptrs[6], ptrs[7],
          output.shards[i].data, this.worldSize, shardBytes, i,
        );
      }
      return true;
    }

    if (this.parallelism === TensorParallelism.Row) {
      const outer = this.shape[0];
      const inner = this.shape.slice(2).reduce((a, b) => a * b, 1);
      const shardDim1 = this.shape[1] / this.worldSize;
      const shardDim1Bytes = shardDim1 * inner * elemBytes;
      const fullDim1Bytes = this.shape[1] * inner * elemBytes;
      // Smem-staged Row AllGather: barrier, then single kernel reads from all peers via TMA.
      this.parallelOps.p2pBarrier();
      this.parallelOps.sourceCleanup();
      this.parallelOps.p2pSources.push(...this.shards.map(s => s.viewClone()));
      for (let i = 0; i < this.worldSize; ++i) {
        addon.p2pAllGatherRowSmem(
          this.devices[i].ctx,
          ptrs[0], ptrs[1], ptrs[2], ptrs[3],
          ptrs[4], ptrs[5], ptrs[6], ptrs[7],
          output.shards[i].data, this.worldSize, shardDim1Bytes, fullDim1Bytes, outer, i,
        );
      }
      return true;
    }

    return false;
  }

  sliceToRowParallel(workspace: WorkspaceBase, shardDim1: number): ParallelTensor {
    if (this.parallelism !== TensorParallelism.Replicated) {
      throw new Error(`sliceToRowParallel: expected Replicated, got ${this.parallelism}`);
    }
    if (this.shape.length !== 2) {
      throw new Error(`sliceToRowParallel: expected 2D tensor, got ${this.shape.length}D`);
    }
    const outer = this.shape[0];
    const dim1 = this.shape[1];
    const elemBytes = ParallelTensor.elemBytes(this.type);
    const shardWss = this.parallelOps.getShardWorkspaces(workspace);
    const shardShape = [outer, shardDim1];
    const outputShards: Tensor[] = [];
    for (let i = 0; i < this.devices.length; i++) {
      const shard = shardWss[i].alloc(shardShape, this.type);
      shard.memcpy2d(
        0, shardDim1 * elemBytes,
        this.shards[i], i * shardDim1 * elemBytes,
        dim1 * elemBytes,
        shardDim1 * elemBytes, outer,
        MemcpyKind.DeviceToDevice,
      );
      outputShards.push(shard);
    }
    return this.parallelOps.wrapShards(workspace, outputShards, this.shape, this.type, TensorParallelism.Row);
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
      const totalElems = ParallelTensor.shapeElems(this.shape);
      const shardBytes = (totalElems / this.worldSize) * eb;
      for (let i = 0; i < this.worldSize; i++) {
        this.shards[i].h2d(data.subarray(i * shardBytes, (i + 1) * shardBytes));
      }
      return;
    }

    if (this.parallelism === TensorParallelism.Row) {
      const outer = this.shape[0];
      const inner = ParallelTensor.shapeElems(this.shape.slice(2));
      const shardDim1 = this.shape[1] / this.worldSize;
      const fullStride = this.shape[1] * inner * eb;
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
      const totalElems = ParallelTensor.shapeElems(this.shape);
      const shardBytes = (totalElems / this.worldSize) * eb;
      for (let i = 0; i < this.worldSize; i++) {
        this.shards[i].d2h(buf.subarray(i * shardBytes, (i + 1) * shardBytes));
      }
      return;
    }

    if (this.parallelism === TensorParallelism.Row) {
      const outer = this.shape[0];
      const inner = ParallelTensor.shapeElems(this.shape.slice(2));
      const shardDim1 = this.shape[1] / this.worldSize;
      const fullStride = this.shape[1] * inner * eb;
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
      const totalElems = ParallelTensor.shapeElems(this.shape);
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

  override linear(weight: Tensor): Tensor {
    super.linear(weight);
    const pWeight = weight as ParallelTensor;
    const batch = this.shape[0];
    const n = weight.shape[0];
    const WP = pWeight.parallelism;
    const XP = this.parallelism;

    // --- Direct paths (no communication) ---

    // Column weight + Replicated input → Row output
    if (WP === TensorParallelism.Column && XP === TensorParallelism.Replicated) {
      const shards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        shards.push(this.shards[i].linear(pWeight.shards[i]));
      }
      return this.parallelOps.wrapShards(this.workspace, shards, [batch, n], this.type, TensorParallelism.Row);
    }

    // Row weight + Row input → PartialSum output
    if (WP === TensorParallelism.Row && XP === TensorParallelism.Row) {
      const shards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        shards.push(this.shards[i].linear(pWeight.shards[i]));
      }
      return this.parallelOps.wrapShards(this.workspace, shards, [batch, n], this.type, TensorParallelism.PartialSum);
    }

    // Replicated weight + Replicated input → Replicated output
    if (WP === TensorParallelism.Replicated && XP === TensorParallelism.Replicated) {
      const shards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        shards.push(this.shards[i].linear(pWeight.shards[i]));
      }
      return this.parallelOps.wrapShards(this.workspace, shards, [batch, n], this.type, TensorParallelism.Replicated);
    }

    // Replicated weight + Column input → Column output
    if (WP === TensorParallelism.Replicated && XP === TensorParallelism.Column) {
      const shards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        shards.push(this.shards[i].linear(pWeight.shards[i]));
      }
      return this.parallelOps.wrapShards(this.workspace, shards, [batch, n], this.type, TensorParallelism.Column);
    }

    // Replicated weight + PartialSum input → PartialSum output (distributivity: Σ(P_i @ W) = (Σ P_i) @ W)
    if (WP === TensorParallelism.Replicated && XP === TensorParallelism.PartialSum) {
      const shards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        shards.push(this.shards[i].linear(pWeight.shards[i]));
      }
      return this.parallelOps.wrapShards(this.workspace, shards, [batch, n], this.type, TensorParallelism.PartialSum);
    }

    // PartialSum weight + Replicated input → PartialSum output (distributivity: X @ (Σ W_i) = Σ(X @ W_i))
    if (WP === TensorParallelism.PartialSum && XP === TensorParallelism.Replicated) {
      const shards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        shards.push(this.shards[i].linear(pWeight.shards[i]));
      }
      return this.parallelOps.wrapShards(this.workspace, shards, [batch, n], this.type, TensorParallelism.PartialSum);
    }

    // --- Fallback paths (need communication) ---

    // PartialSum input: allReduce to Replicated, then retry (hits Replicated+Column or Replicated+Replicated)
    if (XP === TensorParallelism.PartialSum) {
      this.allReduce();
      return this.linear(weight);
    }

    // PartialSum weight with non-Replicated input: can't allReduce a persistent weight
    if (WP === TensorParallelism.PartialSum) {
      throw new Error(`linear: PartialSum weight requires Replicated input (got ${XP}); allReduce would mutate persistent weight`);
    }

    // Row weight: K dimension mismatch, allGather weight to Replicated
    if (WP === TensorParallelism.Row) {
      using gathered = pWeight.allGather(this.workspace);
      return this.linear(gathered);
    }

    // Row input: K dimension mismatch, allGather input to Replicated
    if (XP === TensorParallelism.Row) {
      using gathered = this.allGather(this.workspace);
      return gathered.linear(weight);
    }

    // Column+Column: allGather input to Replicated, then Column+Replicated → Row
    if (XP === TensorParallelism.Column && WP === TensorParallelism.Column) {
      using gathered = this.allGather(this.workspace);
      return gathered.linear(weight);
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

  // Transpose 4D tensor [d0,d1,d2,d3] by permutation [p0,p1,p2,p3].
  // For Column parallelism d1 is the sharded dimension, so each shard uses d1/worldSize.
  transpose4d(d0: number, d1: number, d2: number, d3: number, p0: number, p1: number, p2: number, p3: number): Tensor {
    const shardD1 = this.parallelism === TensorParallelism.Column ? d1 / this.worldSize : d1;
    const outShards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      outShards.push((this.shards[i] as GlmTensor).transpose4d(d0, shardD1, d2, d3, p0, p1, p2, p3));
    }
    const dims = [d0, d1, d2, d3];
    const outDims = [p0, p1, p2, p3].map(p => dims[p]);
    return this.parallelOps.wrapShards(this.workspace, outShards, [outDims[0] * outDims[1], outDims[2] * outDims[3]], this.type, this.parallelism);
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

  rmsnorm(weight: Tensor, eps: number): Tensor {
    super.rmsnorm(weight, eps);
    const batch = this.shape[0];
    const dim = this.shape[1];

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
      const result = gathered.rmsnorm(weight, eps);
      return result;
    }

    if (this.parallelism === TensorParallelism.PartialSum) {
      this.allReduce();
      return this.rmsnorm(weight, eps);
    }

    const pWeight = weight as ParallelTensor;
    this.assertParallel("rmsnorm input", this, TensorParallelism.Replicated);
    this.assertParallel("rmsnorm weight", pWeight, TensorParallelism.Replicated);

    const shards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      shards.push(this.shards[i].rmsnorm(pWeight.shards[i], eps));
    }
    return this.parallelOps.wrapShards(this.workspace, shards, [batch, dim], this.type, TensorParallelism.Replicated);
  }

  layernorm(weight: Tensor, bias: Tensor, eps: number): Tensor {
    super.layernorm(weight, bias, eps);
    const pWeight = weight as ParallelTensor;
    const pBias = bias as ParallelTensor;
    const shards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      shards.push(this.shards[i].layernorm(pWeight.shards[i], pBias.shards[i], eps));
    }
    return this.parallelOps.wrapShards(this.workspace, shards, this.shape, this.type, TensorParallelism.Replicated);
  }

  fusedAddRmsnorm(input: Tensor, weight: Tensor, eps: number): { normed: Tensor, residual: Tensor } {
    super.fusedAddRmsnorm(input, weight, eps);
    const pInput = input as ParallelTensor;

    if (this.parallelism === 'row' && input.parallelism === 'partial_sum') {
      const eb = 2;
      const rows = this.shape[0];
      const fullDim = this.shape[1];
      const shardDim = fullDim / this.worldSize;
      const shardWss = this.parallelOps.getShardWorkspaces(this.workspace);

      const residualShards = shardWss.map((ws, i) => {
        using tempTensor = ws.alloc([rows, fullDim], input.type);
        tempTensor.fill(0, tempTensor.numElements);
        tempTensor.memcpy2d(
          i * shardDim * eb,
          fullDim * eb,
          this.shards[i], 0,
          shardDim * eb,
          shardDim * eb,
          rows,
          MemcpyKind.DeviceToDevice,
        );
        return tempTensor.add(pInput.shards[i]);
      });

      const residual = this.parallelOps.wrapShards(this.workspace, residualShards, input.shape, this.type, TensorParallelism.PartialSum);
      residual.allReduce();

      return {
        normed: residual.rmsnorm(weight, eps),
        residual,
      }
    }

    if (this.parallelism === TensorParallelism.PartialSum) {
      this.allReduce();
      return this.fusedAddRmsnorm(input, weight, eps);
    }

    if (this.parallelism === TensorParallelism.Row || this.parallelism === TensorParallelism.Column) {
      using gathered = this.allGather(this.workspace);
      return gathered.fusedAddRmsnorm(input, weight, eps);
    }

    if (this.parallelism === TensorParallelism.Replicated && input.parallelism === TensorParallelism.PartialSum) {
      // can be used here?
    }

    if (pInput.parallelism === TensorParallelism.PartialSum) {
      pInput.allReduce();
      return this.fusedAddRmsnorm(pInput, weight, eps);
    }

    if (pInput.parallelism === TensorParallelism.Row || pInput.parallelism === TensorParallelism.Column) {
      using gathered = pInput.allGather(pInput.workspace);
      return this.fusedAddRmsnorm(gathered, weight, eps);
    }

    const pWeight = weight as ParallelTensor;
    this.assertParallel("fusedAddRmsnorm inputA", this, TensorParallelism.Replicated);
    this.assertParallel("fusedAddRmsnorm inputB", pInput, TensorParallelism.Replicated);
    this.assertParallel("fusedAddRmsnorm weight", pWeight, TensorParallelism.Replicated);

    const normedShards: Tensor[] = [];
    const residualShards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      const result = this.shards[i].fusedAddRmsnorm(pInput.shards[i], pWeight.shards[i], eps);
      normedShards.push(result.normed);
      residualShards.push(result.residual);
    }
    const normed = this.parallelOps.wrapShards(this.workspace, normedShards, this.shape, this.type, TensorParallelism.Replicated);
    const residual = this.parallelOps.wrapShards(this.workspace, residualShards, this.shape, this.type, TensorParallelism.Replicated);
    return { normed, residual };
  }

  override fusedNormRope(weight: Tensor, cos: Tensor, sin: Tensor, eps: number, ropeDim: number, seqLen: number, batch: number, inStride?: number, interleaved?: boolean): Tensor {
    super.fusedNormRope(weight, cos, sin, eps, ropeDim, seqLen, batch, inStride, interleaved);
    const headDim = weight.numElements;
    const stride = inStride ?? headDim;
    const nHeads = this.shape[1] / stride;
    if (this.parallelism === TensorParallelism.PartialSum) {
      this.allReduce();
      return this.fusedNormRope(weight, cos, sin, eps, ropeDim, seqLen, batch, inStride, interleaved);
    }

    if (this.parallelism === TensorParallelism.Column) {
      using gathered = this.allGather(this.workspace);
      return gathered.fusedNormRope(weight, cos, sin, eps, ropeDim, seqLen, batch, inStride, interleaved);
    }

    const pWeight = weight as ParallelTensor;
    const pCos = cos as ParallelTensor;
    const pSin = sin as ParallelTensor;

    if (this.parallelism === TensorParallelism.Row) {
      const shardNHeads = this.shardDim(nHeads, "fusedNormRope nHeads");
      const shardInStride = this.shape[2] === this.shape[1]
        ? stride / this.worldSize
        : stride;
      this.assertParallel("fusedNormRope weight", pWeight, TensorParallelism.Replicated);
      this.assertParallel("fusedNormRope cos", pCos, TensorParallelism.Replicated);
      this.assertParallel("fusedNormRope sin", pSin, TensorParallelism.Replicated);
      const shards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        shards.push(this.shards[i].fusedNormRope(pWeight.shards[i], pCos.shards[i], pSin.shards[i], eps, ropeDim, seqLen, batch, shardInStride, interleaved));
      }
      return this.parallelOps.wrapShards(this.workspace, shards, [batch, nHeads, seqLen, headDim], this.type, TensorParallelism.Row);
    }

    this.assertParallel("fusedNormRope input", this, TensorParallelism.Replicated);
    this.assertParallel("fusedNormRope weight", pWeight, TensorParallelism.Replicated);
    this.assertParallel("fusedNormRope cos", pCos, TensorParallelism.Replicated);
    this.assertParallel("fusedNormRope sin", pSin, TensorParallelism.Replicated);

    const shards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      shards.push(this.shards[i].fusedNormRope(pWeight.shards[i], pCos.shards[i], pSin.shards[i], eps, ropeDim, seqLen, batch, stride, interleaved));
    }
    return this.parallelOps.wrapShards(this.workspace, shards, [batch, nHeads, seqLen, headDim], this.type, TensorParallelism.Replicated);
  }

  override embedding(ids: Tensor): Tensor {
    super.embedding(ids);
    const pIds = ids as ParallelTensor;
    this.assertParallel("embedding ids", pIds, TensorParallelism.Replicated, TensorParallelism.PartialSum);
    const hidden = this.shape[1];
    const seqLen = ids.numElements;

    if (this.parallelism === TensorParallelism.Row) {
      const shardHidden = hidden / this.worldSize;
      const shards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        shards.push(this.shards[i].embedding(pIds.shards[i]));
      }
      return this.parallelOps.wrapShards(pIds.workspace, shards, [seqLen, hidden], this.type, TensorParallelism.Row);
    }

    this.assertParallel("embedding table", this, TensorParallelism.Replicated);
    const shards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      shards.push(this.shards[i].embedding(pIds.shards[i]));
    }
    return this.parallelOps.wrapShards(pIds.workspace, shards, [seqLen, hidden], this.type, TensorParallelism.Replicated);
  }

  override siluAndMul(up: Tensor): Tensor {
    super.siluAndMul(up);
    const pUp = up as ParallelTensor;
    const intermediate = this.shape[1];
    const batch = this.shape[0];

    if (this.parallelism === TensorParallelism.Column || pUp.parallelism === TensorParallelism.Column) {
      throw new Error(`siluAndMul: unsupported parallelism this=${this.parallelism}, up=${pUp.parallelism}`);
    }

    if (this.parallelism === pUp.parallelism) {
      const outPar = this.parallelism;
      const shardIntermediate = this.parallelism === TensorParallelism.Row
        ? intermediate / this.worldSize
        : intermediate;
      const shards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        shards.push(this.shards[i].siluAndMul(pUp.shards[i]));
      }
      return this.parallelOps.wrapShards(this.workspace, shards, [batch, intermediate], this.type, outPar);
    }

    if (this.parallelism === TensorParallelism.PartialSum) {
      this.allReduce();
      return this.siluAndMul(up);
    }
    if (pUp.parallelism === TensorParallelism.PartialSum) {
      pUp.allReduce();
      return this.siluAndMul(pUp);
    }

    using gatheredGate = this.allGather(this.workspace);
    using gatheredUp = pUp.allGather(pUp.workspace);
    return gatheredGate.siluAndMul(gatheredUp);
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
      const batch = this.shape[0];
      const dim = this.shape[1];
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

      const finalIndices = this.workspace.alloc([batch], "I32", undefined, TensorParallelism.Replicated) as ParallelTensor;
      const pGatheredIndices = gatheredIndices as ParallelTensor;
      const idxBytes = batch * 4;
      for (let i = 0; i < ws; i++) {
        finalIndices.shards[i].memcpy(pGatheredIndices.shards[i], idxBytes, MemcpyKind.DeviceToDevice);
      }

      return { values: rankValues, indices: finalIndices };
    }

    if (this.parallelism === TensorParallelism.Column) {
      const batch = this.shape[0];

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

  indexSelect(indices: Tensor, offset: number = 0): Tensor {
    super.indexSelect(indices, offset);
    const pIndices = indices as ParallelTensor;
    this.assertParallel("indexSelect src", this, TensorParallelism.Replicated, TensorParallelism.Row);
    this.assertParallel("indexSelect indices", pIndices, TensorParallelism.Replicated, TensorParallelism.PartialSum);

    const batch = indices.numElements;
    const dim = this.shape[1];
    const shards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      shards.push(this.shards[i].indexSelect(pIndices.shards[i], offset));
    }
    return this.parallelOps.wrapShards(this.workspace, shards, [batch, dim], this.type, this.parallelism);
  }

  rotateInputIds(qoIndptr: Tensor, newTokens: Tensor, batchSize: number): Tensor {
    super.rotateInputIds(qoIndptr, newTokens, batchSize);
    const pQoIndptr = qoIndptr as ParallelTensor;
    const pNewTokens = newTokens as ParallelTensor;
    this.assertParallel("rotateInputIds inputIds", this, TensorParallelism.Replicated);
    this.assertParallel("rotateInputIds qoIndptr", pQoIndptr, TensorParallelism.Replicated);
    this.assertParallel("rotateInputIds newTokens", pNewTokens, TensorParallelism.Replicated);

    const shards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      shards.push(this.shards[i].rotateInputIds(pQoIndptr.shards[i], pNewTokens.shards[i], batchSize));
    }
    return this.parallelOps.wrapShards(this.workspace, shards, this.shape, this.type, this.parallelism);
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

  causalConv1d(convState: Tensor, input: Tensor, weight: Tensor, cuSeqlens: Tensor, convDim: number, totalSeqLen: number, kernelSize: number, batchSize: number, convStateStride: number, chStride: number, seqStride: number): void {
    super.causalConv1d(convState, input, weight, cuSeqlens, convDim, totalSeqLen, kernelSize, batchSize, convStateStride, chStride, seqStride);
    const pConvState = this.cast(convState);
    const pInput = this.cast(input);
    const pWeight = this.cast(weight);
    const pCuSeqlens = this.cast(cuSeqlens);
    if (pInput.parallelism === TensorParallelism.Column) {
      throw new Error(`causalConv1d: unsupported input parallelism ${pInput.parallelism}`);
    }
    const isRowPar = pInput.parallelism === TensorParallelism.Row;
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
    if (pInput.parallelism === TensorParallelism.Column) {
      throw new Error(`causalConv1dUpdate: unsupported input parallelism ${pInput.parallelism}`);
    }
    const isRowPar = pInput.parallelism === TensorParallelism.Row;
    const parallelism = isRowPar ? TensorParallelism.Row : TensorParallelism.Replicated;
    const shardConvDim = isRowPar ? convDim / this.worldSize : convDim;
    const shardConvStateStride = isRowPar ? convStateStride / this.worldSize : convStateStride;
    const shardOuts: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      shardOuts.push(this.shards[i].causalConv1dUpdate(pConvState.shards[i], pInput.shards[i], pWeight.shards[i], shardConvDim, kernelSize, batchSize, shardConvStateStride));
    }
    return this.parallelOps.wrapShards(this.workspace, shardOuts, [batchSize, convDim], this.type, parallelism);
  }

  rmsnormGated(input: Tensor, gate: Tensor, weight: Tensor, eps: number): void {
    super.rmsnormGated(input, gate, weight, eps);
    const pInput = input as ParallelTensor;
    const pGate = gate as ParallelTensor;
    const pWeight = weight as ParallelTensor;
    const dim = input.shape[1];
    const batch = input.shape[0];

    if (this.parallelism === TensorParallelism.Row && pGate.parallelism === TensorParallelism.Row &&
      pInput.parallelism === TensorParallelism.Row && pWeight.parallelism === TensorParallelism.Replicated) {
      const shardBatch = batch / this.worldSize;
      for (let i = 0; i < this.worldSize; i++) {
        this.shards[i].rmsnormGated(pInput.shards[i], pGate.shards[i], pWeight.shards[i], eps);
      }
      return;
    }

    if (pInput.parallelism === TensorParallelism.Column && pGate.parallelism === TensorParallelism.Column &&
      pWeight.parallelism === TensorParallelism.Replicated &&
      (this.parallelism === TensorParallelism.Column || this.parallelism === TensorParallelism.Row)) {
      const shardBatch = batch / this.worldSize;
      for (let i = 0; i < this.worldSize; i++) {
        this.shards[i].rmsnormGated(pInput.shards[i], pGate.shards[i], pWeight.shards[i], eps);
      }
      return;
    }

    if (pInput.parallelism === TensorParallelism.Row) {
      using gathered = pInput.allGather(pInput.workspace);
      this.rmsnormGated(gathered, gate, weight, eps);
      return;
    }

    if (pInput.parallelism === TensorParallelism.PartialSum) {
      pInput.allReduce();
      this.rmsnormGated(pInput, gate, weight, eps);
      return;
    }

    this.assertParallel("rmsnormGated input", pInput, TensorParallelism.Replicated);
    this.assertParallel("rmsnormGated gate", pGate, TensorParallelism.Replicated);
    this.assertParallel("rmsnormGated weight", pWeight, TensorParallelism.Replicated);
    this.assertParallel("rmsnormGated output", this, TensorParallelism.Replicated);

    for (let i = 0; i < this.worldSize; i++) {
      this.shards[i].rmsnormGated(pInput.shards[i], pGate.shards[i], pWeight.shards[i], eps);
    }
  }

  gateSigmoidMul(gate: Tensor, numHeads: number, headDim: number): void {
    super.gateSigmoidMul(gate, numHeads, headDim);
    const pGate = gate as ParallelTensor;
    const batchSeq = this.numElements / (numHeads * headDim);

    if (this.parallelism === TensorParallelism.Column || pGate.parallelism === TensorParallelism.Column) {
      throw new Error(`gateSigmoidMul: unsupported parallelism this=${this.parallelism}, gate=${pGate.parallelism}`);
    }

    if (this.parallelism === pGate.parallelism) {
      const shardNumHeads = this.parallelism === TensorParallelism.Row
        ? this.shardDim(numHeads, "gateSigmoidMul numHeads")
        : numHeads;
      for (let i = 0; i < this.worldSize; i++) {
        this.shards[i].gateSigmoidMul(pGate.shards[i], shardNumHeads, headDim);
      }
      return;
    }

    if (this.parallelism === TensorParallelism.Row && pGate.parallelism === TensorParallelism.Replicated) {
      using gathered = this.allGather(this.workspace);
      gathered.gateSigmoidMul(gate, numHeads, headDim);
      for (let i = 0; i < this.worldSize; i++) {
        this.shards[i].memcpy((gathered as ParallelTensor).shards[i]);
      }
      return;
    }

    using gatheredThis = this.allGather(this.workspace);
    using gatheredGate = pGate.allGather(pGate.workspace);
    gatheredThis.gateSigmoidMul(gatheredGate, numHeads, headDim);
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
      const outer = this.shape[0];
      const inner = this.shape.slice(2).reduce((a, b) => a * b, 1);
      const fullDim1 = this.shape[1];
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

    // straight copy
    if (this.parallelism === src.parallelism && (size === undefined || this.parallelism === TensorParallelism.Replicated)) {
      for (let i = 0; i < this.shards.length; i++) {
        this.shards[i].memcpy(src.shards[i], size, kind);
      }
      return;
    }

    if (this.parallelism === TensorParallelism.Replicated) {
      using gathered = src.allGather(this.workspace);
      this.memcpy(gathered, size, kind);
      return;
    }

    throw new Error(`ParallelTensor.memcpy: unsupported parallelism combination dst ${this.parallelism} src ${src.parallelism}`);
  }

  memcpy2d(dstOffset: number, dpitch: number, src: Tensor, srcOffset: number, spitch: number, width: number, height: number, kind: MemcpyKind): void {
    if (this.parallelism === TensorParallelism.Replicated) {
      const pSrc = src as ParallelTensor;
      for (let i = 0; i < this.shards.length; i++) {
        this.shards[i].memcpy2d(dstOffset, dpitch, pSrc.shards[i], srcOffset, spitch, width, height, kind);
      }
      return;
    }
    const shardRowBytes = Tensor.byteCount(this.shards[0].shape.slice(1), this.shards[0].type);
    const dstPageId = Math.floor(dstOffset / dpitch);
    const srcPageId = Math.floor(srcOffset / spitch);
    const pSrc = src as ParallelTensor;
    for (let i = 0; i < this.shards.length; i++) {
      this.shards[i].memcpy2d(
        dstPageId * shardRowBytes, shardRowBytes,
        pSrc.shards[i], srcPageId * shardRowBytes,
        shardRowBytes, shardRowBytes, 1,
        kind,
      );
    }
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
    return this.parallelOps.wrapShards(this.workspace, outShards, this.shape, this.type, this.parallelism);
  }

  topk(k: number, dim: number, offset = 0): { values: Tensor, indices: Tensor } {
    if (this.parallelism === TensorParallelism.PartialSum) {
      this.allReduce();
      return this.topk(k, dim, offset);
    }
    if (this.parallelism === TensorParallelism.Row) {
      const batch = this.shape[0];
      const shardDim = this.shardDim(dim, "topk dim");
      const localValuesShards: Tensor[] = [];
      const localIndicesShards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        const shardOffset = i * shardDim + offset;
        const result = this.shards[i].topk(k, shardDim, shardOffset);
        localValuesShards.push(result.values);
        localIndicesShards.push(result.indices);
      }
      using localValues = this.parallelOps.wrapShards(this.workspace, localValuesShards, [batch, k * this.worldSize], this.type, TensorParallelism.Row);
      using localIndices = this.parallelOps.wrapShards(this.workspace, localIndicesShards, [batch, k * this.worldSize], "I32", TensorParallelism.Row);
      using gatheredValues = localValues.allGather(this.workspace);
      using gatheredIndices = localIndices.allGather(this.workspace);
      const kTotal = k * this.worldSize;
      const { values: rankValues, indices: rankIndices } = gatheredValues.topk(k, kTotal);
      using _rankIndices = rankIndices as ParallelTensor;
      const finalIndices = gatheredIndices.gather(rankIndices, k, kTotal, batch);
      return { values: rankValues, indices: finalIndices };
    }
    if (this.parallelism === TensorParallelism.Column) {
      const valuesShards: Tensor[] = [];
      const indicesShards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        const result = this.shards[i].topk(k, dim, offset);
        valuesShards.push(result.values);
        indicesShards.push(result.indices);
      }
      const values = this.parallelOps.wrapShards(this.workspace, valuesShards, [...this.shape.slice(0, -1), k], this.type, TensorParallelism.Column);
      const indices = this.parallelOps.wrapShards(this.workspace, indicesShards, [...this.shape.slice(0, -1), k], "I32", TensorParallelism.Column);
      return { values, indices };
    }
    const valuesShards: Tensor[] = [];
    const indicesShards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      const result = this.shards[i].topk(k, dim, offset);
      valuesShards.push(result.values);
      indicesShards.push(result.indices);
    }
    const values = this.parallelOps.wrapShards(this.workspace, valuesShards, [...this.shape.slice(0, -1), k], this.type, this.parallelism);
    const indices = this.parallelOps.wrapShards(this.workspace, indicesShards, [...this.shape.slice(0, -1), k], "I32", this.parallelism);
    return { values, indices };
  }

  reduceSum(): Tensor {
    const dim = this.shape[1];
    const batch = this.shape[0];
    if (this.parallelism === TensorParallelism.PartialSum) {
      throw new Error("reduceSum: unsupported input parallelism PartialSum (allReduce first)");
    }
    if (this.parallelism === TensorParallelism.Column) {
      throw new Error(`reduceSum: unsupported input parallelism ${this.parallelism}`);
    }
    const outShards: Tensor[] = [];
    const isSharded = this.parallelism === TensorParallelism.Row;
    const shardDim = isSharded ? this.parallelOps.shardDim(dim, "reduceSum dim") : dim;
    const shardBatch = batch;
    for (let i = 0; i < this.worldSize; i++) {
      outShards.push(this.shards[i].reduceSum());
    }
    const outPar = isSharded ? TensorParallelism.PartialSum : this.parallelism;
    return this.parallelOps.wrapShards(this.workspace, outShards, [batch], this.type, outPar);
  }

  rowNormalize(scale: number, normalize: boolean = true): Tensor {
    const dim = this.shape[1];
    const batch = this.shape[0];
    if (this.parallelism === TensorParallelism.PartialSum) {
      throw new Error("rowNormalize: unsupported input parallelism PartialSum (allReduce first)");
    }
    const outShards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      outShards.push(this.shards[i].rowNormalize(scale, normalize));
    }
    return this.parallelOps.wrapShards(this.workspace, outShards, this.shape, this.type, this.parallelism);
  }

  cat(tensors: Tensor[], dim: number): Tensor {
    super.cat(tensors, dim);
    const pTensors = tensors.map(t => t as ParallelTensor);
    const allPar = [this.parallelism, ...pTensors.map(t => t.parallelism)];
    const hasNonReplicated = allPar.some(p => p !== TensorParallelism.Replicated);

    // Row-parallel tensors are sharded on dim 1. Concatenating on any other dim
    // is a purely local per-shard op that preserves Row parallelism — no need to
    // allGather to Replicated (which would hand a full [BS, nHeads, ...] buffer to
    // kernels expecting a per-rank [BS, nHeads/world, ...] contiguous layout, e.g.
    // sparse MLA which takes no q-stride argument).
    if (dim !== 1 && allPar.every(p => p === TensorParallelism.Row)) {
      const outShape = [...this.shape];
      for (const t of pTensors) outShape[dim] += t.shape[dim];
      const outShards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        outShards.push(this.shards[i].cat(pTensors.map(t => t.shards[i]), dim));
      }
      return this.parallelOps.wrapShards(this.workspace, outShards, outShape, this.type, TensorParallelism.Row);
    }

    // Mix of Row and Replicated, cat on non-sharded dim: slice Replicated tensors
    // on dim 1 to match each Row shard's head range, then cat locally per shard.
    // No communication needed — the Replicated data is already present on every GPU.
    if (dim !== 1 && hasNonReplicated && allPar.every(p => p === TensorParallelism.Row || p === TensorParallelism.Replicated)) {
      const outShape = [...this.shape];
      for (const t of pTensors) {
        outShape[dim] += t.shape[dim];
      }
      const rowPar = this.parallelism === TensorParallelism.Row
        ? this
        : pTensors.find(t => (t as ParallelTensor).parallelism === TensorParallelism.Row)! as ParallelTensor;
      const shardDim1 = rowPar.shards[0].shape[1];
      const outShards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        using selfShard = this.parallelism === TensorParallelism.Row
          ? this.shards[i].viewClone()
          : this.shards[i].slice(1, i * shardDim1, shardDim1);
        const tensorShards = pTensors.map(t => {
          const pt = t as ParallelTensor;
          if (pt.parallelism === TensorParallelism.Row) {
            return pt.shards[i].viewClone();
          }
          return pt.shards[i].slice(1, i * shardDim1, shardDim1);
        });
        outShards.push(selfShard.cat(tensorShards, dim));
        for (const t of tensorShards) {
          t[Symbol.dispose]();
        }
      }
      return this.parallelOps.wrapShards(this.workspace, outShards, outShape, this.type, TensorParallelism.Row);
    }

    if (hasNonReplicated) {
      if (this.parallelism === TensorParallelism.PartialSum) {
        this.allReduce();
      }
      using _gatheredSelf = this.parallelism === TensorParallelism.Replicated ? undefined : this.allGather(this.workspace);
      const gatheredSelf = _gatheredSelf || this;
      const gatheredTensors: Tensor[] = [];
      for (const t of pTensors) {
        if (t.parallelism === TensorParallelism.PartialSum) {
          t.allReduce();
        }
        gatheredTensors.push(t.parallelism === TensorParallelism.Replicated ? t : t.allGather(t.workspace));
      }
      return gatheredSelf.cat(gatheredTensors, dim);
    }

    const outShape = [...this.shape];
    for (const t of pTensors) outShape[dim] += t.shape[dim];
    const outShards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      outShards.push(this.shards[i].cat(pTensors.map(t => t.shards[i]), dim));
    }
    return this.parallelOps.wrapShards(this.workspace, outShards, outShape, this.type, TensorParallelism.Replicated);
  }

  slice(dim: number, start: number, length: number): Tensor {
    super.slice(dim, start, length);
    if (this.parallelism === TensorParallelism.PartialSum) {
      this.allReduce();
      return this.slice(dim, start, length);
    }
    if (this.parallelism === TensorParallelism.Row || this.parallelism === TensorParallelism.Column) {
      using gathered = this.allGather(this.workspace);
      return gathered.slice(dim, start, length);
    }
    const outShape = [...this.shape];
    outShape[dim] = length;
    const outShards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      outShards.push(this.shards[i].slice(dim, start, length));
    }
    return this.parallelOps.wrapShards(this.workspace, outShards, outShape, this.type, TensorParallelism.Replicated);
  }

  narrow(start: number, length: number): Tensor {
    super.narrow(start, length);
    if (this.parallelism !== TensorParallelism.Replicated && this.parallelism !== TensorParallelism.Row) {
      throw new Error(`narrow: only supported on Replicated or Row tensors, got ${this.parallelism}`);
    }
    const newShape = [length, ...this.shape.slice(1)];
    const outShards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      outShards.push(this.shards[i].narrow(start, length));
    }
    return this.parallelOps.wrapShards(this.workspace, outShards, newShape, this.type, this.parallelism);
  }

  scatterScalar(indices: Tensor, value: number, k: number): void {
    const pIndices = indices as ParallelTensor;
    const outDim = this.shape[1];
    const batch = this.shape[0];
    if (this.parallelism !== pIndices.parallelism) {
      throw new Error(`scatterScalar: input parallelism ${this.parallelism} must match indices parallelism ${pIndices.parallelism}`);
    }
    for (let i = 0; i < this.worldSize; i++) {
      this.shards[i].scatterScalar(pIndices.shards[i], value, k);
    }
  }

  groupMaskMul(groupMask: Tensor, expertsPerGroup: number, nGroup: number): void {
    const pGroupMask = groupMask as ParallelTensor;
    const batch = this.shape[0];
    const numExperts = this.shape[1];
    if (this.parallelism !== pGroupMask.parallelism) {
      throw new Error(`groupMaskMul: input parallelism ${this.parallelism} must match groupMask parallelism ${pGroupMask.parallelism}`);
    }
    for (let i = 0; i < this.worldSize; i++) {
      this.shards[i].groupMaskMul(pGroupMask.shards[i], expertsPerGroup, nGroup);
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
    if (this.parallelism === TensorParallelism.PartialSum || this.parallelism === TensorParallelism.Column) {
      throw new Error(`ropeTranspose: unsupported input parallelism ${this.parallelism}`);
    }
    const shardNHeads = this.parallelism === TensorParallelism.Row
      ? this.parallelOps.shardDim(nHeads, "ropeTranspose nHeads")
      : nHeads;
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
    if (this.parallelism === TensorParallelism.PartialSum || this.parallelism === TensorParallelism.Column) {
      throw new Error(`applyRotaryPosEmb: unsupported input parallelism ${this.parallelism}`);
    }
    const shardNHeads = this.parallelism === TensorParallelism.Row
      ? this.parallelOps.shardDim(nHeads, "applyRotaryPosEmb nHeads")
      : nHeads;
    const outShards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      outShards.push(this.shards[i].applyRotaryPosEmb(pCos.shards[i], pSin.shards[i], ropeDim, shardNHeads, seqLen, batch, unsqueezeDim, interleaved));
    }
    return this.parallelOps.wrapShards(this.workspace, outShards, this.shape, this.type, this.parallelism);
  }

  mlaVExpand(vProj: Tensor, seqLen: number, batch: number, lse?: Tensor, _headOffset?: number, _attnNHeads?: number, _vProjHeadOffset?: number, _tokenMajor?: boolean): Tensor {
    super.mlaVExpand(vProj, seqLen, batch);
    const kvLoraRank = this.shape[this.shape.length - 1];
    const nHeads = this.shape[1];
    const vHeadDim = vProj.shape[1];
    const pVProj = vProj as ParallelTensor;
    // if (this.parallelism === TensorParallelism.PartialSum || this.parallelism === TensorParallelism.Column ||
    //     pVProj.parallelism === TensorParallelism.PartialSum || pVProj.parallelism === TensorParallelism.Column) {
    //   throw new Error(`mlaVExpand: unsupported parallelism this=${this.parallelism}, vProj=${pVProj.parallelism}`);
    // }
    const shardNHeads = (this.parallelism === TensorParallelism.Row || pVProj.parallelism === TensorParallelism.Row || pVProj.parallelism === TensorParallelism.Column)
      ? this.parallelOps.shardDim(nHeads, "mlaVExpand nHeads")
      : nHeads;
    const isPartialSoftmax = this.parallelism === TensorParallelism.PartialSoftmax;
    const attnNHeads = isPartialSoftmax ? nHeads : shardNHeads;
    const isVProjSharded = pVProj.parallelism === TensorParallelism.Row;
    const isCp = this.parallelism === TensorParallelism.PartialSoftmax;
    const isVProjReplicated = pVProj.parallelism === TensorParallelism.Replicated;
    const isAttnRowSharded = this.parallelism === TensorParallelism.Row;
    if (isCp && pVProj.parallelism === TensorParallelism.Row) {
      throw new Error(`mlaVExpand: context parallelism does not support Row-parallel v_proj`);
    }
    // Column v_proj in CP prefill could be supported by switching to merge-then-expand
    // (merge attn_out in kvLoraRank space, then v_expand locally), but that doubles
    // merge communication (kvLoraRank=512 > vHeadDim=256) for modest memory savings.
    // Token-major attn_out (sparse SM120 output: [BS, nHeads, kvLoraRank]) has the
    // same layout as decode (one row per token), so it takes the merge-then-expand
    // path regardless of the nominal seqLen.
    const isCpDecodeLayout = isCp && (seqLen === 1 || !!_tokenMajor);
    if (isCp && !isCpDecodeLayout && pVProj.parallelism === TensorParallelism.Column) {
      throw new Error(`mlaVExpand: CP prefill (expand-then-merge) requires Replicated v_proj, got Column`);
    }
    const BS = batch * seqLen;

    // this is faster with 4+ gpu
    // CP decode: merge K-dim attn_out first, then v_expand h heads locally.
    // Avoids redundant v_expand of all H heads on every GPU. Only for decode
    // (seqLen=1) where BHSD layout == BSH layout so the merge can read attn_out directly.
    if (isCpDecodeLayout) {
      const pLse = lse as ParallelTensor;
      using merged = new UsingHolder(this.parallelOps.contextParallelMerge(
        this, pLse,
        BS, nHeads, kvLoraRank,
        null, this.workspace,
      ));
      const h = this.parallelOps.shardDim(nHeads, "mlaVExpand cpShardNHeads");
      const expandShards: Tensor[] = [];
      const narrowViews: Tensor[] = [];
      const reshapeViews: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        let vProjShard = pVProj.shards[i];
        if (pVProj.parallelism === TensorParallelism.Replicated) {
          vProjShard = vProjShard.narrow(i * h * kvLoraRank, h * kvLoraRank);
          narrowViews.push(vProjShard);
        }
        const reshapedMerged = merged.value.shards[i].reshape([BS, h, kvLoraRank]);
        reshapeViews.push(reshapedMerged);
        expandShards.push(reshapedMerged.mlaVExpand(
          vProjShard, 1, BS,
          undefined, 0, h, 0, _tokenMajor
        ));
      }
      for (const v of narrowViews) v[Symbol.dispose]();
      for (const v of reshapeViews) v[Symbol.dispose]();
      return this.parallelOps.wrapShards(this.workspace, expandShards, [BS, nHeads * vHeadDim], this.type, TensorParallelism.Row);
    }

    // Non-CP or CP prefill: expand-then-merge
    const outShards: Tensor[] = [];
    const narrowViews: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      let vProjShard = pVProj.shards[i];
      if (isVProjReplicated && isAttnRowSharded) {
        vProjShard = vProjShard.narrow(i * shardNHeads * kvLoraRank, shardNHeads * kvLoraRank);
        narrowViews.push(vProjShard);
      }
      const shardHeadOffset = (isPartialSoftmax && isVProjSharded) ? i * shardNHeads : 0;
      outShards.push(this.shards[i].mlaVExpand(vProjShard, seqLen, batch, undefined, shardHeadOffset, attnNHeads, 0, _tokenMajor));
    }
    for (const v of narrowViews) v[Symbol.dispose]();
    const vExpandedPar = isCp ? TensorParallelism.Column : (shardNHeads === nHeads ? TensorParallelism.Replicated : this.parallelism);
    const vExpandedFullShape = isCp ? [BS * this.parallelOps.worldSize, nHeads * vHeadDim] : [BS, nHeads * vHeadDim];
    using vExpanded = new UsingHolder(this.parallelOps.wrapShards(this.workspace, outShards, vExpandedFullShape, this.type, vExpandedPar));
    if (!isCp) {
      return vExpanded.detach();;
    }

    const pLse = lse as ParallelTensor;
    const merged = this.parallelOps.contextParallelMerge(
      vExpanded.value, pLse,
      BS, nHeads, vHeadDim,
      null, this.workspace,
    );
    return merged;
  }

  mulMatId(weights: Tensor[], expertIds: Tensor, topK: number, count: number, N: number, K: number, name: string): Tensor {
    const pWeights = weights.map(w => w as ParallelTensor);
    const pExpertIds = expertIds as ParallelTensor;
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
      outShards.push(this.shards[i].mulMatId(shardWeights, pExpertIds.shards[i], topK, count, outN, outK, name));
    }
    return this.parallelOps.wrapShards(this.workspace, outShards, [count, N], this.type, outPar);
  }

  swiGluMlpMoe(
    weights: { gate: Tensor[], up: Tensor[], down: Tensor[] },
    topkIndicesFlat: Tensor,
    topK: number, count: number,
    moeIntermediate: number, hs: number,
    pfx: string,
  ): Tensor {
    super.swiGluMlpMoe(weights, topkIndicesFlat, topK, count, moeIntermediate, hs, pfx);

    const inputPar = this.parallelism;
    if (inputPar !== TensorParallelism.Replicated) {
      throw new Error(`swiGluMlpMoe: unsupported input parallelism ${inputPar} (expected Replicated)`);
    }

    const pGate = weights.gate.map(w => w as ParallelTensor);
    const pUp = weights.up.map(w => w as ParallelTensor);
    const pDown = weights.down.map(w => w as ParallelTensor);
    const pExpertIds = topkIndicesFlat as ParallelTensor;

    // Replicated input: gate/up shard N (moeIntermediate) → Row,
    // silu preserves Row, down shards K (moeIntermediate) → PartialSum.
    const shardMoeIntermediate = this.parallelOps.shardDim(moeIntermediate, "swiGluMlpMoe moeIntermediate");

    const outShards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      const shardWeights = {
        gate: pGate.map(w => w.shards[i]),
        up: pUp.map(w => w.shards[i]),
        down: pDown.map(w => w.shards[i]),
      };
      outShards.push(this.shards[i].swiGluMlpMoe(
        shardWeights, pExpertIds.shards[i], topK, count,
        shardMoeIntermediate, hs, pfx,
      ));
    }
    return this.parallelOps.wrapShards(this.workspace, outShards, [count, hs], this.type, TensorParallelism.PartialSum);
  }

  scatterAddRows(scales: Tensor, topK: number, numRows: number): Tensor {
    const pScales = scales as ParallelTensor;
    const dim = this.shape[1];
    const outShards: Tensor[] = [];
    if (this.parallelism === TensorParallelism.PartialSum) {
      for (let i = 0; i < this.worldSize; i++) {
        outShards.push(this.shards[i].scatterAddRows(pScales.shards[i], topK, numRows));
      }
      return this.parallelOps.wrapShards(this.workspace, outShards, [numRows, dim], this.type, TensorParallelism.PartialSum);
    } else {
      for (let i = 0; i < this.worldSize; i++) {
        outShards.push(this.shards[i].scatterAddRows(pScales.shards[i], topK, numRows));
      }
      return this.parallelOps.wrapShards(this.workspace, outShards, [numRows, dim], this.type, this.parallelism);
    }
  }

  rotaryEmbedding(positionIds: Tensor, batch: number, seqLen: number): { cos: Tensor, sin: Tensor } {
    super.rotaryEmbedding(positionIds, batch, seqLen);
    const dimHalf = this.shape[0];
    const pPositionIds = positionIds as ParallelTensor;
    this.assertParallel("rotaryEmbedding invFreq", this, TensorParallelism.Replicated);
    this.assertParallel("rotaryEmbedding positionIds", pPositionIds, TensorParallelism.Replicated);

    const cosShards: Tensor[] = [];
    const sinShards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      const result = this.shards[i].rotaryEmbedding(pPositionIds.shards[i], batch, seqLen);
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
  readonly worldSize: number;
  private readonly devices: readonly GlmOps[];

  constructor(devices: readonly GlmOps[]) {
    this.worldSize = devices.length;
    this.devices = devices;
    const addon = getNativeAddon();
    const deviceIds = devices.map(d => d.device);

    // 1. Create one instance per rank (metadata only).
    this.instances = devices.map((dev, rank) =>
      addon.p2pCreateInstance(dev.ctx, rank, deviceIds));

    // 2. Cache flag pointers (never changes).
    this.flagPtrs = this.instances.map(inst => addon.p2pGetFlagPtr(inst));

    // 3. Initialize peer flags so barrier works.
    for (let i = 0; i < this.worldSize; ++i) {
      addon.p2pSetPeers(this.devices[i].ctx, this.instances[i], this.flagPtrs);
    }
  }

  free(): void {
    for (const inst of this.instances) {
      getNativeAddon().p2pDestroyInstance(inst);
    }
  }

  /** P2P barrier: sync all GPUs without data transfer. */
  barrier(devices: readonly GlmOps[], peerRanks?: number[]): void {
    const addon = getNativeAddon();
    for (let i = 0; i < this.worldSize; ++i) {
      const peerRank = peerRanks ? peerRanks[i] : -1;
      addon.p2pBarrier(devices[i].ctx, this.instances[i], peerRank);
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
  p2pEnabled: boolean;
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
  getP2PGroup(stream: number): P2PAllReduceGroup | null {
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

  p2pSources: Tensor[] = [];
  sourceCleanup() {
    // arrived at new barrier, release the old sources
    while (this.p2pSources.length) {
      using _src = this.p2pSources.pop()!;
    }
  }

  /** NCCL point-to-point send. Must be paired with ncclRecv on peer. */
  ncclSend(shard: Tensor, rank: number, peer: number, count: number, dtype: number): void {
    getNativeAddon().ncclSend(this.comms[rank], this.devices[rank].ctx, shard.data, count, dtype, peer);
  }

  /** NCCL point-to-point recv. Must be paired with ncclSend on peer. */
  ncclRecv(shard: Tensor, rank: number, peer: number, count: number, dtype: number): void {
    getNativeAddon().ncclRecv(this.comms[rank], this.devices[rank].ctx, shard.data, count, dtype, peer);
  }

  /**
   * Merge partial attention outputs from context-parallel shards.
   *
   * Each GPU has partial_v_out [batch, nHeads * vHeadDim] (BF16)
   * and partial_lse [batch, nHeads] (F32).
   *
   * When P2P is enabled, uses a butterfly tree reduction via cpMergeTree
   * (always produces Replicated output, shardNHeads/inputNHeads ignored).
   *
   * Otherwise falls back to NCCL AllGather + local merge. When shardNHeads
   * is provided, only processes and outputs heads
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
  ): ParallelTensor {
    const count = partialVOuts.shards[0].shape.reduce((a, b) => a * b, 1);
    if (this.p2pEnabled && count <= 65536 * 2) {
      return this.cpMergeTreeReduce(partialVOuts.shards, partialLses.shards, batchSize, numHeads, vHeadDim, workspace);
    }
    return this.agRsMerge(partialVOuts.shards, partialLses, batchSize, numHeads, vHeadDim, workspace);
  }

  /**
   * Merge partial attention outputs via butterfly tree reduction using
   * cpMergeTree. Each round pairs GPUs via XOR distance and merges their
   * partial (v_out, lse) with the online softmax kernel. After log2(N)
   * rounds every GPU holds the fully merged result (Replicated).
   *
   * Each GPU has partial_v_out [batch, numHeads * vHeadDim] (BF16)
   * and partial_lse [batch, numHeads] (F32).
   */
  private cpMergeTreeReduce(
    partialVOuts: readonly Tensor[],
    partialLses: readonly Tensor[],
    batchSize: number,
    numHeads: number,
    vHeadDim: number,
    workspace: WorkspaceBase,
  ): ParallelTensor {
    // fast path for decode: single barrier, flat all-to-all merge.
    // Each GPU reads h heads from all N peers directly via P2P, merges in one kernel launch.
    // Output is Row-parallel (each GPU holds h heads), so no need for butterfly rounds.
    const shardNHeads = this.shardDim(numHeads, "cpMergeTreeReduce shardNHeads");
    const shardNumel = batchSize * shardNHeads * vHeadDim;
    const shardWss = this.getShardWorkspaces(workspace);

    this.p2pBarrier();
    this.sourceCleanup();
    this.p2pSources.push(...partialVOuts.map(t => t.viewClone()), ...partialLses.map(t => t.viewClone()));

    const vPtrs: number[] = partialVOuts.map(t => t.data);
    const lsePtrs: number[] = partialLses.map(t => t.data);

    const outputV: Tensor[] = [];
    const outputLse: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      outputV.push(shardWss[i].alloc([batchSize, shardNHeads * vHeadDim], "BF16"));
      outputLse.push(shardWss[i].alloc([batchSize, shardNHeads], "F32"));
    }

    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].cpMergeTree(
        vPtrs,
        lsePtrs,
        this.worldSize,
        outputV[i], outputLse[i],
        shardNumel, batchSize, numHeads, vHeadDim,
        shardNHeads, i * shardNHeads, numHeads,
      );
    }

    for (const lse of outputLse) {
      lse[Symbol.dispose]();
    }

    return this.wrapShards(workspace, outputV, [batchSize, numHeads * vHeadDim], "BF16", TensorParallelism.Row);
  }

  /**
   * AG+RS merge: AllGather LSE, local correction, ReduceScatter v_out.
   *
   * Replaces the log2(N)-round butterfly for prefill (large batches).
   * Communication: 1 AllGather (LSE only, tiny) + 1 ReduceScatter (corrected v_out).
   * Total data sent per rank: (N-1)/N × B×H×D  vs  log2(N) × B×H×D for butterfly.
   *
   * The v_out [B, H, D] row-major buffer is reshaped to [B*N, H/N * D] (zero-copy)
   * so ncclReduceScatter scatters along the head dimension: rank i receives
   * the sum of head group i across all ranks.
   */
  private agRsMerge(
    partialVOuts: readonly Tensor[],
    partialLses: ParallelTensor,
    batchSize: number,
    numHeads: number,
    vHeadDim: number,
    workspace: WorkspaceBase,
  ): ParallelTensor {
    const N = this.worldSize;
    const shardNHeads = this.shardDim(numHeads, "agRsMerge shardNHeads");
    const shardVOutCount = batchSize * shardNHeads * vHeadDim;
    const shardWss = this.getShardWorkspaces(workspace);

    // Step 1: AllGather LSE → [N*B, H] F32 on each GPU.
    // LSE is PartialSoftmax (same [B, H] shape per shard), so we can't use
    // ParallelTensor.allGather (which only supports Column/Row). Use ncclAllGather
    // directly: each rank contributes B*H elements, output is N*B*H on each rank.
    const lseCount = batchSize * numHeads;
    const gatheredLse: Tensor[] = [];
    for (let i = 0; i < N; i++) {
      gatheredLse.push(shardWss[i].alloc([N * batchSize, numHeads], "F32"));
    }
    getNativeAddon().ncclGroupStart();
    for (let i = 0; i < N; i++) {
      getNativeAddon().ncclAllGather(
        this.comms[i], this.devices[i].ctx,
        partialLses.shards[i].data, gatheredLse[i].data,
        lseCount, NCCL_FLOAT32,
      );
    }
    getNativeAddon().ncclGroupEnd();

    // Step 2: Correct v_out in-place (rescale by exp2(lse_local - global_lse))
    for (let i = 0; i < N; i++) {
      this.devices[i].cpCorrectAttnOut(
        partialVOuts[i], gatheredLse[i], null,
        batchSize, numHeads, vHeadDim, N, i,
      );
    }

    for (const lse of gatheredLse) {
      lse[Symbol.dispose]();
    }

    // Step 3: Transpose v_out [B, H, D] → [N, B, H/N, D] so ReduceScatter
    // scatters along the head dimension (block j = all batches, head group j).
    // Without this transpose, ReduceScatter would split across batch+head.
    const transposed: Tensor[] = [];
    for (let i = 0; i < N; i++) {
      transposed.push((partialVOuts[i] as GlmTensor).transpose4d(
        batchSize, N, shardNHeads, vHeadDim, 1, 0, 2, 3,
      ));
    }

    // Step 4: ReduceScatter → [B, H/N * D] on each GPU.
    const outputV: Tensor[] = [];
    for (let i = 0; i < N; i++) {
      outputV.push(shardWss[i].alloc([batchSize, shardNHeads * vHeadDim], "BF16"));
    }
    getNativeAddon().ncclGroupStart();
    for (let i = 0; i < N; i++) {
      getNativeAddon().ncclReduceScatter(
        this.comms[i], this.devices[i].ctx,
        transposed[i].data, outputV[i].data,
        shardVOutCount, NCCL_BFLOAT16, NCCL_SUM,
      );
    }
    getNativeAddon().ncclGroupEnd();

    for (const t of transposed) {
      t[Symbol.dispose]();
    }

    return this.wrapShards(workspace, outputV, [batchSize, numHeads * vHeadDim], "BF16", TensorParallelism.Row);
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

  /** P2P barrier: sync all GPUs without data transfer. */
  p2pBarrier(peerRanks?: number[]): void {
    const group = this.getP2PGroup(this.devices[0].currentStream);
    if (!group) throw new Error('P2P not available for barrier');
    group.barrier(this.devices, peerRanks);
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
      case "U8": return NCCL_UINT8;
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
      pinned ? ws.allocPinned(ss, type, name) : ws.alloc(ss, type, name),
    );
    return new ParallelTensor(workspace, this, par, shards, shape, type, name, pinned, undefined);
  }

  wrapTensor(workspace: WorkspaceBase, data: number, allocSize: number, shape: number[], type: string, pinned: boolean, view: ParallelTensor | undefined): Tensor {
    if (!view)
      throw new Error("ParallelOps.wrapTensor not supported; tensor recycling happens at shard level");
    return new ParallelTensor(workspace, this, view.parallelism, view.shards, shape, type, undefined, pinned, view);
  }

  wrapShards(workspace: WorkspaceBase, shards: Tensor[], fullShape: number[], type: string, parallelism: TensorParallelism, view?: ParallelTensor): ParallelTensor {
    if (shards.length === 0) {
      throw new Error("wrapShards: no shards provided");
    }
    const ws = shards.length;
    const fullElems = fullShape.reduce((a, b) => a * b, 1);
    const shardElems = shards[0].shape.reduce((a, b) => a * b, 1);
    switch (parallelism) {
      case TensorParallelism.Replicated:
      case TensorParallelism.PartialSum:
      case TensorParallelism.PartialSoftmax:
        if (shardElems !== fullElems) {
          throw new Error(`wrapShards(${parallelism}): shard elems ${shardElems} (shape [${shards[0].shape}]) != full elems ${fullElems} (shape [${fullShape}])`);
        }
        break;
      case TensorParallelism.Row:
      case TensorParallelism.Column:
        if (shardElems * ws !== fullElems) {
          throw new Error(`wrapShards(${parallelism}): shard elems ${shardElems} * worldSize ${ws} = ${shardElems * ws} != full elems ${fullElems} (shape [${fullShape}])`);
        }
        break;
    }
    const pt = new ParallelTensor(workspace, this, parallelism, shards, fullShape, type, undefined, !!view?.pinned, view);
    workspace.addTracked(pt);
    return pt;
  }

  tryNarrowToColumnParallel(tensor: ParallelTensor): ParallelTensor | undefined {
    if (tensor.parallelism !== TensorParallelism.Replicated || tensor.shape[0] % this.worldSize !== 0) {
      return undefined;
    }
    const W = this.worldSize;
    const shardSize = tensor.shape[0] / W;
    const shards: Tensor[] = [];
    for (let i = 0; i < W; i++) {
      shards.push(tensor.shards[i].narrow(i * shardSize, shardSize));
    }
    return this.wrapShards(tensor.workspace, shards, tensor.shape, tensor.type, TensorParallelism.Column, tensor);
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

  kvCacheWrite(srcK: Tensor, srcV: Tensor, dstK: Tensor, dstV: Tensor, slotMapping: Tensor, batchSize: number, nKv: number, hd: number, srcKTokenStride: number, srcKHeadStride: number, srcVTokenStride: number, srcVHeadStride: number): void {
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
      this.devices[i].kvCacheWrite(pSrcK.shards[i], pSrcV.shards[i], pDstK.shards[i], pDstV.shards[i], pSlotMapping.shards[i], batchSize, shardNKv, hd, shardKTokenStride, srcKHeadStride, shardVTokenStride, srcVHeadStride);
    }
  }

  positionStep(positionIds: Tensor, lastPageLen: Tensor, slotMapping: Tensor, indptr: Tensor, indices: Tensor, pageSize: number, batchSize: number, steps?: number): void {
    const pPositionIds = this.cast(positionIds);
    const pLastPageLen = this.cast(lastPageLen);
    const pSlotMapping = this.cast(slotMapping);
    const pIndptr = this.cast(indptr);
    const pIndices = this.cast(indices);
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].positionStep(pPositionIds.shards[i], pLastPageLen.shards[i], pSlotMapping.shards[i], pIndptr.shards[i], pIndices.shards[i], pageSize, batchSize, steps);
    }
  }

  mlaPositionStep(positionIds: Tensor, lastPageLen: Tensor, indptr: Tensor, pageSize: number, batchSize: number, contextParallel?: boolean, _cpWorldSize?: number, _cpRank?: number, steps?: number, globalLastPageLen?: Tensor): void {
    const pPositionIds = this.cast(positionIds);
    const pLastPageLen = this.cast(lastPageLen);
    const pIndptr = this.cast(indptr);
    const pGlobalLastPageLen = globalLastPageLen ? this.cast(globalLastPageLen) : undefined;
    const cpWs = contextParallel ? this.worldSize : 1;
    for (let i = 0; i < this.worldSize; i++) {
      const cpR = contextParallel ? i : 0;
      this.devices[i].mlaPositionStep(pPositionIds.shards[i], pLastPageLen.shards[i], pIndptr.shards[i], pageSize, batchSize, contextParallel, cpWs, cpR, steps, pGlobalLastPageLen?.shards[i]);
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

  batchDecodeRun(state: ExecutionState, q: Tensor, o: Tensor, kData: Tensor, vData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, numQoHeads: number, numKvHeads: number, headDim: number, smScale: number): void {
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
      this.devices[i].batchDecodeRun(state, pQ.shards[i], pO.shards[i], pKData.shards[i], pVData.shards[i], pIndices.shards[i], pIndptrD.shards[i], pLastPageLen.shards[i], pFloatWs.shards[i], pIntWs.shards[i], pPlanInfo.shards[i], this.shardDim(numQoHeads, "batchDecodeRun numQoHeads"), this.shardDim(numKvHeads, "batchDecodeRun numKvHeads"), headDim, smScale);
    }
  }

  batchPrefillPagedPlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, qoIndptrH: Tensor, pagedKvIndptrH: Tensor, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, maskMode: MaskMode): void {
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

  batchPrefillPagedRun(state: ExecutionState, q: Tensor, o: Tensor, kData: Tensor, vData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, floatWs: Tensor, intWs: Tensor, qIndptrD: Tensor, planInfo: Tensor, numQoHeads: number, numKvHeads: number, headDim: number, qStrideN: number, qStrideH: number, maskMode: MaskMode, smScale: number): void {
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
      this.devices[i].batchPrefillPagedRun(state, pQ.shards[i], pO.shards[i], pKData.shards[i], pVData.shards[i], pIndices.shards[i], pIndptrD.shards[i], pLastPageLen.shards[i], pFloatWs.shards[i], pIntWs.shards[i], pQIndptrD.shards[i], pPlanInfo.shards[i], this.shardDim(numQoHeads, "batchPrefillPagedRun numQoHeads"), this.shardDim(numKvHeads, "batchPrefillPagedRun numKvHeads"), headDim, qStrideN, qStrideH, maskMode, smScale);
    }
  }

  batchPrefillRaggedPlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, qoIndptrH: Tensor, kvIndptrH: Tensor, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, maskMode: MaskMode): void {
    const pFloatWs = this.cast(floatWs);
    const pIntWs = this.cast(intWs);
    const pPinnedIntWs = this.cast(pinnedIntWs);
    const pPlanInfo = this.cast(planInfo);
    const pQoIndptrH = this.cast(qoIndptrH);
    const pKvIndptrH = this.cast(kvIndptrH);
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].batchPrefillRaggedPlan(pFloatWs.shards[i], floatWsSize, pIntWs.shards[i], pPinnedIntWs.shards[i], intWsSize, pPlanInfo.shards[i], pQoIndptrH.shards[i], pKvIndptrH.shards[i], totalQoRows, batchSize, this.shardDim(numQoHeads, "batchPrefillRaggedPlan numQoHeads"), this.shardDim(numKvHeads, "batchPrefillRaggedPlan numKvHeads"), headDim, maskMode);
    }
  }

  batchPrefillRaggedRun(state: ExecutionState, q: Tensor, k: Tensor, v: Tensor, o: Tensor, floatWs: Tensor, intWs: Tensor, qIndptrD: Tensor, kvIndptrD: Tensor, planInfo: Tensor, numQoHeads: number, numKvHeads: number, headDim: number, qStrideN: number, qStrideH: number, kvStrideN: number, kvStrideH: number, vStrideN: number, vStrideH: number, maskMode: MaskMode, smScale: number): void {
    const pQ = this.cast(q);
    const pK = this.cast(k);
    const pV = this.cast(v);
    const pO = this.cast(o);
    const pFloatWs = this.cast(floatWs);
    const pIntWs = this.cast(intWs);
    const pQIndptrD = this.cast(qIndptrD);
    const pKvIndptrD = this.cast(kvIndptrD);
    const pPlanInfo = this.cast(planInfo);
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].batchPrefillRaggedRun(state, pQ.shards[i], pK.shards[i], pV.shards[i], pO.shards[i], pFloatWs.shards[i], pIntWs.shards[i], pQIndptrD.shards[i], pKvIndptrD.shards[i], pPlanInfo.shards[i], this.shardDim(numQoHeads, "batchPrefillRaggedRun numQoHeads"), this.shardDim(numKvHeads, "batchPrefillRaggedRun numKvHeads"), headDim, qStrideN, qStrideH, kvStrideN, kvStrideH, vStrideN, vStrideH, maskMode, smScale);
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

  mlaPrefillRun(state: ExecutionState, qNope: Tensor, qPe: Tensor, ckvData: Tensor, kpeData: Tensor, kvIndices: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, smScale: number, maskMode: MaskMode, _cpWorldSize?: number, _cpRank?: number, customMask?: Tensor, maskIndptr?: Tensor, maskKvLen?: Tensor): { o: Tensor, lse: Tensor } {
    let pQNope = this.cast(qNope);
    let pQPe = this.cast(qPe);
    const pCkvData = this.cast(ckvData);
    const pKpeData = this.cast(kpeData);
    const pKvIndices = this.cast(kvIndices);
    const pFloatWs = this.cast(floatWs);
    const pIntWs = this.cast(intWs);
    const pPlanInfo = this.cast(planInfo);
    const contextParallel = pCkvData.parallelism === TensorParallelism.Row;
    const numHeads = pQNope.shape[1];
    const headDimCkv = pCkvData.shape[2];
    const headDimKpe = pKpeData.shape[2];
    const totalTokens = state.totalTokens;
    const effectiveCpWorldSize = contextParallel ? this.worldSize : undefined;
    const lsePar = contextParallel ? TensorParallelism.Column : TensorParallelism.Row;
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
    const pCustomMask = customMask ? this.cast(customMask) : undefined;
    const pMaskIndptr = maskIndptr ? this.cast(maskIndptr) : undefined;
    const pMaskKvLen = maskKvLen ? this.cast(maskKvLen) : undefined;
    const oShards: Tensor[] = [];
    const lseShards: Tensor[] = [];
    try {
      for (let i = 0; i < this.worldSize; i++) {
        const effectiveCpRank = contextParallel ? i : undefined;
        const shardCustomMask = pCustomMask ? pCustomMask.shards[i] : undefined;
        const shardMaskIndptr = pMaskIndptr ? pMaskIndptr.shards[i] : undefined;
        const shardMaskKvLen = pMaskKvLen ? pMaskKvLen.shards[i] : undefined;
        const shardResult = this.devices[i].mlaPrefillRun(state, pQNope.shards[i], pQPe.shards[i], pCkvData.shards[i], pKpeData.shards[i], pKvIndices.shards[i], pFloatWs.shards[i], pIntWs.shards[i], pPlanInfo.shards[i], smScale, maskMode, effectiveCpWorldSize, effectiveCpRank, shardCustomMask, shardMaskIndptr, shardMaskKvLen);
        oShards.push(shardResult.o);
        lseShards.push(shardResult.lse);
      }
    } finally {
      if (gatheredQNope) gatheredQNope[Symbol.dispose]();
      if (gatheredQPe) gatheredQPe[Symbol.dispose]();
    }
    const o = this.wrapShards(pFloatWs.workspace, oShards, oFullShape, qNope.type, oPar);
    const lse = this.wrapShards(pFloatWs.workspace, lseShards, lseFullShape, "F32", lsePar);
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

  mlaDecodeRun(state: ExecutionState, qNope: Tensor, qPe: Tensor, ckvData: Tensor, kpeData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, smScale: number): { o: Tensor, lse: Tensor } {
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
    const contextParallel = pCkvData.parallelism === TensorParallelism.Row;
    const numQoHeads = pQNope.shape[1];
    const headDimCkv = pCkvData.shape[2];
    const headDimKpe = pKpeData.shape[2];
    const lsePar = contextParallel ? TensorParallelism.Column : TensorParallelism.Row;
    const lseFullShape = contextParallel ? [state.batchSize * this.worldSize, numQoHeads] : [state.batchSize, numQoHeads];
    const oPar = contextParallel ? TensorParallelism.PartialSoftmax : qNope.parallelism;
    const oFullShape = [state.batchSize, numQoHeads, 1, headDimCkv];
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
        const shardResult = this.devices[i].mlaDecodeRun(state, pQNope.shards[i], pQPe.shards[i], pCkvData.shards[i], pKpeData.shards[i], pIndices.shards[i], pIndptrD.shards[i], pLastPageLen.shards[i], pFloatWs.shards[i], pIntWs.shards[i], pPlanInfo.shards[i], smScale);
        oShards.push(shardResult.o);
        lseShards.push(shardResult.lse);
      }
    } finally {
      if (gatheredQNope) gatheredQNope[Symbol.dispose]();
      if (gatheredQPe) gatheredQPe[Symbol.dispose]();
    }
    const o = this.wrapShards(pFloatWs.workspace, oShards, oFullShape, qNope.type, oPar);
    const lse = this.wrapShards(pFloatWs.workspace, lseShards, lseFullShape, "F32", lsePar);
    return { o, lse };
  }

  mlaKvCacheAppend(ckvData: Tensor, kpeData: Tensor | null, indices: Tensor, indptr: Tensor, lastPageLen: Tensor, appendCkv: Tensor, appendKpe: Tensor | null, batchIndices: Tensor, positions: Tensor, nnz: number, headDimCkv: number, headDimKpe: number, appendCkvStrideN: number, appendKpeStrideN: number, _pageSize?: number, _cpWorldSize?: number, _cpRank?: number): void {
    const pCkvData = this.cast(ckvData);
    const pKpeData = kpeData ? this.cast(kpeData) : null;
    const pIndices = this.cast(indices);
    const pIndptr = this.cast(indptr);
    const pLastPageLen = this.cast(lastPageLen);
    const pAppendCkv = this.cast(appendCkv);
    const pAppendKpe = appendKpe ? this.cast(appendKpe) : null;
    const pBatchIndices = this.cast(batchIndices);
    const pPositions = this.cast(positions);
    const contextParallel = pCkvData.parallelism === TensorParallelism.Row;
    const effectiveCpWorldSize = contextParallel ? this.worldSize : undefined;
    const pageSize = pCkvData.shape[1];
    for (let i = 0; i < this.worldSize; i++) {
      const effectiveCpRank = contextParallel ? i : undefined;
      this.devices[i].mlaKvCacheAppend(pCkvData.shards[i], pKpeData?.shards[i] ?? null, pIndices.shards[i], pIndptr.shards[i], pLastPageLen.shards[i], pAppendCkv.shards[i], pAppendKpe?.shards[i] ?? null, pBatchIndices.shards[i], pPositions.shards[i], nnz, headDimCkv, headDimKpe, appendCkvStrideN, appendKpeStrideN, pageSize, effectiveCpWorldSize, effectiveCpRank);
    }
  }

  concatAndCacheDsMla(kvCache: Tensor, appendCkv: Tensor, appendKpe: Tensor, indices: Tensor, indptr: Tensor, batchIndices: Tensor, positions: Tensor, nnz: number, kvLoraRank: number, peDim: number, appendCkvStrideN: number, appendKpeStrideN: number, _pageSize?: number, _cpWorldSize?: number, _cpRank?: number): void {
    const pKvCache = this.cast(kvCache);
    const pAppendCkv = this.cast(appendCkv);
    const pAppendKpe = this.cast(appendKpe);
    const pIndices = this.cast(indices);
    const pIndptr = this.cast(indptr);
    const pBatchIndices = this.cast(batchIndices);
    const pPositions = this.cast(positions);
    const contextParallel = pKvCache.parallelism === TensorParallelism.Row;
    const cpWorldSize = contextParallel ? this.worldSize : 0;
    const pageSize = pKvCache.shape[1];
    for (let i = 0; i < this.worldSize; i++) {
      const cpRank = contextParallel ? i : 0;
      this.devices[i].concatAndCacheDsMla(pKvCache.shards[i], pAppendCkv.shards[i], pAppendKpe.shards[i], pIndices.shards[i], pIndptr.shards[i], pBatchIndices.shards[i], pPositions.shards[i], nnz, kvLoraRank, peDim, appendCkvStrideN, appendKpeStrideN, pageSize, cpWorldSize, cpRank);
    }
  }

  gdnRecurrentStep(state: ExecutionState, output: Tensor, recurrentState: Tensor, qkv: Tensor, aRaw: Tensor, bRaw: Tensor, aLog: Tensor, dtBias: Tensor, numHeads: number, dK: number, dV: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void {
    const pOutput = this.cast(output);
    const pState = this.cast(recurrentState);
    const pQkv = this.cast(qkv);
    const pARaw = this.cast(aRaw);
    const pBRaw = this.cast(bRaw);
    const pALog = this.cast(aLog);
    const pDtBias = this.cast(dtBias);
    if (pQkv.parallelism === TensorParallelism.Column) {
      throw new Error(`gdnRecurrentStep: unsupported qkv parallelism ${pQkv.parallelism}`);
    }
    const isRowPar = pQkv.parallelism === TensorParallelism.Row;
    const shardHeads = isRowPar ? this.shardDim(numHeads, "gdnRecurrentStep numHeads") : numHeads;
    const shardStateStride = isRowPar ? stateStride / this.worldSize : stateStride;
    const shardSeqStride = isRowPar ? qkvSeqStride / this.worldSize : qkvSeqStride;
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].gdnRecurrentStep(state, pOutput.shards[i], pState.shards[i], pQkv.shards[i], pARaw.shards[i], pBRaw.shards[i], pALog.shards[i], pDtBias.shards[i], shardHeads, dK, dV, shardStateStride, qkvChStride, shardSeqStride);
    }
  }

  gdnPrefill(state: ExecutionState, output: Tensor, recurrentState: Tensor, qkv: Tensor, aRaw: Tensor, bRaw: Tensor, aLog: Tensor, dtBias: Tensor, cuSeqlens: Tensor, numHeads: number, dK: number, dV: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void {
    const pOutput = this.cast(output);
    const pState = this.cast(recurrentState);
    const pQkv = this.cast(qkv);
    const pARaw = this.cast(aRaw);
    const pBRaw = this.cast(bRaw);
    const pALog = this.cast(aLog);
    const pDtBias = this.cast(dtBias);
    const pCuSeqlens = this.cast(cuSeqlens);
    if (pQkv.parallelism === TensorParallelism.Column) {
      throw new Error(`gdnPrefill: unsupported qkv parallelism ${pQkv.parallelism}`);
    }
    const isRowPar = pQkv.parallelism === TensorParallelism.Row;
    const shardHeads = isRowPar ? this.shardDim(numHeads, "gdnPrefill numHeads") : numHeads;
    const shardStateStride = isRowPar ? stateStride / this.worldSize : stateStride;
    const shardSeqStride = isRowPar ? qkvSeqStride / this.worldSize : qkvSeqStride;
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].gdnPrefill(state, pOutput.shards[i], pState.shards[i], pQkv.shards[i], pARaw.shards[i], pBRaw.shards[i], pALog.shards[i], pDtBias.shards[i], pCuSeqlens.shards[i], shardHeads, dK, dV, shardStateStride, qkvChStride, shardSeqStride);
    }
  }

  shouldGatherKv(state: ExecutionState) {
    let shouldGatherKv = CP_GATHER_KV && state.cache.getPagedKV().contextParallel && state.getGraphVariantPaddedQLen() > 32;
    if (shouldGatherKv) {
      console.warn('Gathering KV cache for CP mode due to padded Q length > 32. This may be inefficient for large KV caches.');
      const paddedKvLen = state.getGraphVariantPaddedKvLen();
      const paddedQLen = state.getGraphVariantPaddedQLen();
      shouldGatherKv = paddedQLen * 162 >= paddedKvLen;
    }
    return shouldGatherKv;
  }


  sparseMlaPrefill(state: ExecutionState, qAbsorbed: Tensor, qPe: Tensor, kvCache: Tensor, indices: Tensor, topk: number, smScale: number, topkLength: Tensor, pageIndptrD: Tensor, lastPageLen: Tensor, kvTokenIndptrD: Tensor): { o: Tensor, lse: Tensor } {
    const pagedKV = state.cache.getPagedKV();
    const numTokens = state.totalTokens;
    const pQAbsorbed = this.cast(qAbsorbed);
    const pQPe = this.cast(qPe);
    const pKvCache = this.cast(kvCache);
    const pIndices = this.cast(indices);
    const pTopkLength = this.cast(topkLength);
    const contextParallel = pKvCache.parallelism === TensorParallelism.Row;
    const numHeads = pQAbsorbed.shape[1];
    const headDim = pQAbsorbed.shape[2];

    const shouldGatherKv = this.shouldGatherKv(state);

    using gatheredKv = shouldGatherKv
      ? this.gatherPages(
        kvCache, pagedKV.indices, pageIndptrD, lastPageLen,
        state.batchSize,
        pagedKV.maxPages * pagedKV.pageSize,
        kvTokenIndptrD, true,
      )
      : undefined;
    const effectiveKvCache = gatheredKv ?? kvCache;
    const pEffKvCache = this.cast(effectiveKvCache);
    const nonCp = pEffKvCache.parallelism !== TensorParallelism.Row;
    const oPar = nonCp ? TensorParallelism.Row : TensorParallelism.PartialSoftmax;
    using gatheredQAbsorbed: ParallelTensor = (pQAbsorbed.parallelism === TensorParallelism.Row && nonCp)
      ? pQAbsorbed.viewClone() as ParallelTensor
      : (contextParallel && pQAbsorbed.parallelism === TensorParallelism.Row)
        ? pQAbsorbed.allGather(pQAbsorbed.workspace)
        : pQAbsorbed.viewClone() as ParallelTensor;
    using gatheredQPe: ParallelTensor = (pQPe.parallelism === TensorParallelism.Row && nonCp)
      ? pQPe.viewClone() as ParallelTensor
      : (contextParallel && pQPe.parallelism === TensorParallelism.Row)
        ? pQPe.allGather(pQPe.workspace)
        : pQPe.viewClone() as ParallelTensor;
    const oShards: Tensor[] = [];
    const lseShards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      const result = this.devices[i].sparseMlaPrefill(state, gatheredQAbsorbed.shards[i], gatheredQPe.shards[i], pEffKvCache.shards[i], pIndices.shards[i], topk, smScale, pTopkLength.shards[i], pageIndptrD, lastPageLen, kvTokenIndptrD);
      oShards.push(result.o);
      lseShards.push(result.lse);
    }
    const o = this.wrapShards(qAbsorbed.workspace, oShards, [numTokens, numHeads, headDim], "BF16", oPar);
    const lse = this.wrapShards(qAbsorbed.workspace, lseShards, [numTokens, numHeads], "F32", oPar);
    return { o, lse };
  }

  sparseMlaDecode(state: ExecutionState, qAbsorbed: Tensor, qPe: Tensor, kvCache: Tensor, indices: Tensor, topk: number, numSplits: number, smScale: number, chunksPerBlock: number, topkLength?: Tensor): { o: Tensor, lse: Tensor } {
    const numTokens = state.batchSize;
    const pQAbsorbed = this.cast(qAbsorbed);
    const pQPe = this.cast(qPe);
    const pKvCache = this.cast(kvCache);
    const pIndices = this.cast(indices);
    const pTopkLength = topkLength ? this.cast(topkLength) : undefined;
    const contextParallel = pKvCache.parallelism === TensorParallelism.Row;
    const numHeads = pQAbsorbed.shape[1];
    const headDim = pQAbsorbed.shape[2];
    const effectiveNumHeads = contextParallel ? numHeads : this.shardDim(numHeads, "sparseMlaDecode numHeads");
    const oPar = contextParallel ? TensorParallelism.PartialSoftmax : TensorParallelism.Row;
    using gatheredQAbsorbed: ParallelTensor = (contextParallel && pQAbsorbed.parallelism === TensorParallelism.Row)
      ? pQAbsorbed.allGather(pQAbsorbed.workspace)
      : pQAbsorbed.viewClone() as ParallelTensor;
    using gatheredQPe: ParallelTensor = (contextParallel && pQPe.parallelism === TensorParallelism.Row)
      ? pQPe.allGather(pQPe.workspace)
      : pQPe.viewClone() as ParallelTensor;
    const oShards: Tensor[] = [];
    const lseShards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      const result = this.devices[i].sparseMlaDecode(state, gatheredQAbsorbed.shards[i], gatheredQPe.shards[i], pKvCache.shards[i], pIndices.shards[i], topk, numSplits, smScale, chunksPerBlock, pTopkLength?.shards[i]);
      oShards.push(result.o);
      lseShards.push(result.lse);
    }
    const o = this.wrapShards(qAbsorbed.workspace, oShards, [numTokens, numHeads, headDim], "BF16", oPar);
    const lse = this.wrapShards(qAbsorbed.workspace, lseShards, [numTokens, numHeads], "F32", oPar);
    return { o, lse };
  }

  gatherPages(srcData: Tensor, pageIndices: Tensor, pageIndptrD: Tensor, lastPageLen: Tensor, batchSize: number, paddedKvLen: number, kvTokenIndptrD: Tensor, contextParallel: boolean): Tensor {
    const workspace = pageIndptrD.workspace;
    const pSrc = this.cast(srcData);
    const pIndices = this.cast(pageIndices);
    const pIndptr = this.cast(pageIndptrD);
    const pLastPageLen = this.cast(lastPageLen);
    const pKvIndptr = this.cast(kvTokenIndptrD);
    const pageSize = srcData.shape[1];
    const D = srcData.shape[2];

    if (!contextParallel) {
      const shards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        shards.push(this.devices[i].gatherPages(pSrc.shards[i], pIndices.shards[i], pIndptr.shards[i], pLastPageLen.shards[i], batchSize, paddedKvLen, pKvIndptr.shards[i], false));
      }
      return this.wrapShards(workspace, shards, [paddedKvLen / pageSize, pageSize, D], pSrc.type, pSrc.parallelism);
    }

    // CP: each GPU has every Nth token within each page (Row-parallel, sharded on pageSize dim)
    const cpWorldSize = this.worldSize;
    const paddedLocalLen = paddedKvLen / cpWorldSize;

    // Step 1: Local gather — each GPU gathers from its shard of the KV data
    const localBufs: Tensor[] = [];
    for (let i = 0; i < cpWorldSize; i++) {
      localBufs.push(this.devices[i].gatherPages(pSrc.shards[i], pIndices.shards[i], pIndptr.shards[i], pLastPageLen.shards[i], batchSize, paddedLocalLen, pKvIndptr.shards[i], false));
    }

    const shardPageSize = pSrc.shards[0].shape[1];
    // Step 2: NCCL all-gather (Column → Replicated)
    using localPar = this.wrapShards(workspace, localBufs, [paddedKvLen / shardPageSize, shardPageSize, D], pSrc.type, TensorParallelism.Column);
    using gathered = localPar.allGather(workspace);

    // Step 3: Deinterleave — reorder interleaved tokens to sequential
    const out = workspace.alloc([paddedKvLen / shardPageSize, shardPageSize, D], pSrc.type) as ParallelTensor;
    const pOut = this.cast(out);
    const elemBytes = pSrc.type === "U8" ? 1 : 2;

    for (let i = 0; i < cpWorldSize; i++) {
      getNativeAddon().deinterleave(
        this.devices[i].ctx,
        pOut.shards[i].data, gathered.shards[i].data,
        cpWorldSize, paddedKvLen,
        pIndptr.shards[i].data, pKvIndptr.shards[i].data,
        batchSize, pageSize, D * elemBytes,
      );
    }
    return out;
  }

  indexerScore(out: Tensor, q: Tensor, kData: Tensor, weights: Tensor, pageIndices: Tensor, pageIndptr: Tensor, lastPageLen: Tensor, qoIndptr: Tensor, scale: number, totalQ: number, idxNHeads: number, idxHeadDim: number, pageSize: number, maxKvLen: number, causal: boolean): void {
    const pOut = this.cast(out);
    const pQ = this.cast(q);
    const pKData = this.cast(kData);
    const pWeights = this.cast(weights);
    const pIndices = this.cast(pageIndices);
    const pIndptr = this.cast(pageIndptr);
    const pLastPageLen = this.cast(lastPageLen);
    const pQoIndptr = this.cast(qoIndptr);
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].indexerScore(pOut.shards[i], pQ.shards[i], pKData.shards[i], pWeights.shards[i], pIndices.shards[i], pIndptr.shards[i], pLastPageLen.shards[i], pQoIndptr.shards[i], scale, totalQ, idxNHeads, idxHeadDim, pageSize, maxKvLen, causal);
    }
  }

  // Query-parallel indexer: for single-sequence prefill (qoIndptr has 2 entries
  // → one sequence), each rank processes a contiguous slice of query rows
  // [i*localQ, (i+1)*localQ) from its already-Replicated copy of idxQ/weights
  // (free narrow, no scatter). The full qoIndptr is reused so numQueries stays
  // totalQ_full → correct causal prefix; qGlobalStart = i*localQ shifts the
  // causal limit. Output [localQ, topk] per rank is Column-parallel → AllGather
  // → Replicated [totalQ, topk] (cheap concat path). topkLength is gathered
  // in-place via NCCL AllGather.
  //
  // Fall-back (replicated): decode, context-parallel, multi-sequence prefill,
  // or uneven totalQ — every rank runs the full indexer. Decode is cheap;
  // multi-seq / CP need qoIndptr rebasing which is not yet implemented.
  indexerTopk(idxQ: Tensor, kData: Tensor, weights: Tensor, pageIndices: Tensor, indptr: Tensor, lastPageLen: Tensor, qoIndptr: Tensor, scale: number, topk: number, decode: boolean, qGlobalStart?: number, customMask?: Tensor, maskIndptr?: Tensor, maskKvLen?: Tensor): Tensor {
    const totalQ = idxQ.shape[0];
    const pQ = this.cast(idxQ);
    const pKData = this.cast(kData);
    const pWeights = this.cast(weights);
    const pPageIndices = this.cast(pageIndices);
    const pIndptr = this.cast(indptr);
    const pLastPageLen = this.cast(lastPageLen);
    const pQoIndptr = this.cast(qoIndptr);
    const pCustomMask = customMask ? this.cast(customMask) : undefined;
    const pMaskIndptr = maskIndptr ? this.cast(maskIndptr) : undefined;
    const pMaskKvLen = maskKvLen ? this.cast(maskKvLen) : undefined;

    const W = this.worldSize;
    const kDataReplicated = kData.parallelism === TensorParallelism.Replicated;
    using colIdxQ = this.tryNarrowToColumnParallel(pQ);
    using colWeights = this.tryNarrowToColumnParallel(pWeights);
    const canShard = !decode && W > 1 && kDataReplicated
      && pQoIndptr.shards[0].shape[0] === 2
      && colIdxQ && colWeights
      && totalQ % W === 0;

    if (true || !canShard) {
      const topkIdxShards: Tensor[] = [];
      for (let i = 0; i < W; i++) {
        topkIdxShards.push(this.devices[i].indexerTopk(pQ.shards[i], pKData.shards[i], pWeights.shards[i], pPageIndices.shards[i], pIndptr.shards[i], pLastPageLen.shards[i], pQoIndptr.shards[i], scale, topk, decode, qGlobalStart ?? 0, pCustomMask?.shards[i], pMaskIndptr?.shards[i], pMaskKvLen?.shards[i]));
      }
      return this.wrapShards(idxQ.workspace, topkIdxShards, [totalQ, topk], "I32", TensorParallelism.Replicated);
    }

    // Query-sharded path: each rank processes totalQ/W query rows.
    const localQ = totalQ / W;
    const topkIdxShards: Tensor[] = [];
    for (let i = 0; i < W; i++) {
      const qStart = i * localQ;
      topkIdxShards.push(this.devices[i].indexerTopk(
        colIdxQ!.shards[i], pKData.shards[i], colWeights!.shards[i],
        pPageIndices.shards[i], pIndptr.shards[i], pLastPageLen.shards[i],
        pQoIndptr.shards[i],
        scale, topk,
        decode, qStart,
        pCustomMask?.shards[i], pMaskIndptr?.shards[i], pMaskKvLen?.shards[i]
      ));
    }

    // Column [totalQ, topk] → AllGather → Replicated [totalQ, topk].
    using topkIdxColumn = this.wrapShards(idxQ.workspace, topkIdxShards, [totalQ, topk], "I32", TensorParallelism.Column);
    const topkIdxReplicated = topkIdxColumn.allGather(idxQ.workspace);

    return topkIdxReplicated;
  }

  topkToSlots(state: ExecutionState, topkIdx: Tensor, kvTokenIndptrD: Tensor, pageIndices: Tensor, indptr: Tensor, lastPageLen: Tensor, batchIndices: Tensor, topkLength: Tensor, pageSize: number, maxKv: number, contextParallel?: boolean, _cpWorldSize?: number, _cpRank?: number): Tensor {
    const pTopkIdx = this.cast(topkIdx);
    const pKvTokenIndptr = this.cast(kvTokenIndptrD);
    const pPageIndices = this.cast(pageIndices);
    const pIndptr = this.cast(indptr);
    const pLastPageLen = this.cast(lastPageLen);
    const pBatchIndices = this.cast(batchIndices);
    const pTopkLength = this.cast(topkLength);

    const W = this.worldSize;
    const flatMode = this.shouldGatherKv(state);
    const cpW = flatMode ? 1 : (contextParallel ? W : 0);
    const totalQ = topkIdx.shape[0];
    const topk = topkIdx.shape[1];

    const slotShards: Tensor[] = [];
    for (let i = 0; i < W; i++) {
      const cpR = (cpW > 1) ? i : 0;
      slotShards.push(this.devices[i].topkToSlots(state, pTopkIdx.shards[i], pKvTokenIndptr.shards[i], pPageIndices.shards[i], pIndptr.shards[i], pLastPageLen.shards[i], pBatchIndices.shards[i], pTopkLength.shards[i], pageSize, maxKv, contextParallel, cpW, cpR));
    }
    return this.wrapShards(topkIdx.workspace, slotShards, [totalQ, topk], "I32", TensorParallelism.Replicated);
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
