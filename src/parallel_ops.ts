import { DeviceOps, MaskMode, notifySynchronizedWorkspaces, SlotSet, StridedMmap, TensorParallelism } from "./device_ops";
import { MemcpyKind } from "./enums";
import { ExecutionState } from "./execution-workspace";
import { Glm51Config } from "./glm51_model";
import { bf16BytesToF32, f32ToBf16Bytes, GlmOps, GlmTensor, NCCL_BFLOAT16, NCCL_FLOAT32, NCCL_INT32, NCCL_SUM, NCCL_UINT8 } from "./glm_ops";
import { getNativeAddon } from "./native-addon";
import { SafeTensorFile } from "./safetensors";
import { Tensor } from "./tensor";
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
// Fall back to the read-based (pull) CP merge; the push path is the default.
export const CP_MERGE_PULL = process.env.GLM_CP_MERGE_PULL === "1";
// Sort the CP top-k merge result ascending by global index. ON by default;
// GLM_CP_TOPK_SORT=0 disables it (diagnostic only -- see below).
//
// The gathered buffer is rank-major, so the merge emits positions ordered by
// (P % W, P / W) rather than by global position P. Sorting also makes this path
// bit-identical to the replicated-kData build, which is what makes `diff`
// against that build a usable regression test here.
//
// It is NOT only about parity: with the sort off, long generations degenerate
// into repeated literal "truncated" / "end of output" tokens once the context
// gets large. Sorting fixes that. Since the sort runs after selection it cannot
// change WHICH positions are selected, only their order -- so something
// downstream (topk_to_slots -> gatherTopkCkv -> sparse MLA) depends on the index
// list being ascending, beyond the float accumulation order it is allowed to
// depend on. That dependency has not been found, and this sort is currently
// masking it: any other producer of unsorted top-k indices would corrupt too.
// Costs ~9% of decode throughput (measured 100.4 -> 92.1 tok/s at 8-way CP).
export const CP_TOPK_SORT = process.env.GLM_CP_TOPK_SORT !== "0";

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

  same(other: Tensor): boolean {
    const o = other as ParallelTensor;
    if (this.shards.length !== o.shards.length) return false;
    for (let i = 0; i < this.shards.length; i++) {
      if (!this.shards[i].same(o.shards[i])) return false;
    }
    return true;
  }

  private get worldSize(): number {
    return this.shards.length;
  }

  private static elemBytes(type: string): number {
    // Single source of truth, shared with Tensor.bytes/byteCount. A local
    // switch here silently under-counted types it hadn't heard of (e.g. U32),
    // which would corrupt the sharded h2d/d2h byte math.
    return SafeTensorFile.dtypeBytes(type);
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
    const capturedShards = this.shards.map(s => s.capture());
    const captured = new ParallelTensor(this.workspace, this.parallelOps, this.parallelism, capturedShards, this.shape, this.type, undefined, this.pinned, undefined);
    (captured as { name: string | undefined }).name = this.name;
    captured.captured = true;
    return captured;
  }

  stage() {
    super.stage();
    for (const shard of this.shards) {
      shard.stage();
    }
  }

  unstage() {
    super.unstage();
    for (const shard of this.shards) {
      shard.unstage();
    }
  }

  _uncapture(): Tensor {
    const shards = this.shards.map(s => s.uncapture());
    return this.parallelOps.wrapShards(
      this.workspace, shards, this.shape, this.type, this.parallelism,
    );
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

  override reshape(newShape: number[], newType?: string): Tensor {
    const outType = newType ?? this.type;
    const currentBytes = Math.ceil(this.shape.reduce((a, b) => a * b, 1) * SafeTensorFile.dtypeBytes(this.type));
    const targetBytes = Math.ceil(newShape.reduce((a, b) => a * b, 1) * SafeTensorFile.dtypeBytes(outType));
    if (currentBytes !== targetBytes) {
      throw new Error(`reshape: cannot reshape [${this.shape}] (${this.type}, ${currentBytes} bytes) to [${newShape}] (${outType}, ${targetBytes} bytes)`);
    }
    if (newType && newType !== this.type && this.parallelism !== TensorParallelism.Replicated) {
      throw new Error(`reshape: type change (${this.type} → ${outType}) is only supported on Replicated tensors, got ${this.parallelism}`);
    }
    const newPar = ParallelTensor.computeReshapeParallelism(this.shape, newShape, this.parallelism, this.worldSize);
    const newShardShape = this.parallelOps.shardShape(newShape, newPar);
    const reshapedShards: Tensor[] = this.shards.map(s => s.reshape(newShardShape, newType));
    return this.parallelOps.wrapShards(
      this.workspace, reshapedShards, newShape, outType, newPar, this,
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
    if (process.env.GLM_P2P_ALLREDUCE === "0") return false;
    if (!this.parallelOps.p2pEnabled)
      return false;
    if (this.type !== "BF16" && this.type !== "F32")
      return false;
    const group = this.parallelOps.getP2PGroup(this.shards[0].workspace.glm.currentStream)!;
    if (!group)
      return false;

    const count = this.numElements;

    // all to all reduce: push-based reduce-scatter + gather (write+write).
    // Two write-only kernels with one barrier between: (1) scatter each GPU's N
    // chunks into peers' staging buffers, (2) reduce local staging and write the
    // result chunk back to all peers. All cross-PCIe traffic is posted writes
    // (no slow P2P reads), bandwidth-optimal at ~2N/GPU.
    //
    // No LEADING barrier is needed even though the scatter writes into peers'
    // staging: staging is allocated from the group's private workspace
    // (group.workspaces), which only barrier-bracketed P2P ops ever touch. An
    // address handed out this iteration was last used (and freed) after a prior
    // P2P barrier, so no peer can still be writing it. (Staging on the shared
    // caller workspace WOULD race a lagging peer's unrelated in-flight kernel on
    // the same recycled address -- that is the bug the private workspace fixes.)
    // Too big for the P2P group, or an uneven split, falls back to NCCL.
    if (count > 65536 * 2)
      return false;
    if (count % this.worldSize !== 0)
      return false;

    const elemBytes = ParallelTensor.elemBytes(this.type);
    const chunkLen = count / this.worldSize;
    const chunkBytes = chunkLen * elemBytes;
    const dtype = this.type === "F32" ? 7 : 9;
    const addon = getNativeAddon();

    // Per-GPU staging buffer laid out [worldSize, chunkLen]: slot j receives
    // GPU j's contribution to this rank's chunk.
    const staging: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      staging.push(group.workspaces[i].alloc(this.shape, this.type));
    }

    // Phase 1: scatter. GPU i writes chunk k to peer k's staging slot i. The
    // kernel rotates the peer order by rank; pointers pass in plain order.
    for (let i = 0; i < this.worldSize; i++) {
      const ptrs = new Array<number>(8).fill(0);
      for (let k = 0; k < this.worldSize; k++)
        ptrs[k] = staging[k].data;
      addon.p2pReduceScatterWrite(
        this.devices[i].ctx,
        this.shards[i].data,
        ptrs[0], ptrs[1], ptrs[2], ptrs[3],
        ptrs[4], ptrs[5], ptrs[6], ptrs[7],
        this.worldSize, chunkBytes, i,
      );
    }

    group.barrier(this.devices);

    // Phase 2: reduce local staging, write reduced chunk i back to all peers'
    // shards at offset i * chunkLen.
    for (let i = 0; i < this.worldSize; i++) {
      const ptrs = new Array<number>(8).fill(0);
      for (let k = 0; k < this.worldSize; k++)
        ptrs[k] = this.shards[k].data;
      addon.p2pReduceGatherWrite(
        this.devices[i].ctx,
        staging[i].data,
        ptrs[0], ptrs[1], ptrs[2], ptrs[3],
        ptrs[4], ptrs[5], ptrs[6], ptrs[7],
        this.worldSize, chunkLen, i, dtype,
      );
    }

    group.barrier(this.devices);
    group.cleanupSources();
    group.sources.push(...staging);

    return true;
  }

  allGather(workspace: WorkspaceBase): ParallelTensor {
    if (this.parallelism === TensorParallelism.Replicated) {
      return this;
    }
    if (this.parallelism === TensorParallelism.PartialSum) {
      throw new Error("allGather cannot be used on PartialSum tensors; use allReduce instead");
    }

    const output = workspace.alloc(this.shape, this.type, undefined, TensorParallelism.Replicated) as ParallelTensor;

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
    if (count > 65536 * 8)
      return false;
    const elemBytes = ParallelTensor.elemBytes(this.type);
    const shardBytes = count * elemBytes;
    const group = this.parallelOps.getP2PGroup(this.shards[0].workspace.glm.currentStream);
    if (!group)
      return false;

    const addon = getNativeAddon();

    // Guard before allocating so the unsupported path can't leak the shards.
    if (this.parallelism !== TensorParallelism.Column && this.parallelism !== TensorParallelism.Row) {
      throw new Error(`tryP2PAllGather: unsupported parallelism ${this.parallelism}`);
    }

    const shards: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      shards.push(group.workspaces[i].alloc(this.shape, this.type));
    }

    if (this.parallelism === TensorParallelism.Column) {
      const fullBytes = shardBytes * this.worldSize;
      // Write-based Column AllGather: each GPU writes its shard to all peers'
      // output buffers (concatenation along dim 0), then barrier ensures all
      // writes are visible. Uses same kernel as Row with outer=1.
      for (let i = 0; i < this.worldSize; ++i) {
        const rotatedPtrs = new Array<number>(8).fill(0);
        for (let k = 0; k < this.worldSize; k++) {
          rotatedPtrs[k] = shards[(i + k) % this.worldSize].data;
        }
        addon.p2pAllGatherRowWrite(
          this.devices[i].ctx,
          this.shards[i].data,
          rotatedPtrs[0], rotatedPtrs[1], rotatedPtrs[2], rotatedPtrs[3],
          rotatedPtrs[4], rotatedPtrs[5], rotatedPtrs[6], rotatedPtrs[7],
          shards[i].data, this.worldSize, shardBytes, fullBytes, 1, i,
        );
      }
      group.barrier(this.devices);

      for (let i = 0; i < this.worldSize; i++) {
        output.shards[i].memcpy(shards[i], shards[i].bytes, MemcpyKind.DeviceToDevice);
      }

      // due to the prior barrier, the staging buffers can immediately be recycled.
      // all peers are done writing to them.
      group.sources.push(...shards);
      group.cleanupSources();


      return true;
    }

    if (this.parallelism === TensorParallelism.Row) {
      const outer = this.shape[0];
      const inner = this.shape.slice(2).reduce((a, b) => a * b, 1);
      const shardDim1 = this.shape[1] / this.worldSize;
      const shardDim1Bytes = shardDim1 * inner * elemBytes;
      const fullDim1Bytes = this.shape[1] * inner * elemBytes;
      // Write-based Row AllGather: each GPU writes its shard to all peers'
      // output buffers, then barrier ensures all writes are visible.
      // Peer pointers are rotated by rank so all N GPUs don't target the
      // same peer's NVLink port simultaneously.
      for (let i = 0; i < this.worldSize; ++i) {
        const rotatedPtrs = new Array<number>(8).fill(0);
        for (let k = 0; k < this.worldSize; k++) {
          rotatedPtrs[k] = shards[(i + k) % this.worldSize].data;
        }
        addon.p2pAllGatherRowWrite(
          this.devices[i].ctx,
          this.shards[i].data,
          rotatedPtrs[0], rotatedPtrs[1], rotatedPtrs[2], rotatedPtrs[3],
          rotatedPtrs[4], rotatedPtrs[5], rotatedPtrs[6], rotatedPtrs[7],
          shards[i].data, this.worldSize, shardDim1Bytes, fullDim1Bytes, outer, i,
        );
      }
      group.barrier(this.devices);

      for (let i = 0; i < this.worldSize; i++) {
        output.shards[i].memcpy(shards[i], shards[i].bytes, MemcpyKind.DeviceToDevice);
      }

      group.cleanupSources();
      group.sources.push(...shards);

      return true;
    }

    throw new Error(`tryP2PAllGather: unsupported parallelism ${this.parallelism}`);
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
      const W = this.worldSize;
      if (n % W === 0 && batch <= W && pWeight.numElements > 1048576) {
        // these weights are better as column parallel for decode but replicated for prefill
        if (pWeight.name?.includes(".self_attn.q_a_proj.weight") || pWeight.name?.includes(".indexer.wq_b.weight") || pWeight.name?.includes(".ckv_proj.weight")) {
          using narrowed = pWeight.parallelOps.tryNarrowToColumnParallel(pWeight);
          if (narrowed) {
            return this.linear(narrowed);
          }
        }
        if (!pWeight.name?.includes(".mlp.gate.weight")) {
          console.log(`[shard-candidate] weight=${pWeight.name ?? "(unnamed)"} shape=[${n}, ${weight.shape[1]}] batch=${batch} n/W=${n / W} allgather=${n * batch * 2}B`);
        }
      }
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

    return this.elementwiseBinary(pOther, (a, b) => a.add(b, n));
  }

  mul(other: Tensor, n?: number): Tensor {
    return this.elementwiseBinary(other as ParallelTensor, (a, b) => a.mul(b, n));
  }

  private elementwiseBinary(pOther: ParallelTensor, op: (a: Tensor, b: Tensor) => Tensor): Tensor {
    if (this.parallelism === pOther.parallelism) {
      const outShards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        outShards.push(op(this.shards[i], pOther.shards[i]));
      }
      return this.parallelOps.wrapShards(this.workspace, outShards, this.shape, this.type, this.parallelism);
    }

    if (this.parallelism === TensorParallelism.PartialSum && pOther.parallelism === TensorParallelism.Replicated) {
      const outShards: Tensor[] = [];
      for (let i = 0; i < this.worldSize; i++) {
        outShards.push(op(this.shards[i], pOther.shards[i]));
      }
      return this.parallelOps.wrapShards(this.workspace, outShards, this.shape, this.type, TensorParallelism.PartialSum);
    }
    if (pOther.parallelism === TensorParallelism.PartialSum && this.parallelism === TensorParallelism.Replicated) {
      return pOther.elementwiseBinary(this, op);
    }

    if (this.parallelism === TensorParallelism.PartialSum) {
      using gathered = pOther.allGather(pOther.workspace);
      return this.elementwiseBinary(gathered as ParallelTensor, op);
    }
    if (pOther.parallelism === TensorParallelism.PartialSum) {
      using gathered = this.allGather(this.workspace);
      return pOther.elementwiseBinary(gathered as ParallelTensor, op);
    }

    if ((this.parallelism === TensorParallelism.Row || this.parallelism === TensorParallelism.Column) && pOther.parallelism === TensorParallelism.Replicated) {
      using gathered = this.allGather(this.workspace);
      return (gathered as ParallelTensor).elementwiseBinary(pOther, op);
    }
    if (this.parallelism === TensorParallelism.Replicated && (pOther.parallelism === TensorParallelism.Row || pOther.parallelism === TensorParallelism.Column)) {
      using gathered = pOther.allGather(pOther.workspace);
      return this.elementwiseBinary(gathered as ParallelTensor, op);
    }

    using gatheredThis = this.allGather(this.workspace);
    using gatheredOther = pOther.allGather(pOther.workspace);
    return (gatheredThis as ParallelTensor).elementwiseBinary(gatheredOther as ParallelTensor, op);
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
    if (this.parallelism === TensorParallelism.Row || this.parallelism === TensorParallelism.Column) {
      using gathered = this.allGather(this.workspace);
      return gathered.sigmoid();
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
  workspaces: WorkspaceBase[];

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

    this.workspaces = devices.map(d => new WorkspaceBase(d));
  }

  /**
   * Buffers peers are reading through this group's barriers. A P2P op pushes
   * the tensors its peers will read; they are only released once this group's
   * *next* barrier proves every peer is past those reads.
   *
   * Per-group (i.e. per-stream) on purpose: a barrier on one stream says
   * nothing about a P2P op still in flight on another, so a shared list would
   * let one stream's cleanup recycle buffers another stream's peers are still
   * reading.
   */
  sources: Tensor[] = [];

  /** Release the sources retained since the previous barrier on this group. */
  cleanupSources(): void {
    while (this.sources.length) {
      using _src = this.sources.pop()!;
    }
  }

  free(): void {
    this.cleanupSources();
    for (const ws of this.workspaces) {
      ws.free();
    }
    for (const inst of this.instances) {
      getNativeAddon().p2pDestroyInstance(inst);
    }
  }

  /** P2P barrier: sync all GPUs without data transfer. */
  barrier(devices: readonly GlmOps[], peerRanks?: number[]): void {
    this.arrive(devices, peerRanks);
    this.wait(devices, peerRanks);
  }

  /** Arrive phase: each rank publishes its flag (release). No spinning. */
  arrive(devices: readonly GlmOps[], peerRanks?: number[]): void {
    const addon = getNativeAddon();
    for (let i = 0; i < this.worldSize; ++i) {
      const peerRank = peerRanks ? peerRanks[i] : -1;
      addon.p2pArrive(devices[i].ctx, this.instances[i], peerRank);
    }
  }

  /** Wait phase: each rank spins on peers' flags (acquire). */
  wait(devices: readonly GlmOps[], peerRanks?: number[]): void {
    const addon = getNativeAddon();
    for (let i = 0; i < this.worldSize; ++i) {
      const peerRank = peerRanks ? peerRanks[i] : -1;
      addon.p2pWait(devices[i].ctx, this.instances[i], peerRank);
    }
  }
}

interface SparseMlaPrefetchExtra {
  stream?: Disposable & {
    result: ParallelTensor;
    streamWaitEvent(): void;
    synchronize(): void;
  };
  indexerStream?: Disposable & {
    result: ParallelTensor;
    streamWaitEvent(): void;
    synchronize(): void;
  };
}

export class ParallelOps implements DeviceOps {
  readonly devices: readonly GlmOps[];
  readonly worldSize: number;
  synchronizeListeners: WeakRef<WorkspaceBase>[] = [];
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

  /**
   * Retain `tensors` until the next barrier on the current stream's P2P group,
   * because peers are about to read them and must not see the memory recycled.
   */
  p2pRetainSources(...tensors: Tensor[]): void {
    const group = this.getP2PGroup(this.devices[0].currentStream);
    if (!group) throw new Error('P2P not available for source retention');
    group.sources.push(...tensors);
  }

  /** Release the sources retained by the current stream's group. */
  sourceCleanup() {
    this.getP2PGroup(this.devices[0].currentStream)?.cleanupSources();
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
    if (this.p2pEnabled && count <= 65536 * 8) {
      return CP_MERGE_PULL
        ? this.cpMergeTreeReduce(partialVOuts.shards, partialLses.shards, batchSize, numHeads, vHeadDim, workspace)
        : this.cpMergePushReduce(partialVOuts.shards, partialLses.shards, batchSize, numHeads, vHeadDim, workspace);
    }
    return this.agRsMerge(partialVOuts.shards, partialLses, batchSize, numHeads, vHeadDim, workspace);
  }

  /**
   * Push-based CP merge: scatter each peer's head slice into that peer's staging
   * buffer, barrier, then merge locally. Same Row-parallel result as
   * cpMergeTreeReduce, but every cross-device access is a posted write instead
   * of a blocking P2P read.
   *
   * Ordering, in the terms this file's other write-based collectives use:
   *
   * - No LEADING barrier. Peers write into `stage*`, which comes from the P2P
   *   group's private workspace, and the retention below keeps an address out
   *   of circulation until a barrier downstream of its last reader. Note this
   *   is the reason the pull version *does* need a leading barrier: there peers
   *   read the caller's partials, so the producer's attention kernel has to be
   *   proven complete first. Here nobody touches the partials remotely.
   * - TRAILING barrier before phase 2, which is what makes the local merge's
   *   reads of `stage*` valid.
   * - Retention is cleanup-then-push (one round). The merge reads `stage*`
   *   *after* the trailing barrier, so that barrier does not cover it; the
   *   first sync that does is the next collective's, which is exactly what one
   *   round of deferral waits for. Pushing before cleanup would free the
   *   staging immediately and let a peer's next collective overwrite it
   *   mid-merge.
   */
  private cpMergePushReduce(
    partialVOuts: readonly Tensor[],
    partialLses: readonly Tensor[],
    batchSize: number,
    numHeads: number,
    vHeadDim: number,
    workspace: WorkspaceBase,
  ): ParallelTensor {
    const group = this.getP2PGroup(this.devices[0].currentStream);
    if (!group) throw new Error("cpMergePushReduce: P2P group unavailable");

    const shardNHeads = this.shardDim(numHeads, "cpMergePushReduce shardNHeads");
    const shardWss = this.getShardWorkspaces(workspace);

    // Staging: [world, batch, shardNHeads, vHeadDim] BF16 + [world, batch,
    // shardNHeads] F32. Slot j receives rank j's contribution to this rank's
    // head slice. Total staging is batch*numHeads*vHeadDim -- the same size as
    // one rank's partial, so it fits the same P2P size gate as the caller.
    const stageV: Tensor[] = [];
    const stageLse: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      stageV.push(group.workspaces[i].alloc([this.worldSize * batchSize * shardNHeads, vHeadDim], "BF16"));
      stageLse.push(group.workspaces[i].alloc([this.worldSize * batchSize, shardNHeads], "F32"));
    }

    // Phase 1: scatter. Destination pointers are rotated by rank here; the
    // kernel rotates again by blockIdx.x so concurrent blocks target different
    // peers (see cp_merge_scatter_kernel).
    for (let i = 0; i < this.worldSize; i++) {
      const vPtrs = new Array<number>(8).fill(0);
      const lsePtrs = new Array<number>(8).fill(0);
      for (let k = 0; k < this.worldSize; k++) {
        const peer = (i + k) % this.worldSize;
        vPtrs[k] = stageV[peer].data;
        lsePtrs[k] = stageLse[peer].data;
      }
      this.devices[i].cpMergeScatter(
        partialVOuts[i], partialLses[i], vPtrs, lsePtrs,
        this.worldSize, batchSize, shardNHeads, vHeadDim, numHeads, numHeads, i,
      );
    }

    group.barrier(this.devices);

    // Phase 2: local online-softmax merge of the world_size staging slots.
    const outputV: Tensor[] = [];
    const outputLse: Tensor[] = [];
    for (let i = 0; i < this.worldSize; i++) {
      outputV.push(shardWss[i].alloc([batchSize, shardNHeads * vHeadDim], "BF16"));
      outputLse.push(shardWss[i].alloc([batchSize, shardNHeads], "F32"));
    }

    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].cpMergeLocal(
        stageV[i], stageLse[i], outputV[i], outputLse[i],
        this.worldSize, batchSize, shardNHeads, vHeadDim,
      );
    }

    group.cleanupSources();
    group.sources.push(
      ...stageV,
      ...stageLse,
    );

    // outputLse is local-only and unused downstream; any reuse of its address is
    // on this device's stream, hence ordered behind the merge kernel above.
    for (const lse of outputLse) {
      lse[Symbol.dispose]();
    }

    return this.wrapShards(workspace, outputV, [batchSize, numHeads * vHeadDim], "BF16", TensorParallelism.Row);
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
    this.p2pRetainSources(...partialVOuts.map(t => t.viewClone()), ...partialLses.map(t => t.viewClone()));

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

  [Symbol.dispose](): void {
    this.free();
  }

  /** P2P barrier: sync all GPUs without data transfer. */
  p2pBarrier(peerRanks?: number[]): void {
    const group = this.getP2PGroup(this.devices[0].currentStream);
    if (!group) throw new Error('P2P not available for barrier');
    group.barrier(this.devices, peerRanks);
  }

  /** Arrive phase of p2pBarrier: publish flags without waiting. */
  p2pArrive(peerRanks?: number[]): void {
    const group = this.getP2PGroup(this.devices[0].currentStream);
    if (!group) throw new Error('P2P not available for arrive');
    group.arrive(this.devices, peerRanks);
  }

  /** Wait phase of p2pBarrier: spin on peers' flags. */
  p2pWait(peerRanks?: number[]): void {
    const group = this.getP2PGroup(this.devices[0].currentStream);
    if (!group) throw new Error('P2P not available for wait');
    group.wait(this.devices, peerRanks);
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

    const pOut = outTokens as ParallelTensor;
    const pTopkVals = topkVals as ParallelTensor;
    const pTopkIdxs = topkIdxs as ParallelTensor;
    const pWorkspace = workspace as ParallelTensor;
    const pPenaltyTokens = penaltyTokens as ParallelTensor;
    const pPenaltyCount = penaltyCount as ParallelTensor;
    const pTemps = temperatures as ParallelTensor;
    const pRepPen = repPenalties as ParallelTensor;
    const pPresPen = presPenalties as ParallelTensor;
    const pTopKs = topKs as ParallelTensor;
    const pTopPs = topPs as ParallelTensor;
    const pStepCounter = stepCounter as ParallelTensor;
    if (pLogits.parallelism !== TensorParallelism.Replicated) {
      throw new Error(`sampleBatch logits: unsupported parallelism ${pLogits.parallelism}, expected ${TensorParallelism.Replicated}`);
    }
    if (pOut.shards.length !== this.worldSize) {
      throw new Error(`sampleBatch: outTokens has ${pOut.shards.length} shards, expected ${this.worldSize} (disposed by a borrower?)`);
    }
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].sampleBatch(pOut.shards[i], pTopkVals.shards[i], pTopkIdxs.shards[i], pWorkspace.shards[i], pLogits.shards[i], pPenaltyTokens.shards[i], pPenaltyCount.shards[i], maxWindow, vocabSize, batchSize, pTemps.shards[i], pRepPen.shards[i], pPresPen.shards[i], pTopKs.shards[i], pTopPs.shards[i], pStepCounter.shards[i], maxEffectiveK);
    }
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
    for (const group of this.p2pGroups.values()) {
      group.cleanupSources();
      for (const w of group.workspaces) {
        w.clearTracking();
      }
    }
    notifySynchronizedWorkspaces(this.synchronizeListeners);
  }

  async synchronizeAsync(): Promise<void> {
    await Promise.all(this.devices.map(device => device.synchronizeAsync()));
    for (const group of this.p2pGroups.values()) {
      group.cleanupSources();
      for (const w of group.workspaces) {
        w.clearTracking();
      }
    }
    notifySynchronizedWorkspaces(this.synchronizeListeners);
  }

  synchronizeStream(streamIdx: number): void {
    for (const device of this.devices) {
      device.synchronizeStream(streamIdx);
    }
    const group = this.getP2PGroup(streamIdx);
    if (group) {
      group.cleanupSources();
      for (const w of group.workspaces) {
        w.clearTracking();
      }
    }
  }

  async synchronizeStreamAsync(streamIdx: number): Promise<void> {
    await Promise.all(this.devices.map(device => device.synchronizeStreamAsync(streamIdx)));
    const group = this.getP2PGroup(streamIdx);
    if (group) {
      group.cleanupSources();
      for (const w of group.workspaces) {
        w.clearTracking();
      }
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
    let disposed = false;
    return {
      [Symbol.dispose]: () => {
        if (disposed)
          return;
        disposed = true;
        for (let i = 0; i < this.devices.length; i++) {
          this.devices[i].disposeStream(streams[i]!);
        }
      },
      streamWaitEvent: () => {
        // if (disposed)
        //   return;
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

  private cpLocalKvLen(globalKvLen: number, rank: number): number {
    return globalKvLen > rank ? Math.floor((globalKvLen - 1 - rank) / this.worldSize) + 1 : 0;
  }

  private adjustCpLastPageLen(lastPageLenH: ParallelTensor, batchSize: number, seqKvLens: number[], pageSize: number): void {
    const cpWorldSize = this.worldSize;
    const effectivePageSize = pageSize / cpWorldSize;
    for (let r = 0; r < cpWorldSize; r++) {
      lastPageLenH.shards[r].withPinnedBuffer(buf => {
        for (let s = 0; s < batchSize; s++) {
          const localKvLen = this.cpLocalKvLen(seqKvLens[s], r);
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
    this.adjustCpLastPageLen(pLastPageLenH, batchSize, seqKvLens, pageSize);
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

  mlaDecodePlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, indptrH: Tensor, lastPageLenH: Tensor, batchSize: number, numQoHeads: number, pageSize: number, enableCudaGraph: boolean, headDimCkv: number, headDimKpe: number, seqKvLens: number[], contextParallel?: boolean, _cpWorldSize?: number, _cpRank?: number): void {
    const pFloatWs = this.cast(floatWs);
    const pIntWs = this.cast(intWs);
    const pPinnedIntWs = this.cast(pinnedIntWs);
    const pPlanInfo = this.cast(planInfo);
    const pIndptrH = this.cast(indptrH);
    const pLastPageLenH = this.cast(lastPageLenH);
    const effectiveNumQoHeads = contextParallel ? numQoHeads : this.shardDim(numQoHeads, "mlaDecodePlan numQoHeads");
    const effectivePageSize = contextParallel ? pageSize / this.worldSize : pageSize;
    if (contextParallel) {
      this.adjustCpLastPageLen(pLastPageLenH, batchSize, seqKvLens, pageSize);
    }
    for (let i = 0; i < this.worldSize; i++) {
      const localSeqKvLens = contextParallel
        ? seqKvLens.map(length => this.cpLocalKvLen(length, i))
        : seqKvLens;
      this.devices[i].mlaDecodePlan(pFloatWs.shards[i], floatWsSize, pIntWs.shards[i], pPinnedIntWs.shards[i], intWsSize, pPlanInfo.shards[i], pIndptrH.shards[i], pLastPageLenH.shards[i], batchSize, effectiveNumQoHeads, effectivePageSize, enableCudaGraph, headDimCkv, headDimKpe, localSeqKvLens, contextParallel);
    }
  }

  sparseMlaDecodePlan(lastPageLenH: Tensor, batchSize: number, seqKvLens: number[], pageSize: number, contextParallel: boolean): void {
    if (contextParallel) {
      this.adjustCpLastPageLen(this.cast(lastPageLenH), batchSize, seqKvLens, pageSize);
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

  mlaKvCacheAppend(state: ExecutionState, cacheIdx: number, ckvData: Tensor, kpeData: Tensor | null, indices: Tensor, indptr: Tensor, lastPageLen: Tensor, appendCkv: Tensor, appendKpe: Tensor | null, batchIndices: Tensor, positions: Tensor, nnz: number, headDimCkv: number, headDimKpe: number, appendCkvStrideN: number, appendKpeStrideN: number, _pageSize?: number, _cpWorldSize?: number, _cpRank?: number): { ckv: Tensor; kpe?: Tensor } {
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
      const cache = this.devices[i].mlaKvCacheAppend(state, cacheIdx, pCkvData.shards[i], pKpeData?.shards[i] ?? null, pIndices.shards[i], pIndptr.shards[i], pLastPageLen.shards[i], pAppendCkv.shards[i], pAppendKpe?.shards[i] ?? null, pBatchIndices.shards[i], pPositions.shards[i], nnz, headDimCkv, headDimKpe, appendCkvStrideN, appendKpeStrideN, pageSize, effectiveCpWorldSize, effectiveCpRank);
      using _ckv = cache.ckv;
      using _kpe = cache.kpe;
    }

    const isIndexer = !pKpeData && !pAppendKpe && headDimKpe === 0;
    if (isIndexer && contextParallel) {
      const cfg = state.model.cfg as Glm51Config;
      const prefetchKey = `sparseMlaPrefetchLayer_${cacheIdx}`;
      using prefetchedStream = this.takeSparseMlaPrefetchStream(state, prefetchKey, "indexerStream");
      prefetchedStream?.streamWaitEvent();
      const prefetched = prefetchedStream?.result;

      if (prefetched) {
        const pKvTokenIndptr = this.cast(state.kvTokenIndptrD);
        for (let i = 0; i < this.worldSize; i++) {
          this.devices[i].indexerKvCacheAppendFlat(
            prefetched.shards[i], pAppendCkv.shards[i], pKvTokenIndptr.shards[i],
            pBatchIndices.shards[i], pPositions.shards[i], nnz, headDimCkv,
            appendCkvStrideN,
          );
        }
        return { ckv: prefetched };
      }

      if (this.shouldGatherKv(state, false)) {
        if (cacheIdx !== 0) {
          throw new Error(`mlaKvCacheAppend: expected indexer prefetch for full layer ${cacheIdx}`);
        }
        const pagedKV = state.cache.getPagedKV();
        return {
          ckv: this.gatherPages(
            ckvData, indices, indptr, lastPageLen,
            state.batchSize,
            pagedKV.maxPages * pagedKV.pageSize,
            state.kvTokenIndptrD, true,
          ),
        };
      }
    }
    return { ckv: pCkvData.viewClone(), kpe: pKpeData?.viewClone() };
  }

  sparseMlaPrepareCache(state: ExecutionState, groupSlots: Tensor, cacheIdx: number, kvCache: Tensor, appendCkv: Tensor, appendKpe: Tensor, topk: Tensor | undefined, indices: Tensor, indptr: Tensor, batchIndices: Tensor, positions: Tensor, nnz: number, kvLoraRank: number, peDim: number, appendCkvStrideN: number, appendKpeStrideN: number): Tensor {
    const pKvCache = this.cast(kvCache);
    const pBatchIndices = this.cast(batchIndices);
    const pPositions = this.cast(positions);
    const pAppendCkv = this.cast(appendCkv);
    const pAppendKpe = this.cast(appendKpe);
    const pageSize = pKvCache.shape[1];
    const pIndptr = this.cast(indptr);
    const pKvTokenIndptr = this.cast(state.kvTokenIndptrD);
    const cfg = state.model.cfg as Glm51Config;

    const pagedKV = state.cache.getPagedKV();

    // Get this layer's gathering stream before cleaning up a skipped previous
    // layer. Extras use exact layer keys; buffers use distance-based names.
    const prefetchDist = this.prefetchDistance(cacheIdx, cfg);
    const prefetchBufferKey = `sparseMlaPrefetch_${prefetchDist}`;
    const prefetchKey = `sparseMlaPrefetchLayer_${cacheIdx}`;
    using prefetchedStream = this.takeSparseMlaPrefetchStream(state, prefetchKey, "stream");
    prefetchedStream?.streamWaitEvent();

    // the previous layer may not be processed due to either incorrect usage (so being defensive here)
    // or because it is an mtp layer that was skipped.
    {
      const prevCacheIndex = (cacheIdx - 1 + pagedKV.ckvData.length) % pagedKV.ckvData.length;
      const prevKey = `sparseMlaPrefetchLayer_${prevCacheIndex}`;
      if (prevKey !== prefetchKey) {
        this.cleanupSparseMlaPrefetch(state, prevKey);
      }
    }

    // the gathering stream returns a tensor if its a full gather.
    // a sparse gather will NOT return a tensor, it must be read from the deterministically named tensor in the workspace.
    let prefetched = prefetchedStream?.result;

    const isSharedLayer = cfg.indexerTypes[cacheIdx] === "shared";
    if (isSharedLayer) {
      const isSparseGathering = this.shouldGatherKv(state, true);
      if (!prefetched && isSparseGathering) {
        // Sparse-gather decode: the full layer returned early (no extras
        // stream), so the gathered flat buffer lives as a named workspace
        // tensor materialized by topkToSlots' gatherGroupCkv loop
        // (ensureAlloc), not as a stream result in extras.
        prefetched = pIndptr.workspace.tensors.get(prefetchBufferKey) as ParallelTensor;
        if (!prefetched) {
          if (!groupSlots) {
            throw new Error(`sparseMlaPrepareCache: shared layer ${cacheIdx} has no group slots`);
          }
          const shardPageSize = pageSize / this.worldSize;
          const paddedKvLen = pagedKV.maxPages * pagedKV.pageSize;
          prefetched = pIndptr.workspace.ensureAlloc(
            [paddedKvLen / shardPageSize, shardPageSize, pKvCache.shape[2]],
            pKvCache.type,
            prefetchBufferKey,
          ) as ParallelTensor;
          this.gatherTopkCkv(
            state, kvCache, [prefetched], groupSlots, indices,
            pIndptr, state.kvTokenIndptrD, pBatchIndices,
            groupSlots.shape[1], paddedKvLen,
          );
          this.p2pBarrier();
        }
      }
    }
    else {
      const isFullGathering = this.shouldGatherKv(state, false);
      if (!prefetched && isFullGathering && cacheIdx) {
        throw new Error(`sparseMlaPrepareCache: expected prefetched result for full layer ${cacheIdx}`);
      }
    }

    if (prefetched) {
      // After a prefetch, this layer's NEW ckv values need to be written to
      // the flat gathered tensor: the prefetch started on the previous
      // layer's call and read from pagedKV.ckvData[cacheIdx] BEFORE this
      // layer's concatAndCacheDsMla ran, so positions [old_seq_len,
      // new_seq_len) are stale in the gathered buffer. concatAndCacheDsMla
      // writes them in. This is required in prefill (full gather) and decode (sparse gather)
      // because the topk may reference those new-token positions.

      for (let i = 0; i < this.worldSize; i++) {
        using _cache = this.devices[i].concatAndCacheDsMla(state, cacheIdx, prefetched!.shards[i], pAppendCkv.shards[i], pAppendKpe.shards[i],
          // prefetch tensor is flat, so no need for indices — and in that mode
          // the kernel indexes by kvTokenIndptr (the de-interleaved token prefix
          // sum the gather used), NOT the page indptr. They only coincide for
          // batch 0, so passing the page indptr corrupted the gathered buffer
          // for every forked sequence in MTP's draft passes.
          undefined,
          pKvTokenIndptr.shards[i], pBatchIndices.shards[i], pPositions.shards[i], nnz, kvLoraRank, peDim, appendCkvStrideN, appendKpeStrideN, pageSize, 0, 0);
      }
    }

    // if full gather path isn't in use, return whatever was found or fall back
    if (!this.shouldGatherKv(state, false)) {
      return prefetched?.viewClone() || kvCache.viewClone();
    }

    const contextParallel = pKvCache.parallelism === TensorParallelism.Row;

    const nextCacheIdx = cacheIdx + 1;
    // Do not carry a prefetched tensor across forwardModel's tracking scope.
    // The MTP layer runs in a separate forward and gathers its cache there.
    if (nextCacheIdx < cfg.numHiddenLayers && pagedKV.ckvData[nextCacheIdx]) {
      const nextKData = pagedKV.kData[nextCacheIdx];
      const indexerStream = nextKData?.parallelism === TensorParallelism.Row
        ? this.withStream<ParallelTensor>(() => this.gatherPages(
            nextKData, indices!, indptr, state.lastPageLen,
            state.batchSize,
            pagedKV.maxPages * pagedKV.pageSize,
            state.kvTokenIndptrD, true,
          ) as ParallelTensor)
        : undefined;
      const nextStream = this.withStream<ParallelTensor>(() => {
        const nextKvCache = pagedKV.ckvData[nextCacheIdx];

        return this.gatherPages(
          nextKvCache, indices!, indptr, state.lastPageLen,
          state.batchSize,
          pagedKV.maxPages * pagedKV.pageSize,
          state.kvTokenIndptrD, contextParallel,
        ) as ParallelTensor;
      });

      // due to mtp usage being potentially dynamic (mtp or incorrect usage), only clean up after the layer is finished and before a prefetch overwrites.
      {
        const nextKey = `sparseMlaPrefetchLayer_${nextCacheIdx}`;
        this.cleanupSparseMlaPrefetch(state, nextKey);
      }

      const nextExtra: SparseMlaPrefetchExtra = { stream: nextStream, indexerStream };
      state.ws.extras.set(`sparseMlaPrefetchLayer_${nextCacheIdx}`, nextExtra);
    }

    if (prefetched) {
      // console.log('prefetched', cacheIdx);
      return prefetched;
    }

    // no prefetch was available, so gather the pages now (layer 0).
    return this.gatherPages(
      kvCache, indices, indptr, state.lastPageLen,
      state.batchSize,
      pagedKV.maxPages * pagedKV.pageSize,
      state.kvTokenIndptrD, contextParallel,
    );
  }

  concatAndCacheDsMla(state: ExecutionState, cacheIdx: number, kvCache: Tensor, appendCkv: Tensor, appendKpe: Tensor, indices: Tensor | undefined, indptr: Tensor, batchIndices: Tensor, positions: Tensor, nnz: number, kvLoraRank: number, peDim: number, appendCkvStrideN: number, appendKpeStrideN: number): Tensor {
    const pKvCache = this.cast(kvCache);
    const pAppendCkv = this.cast(appendCkv);
    const pAppendKpe = this.cast(appendKpe);
    const pIndices = indices ? this.cast(indices) : null;
    const pIndptr = this.cast(indptr);
    const pBatchIndices = this.cast(batchIndices);
    const pPositions = this.cast(positions);
    const contextParallel = pKvCache.parallelism === TensorParallelism.Row;
    const cpWorldSize = contextParallel ? this.worldSize : 0;
    const pageSize = pKvCache.shape[1];

    for (let i = 0; i < this.worldSize; i++) {
      const cpRank = contextParallel ? i : 0;
      using _cache = this.devices[i].concatAndCacheDsMla(state, cacheIdx, pKvCache.shards[i], pAppendCkv.shards[i], pAppendKpe.shards[i], pIndices?.shards[i], pIndptr.shards[i], pBatchIndices.shards[i], pPositions.shards[i], nnz, kvLoraRank, peDim, appendCkvStrideN, appendKpeStrideN, pageSize, cpWorldSize, cpRank);
    }
    return pKvCache.viewClone();
  }

  appendSelectedMtpCaches(mlaSrcCkvPtrs: Tensor, mlaSrcKpePtrs: Tensor, mlaDstCkvPtrs: Tensor, mlaDstKpePtrs: Tensor | undefined,
    indexerSrcPtrs: Tensor | undefined, indexerDstPtrs: Tensor | undefined,
    sourceRows: Tensor, indices: Tensor, indptr: Tensor, batchIndices: Tensor, positions: Tensor,
    pageSize: number, kvLoraRank: number, peDim: number, indexHeadDim: number, sparseMode: boolean,
    cpWorldSize: number = 0, _cpRank: number = 0): void {
    const required = [mlaSrcCkvPtrs, mlaSrcKpePtrs, mlaDstCkvPtrs, sourceRows, indices, indptr, batchIndices, positions].map(tensor => this.cast(tensor));
    const pMlaDstKpePtrs = mlaDstKpePtrs ? this.cast(mlaDstKpePtrs) : undefined;
    const pIndexerSrcPtrs = indexerSrcPtrs ? this.cast(indexerSrcPtrs) : undefined;
    const pIndexerDstPtrs = indexerDstPtrs ? this.cast(indexerDstPtrs) : undefined;
    for (const tensor of [...required, pMlaDstKpePtrs, pIndexerSrcPtrs, pIndexerDstPtrs]) {
      if (tensor) this.assertParallel("appendSelectedMtpCaches", tensor, TensorParallelism.Replicated);
    }
    for (let rank = 0; rank < this.worldSize; rank++) {
      this.devices[rank].appendSelectedMtpCaches(
        required[0].shards[rank], required[1].shards[rank], required[2].shards[rank], pMlaDstKpePtrs?.shards[rank],
        pIndexerSrcPtrs?.shards[rank], pIndexerDstPtrs?.shards[rank],
        required[3].shards[rank], required[4].shards[rank], required[5].shards[rank], required[6].shards[rank], required[7].shards[rank],
        pageSize, kvLoraRank, peDim, indexHeadDim, sparseMode, cpWorldSize, cpWorldSize > 0 ? rank : 0,
      );
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

  // Distance from the last full layer: full layers return 0, shared layers
  // return the number of consecutive shared layers since the last full one.
  // Used as the workspace buffer key so layers at the same position within
  // their group reuse the same allocation (4 unique keys instead of 78).
  private prefetchDistance(cacheIdx: number, cfg: Glm51Config): number {
    let d = 0;
    for (let i = cacheIdx; i >= 0 && cfg.indexerTypes[i] === "shared"; i--) d++;
    return d;
  }

  private takeSparseMlaPrefetchStream(state: ExecutionState, key: string, field: "stream" | "indexerStream") {
    const extra = state.ws.extras.get(key) as SparseMlaPrefetchExtra | undefined;
    const stream = extra?.[field];
    if (!extra || !stream) return undefined;
    delete extra[field];
    if (!extra.stream && !extra.indexerStream) {
      state.ws.extras.delete(key);
    }
    return stream;
  }

  private cleanupSparseMlaPrefetch(state: ExecutionState, key: string): void {
    const extra = state.ws.extras.get(key) as SparseMlaPrefetchExtra | undefined;
    if (!extra) return;
    state.ws.extras.delete(key);
    for (const stream of [extra.indexerStream, extra.stream]) {
      if (!stream) continue;
      stream.streamWaitEvent();
      stream.result[Symbol.dispose]();
      stream[Symbol.dispose]();
    }
  }

  // determines the gather type to be used depending on the state.
  // this is called at various states in the pipeline for hooking a all vs sparse gather
  // "full" layers should never be sparse gathered. (enforced elsewhere)
  shouldGatherKv(state: ExecutionState, sparseGather: boolean) {
    // only valid in cp mode
    if (!state.cache.getPagedKV().contextParallel)
      return false;
    // force it off if requested
    if (!CP_GATHER_KV)
      return false;

    // if total tokens is under some threshold, use the sparse gather.
    if (state.totalTokens <= 32) {
      // decode should only sparse gather.
      return sparseGather;
    }

    // prevent high batch decode from using the gather path
    if (state.isDecode) {
      return false;
    }

    // never sparse gather above the threshold
    if (sparseGather)
      return false;

    // should do the actual math to see whether q or ckv has a smaller gather
    // this could be tuned further.
    const paddedKvLen = state.getGraphVariantPaddedKvLen();
    if (paddedKvLen > 65536 * 4)
      return false;

    return true;
  }

  sparseMlaPrefill(state: ExecutionState, qAbsorbed: Tensor, qPe: Tensor, kvCache: Tensor, indices: Tensor, topk: number, smScale: number, topkLength: Tensor, pageIndptrD: Tensor, lastPageLen: Tensor, kvTokenIndptrD: Tensor): { o: Tensor, lse: Tensor } {
    const numTokens = state.totalTokens;
    const pQAbsorbed = this.cast(qAbsorbed);
    const pQPe = this.cast(qPe);
    const pKvCache = this.cast(kvCache);
    const pIndices = this.cast(indices);
    const pTopkLength = this.cast(topkLength);
    const contextParallel = pKvCache.parallelism === TensorParallelism.Row;
    const numHeads = pQAbsorbed.shape[1];
    const headDim = pQAbsorbed.shape[2];

    const pEffKvCache = this.cast(kvCache);
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

  gatherTopkCkv(state: ExecutionState, kvCache: Tensor, outputs: readonly Tensor[], topkIdx: Tensor, pageIndices: Tensor, pageIndptr: Tensor, kvTokenIndptr: Tensor, batchIndices: Tensor, topk: number, paddedKvLen: number, _cpWorldSize?: number, _cpRank?: number, _effPageSize?: number): void {
    // ParallelOps callsite contract: outputs contains exactly ONE ParallelTensor
    // whose N shards are the per-rank flat output buffers (pre-allocated by
    // the caller). Internally we extract those shards and use them as the
    // peer table passed to each per-rank GlmOps.gatherTopkCkv invocation.
    //
    // CP-only: the whole point of gatherTopkCkv is the cross-rank P2P fan-out,
    // which only makes sense when each rank holds a disjoint slice of the KV
    // cache (kvCache is Row-parallel). Non-CP callers should use gatherPages
    // or invoke GlmOps.gatherTopkCkv directly — ParallelOps has no useful work
    // to coordinate when the cache is Replicated.
    if (outputs.length !== 1) {
      throw new Error(`ParallelOps.gatherTopkCkv: expected outputs.length===1 (single ParallelTensor), got ${outputs.length}`);
    }
    const pKvCache = this.cast(kvCache);
    const pOut = this.cast(outputs[0]);
    const pTopkIdx = this.cast(topkIdx);
    const pPageIndices = this.cast(pageIndices);
    const pPageIndptr = this.cast(pageIndptr);
    const pKvTokenIndptr = this.cast(kvTokenIndptr);
    const pBatchIndices = this.cast(batchIndices);
    if (pKvCache.parallelism !== TensorParallelism.Row) {
      throw new Error(`ParallelOps.gatherTopkCkv: kvCache.parallelism=${pKvCache.parallelism}, expected Row (CP). Non-CP is not supported — use gatherPages or call GlmOps.gatherTopkCkv directly.`);
    }
    if (pOut.shards.length !== this.worldSize) {
      throw new Error(`ParallelOps.gatherTopkCkv: output ParallelTensor has ${pOut.shards.length} shards, expected worldSize=${this.worldSize}`);
    }
    const pageSize = pKvCache.shape[1];
    // Under CP shardPageSize = pageSize / worldSize, matching gatherPages' CP
    // output convention so the downstream sparse MLA kernel can read with
    // effPageSize=shardPageSize. Per-shard shape/type validation is left to
    // the GlmOps delegate (which sees each peer shard in its `outputs` array
    // and validates them on every per-rank invocation).
    const shardPageSize = pageSize / this.worldSize;
    if (paddedKvLen % shardPageSize !== 0) {
      throw new Error(`ParallelOps.gatherTopkCkv: paddedKvLen=${paddedKvLen} not divisible by shardPageSize=${shardPageSize}`);
    }

    // kvCache is Row-parallel (each rank stores every Nth token). Each
    // rank reads ONLY the topk positions that live on it (pos % worldSize ==
    // cpRank) and fan-out writes those BPT-byte tokens to all N peer output
    // shards at flat_slot = kvTokenIndptr[seq] + token_pos. After all N
    // per-shard invocations every output shard holds the full topk union.
    // Per-shard GlmOps call with the FULL N-shard peer table (N=worldSize),
    // cpWorldSize=worldSize, cpRank=i, effPageSize=shardPageSize.
    //
    // The peer table is ROTATED by rank: rank i is handed
    // [shard_i, shard_i+1, ..., shard_N-1, shard_0, ...]. The kernel walks its
    // peer array in order, so unrotated every rank would target shard 0 first,
    // shard 1 second, and so on — all N sources contending for one destination
    // link per phase. Rotated, step t is the permutation i -> (i+t) % N, so each
    // destination receives from exactly one source at a time. The kernel only
    // uses the peer index to pick a destination base pointer (no self-case, no
    // rank-derived math), so the rotation is transparent to it — cpRank is
    // passed separately and stays the true rank for the CP ownership filter.
    for (let i = 0; i < this.worldSize; i++) {
      const rotatedPeers = [...pOut.shards.slice(i), ...pOut.shards.slice(0, i)];
      this.devices[i].gatherTopkCkv(
        state,
        pKvCache.shards[i],
        rotatedPeers,
        pTopkIdx.shards[i],
        pPageIndices.shards[i], pPageIndptr.shards[i],
        pKvTokenIndptr.shards[i], pBatchIndices.shards[i],
        topk, paddedKvLen,
        this.worldSize, i, shardPageSize,
      );
    }

  }

  indexerScore(out: Tensor, q: Tensor, kData: Tensor, weights: Tensor, pageIndices: Tensor, pageIndptr: Tensor, lastPageLen: Tensor, qoIndptr: Tensor, scale: number, totalQ: number, idxNHeads: number, idxHeadDim: number, pageSize: number, maxKvLen: number, causal: boolean, kvTokenIndptr?: Tensor): void {
    const pOut = this.cast(out);
    const pQ = this.cast(q);
    const pKData = this.cast(kData);
    const pWeights = this.cast(weights);
    const pIndices = this.cast(pageIndices);
    const pIndptr = this.cast(pageIndptr);
    const pLastPageLen = this.cast(lastPageLen);
    const pQoIndptr = this.cast(qoIndptr);
    const pKvTokenIndptr = kvTokenIndptr ? this.cast(kvTokenIndptr) : undefined;
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].indexerScore(pOut.shards[i], pQ.shards[i], pKData.shards[i], pWeights.shards[i], pIndices.shards[i], pIndptr.shards[i], pLastPageLen.shards[i], pQoIndptr.shards[i], scale, totalQ, idxNHeads, idxHeadDim, pageSize, maxKvLen, causal, pKvTokenIndptr?.shards[i]);
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
  indexerTopk(state: ExecutionState, idxQ: Tensor, kData: Tensor, weights: Tensor, pageIndices: Tensor, indptr: Tensor, lastPageLen: Tensor, qoIndptr: Tensor, scale: number, topk: number, decode: boolean, qGlobalStart?: number, customMask?: Tensor, maskIndptr?: Tensor, maskKvLen?: Tensor): { values: Tensor, indices: Tensor } {
    const totalQ = idxQ.shape[0];
    using pQ = idxQ.parallelism === TensorParallelism.Replicated ? idxQ.viewClone() as ParallelTensor : this.cast(idxQ).allGather(idxQ.workspace);
    using pKData = this.cast(kData).viewClone() as ParallelTensor;
    const pWeights = this.cast(weights);
    const pPageIndices = this.cast(pageIndices);
    const pIndptr = this.cast(indptr);
    const kIsRow = pKData.parallelism === TensorParallelism.Row;
    const effectiveLastPageLen = kIsRow ? state.lastPageLen : lastPageLen;
    const pLastPageLen = this.cast(effectiveLastPageLen);
    const pGlobalLastPageLen = this.cast(state.globalLastPageLen);
    const pQoIndptr = this.cast(qoIndptr);
    const pCustomMask = customMask ? this.cast(customMask) : undefined;
    const pMaskIndptr = maskIndptr ? this.cast(maskIndptr) : undefined;
    const pMaskKvLen = maskKvLen ? this.cast(maskKvLen) : undefined;
    const flatKData = state.cache.getPagedKV().contextParallel && pKData.parallelism === TensorParallelism.Replicated;
    const pKvTokenIndptr = flatKData ? this.cast(state.kvTokenIndptrD) : undefined;

    const W = this.worldSize;
    const kDataReplicated = pKData.parallelism === TensorParallelism.Replicated;
    using colIdxQ = this.tryNarrowToColumnParallel(pQ);
    using colWeights = this.tryNarrowToColumnParallel(pWeights);
    const canShard = !decode && W > 1 && kDataReplicated
      && pQoIndptr.shards[0].shape[0] === 2
      && colIdxQ && colWeights
      && totalQ % W === 0;

    if (!canShard) {
      const topkIdxShards: Tensor[] = [];
      const topkValShards: Tensor[] = [];
      for (let i = 0; i < W; i++) {
        const r = this.devices[i].indexerTopk(state, pQ.shards[i], pKData.shards[i], pWeights.shards[i], pPageIndices.shards[i], pIndptr.shards[i], pLastPageLen.shards[i], pQoIndptr.shards[i], scale, topk, decode, qGlobalStart ?? 0, pCustomMask?.shards[i], pMaskIndptr?.shards[i], pMaskKvLen?.shards[i], kIsRow ? W : 0, kIsRow ? i : 0, pGlobalLastPageLen.shards[i], pKvTokenIndptr?.shards[i]);
        topkIdxShards.push(r.indices);
        topkValShards.push(r.values);
      }
      if (!kIsRow) {
        return {
          indices: this.wrapShards(idxQ.workspace, topkIdxShards, [totalQ, topk], "I32", TensorParallelism.Replicated),
          values: this.wrapShards(idxQ.workspace, topkValShards, [totalQ, topk], "BF16", TensorParallelism.Replicated),
        };
      }

      if (true) {
        // gather indices/values individually here, and then do topk merge
        using localValues = this.wrapShards(idxQ.workspace, topkValShards, [totalQ, topk * W], "BF16", TensorParallelism.Row);
        using localIndices = this.wrapShards(idxQ.workspace, topkIdxShards, [totalQ, topk * W], "I32", TensorParallelism.Row);
        using gatheredValues = localValues.allGather(idxQ.workspace);
        using gatheredIndices = localIndices.allGather(idxQ.workspace);
        const kTotal = topk * W;
        const { values: mergedValues, indices: mergedIndices } = gatheredValues.topk(topk, kTotal);
        using _mergedIndices = mergedIndices as ParallelTensor;
        const finalIndices = gatheredIndices.gather(mergedIndices, topk, kTotal, totalQ);

        // Opt-in: restore ascending-index order so this path is bit-identical to
        // the replicated-kData build. Off by default -- see CP_TOPK_SORT.
        if (CP_TOPK_SORT) {
          const pFinalIndices = this.cast(finalIndices);
          const pMergedValues = this.cast(mergedValues);
          for (let i = 0; i < W; i++) {
            this.devices[i].sortTopkByIndex(pFinalIndices.shards[i], pMergedValues.shards[i], totalQ, topk);
          }
        }
        return { values: mergedValues, indices: finalIndices };
      }
    }

    // Query-sharded path: each rank processes totalQ/W query rows.
    const localQ = totalQ / W;
    const topkIdxShards: Tensor[] = [];
    const topkValShards: Tensor[] = [];
    for (let i = 0; i < W; i++) {
      const qStart = i * localQ;
      const r = this.devices[i].indexerTopk(
        state,
        colIdxQ!.shards[i], pKData.shards[i], colWeights!.shards[i],
        pPageIndices.shards[i], pIndptr.shards[i], pLastPageLen.shards[i],
        pQoIndptr.shards[i],
        scale, topk,
        decode, qStart,
        pCustomMask?.shards[i], pMaskIndptr?.shards[i], pMaskKvLen?.shards[i],
        kIsRow ? W : 0, kIsRow ? i : 0, pGlobalLastPageLen.shards[i], pKvTokenIndptr?.shards[i],
      );
      topkIdxShards.push(r.indices);
      topkValShards.push(r.values);
    }

    // Column [totalQ, topk] → AllGather → Replicated [totalQ, topk].
    using topkIdxColumn = this.wrapShards(idxQ.workspace, topkIdxShards, [totalQ, topk], "I32", TensorParallelism.Column);
    using topkValColumn = this.wrapShards(idxQ.workspace, topkValShards, [totalQ, topk], "BF16", TensorParallelism.Column);
    const topkIdxReplicated = topkIdxColumn.allGather(idxQ.workspace);
    // const topkValReplicated = topkValColumn.allGather(idxQ.workspace);

    return { values: topkValColumn, indices: topkIdxReplicated };
  }

  // Sort each top-k row ascending by index, in place, on every shard. The
  // shards hold identical Replicated copies, so sorting each one keeps them so.
  sortTopkByIndex(indices: Tensor, values: Tensor, batch: number, topk: number): void {
    const pIndices = this.cast(indices);
    const pValues = this.cast(values);
    for (let i = 0; i < this.worldSize; i++) {
      this.devices[i].sortTopkByIndex(pIndices.shards[i], pValues.shards[i], batch, topk);
    }
  }

  // The flat/paged addressing is resolved internally from `cacheIdx` (via
  // topkSlotMode) so the model / ExecutionState never see it:
  //   "flat": de-interleaved slot (kvTokenIndptr[seq] + token_pos), identical on
  //   every rank (cpW=1) — for the gathered/replicated CKV buffer. "paged": the
  //   per-rank CP shard (cpW=W, cpR=i) with the ownership filter, or non-CP
  //   (cpW=0).
  topkToSlots(state: ExecutionState, topkIdx: Tensor, kvTokenIndptrD: Tensor, pageIndices: Tensor, indptr: Tensor, lastPageLen: Tensor, batchIndices: Tensor, pageSize: number, maxKv: number, cacheIdx: number, contextParallel?: boolean, _cpWorldSize?: number, _cpRank?: number): { layer: SlotSet, group: SlotSet, stream?: ReturnType<ParallelOps["withStream"]> } {
    const pTopkIdx = this.cast(topkIdx);
    const pKvTokenIndptr = this.cast(kvTokenIndptrD);
    const pPageIndices = this.cast(pageIndices);
    const pIndptr = this.cast(indptr);
    const pLastPageLen = this.cast(lastPageLen);
    const pBatchIndices = this.cast(batchIndices);

    const W = this.worldSize;
    const totalQ = topkIdx.shape[0];
    const topk = topkIdx.shape[1];
    const maxQ = state.positionIds.shape[0];

    // One SlotSet in the addressing `modeCacheIdx` requires. The per-shard call
    // has no group of its own — its viewClone is discarded here, and must be,
    // or it would keep the layer's shards from being recycled on schedule.
    const computeSet = (modeCacheIdx: number): SlotSet => {
      const cpW = this.topkSlotMode(state, modeCacheIdx) === "flat" ? 1 : (contextParallel ? W : 0);
      // One Replicated length allocated up front, its shards handed down — the
      // per-device alloc order has to stay exactly as it was, or the retained
      // slots block rotates through the recycle pool and graph replay breaks.
      const length = this.cast(topkIdx.workspace.alloc([maxQ], "I32"));
      const slotShards: Tensor[] = [];
      for (let i = 0; i < W; i++) {
        const cpR = (cpW > 1) ? i : 0;
        // cacheIdx is unused at the device level (mode already resolved to cpW/cpR).
        const leaf = this.devices[i].topkToSlots(state, pTopkIdx.shards[i], pKvTokenIndptr.shards[i], pPageIndices.shards[i], pIndptr.shards[i], pLastPageLen.shards[i], pBatchIndices.shards[i], pageSize, maxKv, -1, contextParallel, cpW, cpR, length.shards[i]);
        // The per-shard call has no group of its own; its viewClones must be
        // released or they keep the layer's shards from recycling on schedule.
        leaf.group.slots[Symbol.dispose]();
        leaf.group.length[Symbol.dispose]();
        slotShards.push(leaf.layer.slots);
      }
      return {
        slots: this.wrapShards(topkIdx.workspace, slotShards, [totalQ, topk], "I32", TensorParallelism.Replicated),
        length,
      };
    };

    const cfg = state.model.cfg as Glm51Config;
    const pagedKV = state.cache.getPagedKV();
    const groupIdx = cacheIdx + 1;
    const hasGroup = cfg.indexerTypes[groupIdx] === "shared" && !!pagedKV.ckvData[groupIdx];
    // Only decode sparse-gather splits the two: the full layer reads its paged
    // CP shard while the group reads the gathered flat buffer. Everywhere else
    // one buffer serves both, and computing a second identical copy would put
    // another [maxKv, topk] block into the recycle rotation — which moves the
    // slots address step to step and breaks graph replay.
    const diverges = hasGroup && this.topkSlotMode(state, groupIdx) !== this.topkSlotMode(state, cacheIdx);

    // `group` is always the parent allocation and, when shared, `layer` is the
    // clone — the caller `using`s the layer per-layer while the group survives
    // into the following shared layers and the MTP passes.
    //
    // Allocate the group FIRST. It is the set held across the pass boundary, so
    // it has to take its block from a fixed position in the recycle rotation;
    // letting the layer-local set (freed at end of layer) go first makes the
    // retained block rotate step to step, which breaks graph replay.
    let layer: SlotSet;
    let group: SlotSet;
    let stream: ReturnType<ParallelOps["withStream"]> | undefined;
    if (diverges) {
      group = computeSet(groupIdx);
      stream = this.shouldGatherKv(state, true)
        ? this.withStream(() => this.gatherGroupCkv(state, cacheIdx, group.slots, pageIndices, pIndptr, pBatchIndices))
        : undefined;
      layer = computeSet(cacheIdx);
    } else {
      // Same addressing — cacheIdx resolves it either way, and never indexes
      // indexerTypes past the end when no group follows.
      group = computeSet(cacheIdx);
      layer = { slots: group.slots.viewClone(), length: group.length.viewClone() };
      stream = hasGroup && this.shouldGatherKv(state, true)
        ? this.withStream(() => this.gatherGroupCkv(state, cacheIdx, group.slots, pageIndices, pIndptr, pBatchIndices))
        : undefined;
    }

    return { layer, group, stream };
  }

  // Fan this rank's CKV for every shared layer in `cacheIdx`'s group out to all
  // peers' flat buffers, driven by the group's slots. Decode sparse-gather only.
  private gatherGroupCkv(state: ExecutionState, cacheIdx: number, groupSlots: Tensor, indices: Tensor, pIndptr: ParallelTensor, pBatchIndices: ParallelTensor) {
    const cfg = state.model.cfg as Glm51Config;
    const pagedKV = state.cache.getPagedKV();
    const pKvCache = this.cast(pagedKV.ckvData[cacheIdx]);
    const pageSize = pKvCache.shape[1];
    const paddedKvLen = pagedKV.maxPages * pagedKV.pageSize;
    const pSlots = this.cast(groupSlots);
    const topkCount = pSlots.shape[1];

    for (let i = 1; ; i++) {
      const nextCacheIdx = cacheIdx + i;
      if (!pagedKV.ckvData[nextCacheIdx]) break;
      if (cfg.indexerTypes[nextCacheIdx] !== "shared") break;

      const nextKvCache = pagedKV.ckvData[nextCacheIdx];
      const shardPageSize = pageSize / this.worldSize;
      const BPT = pKvCache.shape[2];
      const out = pIndptr.workspace.ensureAlloc(
        [paddedKvLen / shardPageSize, shardPageSize, BPT],
        pKvCache.type,
        `sparseMlaPrefetch_${this.prefetchDistance(nextCacheIdx, cfg)}`,
      ) as ParallelTensor;

      this.gatherTopkCkv(
        state, nextKvCache, [out], pSlots, this.cast(indices),
        pIndptr, state.kvTokenIndptrD, pBatchIndices, topkCount, paddedKvLen,
      );
    }

    this.p2pBarrier();
  }

  // Which addressing layer `cacheIdx`'s CKV buffer (as returned by
  // sparseMlaPrepareCache) needs — the single source of truth shared with that
  // method's return-value logic:
  //   full-gather (prefill)   -> every layer reads a gathered flat buffer.
  //   sparse-gather (decode)  -> 'full' layers read their paged CP shard;
  //                              'shared' layers read the gathered flat buffer.
  //   no gather               -> every layer reads its paged CP shard.
  topkSlotMode(state: ExecutionState, cacheIdx: number): "flat" | "paged" {
    if (this.shouldGatherKv(state, false))
      return "flat";
    if (this.shouldGatherKv(state, true)) {
      const cfg = state.model.cfg as Glm51Config;
      return cfg.indexerTypes[cacheIdx] === "shared" ? "flat" : "paged";
    }
    return "paged";
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
