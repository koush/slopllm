import { Allocator, ArenaAllocator } from "./allocator";
import { DeviceOps, MaskMode, notifySynchronizedWorkspaces, SlotSet, StridedMmap, TensorParallelism } from "./device_ops";
import type { ExecutionState } from "./execution-workspace";
import { SafeTensorFile } from "./safetensors";
import { Tensor } from "./tensor";
import type { WorkspaceBase } from "./workspace";
import { MemcpyKind } from "./enums";
import { getNativeAddon } from "./native-addon";

// Above `count == topK` (single-token decode), mulMatId can either:
//  - run the direct per-(token,expert) GEMV kernel (mulMatId/nvfp4MulMatId), which
//    re-reads each expert's weight once per routed token (redundant when several
//    tokens share an expert, but a single dependency-free kernel launch), or
//  - run the grouped/sorted pipeline (bf16MulMatIdGroupedMma/nvfp4MulMatIdGroupedMmaCoop), which
//    reads each expert's weight once regardless of how many tokens route to it, at
//    the cost of a 6-stage histogram/scatter/gemv/unscatter dependency chain.
// The grouped pipeline's per-stage overhead dominates at small counts (e.g. MTP
// verification batches), so the direct path wins there despite redundant reads;
// the grouped path only pays off once `count` is large enough to amortize its
// dispatch overhead against avoided redundant weight reads (true prefill territory).
const MUL_MAT_ID_GROUPED_THRESHOLD = 512;

// Below this query-token count, sparse MLA prefill is routed to the split-K
// decode kernel for better GPU occupancy (e.g. MTP tree verify). Above it, the
// prefill kernel's per-token CTAs already fill the GPU and amortize KV loads.
// Tunable — the crossover is roughly the SM count divided by heads/HPB.
const SPARSE_MLA_DECODE_DISPATCH_MAX = Number(process.env.GLM_SPARSE_DECODE_DISPATCH_MAX ?? 64);

// At or below this query-token count the indexer scores via the "direct" v2 path
// (simple per-position score kernel + a single 65536-bucket histogram): lowest
// per-launch overhead and best occupancy for small Q, since the tensor-core
// prefill score kernel underutilizes its TM=64 query tile when Q is tiny (decode,
// MTP tree verify). The v2 histogram is 256 KB/query so it can't scale — above the
// threshold the memory-scalable two-level tensor-core prefill path is used. Both
// paths support custom masks and query-sharding, so this is a pure occupancy/
// memory tradeoff. Tunable to align with SPARSE_MLA_DECODE_DISPATCH_MAX.
const INDEXER_DIRECT_DISPATCH_MAX = Number(process.env.GLM_INDEXER_DIRECT_DISPATCH_MAX ?? 64);
const CUBLASLT_WORKSPACE_BYTES = 2 * 1024 * 1024;

function ptr(t: Tensor | undefined): number {
  return t ? t.data : 0;
}


export class GlmTensor extends Tensor {
  constructor(workspace: WorkspaceBase, public readonly glm: GlmOps, data: number, allocSize: number, shape: number[], type: string, name: string | undefined, pinned: boolean, view: GlmTensor | undefined) {
    super(workspace, data, allocSize, shape, type, name, pinned, view);
  }

  [Symbol.dispose](): void {
    if (!this.canDispose()) {
      return;
    }
    // if stream is active, defer disposal until stream switch
    if (this.glm.currentStream) {
      this.glm.streamTensors.get(this.glm.currentStream)!.add(this);
    }
    else {
      super[Symbol.dispose]();
    }
  }

  free(): void {
    if (this.data !== 0) {
      if (this.pinned) {
        getNativeAddon().freePinned(this.data);
      } else if (!this.view) {
        this.glm.allocator.free(this.data);
      }
      this.detachData();
    }
  }

  h2d(data: Buffer, size?: number): void {
    getNativeAddon().h2d(this.glm.ctx, this.data, data, size ?? data.length);
  }

  d2h(buf: Buffer, size?: number): void {
    getNativeAddon().d2h(this.glm.ctx, buf, this.data, size ?? buf.length);
  }

  linear(weight: Tensor): Tensor {
    super.linear(weight);
    const batch = this.shape[0];
    const n = weight.shape[0];
    let k = weight.shape[1];
    const outShape = [batch, n];
    const out = this.workspace.alloc(outShape, this.type);
    if (weight.type === "F8_E4M3") {
      const scale = weight.workspace.tensors.get(weight.name! + "_scale_inv")!;
      getNativeAddon().fp8LinearDecode(this.glm.ctx, out.data, this.data, weight.data, scale.data, batch, n, k);
    } else if (weight.type === "U8") {
      k = k * 2; // NVFP4: weight is [N, K/2] packed, kernel expects K
      const scale = weight.workspace.tensors.get(weight.name! + "_weight_scale")!;
      const scale2 = weight.workspace.tensors.get(weight.name! + "_weight_scale_2")!;
      getNativeAddon().nvfp4LinearDecode(this.glm.ctx, out.data, this.data, weight.data, scale.data, scale2.data, batch, n, k, 0);
    } else {
      using workspace = batch >= 3 ? this.workspace.allocRaw(CUBLASLT_WORKSPACE_BYTES) : undefined;
      getNativeAddon().linear(this.glm.ctx, out.data, this.data, weight.data, batch, n, k, workspace?.data ?? 0, workspace?.bytes ?? 0);
    }
    return out;
  }

  bmm(B: Tensor, batch: number, M: number, N: number, K: number, transA: boolean = false, transB: boolean = false): Tensor {
    const out = this.workspace.alloc([batch * M, N], this.type);
    getNativeAddon().bmm(this.glm.ctx, out.data, this.data, B.data, 1.0, 0.0, batch, M, N, K, transA ? 1 : 0, transB ? 1 : 0);
    return out;
  }

  transpose4d(d0: number, d1: number, d2: number, d3: number, p0: number, p1: number, p2: number, p3: number): Tensor {
    const out = this.workspace.alloc([d0 * d1 * d2 * d3], this.type);
    getNativeAddon().transpose4d(this.glm.ctx, out.data, this.data, d0, d1, d2, d3, p0, p1, p2, p3);
    return out;
  }

  writePointers(tensors: Tensor[]): void {
    super.writePointers(tensors);
    const n = tensors.length;
    if (n > 8) {
      const ptrs = new BigInt64Array(n);
      for (let i = 0; i < n; i++) {
         ptrs[i] = BigInt(tensors[i].data);
      }
      this.h2d(Buffer.from(ptrs.buffer));
      return;
    }
    const ptrs = new Array(8).fill(0);
    for (let i = 0; i < n; i++) {
      ptrs[i] = tensors[i].data;
    }
    getNativeAddon().writePointers(this.glm.ctx, this.data,
      ptrs[0], ptrs[1], ptrs[2], ptrs[3],
      ptrs[4], ptrs[5], ptrs[6], ptrs[7], n);
  }

  rmsnorm(weight: Tensor, eps: number): Tensor {
    super.rmsnorm(weight, eps);
    const batch = this.shape[0];
    const dim = this.shape[1];
    const out = this.workspace.alloc([batch, dim], this.type);
    getNativeAddon().rmsnorm(this.glm.ctx, out.data, this.data, weight.data, eps, dim, batch);
    return out;
  }

  layernorm(weight: Tensor, bias: Tensor, eps: number): Tensor {
    super.layernorm(weight, bias, eps);
    const batch = this.shape[0];
    const dim = this.shape[1];
    const out = this.workspace.alloc([batch, dim], this.type);
    getNativeAddon().layernorm(this.glm.ctx, out.data, this.data, weight.data, bias.data, eps, dim, batch);
    return out;
  }

  fusedAddRmsnorm(input: Tensor, weight: Tensor, eps: number): { normed: Tensor, residual: Tensor } {
    super.fusedAddRmsnorm(input, weight, eps);
    const batch = this.shape[0];
    const dim = this.shape[1];
    const normed = this.workspace.alloc([batch, dim], this.type);
    const residual = this.workspace.alloc([batch, dim], this.type);
    getNativeAddon().fusedAddRmsnorm(this.glm.ctx, normed.data, residual.data, this.data, input.data, weight.data, eps, dim, batch);
    return { normed, residual };
  }

  fusedNormRope(weight: Tensor, cos: Tensor, sin: Tensor, eps: number, ropeDim: number, seqLen: number, batch: number, inStride?: number, interleaved?: boolean): Tensor {
    super.fusedNormRope(weight, cos, sin, eps, ropeDim, seqLen, batch, inStride, interleaved);
    const headDim = weight.numElements;
    const stride = inStride ?? headDim;
    const nHeads = this.shape[1] / stride;
    using reshaped = this.reshape([batch, seqLen, ...this.shape.slice(1)]);
    const out = this.workspace.alloc([batch, nHeads, seqLen, headDim], this.type);
    getNativeAddon().fusedNormRope(this.glm.ctx, out.data, reshaped.data, weight.data, cos.data, sin.data, eps, ropeDim, headDim, nHeads, seqLen, batch, inStride ?? headDim, interleaved ?? false);
    return out;
  }

  embedding(ids: Tensor): Tensor {
    super.embedding(ids);
    const hidden = this.shape[1];
    const seqLen = ids.numElements;
    const out = ids.workspace.alloc([seqLen, hidden], this.type);
    getNativeAddon().indexSelect(this.glm.ctx, out.data, this.data, ids.data, hidden, seqLen, 0);
    return out;
  }

  siluAndMul(up: Tensor): Tensor {
    super.siluAndMul(up);
    const batch = this.shape[0];
    const intermediate = this.shape[1];
    const out = this.workspace.alloc([batch, intermediate], this.type);
    getNativeAddon().siluAndMul(this.glm.ctx, out.data, this.data, up.data, intermediate, batch);
    return out;
  }

  arange(start: number, step: number, count: number): void {
    super.arange(start, step, count);
    getNativeAddon().arange(this.glm.ctx, this.data, start, step, count);
  }

  argmax(): Tensor {
    super.argmax();
    const { indices, values } = this.max();
    values[Symbol.dispose]();
    return indices;
  }

  max(offset: number = 0): { values: Tensor, indices: Tensor } {
    super.max(offset);
    const dim = this.shape[1];
    const batch = this.shape[0];
    const values = this.workspace.alloc([batch], this.type);
    const indices = this.workspace.alloc([batch], "I32");
    getNativeAddon().max(this.glm.ctx, values.data, indices.data, this.data, dim, batch, offset);
    return { values, indices };
  }

  indexSelect(indices: Tensor, offset: number = 0): Tensor {
    super.indexSelect(indices, offset);
    const batch = indices.numElements;
    const dim = this.shape[1];
    const out = this.workspace.alloc([batch, dim], this.type);
    getNativeAddon().indexSelect(this.glm.ctx, out.data, this.data, indices.data, dim, batch, offset);
    return out;
  }

  gather(indices: Tensor, k: number, inDim: number, batch: number): Tensor {
    super.gather(indices, k, inDim, batch);
    const out = this.workspace.alloc([batch, k], this.type);
    const elemSize = SafeTensorFile.dtypeBytes(this.type);
    getNativeAddon().gather(this.glm.ctx, out.data, this.data, indices.data, k, inDim, batch, elemSize);
    return out;
  }

  rotateInputIds(qoIndptr: Tensor, newTokens: Tensor, batchSize: number): Tensor {
    super.rotateInputIds(qoIndptr, newTokens, batchSize);
    const out = this.workspace.alloc(this.shape, this.type);
    getNativeAddon().rotateInputIds(this.glm.ctx, out.data, this.data, qoIndptr.data, newTokens.data, batchSize);
    return out;
  }

  causalConv1d(convState: Tensor, input: Tensor, weight: Tensor, cuSeqlens: Tensor, convDim: number, totalSeqLen: number, kernelSize: number, batchSize: number, convStateStride: number, chStride: number, seqStride: number): void {
    super.causalConv1d(convState, input, weight, cuSeqlens, convDim, totalSeqLen, kernelSize, batchSize, convStateStride, chStride, seqStride);
    getNativeAddon().causalConv1d(this.glm.ctx, this.data, convState.data, input.data, weight.data, cuSeqlens.data, convDim, totalSeqLen, kernelSize, batchSize, convStateStride, chStride, seqStride);
  }

  causalConv1dUpdate(convState: Tensor, input: Tensor, weight: Tensor, convDim: number, kernelSize: number, batchSize: number, convStateStride: number): Tensor {
    super.causalConv1dUpdate(convState, input, weight, convDim, kernelSize, batchSize, convStateStride);
    const out = this.workspace.alloc([batchSize * convDim], this.type);
    getNativeAddon().causalConv1dUpdate(this.glm.ctx, out.data, convState.data, input.data, weight.data, convDim, kernelSize, batchSize, convStateStride);
    return out;
  }

  rmsnormGated(input: Tensor, gate: Tensor, weight: Tensor, eps: number): void {
    super.rmsnormGated(input, gate, weight, eps);
    const dim = input.shape[1];
    const batch = input.shape[0];
    getNativeAddon().rmsnormGated(this.glm.ctx, this.data, input.data, gate.data, weight.data, eps, dim, batch);
  }

  gateSigmoidMul(gate: Tensor, numHeads: number, headDim: number): void {
    super.gateSigmoidMul(gate, numHeads, headDim);
    const batchSeq = this.numElements / (numHeads * headDim);
    getNativeAddon().gateSigmoidMul(this.glm.ctx, this.data, gate.data, batchSeq, numHeads, headDim);
  }

  fill(value: number, n: number): void {
    getNativeAddon().fill(this.glm.ctx, this.data, value, n);
  }

  async mmapLoad(mmapPtr: number, offset: number, nbytes: number, strided?: StridedMmap): Promise<void> {
    if (process.env.GLM_SKIP_MMAP_LOAD) {
      // for testing: skip actual load
      return;
    }
    if (strided) {
      return this.memcpy2dHostToDeviceAsync(strided.dstOffset, strided.dstPitch, mmapPtr + offset + strided.srcOffset, strided.srcPitch, strided.width, strided.height);
    } else {
      return this.mmapLoadAsync(mmapPtr, offset, nbytes);
    }
  }

  async mmapLoadAsync(mmapPtr: number, offset: number, nbytes: number): Promise<void> {
    if (process.env.GLM_SKIP_MMAP_LOAD) {
      // for testing: skip actual load
      return;
    }
    return getNativeAddon().mmapLoadAsync(this.glm.ctx, this.data, mmapPtr, offset, nbytes);
  }

  memcpy2dHostToDeviceAsync(dstOffset: number, dpitch: number, src: number, spitch: number, width: number, height: number): Promise<void> {
    return getNativeAddon().memcpy2dHostToDeviceAsync(this.glm.ctx, this.data + dstOffset, dpitch, src, spitch, width, height);
  }

  memcpy(src: Tensor, size?: number, kind?: MemcpyKind): void {
    super.memcpy(src, size, kind);
    if (!(src instanceof GlmTensor)) {
      throw new Error("GlmTensor.memcpy requires GlmTensor source");
    }
    const bytes = size ?? Math.min(this.allocSize, src.allocSize);
    const copyKind = kind ?? (src.pinned ? MemcpyKind.HostToDevice : MemcpyKind.DeviceToDevice);
    // if (copyKind === MemcpyKind.DeviceToDevice && this.glm !== src.glm) {
    //   getNativeAddon().memcpyPeer(src.glm.ctx, this.data, this.glm.device, src.data, src.glm.device, bytes);
    // } else {
      getNativeAddon().memcpy(src.glm.ctx, this.data, src.data, bytes, memcpyKindToNative(copyKind));
    // }
  }

  memcpy2d(dstOffset: number, dpitch: number, src: Tensor, srcOffset: number, spitch: number, width: number, height: number, kind: MemcpyKind): void {
    super.memcpy2d(dstOffset, dpitch, src, srcOffset, spitch, width, height, kind);
    if (!(src instanceof GlmTensor)) {
      throw new Error("GlmTensor.memcpy requires GlmTensor source");
    }
    // if (kind === MemcpyKind.DeviceToDevice && this.glm !== (src as GlmTensor).glm) {
    //   const s = src as GlmTensor;
    //   getNativeAddon().memcpy3dPeer(
    //     src.glm.ctx,
    //     this.data + dstOffset, dpitch, width, height, this.glm.device,
    //     0, 0, 0,
    //     s.data + srcOffset, spitch, width, height, s.glm.device,
    //     0, 0, 0,
    //     width, height, 1,
    //   );
    // } else {
      getNativeAddon().memcpy2d(src.glm.ctx, this.data + dstOffset, dpitch, (src as GlmTensor).data + srcOffset, spitch, width, height, memcpyKindToNative(kind));
    // }
  }

  rotaryEmbedding(positionIds: Tensor, batch: number, seqLen: number): { cos: Tensor, sin: Tensor } {
    super.rotaryEmbedding(positionIds, batch, seqLen);
    const dimHalf = this.shape[0];
    const hd = dimHalf * 2;
    using reshaped = positionIds.reshape([batch, seqLen]);
    const cos = positionIds.workspace.alloc([batch, seqLen, hd], this.type);
    const sin = positionIds.workspace.alloc([batch, seqLen, hd], this.type);
    getNativeAddon().rotaryEmbedding(this.glm.ctx, cos.data, sin.data, this.data, reshaped.data, dimHalf, batch, seqLen);
    return { cos, sin };
  }

  ropeTranspose(cos: Tensor, sin: Tensor, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, inStride?: number, interleaved?: boolean): Tensor {
    super.ropeTranspose(cos, sin, ropeDim, headDim, nHeads, seqLen, batch, inStride, interleaved);
    using reshaped = this.reshape([batch, seqLen, ...this.shape.slice(1)]);
    const out = this.workspace.alloc([batch * seqLen, nHeads, headDim], this.type);
    getNativeAddon().ropeTranspose(this.glm.ctx, out.data, reshaped.data, ropeDim > 0 ? cos.data : 0, ropeDim > 0 ? sin.data : 0, ropeDim, headDim, nHeads, seqLen, batch, inStride ?? headDim, interleaved ?? false);
    return out;
  }

  applyRotaryPosEmb(cos: Tensor, sin: Tensor, ropeDim: number, nHeads: number, seqLen: number, batch: number, unsqueezeDim: number, interleaved?: boolean): Tensor {
    super.applyRotaryPosEmb(cos, sin, ropeDim, nHeads, seqLen, batch, unsqueezeDim, interleaved);
    let inputData = this.data;
    if (this.shape.length === 2) {
      using reshaped = this.reshape([batch, seqLen, ...this.shape.slice(1)]);
      inputData = reshaped.data;
    }
    const out = this.workspace.alloc(this.shape, this.type);
    getNativeAddon().applyRotaryPosEmb(this.glm.ctx, out.data, inputData, cos.data, sin.data, ropeDim, nHeads, seqLen, batch, unsqueezeDim, interleaved ?? false);
    return out;
  }

  mlaVExpand(vProj: Tensor, seqLen: number, batch: number, _lse?: Tensor, headOffset: number = 0, attnNHeads?: number, vProjHeadOffset: number = 0, tokenMajor: boolean = false): Tensor {
    super.mlaVExpand(vProj, seqLen, batch);
    const kvLoraRank = this.shape[this.shape.length - 1];
    const nHeads = this.shape[1];
    const vHeadDim = vProj.shape[1];
    attnNHeads = attnNHeads ?? nHeads;
    const BS = batch * seqLen;
    let inputData = this.data;
    if (this.shape.length !== 4) {
      using reshaped = this.reshape([batch, seqLen, ...this.shape.slice(1)]);
      inputData = reshaped.data;
    }
    const out = this.workspace.alloc([BS, nHeads * vHeadDim], this.type);
    const effSeqLen = tokenMajor ? 1 : seqLen;
    const effBatch = tokenMajor ? BS : batch;
    getNativeAddon().mlaVExpand(this.glm.ctx, out.data, inputData, vProj.data, kvLoraRank, vHeadDim, nHeads, effSeqLen, effBatch, attnNHeads, headOffset, vProjHeadOffset);
    return out;
  }

  sigmoid(): Tensor {
    const n = this.shape.reduce((a, b) => a * b, 1);
    const out = this.workspace.alloc(this.shape, this.type);
    getNativeAddon().sigmoid(this.glm.ctx, out.data, this.data, n);
    return out;
  }

  topk(k: number, dim: number, offset = 0): { values: Tensor, indices: Tensor } {
    const batch = this.shape.reduce((a, b) => a * b, 1) / dim;
    const values = this.workspace.alloc([batch, k], this.type);
    const indices = this.workspace.alloc([batch, k], "I32");
    if (k > 8) {
      using hist = this.workspace.alloc([batch, 65536], "I32");
      using meta = this.workspace.alloc([batch, 4], "I32");
      getNativeAddon().topkFromScores(this.glm.ctx, values.data, indices.data, this.data, 0, hist.data, meta.data, batch, dim, k, 1, offset === 0 ? 0 : 1, offset);
    } else {
      getNativeAddon().topk(this.glm.ctx, values.data, indices.data, this.data, k, dim, batch, offset);
    }
    return { values, indices };
  }

  indexAdd(indices: Tensor, values: Tensor, nIndices: number, dim: number): void {
    getNativeAddon().indexAdd(this.glm.ctx, this.data, indices.data, values.data, nIndices, dim);
  }

  add(other: Tensor, n?: number): Tensor {
    if (this.shape.length === 2 && other.shape.length === 1 && this.shape[1] === other.shape[0]) {
      const out = this.workspace.alloc(this.shape, this.type);
      getNativeAddon().addBroadcast(this.glm.ctx, out.data, this.data, other.data, this.shape[1], this.shape[0]);
      return out;
    }
    const count = n ?? this.shape.reduce((a, b) => a * b, 1);
    const out = this.workspace.alloc(this.shape, this.type);
    getNativeAddon().add(this.glm.ctx, out.data, this.data, other.data, count);
    return out;
  }

  private runSumPointers(output: Tensor, inputs: Tensor[], writeback?: boolean): void {
    const N = inputs.length;
    const numel = output.shape.reduce((a, b) => a * b, 1);
    const dtype = output.type === "F32" ? 7 : 9;
    const ptrs = new Array<number>(8).fill(0);
    for (let i = 0; i < N; i++) ptrs[i] = inputs[i].data;
    getNativeAddon().sumPointers(
      this.glm.ctx,
      ptrs[0], ptrs[1], ptrs[2], ptrs[3],
      ptrs[4], ptrs[5], ptrs[6], ptrs[7],
      output.data, N, numel, dtype, writeback,
    );
  }

  /**
   * Fused P2P AllReduce + Add + RMSNorm over N peer partial-sum tensors.
   * Computes, per row of `dim` elements:
   *   s = inputA + sum(peers);  residual = s;  out = weight * s * rsqrt(mean(s^2)+eps)
   * `out` and `residual` receive the results; `peers` are read only.
   * Requires dim % 512 == 0 and 8192 % dim == 0 (row-aligned tiles).
   */
  rmsNormPointers(inputA: Tensor, peers: Tensor[], weight: Tensor, out: Tensor, residual: Tensor, dim: number, eps: number): void {
    const N = peers.length;
    const numel = out.shape.reduce((a, b) => a * b, 1);
    const dtype = out.type === "F32" ? 7 : 9;
    const ptrs = new Array<number>(8).fill(0);
    for (let i = 0; i < N; i++) ptrs[i] = peers[i].data;
    getNativeAddon().rmsNormPointers(
      this.glm.ctx,
      ptrs[0], ptrs[1], ptrs[2], ptrs[3],
      ptrs[4], ptrs[5], ptrs[6], ptrs[7],
      inputA.data, weight.data, out.data, residual.data,
      N, numel, dim, eps, dtype,
    );
  }

  sumInPlace(tensors: Tensor[], writeback?: boolean): void {
    super.sumInPlace(tensors);
    this.runSumPointers(this, tensors, writeback);
  }

  sum(tensors: Tensor[]): Tensor {
    super.sum(tensors);
    const out = this.workspace.alloc(this.shape, this.type);
    this.runSumPointers(out, [this as Tensor, ...tensors]);
    return out;
  }

  scaleInPlace(scale: number, n: number): void {
    getNativeAddon().scale(this.glm.ctx, this.data, this.data, scale, n);
  }

  mul(other: Tensor, n?: number): Tensor {
    if (this.shape.length === 2 && other.shape.length === 1 && this.shape[1] === other.shape[0]) {
      const out = this.workspace.alloc(this.shape, this.type);
      getNativeAddon().mulBroadcast(this.glm.ctx, out.data, this.data, other.data, this.shape[1], this.shape[0]);
      return out;
    }
    const count = n ?? this.shape.reduce((a, b) => a * b, 1);
    const out = this.workspace.alloc(this.shape, this.type);
    getNativeAddon().mul(this.glm.ctx, out.data, this.data, other.data, count);
    return out;
  }

  cat(tensors: Tensor[], dim: number): Tensor {
    super.cat(tensors, dim);
    const all = [this as Tensor, ...tensors];
    const outShape = [...this.shape];
    for (const t of tensors) outShape[dim] += t.shape[dim];
    const out = this.workspace.alloc(outShape, this.type);
    const elemBytes = SafeTensorFile.dtypeBytes(this.type);
    const outerStrides = this.shape.slice(0, dim).reduce((a, b) => a * b, 1);
    const innerStride = this.shape.slice(dim + 1).reduce((a, b) => a * b, 1);
    const dstRowBytes = outShape[dim] * innerStride * elemBytes;
    let offset = 0;
    for (const t of all) {
      const srcRowBytes = t.shape[dim] * innerStride * elemBytes;
      out.memcpy2d(
        offset,
        dstRowBytes,
        t,
        0,
        srcRowBytes,
        srcRowBytes,
        outerStrides,
        MemcpyKind.DeviceToDevice,
      );
      offset += srcRowBytes;
    }
    return out;
  }

  slice(dim: number, start: number, length: number): Tensor {
    super.slice(dim, start, length);
    if (start < 0) {
      start = this.shape[dim] + start;
    }
    const outShape = [...this.shape];
    outShape[dim] = length;
    const out = this.workspace.alloc(outShape, this.type);
    const elemBytes = SafeTensorFile.dtypeBytes(this.type);
    const outerStrides = this.shape.slice(0, dim).reduce((a, b) => a * b, 1);
    const innerStride = this.shape.slice(dim + 1).reduce((a, b) => a * b, 1);
    const srcPitch = this.shape[dim] * innerStride * elemBytes;
    const dstPitch = length * innerStride * elemBytes;
    const srcOffset = start * innerStride * elemBytes;
    out.memcpy2d(0, dstPitch, this, srcOffset, srcPitch, dstPitch, outerStrides, MemcpyKind.DeviceToDevice);
    return out;
  }

  narrow(start: number, length: number): Tensor {
    super.narrow(start, length);
    if (start < 0) {
      start = this.shape[0] + start;
    }
    const innerElements = this.shape.slice(1).reduce((a, b) => a * b, 1);
    const elemBytes = SafeTensorFile.dtypeBytes(this.type);
    const byteOffset = start * innerElements * elemBytes;
    const newShape = [length, ...this.shape.slice(1)];
    const newAllocSize = this.allocSize - byteOffset;
    return this.workspace.glm.wrapTensor(this.workspace, this.data + byteOffset, newAllocSize, newShape, this.type, this.pinned, this);
  }

  scatterScalar(indices: Tensor, value: number, k: number): void {
    const outDim = this.shape[1];
    const batch = this.shape[0];
    getNativeAddon().scatterScalar(this.glm.ctx, this.data, indices.data, value, k, outDim, batch);
  }

  maskedFill(mask: Tensor, value: number, n: number): void {
    getNativeAddon().maskedFill(this.glm.ctx, this.data, this.data, mask.data, value, n);
  }

  applyRotaryPosEmbPartial(cos: Tensor, sin: Tensor, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, unsqueezeDim: number, interleaved?: boolean): Tensor {
    super.applyRotaryPosEmbPartial(cos, sin, ropeDim, headDim, nHeads, seqLen, batch, unsqueezeDim, interleaved);
    using reshaped = this.reshape([batch, seqLen, ...this.shape.slice(1)]);
    const out = this.workspace.alloc(this.shape, this.type);
    getNativeAddon().applyRotaryPosEmbPartial(this.glm.ctx, out.data, reshaped.data, cos.data, sin.data, ropeDim, headDim, nHeads, seqLen, batch, unsqueezeDim, interleaved ?? false);
    return out;
  }

  reduceSum(): Tensor {
    const batch = this.shape[0];
    const dim = this.shape[1];
    const out = this.workspace.alloc([batch], this.type);
    getNativeAddon().reduceSum(this.glm.ctx, out.data, this.data, batch, dim);
    return out;
  }

  rowNormalize(scale: number, normalize: boolean = true): Tensor {
    const batch = this.shape[0];
    const dim = this.shape[1];
    const out = this.workspace.alloc([batch, dim], this.type);
    getNativeAddon().rowNormalize(this.glm.ctx, out.data, this.data, scale, batch, dim, normalize);
    return out;
  }

  groupMaskMul(groupMask: Tensor, expertsPerGroup: number, nGroup: number): void {
    const batch = this.shape[0];
    const numExperts = this.shape[1];
    getNativeAddon().groupMaskMul(this.glm.ctx, this.data, groupMask.data, numExperts, expertsPerGroup, nGroup, batch);
  }

  mulMatId(weights: Tensor[], expertIds: Tensor, topK: number, count: number, N: number, K: number, name: string): Tensor {
    const out = this.workspace.alloc([count, N], this.type);
    const ptrName = `__moe_ptrs.${name}`;
    let weightPtrs = this.workspace.tensors.get(ptrName);
    if (!weightPtrs) {
      weightPtrs = this.workspace.alloc([weights.length], "I64", ptrName);
      weightPtrs.writePointers(weights);
    }
    if (weights[0].type === "U8") {
      const scalePtrName = ptrName + "_weight_scale";
      const scale2PtrName = ptrName + "_weight_scale_2";
      let scalePtrs = this.workspace.tensors.get(scalePtrName);
      if (!scalePtrs) {
        const scaleTensors = weights.map(w => w.workspace.tensors.get(w.name! + "_weight_scale")!);
        scalePtrs = this.workspace.alloc([weights.length], "I64", scalePtrName);
        scalePtrs.writePointers(scaleTensors);
      }
      let scale2Ptrs = this.workspace.tensors.get(scale2PtrName);
      if (!scale2Ptrs) {
        const scale2Tensors = weights.map(w => w.workspace.tensors.get(w.name! + "_weight_scale_2")!);
        scale2Ptrs = this.workspace.alloc([weights.length], "I64", scale2PtrName);
        scale2Ptrs.writePointers(scale2Tensors);
      }
      if (count > MUL_MAT_ID_GROUPED_THRESHOLD) {
        const numExperts = weights.length;
        const wsSize = getNativeAddon().mmaMoeCoopWorkspaceSize(count, N, K, numExperts);
        using wsTensor = this.workspace.allocRaw(wsSize);
        getNativeAddon().nvfp4MulMatIdGroupedMmaCoop(this.glm.ctx, out.data, this.data, weightPtrs.data, scalePtrs.data, scale2Ptrs.data, expertIds.data, topK, count, N, K, numExperts, wsTensor.data);
      } else {
        getNativeAddon().nvfp4MulMatId(this.glm.ctx, out.data, this.data, weightPtrs.data, scalePtrs.data, scale2Ptrs.data, expertIds.data, topK, count, N, K);
      }
    } else if (count > MUL_MAT_ID_GROUPED_THRESHOLD) {
      const numExperts = weights.length;
      const wsSize = getNativeAddon().mmaMoeWorkspaceSize(count, N, K, numExperts);
      using wsTensor = this.workspace.allocRaw(wsSize);
      getNativeAddon().bf16MulMatIdGroupedMma(this.glm.ctx, out.data, this.data, weightPtrs.data, expertIds.data, topK, count, N, K, numExperts, wsTensor.data);
    } else {
      getNativeAddon().mulMatId(this.glm.ctx, out.data, this.data, weightPtrs.data, expertIds.data, topK, count, N, K);
    }
    return out;
  }

  private getMoeNvfp4Ptrs(weights: Tensor[], name: string): { weightPtrs: Tensor, scalePtrs: Tensor, scale2Ptrs: Tensor } {
    const ptrName = `__moe_ptrs.${name}`;
    let weightPtrs = this.workspace.tensors.get(ptrName);
    if (!weightPtrs) {
      weightPtrs = this.workspace.alloc([weights.length], "I64", ptrName);
      weightPtrs.writePointers(weights);
    }
    const scalePtrName = ptrName + "_weight_scale";
    let scalePtrs = this.workspace.tensors.get(scalePtrName);
    if (!scalePtrs) {
      const scaleTensors = weights.map(w => w.workspace.tensors.get(w.name! + "_weight_scale")!);
      scalePtrs = this.workspace.alloc([weights.length], "I64", scalePtrName);
      scalePtrs.writePointers(scaleTensors);
    }
    const scale2PtrName = ptrName + "_weight_scale_2";
    let scale2Ptrs = this.workspace.tensors.get(scale2PtrName);
    if (!scale2Ptrs) {
      const scale2Tensors = weights.map(w => w.workspace.tensors.get(w.name! + "_weight_scale_2")!);
      scale2Ptrs = this.workspace.alloc([weights.length], "I64", scale2PtrName);
      scale2Ptrs.writePointers(scale2Tensors);
    }
    return { weightPtrs, scalePtrs, scale2Ptrs };
  }

  swiGluMlpMoe(
    weights: { gate: Tensor[], up: Tensor[], down: Tensor[] },
    topkIndicesFlat: Tensor,
    topK: number, count: number,
    moeIntermediate: number, hs: number,
    pfx: string,
  ): Tensor {
    super.swiGluMlpMoe(weights, topkIndicesFlat, topK, count, moeIntermediate, hs, pfx);

    if (count <= MUL_MAT_ID_GROUPED_THRESHOLD || weights.gate[0].type !== "U8") {
      using gateOutStream = this.workspace.glm.withStream(() => this.mulMatId(weights.gate, topkIndicesFlat, topK, count, moeIntermediate, hs, `${pfx}.gate_proj`));
      using upOut = this.mulMatId(weights.up, topkIndicesFlat, topK, count, moeIntermediate, hs, `${pfx}.up_proj`);
      gateOutStream.streamWaitEvent();
      using gateOut = gateOutStream.result;
      using siluOut = gateOut.siluAndMul(upOut);
      return siluOut.mulMatId(weights.down, topkIndicesFlat, 1, count, hs, moeIntermediate, `${pfx}.down_proj`);
    }

    const numExperts = weights.gate.length;
    const ctx = this.glm.ctx;
    const gatePtrs = this.getMoeNvfp4Ptrs(weights.gate, `${pfx}.gate_proj`);
    const upPtrs = this.getMoeNvfp4Ptrs(weights.up, `${pfx}.up_proj`);

    const scatterWsSize = getNativeAddon().mmaMoeCoopScatterWorkspaceSize(count, hs, numExperts);
    using scatterWs = this.workspace.allocRaw(scatterWsSize);
    getNativeAddon().mmaMoeCoopScatter(ctx, this.data, topkIndicesFlat.data, topK, count, hs, numExperts, scatterWs.data);

    const gemmWsSize = getNativeAddon().mmaMoeCoopGemmWorkspaceSize(count, moeIntermediate);

    using gateStream = this.workspace.glm.withStream(() => {
      using gemmWs = this.workspace.allocRaw(gemmWsSize);
      const gateOut = this.workspace.alloc([count, moeIntermediate], this.type);
      getNativeAddon().mmaMoeCoopGemm(ctx, gatePtrs.weightPtrs.data, gatePtrs.scalePtrs.data, gatePtrs.scale2Ptrs.data,
                                       numExperts, moeIntermediate, hs, count, scatterWs.data, gemmWs.data, gateOut.data);
      return gateOut;
    });

    using upGemmWs = this.workspace.allocRaw(gemmWsSize);
    using upOut = this.workspace.alloc([count, moeIntermediate], this.type);
    getNativeAddon().mmaMoeCoopGemm(ctx, upPtrs.weightPtrs.data, upPtrs.scalePtrs.data, upPtrs.scale2Ptrs.data,
                                     numExperts, moeIntermediate, hs, count, scatterWs.data, upGemmWs.data, upOut.data);

    gateStream.streamWaitEvent();
    using gateOut = gateStream.result;
    using siluOut = gateOut.siluAndMul(upOut);
    return siluOut.mulMatId(weights.down, topkIndicesFlat, 1, count, hs, moeIntermediate, `${pfx}.down_proj`);
  }

  scatterAddRows(scales: Tensor, topK: number, numRows: number): Tensor {
    const dim = this.shape[1];
    const out = this.workspace.alloc([numRows, dim], this.type);
    getNativeAddon().scatterAddRows(this.glm.ctx, out.data, this.data, scales.data, topK, dim, numRows, 0);
    return out;
  }

}

export class GlmOps implements DeviceOps {
  readonly worldSize = 1;
  synchronizeListeners: WeakRef<WorkspaceBase>[] = [];
  ctx: number;
  device: number;
  allocator: Allocator;
  readonly arenaBase?: number;
  readonly arenaSize?: number;
  capturing = false;

  constructor(deviceId: number = 0, libPath?: string, arenaGb?: number) {
    const native = getNativeAddon(libPath);
    this.ctx = native.init(deviceId);
    if (!this.ctx) {
      throw new Error(`glm_init failed on device ${deviceId}`);
    }
    this.device = deviceId;

    if (arenaGb) {
      const size = arenaGb * 1024 * 1024 * 1024;
      const baseEnv = process.env[`GLM_ARENA_BASE_${deviceId}`];
      const base = baseEnv === undefined ? getNativeAddon().alloc(this.ctx, size) : Number(baseEnv);
      if (!Number.isSafeInteger(base) || base <= 0) {
        getNativeAddon().free(this.ctx);
        throw new Error(`Invalid GLM_ARENA_BASE_${deviceId}: ${baseEnv ?? base}`);
      }
      this.arenaBase = base;
      this.arenaSize = size;
      this.allocator = new ArenaAllocator(base, size);
    }
    else {
      this.allocator = {
        alloc: (size: number) => {
          if (this.capturing) {
            console.warn("Warning: allocating during capture will fail.");
          }
          return getNativeAddon().alloc(this.ctx, size)
        },
        free: (ptr: number) => getNativeAddon().freeBuf(this.ctx, ptr),
      }
    }
  }

  free(): void {
    getNativeAddon().free(this.ctx);
  }

  [Symbol.dispose](): void {
    this.free();
  }

  allocPinned(bytes: number): number {
    const p = getNativeAddon().allocPinned(bytes);
    if (!p) throw new Error(`allocPinned failed for size ${bytes}`);
    return p;
  }

  newTensor(workspace: WorkspaceBase, shape: number[], type: string, pinned: boolean, name?: string, _parallelism?: TensorParallelism): GlmTensor {
    const size = Tensor.byteCount(shape, type);
    const data = pinned ? this.allocPinned(size) : this.allocator.alloc(size);
    return new GlmTensor(workspace, this, data, size, shape, type, name, pinned, undefined);
  }

  wrapTensor(workspace: WorkspaceBase, data: number, allocSize: number, shape: number[], type: string, pinned: boolean, view: GlmTensor | undefined): Tensor {
    return new GlmTensor(workspace, this, data, allocSize, shape, type, undefined, pinned, view);
  }

  sampleBatch(outTokens: Tensor, topkVals: Tensor, topkIdxs: Tensor, workspace: Tensor, logits: Tensor, penaltyTokens: Tensor, penaltyCount: Tensor, maxWindow: number, vocabSize: number, batchSize: number, temperatures: Tensor, repPenalties: Tensor, presPenalties: Tensor, topKs: Tensor, topPs: Tensor, stepCounter: Tensor, maxEffectiveK: number): void {
    getNativeAddon().sampleBatch(this.ctx, outTokens.data, topkVals.data, topkIdxs.data, workspace.data, logits.data, penaltyTokens.data, penaltyCount.data, maxWindow, vocabSize, batchSize, temperatures.data, repPenalties.data, presPenalties.data, topKs.data, topPs.data, stepCounter.data, maxEffectiveK);
  }

  synchronize(): void {
    getNativeAddon().synchronize(this.ctx);
    notifySynchronizedWorkspaces(this.synchronizeListeners);
  }

  async synchronizeAsync(): Promise<void> {
    await getNativeAddon().synchronizeAsync(this.ctx);
    notifySynchronizedWorkspaces(this.synchronizeListeners);
  }

  synchronizeStream(streamIdx: number): void {
    getNativeAddon().synchronizeStream(this.ctx, streamIdx);
  }

  synchronizeStreamAsync(streamIdx: number): Promise<void> {
    return getNativeAddon().synchronizeStreamAsync(this.ctx, streamIdx);
  }

  streamTensors = new Map<number, Set<GlmTensor>>();
  setStream(streamIdx: number): void {
    getNativeAddon().setStream(this.ctx, streamIdx);
    this.currentStream = streamIdx;
    if (streamIdx) {
      if (!this.streamTensors.has(streamIdx))
        this.streamTensors.set(streamIdx, new Set());
    }
  }

  currentStream = 0;
  availableStreams = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
  disposeStream(stream: number) {
    if (this.availableStreams.includes(stream))
      throw new Error(`Stream ${stream} already disposed`);
    this.availableStreams.push(stream);
    const tensors = this.streamTensors.get(stream);
    this.streamTensors.delete(stream);
    for (const tensor of tensors!) {
      tensor[Symbol.dispose]();
    }
  }

  withStream<T>(fn: () => T) {
    const stream = this.availableStreams.pop();
    if (stream === undefined)
      throw new Error("No available streams");
    const currentStream = this.currentStream;
    // Record event on current stream so the alternate stream can wait for
    // all prior work (e.g. rmsnorm output that K/V will read).
    getNativeAddon().eventRecord(this.ctx, currentStream, currentStream);
    if (this.streamTensors.has(stream)) {
      throw new Error(`Stream ${stream} already in use`);
    }
    this.setStream(stream);
    getNativeAddon().streamWaitEvent(this.ctx, stream, currentStream);
    const result = fn();
    // Record event on the alternate stream so others can wait
    getNativeAddon().eventRecord(this.ctx, stream, stream);
    this.setStream(currentStream);
    return {
      [Symbol.dispose]: () => {
        this.disposeStream(stream);
      },
      result,
      synchronize: () => {
        getNativeAddon().synchronizeStream(this.ctx, stream);
      },
      streamWaitEvent: () => {
        getNativeAddon().streamWaitEvent(this.ctx, this.currentStream, stream);
      }
    }
  }

  eventRecord(eventIdx: number, streamIdx: number): void {
    getNativeAddon().eventRecord(this.ctx, eventIdx, streamIdx);
  }

  streamWaitEvent(streamIdx: number, eventIdx: number): void {
    getNativeAddon().streamWaitEvent(this.ctx, streamIdx, eventIdx);
  }

  kvCacheWrite(srcK: Tensor, srcV: Tensor, dstK: Tensor, dstV: Tensor, slotMapping: Tensor, batchSize: number, nKv: number, hd: number, srcKTokenStride: number, srcKHeadStride: number, srcVTokenStride: number, srcVHeadStride: number): void {
    const pageSize = dstK.shape[1] / (nKv * hd);
    getNativeAddon().kvCacheWrite(this.ctx, ptr(srcK), ptr(srcV), ptr(dstK), ptr(dstV), ptr(slotMapping), batchSize, nKv, hd, pageSize, srcKTokenStride, srcKHeadStride, srcVTokenStride, srcVHeadStride);
  }

  gatherPages(srcData: Tensor, pageIndices: Tensor, pageIndptrD: Tensor, lastPageLen: Tensor, batchSize: number, paddedKvLen: number, _kvTokenIndptrD: Tensor, _contextParallel: boolean): Tensor {
    const pageSize = srcData.shape[1];
    const D = srcData.shape[2];
    const out = pageIndptrD.workspace.alloc([paddedKvLen / pageSize, pageSize, D], srcData.type);
    const elemBytes = srcData.type === "U8" ? 1 : 2;
    const maxPages = srcData.shape[0];
    getNativeAddon().gatherPages(this.ctx, out.data, srcData.data, pageIndices.data, pageIndptrD.data, lastPageLen.data, maxPages, batchSize, pageSize, D * elemBytes);
    return out;
  }

  gatherTopkCkv(_state: ExecutionState, kvCache: Tensor, outputs: readonly Tensor[], topkIdx: Tensor, pageIndices: Tensor, pageIndptr: Tensor, kvTokenIndptr: Tensor, batchIndices: Tensor, topk: number, paddedKvLen: number, cpWorldSize: number = 0, cpRank: number = 0, effPageSize?: number): void {
    const pageSize = kvCache.shape[1];
    const bpt = kvCache.shape[2];
    const numTokens = topkIdx.shape[0];
    const N = outputs.length;
    if (N < 1 || N > 8) {
      throw new Error(`gatherTopkCkv: outputs.length=${N} must be in [1, 8]`);
    }
    // Output contract: each peer buffer must be 3D
    // [paddedKvLen/pageSize, pageSize, BPT] U8 (same shape gatherPages' TS
    // impl returns). Caller is responsible for pre-allocating; validate here.
    if (paddedKvLen % pageSize !== 0) {
      throw new Error(`gatherTopkCkv: paddedKvLen=${paddedKvLen} not divisible by pageSize=${pageSize}`);
    }
    const expectedPages = paddedKvLen / pageSize;
    const expectedShape: number[] = [expectedPages, pageSize, bpt];
    for (let i = 0; i < N; i++) {
      const o = outputs[i];
      if (o.type !== kvCache.type) {
        throw new Error(`gatherTopkCkv: outputs[${i}].type=${o.type}, expected ${kvCache.type}`);
      }
      if (o.shape.length !== 3 || o.shape[0] !== expectedShape[0] || o.shape[1] !== expectedShape[1] || o.shape[2] !== expectedShape[2]) {
        throw new Error(`gatherTopkCkv: outputs[${i}].shape=[${o.shape.join("x")}], expected [${expectedShape.join("x")}]`);
      }
    }
    const effPs = effPageSize ?? pageSize;
    // Build the 8-pointer peer table from the caller-provided outputs. Unused
    // slots stay at 0 (kernel only writes peers [0, N)).
    const peerPtrs = new Array<number>(8).fill(0);
    for (let j = 0; j < N; j++) peerPtrs[j] = outputs[j].data;

    // Dedup scratch, used only when numTokens > 1.
    //
    // These are persistent (named) rather than alloc + `using`. Empirically the
    // alloc + `using` form produced CKV corruption under CP+MTP that this form
    // does not; the mechanism was never established, so treat the requirement as
    // observed rather than understood, and re-test rather than assume if it is
    // ever changed back.
    //
    // INVARIANT: at most one gather in flight per device. ParallelOps issues a
    // supergroup's gathers sequentially on a single stream, and the consumer
    // waits before the next supergroup produces, so this holds. Reintroducing
    // concurrent gather streams would require one scratch set per stream.
    const BITS_PER_WORD = 32;
    const ws = kvCache.workspace;
    const bitmapWords = Math.ceil(paddedKvLen / BITS_PER_WORD);  // one bit per flat slot
    const maxEntries = numTokens * topk;                         // one int2 per (query, k)
    using bitmap = ws.alloc([bitmapWords], "I32");
    using unique = ws.alloc([maxEntries * 2], "I32");
    using counter = ws.alloc([1], "I32");

    getNativeAddon().gatherTopkCkv(
      this.ctx,
      peerPtrs[0], peerPtrs[1], peerPtrs[2], peerPtrs[3],
      peerPtrs[4], peerPtrs[5], peerPtrs[6], peerPtrs[7],
      kvCache.data, topkIdx.data, batchIndices.data,
      pageIndices.data, pageIndptr.data, kvTokenIndptr.data,
      N, cpWorldSize, cpRank,
      effPs, bpt,
      numTokens, topk, paddedKvLen,
      bitmap.data, unique.data, counter.data,
    );
  }

  indexerScore(out: Tensor, q: Tensor, kData: Tensor, weights: Tensor, pageIndices: Tensor, pageIndptr: Tensor, lastPageLen: Tensor, qoIndptr: Tensor, scale: number, totalQ: number, idxNHeads: number, idxHeadDim: number, pageSize: number, maxKvLen: number, causal: boolean, kvTokenIndptr?: Tensor): void {
    getNativeAddon().indexerScore(this.ctx, out.data, q.data, kData.data, weights.data, pageIndices.data, pageIndptr.data, lastPageLen.data, qoIndptr.data, scale, totalQ, idxNHeads, idxHeadDim, pageSize, maxKvLen, causal ? 1 : 0, kvTokenIndptr?.data ?? 0);
  }

  // Indexer top-k selection -> physical KV slots. Runs the full pipeline
  // (score+topk, then map token positions to slots) on this device. Decode uses
  // the multi-block score+histogram kernel (v2): totalQ is small (batch), so
  // per-query scratch is tiny. Prefill uses the fused two-level kernel: multi-
  // block grid (numSplits × totalQ) parallelizes across both KV and query dims,
  // with a coarse/fine histogram top-K that avoids materializing scores.
  // Scratch: ~17 MB (coarseHist + fineHist + meta) vs 2.1 GB for v2 on prefill.
  // Writes the compacted valid-slot count per query into `topkLength` (a stable
  // caller buffer), which feeds the sparse kernel's topk_length so it only walks
  // ceil(count/BI) candidate tiles instead of the full topk.
  indexerTopk(state: ExecutionState, idxQ: Tensor, kData: Tensor, weights: Tensor, pageIndices: Tensor, indptr: Tensor, lastPageLen: Tensor, qoIndptr: Tensor, scale: number, topk: number, decode: boolean, qGlobalStart: number = 0, customMask?: Tensor, maskIndptr?: Tensor, maskKvLen?: Tensor, cpWorldSize: number = 0, cpRank: number = 0, globalLastPageLen?: Tensor, kvTokenIndptr?: Tensor): { values: Tensor, indices: Tensor } {
    const totalQ = idxQ.shape[0];
    const idxNHeads = idxQ.shape[1];
    const idxHeadDim = idxQ.shape[2];
    const pageSize = kData.shape[1];
    const maxKv = kData.shape[0] * kData.shape[1];
    const useDirect = totalQ <= INDEXER_DIRECT_DISPATCH_MAX;
    return useDirect
      ? this.indexerScoreTopkV2(idxQ, kData, weights, pageIndices, indptr, lastPageLen, qoIndptr, scale, totalQ, idxNHeads, idxHeadDim, pageSize, topk, maxKv, decode ? 0 : 1, customMask, maskIndptr, maskKvLen, qGlobalStart, cpWorldSize, cpRank, globalLastPageLen, kvTokenIndptr)
      : this.indexerScoreTopkPrefill(idxQ, kData, weights, pageIndices, indptr, lastPageLen, qoIndptr, scale, totalQ, idxNHeads, idxHeadDim, pageSize, topk, maxKv, customMask, maskIndptr, maskKvLen, qGlobalStart, cpWorldSize, cpRank, globalLastPageLen, kvTokenIndptr);
  }

  // Sort each top-k row ascending by index (-1 padding last), in place.
  // Restores the replicated path's ordering after the CP merge, which
  // concatenates per-shard lists rank-major. See idx_sort_by_index_kernel.
  sortTopkByIndex(indices: Tensor, values: Tensor, batch: number, topk: number): void {
    getNativeAddon().sortTopkByIndex(this.ctx, indices.data, values.data, batch, topk);
  }

  topkToSlots(state: ExecutionState, topkIdx: Tensor, kvTokenIndptrD: Tensor, pageIndices: Tensor, indptr: Tensor, lastPageLen: Tensor, batchIndices: Tensor, pageSize: number, maxKv: number, _cacheIdx: number, _contextParallel?: boolean, cpWorldSize: number = 0, cpRank: number = 0, providedLength?: Tensor): { layer: SlotSet, group: SlotSet } {
    // Device level operates on the resolved cpWorldSize/cpRank. Single-GPU
    // (non-CP) callers reach here with the defaults (cpWorldSize 0 = paged);
    // ParallelOps resolves the flat/paged mode from cacheIdx and passes
    // cpWorldSize/cpRank per shard, so cacheIdx is unused here.
    const totalQ = topkIdx.shape[0];
    const topk = topkIdx.shape[1];
    // Constant-size across totalQ / graph variants so the allocator layout
    // stays stable under capture. ParallelOps pre-allocates one Replicated
    // tensor and hands down its shards, so the per-device alloc order stays
    // exactly as it was before the group/layer pair was folded in here.
    const topkLength = providedLength ?? topkIdx.workspace.alloc([state.positionIds.shape[0]], "I32");
    // Allocate the max footprint ([maxKv, topk]) and narrow to [totalQ, topk]:
    // a plain transient alloc, but constant-sized across totalQ / graph variants
    // so the workspace allocator layout stays stable under graph capture. Hold
    // the full allocation with `using` so its disposal is deferred to when the
    // caller disposes the returned narrow view — returning a bare
    // `alloc(...).narrow(...)` would leak the parent allocation, since disposing
    // a view alone never frees its parent.
    using full = topkIdx.workspace.alloc([maxKv, topk], "I32");
    const slots = full.narrow(0, totalQ);
    getNativeAddon().topkToSlots(this.ctx, ptr(slots), ptr(topkLength), ptr(topkIdx), ptr(pageIndices), ptr(indptr), ptr(lastPageLen), ptr(batchIndices), totalQ, topk, pageSize, cpWorldSize, cpRank, ptr(kvTokenIndptrD));
    // No group concept at the device level: a following shared layer reads the
    // same slots. Hand back a viewClone so it shares this memory but disposes
    // independently of the layer's own handle.
    return {
      layer: { slots, length: topkLength },
      group: { slots: slots.viewClone(), length: topkLength.viewClone() },
    };
  }

  private indexerScoreTopkPrefill(q: Tensor, kData: Tensor, weights: Tensor, pageIndices: Tensor, pageIndptr: Tensor, lastPageLen: Tensor, qoIndptr: Tensor, scale: number, totalQ: number, idxNHeads: number, idxHeadDim: number, pageSize: number, topk: number, maxKv: number, customMask?: Tensor, maskIndptr?: Tensor, maskKvLen?: Tensor, qGlobalStart: number = 0, cpWorldSize: number = 0, cpRank: number = 0, globalLastPageLen?: Tensor, kvTokenIndptr?: Tensor): { values: Tensor, indices: Tensor } {
    const numSplits = Math.min(256, Math.max(1, Math.ceil(maxKv / 256)));
    const indices = q.workspace.alloc([totalQ, topk], "I32");
    const values = q.workspace.alloc([totalQ, topk], "BF16");
    using scores = q.workspace.alloc([totalQ, maxKv], "BF16");
    using rowLen = q.workspace.alloc([totalQ], "I32");
    using coarseHist = q.workspace.alloc([totalQ, 1024], "I32");
    using fineHist = q.workspace.alloc([totalQ, 64], "I32");
    using meta = q.workspace.alloc([totalQ, 4], "I32");
    getNativeAddon().indexerScoreTopkPrefill(this.ctx, indices.data, values.data, q.data, kData.data, weights.data, pageIndices.data, pageIndptr.data, lastPageLen.data, qoIndptr.data, scale, totalQ, idxNHeads, idxHeadDim, pageSize, topk, 1 /*causal*/, qGlobalStart, customMask ? customMask.data : 0, maskIndptr ? maskIndptr.data : 0, maskKvLen ? maskKvLen.data : 0, scores.data, rowLen.data, maxKv, coarseHist.data, fineHist.data, meta.data, numSplits, cpWorldSize, cpRank, globalLastPageLen ? globalLastPageLen.data : 0, kvTokenIndptr?.data ?? 0);
    return { values, indices };
  }

  private indexerScoreTopkV2(q: Tensor, kData: Tensor, weights: Tensor, pageIndices: Tensor, pageIndptr: Tensor, lastPageLen: Tensor, qoIndptr: Tensor, scale: number, totalQ: number, idxNHeads: number, idxHeadDim: number, pageSize: number, topk: number, maxKv: number, causal = 0, customMask?: Tensor, maskIndptr?: Tensor, maskKvLen?: Tensor, qGlobalStart = 0, cpWorldSize: number = 0, cpRank: number = 0, globalLastPageLen?: Tensor, kvTokenIndptr?: Tensor): { values: Tensor, indices: Tensor } {
    const numSplits = Math.min(256, Math.max(1, Math.ceil(maxKv / 256)));
    const indices = q.workspace.alloc([totalQ, topk], "I32");
    const values = q.workspace.alloc([totalQ, topk], "BF16");
    using scores = q.workspace.alloc([totalQ, maxKv], "BF16");
    using rowLen = q.workspace.alloc([totalQ], "I32");
    using hist = q.workspace.alloc([totalQ, 65536], "I32");
    using meta = q.workspace.alloc([totalQ, 4], "I32");
    getNativeAddon().indexerScoreTopkV2(this.ctx, indices.data, values.data, q.data, kData.data, weights.data, pageIndices.data, pageIndptr.data, lastPageLen.data, qoIndptr.data, scale, totalQ, idxNHeads, idxHeadDim, pageSize, topk, causal, qGlobalStart, customMask ? customMask.data : 0, maskIndptr ? maskIndptr.data : 0, maskKvLen ? maskKvLen.data : 0, scores.data, rowLen.data, hist.data, meta.data, maxKv, numSplits, cpWorldSize, cpRank, globalLastPageLen ? globalLastPageLen.data : 0, kvTokenIndptr?.data ?? 0);
    return { values, indices };
  }

  hostPointerToBuffer(ptr: number, size: number): Buffer {
    return getNativeAddon().hostPointerToBuffer(ptr, size);
  }

  batchDecodePlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, indptrH: Tensor, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, enableCudaGraph: boolean): void {
    getNativeAddon().batchDecodePlan(this.ctx, ptr(floatWs), floatWsSize, ptr(intWs), ptr(pinnedIntWs), intWsSize, ptr(planInfo), ptr(indptrH), batchSize, numQoHeads, numKvHeads, headDim, pageSize, enableCudaGraph);
  }

  batchDecodeRun(state: ExecutionState, q: Tensor, o: Tensor, kData: Tensor, vData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, numQoHeads: number, numKvHeads: number, headDim: number, smScale: number): void {
    const pageSize = kData.shape[1] / (numKvHeads * headDim);
    getNativeAddon().batchDecodeRun(this.ctx, ptr(q), ptr(o), ptr(kData), ptr(vData), ptr(indices), ptr(indptrD), ptr(lastPageLen), ptr(floatWs), ptr(intWs), ptr(planInfo), state.batchSize, numQoHeads, numKvHeads, headDim, pageSize, smScale);
  }

  batchPrefillPagedPlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, qoIndptrH: Tensor, pagedKvIndptrH: Tensor, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, maskMode: MaskMode): void {
    getNativeAddon().batchPrefillPagedPlan(this.ctx, ptr(floatWs), floatWsSize, ptr(intWs), ptr(pinnedIntWs), intWsSize, ptr(planInfo), ptr(qoIndptrH), ptr(pagedKvIndptrH), totalQoRows, batchSize, numQoHeads, numKvHeads, headDim, pageSize, maskMode);
  }

  batchPrefillPagedRun(state: ExecutionState, q: Tensor, o: Tensor, kData: Tensor, vData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, floatWs: Tensor, intWs: Tensor, qIndptrD: Tensor, planInfo: Tensor, numQoHeads: number, numKvHeads: number, headDim: number, qStrideN: number, qStrideH: number, maskMode: MaskMode, smScale: number): void {
    const pageSize = kData.shape[1] / (numKvHeads * headDim);
    getNativeAddon().batchPrefillPagedRun(this.ctx, ptr(q), ptr(o), ptr(kData), ptr(vData), ptr(indices), ptr(indptrD), ptr(lastPageLen), ptr(floatWs), ptr(intWs), ptr(qIndptrD), ptr(planInfo), state.totalTokens, state.batchSize, numQoHeads, numKvHeads, headDim, pageSize, qStrideN, qStrideH, maskMode, smScale);
  }

  batchPrefillRaggedPlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, qoIndptrH: Tensor, kvIndptrH: Tensor, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, maskMode: MaskMode): void {
    getNativeAddon().batchPrefillRaggedPlan(this.ctx, ptr(floatWs), floatWsSize, ptr(intWs), ptr(pinnedIntWs), intWsSize, ptr(planInfo), ptr(qoIndptrH), ptr(kvIndptrH), totalQoRows, batchSize, numQoHeads, numKvHeads, headDim, maskMode);
  }

  batchPrefillRaggedRun(state: ExecutionState, q: Tensor, k: Tensor, v: Tensor, o: Tensor, floatWs: Tensor, intWs: Tensor, qIndptrD: Tensor, kvIndptrD: Tensor, planInfo: Tensor, numQoHeads: number, numKvHeads: number, headDim: number, qStrideN: number, qStrideH: number, kvStrideN: number, kvStrideH: number, vStrideN: number, vStrideH: number, maskMode: MaskMode, smScale: number): void {
    getNativeAddon().batchPrefillRaggedRun(this.ctx, ptr(q), ptr(k), ptr(v), ptr(o), ptr(floatWs), ptr(intWs), ptr(qIndptrD), ptr(kvIndptrD), ptr(planInfo), state.totalTokens, state.batchSize, numQoHeads, numKvHeads, headDim, qStrideN, qStrideH, kvStrideN, kvStrideH, vStrideN, vStrideH, maskMode, smScale);
  }

  graphBeginCapture(): void {
    getNativeAddon().graphBeginCapture(this.ctx);
    this.capturing = true;
  }

  graphEndCapture(): number {
    const graph = getNativeAddon().graphEndCapture(this.ctx);
    if (!graph) throw new Error("CUDA graph capture failed");
    this.capturing = false;
    return graph;
  }

  graphInstantiate(graph: number): number {
    const exec = getNativeAddon().graphInstantiate(this.ctx, graph);
    if (!exec) throw new Error("CUDA graph instantiation failed");
    return exec;
  }

  graphLaunch(graphExec: number): void {
    getNativeAddon().graphLaunch(this.ctx, graphExec);
  }

  graphDestroy(graph: number): void {
    getNativeAddon().graphDestroy(this.ctx, graph);
  }

  graphExecDestroy(graphExec: number): void {
    getNativeAddon().graphExecDestroy(this.ctx, graphExec);
  }

  mlaPrefillPlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, qoIndptrH: Tensor, kvIndptrH: Tensor, kvLenH: Tensor, lastPageLenH: Tensor, batchSize: number, numHeads: number, headDimO: number, causal: boolean, pageSize: number, seqKvLens: number[], contextParallel?: boolean, cpWorldSize: number = 0, cpRank: number = 0): void {
    getNativeAddon().mlaPrefillPlan(this.ctx, ptr(floatWs), floatWsSize, ptr(intWs), ptr(pinnedIntWs), intWsSize, ptr(planInfo), ptr(qoIndptrH), ptr(kvIndptrH), ptr(kvLenH), batchSize, numHeads, headDimO, causal, cpWorldSize, cpRank);
  }

  mlaPrefillRun(state: ExecutionState, qNope: Tensor, qPe: Tensor, ckvData: Tensor, kpeData: Tensor, kvIndices: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, smScale: number, maskMode: MaskMode, cpWorldSize: number = 0, cpRank: number = 0, customMask?: Tensor, maskIndptr?: Tensor, maskKvLen?: Tensor): { o: Tensor, lse: Tensor } {
    const numHeads = qNope.shape[1];
    const headDimCkv = ckvData.shape[2];
    const headDimKpe = kpeData.shape[2];
    const pageSize = ckvData.shape[1];
    const ckvStridePage = pageSize * headDimCkv;
    const ckvStrideN = headDimCkv;
    const kpeStridePage = kpeData.shape[1] * headDimKpe;
    const kpeStrideN = headDimKpe;
    const qNopeStrideN = qNope.shape[1] * qNope.shape[2];
    const qNopeStrideH = qNope.shape[2];
    const qPeStrideN = qPe.shape[1] * qPe.shape[2];
    const qPeStrideH = qPe.shape[2];
    const totalTokens = state.totalTokens;
    const oStrideN = headDimCkv;
    const oStrideH = totalTokens * headDimCkv;
    const o = qNope.workspace.alloc([1, numHeads, totalTokens, headDimCkv], qNope.type);
    const lse = qNope.workspace.alloc([totalTokens, numHeads], "F32");
    getNativeAddon().mlaPrefillRun(this.ctx, ptr(qNope), ptr(qPe), ptr(ckvData), ptr(kpeData), ptr(kvIndices), ptr(o), ptr(floatWs), ptr(intWs), ptr(planInfo), numHeads, pageSize, maskMode, smScale, qNopeStrideN, qNopeStrideH, qPeStrideN, qPeStrideH, ckvStridePage, ckvStrideN, kpeStridePage, kpeStrideN, oStrideN, oStrideH, headDimCkv, headDimKpe, ptr(lse), cpWorldSize, cpRank, customMask ? ptr(customMask) : 0, maskIndptr ? ptr(maskIndptr) : 0, maskKvLen ? ptr(maskKvLen) : 0);
    return { o, lse };
  }

  mlaDecodePlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, indptrH: Tensor, lastPageLenH: Tensor, batchSize: number, numQoHeads: number, pageSize: number, enableCudaGraph: boolean, headDimCkv: number, headDimKpe: number, _seqKvLens: number[], contextParallel?: boolean, cpWorldSize?: number, cpRank?: number): void {
    getNativeAddon().mlaDecodePlan(this.ctx, ptr(floatWs), floatWsSize, ptr(intWs), ptr(pinnedIntWs), intWsSize, ptr(planInfo), ptr(indptrH), batchSize, numQoHeads, pageSize, enableCudaGraph, headDimCkv, headDimKpe);
  }

  sparseMlaDecodePlan(_lastPageLenH: Tensor, _batchSize: number, _seqKvLens: number[], _pageSize: number, _contextParallel: boolean): void {
  }

  mlaDecodeRun(state: ExecutionState, qNope: Tensor, qPe: Tensor, ckvData: Tensor, kpeData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, smScale: number): { o: Tensor, lse: Tensor } {
    const numQoHeads = qNope.shape[1];
    const headDimCkv = ckvData.shape[2];
    const headDimKpe = kpeData.shape[2];
    const pageSize = ckvData.shape[1];
    const o = qNope.workspace.alloc([state.batchSize, numQoHeads, 1, headDimCkv], qNope.type);
    const lse = qNope.workspace.alloc([state.batchSize, numQoHeads], "F32");
    getNativeAddon().mlaDecodeRun(this.ctx, ptr(qNope), ptr(qPe), ptr(ckvData), ptr(kpeData), ptr(indices), ptr(indptrD), ptr(lastPageLen), ptr(o), ptr(floatWs), ptr(intWs), ptr(planInfo), state.batchSize, numQoHeads, pageSize, smScale, headDimCkv, headDimKpe, ptr(lse));
    return { o, lse };
  }

  mlaKvCacheAppend(_state: ExecutionState, _cacheIdx: number, ckvData: Tensor, kpeData: Tensor | null, indices: Tensor, indptr: Tensor, lastPageLen: Tensor, appendCkv: Tensor, appendKpe: Tensor | null, batchIndices: Tensor, positions: Tensor, nnz: number, headDimCkv: number, headDimKpe: number, appendCkvStrideN: number, appendKpeStrideN: number, pageSize: number = ckvData.shape[1], cpWorldSize: number = 0, cpRank: number = 0): { ckv: Tensor; kpe?: Tensor } {
    getNativeAddon().mlaKvCacheAppend(this.ctx, ptr(ckvData), kpeData ? ptr(kpeData) : 0, ptr(indices), ptr(indptr), ptr(lastPageLen), ptr(appendCkv), appendKpe ? ptr(appendKpe) : 0, ptr(batchIndices), ptr(positions), nnz, pageSize, headDimCkv, headDimKpe, appendCkvStrideN, appendKpeStrideN, cpWorldSize, cpRank);
    return { ckv: ckvData.viewClone(), kpe: kpeData?.viewClone() };
  }

  indexerKvCacheAppendFlat(kData: Tensor, appendK: Tensor, kvTokenIndptr: Tensor, batchIndices: Tensor, positions: Tensor, nnz: number, headDim: number, appendStrideN: number): void {
    getNativeAddon().indexerKvCacheAppendFlat(this.ctx, ptr(kData), ptr(appendK), ptr(kvTokenIndptr), ptr(batchIndices), ptr(positions), nnz, headDim, appendStrideN);
  }

  concatAndCacheDsMla(state: ExecutionState, cacheIdx: number, kvCache: Tensor, appendCkv: Tensor, appendKpe: Tensor, indices: Tensor | undefined, indptr: Tensor, batchIndices: Tensor, positions: Tensor, nnz: number, kvLoraRank: number, peDim: number, appendCkvStrideN: number, appendKpeStrideN: number, pageSize: number = kvCache.shape[1], cpWorldSize: number = 0, cpRank: number = 0): Tensor {
    getNativeAddon().concatAndCacheDsMla(this.ctx, ptr(kvCache), ptr(appendCkv), ptr(appendKpe), ptr(indices), ptr(indptr), ptr(batchIndices), ptr(positions), nnz, pageSize, kvLoraRank, peDim, appendCkvStrideN, appendKpeStrideN, cpWorldSize, cpRank);
    return kvCache.viewClone();
  }

  sparseMlaPrepareCache(state: ExecutionState, groupSlots: Tensor, cacheIdx: number, kvCache: Tensor, appendCkv: Tensor, appendKpe: Tensor, topk: Tensor | undefined, indices: Tensor | null, indptr: Tensor, batchIndices: Tensor, positions: Tensor, nnz: number, kvLoraRank: number, peDim: number, appendCkvStrideN: number, appendKpeStrideN: number): Tensor {
    return kvCache.viewClone();
  }

  gdnRecurrentStep(state: ExecutionState, output: Tensor, recurrentState: Tensor, qkv: Tensor, aRaw: Tensor, bRaw: Tensor, aLog: Tensor, dtBias: Tensor, numHeads: number, dK: number, dV: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void {
    if (recurrentState.type !== "F32") throw new Error(`gdnRecurrentStep: state must be F32, got ${recurrentState.type}`);
    if (aLog.type !== "F32") throw new Error(`gdnRecurrentStep: aLog must be F32, got ${aLog.type}`);
    if (dtBias.type !== "F32") throw new Error(`gdnRecurrentStep: dtBias must be F32, got ${dtBias.type}`);
    getNativeAddon().gdnRecurrentStep(this.ctx, output.data, recurrentState.data, qkv.data, aRaw.data, bRaw.data, aLog.data, dtBias.data, numHeads, dK, dV, state.batchSize, stateStride, qkvChStride, qkvSeqStride);
  }

  gdnPrefill(state: ExecutionState, output: Tensor, recurrentState: Tensor, qkv: Tensor, aRaw: Tensor, bRaw: Tensor, aLog: Tensor, dtBias: Tensor, cuSeqlens: Tensor, numHeads: number, dK: number, dV: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void {
    if (recurrentState.type !== "F32") throw new Error(`gdnPrefill: state must be F32, got ${recurrentState.type}`);
    if (aLog.type !== "F32") throw new Error(`gdnPrefill: aLog must be F32, got ${aLog.type}`);
    if (dtBias.type !== "F32") throw new Error(`gdnPrefill: dtBias must be F32, got ${dtBias.type}`);
    if (cuSeqlens.type !== "I32") throw new Error(`gdnPrefill: cuSeqlens must be I32, got ${cuSeqlens.type}`);
    getNativeAddon().gdnPrefill(this.ctx, output.data, recurrentState.data, qkv.data, aRaw.data, bRaw.data, aLog.data, dtBias.data, cuSeqlens.data, state.totalTokens, numHeads, dK, dV, state.batchSize, stateStride, qkvChStride, qkvSeqStride);
  }

  sparseMlaPrefill(state: ExecutionState, qAbsorbed: Tensor, qPe: Tensor, kvCache: Tensor, indices: Tensor, topk: number, smScale: number, topkLength: Tensor, _pageIndptrD: Tensor, _lastPageLen: Tensor, _kvTokenIndptrD: Tensor): { o: Tensor, lse: Tensor } {
    const numTokens = state.totalTokens;
    const numHeads = qAbsorbed.shape[1];
    const headDim = qAbsorbed.shape[2];
    const elemBytes = SafeTensorFile.dtypeBytes(kvCache.type);
    const pageBlockSize = kvCache.shape[1];
    const effectiveStrideKvBlock = pageBlockSize * kvCache.shape[2] * elemBytes;
    if (pageBlockSize !== 64) throw new Error(`sparseMlaPrefill: SM120 kernel requires pageBlockSize=64, got ${pageBlockSize} ${kvCache.shape}`);
    using q = qAbsorbed.cat([qPe], 2);
    // Pad to 16 tokens so cuBLAS TMA kernels (16-wide n-tile) don't over-read
    // the allocation when numTokens < 16.
    const oTokens = Math.max(numTokens, 16);
    using oFull = q.workspace.alloc([oTokens, numHeads, headDim], "BF16");
    const o = numTokens < 16 ? oFull.narrow(0, numTokens) : oFull.viewClone();
    const lse = q.workspace.alloc([numTokens, numHeads], "F32");
    // Small query counts (e.g. MTP tree verify) starve the prefill kernel: its
    // grid is only numTokens × ceil(NUM_HEADS/HPB) CTAs, leaving the GPU idle.
    // Route to the split-K decode kernel — same mask-free slot attention, but
    // numTokens × ceil(topk/64) CTAs — which fills the SMs. Correctness is
    // identical (causality lives in the slots, not the kernel).
    if (numTokens <= SPARSE_MLA_DECODE_DISPATCH_MAX) {
      const numSplits = Math.ceil(topk / 64);
      using midOut = q.workspace.alloc([numTokens, numHeads, numSplits, headDim], "BF16");
      using midLse = q.workspace.alloc([numTokens, numHeads, numSplits], "F32");
      getNativeAddon().sparseMlaDecode(this.ctx, ptr(q), ptr(kvCache), ptr(indices), ptr(midOut), ptr(midLse), ptr(o), ptr(lse), numTokens, numHeads, topk, numSplits, smScale, effectiveStrideKvBlock, 0, ptr(topkLength));
      return { o, lse };
    }
    getNativeAddon().sparseMlaPrefill(this.ctx, ptr(q), ptr(kvCache), ptr(indices), ptr(o), ptr(lse), numTokens, numHeads, topk, pageBlockSize, smScale, effectiveStrideKvBlock, ptr(topkLength));
    return { o, lse };
  }

  sparseMlaDecode(state: ExecutionState, qAbsorbed: Tensor, qPe: Tensor, kvCache: Tensor, indices: Tensor, topk: number, numSplits: number, smScale: number, chunksPerBlock: number, topkLength?: Tensor): { o: Tensor, lse: Tensor } {
    const numTokens = state.batchSize;
    const numHeads = qAbsorbed.shape[1];
    const headDim = qAbsorbed.shape[2];
    const elemBytes = SafeTensorFile.dtypeBytes(kvCache.type);
    const pageBlockSize = kvCache.shape[1];
    const effectiveStrideKvBlock = pageBlockSize * kvCache.shape[2] * elemBytes;
    if (pageBlockSize !== 64) throw new Error(`sparseMlaDecode: SM120 kernel requires pageBlockSize=64, got ${pageBlockSize}`);
    using q = qAbsorbed.cat([qPe], 2);
    // Pad to 16 tokens so cuBLAS TMA kernels (16-wide n-tile) don't over-read
    // the allocation when numTokens < 16.
    const oTokens = Math.max(numTokens, 16);
    using oFull = q.workspace.alloc([oTokens, numHeads, headDim], "BF16");
    const o = numTokens < 16 ? oFull.narrow(0, numTokens) : oFull.viewClone();
    const lse = q.workspace.alloc([numTokens, numHeads], "F32");
    using midOut = q.workspace.alloc([numTokens, numHeads, numSplits, headDim], "BF16");
    using midLse = q.workspace.alloc([numTokens, numHeads, numSplits], "F32");
    getNativeAddon().sparseMlaDecode(this.ctx, ptr(q), ptr(kvCache), ptr(indices), ptr(midOut), ptr(midLse), ptr(o), ptr(lse), numTokens, numHeads, topk, numSplits, smScale, effectiveStrideKvBlock, chunksPerBlock, topkLength ? ptr(topkLength) : 0);
    return { o, lse };
  }

  cpMergeTree(vPtrs: number[], lsePtrs: number[], numShards: number, outputV: Tensor, outputLse: Tensor | null, numel: number, batchSize: number, numHeads: number, vHeadDim: number, shardNHeads?: number, headOffset?: number, inputNHeads?: number): void {
    const snh = shardNHeads ?? numHeads;
    const ho = headOffset ?? 0;
    const inh = inputNHeads ?? numHeads;
    const v = new Array<number>(8).fill(0);
    for (let i = 0; i < numShards; i++) v[i] = vPtrs[i];
    const lse = new Array<number>(8).fill(0);
    for (let i = 0; i < numShards; i++) lse[i] = lsePtrs[i];
    getNativeAddon().cpMergeTree(
      this.ctx,
      v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
      lse[0], lse[1], lse[2], lse[3], lse[4], lse[5], lse[6], lse[7],
      numShards, ptr(outputV), outputLse ? ptr(outputLse) : 0, numel, batchSize, numHeads, vHeadDim,
      snh, ho, inh,
    );
  }

  /**
   * Push-based CP merge, phase 1. Writes each peer's head slice into that peer's
   * staging buffer at slot `rank`. `stageVPtrs`/`stageLsePtrs` must already be
   * rotated by rank host-side: entry k belongs to peer (rank + k) % worldSize.
   */
  cpMergeScatter(localV: Tensor, localLse: Tensor, stageVPtrs: number[], stageLsePtrs: number[], worldSize: number, batchSize: number, shardNHeads: number, vHeadDim: number, inputNHeads: number, numHeads: number, rank: number): void {
    const dv = new Array<number>(8).fill(0);
    const dl = new Array<number>(8).fill(0);
    for (let i = 0; i < worldSize; i++) {
      dv[i] = stageVPtrs[i];
      dl[i] = stageLsePtrs[i];
    }
    getNativeAddon().cpMergeScatter(
      this.ctx, ptr(localV), ptr(localLse),
      dv[0], dv[1], dv[2], dv[3], dv[4], dv[5], dv[6], dv[7],
      dl[0], dl[1], dl[2], dl[3], dl[4], dl[5], dl[6], dl[7],
      worldSize, batchSize, shardNHeads, vHeadDim, inputNHeads, numHeads, rank,
    );
  }

  /** Push-based CP merge, phase 2. Must follow a barrier over phase 1. */
  cpMergeLocal(stageV: Tensor, stageLse: Tensor, outputV: Tensor, outputLse: Tensor | null, worldSize: number, batchSize: number, shardNHeads: number, vHeadDim: number): void {
    getNativeAddon().cpMergeLocal(
      this.ctx, ptr(stageV), ptr(stageLse), ptr(outputV), outputLse ? ptr(outputLse) : 0,
      worldSize, batchSize, shardNHeads, vHeadDim,
    );
  }

  cpCorrectAttnOut(vOut: Tensor, lses: Tensor, globalLse: Tensor | null, batchSize: number, numHeads: number, vHeadDim: number, worldSize: number, rank: number): void {
    getNativeAddon().cpCorrectAttnOut(
      this.ctx, ptr(vOut), ptr(lses), globalLse ? ptr(globalLse) : 0,
      batchSize, numHeads, vHeadDim, worldSize, rank,
    );
  }

  ncclReduceScatter(comm: number, sendbuff: Tensor, recvbuff: Tensor, recvcount: number, datatype: number, op: number): void {
    getNativeAddon().ncclReduceScatter(comm, this.ctx, ptr(sendbuff), ptr(recvbuff), recvcount, datatype, op);
  }

  p2pArrive(instance: number, peerRank: number = -1): void {
    getNativeAddon().p2pArrive(this.ctx, instance, peerRank);
  }

  p2pWait(instance: number, peerRank: number = -1): void {
    getNativeAddon().p2pWait(this.ctx, instance, peerRank);
  }

  p2pBarrier(instance: number, peerRank: number = -1): void {
    this.p2pArrive(instance, peerRank);
    this.p2pWait(instance, peerRank);
  }
}

export function f32ToBf16Bytes(arr: Float32Array): Buffer {
  const u32 = new Uint32Array(arr.buffer);
  const u16 = new Uint16Array(u32.length);
  for (let i = 0; i < u32.length; i++) {
    u16[i] = u32[i] >>> 16;
  }
  return Buffer.from(u16.buffer);
}

export function bf16BytesToF32(buf: Buffer): Float32Array {
  const u16 = new Uint16Array(buf.buffer, buf.byteOffset, buf.length / 2);
  const u32 = new Uint32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    u32[i] = u16[i] << 16;
  }
  return new Float32Array(u32.buffer);
}

export const BF16 = 2;
export const I32 = 4;
export const F32 = 4;

const MEMCPY_H2H = 0;
const MEMCPY_H2D = 1;
const MEMCPY_D2H = 2;
const MEMCPY_D2D = 3;

function memcpyKindToNative(kind: MemcpyKind): number {
  switch (kind) {
    case MemcpyKind.HostToHost: return MEMCPY_H2H;
    case MemcpyKind.HostToDevice: return MEMCPY_H2D;
    case MemcpyKind.DeviceToHost: return MEMCPY_D2H;
    case MemcpyKind.DeviceToDevice: return MEMCPY_D2D;
    case MemcpyKind.Default: return 4;
  }
}

export const NCCL_UNIQUE_ID_BYTES = 128;
export const NCCL_INT8 = 0;
export const NCCL_UINT8 = 1;
export const NCCL_INT32 = 2;
export const NCCL_UINT32 = 3;
export const NCCL_INT64 = 4;
export const NCCL_UINT64 = 5;
export const NCCL_FLOAT16 = 6;
export const NCCL_FLOAT32 = 7;
export const NCCL_FLOAT64 = 8;
export const NCCL_BFLOAT16 = 9;
export const NCCL_SUM = 0;
export const NCCL_PROD = 1;
export const NCCL_MAX = 2;
export const NCCL_MIN = 3;
