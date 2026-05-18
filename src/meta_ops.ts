import { DeviceOps, StridedMmap, TensorParallelism } from "./device_ops";
import { MemcpyKind, Tensor } from "./tensor";
import { WorkspaceBase } from "./workspace";

export class MetaTensor extends Tensor {
    private fakePinned?: Buffer;

    free(): void {
    }

    h2d(data: Buffer, size?: number): void {
    }

    d2h(buf: Buffer, size?: number): void {
    }

    withPinnedBuffer(fn: (buf: Buffer) => void): void {
        this.fakePinned ??= Buffer.alloc(this.allocSize);
        fn(this.fakePinned);
    }

    readPinnedBuffer(): Buffer {
        this.fakePinned ??= Buffer.alloc(this.allocSize);
        return this.fakePinned;
    }

    fill(value: number, n: number): void {
    }

    mmapLoad(mmapPtr: number, offset: number, nbytes: number, strided?: StridedMmap): Promise<void> {
      return Promise.resolve();
    }

    mmapLoadAsync(mmapPtr: number, offset: number, nbytes: number): Promise<void> {
      return Promise.resolve();
    }

    memcpy2dHostToDeviceAsync(dstOffset: number, dpitch: number, src: number, spitch: number, width: number, height: number): Promise<void> {
      return Promise.resolve();
    }

    memcpy(src: Tensor, size?: number, kind?: MemcpyKind): void {
    }

    memcpy2d(dstOffset: number, dpitch: number, src: number, spitch: number, width: number, height: number, kind: MemcpyKind): void {
    }

    linear(weight: Tensor, batch: number): Tensor {
        super.linear(weight, batch);
        const n = weight.shape[0];
        return this.workspace.alloc([batch, n], this.type);
    }

    bmm(B: Tensor, batch: number, M: number, N: number, K: number, transA: boolean = false, transB: boolean = false): Tensor {
        return this.workspace.alloc([batch * M, N], this.type);
    }

    writePointers(tensors: Tensor[]): void {
    }

    rmsnorm(weight: Tensor, eps: number, dim: number, batch: number): Tensor {
        super.rmsnorm(weight, eps, dim, batch);
        return this.workspace.alloc([batch, dim], this.type);
    }

    fusedAddRmsnorm(input: Tensor, weight: Tensor, eps: number, dim: number, batch: number): { normed: Tensor, residual: Tensor } {
        super.fusedAddRmsnorm(input, weight, eps, dim, batch);
        const normed = this.workspace.alloc([batch, dim], this.type);
        const residual = this.workspace.alloc([batch, dim], this.type);
        return { normed, residual };
    }

    fusedNormRope(weight: Tensor, cos: Tensor, sin: Tensor, eps: number, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, inStride?: number, interleaved?: boolean): Tensor {
        super.fusedNormRope(weight, cos, sin, eps, ropeDim, headDim, nHeads, seqLen, batch, inStride, interleaved);
        return this.workspace.alloc([batch, nHeads, seqLen, headDim], this.type);
    }

    embedding(ids: Tensor, hidden: number, seqLen: number): Tensor {
        super.embedding(ids, hidden, seqLen);
        return ids.workspace.alloc([seqLen, hidden], this.type);
    }

    siluAndMul(up: Tensor, intermediate: number, batch: number): Tensor {
        super.siluAndMul(up, intermediate, batch);
        return this.workspace.alloc([batch, intermediate], this.type);
    }

    arange(start: number, step: number, count: number): void {
        super.arange(start, step, count);
    }

    argmax(): Tensor {
        super.argmax();
        const { indices } = this.max();
        return indices;
    }

    max(offset: number = 0): { values: Tensor, indices: Tensor } {
        super.max(offset);
        const batch = this.shape[0];
        const values = this.workspace.alloc([batch], this.type);
        const indices = this.workspace.alloc([batch], "I32");
        return { values, indices };
    }

    indexSelect(indices: Tensor, dim: number, batch: number): Tensor {
        super.indexSelect(indices, dim, batch);
        return this.workspace.alloc([batch, dim], this.type);
    }

    gather(indices: Tensor, k: number, inDim: number, batch: number): Tensor {
        super.gather(indices, k, inDim, batch);
        return this.workspace.alloc([batch, k], this.type);
    }

    gdnRecurrentStep(state: Tensor, qkv: Tensor, aRaw: Tensor, bRaw: Tensor, aLog: Tensor, dtBias: Tensor, numHeads: number, dK: number, dV: number, batchSize: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void {
        super.gdnRecurrentStep(state, qkv, aRaw, bRaw, aLog, dtBias, numHeads, dK, dV, batchSize, stateStride, qkvChStride, qkvSeqStride);
    }

    gdnPrefill(state: Tensor, qkv: Tensor, aRaw: Tensor, bRaw: Tensor, aLog: Tensor, dtBias: Tensor, cuSeqlens: Tensor, totalSeqLen: number, numHeads: number, dK: number, dV: number, batchSize: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void {
        super.gdnPrefill(state, qkv, aRaw, bRaw, aLog, dtBias, cuSeqlens, totalSeqLen, numHeads, dK, dV, batchSize, stateStride, qkvChStride, qkvSeqStride);
    }

    causalConv1d(convState: Tensor, input: Tensor, weight: Tensor, cuSeqlens: Tensor, convDim: number, totalSeqLen: number, kernelSize: number, batchSize: number, convStateStride: number, chStride: number, seqStride: number): void {
        super.causalConv1d(convState, input, weight, cuSeqlens, convDim, totalSeqLen, kernelSize, batchSize, convStateStride, chStride, seqStride);
    }

    causalConv1dUpdate(convState: Tensor, input: Tensor, weight: Tensor, convDim: number, kernelSize: number, batchSize: number, convStateStride: number): Tensor {
        super.causalConv1dUpdate(convState, input, weight, convDim, kernelSize, batchSize, convStateStride);
        return this.workspace.alloc([batchSize * convDim], this.type);
    }

    rmsnormGated(input: Tensor, gate: Tensor, weight: Tensor, eps: number, dim: number, batch: number): void {
        super.rmsnormGated(input, gate, weight, eps, dim, batch);
    }

    gateSigmoidMul(gate: Tensor, batchSeq: number, numHeads: number, headDim: number): void {
        super.gateSigmoidMul(gate, batchSeq, numHeads, headDim);
    }

    rotaryEmbedding(positionIds: Tensor, dimHalf: number, batch: number, seqLen: number): { cos: Tensor, sin: Tensor } {
        super.rotaryEmbedding(positionIds, dimHalf, batch, seqLen);
        const hd = dimHalf * 2;
        const cos = positionIds.workspace.alloc([batch, seqLen, hd], this.type);
        const sin = positionIds.workspace.alloc([batch, seqLen, hd], this.type);
        return { cos, sin };
    }

    ropeTranspose(cos: Tensor, sin: Tensor, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, inStride?: number, interleaved?: boolean): Tensor {
        super.ropeTranspose(cos, sin, ropeDim, headDim, nHeads, seqLen, batch, inStride, interleaved);
        return this.workspace.alloc([batch * seqLen, nHeads, headDim], this.type);
    }

    applyRotaryPosEmb(cos: Tensor, sin: Tensor, ropeDim: number, nHeads: number, seqLen: number, batch: number, unsqueezeDim: number, interleaved?: boolean): Tensor {
        return this.workspace.alloc(this.shape, this.type);
    }

    mlaVExpand(vProj: Tensor, kvLoraRank: number, vHeadDim: number, nHeads: number, seqLen: number, batch: number, _lse?: Tensor, _headOffset?: number, _attnNHeads?: number): Tensor {
        super.mlaVExpand(vProj, kvLoraRank, vHeadDim, nHeads, seqLen, batch);
        const BS = batch * seqLen;
        return this.workspace.alloc([BS, nHeads * vHeadDim], this.type);
    }

    sigmoid(): Tensor {
        const n = this.shape.reduce((a, b) => a * b, 1);
        return this.workspace.alloc(this.shape, this.type);
    }

    topk(k: number, dim: number): { values: Tensor, indices: Tensor } {
        const batch = this.shape.reduce((a, b) => a * b, 1) / dim;
        const values = this.workspace.alloc([batch, k], this.type);
        const indices = this.workspace.alloc([batch, k], "I32");
        return { values, indices };
    }

    add(other: Tensor, n?: number): Tensor {
        if (this.shape.length === 2 && other.shape.length === 1 && this.shape[1] === other.shape[0]) {
            return this.workspace.alloc(this.shape, this.type);
        }
        const count = n ?? this.shape.reduce((a, b) => a * b, 1);
        return this.workspace.alloc(this.shape, this.type);
    }

    scaleInPlace(scale: number, n: number): void {
    }

    mul(other: Tensor, n?: number): Tensor {
        if (this.shape.length === 2 && other.shape.length === 1 && this.shape[1] === other.shape[0]) {
            return this.workspace.alloc(this.shape, this.type);
        }
        const count = n ?? this.shape.reduce((a, b) => a * b, 1);
        return this.workspace.alloc(this.shape, this.type);
    }

    cat(tensors: Tensor[], dim: number): Tensor {
        super.cat(tensors, dim);
        const outShape = [...this.shape];
        for (const t of tensors) outShape[dim] += t.shape[dim];
        return this.workspace.alloc(outShape, this.type);
    }

    scatterScalar(indices: Tensor, value: number, k: number, outDim: number, batch: number): void {
    }

    maskedFill(mask: Tensor, value: number, n: number): void {
    }

    applyRotaryPosEmbPartial(cos: Tensor, sin: Tensor, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, unsqueezeDim: number, interleaved?: boolean): Tensor {
        return this.workspace.alloc(this.shape, this.type);
    }

    rowScaleAdd(input: Tensor, scales: Tensor, rows: number, dim: number): void {
    }

    reduceSum(dim: number, batch: number): Tensor {
        return this.workspace.alloc([batch], this.type);
    }

    rowNormalize(scale: number, dim: number, batch: number, normalize: boolean = true): Tensor {
        return this.workspace.alloc([batch, dim], this.type);
    }

    groupMaskMul(groupMask: Tensor, numExperts: number, expertsPerGroup: number, nGroup: number, batch: number): void {
    }

    expertScale(weights: Tensor, indices: Tensor, expertId: number, topK: number, batch: number): void {
    }

    mulMatId(weights: Tensor[], expertIds: Tensor, topK: number, count: number, N: number, K: number, name: string): Tensor {
        return this.workspace.alloc([count, N], this.type);
    }

    scatterAddRows(scales: Tensor, topK: number, dim: number, numRows: number): Tensor {
        return this.workspace.alloc([numRows, dim], this.type);
    }

    sampleBatch(outTokens: Tensor, topkVals: Tensor, topkIdxs: Tensor, workspace: Tensor, logits: Tensor, penaltyTokens: Tensor, penaltyCount: Tensor, maxWindow: number, vocabSize: number, batchSize: number, temperatures: Tensor, repPenalties: Tensor, presPenalties: Tensor, topKs: Tensor, topPs: Tensor, stepCounter: Tensor, maxEffectiveK: number): void {
    }
}

export class MetaOps implements DeviceOps {
    totalAllocs = 0;
    totalBytes = 0;

    newTensor(workspace: WorkspaceBase, shape: number[], type: string, pinned: boolean, name?: string, parallelism?: TensorParallelism): Tensor {
        const size = Tensor.byteCount(shape, type);
        this.totalAllocs++;
        this.totalBytes += size;
        return new MetaTensor(workspace, 0, size, shape, type, name, pinned, undefined);
    }

    wrapTensor(workspace: WorkspaceBase, data: number, allocSize: number, shape: number[], type: string, pinned: boolean, view: Tensor | undefined): Tensor {
        return new MetaTensor(workspace, data, allocSize, shape, type, undefined, pinned, view);
    }

    synchronize(): void {
    }

    synchronizeStream(streamIdx: number): void {
    }

    setStream(streamIdx: number): void {
    }

    eventRecord(eventIdx: number, streamIdx: number): void {
    }

    streamWaitEvent(streamIdx: number, eventIdx: number): void {
    }

    currentStream = 0;
    availableStreams: number[] = [];

    withStream<T>(fn: () => T): Disposable & { result: T; streamWaitEvent(): void; synchronize(): void; } {
        const result = fn();
        return {
            [Symbol.dispose]() {},
            result,
            streamWaitEvent() {},
            synchronize() {}
        };
    }

    kvCacheWrite(srcK: Tensor, srcV: Tensor, dstK: Tensor, dstV: Tensor, slotMapping: Tensor, batchSize: number, nKv: number, hd: number, pageSize: number, srcKTokenStride: number, srcKHeadStride: number, srcVTokenStride: number, srcVHeadStride: number): void {
    }

    decodeStep(positionIds: Tensor, lastPageLen: Tensor, slotMapping: Tensor, indptr: Tensor, indices: Tensor, pageSize: number, batchSize: number, steps?: number, contextParallel?: boolean, cpWorldSize?: number, cpRank?: number): void {
    }

    mlaDecodeStep(positionIds: Tensor, lastPageLen: Tensor, indptr: Tensor, pageSize: number, batchSize: number, contextParallel?: boolean, cpWorldSize?: number, cpRank?: number, steps?: number): void {
    }

    batchDecodePlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, indptrH: Tensor, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, enableCudaGraph: boolean): void {
    }

    batchDecodeRun(q: Tensor, o: Tensor, kData: Tensor, vData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, smScale: number): void {
    }

    batchPrefillPagedPlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, qoIndptrH: Tensor, pagedKvIndptrH: Tensor, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, maskMode: number): void {
    }

    batchPrefillPagedRun(q: Tensor, o: Tensor, kData: Tensor, vData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, floatWs: Tensor, intWs: Tensor, qIndptrD: Tensor, planInfo: Tensor, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, qStrideN: number, qStrideH: number, maskMode: number, smScale: number): void {
    }

  mlaPrefillPlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, qoIndptrH: Tensor, kvIndptrH: Tensor, kvLenH: Tensor, lastPageLenH: Tensor, batchSize: number, numHeads: number, headDimO: number, causal: boolean, pageSize: number, seqKvLens: number[], contextParallel?: boolean, cpWorldSize?: number, cpRank?: number): void {
  }

    mlaPrefillRun(qNope: Tensor, qPe: Tensor, ckvData: Tensor, kpeData: Tensor, kvIndices: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, numHeads: number, pageSize: number, maskMode: number, smScale: number, qNopeStrideN: number, qNopeStrideH: number, qPeStrideN: number, qPeStrideH: number, ckvStridePage: number, ckvStrideN: number, kpeStridePage: number, kpeStrideN: number, oStrideN: number, oStrideH: number, headDimCkv: number, headDimKpe: number, contextParallel?: boolean, cpWorldSize?: number, cpRank?: number): { o: Tensor, lse: Tensor } {
        return undefined as never;
    }

    mlaDecodePlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, indptrH: Tensor, lastPageLenH: Tensor, batchSize: number, numQoHeads: number, pageSize: number, enableCudaGraph: boolean, headDimCkv: number, headDimKpe: number, contextParallel?: boolean, cpWorldSize?: number, cpRank?: number, seqKvLens?: number[]): void {
    }

    mlaDecodeRun(qNope: Tensor, qPe: Tensor, ckvData: Tensor, kpeData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, batchSize: number, numQoHeads: number, pageSize: number, smScale: number, headDimCkv: number, headDimKpe: number, contextParallel?: boolean, cpWorldSize?: number, cpRank?: number): { o: Tensor, lse: Tensor } {
        return undefined as never;
    }

    mlaKvCacheAppend(ckvData: Tensor, kpeData: Tensor, indices: Tensor, indptr: Tensor, lastPageLen: Tensor, appendCkv: Tensor, appendKpe: Tensor, batchIndices: Tensor, positions: Tensor, nnz: number, pageSize: number, headDimCkv: number, headDimKpe: number, appendCkvStrideN: number, appendKpeStrideN: number, contextParallel?: boolean, cpWorldSize?: number, cpRank?: number): void {
    }

    graphBeginCapture(): void {
    }

    graphEndCapture(): number {
        return 0;
    }

    graphInstantiate(graph: number): number {
        return 0;
    }

    graphLaunch(graphExec: number): void {
    }

    graphDestroy(graph: number): void {
    }

    graphExecDestroy(graphExec: number): void {
    }
}
