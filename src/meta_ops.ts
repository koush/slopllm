import { DeviceOps, fp8ScaleShape, MaskMode, notifySynchronizedWorkspaces, SlotSet, StridedMmap, TensorParallelism, type WorkspaceMemoryStats } from "./device_ops";
import type { ExecutionState } from "./execution-workspace";
import { SafeTensorFile } from "./safetensors";
import { MemcpyKind } from "./sampling";
import { Tensor } from "./tensor";
import { WorkspaceBase } from "./workspace";
import type { HeapKey } from "./heap";

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
        super.memcpy(src, size, kind);
    }

    memcpy2d(_dstOffset: number, _dpitch: number, _src: Tensor, _srcOffset: number, _spitch: number, _width: number, _height: number, _kind: MemcpyKind): void {
        super.memcpy2d(_dstOffset, _dpitch, _src, _srcOffset, _spitch, _width, _height, _kind);
    }

    linear(weight: Tensor): Tensor {
        super.linear(weight);
        const n = weight.shape[0];
        return this.workspace.alloc([this.shape[0], n], this.type);
    }

    bmm(B: Tensor, batch: number, M: number, N: number, K: number, transA: boolean = false, transB: boolean = false): Tensor {
        return this.workspace.alloc([batch * M, N], this.type);
    }

    writePointers(tensors: Tensor[]): void {
    }

    rmsnorm(weight: Tensor, eps: number): Tensor {
        super.rmsnorm(weight, eps);
        return this.workspace.alloc(this.shape, this.type);
    }

    layernorm(weight: Tensor, bias: Tensor, eps: number): Tensor {
        super.layernorm(weight, bias, eps);
        return this.workspace.alloc(this.shape, this.type);
    }

    fusedAddRmsnorm(input: Tensor, weight: Tensor, eps: number): { normed: Tensor, residual: Tensor } {
        super.fusedAddRmsnorm(input, weight, eps);
        const normed = this.workspace.alloc(this.shape, this.type);
        const residual = this.workspace.alloc(this.shape, this.type);
        return { normed, residual };
    }

    fusedNormRope(weight: Tensor, cos: Tensor, sin: Tensor, eps: number, ropeDim: number, seqLen: number, batch: number, inStride?: number, interleaved?: boolean): Tensor {
        super.fusedNormRope(weight, cos, sin, eps, ropeDim, seqLen, batch, inStride, interleaved);
        const headDim = weight.numElements;
        const stride = inStride ?? headDim;
        const nHeads = this.shape[1] / stride;
        using reshaped = this.reshape([batch, seqLen, ...this.shape.slice(1)]);
        return this.workspace.alloc([batch, nHeads, seqLen, headDim], this.type);
    }

    embedding(ids: Tensor): Tensor {
        super.embedding(ids);
        const seqLen = ids.numElements;
        const hidden = this.shape[1];
        return ids.workspace.alloc([seqLen, hidden], this.type);
    }

    siluAndMul(up: Tensor): Tensor {
        super.siluAndMul(up);
        return this.workspace.alloc(this.shape, this.type);
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

    indexSelect(indices: Tensor, offset: number = 0): Tensor {
        super.indexSelect(indices, offset);
        const batch = indices.numElements;
        const dim = this.shape[1];
        return this.workspace.alloc([batch, dim], this.type);
    }

    gather(indices: Tensor, k: number, inDim: number, batch: number): Tensor {
        super.gather(indices, k, inDim, batch);
        return this.workspace.alloc([batch, k], this.type);
    }

    causalConv1d(convState: Tensor, input: Tensor, weight: Tensor, cuSeqlens: Tensor, convDim: number, totalSeqLen: number, kernelSize: number, batchSize: number, convStateStride: number, chStride: number, seqStride: number): void {
        super.causalConv1d(convState, input, weight, cuSeqlens, convDim, totalSeqLen, kernelSize, batchSize, convStateStride, chStride, seqStride);
    }

    causalConv1dUpdate(convState: Tensor, input: Tensor, weight: Tensor, convDim: number, kernelSize: number, batchSize: number, convStateStride: number): Tensor {
        super.causalConv1dUpdate(convState, input, weight, convDim, kernelSize, batchSize, convStateStride);
        return this.workspace.alloc([batchSize * convDim], this.type);
    }

    rmsnormGated(input: Tensor, gate: Tensor, weight: Tensor, eps: number): void {
        super.rmsnormGated(input, gate, weight, eps);
    }

    gateSigmoidMul(gate: Tensor, numHeads: number, headDim: number): void {
        super.gateSigmoidMul(gate, numHeads, headDim);
    }

    rotaryEmbedding(positionIds: Tensor, batch: number, seqLen: number): { cos: Tensor, sin: Tensor } {
        super.rotaryEmbedding(positionIds, batch, seqLen);
        const dimHalf = this.shape[0];
        const hd = dimHalf * 2;
        using reshaped = positionIds.reshape([batch, seqLen]);
        const cos = positionIds.workspace.alloc([batch, seqLen, hd], this.type);
        const sin = positionIds.workspace.alloc([batch, seqLen, hd], this.type);
        return { cos, sin };
    }

    ropeTranspose(cos: Tensor, sin: Tensor, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, inStride?: number, interleaved?: boolean): Tensor {
        super.ropeTranspose(cos, sin, ropeDim, headDim, nHeads, seqLen, batch, inStride, interleaved);
        using reshaped = this.reshape([batch, seqLen, ...this.shape.slice(1)]);
        return this.workspace.alloc([batch * seqLen, nHeads, headDim], this.type);
    }

    applyRotaryPosEmb(cos: Tensor, sin: Tensor, ropeDim: number, nHeads: number, seqLen: number, batch: number, unsqueezeDim: number, interleaved?: boolean): Tensor {
        super.applyRotaryPosEmb(cos, sin, ropeDim, nHeads, seqLen, batch, unsqueezeDim, interleaved);
        if (this.shape.length === 2) {
            using reshaped = this.reshape([batch, seqLen, ...this.shape.slice(1)]);
        }
        return this.workspace.alloc(this.shape, this.type);
    }

    mlaVExpand(vProj: Tensor, seqLen: number, batch: number, _lse?: Tensor, _headOffset?: number, _attnNHeads?: number, _vProjHeadOffset?: number, _tokenMajor?: boolean): Tensor {
        super.mlaVExpand(vProj, seqLen, batch);
        const nHeads = this.shape[1];
        const vHeadDim = vProj.shape[1];
        const kvLoraRank = this.shape[this.shape.length - 1];
        const BS = batch * seqLen;
        if (this.shape.length !== 4) {
            using reshaped = this.reshape([batch, seqLen, ...this.shape.slice(1)]);
        }
        return this.workspace.alloc([BS, nHeads * vHeadDim], this.type);
    }

    sigmoid(): Tensor {
        const n = this.shape.reduce((a, b) => a * b, 1);
        return this.workspace.alloc(this.shape, this.type);
    }

    topk(k: number, dim: number, offset?: number): { values: Tensor, indices: Tensor } {
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

    slice(dim: number, start: number, length: number): Tensor {
        super.slice(dim, start, length);
        const outShape = [...this.shape];
        outShape[dim] = length;
        return this.workspace.alloc(outShape, this.type);
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
    }

    maskedFill(mask: Tensor, value: number, n: number): void {
    }

    applyRotaryPosEmbPartial(cos: Tensor, sin: Tensor, ropeDim: number, headDim: number, nHeads: number, seqLen: number, batch: number, unsqueezeDim: number, interleaved?: boolean): Tensor {
        super.applyRotaryPosEmbPartial(cos, sin, ropeDim, headDim, nHeads, seqLen, batch, unsqueezeDim, interleaved);
        using reshaped = this.reshape([batch, seqLen, ...this.shape.slice(1)]);
        return this.workspace.alloc(this.shape, this.type);
    }

    reduceSum(): Tensor {
        return this.workspace.alloc([this.shape[0]], this.type);
    }

    rowNormalize(scale: number, normalize: boolean = true): Tensor {
        return this.workspace.alloc(this.shape, this.type);
    }

    groupMaskMul(groupMask: Tensor, expertsPerGroup: number, nGroup: number): void {
    }

    mulMatId(weights: Tensor[], expertIds: Tensor, topK: number, count: number, N: number, K: number, name: string): Tensor {
        return this.workspace.alloc([count, N], this.type);
    }

    scatterAddRows(scales: Tensor, topK: number, numRows: number): Tensor {
        return this.workspace.alloc([numRows, this.shape[1]], this.type);
    }

}

export class MetaOps implements DeviceOps {
    readonly worldSize = 1;
    synchronizeListeners: WeakRef<WorkspaceBase>[] = [];
    totalAllocs = 0;
    totalBytes = 0;

    [Symbol.dispose](): void {
    }

    newTensor(workspace: WorkspaceBase, shape: number[], type: string, pinned: boolean, name?: string, parallelism?: TensorParallelism, recycleKey: HeapKey | null = null): Tensor {
        const size = Tensor.byteCount(shape, type);
        this.totalAllocs++;
        this.totalBytes += size;
        return new MetaTensor(workspace, 0, size, shape, type, name, pinned, undefined, recycleKey);
    }

    wrapTensor(workspace: WorkspaceBase, data: number, allocSize: number, shape: number[], type: string, pinned: boolean, view: Tensor | undefined, recycleKey: HeapKey | null = null): Tensor {
        return new MetaTensor(workspace, data, allocSize, shape, type, undefined, pinned, view, recycleKey);
    }

    workspaceMemoryStats(workspace: WorkspaceBase): WorkspaceMemoryStats[] {
        const heaps = [...workspace.heapByKey.values()];
        return [{
            regions: heaps.reduce((sum, heap) => sum + heap.regionCount, 0),
            freeBytes: heaps.reduce((sum, heap) => sum + heap.freeBytes, 0),
        }];
    }

    reclaimWorkspaceMemory(workspace: WorkspaceBase): void {
        workspace.heapByKey.clear();
    }

    deviceHeapStats(): WorkspaceMemoryStats[] {
        return [{ regions: 0, freeBytes: 0 }];
    }

    sampleBatch(outTokens: Tensor, topkVals: Tensor, topkIdxs: Tensor, workspace: Tensor, logits: Tensor, penaltyTokens: Tensor, penaltyCount: Tensor, maxWindow: number, vocabSize: number, batchSize: number, temperatures: Tensor, repPenalties: Tensor, presPenalties: Tensor, topKs: Tensor, topPs: Tensor, stepCounter: Tensor, maxEffectiveK: number, outProbs?: Tensor, outIds?: Tensor, supportCapacity?: number): void {
    }

    sampleCandidates(outTokens: Tensor, outProbs: Tensor, outIds: Tensor, candidateValues: Tensor, candidateIds: Tensor, temperatures: Tensor, topKs: Tensor, topPs: Tensor, stepCounter: Tensor, batchSize: number, candidateCount: number, supportCapacity: number): void {
    }

    specRejectLinear(outTokens: Tensor, outAccepted: Tensor, draftTokens: Tensor, qProbs: Tensor, qIds: Tensor, pProbs: Tensor, pIds: Tensor, stepCounter: Tensor, batchSize: number, depth: number, capacity: number): void {
    }

    synchronize(): void {
        notifySynchronizedWorkspaces(this.synchronizeListeners);
    }

    async synchronizeAsync(): Promise<void> {
        notifySynchronizedWorkspaces(this.synchronizeListeners);
    }

    synchronizeStream(streamIdx: number): void {
    }

    synchronizeStreamAsync(streamIdx: number): Promise<void> {
        return Promise.resolve();
    }

    setStream(streamIdx: number): void {
    }

    eventRecord(eventIdx: number, streamIdx: number): void {
    }

    streamWaitEvent(streamIdx: number, eventIdx: number): void {
    }

    activeStreams = [0];
    get currentStream() { return this.activeStreams[this.activeStreams.length - 1]; }
    availableStreams: number[] = [];

    withStream<T>(fn: () => T): Disposable & { result: T; streamWaitEvent(): void; synchronize(): void; } {
        const result = fn();
        return {
            [Symbol.dispose]() { },
            result,
            streamWaitEvent() { },
            synchronize() { }
        };
    }

    quantizeFp8(input: Tensor, blockSize: number): { values: Tensor, scales: Tensor } {
        const scaleShape = fp8ScaleShape(input, blockSize);
        return {
            values: input.workspace.alloc(input.shape, 'F8_E4M3'),
            scales: input.workspace.alloc(scaleShape, 'F32'),
        };
    }

    projectMlaQuery(state: ExecutionState, _kvCache: Tensor, qNormed: Tensor, qPeWeight: Tensor, absorbedWeight: Tensor, cos: Tensor, sin: Tensor, qkRopeDim: number, kvLoraRank: number, nHeads: number, seqLen: number, batch: number, ropeInterleave: boolean): { qAbsorbed: Tensor, qPe: Tensor } {
        using qPeStream = this.withStream(() => {
            using qPeLin = qNormed.linear(qPeWeight);
            return qPeLin.ropeTranspose(cos, sin, qkRopeDim, qkRopeDim, nHeads, seqLen, batch, qkRopeDim, ropeInterleave);
        });
        using qAbsorbedLin = qNormed.linear(absorbedWeight);
        const qAbsorbed = state.isDecode
            ? qAbsorbedLin.ropeTranspose(undefined!, undefined!, 0, kvLoraRank, nHeads, seqLen, batch, kvLoraRank)
            : qAbsorbedLin.ropeTranspose(cos, sin, 0, kvLoraRank, nHeads, seqLen, batch, kvLoraRank);
        qPeStream.streamWaitEvent();
        return { qAbsorbed, qPe: qPeStream.result };
    }

    kvCacheWrite(srcK: Tensor, srcV: Tensor, dstK: Tensor, dstV: Tensor, slotMapping: Tensor, batchSize: number, nKv: number, hd: number, srcKTokenStride: number, srcKHeadStride: number, srcVTokenStride: number, srcVHeadStride: number): void {
    }

    batchDecodePlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, indptrH: Tensor, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, enableCudaGraph: boolean): void {
    }

    batchDecodeRun(state: ExecutionState, q: Tensor, o: Tensor, kData: Tensor, vData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, numQoHeads: number, numKvHeads: number, headDim: number, smScale: number): void {
    }

    batchPrefillPagedPlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, qoIndptrH: Tensor, pagedKvIndptrH: Tensor, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, pageSize: number, maskMode: MaskMode): void {
    }

    batchPrefillPagedRun(state: ExecutionState, q: Tensor, o: Tensor, kData: Tensor, vData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, floatWs: Tensor, intWs: Tensor, qIndptrD: Tensor, planInfo: Tensor, numQoHeads: number, numKvHeads: number, headDim: number, qStrideN: number, qStrideH: number, maskMode: MaskMode, smScale: number): void {
    }

    batchPrefillRaggedPlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, qoIndptrH: Tensor, kvIndptrH: Tensor, totalQoRows: number, batchSize: number, numQoHeads: number, numKvHeads: number, headDim: number, maskMode: MaskMode): void {
    }

    batchPrefillRaggedRun(state: ExecutionState, q: Tensor, k: Tensor, v: Tensor, o: Tensor, floatWs: Tensor, intWs: Tensor, qIndptrD: Tensor, kvIndptrD: Tensor, planInfo: Tensor, numQoHeads: number, numKvHeads: number, headDim: number, qStrideN: number, qStrideH: number, kvStrideN: number, kvStrideH: number, vStrideN: number, vStrideH: number, maskMode: MaskMode, smScale: number): void {
    }

    mlaPrefillPlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, qoIndptrH: Tensor, kvIndptrH: Tensor, kvLenH: Tensor, lastPageLenH: Tensor, batchSize: number, numHeads: number, headDimO: number, causal: boolean, pageSize: number, seqKvLens: number[], contextParallel?: boolean, cpWorldSize?: number, cpRank?: number): void {
    }

    mlaPrefillRun(state: ExecutionState, qNope: Tensor, qPe: Tensor, ckvData: Tensor, kpeData: Tensor, kvIndices: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, smScale: number, maskMode: MaskMode, cpWorldSize?: number, cpRank?: number, customMask?: Tensor, maskIndptr?: Tensor, maskKvLen?: Tensor): { o: Tensor, lse: Tensor } {
        return undefined as never;
    }

    mlaDecodePlan(floatWs: Tensor, floatWsSize: number, intWs: Tensor, pinnedIntWs: Tensor, intWsSize: number, planInfo: Tensor, indptrH: Tensor, lastPageLenH: Tensor, batchSize: number, numQoHeads: number, pageSize: number, enableCudaGraph: boolean, headDimCkv: number, headDimKpe: number, seqKvLens: number[], contextParallel?: boolean, cpWorldSize?: number, cpRank?: number): void {
    }

    sparseMlaDecodePlan(lastPageLenH: Tensor, batchSize: number, seqKvLens: number[], pageSize: number, contextParallel: boolean): void {
    }

    mlaDecodeRun(state: ExecutionState, qNope: Tensor, qPe: Tensor, ckvData: Tensor, kpeData: Tensor, indices: Tensor, indptrD: Tensor, lastPageLen: Tensor, floatWs: Tensor, intWs: Tensor, planInfo: Tensor, smScale: number): { o: Tensor, lse: Tensor } {
        return undefined as never;
    }

    mlaKvCacheAppend(_state: ExecutionState, _cacheIdx: number, ckvData: Tensor, kpeData: Tensor | null, indices: Tensor, indptr: Tensor, lastPageLen: Tensor, appendCkv: Tensor, appendKpe: Tensor | null, batchIndices: Tensor, positions: Tensor, nnz: number, headDimCkv: number, headDimKpe: number, appendCkvStrideN: number, appendKpeStrideN: number, pageSize?: number, cpWorldSize?: number, cpRank?: number): { ckv: Tensor; kpe?: Tensor } {
        return { ckv: ckvData.viewClone(), kpe: kpeData?.viewClone() };
    }

    concatAndCacheDsMla(state: ExecutionState, cacheIdx: number, kvCache: Tensor, appendCkv: Tensor, appendKpe: Tensor, indices: Tensor | undefined, indptr: Tensor, batchIndices: Tensor, positions: Tensor, nnz: number, kvLoraRank: number, peDim: number, appendCkvStrideN: number, appendKpeStrideN: number): Tensor {
        return kvCache.viewClone();
    }

    appendSelectedMtpCaches(mlaSrcCkvPtrs: Tensor, mlaSrcKpePtrs: Tensor, mlaDstCkvPtrs: Tensor, mlaDstKpePtrs: Tensor | undefined,
        indexerSrcPtrs: Tensor | undefined, indexerDstPtrs: Tensor | undefined, indexerDstScalePtrs: Tensor | undefined,
        sourceRows: Tensor, indices: Tensor, indptr: Tensor, batchIndices: Tensor, positions: Tensor,
        pageSize: number, kvLoraRank: number, peDim: number, indexHeadDim: number, sparseMode: boolean,
        cpWorldSize?: number, cpRank?: number): void {
    }

    sparseMlaPrepareCache(state: ExecutionState, groupSlots: Tensor, cacheIdx: number, kvCache: Tensor, appendCkv: Tensor, appendKpe: Tensor, topk: Tensor | undefined, indices: Tensor | null, indptr: Tensor, batchIndices: Tensor, positions: Tensor, nnz: number, kvLoraRank: number, peDim: number, appendCkvStrideN: number, appendKpeStrideN: number): Tensor {
        return kvCache.viewClone();
    }

    gdnRecurrentStep(state: ExecutionState, output: Tensor, recurrentState: Tensor, qkv: Tensor, aRaw: Tensor, bRaw: Tensor, aLog: Tensor, dtBias: Tensor, numHeads: number, dK: number, dV: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void {
    }

    gdnPrefill(state: ExecutionState, output: Tensor, recurrentState: Tensor, qkv: Tensor, aRaw: Tensor, bRaw: Tensor, aLog: Tensor, dtBias: Tensor, cuSeqlens: Tensor, numHeads: number, dK: number, dV: number, stateStride: number, qkvChStride: number, qkvSeqStride: number): void {
    }

    sparseMlaPrefill(state: ExecutionState, qAbsorbed: Tensor, qPe: Tensor, kvCache: Tensor, indices: Tensor, topk: number, smScale: number, topkLength: Tensor, pageIndptrD: Tensor, lastPageLen: Tensor, kvTokenIndptrD: Tensor): { o: Tensor, lse: Tensor } {
        return undefined as never;
    }

    sparseMlaDecode(state: ExecutionState, qAbsorbed: Tensor, qPe: Tensor, kvCache: Tensor, indices: Tensor, topk: number, numSplits: number, smScale: number, chunksPerBlock: number, topkLength?: Tensor): { o: Tensor, lse: Tensor } {
        return undefined as never;
    }

    gatherPages(srcData: Tensor, pageIndices: Tensor, pageIndptrD: Tensor, lastPageLen: Tensor, batchSize: number, paddedKvLen: number, kvTokenIndptrD: Tensor, contextParallel: boolean): Tensor {
        return undefined as never;
    }

    gatherTopkCkv(state: ExecutionState, kvCache: Tensor, outputs: readonly Tensor[], topkIdx: Tensor, pageIndices: Tensor, pageIndptr: Tensor, kvTokenIndptr: Tensor, batchIndices: Tensor, topk: number, paddedKvLen: number, cpWorldSize?: number, cpRank?: number, effPageSize?: number): void {
    }

    indexerScore(out: Tensor, q: Tensor, kData: Tensor, kScaleData: Tensor, weights: Tensor, pageIndices: Tensor, pageIndptr: Tensor, lastPageLen: Tensor, qoIndptr: Tensor, scale: number, totalQ: number, idxNHeads: number, idxHeadDim: number, pageSize: number, maxKvLen: number, causal: boolean, kvTokenIndptr?: Tensor): void {
    }

    indexerTopk(state: ExecutionState, idxQ: Tensor, kData: Tensor, kScaleData: Tensor, weights: Tensor, pageIndices: Tensor, indptr: Tensor, lastPageLen: Tensor, qoIndptr: Tensor, scale: number, topk: number, decode: boolean, qGlobalStart?: number, customMask?: Tensor, maskIndptr?: Tensor, maskKvLen?: Tensor, cpWorldSize?: number, cpRank?: number, globalLastPageLen?: Tensor, kvTokenIndptr?: Tensor): { values: Tensor, indices: Tensor } {
        const indices = idxQ.workspace.alloc([idxQ.shape[0], topk], "I32");
        const values = idxQ.workspace.alloc([idxQ.shape[0], topk], "BF16");
        return { values, indices };
    }

    sortTopkByIndex(indices: Tensor, values: Tensor, batch: number, topk: number): void {
    }

    topkToSlots(state: ExecutionState, topkIdx: Tensor, kvTokenIndptrD: Tensor, pageIndices: Tensor, indptr: Tensor, lastPageLen: Tensor, batchIndices: Tensor, pageSize: number, maxKv: number, cacheIdx: number, contextParallel?: boolean, cpWorldSize?: number, cpRank?: number, providedLength?: Tensor): { layer: SlotSet, group: SlotSet } {
        const slots = topkIdx.workspace.alloc([topkIdx.shape[0], topkIdx.shape[1]], "I32");
        const length = providedLength ?? topkIdx.workspace.alloc([topkIdx.shape[0]], "I32");
        return {
            layer: { slots, length },
            group: { slots: slots.viewClone(), length: length.viewClone() },
        };
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
