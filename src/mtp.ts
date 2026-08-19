import { CaptureManager } from "./capture-manager";
import { type ChatCache, type ChatModel } from "./chat_model";
import { MaskMode } from "./device_ops";
import { MemcpyKind } from "./enums";
import { ExecutionState, ExecutionWorkspace } from "./execution-workspace";
import { BF16, I32 } from "./glm_ops";
import { type Tensor } from "./tensor";
import { UsingHolder } from "./using-holder";

/**
 * Accumulator for MTP/speculative-decoding acceptance metrics, mirroring
 * vllm's SpecDecodingStats + SpecDecodingLogging.
 *
 *   draft_acceptance_rate = num_accepted_tokens / num_draft_tokens
 *   mean_acceptance_length = 1 + num_accepted_tokens / num_drafts  (includes bonus token)
 *   per_position_rate[i] = num_accepted_per_pos[i] / num_drafts
 */
export class MtpStats {
  numDrafts = 0;
  numDraftTokens = 0;
  numAcceptedTokens = 0;
  numAcceptedPerPos: number[];
  readonly numSpecTokens: number;

  constructor(numSpecTokens: number) {
    this.numSpecTokens = numSpecTokens;
    this.numAcceptedPerPos = new Array(numSpecTokens).fill(0);
  }

  observe(numDraftTokens: number, numAccepted: number): void {
    this.numDrafts++;
    this.numDraftTokens += numDraftTokens;
    this.numAcceptedTokens += numAccepted;
    for (let i = 0; i < numAccepted && i < this.numAcceptedPerPos.length; i++) {
      this.numAcceptedPerPos[i]++;
    }
  }

  get acceptanceRate(): number {
    return this.numDraftTokens > 0 ? this.numAcceptedTokens / this.numDraftTokens : NaN;
  }

  get meanAcceptanceLength(): number {
    return this.numDrafts > 0 ? 1 + this.numAcceptedTokens / this.numDrafts : NaN;
  }

  perPositionRates(): number[] {
    return this.numAcceptedPerPos.map(a => this.numDrafts > 0 ? a / this.numDrafts : NaN);
  }

  log(): string {
    if (this.numDrafts === 0) return "";
    const rates = this.perPositionRates().map(r => r.toFixed(3)).join(", ");
    return `MTP metrics: mean acceptance length=${this.meanAcceptanceLength.toFixed(2)}, ` +
      `acceptance rate=${(this.acceptanceRate * 100).toFixed(1)}%, ` +
      `accepted=${this.numAcceptedTokens}/${this.numDraftTokens} tokens, ` +
      `drafts=${this.numDrafts}, per-pos=[${rates}]`;
  }
}

function totalPaths(topk: number[]): number {
  return topk.reduce((acc, k) => acc * k, 1);
}

function totalTreeNodes(topk: number[]): number {
  let cumulative = 0;
  let nodesAtDepth = 1;
  for (const k of topk) {
    nodesAtDepth *= k;
    cumulative += nodesAtDepth;
  }
  return cumulative;
}

function depthBoundaries(topk: number[]): number[] {
  const boundaries: number[] = [];
  let cumulative = 0;
  let nodesAtDepth = 1;
  for (const k of topk) {
    nodesAtDepth *= k;
    cumulative += nodesAtDepth;
    boundaries.push(cumulative);
  }
  return boundaries;
}

function treeDepth(topk: number[], nodeIndex: number): number {
  const boundaries = depthBoundaries(topk);
  let depth = 0;
  while (depth < boundaries.length && nodeIndex >= boundaries[depth]) {
    depth++;
  }
  return depth;
}

function childIndex(topk: number[], nodeIndex: number, childDigit: number, boundaries?: number[]): number {
  const b = boundaries ?? depthBoundaries(topk);
  const d = treeDepth(topk, nodeIndex);
  const offsetWithinDepth = nodeIndex - (d > 0 ? b[d - 1] : 0);
  return b[d] + offsetWithinDepth * topk[d + 1] + childDigit;
}

function pathDigit(topks: number[], path: number, layer: number, strides?: number[]): number {
  if (!strides) {
    strides = topks.map((_, i) => totalPaths(topks.slice(i + 1)));
  }
  return Math.floor(path / strides[layer]) % topks[layer];
}

function parentIndex(topk: number[], nodeIndex: number, boundaries?: number[]): number {
  const b = boundaries ?? depthBoundaries(topk);
  const d = treeDepth(topk, nodeIndex);
  if (d === 0) return -1;
  const offsetWithinDepth = nodeIndex - b[d - 1];
  const parentOffset = Math.floor(offsetWithinDepth / topk[d]);
  return (d > 1 ? b[d - 2] : 0) + parentOffset;
}

function getPositionIdsMask(ws: ExecutionWorkspace, originalAllocLens: number[], topk: number[]) {
  const batchSize = originalAllocLens.length;
  const numNodes = totalTreeNodes(topk);
  const maxShape = [ws.maxBatch * ws.maxSeqLen];
  const positionIds = ws.ensureAlloc(maxShape, "I32", `mtp_pos-${numNodes}`);
  const positionIdsH = ws.ensureAllocPinned(maxShape, "I32", `mtp_pos_host-${numNodes}`);
  const boundaries = depthBoundaries(topk);
  positionIdsH.withPinnedBuffer(buf => {
    let posOff = 0;
    for (const originalAllocLen of originalAllocLens) {
      for (let p = 0; p < numNodes; p++) {
        let depth = 0;
        while (depth < boundaries.length && p >= boundaries[depth]) {
          depth++;
        }
        buf.writeInt32LE(originalAllocLen + depth, posOff * I32);
        posOff++;
      }
    }
  });
  positionIds.memcpy(positionIdsH, batchSize * numNodes * I32, MemcpyKind.HostToDevice);
  return positionIds;
}

/**
 * Tree-structured batched MTP decode using prefill with custom masks.
 *
 * Iteration 0 uses the target model's decode outputs to produce each root's
 * candidates. Iterations 1+ append tree tokens to every sequence and run batched
 * prefill passes with tree-shaped custom masks. All sequences are then restored
 * to their individual original lengths before verification.
 *
 * The custom mask ensures each draft token only attends to the prefix plus its
 * ancestors in the binary tree, so different branches don't cross-attend.
 *
 * Prerequisites:
 *   - The target model must have just decoded (planDecode + positionStep + forward
 *     already called for the current position)
 *   - model.forwardMtp must exist (MTP enabled)
 *
 * @param state - The target model's decode ExecutionState
 * @param captureManager - Capture manager for CUDA graph replay
 * @param model - The chat model (must support forwardMtp)
 * @param mtpHiddenStates - MTP hidden-state scratch large enough for the widest intermediate tree depth
 * @param ws - Execution workspace
 * @param topks - Array of top-k values per MTP depth (e.g. [2,2,2] for a binary tree with 3 layers)
 * @param cache - Chat cache (paged KV cache)
 * @returns Per-sequence accepted draft tokens plus each replacement token
 */
export async function mtpTreeDecode(
  captureManager: CaptureManager,
  model: ChatModel,
  mtpHiddenStates: Tensor,
  sharedSlots: Tensor,
  sharedSlotsLength: Tensor,
  ws: ExecutionWorkspace,
  targetTokens: number[],
  topks: number[],
  cache: ChatCache,
) {
  if (!model.forwardMtp) {
    throw new Error("mtpTreeDecode: model does not support MTP (forwardMtp not defined)");
  }

  const tokenizer = model.tokenizer;
  const lmHead = model.tensors.get("lm_head.weight")!;

  const draftTopk = topks;
  const targetTopk = [1, ...topks];
  const numPaths = totalPaths(draftTopk);
  const numTreeNodes = totalTreeNodes(draftTopk);
  const pagedKv = cache.getPagedKV();
  const batchSize = pagedKv.sequences.length;
  if (targetTokens.length !== batchSize) {
    throw new Error(`mtpTreeDecode: received ${targetTokens.length} target tokens for ${batchSize} sequences`);
  }
  const originalAllocLens = pagedKv.sequences.map(sequence => sequence.allocLen);

  const hiddenDim = mtpHiddenStates.shape[1];
  const rowBytes = hiddenDim * BF16; // BF16 = 2 bytes per element
  const maxIntermediateWidth = Math.max(1, ...topks.slice(0, -1).map((_, depth) => totalPaths(topks.slice(0, depth + 1))));
  const scratchRows = mtpHiddenStates.numElements / hiddenDim;
  const requiredScratchRows = batchSize * maxIntermediateWidth;
  if (scratchRows < requiredScratchRows) {
    throw new Error(`mtpTreeDecode: mtpHiddenStates has ${scratchRows} rows, need ${requiredScratchRows} for batch size ${batchSize}`);
  }

  const hostBuf = ws.ensureAllocPinned([ws.maxBatch * numTreeNodes], "I32", `mtp_verify_host_buf_${numTreeNodes}`);


  const start = performance.now();

  let warmup = false;
  const useDecodeDraftGenerator = false;

  // Depth-1 tree (topks.length === 1, e.g. nextn=1): both draft strategies below
  // run zero loop iterations, so the root's top-k candidates must be generated
  // explicitly here. No MTP forward or sequence duplication is needed — the
  // candidates come straight from the seed hidden state. Applies to both the
  // batched-decode and chunked-prefill branches.
  if (topks.length === 1) {
    using _tracker = ws.startTracking(new Set([mtpHiddenStates, sharedSlots, sharedSlotsLength]));
    warmup ||= !captureManager.isCaptured(['mtp-tree-decode-root', topks[0], `batchSize:${batchSize}`]);
    captureManager.run({ sharedSlots, sharedSlotsLength, mtpHiddenStates }, () => {
      // mtpHiddenStates is already shared_head.norm'd by forwardMtp; use directly
      using seedHiddenStates = mtpHiddenStates.narrow(0, batchSize);
      using initialLogits = seedHiddenStates.linear(lmHead);
      const initialTopk = initialLogits.topk(topks[0], model.cfg.vocabSize);
      using _initialValues = initialTopk.values;
      using initialIndices = initialTopk.indices;
      hostBuf.memcpy2d(0, batchSize * I32 * topks[0], initialIndices, 0, batchSize * I32 * topks[0], batchSize * I32 * topks[0], 1, MemcpyKind.DeviceToHost);
    }, ['mtp-tree-decode-root', topks[0], `batchSize:${batchSize}`]);
    mtpHiddenStates.removeTracking();
    sharedSlots.removeTracking();
    sharedSlotsLength.removeTracking();
    await ws.glm.synchronizeAsync();
  }
  else if (useDecodeDraftGenerator) {
    // current path that decodes in batch
    let hostBufOffset = 0;

    for (let i = 1; i < topks.length; i++) {
      using _tracker = ws.startTracking(new Set([mtpHiddenStates, sharedSlots, sharedSlotsLength]));

      // every iteration, duplicate all the sequences to add the top-k for this depth
      const currentBatchSize = pagedKv.sequences.length;
      const k = topks[i - 1];
      const newBatchSize = currentBatchSize * k;

      // Interleaved duplication: produce [orig0, copy0_of_orig0, ..., orig1, copy0_of_orig1, ...]
      // so that parent-grouped (BFS) topk indices can be used directly as inputs without
      // transposition. This keeps the hostBuf layout in BFS order at all depths, matching
      // the BFS tree indexing used during verification (childIndex/parentIndex over targetTopk).
      // Phase 1: save originals to the end of the array
      for (let seqIdx = 0; seqIdx < currentBatchSize; seqIdx++) {
        pagedKv.copySequence(newBatchSize - currentBatchSize + seqIdx, seqIdx);
      }
      // Phase 2: create interleaved copies from the saved originals
      for (let seqIdx = 0; seqIdx < currentBatchSize; seqIdx++) {
        for (let j = 0; j < k; j++) {
          pagedKv.copySequence(seqIdx * k + j, newBatchSize - currentBatchSize + seqIdx);
        }
      }

      const state = ws.planDecode(model, newBatchSize, cache);

      warmup ||= !state.isCaptured(captureManager, ['mtp-tree-decode', i, topks.length]);

      state.capture(captureManager, { sharedSlots, sharedSlotsLength, mtpHiddenStates }, (_capturing, inputs) => {
        state.sharedSlots = new UsingHolder(inputs.sharedSlots?.capture());
        state.sharedSlotsLength = new UsingHolder(inputs.sharedSlotsLength?.capture());
        using chainedMtpHiddenState = inputs.mtpHiddenStates.narrow(0, currentBatchSize);

        // prepare initial input
        if (i === 1) {
          // mtpHiddenStates is already shared_head.norm'd by forwardMtp; use directly
          using seedHiddenStates = mtpHiddenStates.narrow(0, batchSize);
          using initialLogits = seedHiddenStates.linear(lmHead);
          const initialTopk = initialLogits.topk(topks[0], model.cfg.vocabSize);
          using initialIndices = initialTopk.indices;
          using _initialValues = initialTopk.values;
          state.inputIdsBuf.memcpy(initialIndices, initialIndices.bytes, MemcpyKind.DeviceToDevice);
          // copy the initial indices to host buffer
          hostBuf.memcpy2d(0, currentBatchSize * I32 * topks[0], initialIndices, 0, currentBatchSize * I32 * topks[0], currentBatchSize * I32 * topks[0], 1, MemcpyKind.DeviceToHost);
        }

        state.setInput(state.inputIdsBuf);

        ws.positionStep(state, model);

        // Expand hidden states: repeat each row k times consecutively [hs0, hs0, ..., hs1, hs1, ...]
        // to match interleaved sequence order [orig0, copy0, ..., orig1, copy1, ...]. Each child
        // shares its parent's hidden state (the EAGLE/MTP recurrence uses h_{d-1} as input).
        using _expanded = k > 1 ? ws.alloc([newBatchSize, hiddenDim], "BF16") : undefined;
        const expandedHiddenState = _expanded ?? chainedMtpHiddenState;
        if (_expanded) {
          for (let seqIdx = 0; seqIdx < currentBatchSize; seqIdx++) {
            for (let j = 0; j < k; j++) {
              _expanded.memcpy2d(
                (seqIdx * k + j) * rowBytes, rowBytes,
                chainedMtpHiddenState, seqIdx * rowBytes, rowBytes,
                rowBytes, 1,
                MemcpyKind.DeviceToDevice,
              );
            }
          }
        }
        using newMtpHiddenStates = model.forwardMtp!(state, expandedHiddenState);
        // newMtpHiddenStates is already shared_head.norm'd; apply lmHead directly
        using logits = newMtpHiddenStates.linear(lmHead);
        const logitsTopk = logits.topk(topks[i], model.cfg.vocabSize);
        using _values = logitsTopk.values;
        using indices = logitsTopk.indices;

        // append the new indices to host buffer
        hostBufOffset += currentBatchSize * I32 * topks[i - 1];
        hostBuf.memcpy2d(hostBufOffset, newBatchSize * I32 * topks[i], indices, 0, newBatchSize * I32 * topks[i], newBatchSize * I32 * topks[i], 1, MemcpyKind.DeviceToHost);

        // prepare next input — indices are already in parent-grouped (BFS) order,
        // matching the interleaved sequence layout [orig0_child0, orig0_child1, orig1_child0, ...]
        if (i !== topks.length - 1) {
          state.inputIdsBuf.memcpy(indices, indices.bytes, MemcpyKind.DeviceToDevice);
        }

        chainedMtpHiddenState.memcpy(newMtpHiddenStates, newMtpHiddenStates.bytes, MemcpyKind.DeviceToDevice);
      }, ['mtp-tree-decode', i, topks.length]);

      await ws.glm.synchronizeAsync();

      mtpHiddenStates.removeTracking();
      sharedSlots?.removeTracking();
      sharedSlotsLength?.removeTracking();
    }

    // clean up the tree of sequences, reverse order so pages are returned in order.
    while (pagedKv.sequences.length > batchSize) {
      pagedKv.removeSequence(pagedKv.sequences.length - 1);
    }
  }
  else {
    // Chunked prefill with extended causal custom mask.
    // Each depth is a separate planPrefill + forwardMtp. The extended mask
    // allows each token to attend to its parent from the previous depth
    // (now in KV cache). KV cache is retained between depths; only truncated
    // after all depths for verification.
    const draftBoundaries = depthBoundaries(draftTopk);

    using _tracker = ws.startTracking(new Set([mtpHiddenStates, sharedSlots, sharedSlotsLength]));

    // Plan loop — all plans execute back-to-back. Each gets a unique slot,
    // so plan N+1's host writes don't race with plan N's in-flight H2D copies.
    const planStates: ExecutionState[] = [];
    const planMeta: { qoLen: number, hsPrevQoLen: number, expandK: number, depth: number }[] = [];
    for (let depth = 1; depth < topks.length; depth++) {
      const qoLen = totalPaths(topks.slice(0, depth));
      const hsPrevQoLen = depth > 1 ? totalPaths(topks.slice(0, depth - 1)) : 1;
      const expandK = topks[depth - 1];

      const chunkedMask = ensureChunkedMTPMask(ws, topks, depth);
      const posIds = getPositionIdsChunked(ws, originalAllocLens, depth, qoLen);
      const state = ws.planPrefill(model, batchSize, new Array(batchSize).fill(qoLen), cache, {
        mask: chunkedMask.mask,
        indptr: chunkedMask.indptr,
        mode: MaskMode.CausalCustom,
        positionIds: posIds,
        maskKvLen: chunkedMask.maskKvLen,
      });

      planStates.push(state);
      planMeta.push({ qoLen, hsPrevQoLen, expandK, depth });
    }

    // Run loop — all depths captured as a single CUDA graph. Each depth's
    // forward pass runs sequentially (each needs prev's output), but the
    // entire multi-depth MTP draft is a single graph launch on replay.
    const captureKey = ['mtp-chunk-all', topks.join(',')];
    warmup ||= !ExecutionState.isCaptured(captureManager, planStates, captureKey);
    ExecutionState.captureAll(captureManager, planStates, { mtpHiddenStates, sharedSlots, sharedSlotsLength }, (_capturing, inputs) => {
      for (let pi = 0; pi < planStates.length; pi++) {
        const { qoLen, hsPrevQoLen, expandK, depth } = planMeta[pi];
        const state = planStates[pi];
        const next = pi + 1 < planStates.length ? planStates[pi + 1] : null;

        state.sharedSlots = new UsingHolder(inputs.sharedSlots?.capture());
        state.sharedSlotsLength = new UsingHolder(inputs.sharedSlotsLength?.capture());
        const prevHs = inputs.mtpHiddenStates;

        // Depth 1: compute initial logits and topk from mtpHiddenStates
        if (depth === 1) {
          using seedHiddenStates = inputs.mtpHiddenStates.narrow(0, batchSize);
          using initialLogits = seedHiddenStates.linear(lmHead);
          const initialTopk = initialLogits.topk(topks[0], model.cfg.vocabSize);
          using _initialValues = initialTopk.values;
          using initialIndices = initialTopk.indices;
          state.inputIdsBuf.memcpy(initialIndices, initialIndices.bytes, MemcpyKind.DeviceToDevice);
          for (let batch = 0; batch < batchSize; batch++) {
            hostBuf.memcpy2d(
              batch * numTreeNodes * I32, topks[0] * I32,
              initialIndices, batch * topks[0] * I32, topks[0] * I32,
              topks[0] * I32, 1,
              MemcpyKind.DeviceToHost,
            );
          }
        }

        state.setInput(state.inputIdsBuf);

        // Expand prevHs: replicate each parent's hidden state for expandK children
        const prevRows = batchSize * hsPrevQoLen;
        const totalQoLen = batchSize * qoLen;
        using narrowedPrevHs = prevHs.narrow(0, prevRows);
        using _expandedHs = expandK > 1 ? ws.alloc([totalQoLen, hiddenDim], "BF16") : undefined;
        const expandedHs = _expandedHs ?? narrowedPrevHs;
        if (expandK > 1) {
          for (let c = 0; c < expandK; c++) {
            _expandedHs!.memcpy2d(
              c * rowBytes, expandK * rowBytes,
              narrowedPrevHs, 0, rowBytes,
              rowBytes, prevRows,
              MemcpyKind.DeviceToDevice,
            );
          }
        }

        using hiddenStates = model.forwardMtp!(state, expandedHs);

        // hiddenStates is already shared_head.norm'd; apply lmHead directly
        using logits = hiddenStates.linear(lmHead);
        const logitsTopk = logits.topk(topks[depth], model.cfg.vocabSize);
        using _values = logitsTopk.values;
        using indices = logitsTopk.indices;

        // Write next depth's tokens to hostBuf
        const depthOffset = draftBoundaries[depth - 1];
        const hostCount = qoLen * topks[depth];
        for (let batch = 0; batch < batchSize; batch++) {
          hostBuf.memcpy2d(
            (batch * numTreeNodes + depthOffset) * I32, hostCount * I32,
            indices, batch * hostCount * I32, hostCount * I32,
            hostCount * I32, 1,
            MemcpyKind.DeviceToHost,
          );
        }

        // Prepare input for next depth — copy to next state's inputIdsBuf
        if (next) {
          next.inputIdsBuf.memcpy(indices, indices.bytes, MemcpyKind.DeviceToDevice);
        }

        if (next) {
          inputs.mtpHiddenStates.memcpy(hiddenStates, hiddenStates.bytes, MemcpyKind.DeviceToDevice);
        }
      }
    }, captureKey);

    await ws.glm.synchronizeAsync();

    mtpHiddenStates.removeTracking();
    sharedSlots?.removeTracking();
    sharedSlotsLength?.removeTracking();
  }

  const draft = performance.now();

  // after iterative tree decode/prefill, rewind for tree verification prefill
  for (let batch = 0; batch < batchSize; batch++) {
    pagedKv.sequences[batch].truncate(originalAllocLens[batch]);
  }

  const verificationTokens: number[][] = [];
  for (let batch = 0; batch < batchSize; batch++) {
    let batchOffset = batch * numTreeNodes * I32;
    const batchTokens: number[] = [targetTokens[batch]];
    for (let i = 0; i < numTreeNodes; i++) {
      batchTokens.push(hostBuf.readPinnedBuffer().readInt32LE(batchOffset + i * I32));
    }
    verificationTokens.push(batchTokens);
  }

  const numVerificationTokens = numTreeNodes + 1; // include target token
  const targetCustomMask = ensureTargetCustomMask(ws, targetTopk);
  const targetPrefillState = ws.planPrefill(model, batchSize, new Array(batchSize).fill(numVerificationTokens), cache, {
    ...targetCustomMask,
    positionIds: getPositionIdsMask(ws, originalAllocLens, targetTopk),
  });

  targetPrefillState.setInput(verificationTokens);

  const hiddenStateStaging = ws.ensureAlloc([ws.maxBatch * numVerificationTokens, hiddenDim], "BF16", `mtp-tree-hs-staging-${numVerificationTokens}`, undefined, 0);
  await ws.glm.synchronizeAsync();

  const track1 = ws.startTracking(new Set([mtpHiddenStates, sharedSlots, sharedSlotsLength]));

  warmup ||= !targetPrefillState.isCaptured(captureManager, ['mtp-verify', numVerificationTokens]);
  const { kvCacheLayers, indexerKvCacheLayers } = targetPrefillState.capture(captureManager, { mtpHiddenStates, sharedSlots, sharedSlotsLength }, () => {
    targetPrefillState.sharedSlots = new UsingHolder(undefined!);
    targetPrefillState.sharedSlotsLength = new UsingHolder(undefined!);

    const kvCacheLayers: { appendCkv: Tensor, appendKpe: Tensor, appendCkvOrig: Tensor, appendKpeOrig: Tensor, cacheIdx: number, kvLoraRank: number, qkRopeDim: number }[] = [];
    const indexerKvCacheLayers: { appendIdxK: Tensor, appendIdxKOrig: Tensor, cacheIdx: number, indexHeadDim: number }[] = [];

    const mlaKVCacheAppendOrig = targetPrefillState.mlaKvCacheAppend.bind(targetPrefillState);
    targetPrefillState.mlaKvCacheAppend = (appendCkv, appendKpe, cacheIdx, kvLoraRank, qkRopeDim) => {
      // need to prevent this from being recycled into workspace
      appendCkv.removeTracking();
      appendKpe.removeTracking();
      // and capture the tensors for graph playback
      kvCacheLayers.push({ appendCkv: appendCkv.capture(), appendKpe: appendKpe.capture(), appendCkvOrig: appendCkv, appendKpeOrig: appendKpe, cacheIdx, kvLoraRank, qkRopeDim });
      return mlaKVCacheAppendOrig(appendCkv, appendKpe, cacheIdx, kvLoraRank, qkRopeDim);
    };

    const indexerKvCacheAppendOrig = targetPrefillState.indexerKvCacheAppend.bind(targetPrefillState);
    targetPrefillState.indexerKvCacheAppend = (idxKOut, cacheIdx, indexHeadDim) => {
      idxKOut.removeTracking();
      indexerKvCacheLayers.push({ appendIdxK: idxKOut.capture(), appendIdxKOrig: idxKOut, cacheIdx, indexHeadDim });
      return indexerKvCacheAppendOrig(idxKOut, cacheIdx, indexHeadDim);
    };

    using hiddenStates = model.forwardModel(targetPrefillState);
    using logits = targetPrefillState.computeLogits(hiddenStates, model, true);
    using argmaxResult = logits.argmax();
    const argmaxHost = ws.ensureAllocPinned([ws.maxBatch * numVerificationTokens], argmaxResult.type, `mtp_verify_argmax_host_${numVerificationTokens}`);
    argmaxHost.memcpy(argmaxResult, argmaxResult.bytes, MemcpyKind.DeviceToHost);

    hiddenStateStaging.memcpy(hiddenStates, undefined, MemcpyKind.DeviceToDevice);

    // need the new shared slots for the 
    const capturedSharedSlots = targetPrefillState.sharedSlots.value;
    const capturedSharedSlotsLength = targetPrefillState.sharedSlotsLength.value;
    sharedSlots?.memcpy(capturedSharedSlots, capturedSharedSlots!.bytes, MemcpyKind.DeviceToDevice);
    sharedSlotsLength?.memcpy(capturedSharedSlotsLength, capturedSharedSlotsLength.bytes, MemcpyKind.DeviceToDevice);

    return { kvCacheLayers, indexerKvCacheLayers };
  }, ['mtp-verify', numVerificationTokens]);

  ws.freeze();

  sharedSlots?.removeTracking();
  sharedSlotsLength?.removeTracking();
  mtpHiddenStates.removeTracking();

  await ws.glm.synchronizeAsync();

  const verify = performance.now();

  const argmaxHost = ws.tensors.get(`mtp_verify_argmax_host_${numVerificationTokens}`)!;
  const argmaxBuf = argmaxHost.readPinnedBuffer();

  // const tokenTree = argmaxHost.readInt32LEArray().map(t => tokenizer.decode([t], { skip_special_tokens: false }));
  // const draftTree = verificationTokens[0].map(t => tokenizer.decode([t], { skip_special_tokens: false }));
  // console.warn(`MTP target predictions: ${tokenTree.join(", ")}`);
  // console.warn(`MTP draft tokens:       ${draftTree.join(", ")}`);

  const targetBoundaries = depthBoundaries(targetTopk);
  const draftStrides = topks.map((_, i) => totalPaths(topks.slice(i + 1)));
  const bestAccepted: number[] = [];
  const acceptedTokens: number[][] = [];
  const replacementTokens: number[] = [];
  const acceptedNodeIndices: number[][] = [];

  for (let batch = 0; batch < batchSize; batch++) {
    const argmaxOffset = batch * numVerificationTokens;
    let batchBestPath = 0;
    let batchBestAccepted = -1;
    let batchBestReplacement = -1;

    for (let path = 0; path < numPaths; path++) {
      let accepted = 0;
      let nodeIdx = 0;
      for (let layer = 0; layer < topks.length; layer++) {
        const digit = pathDigit(topks, path, layer, draftStrides);
        const childIdx = childIndex(targetTopk, nodeIdx, digit, targetBoundaries);
        const draftToken = verificationTokens[batch][childIdx];
        const targetPrediction = argmaxBuf.readInt32LE((argmaxOffset + nodeIdx) * I32);
        if (draftToken === targetPrediction) {
          accepted++;
          nodeIdx = childIdx;
        } else {
          break;
        }
      }
      if (accepted > batchBestAccepted) {
        batchBestAccepted = accepted;
        batchBestPath = path;
        batchBestReplacement = argmaxBuf.readInt32LE((argmaxOffset + nodeIdx) * I32);
      }
    }

    const batchAcceptedTokens: number[] = [];
    const batchAcceptedNodeIndices = [0];
    let nodeIdx = 0;
    for (let layer = 0; layer < batchBestAccepted; layer++) {
      const digit = pathDigit(topks, batchBestPath, layer, draftStrides);
      nodeIdx = childIndex(targetTopk, nodeIdx, digit, targetBoundaries);
      batchAcceptedTokens.push(verificationTokens[batch][nodeIdx]);
      batchAcceptedNodeIndices.push(nodeIdx);
    }

    bestAccepted.push(batchBestAccepted);
    acceptedTokens.push(batchAcceptedTokens);
    replacementTokens.push(batchBestReplacement);
    acceptedNodeIndices.push(batchAcceptedNodeIndices);
  }

  const finishCounts = bestAccepted.map(accepted => accepted + 1); // target + accepted
  const totalFinishCount = finishCounts.reduce((sum, count) => sum + count, 0);

  {
    // Accepted tokens along bestPath: root (node 0) + accepted children.
    // The replacement token is handled separately via a decode step, not from the tree.
    // After capture block, in-place reorder accepted rows to front:
    // targetHiddenStates has one fixed-size tree block per sequence. Compact
    // each accepted path into the variable-length batched prefill layout.
    let dstBase = 0;
    for (let batch = 0; batch < batchSize; batch++) {
      const srcBase = batch * numVerificationTokens;
      for (let i = 0; i < finishCounts[batch]; i++) {
        const srcIdx = srcBase + acceptedNodeIndices[batch][i];
        const dstIdx = dstBase + i;
        if (srcIdx === dstIdx) continue;
        hiddenStateStaging.memcpy2d(
          dstIdx * rowBytes, rowBytes,
          hiddenStateStaging, srcIdx * rowBytes, rowBytes,
          rowBytes, 1,
          MemcpyKind.DeviceToDevice,
        );
      }
      dstBase += finishCounts[batch];
    }

    for (const layer of kvCacheLayers) {
      const kvLoraRank = layer.kvLoraRank;
      const qkRopeDim = layer.qkRopeDim;
      let dstBase = 0;
      for (let batch = 0; batch < batchSize; batch++) {
        const srcBase = batch * numVerificationTokens;
        for (let i = 0; i < finishCounts[batch]; i++) {
          const srcIdx = srcBase + acceptedNodeIndices[batch][i];
          const dstIdx = dstBase + i;
          if (srcIdx === dstIdx) continue;
          layer.appendCkv.memcpy2d(
            dstIdx * kvLoraRank * 2, kvLoraRank * 2,
            layer.appendCkv, srcIdx * kvLoraRank * 2, kvLoraRank * 2,
            kvLoraRank * 2, 1,
            MemcpyKind.DeviceToDevice,
          );
          layer.appendKpe.memcpy2d(
            dstIdx * qkRopeDim * 2, qkRopeDim * 2,
            layer.appendKpe, srcIdx * qkRopeDim * 2, qkRopeDim * 2,
            qkRopeDim * 2, 1,
            MemcpyKind.DeviceToDevice,
          );
        }
        dstBase += finishCounts[batch];
      }
    }
    for (const layer of indexerKvCacheLayers) {
      const indexHeadDim = layer.indexHeadDim;
      let dstBase = 0;
      for (let batch = 0; batch < batchSize; batch++) {
        const srcBase = batch * numVerificationTokens;
        for (let i = 0; i < finishCounts[batch]; i++) {
          const srcIdx = srcBase + acceptedNodeIndices[batch][i];
          const dstIdx = dstBase + i;
          if (srcIdx === dstIdx) continue;
          layer.appendIdxK.memcpy2d(
            dstIdx * indexHeadDim * 2, indexHeadDim * 2,
            layer.appendIdxK, srcIdx * indexHeadDim * 2, indexHeadDim * 2,
            indexHeadDim * 2, 1,
            MemcpyKind.DeviceToDevice,
          );
        }
        dstBase += finishCounts[batch];
      }
    }
  }

  // target can now be truncate to the original length, plan another prefill with
  // the accepted token length.
  // target layers: write the captured kvCacheLayers
  // mtp layers: perform extended prefill as usual
  for (let batch = 0; batch < batchSize; batch++) {
    pagedKv.sequences[batch].truncate(originalAllocLens[batch]);
  }
  await ws.glm.synchronizeAsync();

  const mtpExtendPrefill = ws.planPrefill(model, batchSize, finishCounts, cache);
  mtpExtendPrefill.setInput(acceptedTokens.map((tokens, batch) => [...tokens, replacementTokens[batch]]));
  for (const layer of kvCacheLayers) {
    mtpExtendPrefill.mlaKvCacheAppend(layer.appendCkv, layer.appendKpe, layer.cacheIdx, layer.kvLoraRank, layer.qkRopeDim);
    layer.appendCkvOrig[Symbol.dispose]();
    layer.appendKpeOrig[Symbol.dispose]();
  }
  for (const layer of indexerKvCacheLayers) {
    mtpExtendPrefill.indexerKvCacheAppend(layer.appendIdxK, layer.cacheIdx, layer.indexHeadDim);
    layer.appendIdxKOrig[Symbol.dispose]();
  }
  await ws.glm.synchronizeAsync();

  ws.unfreeze()

  track1[Symbol.dispose]();
  using track2 = ws.startTracking(new Set([mtpHiddenStates, sharedSlots, sharedSlotsLength]));

  warmup ||= !mtpExtendPrefill.isCaptured(captureManager, ['mtp-replace', finishCounts.join(',')]);
  mtpExtendPrefill.capture(captureManager, { mtpHiddenStates, sharedSlots, sharedSlotsLength }, () => {
    mtpExtendPrefill.sharedSlots = new UsingHolder(sharedSlots?.capture());
    mtpExtendPrefill.sharedSlotsLength = new UsingHolder(sharedSlotsLength?.capture());

    using verfiedHiddenStates = hiddenStateStaging.slice(0, 0, totalFinishCount);
    using mtpHs = model.forwardMtp!(mtpExtendPrefill, verfiedHiddenStates);
    // MTP convention (same as rotateInputIds in prefill): at position P, the input
    // token is P+1 paired with hidden state at P. Here input[0] = acceptedTokens[0]
    // (the token after targetToken) with verifiedHiddenStates[0] (target model HS at
    // targetToken). The MTP KV at position P thus encodes token P+1, while the target
    // model KV at the same position encodes token P — each layer has its own KV slot
    // so this is safe. The last row is the seed for the next draft iteration.
    using lastIdx = mtpExtendPrefill.lastIdx;
    using newMtpHiddenStates = mtpHs.indexSelect(lastIdx, -1);
    mtpHiddenStates.memcpy(newMtpHiddenStates);
  }, ['mtp-replace', finishCounts.join(',')]);
  mtpHiddenStates.removeTracking();
  sharedSlots?.removeTracking();
  sharedSlotsLength?.removeTracking();
  await ws.glm.synchronizeAsync();


  // if (acceptedTokens.length) {
  //  console.warn(`MTP verify: accepted ${acceptedTokens.length} tokens: ${acceptedTokens.map(t => tokenizer.decode([t], { skip_special_tokens: false }))}, replacement: ${tokenizer.decode([bestReplacement], { skip_special_tokens: false })}, target: ${tokenizer.decode([targetToken], { skip_special_tokens: false })}`);
  // }

  // timings
  // console.log(`MTP tree decode: ${draft - start}ms, verification prefill ${verify - draft}ms, extend prefill ${performance.now() - verify}ms`);

  return {
    warmup,
    tokens: acceptedTokens.map((tokens, batch) => [...tokens, replacementTokens[batch]]),
    numAccepted: bestAccepted,
    numDraftTokens: topks.length,
  }
}

function buildTargetMask(ws: ExecutionWorkspace, topk: number[]): { data: Tensor; indptr: Tensor } {
  const numTokens = totalTreeNodes(topk);
  const totalBits = numTokens * numTokens;
  const byteLen = Math.ceil(totalBits / 8);
  const key = topk.join('_');
  const data = ws.tensors.get(`mtp_target_mask_host_${key}`) || ws.allocPinned([ws.maxBatch * byteLen], "U8", `mtp_target_mask_host_${key}`);

  const boundaries = depthBoundaries(topk);
  data.withPinnedBuffer(data => {
    data.fill(0);
    for (let batch = 0; batch < ws.maxBatch; batch++) {
      const byteOffset = batch * byteLen;
      for (let q = 0; q < numTokens; q++) {
        let cur = q;
        while (true) {
          const bit = q * numTokens + cur;
          data[byteOffset + (bit >> 3)] |= 1 << (bit & 7);
          const p = parentIndex(topk, cur, boundaries);
          if (p === -1) break;
          cur = p;
        }
      }
    }
  });

  const indptr = ws.tensors.get(`mtp_target_mask_indptr_host_${key}`) || ws.allocPinned([ws.maxBatch + 1], "I32", `mtp_target_mask_indptr_host_${key}`);
  indptr.withPinnedBuffer(indptr => {
    for (let batch = 0; batch <= ws.maxBatch; batch++) {
      indptr.writeInt32LE(batch * byteLen, batch * I32);
    }
  });

  return { data, indptr };
}

function ensureTargetCustomMask(ws: ExecutionWorkspace, topk: number[]) {
  const key = topk.join('_');
  let mask = ws.tensors.get(`mtp_target_mask_${key}`);
  let indptr = ws.tensors.get(`mtp_target_mask_indptr_${key}`);
  if (!mask || !indptr) {
    const { data: maskData, indptr: maskIndptrData } = buildTargetMask(ws, topk);
    mask = ws.alloc(maskData.shape, "U8", `mtp_target_mask_${key}`);
    mask.memcpy(maskData, maskData.bytes, MemcpyKind.HostToDevice);
    indptr = ws.alloc(maskIndptrData.shape, "I32", `mtp_target_mask_indptr_${key}`);
    indptr.memcpy(maskIndptrData, maskIndptrData.bytes, MemcpyKind.HostToDevice);
  }

  return {
    mask,
    indptr,
    mode: MaskMode.CausalCustom,
  };
};

function buildChunkedMTPMask(ws: ExecutionWorkspace, topks: number[], depth: number): { data: Tensor; indptr: Tensor; maskKvLenValue: number } {
  const qoLen = totalPaths(topks.slice(0, depth));
  const boundaries = depthBoundaries(topks);
  // Total prior draft tokens in KV cache: all tree nodes at depths 0..depth-2.
  // These map directly to mask columns 0..priorTokens-1.
  const priorTokens = depth > 1 ? boundaries[depth - 2] : 0;
  const maskKvLen = priorTokens + qoLen;
  const totalBits = qoLen * maskKvLen;
  const byteLen = Math.ceil(totalBits / 8);
  const key = `${topks.join('_')}_d${depth}`;

  const data = ws.tensors.get(`mtp_chunk_mask_host_${key}`) || ws.allocPinned([ws.maxBatch * byteLen], "U8", `mtp_chunk_mask_host_${key}`);

  data.withPinnedBuffer(buf => {
    buf.fill(0);
    for (let batch = 0; batch < ws.maxBatch; batch++) {
      const byteOffset = batch * byteLen;
      for (let q = 0; q < qoLen; q++) {
        // Walk the full ancestor chain from this query's tree node to root.
        // Tree node index = priorTokens + q; each ancestor's tree node index
        // is also its mask column (ancestors are all in [0, priorTokens)).
        let cur = priorTokens + q;
        while (cur !== -1) {
          const bit = q * maskKvLen + cur;
          buf[byteOffset + (bit >> 3)] |= 1 << (bit & 7);
          cur = parentIndex(topks, cur, boundaries);
        }
      }
    }
  });

  const indptr = ws.tensors.get(`mtp_chunk_mask_indptr_host_${key}`) || ws.allocPinned([ws.maxBatch + 1], "I32", `mtp_chunk_mask_indptr_host_${key}`);
  indptr.withPinnedBuffer(buf => {
    for (let batch = 0; batch <= ws.maxBatch; batch++) {
      buf.writeInt32LE(batch * byteLen, batch * I32);
    }
  });

  return { data, indptr, maskKvLenValue: maskKvLen };
}

function ensureChunkedMTPMask(ws: ExecutionWorkspace, topks: number[], depth: number): { mask: Tensor; indptr: Tensor; maskKvLen: Tensor } {
  const key = `${topks.join('_')}_d${depth}`;
  let mask = ws.tensors.get(`mtp_chunk_mask_${key}`);
  let indptr = ws.tensors.get(`mtp_chunk_mask_indptr_${key}`);
  let maskKvLen = ws.tensors.get(`mtp_chunk_mask_kvlen_${key}`);

  if (!mask || !indptr || !maskKvLen) {
    const { data: maskData, indptr: maskIndptrData, maskKvLenValue } = buildChunkedMTPMask(ws, topks, depth);

    mask = ws.alloc(maskData.shape, "U8", `mtp_chunk_mask_${key}`);
    mask.memcpy(maskData, maskData.bytes, MemcpyKind.HostToDevice);

    indptr = ws.alloc(maskIndptrData.shape, "I32", `mtp_chunk_mask_indptr_${key}`);
    indptr.memcpy(maskIndptrData, maskIndptrData.bytes, MemcpyKind.HostToDevice);

    const maskKvLenH = ws.allocPinned([ws.maxBatch], "I32", `mtp_chunk_mask_kvlen_host_${key}`);
    maskKvLenH.withPinnedBuffer(buf => {
      for (let batch = 0; batch < ws.maxBatch; batch++) {
        buf.writeInt32LE(maskKvLenValue, batch * I32);
      }
    });
    maskKvLen = ws.alloc([ws.maxBatch], "I32", `mtp_chunk_mask_kvlen_${key}`);
    maskKvLen.memcpy(maskKvLenH, ws.maxBatch * I32, MemcpyKind.HostToDevice);
  }

  return { mask, indptr, maskKvLen };
}

function getPositionIdsChunked(ws: ExecutionWorkspace, originalAllocLens: number[], depth: number, numTokens: number): Tensor {
  const batchSize = originalAllocLens.length;
  const totalTokens = batchSize * numTokens;
  const key = `mtp_chunk_pos_d${depth}_n${numTokens}`;
  const positionIds = ws.ensureAlloc([ws.maxBatch * numTokens], "I32", key);
  const positionIdsH = ws.ensureAllocPinned([ws.maxBatch * numTokens], "I32", `${key}_host`);
  positionIdsH.withPinnedBuffer(buf => {
    for (let batch = 0; batch < batchSize; batch++) {
      for (let p = 0; p < numTokens; p++) {
        buf.writeInt32LE(originalAllocLens[batch] + depth, (batch * numTokens + p) * I32);
      }
    }
  });
  positionIds.memcpy(positionIdsH, totalTokens * I32, MemcpyKind.HostToDevice);
  return positionIds;
}
