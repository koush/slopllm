import { CaptureManager } from "./capture-manager";
import { type ChatCache, type ChatModel } from "./chat_model";
import { MaskMode } from "./device_ops";
import { MemcpyKind } from "./enums";
import { ExecutionState, ExecutionWorkspace } from "./execution-workspace";
import { BF16, I32 } from "./glm_ops";
import { type Tensor } from "./tensor";
import { UsingHolder } from "./using-holder";
import { WorkspaceBase } from "./workspace";

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

function getPositionIdsMask(ws: ExecutionWorkspace, originalAllocLen: number, topk: number[]) {
  const numNodes = totalTreeNodes(topk);
  const maxShape = [ws.maxBatch * ws.maxSeqLen];
  const positionIds = ws.ensureAlloc(maxShape, "I32", `mtp_pos-${numNodes}`);
  const positionIdsH = ws.ensureAllocPinned(maxShape, "I32", `mtp_pos_host-${numNodes}`);
  const boundaries = depthBoundaries(topk);
  positionIdsH.withPinnedBuffer(buf => {
    let posOff = 0;
    for (let p = 0; p < numNodes; p++) {
      let depth = 0;
      while (depth < boundaries.length && p >= boundaries[depth]) {
        depth++;
      }
      buf.writeInt32LE(originalAllocLen + depth, posOff * I32);
      posOff++;
    }
  });
  positionIds.memcpy(positionIdsH, numNodes * I32, MemcpyKind.HostToDevice);
  return positionIds;
}

/**
 * Tree-structured MTP decode using single-sequence prefill with custom masks.
 *
 * Iteration 0 reuses the target model's decode slot (batch=1) to produce the
 * root's top-2 predictions. Iterations 1+ append all tree tokens built so far
 * to seq0, run a single prefill pass with a tree-shaped custom mask, and extract
 * top-2 at the leaf positions. After each prefill iteration, seq0 is rolled back
 * to its original length. This avoids forking KV cache sequences entirely.
 *
 * The custom mask ensures each draft token only attends to the prefix plus its
 * ancestors in the binary tree, so different branches don't cross-attend.
 *
 * Prerequisites:
 *   - The target model must have just decoded (planDecode + positionStep + forward
 *     already called for the current position)
 *   - state must be the target model's decode state (batchSize = 1)
 *   - gpuSampleResult must be [1] I32 GPU tensor: target model's sampled token
 *   - model.forwardMtp must exist (MTP enabled)
 *   - cache.getPagedKV().sequences must have exactly 1 sequence (seq 0)
 *
 * @param state - The target model's decode ExecutionState
 * @param captureManager - Capture manager for CUDA graph replay
 * @param model - The chat model (must support forwardMtp)
 * @param mtpHiddenStates - The mtp layer's hidden states [1, hidden] BF16 from draft prefill extend
 * @param ws - Execution workspace
 * @param topks - Array of top-k values per MTP depth (e.g. [2,2,2] for a binary tree with 3 layers)
 * @param cache - Chat cache (paged KV cache)
 * @returns Array of accepted draft tokens plus the replacement token
 */
export async function mtpTreeDecode(
  captureManager: CaptureManager,
  model: ChatModel,
  mtpHiddenStates: Tensor,
  sharedSlots: Tensor,
  sharedSlotsLength: Tensor,
  ws: ExecutionWorkspace,
  targetToken: number,
  topks: number[],
  cache: ChatCache,
) {
  if (!model.forwardMtp) {
    throw new Error("mtpTreeDecode: model does not support MTP (forwardMtp not defined)");
  }

  const tokenizer = model.tokenizer;
  const lmHead = model.tensors.get("lm_head.weight")!;

  const pagedKV = cache.getPagedKV();
  const draftTopk = topks;
  const targetTopk = [1, ...topks];
  const numPaths = totalPaths(draftTopk);
  const numTreeNodes = totalTreeNodes(draftTopk);
  const seq0 = pagedKV.sequences[0];
  const originalAllocLen = seq0.allocLen;
  const pagedKv = cache.getPagedKV();
  const batchSize = pagedKv.sequences.length;

  const hiddenDim = mtpHiddenStates.shape[1];
  const rowBytes = hiddenDim * BF16; // BF16 = 2 bytes per element

  const hostBuf = ws.ensureAllocPinned([numTreeNodes], "I32", `mtp_verify_host_buf_${numTreeNodes}`);


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
      using initialLogits = mtpHiddenStates.linear(lmHead);
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
          using initialLogits = mtpHiddenStates.linear(lmHead);
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
      const posIds = getPositionIdsChunked(ws, originalAllocLen, depth, qoLen);
      const state = ws.planPrefill(model, batchSize, [qoLen], cache, {
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
          using initialLogits = mtpHiddenStates.linear(lmHead);
          const initialTopk = initialLogits.topk(topks[0], model.cfg.vocabSize);
          using _initialValues = initialTopk.values;
          using initialIndices = initialTopk.indices;
          state.inputIdsBuf.memcpy(initialIndices, initialIndices.bytes, MemcpyKind.DeviceToDevice);
          hostBuf.memcpy2d(0, topks[0] * I32, initialIndices, 0, topks[0] * I32, topks[0] * I32, 1, MemcpyKind.DeviceToHost);
        }

        state.setInput(state.inputIdsBuf);

        // Expand prevHs: replicate each parent's hidden state for expandK children
        using _expandedHs = expandK > 1 ? ws.alloc([qoLen, hiddenDim], "BF16") : undefined;
        const expandedHs = _expandedHs ?? prevHs;
        if (expandK > 1) {
          for (let c = 0; c < expandK; c++) {
            _expandedHs!.memcpy2d(
              c * rowBytes, expandK * rowBytes,
              prevHs, 0, rowBytes,
              rowBytes, hsPrevQoLen,
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
        const hostOffset = draftBoundaries[depth - 1] * I32;
        const hostCount = qoLen * topks[depth];
        hostBuf.memcpy2d(hostOffset, hostCount * I32, indices, 0, hostCount * I32, hostCount * I32, 1, MemcpyKind.DeviceToHost);

        // Prepare input for next depth — copy to next state's inputIdsBuf
        if (next) {
          next.inputIdsBuf.memcpy(indices, indices.bytes, MemcpyKind.DeviceToDevice);
        }

        prevHs.memcpy(hiddenStates, hiddenStates.bytes, MemcpyKind.DeviceToDevice);
      }
    }, captureKey);

    await ws.glm.synchronizeAsync();

    mtpHiddenStates.removeTracking();
    sharedSlots?.removeTracking();
    sharedSlotsLength?.removeTracking();
  }

  const draft = performance.now();

  // after iterative tree decode/prefill, rewind for tree verification prefill
  seq0.truncate(originalAllocLen);

  const verificationTokens: number[][] = [];
  for (let batch = 0; batch < batchSize; batch++) {
    let batchOffset = batch * numTreeNodes * I32;
    // this is a code smell, fix later when real batch support is added, right now code is batch 1
    const batchTokens: number[] = [targetToken];
    for (let i = 0; i < numTreeNodes; i++) {
      batchTokens.push(hostBuf.readPinnedBuffer().readInt32LE(batchOffset + i * I32));
    }
    verificationTokens.push(batchTokens);
  }

  const numVerificationTokens = numTreeNodes + 1; // include target token
  const targetCustomMask = ensureTargetCustomMask(ws, targetTopk);
  const targetPrefillState = ws.planPrefill(model, batchSize, [numVerificationTokens], cache, {
    ...targetCustomMask,
    positionIds: getPositionIdsMask(ws, originalAllocLen, targetTopk),
  });

  targetPrefillState.setInput(verificationTokens);

  const hiddenStateStaging = ws.ensureAlloc([numVerificationTokens, hiddenDim], "BF16", `mtp-tree-hs-staging-${numVerificationTokens}`, undefined, 0);
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
    const argmaxHost = ws.ensureAllocPinned(argmaxResult.shape, argmaxResult.type, `mtp_verify_argmax_host_${numVerificationTokens}`);
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

  let bestPath = 0;
  let bestAccepted = -1;
  let bestReplacement = -1;

  const targetBoundaries = depthBoundaries(targetTopk);
  const draftStrides = topks.map((_, i) => totalPaths(topks.slice(i + 1)));

  for (let path = 0; path < numPaths; path++) {
    let accepted = 0;
    let nodeIdx = 0;
    for (let layer = 0; layer < topks.length; layer++) {
      const digit = pathDigit(topks, path, layer, draftStrides);
      const childIdx = childIndex(targetTopk, nodeIdx, digit, targetBoundaries);
      const draftToken = verificationTokens[0][childIdx];
      const targetToken = argmaxBuf.readInt32LE(nodeIdx * 4);
      if (draftToken === targetToken) {
        accepted++;
        nodeIdx = childIdx;
      } else {
        break;
      }
    }
    if (accepted > bestAccepted) {
      bestAccepted = accepted;
      bestPath = path;
      bestReplacement = argmaxBuf.readInt32LE(nodeIdx * 4);
    }
  }

  const finishCount = bestAccepted + 1; // target + accepted (replacement added via decode)

  const acceptedTokens: number[] = [];
  let nodeIdx = 0;
  for (let layer = 0; layer < bestAccepted; layer++) {
    const digit = pathDigit(topks, bestPath, layer, draftStrides);
    const childIdx = childIndex(targetTopk, nodeIdx, digit, targetBoundaries);
    acceptedTokens.push(verificationTokens[0][childIdx]);
    nodeIdx = childIdx;
  }

  {
    // Accepted tokens along bestPath: root (node 0) + accepted children.
    // The replacement token is handled separately via a decode step, not from the tree.
    const acceptedNodeIndices: number[] = [0];
    let ni = 0;
    for (let layer = 0; layer < bestAccepted; layer++) {
      const digit = pathDigit(topks, bestPath, layer, draftStrides);
      ni = childIndex(targetTopk, ni, digit, targetBoundaries);
      acceptedNodeIndices.push(ni);
    }

    // After capture block, in-place reorder accepted rows to front:
    // targetHiddenStates now has shape [numVerificationTokens, hiddenDim]
    // Move accepted rows to positions 0..finishCount-1
    for (let i = 0; i < finishCount; i++) {
      const srcIdx = acceptedNodeIndices[i];
      if (srcIdx !== i) {
        hiddenStateStaging.memcpy2d(
          i * rowBytes, rowBytes,
          hiddenStateStaging, srcIdx * rowBytes, rowBytes,
          rowBytes, 1,
          MemcpyKind.DeviceToDevice,
        );
      }
    }

    for (const layer of kvCacheLayers) {
      const kvLoraRank = layer.kvLoraRank;
      const qkRopeDim = layer.qkRopeDim;
      for (let i = 0; i < finishCount; i++) {
        const srcIdx = acceptedNodeIndices[i];
        if (srcIdx !== i) {
          layer.appendCkv.memcpy2d(
            i * kvLoraRank * 2, kvLoraRank * 2,
            layer.appendCkv, srcIdx * kvLoraRank * 2, kvLoraRank * 2,
            kvLoraRank * 2, 1,
            MemcpyKind.DeviceToDevice,
          );
          layer.appendKpe.memcpy2d(
            i * qkRopeDim * 2, qkRopeDim * 2,
            layer.appendKpe, srcIdx * qkRopeDim * 2, qkRopeDim * 2,
            qkRopeDim * 2, 1,
            MemcpyKind.DeviceToDevice,
          );
        }
      }
    }
    for (const layer of indexerKvCacheLayers) {
      const indexHeadDim = layer.indexHeadDim;
      for (let i = 0; i < finishCount; i++) {
        const srcIdx = acceptedNodeIndices[i];
        if (srcIdx !== i) {
          layer.appendIdxK.memcpy2d(
            i * indexHeadDim * 2, indexHeadDim * 2,
            layer.appendIdxK, srcIdx * indexHeadDim * 2, indexHeadDim * 2,
            indexHeadDim * 2, 1,
            MemcpyKind.DeviceToDevice,
          );
        }
      }
    }
  }

  // target can now be truncate to the original length, plan another prefill with
  // the accepted token length.
  // target layers: write the captured kvCacheLayers
  // mtp layers: perform extended prefill as usual
  seq0.truncate(originalAllocLen);
  await ws.glm.synchronizeAsync();

  const mtpExtendPrefill = ws.planPrefill(model, batchSize, [finishCount], cache);
  mtpExtendPrefill.setInput([[...acceptedTokens, bestReplacement]]);
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

  warmup ||= !mtpExtendPrefill.isCaptured(captureManager, ['mtp-replace', finishCount]);
  mtpExtendPrefill.capture(captureManager, { mtpHiddenStates, sharedSlots, sharedSlotsLength }, () => {
    mtpExtendPrefill.sharedSlots = new UsingHolder(sharedSlots?.capture());
    mtpExtendPrefill.sharedSlotsLength = new UsingHolder(sharedSlotsLength?.capture());

    using verfiedHiddenStates = hiddenStateStaging.slice(0, 0, finishCount);
    using mtpHs = model.forwardMtp!(mtpExtendPrefill, verfiedHiddenStates);
    // MTP convention (same as rotateInputIds in prefill): at position P, the input
    // token is P+1 paired with hidden state at P. Here input[0] = acceptedTokens[0]
    // (the token after targetToken) with verifiedHiddenStates[0] (target model HS at
    // targetToken). The MTP KV at position P thus encodes token P+1, while the target
    // model KV at the same position encodes token P — each layer has its own KV slot
    // so this is safe. The last row is the seed for the next draft iteration.
    using newMtpHiddenStates = mtpHs.slice(0, -1, 1);
    mtpHiddenStates.memcpy(newMtpHiddenStates);
  }, ['mtp-replace', finishCount]);
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
    tokens: [...acceptedTokens, bestReplacement],
    numAccepted: bestAccepted,
    numDraftTokens: topks.length,
  }
}

function buildTargetMask(ws: WorkspaceBase, topk: number[]): { data: Tensor; indptr: Tensor } {
  const numTokens = totalTreeNodes(topk);
  const totalBits = numTokens * numTokens;
  const byteLen = Math.ceil(totalBits / 8);
  const key = topk.join('_');
  const data = ws.tensors.get(`mtp_target_mask_host_${key}`) || ws.allocPinned([byteLen], "U8", `mtp_target_mask_host_${key}`);

  const boundaries = depthBoundaries(topk);
  data.withPinnedBuffer(data => {
    data.fill(0);
    for (let q = 0; q < numTokens; q++) {
      let cur = q;
      while (true) {
        const bit = q * numTokens + cur;
        data[bit >> 3] |= 1 << (bit & 7);
        const p = parentIndex(topk, cur, boundaries);
        if (p === -1) break;
        cur = p;
      }
    }
  });

  const indptr = ws.tensors.get(`mtp_target_mask_indptr_host_${key}`) || ws.allocPinned([2], "I32", `mtp_target_mask_indptr_host_${key}`);
  indptr.withPinnedBuffer(indptr => {
    indptr.writeInt32LE(0, 0);
    indptr.writeInt32LE(byteLen, 4);
  });

  return { data, indptr };
}

function ensureTargetCustomMask(ws: WorkspaceBase, topk: number[]) {
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

function buildChunkedMTPMask(ws: WorkspaceBase, topks: number[], depth: number): { data: Tensor; indptr: Tensor; maskKvLenValue: number } {
  const qoLen = totalPaths(topks.slice(0, depth));
  const boundaries = depthBoundaries(topks);
  // Total prior draft tokens in KV cache: all tree nodes at depths 0..depth-2.
  // These map directly to mask columns 0..priorTokens-1.
  const priorTokens = depth > 1 ? boundaries[depth - 2] : 0;
  const maskKvLen = priorTokens + qoLen;
  const totalBits = qoLen * maskKvLen;
  const byteLen = Math.ceil(totalBits / 8);
  const key = `${topks.join('_')}_d${depth}`;

  const data = ws.tensors.get(`mtp_chunk_mask_host_${key}`) || ws.allocPinned([byteLen], "U8", `mtp_chunk_mask_host_${key}`);

  data.withPinnedBuffer(buf => {
    buf.fill(0);
    for (let q = 0; q < qoLen; q++) {
      // Walk the full ancestor chain from this query's tree node to root.
      // Tree node index = priorTokens + q; each ancestor's tree node index
      // is also its mask column (ancestors are all in [0, priorTokens)).
      let cur = priorTokens + q;
      while (cur !== -1) {
        const bit = q * maskKvLen + cur;
        buf[bit >> 3] |= 1 << (bit & 7);
        cur = parentIndex(topks, cur, boundaries);
      }
    }
  });

  const indptr = ws.tensors.get(`mtp_chunk_mask_indptr_host_${key}`) || ws.allocPinned([2], "I32", `mtp_chunk_mask_indptr_host_${key}`);
  indptr.withPinnedBuffer(buf => {
    buf.writeInt32LE(0, 0);
    buf.writeInt32LE(byteLen, 4);
  });

  return { data, indptr, maskKvLenValue: maskKvLen };
}

function ensureChunkedMTPMask(ws: WorkspaceBase, topks: number[], depth: number): { mask: Tensor; indptr: Tensor; maskKvLen: Tensor } {
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

    const maskKvLenH = ws.allocPinned([1], "I32", `mtp_chunk_mask_kvlen_host_${key}`);
    maskKvLenH.withPinnedBuffer(buf => {
      buf.writeInt32LE(maskKvLenValue, 0);
    });
    maskKvLen = ws.alloc([1], "I32", `mtp_chunk_mask_kvlen_${key}`);
    maskKvLen.memcpy(maskKvLenH, 4, MemcpyKind.HostToDevice);
  }

  return { mask, indptr, maskKvLen };
}

function getPositionIdsChunked(ws: ExecutionWorkspace, originalAllocLen: number, depth: number, numTokens: number): Tensor {
  const key = `mtp_chunk_pos_d${depth}_n${numTokens}`;
  const positionIds = ws.ensureAlloc([numTokens], "I32", key);
  const positionIdsH = ws.ensureAllocPinned([numTokens], "I32", `${key}_host`);
  positionIdsH.withPinnedBuffer(buf => {
    for (let p = 0; p < numTokens; p++) {
      buf.writeInt32LE(originalAllocLen + depth, p * I32);
    }
  });
  positionIds.memcpy(positionIdsH, numTokens * I32, MemcpyKind.HostToDevice);
  return positionIds;
}
