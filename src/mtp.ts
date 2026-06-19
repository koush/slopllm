import { CaptureManager } from "./capture-manager";
import { type ChatCache, type ChatModel } from "./chat_model";
import { MaskMode } from "./device_ops";
import { ExecutionWorkspace } from "./execution-workspace";
import { BF16, I32 } from "./glm_ops";
import { MemcpyKind, Tensor } from "./tensor";
import { WorkspaceBase } from "./workspace";

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
  const positionIds = ws.ensureAlloc(ws.positionIds.shape, ws.positionIds.type, `mtp_pos-${numNodes}`);
  const positionIdsH = ws.ensureAllocPinned(ws.positionIds.shape, ws.positionIds.type, `mtp_pos_host-${numNodes}`);
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
export function mtpTreeDecode(
  captureManager: CaptureManager,
  model: ChatModel,
  mtpHiddenStates: Tensor,
  ws: ExecutionWorkspace,
  targetToken: number,
  topks: number[],
  cache: ChatCache,
  tokenizer?: any,
) {
  if (!model.forwardMtp) {
    throw new Error("mtpTreeDecode: model does not support MTP (forwardMtp not defined)");
  }

  const pagedKV = cache.getPagedKV();
  const draftTopk = topks;
  const targetTopk = [1, ...topks];
  const numPaths = totalPaths(draftTopk);
  const numTreeNodes = totalTreeNodes(draftTopk);
  const seq0 = pagedKV.sequences[0];
  const originalAllocLen = seq0.allocLen;
  const pagedKv = cache.getPagedKV();
  const batchSize = pagedKv.sequences.length;

  using _tracker = ws.startTracking(new Set([mtpHiddenStates]));

  const hiddenDim = mtpHiddenStates.shape[1];
  const rowBytes = hiddenDim * BF16; // BF16 = 2 bytes per element

  // Tiled hidden states for the draft tree. Populated incrementally:
  // Used by the prefill-based path (else branch below).
  const tiledHs = ws.ensureAlloc([numTreeNodes, hiddenDim], "BF16", `mtp-tree-hs-${numTreeNodes}`, undefined, 0);

  const hostBuf = ws.ensureAllocPinned([numTreeNodes], "I32", `mtp_verify_host_buf_${numTreeNodes}`);


  const start = performance.now();

  let warmup = false;

  if (true) {
    // current path that decodes in batch
    let hostBufOffset = 0;
    let chainedMtpHiddenState = mtpHiddenStates;
    for (let i = 1; i < topks.length; i++) {
      // every iteration, duplicate all the sequences to add the top-k for this depth
      const currentBatchSize = pagedKv.sequences.length;
      const k = topks[i - 1];
      for (let j = 1; j < k; j++) {
        for (let seqIdx = 0; seqIdx < currentBatchSize; seqIdx++) {
          pagedKv.copySequence(seqIdx + j * currentBatchSize, seqIdx);
        }
      }

      const newBatchSize = pagedKv.sequences.length;
      const state = ws.planDecode(model, newBatchSize, cache);


      warmup ||= !captureManager.isCaptured(['mtp-tree-decode', i, topks.length]);
      chainedMtpHiddenState = captureManager.run(() => {
        // prepare initial input
        if (i === 1) {
          using initialLogits = mtpHiddenStates.linear(model.tensors.get("lm_head.weight")!, currentBatchSize);
          const initialTopk = initialLogits.topk(topks[0], model.cfg.vocabSize);
          using initialIndices = initialTopk.indices;
          using _initialValues = initialTopk.values;
          ws.inputIdsBuf.memcpy(initialIndices, initialIndices.bytes, MemcpyKind.DeviceToDevice);
          // copy the initial indices to host buffer
          hostBuf.memcpy2d(0, currentBatchSize * I32 * topks[0], initialIndices, 0, currentBatchSize * I32 * topks[0], currentBatchSize * I32 * topks[0], 1, MemcpyKind.DeviceToHost);
        }

        state.setInput(ws.inputIdsBuf);

        ws.positionStep(state, model);

        using _expanded = k > 1 ? chainedMtpHiddenState.cat(Array(k - 1).fill(chainedMtpHiddenState), 0) : undefined;
        const expandedHiddenState = _expanded ?? chainedMtpHiddenState;
        using newMtpHiddenStates = model.forwardMtp!(state, expandedHiddenState);
        using logits = newMtpHiddenStates.linear(model.tensors.get("lm_head.weight")!, newBatchSize);
        const logitsTopk = logits.topk(topks[i], model.cfg.vocabSize);
        using _values = logitsTopk.values;
        using indices = logitsTopk.indices;

        // append the new indices to host buffer
        hostBufOffset += currentBatchSize * I32 * topks[i - 1];
        hostBuf.memcpy2d(hostBufOffset, newBatchSize * I32 * topks[i], indices, 0, newBatchSize * I32 * topks[i], newBatchSize * I32 * topks[i], 1, MemcpyKind.DeviceToHost);

        // prepare next input
        if (i !== topks.length - 1) {
          ws.inputIdsBuf.memcpy(indices, indices.bytes, MemcpyKind.DeviceToDevice);
        }

        return newMtpHiddenStates.capture();
      }, ['mtp-tree-decode', i, topks.length]);

      ws.glm.synchronize();
    }

    //

    // clean up the tree of sequences
    while (pagedKv.sequences.length > batchSize) {
      pagedKv.removeSequence(batchSize);
    }
  }
  else {
    // Single-sequence prefill with custom mask: all tree tokens processed at once,
    // with a tree-shaped causal mask preventing cross-branch attention.
    const draftBoundaries = depthBoundaries(draftTopk);
    const mtpCustomMask = ensureMTPCustomMask(ws, draftTopk);
    const treePrefillState = ws.planPrefill(model, batchSize, [numTreeNodes], cache, {
      ...mtpCustomMask,
      positionIds: getPositionIdsMask(ws, originalAllocLen, draftTopk),
    });
    ws.ensureInputCleared();

    warmup ||= !captureManager.isCaptured(['mtp-tree', numTreeNodes]);
    captureManager.run(() => {
      using initialLogits = mtpHiddenStates.linear(model.tensors.get("lm_head.weight")!, batchSize);
      const initialTopk = initialLogits.topk(topks[0], model.cfg.vocabSize);
      using initialIndices = initialTopk.indices;
      using _initialValues = initialTopk.values;
      ws.inputIdsBuf.memcpy(initialIndices, initialIndices.bytes, MemcpyKind.DeviceToDevice);
      treePrefillState.setInput(ws.inputIdsBuf);

      for (let i = 1; i < topks.length; i++) {
        const numPrefillTokens = totalTreeNodes(topks.slice(0, i));

        if (i === 1) {
          if (topks[0] > 1) {
            using cat = mtpHiddenStates.cat(Array(topks[0] - 1).fill(mtpHiddenStates), 0);
            tiledHs.memcpy(cat);
          } else {
            tiledHs.memcpy(mtpHiddenStates);
          }
        }

        using hiddenStates = model.forwardMtp!(treePrefillState, tiledHs);

        // Copy each node's hidden state to its children's positions in tiledHs.
        // For each depth d and child offset c, memcpy2d copies all nodes at depth d
        // to their c-th child's position (pitch = topks[d+1] rows between siblings).
        using tiledHsStream = i < topks.length - 1
          ? ws.glm.withStream(() => {
            const streams: { streamWaitEvent: () => void }[] = [];
            for (let d = 0; d < i; d++) {
              const nodesAtDepth = d > 0 ? draftBoundaries[d] - draftBoundaries[d - 1] : draftBoundaries[0];
              const srcStartRow = d > 0 ? draftBoundaries[d - 1] : 0;
              const childBranchFactor = topks[d + 1];
              for (let c = 1; c < childBranchFactor; c++) {
                const stream = ws.glm.withStream(() => {
                  tiledHs.memcpy2d(
                    (draftBoundaries[d] + c) * rowBytes, childBranchFactor * rowBytes,
                    hiddenStates, srcStartRow * rowBytes, rowBytes,
                    rowBytes, nodesAtDepth,
                    MemcpyKind.DeviceToDevice,
                  );
                });
                streams.push(stream);
              }
            }
            // c=0 copies (child offset 0) run on the default stream
            for (let d = 0; d < i; d++) {
              const nodesAtDepth = d > 0 ? draftBoundaries[d] - draftBoundaries[d - 1] : draftBoundaries[0];
              const srcStartRow = d > 0 ? draftBoundaries[d - 1] : 0;
              const childBranchFactor = topks[d + 1];
              tiledHs.memcpy2d(
                draftBoundaries[d] * rowBytes, childBranchFactor * rowBytes,
                hiddenStates, srcStartRow * rowBytes, rowBytes,
                rowBytes, nodesAtDepth,
                MemcpyKind.DeviceToDevice,
              );
            }
            for (const s of streams) s.streamWaitEvent();
          })
          : undefined;

        using logits = treePrefillState.computeLogits(hiddenStates, model, true);
        const logitsTopk = logits.topk(topks[i], model.cfg.vocabSize);
        using _values = logitsTopk.values;
        using indices = logitsTopk.indices;

        const leafCount = totalPaths(topks.slice(0, i));
        const nodeCount = numPrefillTokens - leafCount;
        const writeOffset = numPrefillTokens * I32;
        const writeCount = leafCount * topks[i];
        ws.inputIdsBuf.memcpy2d(
          writeOffset,
          writeCount * I32,
          indices,
          (nodeCount * topks[i]) * I32,
          writeCount * I32,
          writeCount * I32,
          1,
          MemcpyKind.DeviceToDevice,
        );

        tiledHsStream?.streamWaitEvent();
      }

      hostBuf.memcpy(ws.inputIdsBuf, hostBuf.bytes, MemcpyKind.DeviceToHost);
    }, ['mtp-tree', numTreeNodes]);
    ws.glm.synchronize();
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

  warmup ||= !captureManager.isCaptured(['mtp-verify', numVerificationTokens]);
  const kvCacheLayers = captureManager.run(() => {
    const kvCacheLayers: { appendCkv: Tensor, appendKpe: Tensor, appendCkvOrig: Tensor, appendKpeOrig: Tensor, cacheIdx: number, kvLoraRank: number, qkRopeDim: number }[] = [];

    const mlaKVCacheAppendOrig = targetPrefillState.mlaKvCacheAppend.bind(targetPrefillState);
    targetPrefillState.mlaKvCacheAppend = (appendCkv, appendKpe, cacheIdx, kvLoraRank, qkRopeDim) => {
      // appendCkv.removeTracking();
      // appendKpe.removeTracking();
      kvCacheLayers.push({ appendCkv: appendCkv.capture(), appendKpe: appendKpe.capture(), appendCkvOrig: appendCkv, appendKpeOrig: appendKpe, cacheIdx, kvLoraRank, qkRopeDim });
      mlaKVCacheAppendOrig(appendCkv, appendKpe, cacheIdx, kvLoraRank, qkRopeDim);
    };

    using hiddenStates = model.forwardModel(targetPrefillState);
    using logits = targetPrefillState.computeLogits(hiddenStates, model, true);
    using argmaxResult = logits.argmax();
    const argmaxHost = ws.ensureAllocPinned(argmaxResult.shape, argmaxResult.type, `mtp_verify_argmax_host_${numVerificationTokens}`);
    argmaxHost.memcpy(argmaxResult, argmaxResult.bytes, MemcpyKind.DeviceToHost);

    hiddenStateStaging.memcpy(hiddenStates, undefined, MemcpyKind.DeviceToDevice);

    return kvCacheLayers;
  }, ['mtp-verify', numVerificationTokens]);

  ws.glm.synchronize();

  const verify = performance.now();

  const argmaxHost = ws.tensors.get(`mtp_verify_argmax_host_${numVerificationTokens}`)!;
  const argmaxBuf = argmaxHost.readPinnedBuffer();

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
  }

  // target can now be truncate to the original length, plan another prefill with
  // the accepted token length.
  // target layers: write the captured kvCacheLayers
  // mtp layers: perform extended prefill as usual
  seq0.truncate(originalAllocLen);

  const mtpExtendPrefill = ws.planPrefill(model, batchSize, [finishCount], cache);
  mtpExtendPrefill.setInput([[...acceptedTokens, bestReplacement]]);

  warmup ||= !captureManager.isCaptured(['mtp-replace', finishCount]);
  captureManager.run(() => {
    for (const layer of kvCacheLayers) {
      mtpExtendPrefill.mlaKvCacheAppend(layer.appendCkv, layer.appendKpe, layer.cacheIdx, layer.kvLoraRank, layer.qkRopeDim);
      layer.appendCkvOrig[Symbol.dispose]();
      layer.appendKpeOrig[Symbol.dispose]();
    }

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

  ws.glm.synchronize();


  // if (acceptedTokens.length) {
  //   console.warn(`MTP verify: accepted ${acceptedTokens.length} tokens: ${acceptedTokens.map(t => tokenizer.decode([t], { skip_special_tokens: false }))}, replacement: ${tokenizer.decode([bestReplacement], { skip_special_tokens: false })}, target: ${tokenizer.decode([targetToken], { skip_special_tokens: false })}`);
  // }

  // timings
  // console.log(`MTP tree decode: ${draft - start}ms, verification prefill ${verify - draft}ms, extend prefill ${performance.now() - verify}ms`);

  return {
    warmup,
    tokens: [...acceptedTokens, bestReplacement],
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

function buildMTPMask(ws: WorkspaceBase, topk: number[]): { data: Tensor; indptr: Tensor } {
  const numTokens = totalTreeNodes(topk);
  const totalBits = numTokens * numTokens;
  const byteLen = Math.ceil(totalBits / 8);
  const key = topk.join('_');
  const data = ws.tensors.get(`mtp_draft_mask_host_${key}`) || ws.allocPinned([byteLen], "U8", `mtp_draft_mask_host_${key}`);

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

  const indptr = ws.tensors.get(`mtp_draft_mask_indptr_host_${key}`) || ws.allocPinned([2], "I32", `mtp_draft_mask_indptr_host_${key}`);
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

function ensureMTPCustomMask(ws: WorkspaceBase, topk: number[]) {
  const key = topk.join('_');
  let mask = ws.tensors.get(`mtp_draft_mask_${key}`);
  let indptr = ws.tensors.get(`mtp_draft_mask_indptr_${key}`);
  if (!mask || !indptr) {
    const { data: maskData, indptr: maskIndptrData } = buildMTPMask(ws, topk);
    mask = ws.alloc(maskData.shape, "U8", `mtp_draft_mask_${key}`);
    mask.memcpy(maskData, maskData.bytes, MemcpyKind.HostToDevice);
    indptr = ws.alloc(maskIndptrData.shape, "I32", `mtp_draft_mask_indptr_${key}`);
    indptr.memcpy(maskIndptrData, maskIndptrData.bytes, MemcpyKind.HostToDevice);
  }

  return {
    mask,
    indptr,
    mode: MaskMode.CausalCustom,
  };
};

