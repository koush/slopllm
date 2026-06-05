import { CaptureManager } from "./capture-manager";
import { type ChatCache, type ChatModel } from "./chat_model";
import { MaskMode } from "./device_ops";
import { ExecutionWorkspace } from "./execution-workspace";
import { BF16, I32 } from "./glm_ops";
import { MemcpyKind, Tensor } from "./tensor";
import { WorkspaceBase } from "./workspace";

function getTargetPositionIdsMask(ws: ExecutionWorkspace, originalAllocLen: number, totalTreeNodes: number) {
  const positionIds = ws.ensureAlloc(ws.positionIds.shape, ws.positionIds.type, `mtp_target_pos-${totalTreeNodes}`);
  function treeDepth(n: number): number {
    let depth = 0;
    while (n > 0) {
      n = (n - 1) >> 1;
      depth++;
    }
    return depth;
  }

  const positionIdsH = ws.ensureAllocPinned(ws.positionIds.shape, ws.positionIds.type, `mtp_target_pos_host-${totalTreeNodes}`);
  positionIdsH.withPinnedBuffer(buf => {
    let posOff = 0;
    for (let p = 0; p < totalTreeNodes; p++) {
      buf.writeInt32LE(originalAllocLen + treeDepth(p), posOff * I32);
      posOff++;
    }
  });
  positionIds.memcpy(positionIdsH, totalTreeNodes * I32, MemcpyKind.HostToDevice);
  return positionIds;
}

function getMTPPositionIdsMask(ws: ExecutionWorkspace, originalAllocLen: number, totalTreeNodes: number) {
  const positionIds = ws.ensureAlloc(ws.positionIds.shape, ws.positionIds.type, `mtp_draft_pos-${totalTreeNodes}`);
  function mtpTreeDepth(n: number): number {
    if (n < 2) return 0;
    let depth = 0;
    while (n >= 2) {
      n = (n - 2) >> 1;
      depth++;
    }
    return depth;
  }

  const positionIdsH = ws.ensureAllocPinned(ws.positionIds.shape, ws.positionIds.type, `mtp_draft_pos_host-${totalTreeNodes}`);
  positionIdsH.withPinnedBuffer(buf => {
    let posOff = 0;
    for (let p = 0; p < totalTreeNodes; p++) {
      buf.writeInt32LE(originalAllocLen + mtpTreeDepth(p), posOff * I32);
      posOff++;
    }
  });
  positionIds.memcpy(positionIdsH, totalTreeNodes * I32, MemcpyKind.HostToDevice);
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
 * @param gpuSampleResult - [1] I32 GPU tensor: target model's sampled token
 * @param nextn - Number of MTP layers (tree depth)
 * @param cache - Chat cache (paged KV cache)
 * @returns MtpTreeResult with validation sequences tensor and nextn
 */
export function mtpTreeDecode(
  captureManager: CaptureManager,
  model: ChatModel,
  mtpHiddenStates: Tensor,
  ws: ExecutionWorkspace,
  targetToken: number,
  nextn: number,
  cache: ChatCache,
  tokenizer?: any,
) {
  if (!model.forwardMtp) {
    throw new Error("mtpTreeDecode: model does not support MTP (forwardMtp not defined)");
  }

  const pagedKV = cache.getPagedKV();
  const totalPaths = 1 << nextn;
  const totalTreeNodes = 2 * totalPaths - 2;
  const seq0 = pagedKV.sequences[0];
  const originalAllocLen = seq0.allocLen;
  const batchSize = 1;

  using _tracker = ws.startTracking(new Set([mtpHiddenStates]));

  const hiddenDim = mtpHiddenStates.shape[1];
  const rowBytes = hiddenDim * BF16; // BF16 = 2 bytes per element

  // Tiled hidden states for the 2-root binary tree. Populated incrementally:
  //   tiledHs[0]     = mtpHS                 (root 0, written at i=1)
  //   tiledHs[1]     = mtpHS                 (root 1, written at i=1)
  //   tiledHs[2p+2] = hiddenStates[p]       (child 1, written at end of iter i)
  //   tiledHs[2p+3] = hiddenStates[p]       (child 2, written at end of iter i)
  // Each iteration's hiddenStates output is copied directly to tiledHs for the
  // next iteration, eliminating the need for a separate prevMtpOutput buffer.
  const tiledHs = ws.ensureAlloc([totalTreeNodes, hiddenDim], "BF16", `mtp-tree-hs-${totalTreeNodes}`, undefined, 0);

  const hostBuf = ws.ensureAllocPinned([totalTreeNodes], "I32", `mtp_verify_host_buf_${totalTreeNodes}`);

  const mtpCustomMask = ensureMTPCustomMask(ws, totalTreeNodes);
  // this creates an oversized tree for all iterations, but the masking prevents the padding tokens from affecting the results that
  // we care about per iteration.
  const treePrefillState = ws.planPrefill(model, batchSize, [totalTreeNodes], cache, {
    ...mtpCustomMask,
    positionIds: getMTPPositionIdsMask(ws, originalAllocLen, totalTreeNodes),
  });
  ws.ensureInputCleared();

  const start = performance.now();
  const mlaKVCacheAppendOrig = treePrefillState.mlaKvCacheAppend.bind(treePrefillState);

  captureManager.run(() => {
    using initialLogits = mtpHiddenStates.linear(model.tensors.get("lm_head.weight")!, batchSize);
    const initialTopk = initialLogits.topk(2, model.cfg.vocabSize);
    using initialIndices = initialTopk.indices;
    using _initialValues = initialTopk.values;
    ws.inputIdsBuf.memcpy(initialIndices, initialIndices.bytes, MemcpyKind.DeviceToDevice);
    treePrefillState.setInput(ws.inputIdsBuf);

    // Iteration: single-sequence prefill with tree-shaped custom mask
    // 2 -> 6 -> 14 for i = 0,1,2 (nextn=3)
    for (let i = 1; i < nextn; i++) {
      const numPrefillTokens = (1 << (i + 1)) - 2;

      if (i === 1) {
        using cat = mtpHiddenStates.cat([mtpHiddenStates], 0);
        tiledHs.memcpy(cat);
      }

      using hiddenStates = model.forwardMtp!(treePrefillState, tiledHs);

      // Copy this iteration's hiddenStates directly to tiledHs for next iteration.
      using tiledHsStream = i < nextn - 1
        ? ws.glm.withStream(() => {
          using oddStream = ws.glm.withStream(() => {
            // Rows 3,5,7,…: child 2 of each parent (2p+3)
            tiledHs.memcpy2d(
              3 * rowBytes, 2 * rowBytes,
              hiddenStates, 0, rowBytes,
              rowBytes, numPrefillTokens,
              MemcpyKind.DeviceToDevice,
            );
          });
          // Rows 2,4,6,…: child 1 of each parent (2p+2)
          tiledHs.memcpy2d(
            2 * rowBytes, 2 * rowBytes,
            hiddenStates, 0, rowBytes,
            rowBytes, numPrefillTokens,
            MemcpyKind.DeviceToDevice,
          );

          oddStream.streamWaitEvent();
        })
        : undefined;

      using logits = treePrefillState.computeLogits(hiddenStates, model, true);
      const topk = logits.topk(2, model.cfg.vocabSize);
      using _values = topk.values;
      using indices = topk.indices;

      // number of leaves in the prefill
      const leafCount = 1 << i;
      // number of nodes in the prefill
      const nodeCount = numPrefillTokens - leafCount;
      // start writing after existing tokens (ie 1, 3, 7) to append the new layer's tree top 2 predictions
      const writeOffset = numPrefillTokens * I32;
      // each leaf has 2 predictions
      const writeCount = leafCount * 2;
      ws.inputIdsBuf.memcpy2d(
        writeOffset, // dst offset
        writeCount * I32, // dst pitch (unused)
        indices, // src
        (nodeCount * 2) * I32, // src offset (each node which has already processed will have 2 predictions that we can skip)
        writeCount * I32, // src pitch (unused)
        writeCount * I32, // width
        1, // height (no pitch)
        MemcpyKind.DeviceToDevice,
      );

      tiledHsStream?.streamWaitEvent();
    }

    hostBuf.memcpy(ws.inputIdsBuf, hostBuf.bytes, MemcpyKind.DeviceToHost);
  }, ['mtp-tree', totalTreeNodes]);

  ws.glm.synchronize();

  // after iterative tree decode/prefill, rewind for tree verification prefill
  seq0.truncate(originalAllocLen);

  const verificationTokens: number[][] = [];
  for (let batch = 0; batch < batchSize; batch++) {
    let batchOffset = batch * totalTreeNodes * I32;
    // this is a code smell, fix later when real batch support is added, right now code is batch 1
    const batchTokens: number[] = [targetToken];
    for (let i = 0; i < totalTreeNodes; i++) {
      batchTokens.push(hostBuf.readPinnedBuffer().readInt32LE(batchOffset + i * I32));
    }
    verificationTokens.push(batchTokens);
  }

  const totalVerificationTokens = totalTreeNodes + 1; // include target token
  const targetCustomMask = ensureTargetCustomMask(ws, totalVerificationTokens);
  const targetPrefillState = ws.planPrefill(model, batchSize, [totalVerificationTokens], cache, {
    ...targetCustomMask,
    positionIds: getTargetPositionIdsMask(ws, originalAllocLen, totalVerificationTokens),
  });

  targetPrefillState.setInput(verificationTokens);

  const hiddenStateStaging = ws.ensureAlloc([totalVerificationTokens, hiddenDim], "BF16", `mtp-tree-hs-staging-${totalVerificationTokens}`, undefined, 0);

  const kvCacheLayers = captureManager.run(() => {
    const kvCacheLayers: { appendCkv: Tensor, appendKpe: Tensor, cacheIdx: number, kvLoraRank: number, qkRopeDim: number }[] = [];

    targetPrefillState.mlaKvCacheAppend = (appendCkv, appendKpe, cacheIdx, kvLoraRank, qkRopeDim) => {
      appendCkv.removeTracking();
      appendKpe.removeTracking();
      kvCacheLayers.push({ appendCkv, appendKpe, cacheIdx, kvLoraRank, qkRopeDim });
      mlaKVCacheAppendOrig(appendCkv, appendKpe, cacheIdx, kvLoraRank, qkRopeDim);
    };

    using hiddenStates = model.forwardModel(targetPrefillState);
    using logits = targetPrefillState.computeLogits(hiddenStates, model, true);
    using argmaxResult = logits.argmax();
    const argmaxHost = ws.ensureAllocPinned(argmaxResult.shape, argmaxResult.type, `mtp_verify_argmax_host_${totalVerificationTokens}`);
    argmaxHost.memcpy(argmaxResult, argmaxResult.bytes, MemcpyKind.DeviceToHost);

    hiddenStateStaging.memcpy(hiddenStates, undefined, MemcpyKind.DeviceToDevice);

    return kvCacheLayers;
  }, ['mtp-verify', totalVerificationTokens]);

  const draft = performance.now();

  const argmaxHost = ws.tensors.get(`mtp_verify_argmax_host_${totalVerificationTokens}`)!;
  const argmaxBuf = argmaxHost.readPinnedBuffer();

  let bestPath = 0;
  let bestAccepted = -1;
  let bestReplacement = -1;

  for (let path = 0; path < totalPaths; path++) {
    let accepted = 0;
    let nodeIdx = 0;
    for (let layer = 0; layer < nextn; layer++) {
      const childIdx = nodeIdx * 2 + 1 + ((path >> (nextn - 1 - layer)) & 1);
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


  const acceptedTokens: number[] = [];
  let nodeIdx = 0;
  for (let layer = 0; layer < bestAccepted; layer++) {
    const childIdx = nodeIdx * 2 + 1 + ((bestPath >> (nextn - 1 - layer)) & 1);
    acceptedTokens.push(verificationTokens[0][childIdx]);
    nodeIdx = childIdx;
  }

  // Accepted tokens along bestPath: root (node 0) + accepted children.
  // The replacement token is handled separately via a decode step, not from the tree.
  const finishCount = bestAccepted + 1; // target + accepted (replacement added via decode)
  const acceptedNodeIndices: number[] = [0];
  {
    let ni = 0;
    for (let layer = 0; layer < bestAccepted; layer++) {
      ni = ni * 2 + 1 + ((bestPath >> (nextn - 1 - layer)) & 1);
      acceptedNodeIndices.push(ni);
    }
  }
  const acceptedSet = new Set(acceptedNodeIndices);

  // Build positionIds: accepted tokens → sequential positions, rejected → scratch
  targetPrefillState.ws.positionIdsH.withPinnedBuffer(buf => {
    let nextAccepted = 0;
    let nextScratch = originalAllocLen + finishCount;
    for (let i = 0; i < totalVerificationTokens; i++) {
      if (acceptedSet.has(i)) {
        buf.writeInt32LE(originalAllocLen + nextAccepted, i * I32);
        nextAccepted++;
      } else {
        buf.writeInt32LE(nextScratch, i * I32);
        nextScratch++;
      }
    }
  });

  captureManager.run(() => {
    targetPrefillState.ws.positionIds.memcpy(targetPrefillState.ws.positionIdsH, totalVerificationTokens * I32, MemcpyKind.HostToDevice);

    for (const layer of kvCacheLayers) {
      mlaKVCacheAppendOrig(layer.appendCkv, layer.appendKpe, layer.cacheIdx, layer.kvLoraRank, layer.qkRopeDim);
    }
  }, ['mtp-tree-append', totalVerificationTokens]);

  {
    // After capture block, in-place reorder accepted rows to front:
    // targetHiddenStates now has shape [totalVerificationTokens, hiddenDim]
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
  }

  // target can now be truncate to the accepted + replacement length
  seq0.truncate(originalAllocLen + finishCount);

  // mtp still needs to extend prefill
  const mtpExtendPrefill = ws.planPrefill(model, batchSize, [finishCount], cache, undefined,
    // this will start the extend prefill operation to the original alloc len
    cache.getPagedKV().sequences.map(s => originalAllocLen));
  mtpExtendPrefill.setInput([[...acceptedTokens, bestReplacement]]);

  captureManager.run(() => {
    using verfiedHiddenStates = hiddenStateStaging.slice(0, finishCount, 1);
    using mtpHs = model.forwardMtp!(mtpExtendPrefill, verfiedHiddenStates);
    // IS THIS WRONG? FIX?
    using newMtpHiddenStates = mtpHs.slice(finishCount - 1, finishCount, 1);
    mtpHiddenStates.memcpy(newMtpHiddenStates);
  }, ['mtp-replace', finishCount]);

  ws.glm.synchronize();

  const verify = performance.now();

  if (acceptedTokens.length) {
    console.warn(`MTP verify: accepted ${acceptedTokens.length} tokens: ${acceptedTokens.map(t => tokenizer.decode([t], { skip_special_tokens: false }))}, replacement: ${tokenizer.decode([bestReplacement], { skip_special_tokens: false })}, target: ${tokenizer.decode([targetToken], { skip_special_tokens: false })}`);
  }

  return [...acceptedTokens, bestReplacement];
}

function buildTargetMask(ws: WorkspaceBase, totalTreeNodes: number): { data: Tensor; indptr: Tensor } {
  const totalBits = totalTreeNodes * totalTreeNodes;
  const byteLen = Math.ceil(totalBits / 8);
  const data = ws.tensors.get(`mtp_target_mask_host_${totalTreeNodes}`) || ws.allocPinned([byteLen], "U8", `mtp_target_mask_host_${totalTreeNodes}`);

  data.withPinnedBuffer(data => {
    data.fill(0);
    for (let q = 0; q < totalTreeNodes; q++) {
      let cur = q;
      while (true) {
        const bit = q * totalTreeNodes + cur;
        data[bit >> 3] |= 1 << (bit & 7);
        if (cur === 0) break;
        cur = (cur - 1) >> 1;
      }
    }
  });

  const indptr = ws.tensors.get(`mtp_target_mask_indptr_host_${totalTreeNodes}`) || ws.allocPinned([2], "I32", `mtp_target_mask_indptr_host_${totalTreeNodes}`);
  indptr.withPinnedBuffer(indptr => {
    indptr.writeInt32LE(0, 0);
    indptr.writeInt32LE(byteLen, 4);
  });

  return { data, indptr };
}

function buildMTPMask(ws: WorkspaceBase, totalTreeNodes: number): { data: Tensor; indptr: Tensor } {
  const totalBits = totalTreeNodes * totalTreeNodes;
  const byteLen = Math.ceil(totalBits / 8);
  const data = ws.tensors.get(`mtp_draft_mask_host_${totalTreeNodes}`) || ws.allocPinned([byteLen], "U8", `mtp_draft_mask_host_${totalTreeNodes}`);

  data.withPinnedBuffer(data => {
    data.fill(0);
    for (let q = 0; q < totalTreeNodes; q++) {
      let cur = q;
      while (true) {
        const bit = q * totalTreeNodes + cur;
        data[bit >> 3] |= 1 << (bit & 7);
        if (cur < 2) break;
        cur = (cur - 2) >> 1;
      }
    }
  });

  const indptr = ws.tensors.get(`mtp_draft_mask_indptr_host_${totalTreeNodes}`) || ws.allocPinned([2], "I32", `mtp_draft_mask_indptr_host_${totalTreeNodes}`);
  indptr.withPinnedBuffer(indptr => {
    indptr.writeInt32LE(0, 0);
    indptr.writeInt32LE(byteLen, 4);
  });

  return { data, indptr };
}

function ensureTargetCustomMask(ws: WorkspaceBase, numTokens: number) {
  let mask = ws.tensors.get(`mtp_target_mask_${numTokens}`);
  let indptr = ws.tensors.get(`mtp_target_mask_indptr_${numTokens}`);
  if (!mask || !indptr) {
    const { data: maskData, indptr: maskIndptrData } = buildTargetMask(ws, numTokens);
    mask = ws.alloc(maskData.shape, "U8", `mtp_target_mask_${numTokens}`);
    mask.memcpy(maskData, maskData.bytes, MemcpyKind.HostToDevice);
    indptr = ws.alloc(maskIndptrData.shape, "I32", `mtp_target_mask_indptr_${numTokens}`);
    indptr.memcpy(maskIndptrData, maskIndptrData.bytes, MemcpyKind.HostToDevice);
  }

  return {
    mask,
    indptr,
    mode: MaskMode.CausalCustom,
  };
};

function ensureMTPCustomMask(ws: WorkspaceBase, numTokens: number) {
  let mask = ws.tensors.get(`mtp_draft_mask_${numTokens}`);
  let indptr = ws.tensors.get(`mtp_draft_mask_indptr_${numTokens}`);
  if (!mask || !indptr) {
    const { data: maskData, indptr: maskIndptrData } = buildMTPMask(ws, numTokens);
    mask = ws.alloc(maskData.shape, "U8", `mtp_draft_mask_${numTokens}`);
    mask.memcpy(maskData, maskData.bytes, MemcpyKind.HostToDevice);
    indptr = ws.alloc(maskIndptrData.shape, "I32", `mtp_draft_mask_indptr_${numTokens}`);
    indptr.memcpy(maskIndptrData, maskIndptrData.bytes, MemcpyKind.HostToDevice);
  }

  return {
    mask,
    indptr,
    mode: MaskMode.CausalCustom,
  };
};

