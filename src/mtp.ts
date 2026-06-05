import { CaptureManager } from "./capture-manager";
import { type ChatCache, type ChatModel } from "./chat_model";
import { MaskMode } from "./device_ops";
import { ExecutionState, ExecutionWorkspace } from "./execution-workspace";
import { I32 } from "./glm_ops";
import { MemcpyKind, Tensor } from "./tensor";
import { WorkspaceBase } from "./workspace";

function getPositionIdsMask(ws: ExecutionWorkspace, originalAllocLen: number, totalTreeNodes: number) {
  const positionIds = ws.ensureAlloc(ws.positionIds.shape, ws.positionIds.type, `mtp_tree_position_ids-${totalTreeNodes}`);
  function treeDepth(n: number): number {
    let depth = 0;
    while (n > 0) {
      n = (n - 1) >> 1;
      depth++;
    }
    return depth;
  }

  const positionIdsH = ws.ensureAllocPinned(ws.positionIds.shape, ws.positionIds.type, `mtp_tree_position_ids_host-${totalTreeNodes}`);
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
 * @param targetHiddenStates - The target model's hidden states [1, hidden] BF16
 * @param ws - Execution workspace
 * @param gpuSampleResult - [1] I32 GPU tensor: target model's sampled token
 * @param nextn - Number of MTP layers (tree depth)
 * @param cache - Chat cache (paged KV cache)
 * @returns MtpTreeResult with validation sequences tensor and nextn
 */
export function mtpTreeDecode(
  captureManager: CaptureManager,
  model: ChatModel,
  targetHiddenStates: Tensor,
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
  const totalTreeNodes = 2 * totalPaths - 1;
  const seq0 = pagedKV.sequences[0];
  const originalAllocLen = seq0.allocLen;

  using _tracker = ws.startTracking(new Set([targetHiddenStates]));

  const hiddenDim = targetHiddenStates.shape[1];
  const rowBytes = hiddenDim * 2; // BF16 = 2 bytes per element

  // Tiled hidden states for the binary tree. Populated incrementally:
  //   tiledHs[0]     = targetHS              (root, written once at i=0)
  //   tiledHs[2p+1] = hiddenStates[p]       (child 1, written at end of iter i)
  //   tiledHs[2p+2] = hiddenStates[p]       (child 2, written at end of iter i)
  // Each iteration's hiddenStates output is copied directly to tiledHs for the
  // next iteration, eliminating the need for a separate prevMtpOutput buffer.
  const tiledHs = ws.ensureAlloc([totalTreeNodes, hiddenDim], "BF16", `mtp-tree-hs-${totalTreeNodes}`, undefined, 0);

  const hostBuf = ws.ensureAllocPinned([totalTreeNodes], "I32", `mtp_verify_host_buf_${totalTreeNodes}`);

  const customMask = ensureCustomMask(ws, totalTreeNodes);
  const prefillState = ws.planPrefill(model, 1, [totalTreeNodes], cache, {
    ...customMask,
    positionIds: getPositionIdsMask(ws, originalAllocLen, totalTreeNodes),
  });
  ws.ensureInputCleared();
  prefillState.setInput([[targetToken]]);

  const start = performance.now();
  const mlaKVCacheAppendOrig = prefillState.mlaKvCacheAppend.bind(prefillState);

  const kvCacheLayers = captureManager.run(() => {
    const kvCacheLayers: { appendCkv: Tensor, appendKpe: Tensor, cacheIdx: number, kvLoraRank: number, qkRopeDim: number }[] = [];

    // Iteration: single-sequence prefill with tree-shaped custom mask
    // 1 -> 3 -> 7 -> 15 for i = 0,1,2,3 (nextn=3)
    for (let i = 0; i < nextn; i++) {
      const numPrefillTokens = (1 << (i + 1)) - 1;

      if (i === 0) {
        tiledHs.memcpy(targetHiddenStates);
      }

      if (i === nextn - 1) {
        prefillState.mlaKvCacheAppend = (appendCkv, appendKpe, cacheIdx, kvLoraRank, qkRopeDim) => {
          appendCkv.removeTracking();
          appendKpe.removeTracking();
          kvCacheLayers.push({ appendCkv, appendKpe, cacheIdx, kvLoraRank, qkRopeDim });
          mlaKVCacheAppendOrig(appendCkv, appendKpe, cacheIdx, kvLoraRank, qkRopeDim);
        };
      }

      using hiddenStates = model.forwardMtp!(prefillState, tiledHs);

      // Copy this iteration's hiddenStates directly to tiledHs for next iteration.
      // Each parent p's output becomes both children 2p+1 and 2p+2.
      using tiledHsStream = i < nextn - 1
        ? ws.glm.withStream(() => {
          using oddStream = ws.glm.withStream(() => {
            // Odd rows (1,3,5,…): child 1 of each parent
            tiledHs.memcpy2d(
              1 * rowBytes, 2 * rowBytes,
              hiddenStates, 0, rowBytes,
              rowBytes, numPrefillTokens,
              MemcpyKind.DeviceToDevice,
            );
          });
          // Even rows (2,4,6,…): child 2 of each parent (sibling shares same parent)
          tiledHs.memcpy2d(
            2 * rowBytes, 2 * rowBytes,
            hiddenStates, 0, rowBytes,
            rowBytes, numPrefillTokens,
            MemcpyKind.DeviceToDevice,
          );

          oddStream.streamWaitEvent();
        })
        : undefined;

      using logits = prefillState.computeLogits(hiddenStates, model, true);
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

    using stream = ws.glm.withStream(() => {
      hostBuf.memcpy(ws.inputIdsBuf, hostBuf.bytes, MemcpyKind.DeviceToHost);
    });

    const hiddenStates = model.forwardModel(prefillState);
    using logits = prefillState.computeLogits(hiddenStates, model, true);
    using argmaxResult = logits.argmax();
    const argmaxHost = ws.ensureAllocPinned(argmaxResult.shape, argmaxResult.type, `mtp_verify_argmax_host_${totalTreeNodes}`);
    argmaxHost.memcpy(argmaxResult, argmaxResult.bytes, MemcpyKind.DeviceToHost);
    stream.streamWaitEvent();

    return kvCacheLayers;
  }, ['mtp-tree']);

  ws.glm.synchronize();

  const draft = performance.now();

  const argmaxHost = ws.tensors.get(`mtp_verify_argmax_host_${totalTreeNodes}`)!;
  const argmaxBuf = argmaxHost.readPinnedBuffer();
  const treeTokens = hostBuf.readPinnedBuffer();

  let bestPath = 0;
  let bestAccepted = -1;
  let bestReplacement = -1;

  for (let path = 0; path < totalPaths; path++) {
    let accepted = 0;
    let nodeIdx = 0;
    for (let layer = 0; layer < nextn; layer++) {
      const childIdx = nodeIdx * 2 + 1 + ((path >> (nextn - 1 - layer)) & 1);
      const draftToken = treeTokens.readInt32LE(childIdx * 4);
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
    acceptedTokens.push(treeTokens.readInt32LE(childIdx * 4));
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
  prefillState.ws.positionIdsH.withPinnedBuffer(buf => {
    let nextAccepted = 0;
    let nextScratch = originalAllocLen + finishCount;
    for (let i = 0; i < totalTreeNodes; i++) {
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
    prefillState.ws.positionIds.memcpy(prefillState.ws.positionIdsH, totalTreeNodes * I32, MemcpyKind.HostToDevice);

    for (const layer of kvCacheLayers) {
      mlaKVCacheAppendOrig(layer.appendCkv, layer.appendKpe, layer.cacheIdx, layer.kvLoraRank, layer.qkRopeDim);
    }
  }, ['mtp-tree-append']);

  seq0.truncate(originalAllocLen + finishCount);

  // Decode the replacement token: this correctly populates its KV cache entry
  // (with the right input token and causal attention) and produces the next token.
  const replaceState = ws.planDecode(model, 1, cache);
  replaceState.setInput([[bestReplacement]]);

  captureManager.run(() => {
    ws.positionStep(replaceState, model);
    using replaceHidden = model.forwardModel(replaceState);
    using logits = replaceState.computeLogits(replaceHidden, model, true);
    using argmaxResult = logits.argmax();
    using rotatedInputIds = replaceState.input!.rotateInputIds(ws.qoIndptrD, argmaxResult, replaceState.batchSize);
    replaceState.setInput(rotatedInputIds);
    using _mtpHs = model.forwardMtp!(replaceState, replaceHidden);
    targetHiddenStates.memcpy(replaceHidden, undefined, MemcpyKind.DeviceToDevice);
    hostBuf.memcpy(argmaxResult, I32, MemcpyKind.DeviceToHost);
  }, ['mtp-replace']);

  ws.glm.synchronize();

  const newTargetToken = hostBuf.readPinnedBuffer().readInt32LE();

  const verify = performance.now();

  if (acceptedTokens.length) {
    console.warn(`MTP verify: accepted ${acceptedTokens.length} tokens: ${acceptedTokens.map(t => tokenizer.decode([t], { skip_special_tokens: false }))}, replacement: ${tokenizer.decode([bestReplacement], { skip_special_tokens: false })}, target: ${tokenizer.decode([targetToken], { skip_special_tokens: false })}`);
  }

  return [...acceptedTokens, bestReplacement, newTargetToken];
}

function buildTreeMask(ws: WorkspaceBase, totalTreeNodes: number): { data: Tensor; indptr: Tensor } {
  const totalBits = totalTreeNodes * totalTreeNodes;
  const byteLen = Math.ceil(totalBits / 8);
  const data = ws.tensors.get(`mtp_tree_mask_host_${totalTreeNodes}`) || ws.allocPinned([byteLen], "U8", `mtp_tree_mask_host_${totalTreeNodes}`);

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

  const indptr = ws.tensors.get(`mtp_tree_mask_indptr_host_${totalTreeNodes}`) || ws.allocPinned([2], "I32", `mtp_tree_mask_indptr_host_${totalTreeNodes}`);
  indptr.withPinnedBuffer(indptr => {
    indptr.writeInt32LE(0, 0);
    indptr.writeInt32LE(byteLen, 4);
  });

  return { data, indptr };
}


function ensureCustomMask(ws: WorkspaceBase, numTokens: number) {
  let mask = ws.tensors.get(`mtp_tree_mask_${numTokens}`);
  let indptr = ws.tensors.get(`mtp_tree_mask_indptr_${numTokens}`);
  if (!mask || !indptr) {
    const { data: maskData, indptr: maskIndptrData } = buildTreeMask(ws, numTokens);
    mask = ws.alloc(maskData.shape, "U8", `mtp_tree_mask_${numTokens}`);
    mask.memcpy(maskData, maskData.bytes, MemcpyKind.HostToDevice);
    indptr = ws.alloc(maskIndptrData.shape, "I32", `mtp_tree_mask_indptr_${numTokens}`);
    indptr.memcpy(maskIndptrData, maskIndptrData.bytes, MemcpyKind.HostToDevice);
  }

  return {
    mask,
    indptr,
    mode: MaskMode.CausalCustom,
  };
};

