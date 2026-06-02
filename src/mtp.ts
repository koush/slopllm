import { CaptureManager } from "./capture-manager";
import { type ChatCache, type ChatModel } from "./chat_model";
import { MaskMode } from "./device_ops";
import { ExecutionState, ExecutionWorkspace } from "./execution-workspace";
import { I32 } from "./glm_ops";
import { MemcpyKind, Tensor } from "./tensor";
import { UsingHolder } from "./using-holder";
import { WorkspaceBase } from "./workspace";

/**
 * MTP (Multi-Token Prediction) prefill rotation.
 *
 * During prefill, each MTP layer needs to see a rotated version of the input:
 *   Target:     [t0, t1, ..., t{S-1}]           → sample T
 *   MTP layer 0: [t1, t2, ..., t{S-1}, T]        → sample P0
 *   MTP layer 1: [t2, t3, ..., t{S-1}, T, P0]    → sample P1
 *   MTP layer 2: [t3, t4, ..., t{S-1}, T, P0, P1] → sample P2
 *
 * Each layer shifts the input left by 1 and appends the previous layer's prediction.
 * The rotation token comes from:
 *   - Layer 0: the target model's sampled token (gpuSampleResult)
 *   - Layer 1+: the previous MTP layer's top-1 prediction (topk.indices)
 *
 * Returns the top-1 prediction indices for each MTP layer (shape [batchSize, 1] I32 each).
 * The caller is responsible for disposing the returned tensors when done.
 *
 * Prerequisites:
 *   - ws.qoIndptrD must be populated on GPU (done by planPrefill)
 *   - state must be in prefill mode (isDecode = false)
 *   - gpuSampleResult must be [batchSize] I32 on GPU (target model's sampled token)
 *   - model.forwardMtp must exist (MTP enabled)
 *
 * Example usage in prefill path:
 *
 *   // After target model forward + sampling:
 *   targetHiddenStates.replace(model.forward(state));
 *   using firstTokens = state.computeLogits(targetHiddenStates.value, model);
 *   doSample(firstTokens);
 *
 *   // MTP prefill with rotation
 *   const mtpPredictions = mtpPrefill(
 *     state, model, targetHiddenStates.value, ws, gpuSampleResult!, nextn
 *   );
 *
 *   readSample();
 *
 *   // Use mtpPredictions[i] for layer i's top-1 prediction...
 *   // Dispose when done:
 *   for (const pred of mtpPredictions) pred[Symbol.dispose]();
 */
export function mtpPrefill(
  state: ExecutionState,
  model: ChatModel,
  targetHiddenStates: Tensor,
  gpuSampleResult: Tensor,
) {
  if (!model.forwardMtp) {
    throw new Error("mtpPrefill: model does not support MTP (forwardMtp not defined)");
  }

  // Forward through one MTP layer with rotated input
  using hiddenStates = model.forwardMtp(state, targetHiddenStates, state.input);

  // Sample top-1 from this layer's output for the next rotation
  using logits = state.computeLogits(hiddenStates, model);
  const topk = logits.topk(1, model.cfg.vocabSize);
  using _values = topk.values;
  using _indices = topk.indices;
}

/**
 * Result of tree-structured MTP draft generation.
 *
 * The validation sequences tensor has shape [2 * (1 << nextn) - 1] I32,
 * laid out in breadth-first order:
 *
 * For nextn=3, the 15 tokens are:
 *   [root, D0_top1, D0_top2, D1_top1(top1), D1_top2(top1),
 *    D1_top1(top2), D1_top2(top2), D2_top1(@3), D2_top2(@3),
 *    D2_top1(@4), D2_top2(@4), D2_top1(@5), D2_top2(@5),
 *    D2_top1(@6), D2_top2(@6)]
 *
 * Token at index i has parent at index floor((i - 1) / 2).
 * This flat layout enables single-sequence verification with a custom
 * tree-shaped attention mask.
 */
export interface MtpTreeResult {
  validationSequences: Tensor;
  nextn: number;
}

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
 *   - The target model must have just decoded (planDecode + decodeStep + forward
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
  currentToken: number,
  nextn: number,
  cache: ChatCache,
): MtpTreeResult {
  if (!model.forwardMtp) {
    throw new Error("mtpTreeDecode: model does not support MTP (forwardMtp not defined)");
  }

  const pagedKV = cache.getPagedKV();
  const totalPaths = 1 << nextn;
  const totalTreeNodes = 2 * totalPaths - 1;
  const validationSequences = ws.ensureAlloc([totalTreeNodes], "I32", `${totalTreeNodes}-validation-sequences`);
  const seq0 = pagedKV.sequences[0];
  const originalAllocLen = seq0.allocLen;

  using _tracker = ws.startTracking(new Set([targetHiddenStates]));

  const hiddenDim = targetHiddenStates.shape[1];
  const rowBytes = hiddenDim * 2; // BF16 = 2 bytes per element
  // Buffer to save MTP output from previous iteration for chained hidden states.
  // Each node p > 0 is conditioned on its parent's MTP output (matching the
  // single-layer EAGLE worker's decode path where hidden_states are chained).
  const prevMtpOutput = ws.ensureAlloc([(1 << nextn) - 1, hiddenDim], "BF16", `mtp-tree-prev-mtp-${totalTreeNodes}`);



  // Iteration: single-sequence prefill with tree-shaped custom mask
  // 1 -> 3 -> 7 -> 15 for i = 0,1,2,3 (nextn=3)
  for (let i = 0; i < nextn; i++) {
    const numPrefillTokens = (1 << (i + 1)) - 1;

    let prefillState: ExecutionState;
    if (i === 0) {
      prefillState = ws.planPrefill(model, 1, [1], cache);
      prefillState.setInput([[currentToken]]);
    }
    else {
      const customMask = ensureCustomMask(ws, numPrefillTokens, buildTreeMask);
      prefillState = ws.planPrefill(model, 1, [numPrefillTokens], cache, {
        ...customMask,
        positionIds: getPositionIdsMask(ws, originalAllocLen, totalTreeNodes),
      });
    }

    captureManager.run(() => {
      prefillState.setInput(ws.inputIdsBuf);

      // Build tiled hidden states: node 0 gets targetHS, node p>0 gets its
      // parent's MTP output from the previous iteration (chained, matching the
      // single-layer EAGLE worker's decode path).
      //   tiledHs[0]         = targetHS            (root)
      //   tiledHs[2p+1]     = prevMtpOutput[p]    (child 1 of node p)
      //   tiledHs[2p+2]     = prevMtpOutput[p]    (child 2 of node p)
      const tiledHs = ws.ensureAlloc([numPrefillTokens, hiddenDim], "BF16", `mtp-tree-hs-${numPrefillTokens}`);

      if (i === 0) {
        tiledHs.memcpy(targetHiddenStates);
      } else {
        // Row 0: target model hidden state (root node)
        tiledHs.memcpy2d(
          0, rowBytes,
          targetHiddenStates, 0, rowBytes,
          rowBytes, 1,
          MemcpyKind.DeviceToDevice,
        );
        // Odd rows (1,3,5,…): one copy of each parent's MTP output
        const numParents = (numPrefillTokens - 1) >> 1;
        tiledHs.memcpy2d(
          1 * rowBytes, 2 * rowBytes,
          prevMtpOutput, 0, rowBytes,
          rowBytes, numParents,
          MemcpyKind.DeviceToDevice,
        );
        // Even rows (2,4,6,…): duplicate from odd rows (sibling shares same parent)
        tiledHs.memcpy2d(
          2 * rowBytes, 2 * rowBytes,
          tiledHs, 1 * rowBytes, 2 * rowBytes,
          rowBytes, numParents,
          MemcpyKind.DeviceToDevice,
        );
      }

      using hiddenStates = model.forwardMtp!(prefillState, tiledHs);
      using logits = prefillState.computeLogits(hiddenStates, model, null);

      // Save this iteration's MTP output for the next iteration's chaining
      prevMtpOutput.memcpy2d(
        0, rowBytes,
        hiddenStates, 0, rowBytes,
        rowBytes, numPrefillTokens,
        MemcpyKind.DeviceToDevice,
      );

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
    }, ['mtp-tree', `layer${i}`]);

    seq0.truncate(originalAllocLen);
    pagedKV.positionIdsDirty = true;
    pagedKV.pagesDirtyHost = true;
    pagedKV.pagesDirtyDevice = true;

    // why is this necessary? without it there's an illegal memory access.
    // actually without the sync all the host pointers are written in a tight loop,
    // and the memcpy may happen before the data has been read and send to gpu.
    ws.glm.synchronize();
  }

  validationSequences.memcpy(ws.inputIdsBuf, undefined, MemcpyKind.DeviceToDevice);

  return { validationSequences, nextn };
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


function ensureCustomMask(ws: WorkspaceBase, numTokens: number, buildMask: (ws: WorkspaceBase, numTokens: number) => { data: Tensor; indptr: Tensor }) {
  let mask = ws.tensors.get(`mtp_tree_mask_${numTokens}-${buildMask.name}`);
  let indptr = ws.tensors.get(`mtp_tree_mask_indptr_${numTokens}-${buildMask.name}`);
  if (!mask || !indptr) {
    const { data: maskData, indptr: maskIndptrData } = buildMask(ws, numTokens);
    mask = ws.alloc(maskData.shape, "U8", `mtp_tree_mask_${numTokens}-${buildMask.name}`);
    mask.memcpy(maskData, maskData.bytes, MemcpyKind.HostToDevice);
    indptr = ws.alloc(maskIndptrData.shape, "I32", `mtp_tree_mask_indptr_${numTokens}-${buildMask.name}`);
    indptr.memcpy(maskIndptrData, maskIndptrData.bytes, MemcpyKind.HostToDevice);
  }

  return {
    mask,
    indptr,
    mode: MaskMode.CausalCustom,
  };
};

/**
 * Verify MTP draft tokens against the target model using a single-sequence
 * tree-shaped custom mask.
 *
 * Appends all tree tokens (root + draft) to seq0, runs a single prefill pass
 * with a custom attention mask that enforces the tree structure (each node
 * attends to prefix + itself + ancestors), and compares argmax logits against
 * the draft tokens. After verification, restores the KV cache.
 *
 * The root token (index 0 in the tree) is included in the prefill so the
 * target model produces a prediction at the root position, which is compared
 * against the root's children. The root is already in the KV cache at
 * position originalAllocLen-1; including it again at originalAllocLen
 * duplicates it but matches the convention used in batch verification.
 *
 * Prerequisites:
 *   - mtpTreeDecode has been called, treeResult.validationSequences is populated
 *   - cache.getPagedKV().sequences has exactly 1 sequence (seq 0)
 *   - model.forwardMtp must exist (MTP enabled)
 *
 * @param model - The chat model
 * @param ws - Execution workspace
 * @param cache - Chat cache (paged KV cache)
 * @param treeResult - MtpTreeResult from mtpTreeDecode
 * @param tokenizer - Optional tokenizer for debug logging
 * @returns MtpVerifyResult with numAccepted, acceptedTokens, and replacementToken
 */
export function mtpVerify(
  captureManager: CaptureManager,
  model: ChatModel,
  targetHiddenStates: UsingHolder<Tensor>,
  ws: ExecutionWorkspace,
  cache: ChatCache,
  treeResult: MtpTreeResult,
  currentToken: number,
  tokenizer?: any,
) {
  const pagedKV = cache.getPagedKV();
  const nextn = treeResult.nextn;
  const totalPaths = 1 << nextn;
  const totalTreeNodes = 2 * totalPaths - 1;
  
  if (pagedKV.sequences.length !== 1) {
    throw new Error(`mtpVerify: expected 1 sequence, got ${pagedKV.sequences.length}`);
  }

  using _tracker = ws.startTracking(new Set([treeResult.validationSequences, targetHiddenStates.value]));

  const seq0 = pagedKV.sequences[0];
  const originalAllocLen = seq0.allocLen;

  const hostBuf = ws.ensureAllocPinned(treeResult.validationSequences.shape, treeResult.validationSequences.type, `mtp_verify_host_buf_${totalTreeNodes}`);

  pagedKV.positionIdsDirty = true;
  pagedKV.pagesDirtyHost = true;
  pagedKV.pagesDirtyDevice = true;

  const customMask = ensureCustomMask(ws, totalTreeNodes, buildTreeMask);

  const state = ws.planPrefill(model, 1, [totalTreeNodes], cache, {
    ...customMask,
    positionIds: getPositionIdsMask(ws, originalAllocLen, totalTreeNodes),
  });

  captureManager.run(() => {
    using stream = ws.glm.withStream(() => {
      hostBuf.memcpy(treeResult.validationSequences, treeResult.validationSequences.bytes, MemcpyKind.DeviceToHost);
    });

    state.setInput(treeResult.validationSequences);

    const hiddenStates = model.forwardModel(state);
    using logits = state.computeLogits(hiddenStates, model, null);
    using argmaxResult = logits.argmax();

    console.log(argmaxResult.readInt32LEArray().map(v => tokenizer.decode([v], { skip_special_tokens: false })));

    const argmaxHost = ws.ensureAllocPinned(argmaxResult.shape, argmaxResult.type, `mtp_verify_argmax_host_${totalTreeNodes}`);
    argmaxHost.memcpy(argmaxResult, argmaxResult.bytes, MemcpyKind.DeviceToHost);

    stream.streamWaitEvent();
  }, ['mtp-verify']);

  ws.glm.synchronize();
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
    const replacement = argmaxBuf.readInt32LE(nodeIdx * 4);
    if (accepted > bestAccepted) {
      bestAccepted = accepted;
      bestPath = path;
      bestReplacement = replacement;
    }
  }

  const acceptedTokens: number[] = [];
  let nodeIdx = 0;
  for (let layer = 0; layer < bestAccepted; layer++) {
    const childIdx = nodeIdx * 2 + 1 + ((bestPath >> (nextn - 1 - layer)) & 1);
    acceptedTokens.push(treeTokens.readInt32LE(childIdx * 4));
    nodeIdx = childIdx;
  }

  seq0.truncate(originalAllocLen);
  pagedKV.positionIdsDirty = true;
  pagedKV.pagesDirtyHost = true;
  pagedKV.pagesDirtyDevice = true;

  const finishTokens = [currentToken, ...acceptedTokens, bestReplacement];
  const finishState = ws.planPrefill(model, 1, [finishTokens.length], cache);
  finishState.setInput([finishTokens]);
  captureManager.run(() => {
    targetHiddenStates.replace(model.forwardModel(finishState));
    using logits = finishState.computeLogits(targetHiddenStates.value, model);
    using logits2 = finishState.computeLogits(targetHiddenStates.value, model, null);


    using argmaxResult = logits.argmax();
    console.log(argmaxResult.readInt32LEArray().map(v => tokenizer.decode([v], { skip_special_tokens: false })));

    using argmaxResult2 = logits2.argmax();
    console.log(argmaxResult2.readInt32LEArray().map(v => tokenizer.decode([v], { skip_special_tokens: false })));

    const argmaxHost = ws.tensors.get(`mtp_verify_argmax_host_${totalTreeNodes}`)!;
    argmaxHost.memcpy(argmaxResult, argmaxResult.bytes, MemcpyKind.DeviceToHost);

  }, ['mtp-verify-finish']);

  ws.glm.synchronize();

  const targetToken = argmaxBuf.readInt32LE(0);

  finishState.setInput([[targetToken]]);

  const verifiedTokens = finishTokens.slice();
  // remove the input token.
  verifiedTokens.shift();

  return [...verifiedTokens, targetToken];
}


