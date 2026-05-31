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
  ws: ExecutionWorkspace,
  gpuSampleResult: Tensor,
  nextn: number,
) {
  if (!model.forwardMtp) {
    throw new Error("mtpPrefill: model does not support MTP (forwardMtp not defined)");
  }

  const batchSize = state.batchSize;
  // topkIndices tracks the rotation token for the next layer.
  // Layer 0 uses the target model's sampled token; layers 1+ use the previous MTP layer's top-1.
  using topkIndices = new UsingHolder<Tensor>(undefined!);

  // rotatedIds ping-pongs between workspace allocations.
  // Source for layer 0 is state.input (the original prompt tokens).
  using rotatedIds = new UsingHolder<Tensor>(undefined!);

  for (let i = 0; i < nextn; i++) {
    // Rotate input IDs: shift each sequence left by 1, append previous prediction
    const source = rotatedIds.value ?? state.input!;
    rotatedIds.replace(source.rotateInputIds(ws.qoIndptrD, topkIndices.value || gpuSampleResult, batchSize));

    // Forward through one MTP layer with rotated input
    using hiddenStates = model.forwardMtp(state, targetHiddenStates, rotatedIds.value);

    // Sample top-1 from this layer's output for the next rotation
    using logits = state.computeLogits(hiddenStates, model);
    const topk = logits.topk(1, model.cfg.vocabSize);
    using _values = topk.values;
    topkIndices.replace(topk.indices); // [batchSize, 1] I32 — stride-compatible with rotateInputIds
  }
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
  gpuSampleResult: Tensor,
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

  using previousTiledHs = new UsingHolder<Tensor>(undefined!);
  using _tracker = ws.startTracking(new Set([targetHiddenStates]));

  // Iteration: single-sequence prefill with tree-shaped custom mask
  // 1 -> 3 -> 7 -> 15 for i = 0,1,2,3 (nextn=3)
  for (let i = 0; i < nextn; i++) {
    const numPrefillTokens = (1 << (i + 1)) - 1;

    let prefillState: ExecutionState;
    if (i === 0) {
      prefillState = ws.planPrefill(model, 1, [1], cache);
    }
    else {
      const customMask = ensureCustomMask(ws, numPrefillTokens, buildMtpTreeMask);
      prefillState = ws.planPrefill(model, 1, [numPrefillTokens], cache, customMask);
    }

    captureManager.run(() => {

      if (i === 0) {
        using gpuReshaped = gpuSampleResult.reshape([gpuSampleResult.numElements]);
        prefillState.setInput(gpuReshaped);
      }
      else {
        prefillState.setInput(ws.inputIdsBuf);
      }

      // results in i ^ 2 + 1 tiled hidden states
      if (i === 0) {
        const copy = ws.alloc(targetHiddenStates.shape, targetHiddenStates.type)
        previousTiledHs.replace(copy);
        copy.memcpy(targetHiddenStates);
      }
      else {
        previousTiledHs.replace(previousTiledHs.value.cat([previousTiledHs.value], 0));
        previousTiledHs.replace(previousTiledHs.value.cat([targetHiddenStates], 0));
      }

      using hiddenStates = model.forwardMtp!(prefillState, previousTiledHs.value);
      using logits = prefillState.computeLogits(hiddenStates, model, null);

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
    if (i === 1)
      ws.glm.synchronize();
  }

  validationSequences.memcpy(ws.inputIdsBuf, undefined, MemcpyKind.DeviceToDevice);

  return { validationSequences, nextn };
}

function buildMtpTreeMask(ws: WorkspaceBase, numPrefillTokens: number): { data: Tensor; indptr: Tensor } {
  const totalBits = numPrefillTokens * numPrefillTokens;
  const byteLen = Math.ceil(totalBits / 8);
  const data = ws.tensors.get(`mtp_tree_mask_host_${numPrefillTokens}`) || ws.allocPinned([byteLen], "U8", `mtp_tree_mask_host_${numPrefillTokens}`);

  data.withPinnedBuffer(data => {
    for (let q = 0; q < numPrefillTokens; q++) {
      let cur = q + 1;
      while (cur > 0) {
        const suffixPos = cur - 1;
        const bit = q * numPrefillTokens + suffixPos;
        data[bit >> 3] |= 1 << (bit & 7);
        cur = (cur - 1) >> 1;
      }
    }
  });

  const indptr = ws.tensors.get(`mtp_tree_mask_indptr_host_${numPrefillTokens}`) || ws.allocPinned([2], "I32", `mtp_tree_mask_indptr_host_${numPrefillTokens}`);
  indptr.withPinnedBuffer(indptr => {
    indptr.writeInt32LE(0, 0);
    indptr.writeInt32LE(byteLen, 4);
  });

  return { data, indptr };
}

export interface MtpVerifyResult {
  /** Number of draft tokens accepted (0..nextn). */
  numAccepted: number;
  /** The accepted draft token IDs from the winning path. */
  acceptedTokens: number[];
  /** Target model's argmax at the first mismatch position (or bonus token if all accepted).
   *  This is the token the target model would generate instead of the rejected draft. */
  replacementToken: number;
}

function buildTreeMask(ws: WorkspaceBase, totalTreeNodes: number): { data: Tensor; indptr: Tensor } {
  const totalBits = totalTreeNodes * totalTreeNodes;
  const byteLen = Math.ceil(totalBits / 8);
  const data = ws.tensors.get(`mtp_tree_mask_host_${totalTreeNodes}`) || ws.allocPinned([byteLen], "U8", `mtp_tree_mask_host_${totalTreeNodes}`);

  data.withPinnedBuffer(data => {
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
  model: ChatModel,
  ws: ExecutionWorkspace,
  cache: ChatCache,
  treeResult: MtpTreeResult,
  tokenizer?: any,
): MtpVerifyResult {
  const pagedKV = cache.getPagedKV();
  const nextn = treeResult.nextn;
  const totalPaths = 1 << nextn;
  const totalTreeNodes = 2 * totalPaths - 1;
  const pageSize = pagedKV.pageSize;

  if (pagedKV.sequences.length !== 1) {
    throw new Error(`mtpVerify: expected 1 sequence, got ${pagedKV.sequences.length}`);
  }

  const seq0 = pagedKV.sequences[0];
  const originalAllocLen = seq0.allocLen;
  const tokenIds = seq0.getTokenIds();

  using hostBuf = ws.allocPinned(treeResult.validationSequences.shape, treeResult.validationSequences.type);
  hostBuf.memcpy(treeResult.validationSequences, treeResult.validationSequences.bytes, MemcpyKind.DeviceToHost);
  ws.glm.synchronize();
  const treeTokens = Buffer.from(hostBuf.readPinnedBuffer());

  const allTokenIds: number[] = [];
  for (let i = 0; i < totalTreeNodes; i++) {
    allTokenIds.push(treeTokens.readInt32LE(i * 4));
  }

  pagedKV.positionIdsDirty = true;
  pagedKV.pagesDirtyHost = true;

  const customMask = ensureCustomMask(ws, totalTreeNodes, buildTreeMask);

  const state = ws.planPrefill(model, 1, [totalTreeNodes], cache, customMask);

  state.setInput([allTokenIds]);

  using hiddenStates = model.forward(state);
  using logits = state.computeLogits(hiddenStates, model, null);

  using argmaxResult = logits.argmax();
  using argmaxHost = ws.allocPinned(argmaxResult.shape, argmaxResult.type);
  argmaxHost.memcpy(argmaxResult, argmaxResult.bytes, MemcpyKind.DeviceToHost);

  ws.glm.synchronize();
  const argmaxBuf = Buffer.from(argmaxHost.readPinnedBuffer());

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
  const lastContentPageIdx = seq0.contentPages - 1;
  if (lastContentPageIdx >= 0) {
    const expected = tokenIds.length - lastContentPageIdx * pageSize;
    if (seq0.pages[lastContentPageIdx].tokenIds.length > expected) {
      seq0.pages[lastContentPageIdx].tokenIds.length = Math.max(0, expected);
    }
  }

  pagedKV.positionIdsDirty = true;
  pagedKV.pagesDirtyHost = true;
  pagedKV.pagesDirtyDevice = true;

  return { numAccepted: bestAccepted, acceptedTokens, replacementToken: bestReplacement };
}


