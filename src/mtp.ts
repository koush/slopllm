import { CaptureManager } from "./capture-manager";
import { type ChatCache, type ChatModel } from "./chat_model";
import { ExecutionState, ExecutionWorkspace } from "./execution-workspace";
import { I32 } from "./glm_ops";
import { MemcpyKind, Tensor } from "./tensor";
import { UsingHolder } from "./using-holder";

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
 * The validation sequences tensor has shape [(1 << nextn) * (nextn + 1)] I32,
 * laid out as (1 << nextn) rows of (nextn + 1) columns in row-major order.
 *
 * Column 0 is the root token (the target model's sampled token, broadcast to all rows).
 * Column i+1 contains MTP layer i's predictions (top-1 and top-2 per batch element,
 * in natural topk order: [top1_seq0, top2_seq0, top1_seq1, top2_seq1, ...]).
 *
 * For nextn=3, the 8 rows are:
 *   [root, D0_top1, D1_top1_of_top1, D2_top1_of_top1_of_top1]
 *   [root, D0_top1, D1_top1_of_top1, D2_top2_of_top1_of_top1]
 *   [root, D0_top1, D1_top2_of_top1, D2_top1_of_top2_of_top1]
 *   [root, D0_top1, D1_top2_of_top1, D2_top2_of_top2_of_top1]
 *   [root, D0_top2, D1_top1_of_top2, D2_top1_of_top1_of_top2]
 *   [root, D0_top2, D1_top1_of_top2, D2_top2_of_top1_of_top2]
 *   [root, D0_top2, D1_top2_of_top2, D2_top1_of_top2_of_top2]
 *   [root, D0_top2, D1_top2_of_top2, D2_top2_of_top2_of_top2]
 */
export interface MtpTreeResult {
  validationSequences: Tensor;
  nextn: number;
}

/**
 * Tree-structured MTP decode: produce draft predictions using topk=2 batch expansion.
 *
 * At each MTP layer, top-2 predictions are sampled, and the batch is doubled by
 * forking KV cache sequences. This creates a binary tree of depth `nextn` with
 * 2^nextn candidate paths. After generation, forked sequences are cleaned up and
 * the KV cache is rewound.
 *
 * Iteration 0 processes at the target model's decode position (S), reusing its
 * slot and decode plan. This ensures the MTP KV entry is written at position S
 * (filling the gap that would otherwise exist) and that the MTP layer sees the
 * correct RoPE position. Iterations 1+ advance the position normally via
 * planDecode + decodeStep.
 *
 * Prerequisites:
 *   - The target model must have just decoded (planDecode + decodeStep + forward
 *     already called for the current position)
 *   - state must be the target model's decode state (batchSize = 1)
 *   - gpuSampleResult must be [1] I32 GPU tensor: target model's sampled token
 *   - model.forwardMtp must exist (MTP enabled)
 *   - The paged KV cache must have enough pages for nextn-1 additional tokens
 *     (iteration 0 reuses the target model's slot) and enough sequences for
 *     (1 << (nextn-1)) forked sequences
 *   - cache.getPagedKV().maxBatch must be >= (1 << (nextn - 1))
 *
 * @param state - The target model's decode ExecutionState
 * @param model - The chat model (must support forwardMtp)
 * @param targetHiddenStates - The target model's hidden states [1, hidden] BF16
 * @param ws - Execution workspace
 * @param gpuSampleResult - [1] I32 GPU tensor: target model's sampled token
 * @param nextn - Number of MTP layers (tree depth)
 * @param cache - Chat cache (paged KV cache)
 * @returns MtpTreeResult with validation sequences tensor and nextn
 */
export function mtpTreeDecode(
  state: ExecutionState,
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
  const maxMtpBatch = 1 << (nextn - 1);
  if (pagedKV.maxBatch < maxMtpBatch) {
    throw new Error(
      `mtpTreeDecode: maxBatch (${pagedKV.maxBatch}) too small for nextn=${nextn}, need >= ${maxMtpBatch}`,
    );
  }

  const totalPaths = 1 << nextn;
  const rowLen = nextn + 1;
  const validationSequences = ws.alloc([totalPaths * rowLen], "I32");

  for (let j = 0; j < totalPaths; j++) {
    validationSequences.memcpy2d(
      j * rowLen * 4, rowLen * 4,
      gpuSampleResult, 0, 4,
      4, 1,
      MemcpyKind.DeviceToDevice,
    );
  }


  let nextInput: Tensor = gpuSampleResult;
  state.setInput(nextInput);

  let total = 1;
  using doubledTargetHs = new UsingHolder<Tensor>(undefined!);

  for (let i = 0; i < nextn; i++) {
    const batchSize = 1 << i;

    if (total > 1) {
      for (let j = 0; j < total / 2; j++) {
        pagedKV.copySequence(j + total / 2, j);
      }
    }

    if (batchSize > 1) {
      const prev: Tensor = doubledTargetHs.value ?? targetHiddenStates;
      doubledTargetHs.replace(prev.cat([prev], 0));
    }
    const hs = batchSize === 1 ? targetHiddenStates : doubledTargetHs.value!;

    let mtpState: ExecutionState;
    if (i === 0) {
      // Iteration 0: process at the target model's decode position (S).
      // The target model already allocated the slot at S and set up the decode
      // plan. Skipping planDecode/decodeStep avoids advancing past position S,
      // which would leave a gap in the MTP KV cache (no entry at S) and cause
      // the MTP to process at S+1 instead of S.
      mtpState = state;
    } else {
      pagedKV.positionIdsDirty = true;
      pagedKV.pagesDirtyHost = true;
      mtpState = ws.planDecode(model, batchSize, cache);
      mtpState.setInput(nextInput.reshape([nextInput.numElements]));
      ws.decodeStep(mtpState, model);
    }

    using hiddenStates = model.forwardMtp!(mtpState, hs);
    using logits = mtpState.computeLogits(hiddenStates, model);

    const topk = logits.topk(2, model.cfg.vocabSize);
    using _values = topk.values;
    using indices = topk.indices;

    const batch = total * 2;
    const half = batch / 2;

    const fanout = totalPaths >>> (i + 1);
    for (let j = 0; j < batch; j++) {
      for (let k = 0; k < fanout; k++) {
        validationSequences.memcpy2d(
          ((j * fanout + k) * rowLen + i + 1) * 4, rowLen * 4,
          indices, j * 4, 4,
          4, 1,
          MemcpyKind.DeviceToDevice,
        );
      }
    }

    if (i < nextn - 1) {
      const reordered = ws.alloc(topk.indices.shape, topk.indices.type);
      reordered.memcpy2d(0, 4, topk.indices, 0, 8, 4, half, MemcpyKind.DeviceToDevice);
      reordered.memcpy2d(half * 4, 4, topk.indices, 4, 8, 4, half, MemcpyKind.DeviceToDevice);
      nextInput = reordered;
    }

    total *= 2;
  }

  for (let i = 0; i < total / 2 - 1; i++) {
    const seq = pagedKV.sequences.pop();
    seq!.clear();
  }

  // Iteration 0 does not advance the position (no planDecode/decodeStep),
  // so only nextn-1 positions were allocated beyond the target model's slot.
  const mtpAdvance = nextn - 1;
  const seq0 = pagedKV.sequences[0];
  seq0.truncate(seq0.allocLen - mtpAdvance);
  pagedKV.positionIdsDirty = true;
  pagedKV.pagesDirtyHost = true;
  pagedKV.pagesDirtyDevice = true;
  ws.decodeStep(state, model, -mtpAdvance);

  return { validationSequences, nextn };
}

/**
 * Initiate async GPU→Host copy of tree validation sequences.
 *
 * The caller MUST synchronize the stream before reading the returned buffer.
 */
export function mtpTreeReadDrafts(
  result: MtpTreeResult,
  ws: ExecutionWorkspace,
): Tensor {
  const host = ws.allocPinned(result.validationSequences.shape, result.validationSequences.type);
  host.memcpy(result.validationSequences, result.validationSequences.bytes, MemcpyKind.DeviceToHost);
  return host;
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

/**
 * Verify MTP draft tokens against the target model via naive batch prefill.
 *
 * Creates 2^nextn sequences by forking the target model's KV cache, prefills
 * each with [currentToken, D0, D1, ..., D{nextn-1}] from the validation
 * sequences, runs the target model forward, and compares argmax logits against
 * the draft tokens. After verification, restores the KV cache to its
 * pre-verification state.
 *
 * Prerequisites:
 *   - mtpTreeDecode + mtpTreeReadDrafts have been called, hostBuf is synchronized
 *   - cache.getPagedKV().sequences has exactly 1 sequence (seq 0)
 *   - cache.getPagedKV().maxBatch >= (1 << nextn)
 *   - model.forwardMtp must exist (MTP enabled)
 *
 * @param model - The chat model
 * @param ws - Execution workspace
 * @param cache - Chat cache (paged KV cache)
 * @param treeResult - MtpTreeResult from mtpTreeDecode
 * @param validationSequences - Pinned host buffer from mtpTreeReadDrafts (synchronized).
 *   Consumed (disposed) by this function — the data is copied before the forward pass
 *   to avoid corruption from workspace tensor recycling.
 * @param tokenizer - Optional tokenizer for debug logging
 * @returns MtpVerifyResult with numAccepted, acceptedTokens, and replacementToken
 */
export function mtpVerify(
  model: ChatModel,
  ws: ExecutionWorkspace,
  cache: ChatCache,
  treeResult: MtpTreeResult,
  validationSequences: Tensor,
  tokenizer?: any,
): MtpVerifyResult {
  const pagedKV = cache.getPagedKV();
  const nextn = treeResult.nextn;
  const totalPaths = 1 << nextn;
  const suffixLen = nextn + 1; // [currentToken, D0, D1, ..., D{nextn-1}]
  const pageSize = pagedKV.pageSize;

  if (pagedKV.maxBatch < totalPaths) {
    throw new Error(
      `mtpVerify: maxBatch (${pagedKV.maxBatch}) too small for nextn=${nextn}, need >= ${totalPaths}`,
    );
  }

  const seq0 = pagedKV.sequences[0];
  const originalAllocLen = seq0.allocLen;
  const tokenIds = seq0.getTokenIds();

  // allocLen = originalAllocLen → planPrefill starts at originalAllocLen, which is
  // exactly where the next decode step would process currentToken. The existing KV
  // at 0..originalAllocLen-1 is the correct prefix context; no prevToken prepend needed.
  pagedKV.positionIdsDirty = true;
  pagedKV.pagesDirtyHost = true;

  // Fork copies for all verification paths (totalPaths = 2^nextn)
  for (let i = 1; i < totalPaths; i++) {
    pagedKV.copySequence(i, 0);
  }
  
  using hostBuf = ws.allocPinned(validationSequences.shape, validationSequences.type);
  using stream = ws.glm.withStream(() => {
    hostBuf.memcpy(validationSequences, validationSequences.bytes, MemcpyKind.DeviceToHost);
  });

  // Run verification prefill: batch=totalPaths, each sequence gets suffixLen tokens
  const seqLens = new Array(totalPaths).fill(suffixLen) as number[];
  const state = ws.planPrefill(model, totalPaths, seqLens, cache);
  state.setInput(validationSequences);

  using hiddenStates = model.forwardInternal(state);
  // Pass null (not undefined) for lastIdx — undefined triggers the default
  // parameter (this.ws.lastIdx, a truthy Tensor), which selects only the last
  // token per sequence via indexSelect. We need all-positions logits.
  using logits = state.computeLogits(hiddenStates, model, null);

  // Argmax all positions: [totalPaths * suffixLen] I32
  using argmaxResult = logits.argmax();

  // Read argmax to host (copy before any subsequent workspace allocations can reclaim the memory)
  using argmaxHost = ws.allocPinned(argmaxResult.shape, argmaxResult.type);
  argmaxHost.memcpy(argmaxResult, argmaxResult.bytes, MemcpyKind.DeviceToHost);

  // Verify: compare draft tokens vs target model argmax for each path.
  // Suffix = [currentToken, D0, D1, ..., D{nextn-1}] starting at originalAllocLen.
  // argmax at position j predicts the token at position j+1:
  //   j=0 (currentToken) → predicts D0; j=1 (D0) → predicts D1; etc.
  let bestPath = 0;
  let bestAccepted = -1;
  let bestReplacement = -1;

  stream.synchronize();
  ws.glm.synchronize();
  const argmaxBuf = Buffer.from(argmaxHost.readPinnedBuffer());
  const validationSequencesBuf = Buffer.from(hostBuf.readPinnedBuffer());

  for (let path = 0; path < totalPaths; path++) {
    let accepted = 0;
    for (let j = 0; j < nextn; j++) {
      const draftToken = validationSequencesBuf.readInt32LE((path * (nextn + 1) + j + 1) * 4);
      const targetToken = argmaxBuf.readInt32LE((path * suffixLen + j) * 4);
      if (draftToken === targetToken) {
        accepted++;
      } else {
        break;
      }
    }
    const replacement = argmaxBuf.readInt32LE((path * suffixLen + accepted) * 4);
    if (accepted > bestAccepted) {
      bestAccepted = accepted;
      bestPath = path;
      bestReplacement = replacement;
    }
  }

  const acceptedTokens: number[] = [];
  for (let j = 0; j < bestAccepted; j++) {
    acceptedTokens.push(validationSequencesBuf.readInt32LE((bestPath * (nextn + 1) + j + 1) * 4));
  }

  // Log verification results
  // const currentToken = validationSequencesBuf.readInt32LE(0);
  // console.log(`MTP Verify: root=${tokenizer?.decode([currentToken]) ?? currentToken} best=[${bestPath}] accepted=${bestAccepted}/${nextn} replacement=${tokenizer?.decode([bestReplacement]) ?? bestReplacement}`);
  // for (let path = 0; path < totalPaths; path++) {
  //   const parts: string[] = [];
  //   for (let j = 0; j < nextn; j++) {
  //     const draftToken = validationSequencesBuf.readInt32LE((path * (nextn + 1) + j + 1) * 4);
  //     const targetToken = argmaxBuf.readInt32LE((path * suffixLen + j) * 4);
  //     const ok = draftToken === targetToken;
  //     parts.push(`${ok ? "✓" : "✗"}${tokenizer?.decode([draftToken]) ?? `?${draftToken}`}`);
  //     if (!ok) break;
  //   }
  //   const marker = path === bestPath ? "*" : " ";
  //   console.log(`  ${marker}[${path}] ${parts.join(" ")}`);
  // }

  // Cleanup: pop forked sequences and restore seq 0 to original state.
  // We don't integrate accepted KV entries yet — just restore and return results.
  for (let i = totalPaths - 1; i > 0; i--) {
    const seq = pagedKV.sequences.pop();
    seq!.clear();
  }

  // Truncate seq 0 back to originalAllocLen. After prefill, allocLen =
  // originalAllocLen + suffixLen. Truncating discards the suffix KV entries.
  seq0.truncate(originalAllocLen);
  const lastPageIdx = seq0.pages.length - 1;
  if (lastPageIdx >= 0) {
    const expected = tokenIds.length - lastPageIdx * pageSize;
    if (seq0.pages[lastPageIdx].tokenIds.length > expected) {
      seq0.pages[lastPageIdx].tokenIds.length = Math.max(0, expected);
    }
  }

  pagedKV.positionIdsDirty = true;
  pagedKV.pagesDirtyHost = true;
  pagedKV.pagesDirtyDevice = true;

  return { numAccepted: bestAccepted, acceptedTokens, replacementToken: bestReplacement };
}


