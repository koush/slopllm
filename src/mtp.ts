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
 *   state.finishPrefill();
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
): Tensor[] {
  if (!model.forwardMtp) {
    throw new Error("mtpPrefill: model does not support MTP (forwardMtp not defined)");
  }

  const batchSize = state.batchSize;
  // topkIndices tracks the rotation token for the next layer.
  // Layer 0 uses the target model's sampled token; layers 1+ use the previous MTP layer's top-1.
  let topkIndices: Tensor = gpuSampleResult;
  const predictions: Tensor[] = [];

  // rotatedIds ping-pongs between workspace allocations.
  // Source for layer 0 is ws.inputIdsBuf (the original prompt tokens).
  using rotatedIds = new UsingHolder<Tensor>(undefined!);

  for (let i = 0; i < nextn; i++) {
    // Rotate input IDs: shift each sequence left by 1, append previous prediction
    const source = rotatedIds.value ?? ws.inputIdsBuf;
    rotatedIds.replace(source.rotateInputIds(ws.qoIndptrD, topkIndices, batchSize));

    // Forward through one MTP layer with rotated input
    const hiddenStates = model.forwardMtp(state, targetHiddenStates, rotatedIds.value);

    // Sample top-1 from this layer's output for the next rotation
    using logits = state.computeLogits(hiddenStates, model);
    hiddenStates[Symbol.dispose]();
    const topk = logits.topk(1, model.cfg.vocabSize);
    topkIndices = topk.indices; // [batchSize, 1] I32 — stride-compatible with rotateInputIds
    topk.values[Symbol.dispose]();

    // Detach topk.indices so it survives the loop; caller disposes predictions.
    predictions.push(topk.indices);
  }

  return predictions;
}

/**
 * MTP decode: produce draft predictions for the next `nextn` tokens.
 *
 * During decode, each MTP layer decodes one token using the target model's
 * hidden states and the previous layer's prediction as input:
 *   Layer 0: input = target model's sampled token → output = draft D0
 *   Layer 1: input = D0 → output = draft D1
 *   Layer 2: input = D1 → output = draft D2
 *
 * Each layer's KV cache is advanced by one position (via planDecode + decodeStep).
 * After all layers, the KV cache is rewound by `nextn` positions so the next
 * target decode step starts at the correct position.
 *
 * Returns the top-1 prediction indices for each MTP layer (shape [1, 1] I32 each).
 * These are token IDs, suitable for mtpReadDrafts.
 * The caller is responsible for disposing the returned tensors when done.
 * The caller should also call `mtpReadDrafts` and synchronize before reading.
 *
 * Prerequisites:
 *   - The target model must have just decoded (planDecode + decodeStep + forward
 *     already called for the current position)
 *   - `state` must be the target model's decode state (isDecode = true, batchSize = 1)
 *   - `gpuSampleResult` must be [1] I32 on GPU (target model's sampled token)
 *   - model.forwardMtp must exist (MTP enabled)
 *   - The paged KV cache must have enough pages for `nextn` additional tokens
 *
 * @param state - The target model's decode ExecutionState
 * @param model - The chat model (must support forwardMtp)
 * @param targetHiddenStates - The target model's hidden states from the current decode step
 * @param ws - Execution workspace
 * @param gpuSampleResult - [1] I32 GPU tensor: target model's sampled token
 * @param nextn - Number of MTP layers (draft predictions to produce)
 * @param cache - Chat cache (paged KV cache, must have 1 sequence)
 * @returns Array of prediction tensors (one per MTP layer, shape [1, 1] I32)
 */
export function mtpDecode(
  state: ExecutionState,
  model: ChatModel,
  targetHiddenStates: Tensor,
  ws: ExecutionWorkspace,
  gpuSampleResult: Tensor,
  nextn: number,
  cache: ChatCache,
): Tensor[] {
  if (!model.forwardMtp) {
    throw new Error("mtpDecode: model does not support MTP (forwardMtp not defined)");
  }

  const pagedKV = cache.getPagedKV();
  const predictions: Tensor[] = [];
  let topkIndices: Tensor | null = null;

  for (let i = 0; i < nextn; i++) {
    // Copy input token to ws.inputIdsBuf for forwardMtp's embedding lookup.
    // Layer 0 uses the target model's sampled token; layers 1+ use previous top-1.
    if (i === 0) {
      ws.inputIdsBuf.memcpy(gpuSampleResult, I32, MemcpyKind.DeviceToDevice);
    } else {
      // topkIndices is [1, 1] I32; copy first element to inputIdsBuf [1] I32
      ws.inputIdsBuf.memcpy(topkIndices!, I32, MemcpyKind.DeviceToDevice);
    }

    // Force position IDs and decode plan to update for the new position.
    // planDecode caches these; setting dirty flags ensures recalculation.
    pagedKV.positionIdsDirty = true;
    pagedKV.pagesDirtyHost = true;

    const mtpState = ws.planDecode(model, 1, cache);
    ws.decodeStep(mtpState, model);

    // Forward through one MTP layer (uses ws.inputIdsBuf for embedding)
    const hiddenStates = model.forwardMtp(mtpState, targetHiddenStates);

    // Sample top-1 from this layer's output
    using logits = mtpState.computeLogits(hiddenStates, model);
    hiddenStates[Symbol.dispose]();
    const topk = logits.topk(1, model.cfg.vocabSize);
    topkIndices = topk.indices; // [1, 1] I32
    topk.values[Symbol.dispose]();

    predictions.push(topk.indices);
  }

  // Rewind: undo the nextn decode tokens added by planDecode.
  // This restores the KV cache position so the next target decode
  // starts at the correct position.
  const seq = pagedKV.sequences[0];
  seq.truncate(seq.allocLen - nextn);
  pagedKV.positionIdsDirty = true;
  pagedKV.pagesDirtyHost = true;
  pagedKV.pagesDirtyDevice = true;
  ws.decodeStep(state, model, -nextn);

  return predictions;
}

/**
 * Initiate async GPU→Host copies of MTP draft predictions.
 *
 * Each prediction tensor is [batchSize, 1] I32 on GPU. The returned pinned host
 * buffers receive the data via async memcpy. The caller MUST synchronize the
 * stream (e.g., `glm.synchronize()`) before reading the buffers.
 *
 * @param predictions - MTP prediction tensors from mtpPrefill or mtpDecode
 * @param ws - Workspace for allocating pinned host buffers
 * @returns Array of pinned host tensors, one per MTP layer (caller reads after sync)
 */
export function mtpReadDrafts(
  predictions: Tensor[],
  ws: ExecutionWorkspace,
): Tensor[] {
  const hosts: Tensor[] = [];
  for (const pred of predictions) {
    const host = ws.allocPinned(pred.shape, pred.type);
    host.memcpy(pred, pred.bytes, MemcpyKind.DeviceToHost);
    hosts.push(host);
  }
  return hosts;
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

  // The target model's decode left ws.inputIdsBuf containing the previous token
  // (set before the target forward pass). The MTP layer needs the current decoded
  // token (gpuSampleResult) as input for iteration 0.
  ws.inputIdsBuf.memcpy(gpuSampleResult, gpuSampleResult.bytes, MemcpyKind.DeviceToDevice);

  let total = 1;
  let doubledTargetHs: Tensor | null = null;

  for (let i = 0; i < nextn; i++) {
    const batchSize = 1 << i;

    if (total > 1) {
      for (let j = 0; j < total / 2; j++) {
        pagedKV.copySequence(j + total / 2, j);
      }
    }

    if (batchSize > 1) {
      const prev: Tensor = doubledTargetHs ?? targetHiddenStates;
      const doubled: Tensor = prev.cat([prev], 0);
      if (doubledTargetHs) doubledTargetHs[Symbol.dispose]();
      doubledTargetHs = doubled;
    }
    const hs = batchSize === 1 ? targetHiddenStates : doubledTargetHs!;

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
      ws.decodeStep(mtpState, model);
    }

    const hiddenStates = model.forwardMtp!(mtpState, hs);
    using logits = mtpState.computeLogits(hiddenStates, model);
    hiddenStates[Symbol.dispose]();

    const topk = logits.topk(2, model.cfg.vocabSize);
    topk.values[Symbol.dispose]();

    const batch = total * 2;
    const half = batch / 2;

    const fanout = totalPaths >>> (i + 1);
    for (let j = 0; j < batch; j++) {
      for (let k = 0; k < fanout; k++) {
        validationSequences.memcpy2d(
          ((j * fanout + k) * rowLen + i + 1) * 4, rowLen * 4,
          topk.indices, j * 4, 4,
          4, 1,
          MemcpyKind.DeviceToDevice,
        );
      }
    }

    if (i < nextn - 1) {
      const reordered = ws.alloc(topk.indices.shape, topk.indices.type);
      reordered.memcpy2d(0, 4, topk.indices, 0, 8, 4, half, MemcpyKind.DeviceToDevice);
      reordered.memcpy2d(half * 4, 4, topk.indices, 4, 8, 4, half, MemcpyKind.DeviceToDevice);
      ws.inputIdsBuf.memcpy(reordered, batch * I32, MemcpyKind.DeviceToDevice);
      reordered[Symbol.dispose]();
    }

    topk.indices[Symbol.dispose]();
    total *= 2;
  }

  if (doubledTargetHs) doubledTargetHs[Symbol.dispose]();

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


