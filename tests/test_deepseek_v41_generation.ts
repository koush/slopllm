import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { it } from "node:test";
import { CaptureManager } from "../src/capture-manager";
import { DeepseekV41Model } from "../src/deepseek_v41_model";
import { ExecutionWorkspace } from "../src/execution-workspace";
import { freeModelRuntime, loadModelRuntime, parseModelArgs } from "../src/model_cli";
import { TensorParallelism } from "../src/device_ops";
import { ParallelTensor } from "../src/parallel_ops";
import { bf16Round } from "./deepseek_v41_oracles";
import { WorkspaceBase } from "../src/workspace";

// Run through the resident loader with SLOPLLM_TEST_DSV41_RESIDENT=1 and its shared model flags.
it("V4.1 resident generation supports batched chunks, captured decode, and rollback replay", {
  skip: process.env.SLOPLLM_TEST_DSV41_RESIDENT !== "1",
}, async () => {
  const runtime = await loadModelRuntime(parseModelArgs(process.argv.slice(2)));
  try {
    assert.ok(runtime.model instanceof DeepseekV41Model);
    const model = runtime.model;
    assert.equal(model.frozen, true);
    for (const name of ["engram.token_map", "engram.primes", "engram.offsets"]) {
      const tensor = model.tensors.get(name)!;
      assert.ok(tensor, `${name} must be initialized before cache creation`);
      assert.equal(tensor.workspace, model);
      assert.equal(tensor.type, "I32");
    }
    const tokenMap = model.tensors.get("engram.token_map")!;
    const tokenMapBytes = Buffer.alloc(tokenMap.bytes); tokenMap.d2h(tokenMapBytes);
    assert.equal(createHash("sha256").update(tokenMapBytes).digest("hex"),
      "c60a86322ec17b4142bfef3c57a8d81fb428550cdf487f88a4320cb59fe46481", "Resident replay must expose loader-initialized constants");
    using cache = model.createChatCache(16, 2, 128);
    using ws = new ExecutionWorkspace(runtime.ops, 2, 128);
    using manager = new CaptureManager(runtime.ops);
    // Compare actual loaded column shards against the original replicated
    // projection contract, including scale-block boundaries and output gather.
    for (const name of ["layers.0.attn.wq_a", "layers.0.attn.wkv", "layers.2.attn.indexer.wq_b",
      "layers.1.engram.wkv", "layers.14.engram.wkv"]) {
      const weight = model.tensors.get(name + ".weight")!;
      const scales = model.tensors.get(name + ".scale")!;
      if (weight instanceof ParallelTensor) {
        assert.equal(weight.parallelism, TensorParallelism.Column, name);
        assert.equal((scales as ParallelTensor).parallelism, TensorParallelism.Column, name);
      }
      using input = ws.alloc([1, weight.shape[1]], "BF16");
      input.h2d(bf16Round(Float32Array.from({ length: weight.shape[1] }, (_, i) => Math.sin(i * 0.17))));
      using partial = input.linear(weight);
      using actual = partial.replicate();
      using fullWeight = weight instanceof ParallelTensor ? weight.allGather(ws) : weight.viewClone();
      using fullScales = scales instanceof ParallelTensor ? scales.allGather(ws) : scales.viewClone();
      using reference = new WorkspaceBase(runtime.ops);
      const referenceWeight = reference.alloc(weight.shape, weight.type, "projection.weight");
      const referenceScales = reference.alloc(scales.shape, scales.type, "projection.scale");
      referenceWeight.memcpy(fullWeight); referenceScales.memcpy(fullScales);
      using expected = input.linear(referenceWeight);
      const actualBytes = Buffer.alloc(actual.bytes), expectedBytes = Buffer.alloc(expected.bytes);
      actual.d2h(actualBytes);
      expected.d2h(expectedBytes);
      assert.deepEqual(actualBytes, expectedBytes, `${name}: sharded projection must match replicated projection`);
      await runtime.ops.synchronizeAsync();
    }
    cache.reset(2);
    const prompts = ["The capital of France is", "The capital of Germany is"];
    const inputs = prompts.map(p => model.tokenizer.encode(p));
    let remaining = inputs;
    let targets: (number | undefined)[] = [];
    const phasedForward = model.forwardPhased;
    const mhcPost = runtime.ops.mhcPost;
    const localSlots = runtime.ops.dsv4LocalSlots;
    const sharedSlots = runtime.ops.dsv4TopkToSlots;
    let posts = 0, localMappings = 0, sharedMappings = 0;
    runtime.ops.mhcPost = function (...args) { posts++; return mhcPost.apply(this, args); };
    runtime.ops.dsv4LocalSlots = function (...args) { localMappings++; return localSlots.apply(this, args); };
    runtime.ops.dsv4TopkToSlots = function (...args) { sharedMappings++; return sharedSlots.apply(this, args); };
    model.forwardPhased = function* (state) {
      posts = localMappings = sharedMappings = 0;
      const forward = phasedForward.call(this, state);
      let phases = 0;
      try {
        for (;;) {
          const step = forward.next();
          if (step.done) {
            assert.equal(phases, 40 * 4, "Each layer yields around both communication boundaries");
            assert.equal(localMappings, 1, "Local slot mapping is shared across all layers");
            assert.equal(sharedMappings, 8, "Only index-owner layers translate shared slots");
            return step.value;
          }
          phases++;
          assert.equal(posts, Math.floor(phases / 2), "Yield before and after each mHC post/reduction");
          yield;
        }
      } finally { forward.return(undefined!); }
    };
    try {
      while (remaining.some(ids => ids.length)) {
        const result = await model.executePrefill(ws, cache, remaining, manager, 3);
        targets = result.targetTokens.map((token, i) => token ?? targets[i]);
        remaining = result.remainingInputIdsList;
      }
    } finally {
      model.forwardPhased = phasedForward;
      runtime.ops.mhcPost = mhcPost;
      runtime.ops.dsv4LocalSlots = localSlots;
      runtime.ops.dsv4TopkToSlots = sharedSlots;
    }
    assert.deepEqual(targets.map(token => model.tokenizer.decode([token!])), [" Paris", " Berlin"]);
    const decode = model.generateDecode(ws, cache, manager);
    const capturedTokens: number[][][] = [];
    try {
      for (let i = 0; i < 7; i++) {
        const step = await decode.next();
        capturedTokens.push(step.value!.tokens);
        assert.equal(step.done, false);
        // The first call discovers KV-length variance; the bucketed key then
        // performs its normal warmup before capture and replay.
        if (i >= 5) assert.equal(step.value!.warmup, false);
        assert.equal(step.value!.tokens.length, 2);
      }
    } finally { await decode.return(); }
    assert.ok([...manager.captured.values()].some(c => c.graphExec !== null), "Decode must instantiate and replay a CUDA graph");
    using eagerCache = model.createChatCache(16, 2, 128);
    eagerCache.reset(2);
    // Hold prefill geometry constant when testing graph/eager equality: GEMV,
    // batched GEMM and prefill attention have different BF16 rounding paths.
    let eagerRemaining = inputs;
    while (eagerRemaining.some(ids => ids.length)) {
      const result = await model.executePrefill(ws, eagerCache, eagerRemaining, undefined, 3);
      eagerRemaining = result.remainingInputIdsList;
    }
    const eager = model.generateDecode(ws, eagerCache);
    try {
      for (const expected of capturedTokens) {
        const step = await eager.next();
        assert.deepEqual(step.value!.tokens, expected, "Graph replay must agree with eager GPU decode");
      }
    } finally { await eager.return(); }
    cache.getPagedKV().sequences[0].truncate(2);
    const restored = await model.executePrefill(ws, cache, [inputs[0].slice(2), []], manager);
    assert.equal(restored.targetTokens[0], targets[0]);
    assert.equal(restored.targetTokens[1], undefined);

    // Capture before position 128, then replay across both the local-window
    // rollover and a logical page boundary. Odd/even positions also exercise
    // the ratio-2 compressor's pending-pair state during warmup and replay.
    using boundaryCache = model.createChatCache(16, 1, 128);
    using boundaryEagerCache = model.createChatCache(16, 1, 128);
    const padding = model.tokenizer.encode("A short note.\n".repeat(100));
    const boundaryInput = [...padding.slice(0, 120 - inputs[0].length), ...inputs[0]];
    assert.equal(boundaryInput.length, 120);
    for (const c of [boundaryCache, boundaryEagerCache]) {
      c.reset(1);
      let remaining = [boundaryInput];
      while (remaining[0].length) {
        const result = await model.executePrefill(ws, c, remaining, undefined, 32);
        remaining = result.remainingInputIdsList;
      }
    }
    const boundaryTokens: number[][][] = [];
    const boundaryDecode = model.generateDecode(ws, boundaryCache, manager);
    try {
      for (let i = 0; i < 12; i++) {
        const step = await boundaryDecode.next();
        assert.equal(step.done, false);
        if (i >= 5) assert.equal(step.value!.warmup, false, "Boundary crossing must replay the existing graph");
        boundaryTokens.push(step.value!.tokens);
      }
    } finally { await boundaryDecode.return(); }
    const boundaryEager = model.generateDecode(ws, boundaryEagerCache);
    try {
      for (const expected of boundaryTokens) {
        const step = await boundaryEager.next();
        assert.deepEqual(step.value!.tokens, expected, "Captured window rollover must agree with eager decode");
      }
    } finally { await boundaryEager.return(); }

    // A single chunk larger than the local ring must retain early queries'
    // keys until the complete batched attention launch has consumed them.
    using wideWs = new ExecutionWorkspace(runtime.ops, 1, 256);
    using wideCache = model.createChatCache(16, 1, 256);
    wideCache.reset(1);
    const wideInput = [...padding.slice(0, 256 - inputs[0].length), ...inputs[0]];
    const wide = await model.executePrefill(wideWs, wideCache, [wideInput], undefined, 256);
    assert.deepEqual(wide.remainingInputIdsList, [[]]);
    assert.equal(model.tokenizer.decode([wide.targetTokens[0]!]), " Paris");
    // Prefix reuse must forward only the suffix. An odd truncation adds exactly
    // one overlap token, while leaving the donor's pages/history intact.
    using reuseCache = model.createChatCache(16, 2, 128);
    reuseCache.reset(2);
    let seed = [wideInput, []];
    while (seed.some(row => row.length)) {
      seed = (await model.executePrefill(ws, reuseCache, seed, undefined, 64)).remainingInputIdsList;
    }
    const paged = reuseCache.getPagedKV();
    const donor = paged.sequences[0];
    const suffix = reuseCache.prefixMatch(1, wideInput);
    assert.equal(paged.sequences[1].allocLen, 192);
    assert.equal(suffix.length, 64);
    assert.equal(paged.sequences[1].pages[0], donor.pages[0]);
    const originalForward = model.forwardPhased;
    let forwarded = 0;
    model.forwardPhased = function* (state) {
      forwarded += state.totalTokens;
      return yield* originalForward.call(this, state);
    };
    try {
      const reused = await model.executePrefill(ws, reuseCache, [[], suffix], undefined, 64);
      assert.equal(forwarded, 64, "a page-aligned hit must not replay the prefix");
      assert.equal(model.tokenizer.decode([reused.targetTokens[1]!]), " Paris");
      paged.sequences[1].truncate(65);
      forwarded = 0;
      let remaining = [[], wideInput.slice(65)];
      while (remaining.some(row => row.length)) {
        remaining = (await model.executePrefill(ws, reuseCache, remaining, undefined, 64)).remainingInputIdsList;
      }
      assert.equal(forwarded, wideInput.length - 65 + 1, "odd boundary adds just one overlap token");
      assert.deepEqual(paged.sequences[1].getTokenIds(), wideInput);
      assert.deepEqual(donor.getTokenIds(), wideInput);
      paged.sequences[1].truncate(65);
      const pending = paged.sequences[1].targetToken;
      forwarded = 0;
      const resumedDecode = model.generateDecode(ws, reuseCache);
      try {
        await resumedDecode.next();
        assert.equal(forwarded, 3, "one reconditioning token plus two decode inputs");
        assert.equal(paged.sequences[1].allocLen, 66);
        assert.equal(paged.sequences[1].getTokenIds().at(-1), pending, "reconditioning preserves the pending target");
      } finally { await resumedDecode.return(); }
      paged.sequences[1].truncate(65);
      forwarded = 0;
      const single = await model.executePrefill(ws, reuseCache, [[], [wideInput[65]]], undefined, 1);
      assert.deepEqual(single.remainingInputIdsList, [[], []]);
      assert.equal(forwarded, 2, "a one-token chunk uses separate overlap and input plans");
      assert.deepEqual(paged.sequences[1].getTokenIds(), wideInput.slice(0, 66));
      paged.sequences[1].truncate(65);
      const shiftedTarget = paged.sequences[1].targetToken;
      paged.stageSequence(0, 42);
      assert.equal(paged.sequences.length, 1);
      forwarded = 0;
      const shifted = model.generateDecode(ws, reuseCache, manager);
      try {
        await shifted.next();
        assert.equal(forwarded, 2, "a changed batch restores only its odd boundary");
        assert.equal(paged.sequences[0].getTokenIds().at(-1), shiftedTarget);
      } finally { await shifted.return(); }
      const stagedSuffix = reuseCache.prefixMatch(0, wideInput);
      assert.equal(stagedSuffix.length, 64);
      forwarded = 0;
      const staged = await model.executePrefill(ws, reuseCache, [stagedSuffix], undefined, 64);
      assert.equal(forwarded, 64, "staged prefix hits must reuse cache data too");
      assert.equal(model.tokenizer.decode([staged.targetTokens[0]!]), " Paris");
    } finally { model.forwardPhased = originalForward; }
    console.log("V4.1 BATCH/CHUNK/CAPTURE/ROLLOVER/ROLLBACK/PREFIX REUSE GENERATION PASS");
  } finally {
    freeModelRuntime(runtime);
  }
});
