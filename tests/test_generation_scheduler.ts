import assert from "node:assert/strict";
import { setImmediate as nextTurn } from "node:timers/promises";
import test from "node:test";
import { createAsyncQueue, Deferred } from "@scrypted/deferred";
import type { ChatModel } from "../src/chat_model";
import type { CaptureManager } from "../src/capture-manager";
import type { ExecutionWorkspace } from "../src/execution-workspace";
import { GenerationScheduler, type GenerationEvent, type GenerationRequest, type ServerMetrics } from "../src/generation-scheduler";
import { MetaOps } from "../src/meta_ops";
import { PAGE_SIZE, PagedKVCache } from "../src/paged_kv";
import type { SamplingWorkspace } from "../src/sampling";

function setup() {
  const ops = new MetaOps();
  const cache = new PagedKVCache(ops, 1, 8, 1, 16, 2);
  const host = new PagedKVCache(ops, 1, 8, 1, 16, 2, PAGE_SIZE, 0, 0, false, 0, [], true);
  const requests = createAsyncQueue<GenerationEvent>();
  const metrics: ServerMetrics = {
    runningRequests: 0, generationTokensTotal: 0, promptTokensTotal: 0,
    specDecodeNumDraftsTotal: 0, specDecodeNumDraftTokensTotal: 0, specDecodeNumAcceptedTokensTotal: 0,
    mtpPhaseSeconds: new Map(), mtpPhaseCount: new Map(), requestSuccessTotal: 0,
    prefillTimeSecondsCount: 0, prefillTimeSecondsSum: 0,
  };
  const model = {
    ops,
    eosIds: new Set<number>(),
    planChunkedPrefill(_ws: unknown, kv: PagedKVCache, inputs: number[][]) {
      inputs.forEach((input, i) => kv.allocAppendPages(i, input.length));
      return {
        states: [],
        prefillInputIdsList: inputs,
        remainingInputIdsList: inputs.map(() => []),
        generator: (function* () { return inputs.map(() => 1); })(),
        reportTokens() { inputs.forEach((input, i) => kv.reportTokens(i, input, 1)); },
      };
    },
  } as unknown as ChatModel;
  const scheduler = new GenerationScheduler({
    requests, model, cache, hostCache: host, metrics, maxBatchSize: 2, chunkSize: 128, decodeLatencyMs: 0,
    ws: { assertClear() {}, clearTracking() {} } as unknown as ExecutionWorkspace,
    captureManager: { execute(_request: unknown, fn: () => unknown) { return { result: fn() }; } } as unknown as CaptureManager,
    samplingWorkspace: {} as SamplingWorkspace,
  });
  return { scheduler, requests, model, cache, host, metrics };
}

function request(id: string, maxTokens = 1): GenerationRequest {
  const inputIds = Array.from({ length: PAGE_SIZE + 1 }, (_, i) => i + 100);
  return {
    id, inputIds, maxTokens, tokens: createAsyncQueue<number>(),
    samplingParams: { temperature: 0, topP: 1, topK: 1, repetitionPenalty: 1, presencePenalty: 0, repetitionPenaltyWindow: 64 },
    generatedTokenCount: 0, finishReason: "length", promptTokenCount: inputIds.length,
    cachedTokenCount: 0, prefillTokenCount: 0, prefillSeconds: 0,
  };
}

test("flush wakes an idle scheduler and serializes consecutive control events", { timeout: 5000 }, async () => {
  const { scheduler, cache, host } = setup();
  const running = scheduler.run();
  try {
    await nextTurn(); // Scheduler is blocked waiting for admission.
    const results = await Promise.all([scheduler.flushGpuCache(), scheduler.flushGpuCache()]);
    assert.deepEqual(results, [
      { gpuPagesFreed: 0, hostPagesUsed: 0 },
      { gpuPagesFreed: 0, hostPagesUsed: 0 },
    ]);
  } finally {
    scheduler.stop();
    await running;
    cache.free();
    host.free();
  }
});

test("flush waits for generation to finish and precedes later admission and host restore", { timeout: 5000 }, async t => {
  const { scheduler, requests, model, cache, host, metrics } = setup();
  const decoding = new Deferred<void>();
  const finishDecode = new Deferred<void>();
  let generatorClosed = false;
  const logs = t.mock.method(console, "log", () => {});
  model.generateDecode = async function* () {
    try {
      decoding.resolve();
      await finishDecode.promise;
      cache.allocAppendPages(0, 1);
      cache.reportTokens(0, [1], 2);
      yield { tokens: [[2]], numDraftTokens: 0, numAccepted: [0], warmup: false,
        draft: { targetTokens: [1], treeTokens: [[]], topks: [] } };
    } finally {
      generatorClosed = true;
    }
  };
  const first = request("first", 2);
  const second = request("second");
  requests.submit(first);
  const running = scheduler.run();
  try {
    await decoding.promise;
    let flushed = false;
    const flush = scheduler.flushGpuCache().then(result => { flushed = true; return result; });
    requests.submit(second);
    await nextTurn();
    assert.equal(flushed, false);
    assert.equal(second.generatedTokenCount, 0);
    assert.equal(host.sequences.length, 0, "no offload while the generator is open");
    finishDecode.resolve();
    assert.deepEqual(await flush, { gpuPagesFreed: 2, hostPagesUsed: 1 });
    assert.equal(generatorClosed, true);
    assert.equal(cache.availablePages.length, cache.maxPages);
    await second.tokens.endPromise;
    assert.equal(second.cachedTokenCount, PAGE_SIZE, "later admission restores the offloaded prefix");
    assert.equal(second.prefillTokenCount, 1);
    assert.equal(metrics.promptTokensTotal, 2 * (PAGE_SIZE + 1), "control events are not generation requests");
    const messages = logs.mock.calls.map(call => String(call.arguments[0]));
    assert(messages.some(message => message.includes("Host cache hit:")));
    assert(messages.some(message => message.includes("Host cache restore complete:")));
  } finally {
    finishDecode.resolve();
    scheduler.stop();
    await running;
    cache.free();
    host.free();
  }
});

test("shutdown rejects queued flushes and new flush requests", async () => {
  const { scheduler, cache, host } = setup();
  const flush = scheduler.flushGpuCache();
  const rejected = assert.rejects(flush, /shutting down/);
  scheduler.stop();
  await rejected;
  await assert.rejects(scheduler.flushGpuCache(), /shutting down/);
  cache.free();
  host.free();
});
