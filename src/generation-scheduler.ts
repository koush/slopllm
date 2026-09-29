import { createAsyncQueue } from "@scrypted/deferred";
import type { CaptureManager } from "./capture-manager";
import type { ChatCache, ChatModel, SamplingParams, TokenSelector } from "./chat_model";
import type { DeviceOps } from "./device_ops";
import type { ExecutionWorkspace } from "./execution-workspace";
import { PageAllocationError, type PagedKVCache, type Sequence } from "./paged_kv";
import type { SamplingWorkspace } from "./sampling";

export interface GenerationRequest {
  id: string;
  inputIds: number[];
  maxTokens: number;
  samplingParams: SamplingParams;
  tokens: ReturnType<typeof createAsyncQueue<number>>;
  generatedTokenCount: number;
  finishReason: string;
  promptTokenCount: number;
  cachedTokenCount: number;
  prefillTokenCount: number;
  prefillSeconds: number;
  decodeStartedAt?: number;
}

export interface GpuCacheFlushResult {
  gpuPagesFreed: number;
  hostPagesUsed: number;
}

export interface HostCacheFlushResult {
  hostPagesFreed: number;
}

interface GpuCacheFlushEvent {
  kind: "flush_gpu_cache";
  resolve: (result: GpuCacheFlushResult) => void;
  reject: (error: unknown) => void;
}

interface HostCacheFlushEvent {
  kind: "flush_host_cache";
  resolve: (result: HostCacheFlushResult) => void;
  reject: (error: unknown) => void;
}

export type CacheFlushEvent = GpuCacheFlushEvent | HostCacheFlushEvent;

export type GenerationEvent = GenerationRequest | CacheFlushEvent;

export interface ServerMetrics {
  runningRequests: number;
  generationTokensTotal: number;
  promptTokensTotal: number;
  specDecodeNumDraftsTotal: number;
  specDecodeNumDraftTokensTotal: number;
  specDecodeNumAcceptedTokensTotal: number;
  mtpPhaseSeconds: Map<string, number>;
  mtpPhaseCount: Map<string, number>;
  requestSuccessTotal: number;
  prefillTimeSecondsCount: number;
  prefillTimeSecondsSum: number;
}

export function isFatalCudaError(error: unknown): boolean {
  const message = error instanceof Error ? `${error.message}\n${error.stack ?? ""}` : String(error);
  return /illegal memory access|misaligned address|device-side assert|unspecified launch failure|CUDA context.*destroyed/i.test(message);
}

/**
 * Two-tier prefix policy over the device KV cache and an optional pinned-host
 * KV cache.
 *
 * Evictability is staging membership and nothing else: an active sequence
 * holds its own page refs and is protected; a staged sequence is reclaimable.
 * Staged GPU content is limited to finished rows retained via
 * retainSequence() (negative staging keys) — rows the scheduler stages around
 * cache.reset() use non-negative keys and are never eviction candidates. The
 * policy is the only writer of host staging.
 */
export class PrefixTierPolicy {
  private nextKey = -1;
  private readonly retainedKeys = new Set<number>();
  private nextHostKey = 0;
  private host?: PagedKVCache;

  constructor(private readonly pagedKV: PagedKVCache, private readonly ops: DeviceOps, host?: PagedKVCache) {
    pagedKV.onPagePressure = requiredPages => this.evictGpuStaged(requiredPages);
    if (host) {
      this.host = host;
      host.onPagePressure = requiredPages => this.evictHostStaged(requiredPages);
    }
  }

  retainSequence(index: number): void {
    const key = this.nextKey--;
    this.pagedKV.stageSequence(index, key);
    this.retainedKeys.add(key);
  }

  clear(): void {
    this.pagedKV.clearStaging();
    this.retainedKeys.clear();
    this.host?.clearStaging();
  }

  private matchGpuPrefixes(prompts: number[][]): number[][] {
    const gpu = this.pagedKV;
    const resumable = new Map<Sequence, number>();
    for (const key of this.retainedKeys) {
      const sequence = gpu.staging.get(key);
      if (sequence) resumable.set(sequence, key);
      else this.retainedKeys.delete(key);
    }
    return prompts.map((prompt, index) => {
      const suffix = gpu.prefixMatch(index, prompt, sequence => resumable.has(sequence));
      const key = resumable.get(gpu.sequences[index]);
      if (key !== undefined && !gpu.staging.has(key)) {
        // Ownership moved to this request. Do not let a later request in the
        // same admission batch consume it again; permission checks are pure.
        resumable.delete(gpu.sequences[index]);
        this.retainedKeys.delete(key);
        console.log(`GPU cache resume: sequence=${index} cached-tokens=${prompt.length - suffix.length} partial-tokens=${gpu.sequences[index].allocLen % gpu.pageSize}`);
      }
      return suffix;
    });
  }

  // Only stage host slots after all earlier transfers have drained.
  private stageHostSequences(): void {
    const host = this.host;
    if (!host) return;
    for (let i = host.sequences.length - 1; i >= 0; i--) {
      if (host.sequences[i].pages.length) {
        host.stageSequence(i, this.nextHostKey++);
      } else {
        host.removeSequence(i);
      }
    }
  }

  /** Called by the scheduler with no active requests or open generators. */
  async flushGpuCache(): Promise<GpuCacheFlushResult> {
    const gpu = this.pagedKV;
    if (gpu.sequences.some(seq => seq.pages.length)
      || [...gpu.staging.keys()].some(key => !this.retainedKeys.has(key))) {
      throw new Error("Cannot flush GPU cache while sequences are active");
    }
    await this.ops.synchronizeAsync();
    this.stageHostSequences();
    const gpuPagesFreed = gpu.maxPages - gpu.availablePages.length;
    try {
      this.evictGpuStaged(gpu.maxPages);
    } finally {
      // Finish offloads before acknowledging the flush or allowing admission.
      await this.ops.synchronizeAsync();
    }
    gpu.reset(0);
    const hostPagesUsed = this.host ? this.host.maxPages - this.host.availablePages.length : 0;
    console.log(`GPU cache flushed: freed-pages=${gpuPagesFreed} host-pages=${hostPagesUsed}`);
    return { gpuPagesFreed, hostPagesUsed };
  }

  /** Called by the scheduler with no active requests or open generators. */
  async flushHostCache(): Promise<HostCacheFlushResult> {
    const host = this.host;
    if (!host) {
      throw new Error("Host cache flush requires a host cache (--max-host-pages > 0)");
    }
    await this.ops.synchronizeAsync();
    const hostPagesFreed = host.maxPages - host.availablePages.length;
    // clearStaging releases staged page refs; reset(0) drops the remaining
    // active slots and rebuilds the free list. Host pages are never shared
    // with the device tier, so dropping every ref frees them all.
    host.clearStaging();
    host.reset(0);
    console.log(`Host cache flushed: freed-pages=${hostPagesFreed}`);
    return { hostPagesFreed };
  }

  /**
   * Admit-time tier match: returns each request's uncached suffix to prefill.
   * Finished GPU rows may be consumed to resume a private partial page.
   * Without a host tier no await is reached inside. Suffixes derive from the
   * prompt, never from cache bookkeeping.
   */
  async prefixMatch(prompts: number[][]): Promise<number[][]> {
    const gpu = this.pagedKV;
    const host = this.host;
    if (!host) {
      return this.matchGpuPrefixes(prompts);
    }
    const pageSize = gpu.pageSize;

    // Stage the entire host tier; discard empties. Everything unstaged here
    // is provably drained — every earlier invocation ended in the trailing
    // sync and decode steps sync in between.
    this.stageHostSequences();

    // Host prime: the request's reusable prefix moves out of staging into
    // an active, request-indexed host slot. The slot's page refs protect
    // the content for the whole window: eviction may drop the staged
    // original, the shared pages survive.
    const hostSuffix = prompts.map((p, i) => host.prefixMatch(i, p));

    // Device prime shares full pages or resumes a finished retained row, so
    // nothing allocates. Rows the
    // scheduler staged around cache.reset() are visible candidates but
    // never evictable (negative keys only).
    const suffixes = this.matchGpuPrefixes(prompts);
    for (let i = 0; i < prompts.length; i++) {
      const hostTokens = prompts[i].length - hostSuffix[i].length;
      if (hostTokens) {
        console.log(`Host cache hit: sequence=${i} matched-tokens=${hostTokens} gpu-matched-tokens=${prompts[i].length - suffixes[i].length}`);
      }
    }

    // Restage superseded host slots: where the device match is at least as
    // long as the host match, no restore will read the slot, so it rejoins
    // the eviction pool now instead of holding refs through the restore
    // below. Snapshot restore sources — the staging splices scramble
    // indices.
    const restoreSources = new Map<number, Sequence>();
    for (let i = prompts.length - 1; i >= 0; i--) {
      const hostSeq = host.sequences[i];
      if (suffixes[i].length <= hostSuffix[i].length) {
        if (hostSeq.pages.length) {
          host.stageSequence(i, this.nextHostKey++);
        } else {
          host.removeSequence(i);
        }
      } else {
        restoreSources.set(i, hostSeq);
      }
    }

    // Restore: where the host tier holds strictly more full pages than the
    // device, move the delta across. Page counts stay floored (the
    // partial-page margin is intentionally forgone) and the prompt bounds
    // the copy, so a host history running past the prompt cannot inject
    // foreign tokens. Pressure here offloads whole finished rows to host,
    // best-effort.
    const restored: { sequence: number; tokens: number }[] = [];
    for (const [i, src] of restoreSources) {
      const prompt = prompts[i];
      const gpuPages = Math.floor((prompt.length - suffixes[i].length) / pageSize);
      const hostPages = Math.floor((prompt.length - hostSuffix[i].length) / pageSize);
      const pages = Math.min(hostPages, Math.floor(prompt.length / pageSize));
      if (pages <= gpuPages) {
        continue;
      }
      gpu.copyPrefixFrom(host, src, i, pages);
      suffixes[i] = prompt.slice(pages * pageSize);
      restored.push({ sequence: i, tokens: pages * pageSize });
    }

    // Drain every copy this invocation enqueued. The active host slots left
    // behind (restore sources, offload destinations) are staged by the
    // stage-all of the next invocation, from a provably drained state.
    await this.ops.synchronizeAsync();
    for (const { sequence, tokens } of restored) {
      console.log(`Host cache restore complete: sequence=${sequence} prefix-tokens=${tokens} prefill-tokens=${suffixes[sequence].length}`);
    }
    return suffixes;
  }

  // Device page pressure: whole-victim eviction of finished rows. Offload to
  // the host tier is best-effort: its allocator reclaims drained host staging
  // as needed. If it still cannot allocate, discard the failed destination
  // and drop the victim. The destination appends a fresh active host slot
  // (born protected; staged by the next prefixMatch).
  private evictGpuStaged(requiredPages: number): void {
    const host = this.host;
    while (this.pagedKV.availablePages.length < requiredPages && this.retainedKeys.size) {
      let victimKey: number | undefined;
      let victimPages = 0;
      for (const key of this.retainedKeys) {
        const seq = this.pagedKV.staging.get(key);
        if (!seq || !seq.pages.length) {
          this.retainedKeys.delete(key);
          if (seq) {
            this.pagedKV.removeStagedSequence(key);
          }
        } else if (seq.pages.length > victimPages) {
          victimKey = key;
          victimPages = seq.pages.length;
        }
      }
      if (victimKey === undefined) {
        break;
      }
      const victim = this.pagedKV.staging.get(victimKey)!;
      if (host) {
        const dstSeqIdx = host.sequences.length;
        try {
          host.copyPrefixFrom(this.pagedKV, victim, dstSeqIdx);
        } catch (error) {
          if (!(error instanceof PageAllocationError)) {
            throw error;
          }
          // Page reservation fails before the copy is submitted. Release any
          // prefix refs acquired by this fresh destination before dropping it.
          host.removeSequence(dstSeqIdx);
        }
      }
      this.retainedKeys.delete(victimKey);
      this.pagedKV.removeStagedSequence(victimKey);
    }
  }

  // Host page pressure: drop the largest staged whole sequence first. No
  // exemptions — anything worth protecting is active and holds its own refs.
  private evictHostStaged(requiredPages: number): void {
    const host = this.host;
    if (!host) {
      return;
    }
    while (host.availablePages.length < requiredPages && host.staging.size) {
      let victimKey: number | undefined;
      let victimPages = 0;
      let emptied = false;
      for (const [key, seq] of host.staging) {
        if (!seq.pages.length) {
          host.removeStagedSequence(key);
          emptied = true;
          continue;
        }
        if (seq.pages.length > victimPages) {
          victimKey = key;
          victimPages = seq.pages.length;
        }
      }
      if (victimKey !== undefined) {
        host.removeStagedSequence(victimKey);
      } else if (!emptied) {
        break;
      }
    }
  }
}

interface SchedulerOptions {
  requests: ReturnType<typeof createAsyncQueue<GenerationEvent>>;
  model: ChatModel;
  ws: ExecutionWorkspace;
  cache: ChatCache;
  /** Pinned-host KV tier for prefix offload/restore; omitted => single-tier. */
  hostCache?: ChatCache;
  captureManager: CaptureManager;
  samplingWorkspace: SamplingWorkspace;
  metrics: ServerMetrics;
  maxBatchSize: number;
  chunkSize: number;
  decodeLatencyMs: number;
  numDraftTokens?: number;
}

/** Owns batch membership and GPU execution; HTTP handlers own token consumption. */
export class GenerationScheduler {
  private active: GenerationRequest[] = [];
  private readonly admitted = new Set<GenerationRequest>();
  private readonly prefixes: PrefixTierPolicy;
  private nextStagingKey = 0;
  private pendingFlush?: CacheFlushEvent;

  constructor(private readonly options: SchedulerOptions) {
    this.prefixes = new PrefixTierPolicy(options.cache.getPagedKV(), options.model.ops, options.hostCache?.getPagedKV());
  }

  /** Queue an admission barrier; resolves after existing requests finish and the cache is flushed. */
  flushGpuCache(): Promise<GpuCacheFlushResult> {
    return new Promise((resolve, reject) => {
      if (!this.options.requests.submit({ kind: "flush_gpu_cache", resolve, reject })) {
        reject(new Error("Server is shutting down"));
      }
    });
  }

  /** Queue an admission barrier; resolves after existing requests finish and the host cache is flushed. */
  flushHostCache(): Promise<HostCacheFlushResult> {
    return new Promise((resolve, reject) => {
      if (!this.options.requests.submit({ kind: "flush_host_cache", resolve, reject })) {
        reject(new Error("Server is shutting down"));
      }
    });
  }

  stop(error = new Error("Server shutting down")): void {
    this.options.requests.end();
    for (const request of this.options.requests.clear()) {
      if ("kind" in request) request.reject(error);
      else request.tokens.end(error);
    }
    this.pendingFlush?.reject(error);
    this.pendingFlush = undefined;
    for (const request of this.admitted) {
      request.tokens.end(error);
    }
  }

  private finish(request: GenerationRequest): void {
    request.tokens.end();
    if (!this.admitted.delete(request)) {
      return;
    }
    this.options.metrics.runningRequests--;
    this.options.metrics.requestSuccessTotal++;
  }

  private removeFinished(): void {
    for (let index = this.active.length - 1; index >= 0; index--) {
      const request = this.active[index];
      if (!request.tokens.ended) {
        continue;
      }
      if (request.inputIds.length) {
        this.options.cache.getPagedKV().removeSequence(index);
      } else {
        this.prefixes.retainSequence(index);
      }
      this.active.splice(index, 1);
      this.finish(request);
    }
  }

  private publish(request: GenerationRequest, tokens: readonly number[]): void {
    for (const token of tokens) {
      if (!request.tokens.submit(token)) {
        break;
      }
      request.generatedTokenCount++;
      if (this.options.model.eosIds.has(token) || request.generatedTokenCount >= request.maxTokens) {
        request.finishReason = this.options.model.eosIds.has(token) ? "stop" : "length";
        request.tokens.end();
        break;
      }
    }
  }

  private recordPhase(name: string, batchSize: number, seconds: number): void {
    const { metrics, numDraftTokens } = this.options;
    if (!numDraftTokens) {
      return;
    }
    const key = `${name}|${batchSize}`;
    metrics.mtpPhaseSeconds.set(key, (metrics.mtpPhaseSeconds.get(key) ?? 0) + seconds);
    metrics.mtpPhaseCount.set(key, (metrics.mtpPhaseCount.get(key) ?? 0) + 1);
  }

  /** Only called at a generator boundary, with active rows in cache order. */
  private prepareSampling(prefill: boolean): TokenSelector | undefined {
    const { samplingWorkspace: sampler, cache, numDraftTokens } = this.options;
    const params = this.active.map(request => request.samplingParams);
    if (numDraftTokens) {
      if (sampler.mtpEnabled) {
        sampler.updateMtpSampler(params);
      }
      if (prefill || !sampler.mtpEnabled) {
        const targetParams = prefill ? params : params.flatMap(param =>
          Array.from({ length: numDraftTokens + 1 }, () => param));
        sampler.updateSampler(targetParams, targetParams.map(() => []));
      }
      return sampler;
    }
    if (params.every(param => param.temperature <= 0 && param.repetitionPenalty === 1 && param.presencePenalty === 0)) {
      return undefined;
    }
    sampler.updateSampler(params, cache.getPagedKV().sequences.map((sequence, index) => {
      const request = this.active[index];
      return [...sequence.getTokenIds(), ...(request.inputIds.length ? request.inputIds
        : sequence.targetToken === undefined ? [] : [sequence.targetToken])];
    }));
    return sampler;
  }

  /**
   * Moves incoming requests into the batch: saves active rows into staging,
   * resets the cache batch, prime-matches each request's prefix against both
   * tiers (see PrefixTierPolicy.prefixMatch), then restores the saved rows.
   * Async only when a host tier restores copies; a tier-less call reaches no
   * await.
   */
  private async admit(requests: GenerationRequest[]): Promise<void> {
    const { cache, metrics } = this.options;
    const pagedKV = cache.getPagedKV();
    for (const request of requests) {
      this.admitted.add(request);
      metrics.runningRequests++;
      metrics.promptTokensTotal += request.promptTokenCount;
    }
    const saved = this.active.map(row => ({ row, key: this.nextStagingKey++ }));
    for (const entry of saved) {
      pagedKV.stageSequence(0, entry.key);
    }
    this.active = [];
    cache.reset(requests.length);
    // Tier-aware prefix match: device slots primed, host prefix restored
    // where it extends the device match. Async only when a host tier is
    // attached (the trailing copy drain); otherwise no await is reached.
    const suffixes = await this.prefixes.prefixMatch(requests.map(request => request.inputIds));
    for (const [index, request] of requests.entries()) {
      request.inputIds = suffixes[index];
      request.cachedTokenCount = request.promptTokenCount - request.inputIds.length;
      request.prefillTokenCount = 0;
    }
    for (const entry of saved) {
      pagedKV.unstageSequence(entry.key);
    }
    this.active = [...requests, ...saved.map(entry => entry.row)];
    console.log(`Prefill: requests=${requests.length} tokens=${requests.reduce((sum, request) => sum + request.inputIds.length, 0)}`);
  }

  /** Run one chunk, leaving unfinished input on its request for the next scheduler turn. */
  private async prefill(): Promise<boolean> {
    const { model, ws, cache, captureManager, metrics, chunkSize } = this.options;
    const inputIds = this.active.map(request => request.inputIds);
    if (!inputIds.some(ids => ids.length)) {
      return true;
    }
    const start = performance.now();
    ws.assertClear();
    ws.clearTracking();
    const plan = model.planChunkedPrefill(ws, cache, inputIds, chunkSize, this.prepareSampling(true));
    let targetTokens: (number | undefined)[];
    try {
      // All prefill chunks are eager; graph capture belongs to decode only.
      const execution = captureManager.execute({ states: plan.states, inputs: {}, key: [] }, () => {
        while (true) {
          const result = plan.generator.next();
          if (result.done) return result.value;
        }
      });
      targetTokens = execution.result;
      await model.ops.synchronizeAsync();
      plan.reportTokens();
    } finally {
      plan.generator.return([]);
    }
    ws.assertClear();
    ws.clearTracking();
    const seconds = (performance.now() - start) / 1000;
    this.recordPhase("prefill_chunk", this.active.length, seconds);
    // Adopt remainders only after execution and token reporting succeed.
    this.active.forEach((request, index) => {
      if (request.inputIds.length) {
        request.inputIds = plan.remainingInputIdsList[index];
        request.prefillTokenCount += plan.prefillInputIdsList[index].length;
        request.prefillSeconds += seconds;
        if (!request.inputIds.length) {
          metrics.prefillTimeSecondsCount++;
          metrics.prefillTimeSecondsSum += request.prefillSeconds;
          this.publish(request, [targetTokens[index]!]);
        }
      }
    });
    return plan.remainingInputIdsList.every(ids => !ids.length);
  }

  private async yieldToRequests(): Promise<void> {
    const delay = this.options.decodeLatencyMs;
    if (delay > 0) {
      await new Promise(resolve => setTimeout(resolve, delay));
    } else {
      await new Promise<void>(resolve => setImmediate(resolve));
    }
  }

  /** One lifetime-long scheduler task; generators are scoped to stable batches. */
  async run(): Promise<void> {
    const { requests, model, ws, cache, captureManager, metrics, maxBatchSize, numDraftTokens } = this.options;
    cache.reset(0);
    while (!requests.ended) {
      // Admission: remove terminated rows, wait if empty, then fill available slots.
      this.removeFinished();
      if (!this.active.length && this.pendingFlush) {
        const flush = this.pendingFlush;
        try {
          if (flush.kind === "flush_gpu_cache") {
            flush.resolve(await this.prefixes.flushGpuCache());
          } else {
            flush.resolve(await this.prefixes.flushHostCache());
          }
        } catch (error) {
          flush.reject(error);
          throw error;
        } finally {
          this.pendingFlush = undefined;
        }
        continue;
      }
      const incoming: GenerationRequest[] = [];
      if (!this.active.length) {
        try {
          const request = await requests.dequeue();
          if ("kind" in request) {
            if (requests.ended) request.reject(new Error("Server is shutting down"));
            else this.pendingFlush = request;
            continue;
          }
          if (!request.tokens.ended) {
            incoming.push(request);
          }
        } catch (error) {
          if (requests.ended) {
            return;
          }
          throw error;
        }
        // Collect other HTTP callbacks already ready in this event-loop turn.
        await this.yieldToRequests();
      }
      if (requests.ended) {
        for (const request of incoming) {
          request.tokens.end(new Error("Server shutting down"));
        }
        break;
      }
      while (!this.pendingFlush && this.active.length + incoming.length < maxBatchSize) {
        const request = requests.take();
        if (!request) {
          break;
        }
        if ("kind" in request) {
          this.pendingFlush = request;
          break;
        }
        if (!request.tokens.ended) {
          incoming.push(request);
        }
      }
      try {
        if (incoming.length) {
          await this.admit(incoming);
        }
        // Prefill: run one chunk, then revisit admission if more input or batch changes remain.
        const finalPrefill = await this.prefill();
        await this.yieldToRequests();
        if (!finalPrefill || requests.ended || this.active.some(request => request.tokens.ended)
          || (!this.pendingFlush && this.active.length < maxBatchSize && requests.queued.length > 0)) {
          continue;
        }
        if (!this.active.length) {
          continue;
        }
        // Decode: keep the batch stable until termination or a pending admission.
        for (const request of this.active) {
          request.decodeStartedAt ??= performance.now();
        }
        const samplingPolicy = this.prepareSampling(false);
        const generator = numDraftTokens
          ? model.generateMtpDecode!(ws, cache, numDraftTokens, captureManager, samplingPolicy)
          : model.generateDecode(ws, cache, captureManager, samplingPolicy);
        for await (const step of generator) {
          if (step.numDraftTokens > 0) {
            metrics.specDecodeNumDraftsTotal += this.active.length;
          }
          metrics.specDecodeNumDraftTokensTotal += step.numDraftTokens * this.active.length;
          metrics.specDecodeNumAcceptedTokensTotal += step.numAccepted.reduce((sum, count) => sum + count, 0);
          step.tokens.forEach((tokens, index) => {
            this.publish(this.active[index], tokens);
          });
          await this.yieldToRequests();
          if (requests.ended || this.active.some(request => request.tokens.ended)
            || (!this.pendingFlush && this.active.length < maxBatchSize && requests.queued.length > 0)) {
            break;
          }
        }
        // for-await has closed the generator before any sequence is removed.
        this.removeFinished();
      } catch (error) {
        // Planning/allocation can enqueue offloads before a later allocation
        // fails. Drain them before the batch rollback recycles their pages.
        try {
          await model.ops.synchronizeAsync();
        } catch (syncError) {
          error = syncError;
        }
        console.error("Generation batch error:", error);
        for (const request of this.admitted) {
          request.tokens.end(error instanceof Error ? error : new Error(String(error)));
          metrics.runningRequests--;
        }
        this.admitted.clear();
        this.active = [];
        ws.clearTracking();
        this.prefixes.clear();
        cache.reset(0);
        if (isFatalCudaError(error)) {
          throw error;
        }
      }
    }
    this.removeFinished();
  }
}
