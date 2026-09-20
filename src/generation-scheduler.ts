import { createAsyncQueue } from "@scrypted/deferred";
import type { CaptureManager } from "./capture-manager";
import type { ChatCache, ChatModel, SamplingParams, TokenSelector } from "./chat_model";
import { executePlan, type ExecutionPhase, type ExecutionWorkspace } from "./execution-workspace";
import { mtpTotalTreeNodes } from "./glm51_model";
import { PhasedPrefillRunner, splitRaggedInput } from "./phased-prefill";
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

class StagedPrefixPolicy {
  private nextKey = -1;
  private readonly retainedKeys = new Set<number>();

  constructor(private readonly pagedKV: ReturnType<ChatCache["getPagedKV"]>) {
    pagedKV.onPagePressure = requiredPages => this.evictUntilAvailable(requiredPages);
  }

  retainSequence(index: number): void {
    const key = this.nextKey--;
    this.pagedKV.stageSequence(index, key);
    this.retainedKeys.add(key);
  }

  clear(): void {
    this.pagedKV.clearStaging();
    this.retainedKeys.clear();
  }

  private evictUntilAvailable(requiredPages: number): void {
    while (this.pagedKV.availablePages.length < requiredPages && this.retainedKeys.size) {
      let longestKey: number | undefined;
      let longestPages = 0;
      for (const key of this.retainedKeys) {
        const sequence = this.pagedKV.staging.get(key);
        if (!sequence || !sequence.pages.length) {
          this.retainedKeys.delete(key);
          if (sequence) {
            this.pagedKV.removeStagedSequence(key);
          }
        } else if (sequence.pages.length > longestPages) {
          longestKey = key;
          longestPages = sequence.pages.length;
        }
      }
      if (longestKey === undefined) {
        break;
      }
      const sequence = this.pagedKV.staging.get(longestKey)!;
      sequence.popPage();
      if (!sequence.pages.length) {
        this.retainedKeys.delete(longestKey);
        this.pagedKV.removeStagedSequence(longestKey);
      }
    }
  }
}

interface ActiveSequence {
  request: GenerationRequest;
  lastToken: number;
}

interface SchedulerOptions {
  requests: ReturnType<typeof createAsyncQueue<GenerationRequest>>;
  model: ChatModel;
  ws: ExecutionWorkspace;
  cache: ChatCache;
  captureManager: CaptureManager;
  samplingWorkspace: SamplingWorkspace;
  metrics: ServerMetrics;
  maxBatchSize: number;
  chunkSize: number;
  decodeLatencyMs: number;
  phasedPrefill: boolean;
  topks?: readonly number[];
}

/** Owns batch membership and GPU execution; HTTP handlers own token consumption. */
export class GenerationScheduler {
  private active: ActiveSequence[] = [];
  private readonly admitted = new Set<GenerationRequest>();
  private readonly prefixes: StagedPrefixPolicy;
  private nextStagingKey = 0;

  constructor(private readonly options: SchedulerOptions) {
    this.prefixes = new StagedPrefixPolicy(options.cache.getPagedKV());
  }

  stop(error = new Error("Server shutting down")): void {
    this.options.requests.end();
    for (const request of this.options.requests.clear()) {
      request.tokens.end(error);
    }
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
      const { request } = this.active[index];
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
    const { metrics, topks } = this.options;
    if (!topks) {
      return;
    }
    const key = `${name}|${batchSize}`;
    metrics.mtpPhaseSeconds.set(key, (metrics.mtpPhaseSeconds.get(key) ?? 0) + seconds);
    metrics.mtpPhaseCount.set(key, (metrics.mtpPhaseCount.get(key) ?? 0) + 1);
  }

  private observePhase = (phase: ExecutionPhase, seconds: number): void => {
    if (phase.timingName) {
      this.recordPhase(phase.timingName, phase.states[0]?.batchSize ?? 0, seconds);
    }
  };

  /** Only called at a generator boundary, with active rows in cache order. */
  private prepareSampling(bootstrap: boolean): TokenSelector | undefined {
    const { samplingWorkspace: sampler, cache, topks } = this.options;
    const params = this.active.map(row => row.request.samplingParams);
    if (topks) {
      if (sampler.mtpEnabled) {
        sampler.updateMtpSampler(params);
      }
      if (bootstrap || !sampler.mtpEnabled) {
        const targetParams = bootstrap ? params : params.flatMap(param =>
          Array.from({ length: mtpTotalTreeNodes(topks) + 1 }, () => param));
        sampler.updateSampler(targetParams, targetParams.map(() => []));
      }
      return sampler;
    }
    if (params.every(param => param.temperature <= 0 && param.repetitionPenalty === 1 && param.presencePenalty === 0)) {
      return undefined;
    }
    sampler.updateSampler(params, cache.getPagedKV().sequences.map(sequence => sequence.getTokenIds()));
    return sampler;
  }

  private async runChunk(inputIds: number[][], nextTokens: number[], sample = false): Promise<number[] | undefined> {
    const { model, ws, cache, captureManager, phasedPrefill, topks } = this.options;
    let firstTokens: number[] | undefined;
    const split = phasedPrefill ? splitRaggedInput(inputIds) : undefined;
    if (split) {
      const start = performance.now();
      const runner = new PhasedPrefillRunner(model, model.glm);
      if (topks) {
        using a = model.planPrefillMtpChunkPhased!(ws, cache, split.inputA, split.nextA);
        using b = model.planPrefillMtpChunkPhased!(ws, cache, split.inputB, nextTokens);
        runner.runPlanPair(a, b);
      } else {
        const a = ws.planPrefill(model, split.inputA.length, split.inputA.map(ids => ids.length), cache);
        a.setInput(split.inputA);
        const b = ws.planPrefill(model, split.inputB.length, split.inputB.map(ids => ids.length), cache);
        b.setInput(split.inputB);
        runner.runPair(a, b, (_stateA, _hiddenA, stateB, hiddenB) => {
          if (sample) {
            using logits = stateB.computeLogits(hiddenB, model);
            using selected = logits.argmax();
            firstTokens = selected.readInt32LEArray();
          }
        });
      }
      await model.glm.synchronizeAsync();
      this.recordPhase("prefill_chunk", inputIds.length, (performance.now() - start) / 1000);
    } else if (topks) {
      await executePlan(captureManager, ws, model.planPrefillMtpChunk!(ws, cache, inputIds, nextTokens), this.observePhase);
    } else {
      using logits = ws.forwardPrefill(model, inputIds, cache);
      if (sample) {
        using selected = logits.argmax();
        firstTokens = selected.readInt32LEArray();
      }
      await model.glm.synchronizeAsync();
    }
    ws.assertClear();
    ws.clearTracking();
    return firstTokens;
  }

  private admit(requests: GenerationRequest[]): void {
    const { cache, metrics, topks } = this.options;
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
    for (const [index, request] of requests.entries()) {
      request.inputIds = cache.prefixMatch(index, request.inputIds);
      request.cachedTokenCount = request.promptTokenCount - request.inputIds.length;
      request.prefillTokenCount = request.inputIds.length + (topks && request.cachedTokenCount > 0 ? 1 : 0);
    }
    for (const entry of saved) {
      pagedKV.unstageSequence(entry.key);
    }
    this.active = [...requests.map(request => ({ request, lastToken: -1 })), ...saved.map(entry => entry.row)];
    console.log(`Prefill: requests=${requests.length} tokens=${requests.reduce((sum, request) => sum + request.inputIds.length, 0)}`);
  }

  /** Run one chunk, leaving unfinished input on its request for the next scheduler turn. */
  private async prefill(): Promise<boolean> {
    const { model, ws, cache, captureManager, samplingWorkspace, metrics, topks, chunkSize } = this.options;
    const pending = this.active.filter(row => row.request.inputIds.length);
    if (!pending.length) {
      return true;
    }
    const pagedKV = cache.getPagedKV();
    const start = performance.now();
    // prepareMtpInput may prepend an overlap token to every row, including decode rows.
    const finalSize = pending.reduce((sum, row) => sum + row.request.inputIds.length, 0)
      + (topks ? this.active.length : 0);
    if (!topks || finalSize > chunkSize) {
      // Take one chunk. Only MTP reserves a boundary token for final prefill.
      const staged = this.active.map(row => ({ row, key: this.nextStagingKey++ }));
      for (const entry of staged) {
        pagedKV.stageSequence(0, entry.key);
      }
      let budget = chunkSize;
      const inputIds: number[][] = [];
      const nextTokens: number[] = [];
      const chunked: ActiveSequence[] = [];
      const deferred: typeof staged = [];
      for (const entry of staged) {
        const { request } = entry.row;
        const take = Math.min(Math.max(0, request.inputIds.length - (topks ? 1 : 0)), budget);
        if (!take) {
          deferred.push(entry);
          continue;
        }
        pagedKV.unstageSequence(entry.key);
        inputIds.push(request.inputIds.slice(0, take));
        request.inputIds = request.inputIds.slice(take);
        if (topks) {
          nextTokens.push(request.inputIds[0]);
        }
        chunked.push(entry.row);
        budget -= take;
      }
      if (!chunked.length) {
        throw new Error(`Unable to fit prefill batch within ${chunkSize} tokens`);
      }
      const sample = !topks && chunked.some(row => !row.request.inputIds.length);
      const firstTokens = await this.runChunk(inputIds, nextTokens, sample);
      inputIds.forEach((ids, index) => cache.reportTokens(index, ids));
      for (const entry of deferred) {
        pagedKV.unstageSequence(entry.key);
      }
      this.active = [...chunked, ...deferred.map(entry => entry.row)];
      const seconds = (performance.now() - start) / 1000;
      for (const { request } of pending) {
        request.prefillSeconds += seconds;
      }
      // Non-MTP rows finish independently; their first generated token seeds decode.
      chunked.forEach((row, index) => {
        const { request } = row;
        if (request.inputIds.length) {
          return;
        }
        metrics.prefillTimeSecondsCount++;
        metrics.prefillTimeSecondsSum += request.prefillSeconds;
        request.decodeStartedAt = performance.now();
        row.lastToken = firstTokens![index];
        cache.reportTokens(index, [row.lastToken]);
        this.publish(request, [row.lastToken]);
      });
      return this.active.every(row => !row.request.inputIds.length);
    }
    // MTP final prefill: consume remaining input and recondition existing decode rows.
    const prefilled = this.active;
    const input = model.prepareMtpInput(cache, prefilled.map(row => row.request.inputIds));
    this.prepareSampling(true);
    const firstTokens = (await executePlan(captureManager, ws,
      model.planPrefillMtp!(ws, cache, input, samplingWorkspace), this.observePhase)).result.targetTokens;
    const seconds = (performance.now() - start) / 1000;
    metrics.prefillTimeSecondsCount += pending.length;
    for (const { request } of pending) {
      request.prefillSeconds += seconds;
      metrics.prefillTimeSecondsSum += request.prefillSeconds;
      request.decodeStartedAt = performance.now();
    }
    firstTokens.forEach((token, index) => {
      const { request } = prefilled[index];
      if (request.inputIds.length) {
        cache.reportTokens(index, request.inputIds);
      }
      request.inputIds = [];
      cache.reportTokens(index, [token]);
      prefilled[index].lastToken = token;
      this.publish(request, [token]);
    });
    return true;
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
    const { requests, model, ws, cache, captureManager, metrics, maxBatchSize, topks } = this.options;
    cache.reset(0);
    while (!requests.ended) {
      // Admission: remove terminated rows, wait if empty, then fill available slots.
      this.removeFinished();
      const incoming: GenerationRequest[] = [];
      if (!this.active.length) {
        try {
          const request = await requests.dequeue();
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
      while (this.active.length + incoming.length < maxBatchSize) {
        const request = requests.take();
        if (!request) {
          break;
        }
        if (!request.tokens.ended) {
          incoming.push(request);
        }
      }
      try {
        if (incoming.length) {
          this.admit(incoming);
        }
        // Prefill: run one chunk, then revisit admission if more input or batch changes remain.
        const finalPrefill = await this.prefill();
        await this.yieldToRequests();
        if (!finalPrefill || requests.ended || this.active.some(row => row.request.tokens.ended)
          || (this.active.length < maxBatchSize && requests.queued.length > 0)) {
          continue;
        }
        if (!this.active.length) {
          continue;
        }
        // Decode: keep the batch stable until termination or a pending admission.
        const samplingPolicy = this.prepareSampling(true);
        const targetTokens = this.active.map(row => row.lastToken);
        const generator = topks
          ? model.generateMtpDecode!(ws, cache, targetTokens, topks, captureManager, samplingPolicy)
          : model.generateDecode(ws, cache, targetTokens, captureManager, samplingPolicy);
        let stepStart = performance.now();
        for await (const step of generator) {
          if (process.env.GLM_MTP_TIMING === "1") {
            this.recordPhase("decode", this.active.length, (performance.now() - stepStart) / 1000);
          }
          if (step.numDraftTokens > 0) {
            metrics.specDecodeNumDraftsTotal += this.active.length;
          }
          metrics.specDecodeNumDraftTokensTotal += step.numDraftTokens * this.active.length;
          metrics.specDecodeNumAcceptedTokensTotal += step.numAccepted.reduce((sum, count) => sum + count, 0);
          step.tokens.forEach((tokens, index) => {
            cache.reportTokens(index, tokens);
            this.active[index].lastToken = tokens.at(-1)!;
            this.publish(this.active[index].request, tokens);
          });
          await this.yieldToRequests();
          if (requests.ended || this.active.some(row => row.request.tokens.ended)
            || (this.active.length < maxBatchSize && requests.queued.length > 0)) {
            break;
          }
          this.prepareSampling(false);
          stepStart = performance.now();
        }
        // for-await has closed the generator before any sequence is removed.
        this.removeFinished();
      } catch (error) {
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
