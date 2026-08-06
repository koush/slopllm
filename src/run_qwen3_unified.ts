import { AutoTokenizer } from "@huggingface/transformers";
import fs from "node:fs";
import path from "node:path";
import { createInterface } from "node:readline";
import { CaptureManager } from "./capture-manager";
import { ChatCache, ChatModel, SamplingParams, makeSamplingParams } from "./chat_model";
import { DeviceOps } from "./device_ops";
import { ExecutionWorkspace } from "./execution-workspace";
import { Glm51Model } from "./glm51_model";
import { GlmOps } from "./glm_ops";
import { MetaOps } from "./meta_ops";
import { resolveModelPath } from "./model_path";
import { MtpStats, mtpTreeDecode } from "./mtp";
import { ParallelOps } from "./parallel_ops";
import { Qwen35Model } from "./qwen35_model";
import { Qwen3Model } from "./qwen3_model";
import { SamplingWorkspace, Tensor } from "./tensor";
import { UsingHolder } from "./using-holder";
import { WorkspaceBase } from "./workspace";
import { MemcpyKind } from "./enums";

const QWEN3_REPO = "Qwen/Qwen3-0.6B";
const QWEN3_FP8_REPO = "Qwen/Qwen3-0.6B-FP8";
const QWEN35_REPO = "Qwen/Qwen3.5-0.8B";
const GLM51_REPO = "zai-org/GLM-5.1";

export interface GraphState {
  graphExec: number | null;
  warmupRemaining: number;
}

interface CliArgs {
  gpus: number[];
  maxNewTokens: number;
  maxSeqLen: number;
  warmupSteps: number;
  maxPages: number;
  maxBatch: number;
  noReset: boolean;
  prompt: string | undefined;
  useQwen35: boolean;
  useGlm51: boolean;
  glm51Small: boolean;
  useFp8: boolean;
  useNvfp4: boolean;
  useBatch: boolean;
  modelDir: string | undefined;
  noCudaGraph: boolean;
  temperature: number;
  topP: number;
  topK: number;
  repetitionPenalty: number;
  presencePenalty: number;
  repetitionPenaltyWindow: number;
  greedy: boolean;
  stats: boolean;
  meta: boolean;
  arena: number;
  cp: boolean;
  mtp: boolean;
  mtpDraftTopk: number[];
}

function parseArgs(argv: string[]): CliArgs {
  const gpusEnv = process.env.GLM_GPUS ?? process.env.GLM_GPU ?? "0";
  const args: CliArgs = {
    gpus: gpusEnv.split(",").map(s => parseInt(s.trim(), 10)),
    maxNewTokens: 256,
    maxSeqLen: 4096,
    warmupSteps: 3,
    maxPages: 256,
    maxBatch: 4,
    noReset: true,
    modelDir: undefined,
    prompt: undefined,
    useQwen35: false,
    useGlm51: false,
    glm51Small: false,
    useFp8: false,
    useNvfp4: false,
    useBatch: false,
    noCudaGraph: false,
    temperature: 0.6,
    topP: 0.95,
    topK: 0,
    repetitionPenalty: 1.0,
    presencePenalty: 0,
    repetitionPenaltyWindow: 64,
    greedy: false,
    stats: false,
    meta: false,
    arena: 0,
    cp: false,
    mtp: false,
    mtpDraftTopk: [],
  };

  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--prompt" && i + 1 < argv.length) args.prompt = argv[++i];
    else if (a === "--max-new-tokens" && i + 1 < argv.length) args.maxNewTokens = parseInt(argv[++i], 10);
    else if (a === "--warmup-steps" && i + 1 < argv.length) args.warmupSteps = parseInt(argv[++i], 10);
    else if (a === "--gpus" && i + 1 < argv.length) args.gpus = argv[++i].split(",").map(s => parseInt(s.trim(), 10));
    else if (a === "--gpu" && i + 1 < argv.length) args.gpus = [parseInt(argv[++i], 10)];
    else if (a === "--max-seq-len" && i + 1 < argv.length) args.maxSeqLen = parseInt(argv[++i], 10);
    else if (a === "--max-pages" && i + 1 < argv.length) args.maxPages = parseInt(argv[++i], 10);
    else if (a === "--max-batch" && i + 1 < argv.length) args.maxBatch = parseInt(argv[++i], 10);
    else if (a === "--no-kv-persist") args.noReset = false;
    else if (a === "--qwen35") args.useQwen35 = true;
    else if (a === "--glm51") args.useGlm51 = true;
    else if (a === "--glm51-small") { args.useGlm51 = true; args.glm51Small = true; }
    else if (a === "--model-dir" && i + 1 < argv.length) args.modelDir = argv[++i];
    else if (a === "--fp8") args.useFp8 = true;
    else if (a === "--nvfp4") args.useNvfp4 = true;
    else if (a === "--batch") args.useBatch = true;
    else if (a === "--no-cuda-graph") args.noCudaGraph = true;
    else if (a === "--temperature" && i + 1 < argv.length) args.temperature = parseFloat(argv[++i]);
    else if (a === "--top-p" && i + 1 < argv.length) args.topP = parseFloat(argv[++i]);
    else if (a === "--top-k" && i + 1 < argv.length) args.topK = parseInt(argv[++i], 10);
    else if (a === "--presence-penalty" && i + 1 < argv.length) args.presencePenalty = parseFloat(argv[++i]);
    else if (a === "--repetition-penalty" && i + 1 < argv.length) args.repetitionPenalty = parseFloat(argv[++i]);
    else if (a === "--repetition-penalty-window" && i + 1 < argv.length) args.repetitionPenaltyWindow = parseInt(argv[++i], 10);
    else if (a === "--greedy") {
      args.greedy = true;
    }
    else if (a === "--stats") args.stats = true;
    else if (a === "--meta") args.meta = true;
    else if (a === "--arena" && i + 1 < argv.length) args.arena = parseInt(argv[++i], 10);
    else if (a === "--cp") args.cp = true;
    else if (a === "--mtp") args.mtp = true;
    else if (a === "--mtp-draft-topk" && i + 1 < argv.length) args.mtpDraftTopk = argv[++i].split(",").map(s => parseInt(s.trim(), 10));
  }

  if (args.useQwen35 && args.useFp8) {
    console.error("Error: --fp8 is not supported with --qwen35");
    process.exit(1);
  }
  if (args.useGlm51 && args.useFp8) {
    console.error("Error: --fp8 is not supported with --glm51");
    process.exit(1);
  }
  if (args.useNvfp4 && !args.useGlm51) {
    console.error("Error: --nvfp4 is only supported with --glm51");
    process.exit(1);
  }

  if (args.mtp && args.mtpDraftTopk.length === 0) {
    args.mtpDraftTopk = [2, 2, 2];
  }

  if (args.mtp && args.maxBatch < 8) {
    args.maxBatch = 8;
  }

  if (args.useQwen35 && args.temperature > 0 && args.topP === 0.95 && args.topK === 0 && args.repetitionPenalty === 1.0 && args.presencePenalty === 0) {
    args.topK = 20;
    args.repetitionPenalty = 1.1;
  }

  return args;
}

function modelLabel(args: CliArgs): string {
  if (args.useQwen35) return "Qwen3.5-0.8B";
  if (args.useGlm51) return args.glm51Small ? "GLM-5.1-small" : (args.useNvfp4 ? "GLM-5.1-NVFP4" : "GLM-5.1");
  return args.useFp8 ? "Qwen3-0.6B-FP8" : "Qwen3-0.6B";
}

function tokenizeMessages(
  tokenizer: any,
  messages: Array<{ role: string; content: string }>,
  chatTemplate?: string,
): number[] {
  try {
    const opts: any = {
      tokenize: true,
      add_generation_prompt: true,
      return_tensor: false,
      return_dict: true,
    };
    if (chatTemplate) opts.chat_template = chatTemplate;
    const result = tokenizer.apply_chat_template(messages, opts) as { input_ids: number[] | number[][] };
    return (Array.isArray(result.input_ids[0]) ? result.input_ids[0] : result.input_ids) as number[];
  } catch {
    const text = messages.map(m => `<|${m.role}|>\n${m.content}`).join("\n") + "\n\n\n";
    return tokenizer.encode(text, { add_special_tokens: false });
  }
}

// --- Generation primitives ---

export interface DecodeTiming {
  planMs: number;
  execMs: number;
  idleMs: number;
  warmupSteps: number;
  graphSteps: number;
  warmupTokPerSec: number;
  mtpStats?: MtpStats;
}

export function* generateStream(
  model: ChatModel, ws: ExecutionWorkspace, glm: DeviceOps, cache: ChatCache,
  inputIds: number[], maxNewTokens: number, eosIds: Set<number>,
  sampling: SamplingParams | undefined, graphState?: GraphState,
  timing?: DecodeTiming, mtp?: boolean, mtpDraftTopk?: number[], tokenizer?: any,
): Generator<number> {
  const suffixIds = cache.prefixMatch(0, inputIds);

  using sampleWorkspace = new WorkspaceBase(glm);
  const greedy = !sampling;
  let sampleResult: Tensor | null = null;
  let gpuSampleResult: Tensor | null = null;
  using samplingWorkspace = sampling ? new SamplingWorkspace(glm, 1, model.cfg.vocabSize, sampling!.repetitionPenaltyWindow) : undefined;
  if (samplingWorkspace) samplingWorkspace.updateSampler([sampling!], [inputIds]);
  function doSample(logits: Tensor) {
    if (greedy) {
      using argmax = logits.argmax();
      gpuSampleResult = sampleWorkspace.ensureAlloc(argmax.shape, argmax.type, "gpuSampleResult");
      gpuSampleResult.memcpy(argmax, argmax.bytes, MemcpyKind.DeviceToDevice);
    }
    else {
      // Borrowed, not owned: sample() returns the SamplingWorkspace's
      // persistent outToken, which must survive for the next decode step.
      const sampled = samplingWorkspace!.sample(logits);
      gpuSampleResult = sampleWorkspace.ensureAlloc(sampled.shape, sampled.type, "gpuSampleResult");
      gpuSampleResult.memcpy(sampled, sampled.bytes, MemcpyKind.DeviceToDevice);
    }
  }

  const tokenHistory = inputIds.slice();

  using captureManager = new CaptureManager(glm);
  const topks = (mtp && model.forwardMtp && mtpDraftTopk && mtpDraftTopk.length > 0) ? mtpDraftTopk : [];
  const mtpStats = topks.length > 0 ? new MtpStats(topks.length) : undefined;
  if (timing && mtpStats) timing.mtpStats = mtpStats;
  using mtpHiddenStates = new UsingHolder<Tensor>(undefined!);
  let currentToken: number;

  captureManager.disabled = graphState === undefined;

  // needed by mtp
  using sharedSlots = new UsingHolder<Tensor>(undefined!);
  using sharedSlotsLength = new UsingHolder<Tensor>(undefined!);
  {
    const inputIdsList = [suffixIds];
    const batchSize = inputIdsList.length;
    const seqLens = inputIdsList.map(ids => ids.length);
    const state = ws.planPrefill(model, batchSize, seqLens, cache);
    state.sharedSlots = sharedSlots;
    state.sharedSlotsLength = sharedSlotsLength;
    state.setInput(inputIdsList);
    using hiddenStates = model.forward(state);
    using firstTokens = state.computeLogits(hiddenStates, model);
    doSample(firstTokens);

    if (mtp && model.forwardMtp && topks.length > 0) {
      // MTP convention: at position P, the input token is the token at P+1 (not P),
      // paired with the target model's hidden state at P. This means the MTP KV entry
      // at position P encodes info about token P+1, whereas the target model KV at the
      // same position encodes token P. This is safe because each MLA layer has its own
      // KV slot — the two never interfere.
      using rotatedInputIds = state.input!.rotateInputIds(state.qoIndptrD, gpuSampleResult!, state.batchSize);
      state.setInput(rotatedInputIds);
      using _mtpHiddenStates = model.forwardMtp(state, hiddenStates, true);

      const maxTopK = Math.max(...topks);
      const maxTopKShape = [maxTopK, ..._mtpHiddenStates.shape.slice(1)];
      const mtpHiddenStatesMaxTopK = ws.alloc(maxTopKShape, _mtpHiddenStates.type);
      using sliced = _mtpHiddenStates.slice(0, -1, 1);
      mtpHiddenStatesMaxTopK.memcpy(sliced, sliced.bytes, MemcpyKind.DeviceToDevice);

      mtpHiddenStates.replace(mtpHiddenStatesMaxTopK.removeTracking());
    }
  }

  sampleResult = sampleWorkspace.ensureAllocPinned(gpuSampleResult!.shape, gpuSampleResult!.type, "sampleResult");
  sampleResult.memcpy(gpuSampleResult!, gpuSampleResult!.bytes, MemcpyKind.DeviceToHost);
  glm.synchronize();
  currentToken = sampleResult!.readPinnedBuffer().readInt32LE();
  cache.reportTokens(0, suffixIds);
  cache.reportTokens(0, [currentToken]);
  tokenHistory.push(currentToken);
  yield currentToken;
  if (eosIds.has(currentToken)) return;

  // Budget is in TOKENS, not loop iterations. One plain decode step emits one
  // token, but one MTP step emits 1 + (accepted drafts), so bounding the loop
  // counter overshoots by the mean acceptance length (~2.2x at topk 1,1,1).
  let generated = 1;
  let planMs = 0;
  let execMs = 0;
  let idleMs = 0;
  let warmupSteps = 0;
  let graphSteps = 0;
  let firstPostWarmupTime = 0;
  let lastTokenTime = 0;
  let postWarmupTokenCount = 0;
  let tAfterSync = 0;

  try {
    for (let i = 1; generated < maxNewTokens; i++) {
      using _tracking = sampleWorkspace.startTracking();

      if (mtp && model.forwardMtp && topks.length > 0) {
        if (process.env.GLM_STEP_LOG === '1') process.stderr.write(`[step ${i}] seqLen=${cache.getPagedKV().sequences[0].allocLen} histLen=${tokenHistory.length}\n`);
        const { warmup, tokens, numAccepted, numDraftTokens } = mtpTreeDecode(captureManager, model, mtpHiddenStates.value, sharedSlots.value, sharedSlotsLength.value, ws, currentToken, topks, cache, tokenizer);
        if (mtpStats && !warmup) mtpStats.observe(numDraftTokens, numAccepted);
        // glm.synchronize();
        for (const t of tokens) {
          currentToken = t;
          cache.reportTokens(0, [t]);
          tokenHistory.push(t);

          const now = performance.now();
          if (firstPostWarmupTime === 0)
            firstPostWarmupTime = now;
          lastTokenTime = now;

          if (warmup) {
            warmupSteps++;
            firstPostWarmupTime = 0;
            postWarmupTokenCount = 0;
          }
          else {
            graphSteps++;
            postWarmupTokenCount++;
          }

          yield t;
          generated++;
          if (eosIds.has(t))
            return;
          // Stop mid-batch: the rest of this step's accepted tokens are over
          // budget. Their KV is already in the cache, exactly as on the EOS
          // path above -- reportTokens has run for them but they are not
          // yielded, so the caller never sees them.
          if (generated >= maxNewTokens)
            return;
        }

        continue;
        // console.log(`MTP accepted=${verifyResult.numAccepted}/${topks.length} replacement=${tokenizer?.decode([verifyResult.replacementToken]) ?? verifyResult.replacementToken}`);
        // if (verifyResult.acceptedTokens.length > 0) {
        //   console.log(`MTP accepted tokens: ${verifyResult.acceptedTokens.map(t => tokenizer?.decode([t]) ?? `?${t}`).join(" ")}`);
        // }
      }

      using _tracking2 = ws.startTracking();

      const tPlan = performance.now();
      let isPostWarmupToken = false;
      if (true) {
        const state = ws.planDecode(model, 1, cache, !captureManager.disabled);
        state.setInput([[currentToken]]);
        planMs += performance.now() - tPlan;

        const isCaptured = state.isCaptured(captureManager, ['decode']);
        isPostWarmupToken = captureManager.disabled || isCaptured;
        if (!isCaptured) {
          warmupSteps++;
        }
        else {
          graphSteps++;
        }

        state.capture(captureManager, {}, () => {
          ws.positionStep(state, model);
          using hiddenStates = model.forwardModel(state);
          doSample(state.computeLogits(hiddenStates, model));
        }, ['decode']);

      }
      else {
        const state = ws.planPrefill(model, 1, [1], cache);
        state.setInput([[currentToken]]);

        const isCaptured = state.isCaptured(captureManager, ['decode']);
        isPostWarmupToken = captureManager.disabled || isCaptured;
        if (!isCaptured) {
          warmupSteps++;
        }
        else {
          graphSteps++;
        }

        state.capture(captureManager, {}, () => {
          using hiddenStates = model.forward(state);
          using tokens = state.computeLogits(hiddenStates, model);
          doSample(tokens);
        }, ['decode']);
      }
      const tExec = performance.now();

      sampleResult ||= sampleWorkspace.allocPinned(gpuSampleResult!.shape, gpuSampleResult!.type);
      sampleResult.memcpy(gpuSampleResult!, gpuSampleResult!.bytes, MemcpyKind.DeviceToHost);
      glm.synchronize();

      currentToken = sampleResult!.readPinnedBuffer().readInt32LE();

      execMs += performance.now() - tExec;
      tAfterSync = performance.now();
      cache.reportTokens(0, [currentToken]);
      tokenHistory.push(currentToken);

      yield currentToken;
      generated++;
      if (eosIds.has(currentToken))
        return;

      if (isPostWarmupToken) {
        const now = performance.now();
        if (firstPostWarmupTime === 0) firstPostWarmupTime = now;
        lastTokenTime = now;
        postWarmupTokenCount++;
      }
    }
  } finally {
    if (timing) {
      timing.planMs = planMs;
      timing.execMs = execMs;
      timing.idleMs = idleMs;
      timing.warmupSteps = warmupSteps;
      timing.graphSteps = graphSteps;
      timing.warmupTokPerSec = (postWarmupTokenCount > 1 && firstPostWarmupTime > 0)
        ? postWarmupTokenCount / ((lastTokenTime - firstPostWarmupTime) / 1000)
        : 0;
    }
  }
}

export function generateBatchTokens(
  model: ChatModel, ws: ExecutionWorkspace, cache: ChatCache,
  inputIdsList: number[][], maxNewTokens: number, eosIds: Set<number>,
): number[][] {
  const batchSize = inputIdsList.length;
  cache.reset(batchSize);
  const firstTokens = ws.forwardEagerPrefill(model, inputIdsList, cache);

  const nextTokens = [...firstTokens];
  const generated: number[][] = nextTokens.map(t => [t]);
  const finished = nextTokens.map(t => eosIds.has(t));

  for (let step = 0; step < maxNewTokens - 1; step++) {
    if (finished.every(f => f)) break;

    const newTokens = ws.forwardEagerDecode(model, nextTokens, cache);

    for (let i = 0; i < batchSize; i++) {
      nextTokens[i] = newTokens[i];
      if (!finished[i]) {
        if (eosIds.has(newTokens[i])) {
          finished[i] = true;
        } else {
          generated[i].push(newTokens[i]);
        }
      }
    }
  }

  return generated;
}

// --- Interactive / single-prompt modes ---

async function interactiveChat(
  model: ChatModel, ws: ExecutionWorkspace, glm: DeviceOps, cache: ChatCache,
  tokenizer: any, args: CliArgs, graphState: GraphState | undefined, chatTemplate?: string,
): Promise<void> {
  const sp = makeSamplingParams(args);
  const eosIds = model.eosIds;
  const messages: Array<{ role: string; content: string }> = [];

  console.log("Type /quit to exit, /clear to reset conversation\n");

  const rl = createInterface({ input: process.stdin, output: process.stdout });
  const askLine = (prompt: string): Promise<string> =>
    new Promise((resolve) => rl.question(prompt, resolve));

  try {
    while (true) {
      const userInput = (await askLine("> ")).trim();
      if (!userInput) continue;
      if (userInput.toLowerCase() === "quit" || userInput.toLowerCase() === "exit" || userInput.toLowerCase() === "/q") break;
      if (userInput.toLowerCase() === "/clear") {
        messages.length = 0;
        if (graphState) graphState.graphExec = null;
        cache.reset(1);
        console.log("Conversation cleared.\n");
        continue;
      }

      messages.push({ role: "user", content: userInput });
      const inputIds = tokenizeMessages(tokenizer, messages, chatTemplate);

      if (inputIds.length > args.maxSeqLen - args.maxNewTokens) {
        console.log(`Warning: prompt (${inputIds.length} tokens) too long, truncating conversation`);
        while (inputIds.length > args.maxSeqLen - args.maxNewTokens && messages.length > 1) {
          messages.splice(1, 2);
          const retryIds = tokenizeMessages(tokenizer, messages, chatTemplate);
          if (retryIds.length <= args.maxSeqLen - args.maxNewTokens) break;
        }
        if (messages.length === 1 && tokenizeMessages(tokenizer, messages, chatTemplate).length > args.maxSeqLen - args.maxNewTokens) {
          console.log("Conversation too long even after truncation. Use /clear to reset.");
          messages.pop();
          continue;
        }
      }

      process.stdout.write("Assistant: ");
      const t0 = performance.now();
      let tokCount = 0;
      const generatedIds: number[] = [];
      const timing: DecodeTiming = { planMs: 0, execMs: 0, idleMs: 0, warmupSteps: 0, graphSteps: 0, warmupTokPerSec: 0 };

      for (const tokenId of generateStream(model, ws, glm, cache, inputIds, args.maxNewTokens, eosIds, sp, graphState, timing, args.mtp, args.mtpDraftTopk, tokenizer)) {
        generatedIds.push(tokenId);
        tokCount++;
        const chunk = tokenizer.decode([tokenId], { skip_special_tokens: false });
        process.stdout.write(chunk);
        if (eosIds.has(tokenId)) break;
      }

      const elapsed = performance.now() - t0;
      console.log(`\n  [${tokCount} tokens in ${(elapsed / 1000).toFixed(1)}s, ${(tokCount / (elapsed / 1000)).toFixed(1)} tok/s]`);
      console.log(`  timing: plan=${timing.planMs.toFixed(1)}ms exec=${timing.execMs.toFixed(1)}ms idle=${timing.idleMs.toFixed(1)}ms (warmup=${timing.warmupSteps} graph=${timing.graphSteps}) decode=${timing.warmupTokPerSec.toFixed(1)} tok/s`);
      if (timing.mtpStats) console.log(`  ${timing.mtpStats.log()}`);

      const responseText = tokenizer.decode(generatedIds.filter(t => !eosIds.has(t)), { skip_special_tokens: true });
      messages.push({ role: "assistant", content: responseText });
    }
  } finally {
    if (graphState?.graphExec !== null && graphState?.graphExec !== undefined) glm.graphExecDestroy(graphState.graphExec);
    rl.close();
  }
}

async function singlePrompt(
  model: ChatModel, ws: ExecutionWorkspace, glm: DeviceOps, cache: ChatCache,
  tokenizer: any, args: CliArgs, graphState: GraphState | undefined, chatTemplate?: string,
): Promise<void> {
  const sp = !args.greedy ? makeSamplingParams(args) : undefined;
  const eosIds = model.eosIds;
  const messages = [{ role: "user", content: args.prompt! }];
  const inputIds = tokenizeMessages(tokenizer, messages, chatTemplate);

  console.log(`Prompt: ${args.prompt}`);
  console.log(`Tokens: ${inputIds.length}`);

  process.stdout.write("\n");
  const t0 = performance.now();
  let tokCount = 0;
  const generatedIds: number[] = [];
  const timing: DecodeTiming = { planMs: 0, execMs: 0, idleMs: 0, warmupSteps: 0, graphSteps: 0, warmupTokPerSec: 0 };

  for (const tokenId of generateStream(model, ws, glm, cache, inputIds, args.maxNewTokens, eosIds, sp, graphState, timing, args.mtp, args.mtpDraftTopk, tokenizer)) {
    generatedIds.push(tokenId);
    tokCount++;
    const chunk = tokenizer.decode([tokenId], { skip_special_tokens: false });
    process.stdout.write(chunk);
    if (eosIds.has(tokenId)) break;
  }

  const elapsed = performance.now() - t0;
  console.log(`\n\n${tokCount} tokens in ${elapsed.toFixed(1)}ms (${(tokCount / (elapsed / 1000)).toFixed(1)} tok/s)`);
  console.log(`timing: plan=${timing.planMs.toFixed(1)}ms exec=${timing.execMs.toFixed(1)}ms idle=${timing.idleMs.toFixed(1)}ms (warmup=${timing.warmupSteps} graph=${timing.graphSteps}) decode=${timing.warmupTokPerSec.toFixed(1)} tok/s`);
  if (timing.mtpStats) console.log(timing.mtpStats.log());

  if (graphState?.graphExec !== null && graphState?.graphExec !== undefined) glm.graphExecDestroy(graphState.graphExec);
}

// --- Batch mode ---

async function interactiveBatch(
  model: ChatModel, ws: ExecutionWorkspace, cache: ChatCache,
  tokenizer: any, args: CliArgs, chatTemplate?: string,
): Promise<void> {
  console.log("Enter prompts one per line. Empty line to submit batch. /clear to reset, /q to quit.");

  const rl = createInterface({ input: process.stdin, output: process.stdout });
  const askLine = (prompt: string): Promise<string> =>
    new Promise((resolve) => rl.question(prompt, resolve));

  try {
    while (true) {
      const prompts: string[] = [];
      while (true) {
        const line = (await askLine(`\nPrompt ${prompts.length + 1} (empty=done): `)).trim();
        if (line === "") break;
        if (line.toLowerCase() === "quit" || line.toLowerCase() === "exit" || line.toLowerCase() === "/q") {
          rl.close();
          return;
        }
        if (line.toLowerCase() === "/clear") {
          prompts.length = 0;
          console.log("Batch cleared.");
          continue;
        }
        prompts.push(line);
      }

      if (prompts.length === 0) continue;
      if (prompts.length > args.maxBatch) {
        console.log(`Too many prompts (${prompts.length}), max batch is ${args.maxBatch}. Reducing to first ${args.maxBatch}.`);
        prompts.length = args.maxBatch;
      }

      const inputIdsList: number[][] = [];
      for (const prompt of prompts) {
        const messages = [{ role: "user" as const, content: prompt }];
        const ids = tokenizeMessages(tokenizer, messages, chatTemplate);
        inputIdsList.push(ids);
      }

      const maxPromptLen = Math.max(...inputIdsList.map(ids => ids.length));
      if (maxPromptLen > args.maxSeqLen) {
        console.log(`Longest prompt (${maxPromptLen} tokens) exceeds max_seq_len (${args.maxSeqLen}). Skipping.`);
        continue;
      }

      const start = Date.now();
      const generatedIds = generateBatchTokens(model, ws, cache, inputIdsList, args.maxNewTokens, model.eosIds);
      const elapsed = (Date.now() - start) / 1000;
      const totalTokens = generatedIds.reduce((sum: number, ids: number[]) => sum + ids.length, 0);

      for (let i = 0; i < prompts.length; i++) {
        const text = tokenizer.decode(generatedIds[i], { skip_special_tokens: true });
        console.log(`\n--- Response ${i + 1} ---`);
        console.log(text);
      }

      console.log(`\n  [${prompts.length} prompts, ${totalTokens} tokens, ${elapsed.toFixed(1)}s, ${(totalTokens / elapsed).toFixed(1)} tok/s]`);
    }
  } finally {
    rl.close();
  }
}

// --- Main ---

async function main(): Promise<void> {
  Error.stackTraceLimit = 20; 

  const args = parseArgs(process.argv.slice(2));

  const GLM51_SMALL_BF16 = "tests/python/test_models/glm51_small/glm51_small_bf16";
  const GLM51_SMALL_NVFP4 = "tests/python/test_models/glm51_small/glm51_small_nvfp4";

  const modelDir = args.modelDir ?? (args.useGlm51
    ? (args.glm51Small
      ? (args.useNvfp4 ? GLM51_SMALL_NVFP4 : GLM51_SMALL_BF16)
      : '/mnt/storage/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/2eff962076815828e4031aec2834ac6e22fb4434/')
    : resolveModelPath(args.useQwen35 ? QWEN35_REPO : (args.useFp8 ? QWEN3_FP8_REPO : QWEN3_REPO)));

  if (args.meta) {
    const metaOps = new MetaOps();
    const model: ChatModel = args.useGlm51
      ? await Glm51Model.fromPretrained(metaOps, modelDir, args.cp, args.mtp)
      : args.useQwen35
        ? await Qwen35Model.fromPretrained(metaOps, modelDir)
        : await Qwen3Model.fromPretrained(metaOps, modelDir);
    const loadAllocs = metaOps.totalAllocs;
    const loadBytes = metaOps.totalBytes;

    const cache = model.createChatCache(args.maxPages, args.maxBatch, args.maxSeqLen);
    const ws = new ExecutionWorkspace(metaOps, args.maxBatch, args.maxSeqLen);
    const inputIds = [1, 2, 3, 4, 5];
    const logits = ws.forwardPrefill(model, [inputIds], cache);
    cache.reportTokens(0, inputIds);
    const forwardAllocs = metaOps.totalAllocs;
    const forwardBytes = metaOps.totalBytes;

    const mb = (b: number) => (b / (1024 * 1024)).toFixed(1);
    console.log(`Meta mode: ${modelLabel(args)}`);
    console.log(`  model weight allocs: ${loadAllocs} (${mb(loadBytes)} MB)`);
    console.log(`  + workspace/cache allocs: ${forwardAllocs - loadAllocs} (${mb(forwardBytes - loadBytes)} MB)`);
    console.log(`  total allocs: ${forwardAllocs} (${mb(forwardBytes)} MB)`);
    return;
  }

  const gpuDevices = args.gpus.map(id => new GlmOps(id, undefined, args.arena || undefined));
  const glm: DeviceOps = gpuDevices.length > 1
    ? new ParallelOps(gpuDevices)
    : gpuDevices[0];

  const gpuLabel = args.gpus.join(",");

  const repoId = args.useGlm51 ? GLM51_REPO
    : args.useQwen35 ? QWEN35_REPO
      : (args.useFp8 ? QWEN3_FP8_REPO : QWEN3_REPO);

  const model: ChatModel = args.useGlm51
    ? await Glm51Model.fromPretrained(glm, modelDir, args.cp, args.mtp)
    : args.useQwen35
      ? await Qwen35Model.fromPretrained(glm, modelDir)
      : await Qwen3Model.fromPretrained(glm, modelDir);

  const cache = model.createChatCache(args.maxPages, args.maxBatch, args.maxSeqLen);
  const ws = new ExecutionWorkspace(glm, args.maxBatch, args.maxSeqLen);

  // Prefer the *resolved* model directory, not just an explicit --model-dir.
  // The served checkpoint ships its own chat_template.jinja, and for GLM-5.x
  // that template emits a "<|system|>Reasoning Effort: ..." prompt that governs
  // how long the model thinks before closing </think>. Falling back to the
  // tokenizer repo's older template drops that system prompt entirely, which
  // leaves the model off-distribution: it either stubs out after a few hundred
  // tokens or never closes the thinking block at all.
  const tokenizerDir = fs.existsSync(path.join(modelDir, "tokenizer_config.json"))
    ? modelDir : resolveModelPath(repoId);
  const tokenizer = await AutoTokenizer.from_pretrained(tokenizerDir, { local_files_only: true });
  const chatTemplatePath = path.join(tokenizerDir, "chat_template.jinja");
  const chatTemplate = fs.existsSync(chatTemplatePath) ? fs.readFileSync(chatTemplatePath, "utf-8") : undefined;

  const sp = makeSamplingParams(args);
  const samplingParts: string[] = [];
  if (sp.temperature > 0) samplingParts.push(`temp=${sp.temperature}`);
  if (sp.topP < 1.0) samplingParts.push(`top_p=${sp.topP}`);
  if (sp.topK > 0) samplingParts.push(`top_k=${sp.topK}`);
  if (sp.repetitionPenalty !== 1.0) samplingParts.push(`rep_pen=${sp.repetitionPenalty}`);
  if (sp.presencePenalty !== 0) samplingParts.push(`pres_pen=${sp.presencePenalty}`);
  const samplingStr = !args.greedy ? samplingParts.join(" ") : "greedy";

  const arenaStr = args.arena ? `  |  arena=${args.arena}GB` : "";
  const mtpStr = args.mtp ? `  |  mtp=${args.mtpDraftTopk.join(',')}` : "";
  console.log(`${modelLabel(args)}  |  GPU${args.gpus.length > 1 ? "s" : ""} ${gpuLabel}  |  max_seq_len=${args.maxSeqLen}  |  max_tokens=${args.maxNewTokens}  |  ${args.useBatch ? `batch=${args.maxBatch}` : (args.noCudaGraph ? "cuda_graph=off" : `cuda_graph=on(warmup=${args.warmupSteps})`)}  |  ${samplingStr}${arenaStr}${mtpStr}`);

  const cleanup = () => {
    glm.synchronize();
    cache.free();
    ws.free();
    model.free();
    if (glm instanceof ParallelOps) glm.free();
    for (const d of gpuDevices) d.free();
  };

  if (args.useBatch) {
    await interactiveBatch(model, ws, cache, tokenizer, args, chatTemplate);
  } else {
    const graphState = args.noCudaGraph ? undefined : { graphExec: null as number | null, warmupRemaining: args.warmupSteps };

    if (args.prompt) {
      await singlePrompt(model, ws, glm, cache, tokenizer, args, graphState, chatTemplate);
    } else {
      await interactiveChat(model, ws, glm, cache, tokenizer, args, graphState, chatTemplate);
    }
  }

  cleanup();
}

// Only run as a CLI. This module also exports generateStream, and without the
// guard any `import { generateStream } from "./run_qwen3_unified"` would load a
// whole second model as a side effect of the import.
if (require.main === module) {
  main().catch((err) => {
    console.error("Failed:", err);
    process.exit(1);
  });
}
