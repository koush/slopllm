import { CaptureManager } from "./capture-manager";
import { ChatModel } from "./chat_model";
import { DeviceOps } from "./device_ops";
import { ExecutionWorkspace } from "./execution-workspace";
import { Glm51Model } from "./glm51_model";
import { GlmOps } from "./glm_ops";
import { ParallelOps } from "./parallel_ops";

const GLM51_MODEL_DIR = '/mnt/storage/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/2eff962076815828e4031aec2834ac6e22fb4434/';
const GLM51_SMALL_NVFP4 = "tests/python/test_models/glm51_small/glm51_small_nvfp4";

interface BenchArgs {
  gpus: number[];
  seqLen: number;
  chunkSize: number;
  contextLen: number;
  arena: number;
  maxBatch: number;
  warmupRuns: number;
  benchRuns: number;
  cp: boolean;
  pageSize: number;
  glm51Small: boolean;
}

function parseArgs(argv: string[]): BenchArgs {
  const gpusEnv = process.env.GLM_GPUS ?? process.env.GLM_GPU ?? "0";
  const args: BenchArgs = {
    gpus: gpusEnv === "0" ? [0, 1, 2, 3, 4, 5, 6, 7] : gpusEnv.split(",").map(s => parseInt(s.trim(), 10)),
    seqLen: 8192,
    chunkSize: 4096,
    contextLen: 0,
    arena: undefined!,
    maxBatch: 1,
    warmupRuns: 1,
    benchRuns: 1,
    cp: false,
    pageSize: 64,
    glm51Small: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--gpus" && i + 1 < argv.length) args.gpus = argv[++i].split(",").map(s => parseInt(s.trim(), 10));
    else if (a === "--seq-len" && i + 1 < argv.length) args.seqLen = parseInt(argv[++i], 10);
    else if (a === "--chunk-size" && i + 1 < argv.length) args.chunkSize = parseInt(argv[++i], 10);
    else if (a === "--context-len" && i + 1 < argv.length) args.contextLen = parseInt(argv[++i], 10);
    else if (a === "--arena" && i + 1 < argv.length) args.arena = parseInt(argv[++i], 10);
    else if (a === "--warmup" && i + 1 < argv.length) args.warmupRuns = parseInt(argv[++i], 10);
    else if (a === "--runs" && i + 1 < argv.length) args.benchRuns = parseInt(argv[++i], 10);
    else if (a === "--cp") args.cp = true;
    else if (a === "--page-size" && i + 1 < argv.length) args.pageSize = parseInt(argv[++i], 10);
    else if (a === "--glm51-small") args.glm51Small = true;
    else if (a === "--help") {
      console.log(`Usage: npx tsx src/run_prefill_benchmark.ts [options]
Options:
  --gpus <ids>       GPU IDs (default: 0-7)
  --seq-len <n>      Total prefill sequence length (default: 65536)
  --chunk-size <n>   Prefill chunk size (default: 4096)
  --context-len <n>  Dummy context length to pre-fill before the timed prefill (default: 0)
  --arena <n>        Arena size in GB (default: none)
  --warmup <n>       Warmup runs (default: 1)
  --runs <n>         Benchmark runs (default: 3)
  --cp               Enable context parallelism
  --page-size <n>    KV cache page size (default: 16)
  --glm51-small      Use GLM-5.1 small model
  --help             Show this help`);
      process.exit(0);
    }
  }
  return args;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const gpuDevices = args.gpus.map(id => new GlmOps(id, undefined, args.arena));
  const glm: DeviceOps = gpuDevices.length > 1 ? new ParallelOps(gpuDevices) : gpuDevices[0];
  const gpuLabel = args.gpus.length > 1 ? `${args.gpus[0]}-${args.gpus[args.gpus.length - 1]}` : `${args.gpus[0]}`;

  const modelDir = args.glm51Small ? GLM51_SMALL_NVFP4 : GLM51_MODEL_DIR;
  console.log(`GLM-5.1 Prefill Benchmark | GPUs ${gpuLabel} (${args.gpus.length}) | seq_len=${args.seqLen} | chunk_size=${args.chunkSize} | context_len=${args.contextLen} | cp=${args.cp} | model=${args.glm51Small ? "small" : "full"}`);

  const model: ChatModel = await Glm51Model.fromPretrained(glm, modelDir, args.cp, false);
  const worldSize = args.gpus.length;
  const cachePageSize = args.cp ? args.pageSize * worldSize : args.pageSize;
  const totalLen = args.contextLen + args.seqLen;
  const maxPages = Math.ceil(totalLen / cachePageSize) + 64;
  const cache = model.createChatCache(maxPages, args.maxBatch, totalLen + 1, cachePageSize);
  const ws = new ExecutionWorkspace(glm, args.maxBatch, args.chunkSize + 1);

  const inputIds = new Array(totalLen).fill(1);

  const cfg = (model as any).cfg ?? {};
  const numLayers = cfg.numHiddenLayers ?? "?";
  const hiddenSize = cfg.hiddenSize ?? "?";
  const numHeads = cfg.numAttentionHeads ?? "?";
  const kvLoraRank = cfg.kvLoraRank ?? "?";
  console.log(`Model: ${numLayers} layers, hidden=${hiddenSize}, heads=${numHeads}, kv_lora_rank=${kvLoraRank}, page_size=${cachePageSize} (effective ${args.pageSize}/gpu${args.cp ? ` ×${worldSize}` : ""})`);
  console.log(`KV cache: ${maxPages} pages (${(maxPages * cachePageSize * 2 * (kvLoraRank ?? 512 + 64) / 1024 / 1024 / 1024).toFixed(1)} GB for MLA cache)`);

  const numChunks = Math.ceil(args.seqLen / args.chunkSize);

  const captureManager = new CaptureManager(glm);
  captureManager.disabled = true;

  let warmupRuns = 0;

  while (!captureManager.isCaptured(['prefill']) && !captureManager.disabled) {
    cache.reset(1);


    warmupRuns++;
    const chunkStart = 0;

    const chunkLen = Math.min(args.chunkSize, args.seqLen - chunkStart);

    const t0 = performance.now();
    const tc0 = performance.now();
    const state = ws.planPrefill(model, 1, [chunkLen], cache);
    state.setInput([inputIds.slice(chunkStart, chunkStart + chunkLen)]);

    state.capture(captureManager, () => {
      using hiddenStates = model.forward(state);
    }, ['prefill']);

    glm.synchronize();
    const tc1 = performance.now();
    const chunkTokPerSec = chunkLen / ((tc1 - tc0) / 1000);

    const runLabel = "warmup";

    console.log(`  ${runLabel} chunk: ${chunkLen} tokens, ${(tc1 - tc0).toFixed(0)}ms (${chunkTokPerSec.toFixed(0)} tok/s)`);
  }
  console.log('warmup finished');

  // Pre-fill a dummy context so the timed prefill runs on top of a non-empty KV cache.
  // Only page allocation + allocLen advance is needed — no plan, no forward. The timed
  // planPrefill's updateIndptr unconditionally rebuilds and uploads the full page table.
  // KV data is garbage but attention timing is data-independent (fixed FLOPs by plan).
  if (args.contextLen > 0) {
    cache.reset(1);
    const cf0 = performance.now();
    cache.getPagedKV().allocAppendPages(0, args.contextLen);
    glm.synchronize();
    const cf1 = performance.now();
    console.log(`context prepared: ${args.contextLen} tokens in ${(cf1 - cf0).toFixed(0)}ms (pages allocated, no forward)`);
  }

  let captureRuns = 0;
  while (captureRuns < args.benchRuns) {
    const runLabel = `run ${captureRuns + 1}`;

    if (args.contextLen > 0) {
      // Rewind to the dummy context, freeing the previous run's pages but keeping
      // the context KV intact (truncate pops pages back to the available pool).
      for (const seq of cache.getPagedKV().sequences) {
        seq.truncate(args.contextLen);
      }
    } else {
      cache.reset(1);
    }

    const t0 = performance.now();
    for (let c = 0; c < numChunks; c++) {
      const chunkStart = args.contextLen + c * args.chunkSize;
      const chunkLen = Math.min(args.chunkSize, args.seqLen - c * args.chunkSize);
      const isLast = c === numChunks - 1;

      const tc0 = performance.now();
      const state = ws.planPrefill(model, 1, [chunkLen], cache);
      state.setInput([inputIds.slice(chunkStart, chunkStart + chunkLen)]);

      state.capture(captureManager, () => {
        using hiddenStates = model.forward(state);
      }, ['prefill']);

      glm.synchronize();
      const tc1 = performance.now();
      const chunkTokPerSec = chunkLen / ((tc1 - tc0) / 1000);
      console.log(`  ${runLabel} chunk ${c + 1}/${numChunks}: ${chunkLen} tokens, ${(tc1 - tc0).toFixed(0)}ms (${chunkTokPerSec.toFixed(0)} tok/s)`);
    }
    const elapsed = performance.now() - t0;

    const tokPerSec = args.seqLen / (elapsed / 1000);
    const msPerTok = elapsed / args.seqLen;

    console.log(`  ${runLabel}: ${elapsed.toFixed(0)}ms (${tokPerSec.toFixed(0)} tok/s, ${msPerTok.toFixed(3)} ms/tok)`);
    captureRuns++;
  }

  glm.synchronize();
  cache.free();
  ws.free();
  model.free();
  if (glm instanceof ParallelOps) glm.free();
  for (const d of gpuDevices) d.free();
}

main().catch((err) => {
  console.error("Failed:", err);
  process.exit(1);
});
