import fs from "node:fs";
import path from "node:path";
import { ChatModel } from "./chat_model";
import { DeviceOps } from "./device_ops";
import { ExecutionWorkspace } from "./execution-workspace";
import { Glm51Model } from "./glm51_model";
import { GlmOps } from "./glm_ops";
import { ParallelOps } from "./parallel_ops";
import { resolveModelPath } from "./model_path";
import { WorkspaceBase } from "./workspace";

const GLM51_MODEL_DIR = "/mnt/storage/GLM-5.1-NVFP4-Fixed";

interface BenchArgs {
  gpus: number[];
  seqLen: number;
  maxBatch: number;
  warmupRuns: number;
  benchRuns: number;
  cp: boolean;
  pageSize: number;
}

function parseArgs(argv: string[]): BenchArgs {
  const gpusEnv = process.env.GLM_GPUS ?? process.env.GLM_GPU ?? "0";
  const args: BenchArgs = {
    gpus: gpusEnv === "0" ? [0, 1, 2, 3, 4, 5, 6, 7] : gpusEnv.split(",").map(s => parseInt(s.trim(), 10)),
  seqLen: 65536,
  maxBatch: 1,
    warmupRuns: 1,
    benchRuns: 3,
    cp: true,
    pageSize: 16,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--gpus" && i + 1 < argv.length) args.gpus = argv[++i].split(",").map(s => parseInt(s.trim(), 10));
    else if (a === "--seq-len" && i + 1 < argv.length) args.seqLen = parseInt(argv[++i], 10);
    else if (a === "--warmup" && i + 1 < argv.length) args.warmupRuns = parseInt(argv[++i], 10);
    else if (a === "--runs" && i + 1 < argv.length) args.benchRuns = parseInt(argv[++i], 10);
    else if (a === "--cp") args.cp = true;
    else if (a === "--help") {
      console.log(`Usage: npx tsx src/run_prefill_benchmark.ts [options]
Options:
  --gpus <ids>       GPU IDs (default: 0)
  --seq-len <n>      Prefill sequence length (default: 65536)
  --warmup <n>       Warmup runs (default: 1)
  --runs <n>         Benchmark runs (default: 3)
  --cp               Enable context parallelism
  --help             Show this help`);
      process.exit(0);
    }
  }
  return args;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  const gpuDevices = args.gpus.map(id => new GlmOps(id, undefined, 92));
  const glm: DeviceOps = gpuDevices.length > 1 ? new ParallelOps(gpuDevices) : gpuDevices[0];
  const gpuLabel = args.gpus.length > 1 ? `${args.gpus[0]}-${args.gpus[args.gpus.length - 1]}` : `${args.gpus[0]}`;

  console.log(`GLM-5.1 Prefill Benchmark | GPUs ${gpuLabel} (${args.gpus.length}) | seq_len=${args.seqLen} | cp=${args.cp}`);

  const model: ChatModel = await Glm51Model.fromPretrained(glm, GLM51_MODEL_DIR, args.cp, false);
  const pageSize = args.pageSize;
  const maxPages = Math.ceil(args.seqLen / pageSize) + 64;
  const cache = model.createChatCache(maxPages, args.maxBatch, args.seqLen + 1);
  const ws = new ExecutionWorkspace(glm, args.maxBatch, args.seqLen + 1);

  const inputIds = new Array(args.seqLen).fill(1);

  const cfg = (model as any).cfg ?? {};
  const numLayers = cfg.numHiddenLayers ?? "?";
  const hiddenSize = cfg.hiddenSize ?? "?";
  const numHeads = cfg.numAttentionHeads ?? "?";
  const kvLoraRank = cfg.kvLoraRank ?? "?";
  console.log(`Model: ${numLayers} layers, hidden=${hiddenSize}, heads=${numHeads}, kv_lora_rank=${kvLoraRank}, page_size=${pageSize}`);
  console.log(`KV cache: ${maxPages} pages (${(maxPages * pageSize * 2 * (kvLoraRank ?? 512 + 64) / 1024 / 1024 / 1024).toFixed(1)} GB for MLA cache)`);

  for (let run = -args.warmupRuns; run < args.benchRuns; run++) {
    const isWarmup = run < 0;
    const runLabel = isWarmup ? "warmup" : `run ${run + 1}`;

    cache.reset(1);

    const t0 = performance.now();
    const state = ws.planPrefill(model, 1, [args.seqLen], cache);
    state.setInput([inputIds]);
    using hiddenStates = model.forward(state);
    using logits = state.computeLogits(hiddenStates, model);
    glm.synchronize();
    const elapsed = performance.now() - t0;

    const tokPerSec = args.seqLen / (elapsed / 1000);
    const msPerTok = elapsed / args.seqLen;

    if (isWarmup) {
      console.log(`  ${runLabel}: ${elapsed.toFixed(0)}ms (${tokPerSec.toFixed(0)} tok/s, ${msPerTok.toFixed(3)} ms/tok) [discarded]`);
    } else {
      console.log(`  ${runLabel}: ${elapsed.toFixed(0)}ms (${tokPerSec.toFixed(0)} tok/s, ${msPerTok.toFixed(3)} ms/tok)`);
    }
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
