import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { GlmOps } from "../src/glm_ops";
import { ParallelOps } from "../src/parallel_ops";
import { Qwen3Model } from "../src/qwen3_model";
import { Qwen35Model } from "../src/qwen35_model";
import { ExecutionWorkspace, PagedKVCache } from "../src/paged_kv";
import { AutoTokenizer } from "@huggingface/transformers";
import { resolveModelPath } from "../src/model_path";
import type { ChatCache } from "../src/chat_model";
import type { DeviceOps } from "../src/device_ops";

const QWEN3_REPO = "Qwen/Qwen3-0.6B";
const FP8_REPO = "Qwen/Qwen3-0.6B-FP8";
const QWEN35_REPO = "Qwen/Qwen3.5-0.8B";

interface ModelContext extends Disposable {
  model: Qwen3Model | Qwen35Model;
  ws: ExecutionWorkspace;
  cache: ChatCache;
  eosIds: Set<number>;
  parisTokenId: number;
}

interface GraphOps {
  graphBeginCapture(): void;
  graphEndCapture(): number;
  graphInstantiate(graph: number): number;
  graphLaunch(graphExec: number): void;
  graphDestroy(graph: number): void;
  graphExecDestroy(graphExec: number): void;
  synchronize(): void;
}

function tokenizePrompt(tokenizer: any, prompt: string): number[] {
  const messages = [{ role: "user" as const, content: prompt }];
  const result = tokenizer.apply_chat_template(messages as any, {
    tokenize: true,
    add_generation_prompt: true,
    return_tensor: false,
    return_dict: true,
    tokenizer_kwargs: { enable_thinking: false },
  }) as { input_ids: number[] | number[][] };
  return (Array.isArray(result.input_ids[0]) ? result.input_ids[0] : result.input_ids) as number[];
}

function loadQwen3(glm: DeviceOps, repoId: string): ModelContext {
  const model = Qwen3Model.fromPretrained(glm, repoId, 1, 128);
  const ws = new ExecutionWorkspace(glm, 1, 128);
  const cache = new PagedKVCache(glm, model.cfg.numKeyValueHeads, model.cfg.headDim, model.cfg.numHiddenLayers, 256, 1);
  return {
    model, ws, cache,
    eosIds: model.eosIds,
    parisTokenId: 59604,
    [Symbol.dispose]() { cache[Symbol.dispose](); ws[Symbol.dispose](); model[Symbol.dispose](); },
  };
}

function loadQwen35(glm: DeviceOps): ModelContext {
  const model = Qwen35Model.fromPretrained(glm, QWEN35_REPO, 1, 128);
  const ws = new ExecutionWorkspace(glm, 1, 128);
  const cache = model.createChatCache(256);
  return {
    model, ws, cache,
    eosIds: model.eosIds,
    parisTokenId: 57590,
    [Symbol.dispose]() { cache[Symbol.dispose](); ws[Symbol.dispose](); model[Symbol.dispose](); },
  };
}

function generateWithGraph(
  ctx: ModelContext, graph: GraphOps, inputIds: number[], maxNewTokens: number,
): number[] {
  const { eosIds, cache, ws, model } = ctx;
  cache.reset(1);

  const firstTokens = ws.forwardEagerPrefill(model, [inputIds], cache);
  let currentToken = firstTokens[0];
  const generated: number[] = [currentToken];
  cache.appendTokens(0, [currentToken]);

  let graphExec: number | null = null;
  let warmupRemaining = 3;
  let capturing = false;
  let argmaxResult: any = null;

  for (let i = 1; i < maxNewTokens && !eosIds.has(currentToken); i++) {
    const state = ws.planDecode(model, 1, cache, true);
    state.prepareInput([currentToken]);

    if (graphExec === null) {
      if (warmupRemaining === 0 && !capturing) {
        capturing = true;
        graph.graphBeginCapture();
      }

      ws.decodeStep(state);
      ws.forwardInput(state);
      const logits = model.forward(state);
      argmaxResult = logits.argmax();

      if (capturing) {
        const graphIdx = graph.graphEndCapture();
        ws.freeze();
        graphExec = graph.graphInstantiate(graphIdx);
        graph.graphDestroy(graphIdx);
        capturing = false;
        warmupRemaining = 0;
      }

      if (warmupRemaining > 0) warmupRemaining--;
    }

    if (graphExec !== null) {
      graph.graphLaunch(graphExec);
      graph.synchronize();
    }

    currentToken = argmaxResult.readInt32LE()[0];
    generated.push(currentToken);
    cache.appendTokens(0, [currentToken]);
  }

  if (graphExec !== null) {
    graph.graphExecDestroy(graphExec);
  }

  return generated;
}

function assertParis(generated: number[], text: string, parisTokenId: number) {
  assert.ok(generated.length > 0, "No tokens generated");
  const hasParis = text.includes("Paris") || generated.includes(parisTokenId);
  assert.ok(hasParis, `Expected 'Paris' in generated text, got: ${JSON.stringify(text.slice(0, 200))}`);
  console.log(`  Generated ${generated.length} tokens: ${JSON.stringify(text.slice(0, 200))}`);
}

// --- Single GPU ---

describe("Qwen3-0.6B Paris (1 GPU, graph)", () => {
  let glm: GlmOps;
  let ctx: ModelContext;
  let tokenizer: any;

  before(async () => {
    glm = new GlmOps(parseInt(process.env.GLM_GPU ?? "0", 10));
    ctx = loadQwen3(glm, QWEN3_REPO);
    tokenizer = await AutoTokenizer.from_pretrained(resolveModelPath(QWEN3_REPO), { local_files_only: true });
  });
  after(() => { ctx[Symbol.dispose](); glm.free(); });

  it("generates 'Paris'", () => {
    const inputIds = tokenizePrompt(tokenizer, "The capital of France is");
    const generated = generateWithGraph(ctx, glm, inputIds, 64);
    assertParis(generated, tokenizer.decode(generated, { skip_special_tokens: true }), ctx.parisTokenId);
  });
});

describe("Qwen3-0.6B-FP8 Paris (1 GPU, graph)", () => {
  let glm: GlmOps;
  let ctx: ModelContext;
  let tokenizer: any;

  before(async () => {
    glm = new GlmOps(parseInt(process.env.GLM_GPU ?? "0", 10));
    ctx = loadQwen3(glm, FP8_REPO);
    tokenizer = await AutoTokenizer.from_pretrained(resolveModelPath(FP8_REPO), { local_files_only: true });
  });
  after(() => { ctx[Symbol.dispose](); glm.free(); });

  it("generates 'Paris'", () => {
    const inputIds = tokenizePrompt(tokenizer, "The capital of France is");
    const generated = generateWithGraph(ctx, glm, inputIds, 64);
    assertParis(generated, tokenizer.decode(generated, { skip_special_tokens: true }), ctx.parisTokenId);
  });
});

describe("Qwen3.5-0.8B Paris (1 GPU, graph)", () => {
  let glm: GlmOps;
  let ctx: ModelContext;
  let tokenizer: any;

  before(async () => {
    glm = new GlmOps(parseInt(process.env.GLM_GPU ?? "0", 10));
    ctx = loadQwen35(glm);
    tokenizer = await AutoTokenizer.from_pretrained(resolveModelPath(QWEN35_REPO), { local_files_only: true });
  });
  after(() => { ctx[Symbol.dispose](); glm.free(); });

  it("generates 'Paris'", () => {
    const inputIds = tokenizePrompt(tokenizer, "The capital of France is");
    const generated = generateWithGraph(ctx, glm, inputIds, 64);
    assertParis(generated, tokenizer.decode(generated, { skip_special_tokens: true }), ctx.parisTokenId);
  });
});

// --- Parallel GPU ---

describe("Qwen3-0.6B Paris (2 GPU, graph)", () => {
  let glm0: GlmOps;
  let glm1: GlmOps;
  let po: ParallelOps;
  let ctx: ModelContext;
  let tokenizer: any;

  before(async () => {
    glm0 = new GlmOps(0);
    glm1 = new GlmOps(1);
    po = new ParallelOps([glm0, glm1]);
    ctx = loadQwen3(po as any, QWEN3_REPO);
    tokenizer = await AutoTokenizer.from_pretrained(resolveModelPath(QWEN3_REPO), { local_files_only: true });
  });
  after(() => { ctx[Symbol.dispose](); po.free(); glm0.free(); glm1.free(); });

  it("generates 'Paris'", () => {
    const inputIds = tokenizePrompt(tokenizer, "The capital of France is");
    const generated = generateWithGraph(ctx, po, inputIds, 64);
    assertParis(generated, tokenizer.decode(generated, { skip_special_tokens: true }), ctx.parisTokenId);
  });
});

describe("Qwen3-0.6B-FP8 Paris (2 GPU, graph)", () => {
  let glm0: GlmOps;
  let glm1: GlmOps;
  let po: ParallelOps;
  let ctx: ModelContext;
  let tokenizer: any;

  before(async () => {
    glm0 = new GlmOps(0);
    glm1 = new GlmOps(1);
    po = new ParallelOps([glm0, glm1]);
    ctx = loadQwen3(po as any, FP8_REPO);
    tokenizer = await AutoTokenizer.from_pretrained(resolveModelPath(FP8_REPO), { local_files_only: true });
  });
  after(() => { ctx[Symbol.dispose](); po.free(); glm0.free(); glm1.free(); });

  it("generates 'Paris'", () => {
    const inputIds = tokenizePrompt(tokenizer, "The capital of France is");
    const generated = generateWithGraph(ctx, po, inputIds, 64);
    assertParis(generated, tokenizer.decode(generated, { skip_special_tokens: true }), ctx.parisTokenId);
  });
});

describe("Qwen3.5-0.8B Paris (2 GPU, graph)", () => {
  let glm0: GlmOps;
  let glm1: GlmOps;
  let po: ParallelOps;
  let ctx: ModelContext;
  let tokenizer: any;

  before(async () => {
    glm0 = new GlmOps(0);
    glm1 = new GlmOps(1);
    po = new ParallelOps([glm0, glm1]);
    ctx = loadQwen35(po as any);
    tokenizer = await AutoTokenizer.from_pretrained(resolveModelPath(QWEN35_REPO), { local_files_only: true });
  });
  after(() => { ctx[Symbol.dispose](); po.free(); glm0.free(); glm1.free(); });

  it("generates 'Paris'", () => {
    const inputIds = tokenizePrompt(tokenizer, "The capital of France is");
    const generated = generateWithGraph(ctx, po, inputIds, 64);
    assertParis(generated, tokenizer.decode(generated, { skip_special_tokens: true }), ctx.parisTokenId);
  });
});
