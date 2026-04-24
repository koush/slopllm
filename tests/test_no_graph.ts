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

interface ModelContext {
  model: Qwen3Model | Qwen35Model;
  ws: ExecutionWorkspace;
  cache: ChatCache;
  eosIds: Set<number>;
  parisTokenId: number;
  free: () => void;
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
    free: () => { cache.free(); ws.free(); model.free(); },
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
    free: () => { cache.free(); ws.free(); model.free(); },
  };
}

function assertParis(generated: number[], text: string, parisTokenId: number) {
  assert.ok(generated.length > 0, "No tokens generated");
  const hasParis = text.includes("Paris") || generated.includes(parisTokenId);
  assert.ok(hasParis, `Expected 'Paris' in generated text, got: ${JSON.stringify(text.slice(0, 200))}`);
  console.log(`  Generated ${generated.length} tokens: ${JSON.stringify(text.slice(0, 200))}`);
}

// --- Single GPU ---

describe("Qwen3-0.6B Paris (1 GPU, no graph)", () => {
  let glm: GlmOps;
  let ctx: ModelContext;
  let tokenizer: any;

  before(async () => {
    glm = new GlmOps(parseInt(process.env.GLM_GPU ?? "0", 10));
    ctx = loadQwen3(glm, QWEN3_REPO);
    tokenizer = await AutoTokenizer.from_pretrained(resolveModelPath(QWEN3_REPO), { local_files_only: true });
  });
  after(() => { ctx.free(); glm.free(); });

  it("generates 'Paris'", () => {
    const { eosIds, cache, ws, model, parisTokenId } = ctx;
    const inputIds = tokenizePrompt(tokenizer, "The capital of France is");
    cache.reset(1);
    let currentToken = ws.forwardEager(model, [inputIds], cache)[0];
    const generated: number[] = [currentToken];
    for (let i = 1; i < 64 && !eosIds.has(currentToken); i++) {
      currentToken = ws.forwardEagerDecode(model, [currentToken], cache)[0];
      generated.push(currentToken);
    }
    assertParis(generated, tokenizer.decode(generated, { skip_special_tokens: true }), parisTokenId);
  });
});

describe("Qwen3-0.6B-FP8 Paris (1 GPU, no graph)", () => {
  let glm: GlmOps;
  let ctx: ModelContext;
  let tokenizer: any;

  before(async () => {
    glm = new GlmOps(parseInt(process.env.GLM_GPU ?? "0", 10));
    ctx = loadQwen3(glm, FP8_REPO);
    tokenizer = await AutoTokenizer.from_pretrained(resolveModelPath(FP8_REPO), { local_files_only: true });
  });
  after(() => { ctx.free(); glm.free(); });

  it("generates 'Paris'", () => {
    const { eosIds, cache, ws, model, parisTokenId } = ctx;
    const inputIds = tokenizePrompt(tokenizer, "The capital of France is");
    cache.reset(1);
    let currentToken = ws.forwardEager(model, [inputIds], cache)[0];
    const generated: number[] = [currentToken];
    for (let i = 1; i < 64 && !eosIds.has(currentToken); i++) {
      currentToken = ws.forwardEagerDecode(model, [currentToken], cache)[0];
      generated.push(currentToken);
    }
    assertParis(generated, tokenizer.decode(generated, { skip_special_tokens: true }), parisTokenId);
  });
});

describe("Qwen3.5-0.8B Paris (1 GPU, no graph)", () => {
  let glm: GlmOps;
  let ctx: ModelContext;
  let tokenizer: any;

  before(async () => {
    glm = new GlmOps(parseInt(process.env.GLM_GPU ?? "0", 10));
    ctx = loadQwen35(glm);
    tokenizer = await AutoTokenizer.from_pretrained(resolveModelPath(QWEN35_REPO), { local_files_only: true });
  });
  after(() => { ctx.free(); glm.free(); });

  it("generates 'Paris'", () => {
    const { eosIds, cache, ws, model, parisTokenId } = ctx;
    const inputIds = tokenizePrompt(tokenizer, "The capital of France is");
    cache.reset(1);
    let currentToken = ws.forwardEager(model, [inputIds], cache)[0];
    const generated: number[] = [currentToken];
    for (let i = 1; i < 20 && !eosIds.has(currentToken); i++) {
      currentToken = ws.forwardEagerDecode(model, [currentToken], cache)[0];
      generated.push(currentToken);
    }
    assertParis(generated, tokenizer.decode(generated, { skip_special_tokens: true }), parisTokenId);
  });
});

// --- Parallel GPU ---

describe("Qwen3-0.6B Paris (2 GPU, no graph)", () => {
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
  after(() => { ctx.free(); po.free(); glm0.free(); glm1.free(); });

  it("generates 'Paris'", () => {
    const { eosIds, cache, ws, model, parisTokenId } = ctx;
    const inputIds = tokenizePrompt(tokenizer, "The capital of France is");
    cache.reset(1);
    let currentToken = ws.forwardEager(model, [inputIds], cache)[0];
    const generated: number[] = [currentToken];
    for (let i = 1; i < 64 && !eosIds.has(currentToken); i++) {
      currentToken = ws.forwardEagerDecode(model, [currentToken], cache)[0];
      generated.push(currentToken);
    }
    assertParis(generated, tokenizer.decode(generated, { skip_special_tokens: true }), parisTokenId);
  });
});

describe("Qwen3-0.6B-FP8 Paris (2 GPU, no graph)", () => {
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
  after(() => { ctx.free(); po.free(); glm0.free(); glm1.free(); });

  it("generates 'Paris'", () => {
    const { eosIds, cache, ws, model, parisTokenId } = ctx;
    const inputIds = tokenizePrompt(tokenizer, "The capital of France is");
    cache.reset(1);
    let currentToken = ws.forwardEager(model, [inputIds], cache)[0];
    const generated: number[] = [currentToken];
    for (let i = 1; i < 64 && !eosIds.has(currentToken); i++) {
      currentToken = ws.forwardEagerDecode(model, [currentToken], cache)[0];
      generated.push(currentToken);
    }
    assertParis(generated, tokenizer.decode(generated, { skip_special_tokens: true }), parisTokenId);
  });
});

describe("Qwen3.5-0.8B Paris (2 GPU, no graph)", () => {
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
  after(() => { ctx.free(); po.free(); glm0.free(); glm1.free(); });

  it("generates 'Paris'", () => {
    const { eosIds, cache, ws, model, parisTokenId } = ctx;
    const inputIds = tokenizePrompt(tokenizer, "The capital of France is");
    cache.reset(1);
    let currentToken = ws.forwardEager(model, [inputIds], cache)[0];
    const generated: number[] = [currentToken];
    for (let i = 1; i < 20 && !eosIds.has(currentToken); i++) {
      currentToken = ws.forwardEagerDecode(model, [currentToken], cache)[0];
      generated.push(currentToken);
    }
    assertParis(generated, tokenizer.decode(generated, { skip_special_tokens: true }), parisTokenId);
  });
});
