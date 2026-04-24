import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { GlmOps } from "../src/glm_ops";
import { Qwen3Model } from "../src/qwen3_model";
import { ExecutionWorkspace, PagedKVCache } from "../src/paged_kv";
import { AutoTokenizer } from "@huggingface/transformers";
import { resolveModelPath } from "../src/model_path";

const QWEN3_REPO = "Qwen/Qwen3-0.6B";

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

describe("Qwen3-0.6B generate Paris", () => {
  let glm: GlmOps;
  let model: Qwen3Model;
  let ws: ExecutionWorkspace;
  let tokenizer: any;
  let cache: PagedKVCache;

  before(async () => {
    const deviceId = parseInt(process.env.GLM_GPU ?? "0", 10);
    glm = new GlmOps(deviceId);
    model = Qwen3Model.fromPretrained(glm, QWEN3_REPO, 1, 128);
    ws = new ExecutionWorkspace(glm, 1, 128);
    const cfg = model.cfg;
    cache = new PagedKVCache(glm, cfg.numKeyValueHeads, cfg.headDim, cfg.numHiddenLayers, 256, 1);
    const modelDir = resolveModelPath(QWEN3_REPO);
    tokenizer = await AutoTokenizer.from_pretrained(modelDir, { local_files_only: true });
  });

  after(() => {
    cache.free();
    ws.free();
    model.free();
    glm.free();
  });

  it("generates 'Paris' from 'The capital of France is' prompt (greedy)", () => {
    const eosIds = model.eosIds;
    const inputIds = tokenizePrompt(tokenizer, "The capital of France is");
    const maxNewTokens = 64;

    cache.reset(1);
    let currentToken = ws.forwardEager(model, [inputIds], cache)[0];
    const generated: number[] = [currentToken];

    for (let i = 1; i < maxNewTokens && !eosIds.has(currentToken); i++) {
      currentToken = ws.forwardEagerDecode(model, [currentToken], cache)[0];
      generated.push(currentToken);
    }

    assert.ok(generated.length > 0, "No tokens generated");

    const text = tokenizer.decode(generated, { skip_special_tokens: true });
    const parisTokenId = 59604;
    const hasParis = text.includes("Paris") || generated.includes(parisTokenId);
    assert.ok(hasParis, `Expected 'Paris' in generated text, got: ${JSON.stringify(text.slice(0, 200))}`);
    console.log(`  Generated ${generated.length} tokens: ${JSON.stringify(text.slice(0, 200))}`);
  });
});
