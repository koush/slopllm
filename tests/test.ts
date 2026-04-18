import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { GlmOps } from "../src/glm_ops";
import { Qwen3Model } from "../src/qwen3_model";
import { PagedKVCache } from "../src/paged_kv";
import { AutoTokenizer } from "@huggingface/transformers";
import { resolveModelPath } from "../src/model_path";
import { generateBatchTokens } from "./test_helper";

const QWEN3_REPO = "Qwen/Qwen3-0.6B";
const EOS_TOKEN_IDS = new Set([151645, 151643]);

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

describe("Qwen3-0.6B batch smoke test", () => {
  let glm: GlmOps;
  let model: Qwen3Model;
  let tokenizer: any;
  let cache: PagedKVCache;

  before(async () => {
    process.env.CUDA_VISIBLE_DEVICES = process.env.GLM_GPU ?? "0";
    glm = new GlmOps(0);
    model = Qwen3Model.fromPretrained(glm, QWEN3_REPO, 4, 2048);
    const cfg = model.cfg;
    cache = new PagedKVCache(glm, cfg.numKeyValueHeads, cfg.headDim, cfg.numHiddenLayers, 128, 4);
    const modelDir = resolveModelPath(QWEN3_REPO);
    tokenizer = await AutoTokenizer.from_pretrained(modelDir, { local_files_only: true });
  });

  after(() => {
    cache.free();
    model.free();
  });

  it("2 identical 'hi' prompts produce responses containing 'hello' and 'assist'", () => {
    const promptIds = tokenizePrompt(tokenizer, "hi");
    const generated = generateBatchTokens(model, cache, [promptIds, promptIds], 128, EOS_TOKEN_IDS);

    assert.equal(generated.length, 2);

    for (let i = 0; i < generated.length; i++) {
      const text = tokenizer.decode(generated[i], { skip_special_tokens: true }).toLowerCase();
      assert.ok(text.includes("hello"), `Response ${i + 1}: expected "hello" in: ${text.slice(0, 80)}`);
      assert.ok(text.includes("assist"), `Response ${i + 1}: expected "assist" in: ${text.slice(0, 80)}`);
    }
  });
});
