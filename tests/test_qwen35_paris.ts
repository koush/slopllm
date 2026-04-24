import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { GlmOps } from "../src/glm_ops";
import { Qwen35Model } from "../src/qwen35_model";
import { ExecutionWorkspace } from "../src/paged_kv";
import { AutoTokenizer } from "@huggingface/transformers";
import { resolveModelPath } from "../src/model_path";

const QWEN35_REPO = "Qwen/Qwen3.5-0.8B";

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

describe("Qwen3.5-0.8B generate Paris", () => {
  let glm: GlmOps;
  let model: Qwen35Model;
  let ws: ExecutionWorkspace;
  let tokenizer: any;
  let cache: ReturnType<Qwen35Model["createChatCache"]>;

  before(async () => {
    const deviceId = parseInt(process.env.GLM_GPU ?? "0", 10);
    glm = new GlmOps(deviceId);
    model = Qwen35Model.fromPretrained(glm, QWEN35_REPO, 1, 128);
    ws = new ExecutionWorkspace(glm, 1, 128);
    cache = model.createChatCache(256);
    const modelDir = resolveModelPath(QWEN35_REPO);
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
    const maxNewTokens = 20;

    cache.reset(1);
    let currentToken = ws.forwardEager(model, [inputIds], cache)[0];
    const generated: number[] = [currentToken];

    for (let i = 1; i < maxNewTokens && !eosIds.has(currentToken); i++) {
      currentToken = ws.forwardEagerDecode(model, [currentToken], cache)[0];
      generated.push(currentToken);
    }

    assert.ok(generated.length > 0, "No tokens generated");

    const text = tokenizer.decode(generated, { skip_special_tokens: true });
    const parisTokenId = 57590;
    const hasParis = text.includes("Paris") || generated.includes(parisTokenId);
    assert.ok(hasParis, `Expected 'Paris' in generated text, got: ${JSON.stringify(text.slice(0, 200))}`);
    console.log(`  Generated ${generated.length} tokens: ${JSON.stringify(text.slice(0, 200))}`);
  });
});
