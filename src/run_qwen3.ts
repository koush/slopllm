import { GlmOps } from "./glm_ops";
import { Qwen3Model } from "./qwen3_model";
import { FlatKVCache } from "./flat_kv";
import { AutoTokenizer, PreTrainedTokenizer } from "@huggingface/transformers";
import { resolveModelPath } from "./model_path";
import { createInterface } from "node:readline";

const QWEN3_REPO = "Qwen/Qwen3-0.6B";
const QWEN3_FP8_REPO = "Qwen/Qwen3-0.6B-FP8";
const EOS_TOKEN_IDS = new Set([151645, 151643]);

async function chatLoop(model: Qwen3Model, cache: FlatKVCache, tokenizer: PreTrainedTokenizer, maxNewTokens: number, noThink: boolean): Promise<void> {
  const messages: Array<{ role: string; content: string }> = [];
  const enableThinking = !noThink;
  const rl = createInterface({ input: process.stdin, output: process.stdout });

  const askQuestion = (prompt: string): Promise<string> =>
    new Promise((resolve) => rl.question(prompt, resolve));

  while (true) {
    const userInput = (await askQuestion("\nYou: ")).trim();
    if (!userInput) continue;
    if (userInput.toLowerCase() === "quit" || userInput.toLowerCase() === "exit" || userInput.toLowerCase() === "/q") break;
    if (userInput.toLowerCase() === "/clear") {
      messages.length = 0;
      console.log("Conversation cleared.");
      cache.reset();
      continue;
    }

    messages.push({ role: "user", content: userInput });

    const result = tokenizer.apply_chat_template(messages, {
      tokenize: true,
      add_generation_prompt: true,
      return_tensor: false,
      return_dict: true,
      tokenizer_kwargs: { enable_thinking: enableThinking },
    }) as { input_ids: number[] | number[][] };
    const inputIds = (Array.isArray(result.input_ids[0]) ? result.input_ids : [result.input_ids]) as number[][];

    if (inputIds[0].length > model.maxSeqLen) {
      console.log(`Prompt (${inputIds[0].length} tokens) exceeds max_seq_len (${model.maxSeqLen}). Truncating conversation.`);
      messages.pop();
      continue;
    }

    process.stdout.write("\nAssistant: ");
    const generatedIds: number[] = [];
    const start = Date.now();
    let tokenCount = 0;

    for (const tokenId of model.streamTokens(inputIds, cache, maxNewTokens, EOS_TOKEN_IDS)) {
      generatedIds.push(tokenId);
      tokenCount++;
      const chunk = tokenizer.decode([tokenId], { skip_special_tokens: false });
      process.stdout.write(chunk);
      if (EOS_TOKEN_IDS.has(tokenId)) break;
    }

    const elapsed = (Date.now() - start) / 1000;
    console.log(`\n  [${tokenCount} tokens, ${elapsed.toFixed(1)}s, ${(tokenCount / elapsed).toFixed(1)} tok/s]`);

    const assistantText = tokenizer.decode(generatedIds, { skip_special_tokens: true });
    messages.push({ role: "assistant", content: assistantText });
  }

  rl.close();
  model.free();
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  let maxTokens = 512;
  let noThink = false;
  let gpu: number | undefined;
  let maxSeqLen = 2048;
  let useFp8 = false;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--max-tokens" && i + 1 < args.length) {
      maxTokens = parseInt(args[++i], 10);
    } else if (args[i] === "--no-think") {
      noThink = true;
    } else if (args[i] === "--gpu" && i + 1 < args.length) {
      gpu = parseInt(args[++i], 10);
    } else if (args[i] === "--max-seq-len" && i + 1 < args.length) {
      maxSeqLen = parseInt(args[++i], 10);
    } else if (args[i] === "--fp8") {
      useFp8 = true;
    }
  }

  const gpuId = gpu ?? parseInt(process.env.GLM_GPU ?? "0", 10);
  process.env.CUDA_VISIBLE_DEVICES = String(gpuId);

  const repoId = useFp8 ? QWEN3_FP8_REPO : QWEN3_REPO;

  console.log(`Loading model on GPU ${gpuId}${useFp8 ? " (FP8)" : ""}...`);
  const glm = new GlmOps(0);
  const model = Qwen3Model.fromPretrained(glm, repoId, 1, maxSeqLen);
  const cache = model.createFlatKVCache();

  const modelDir = resolveModelPath(repoId);
  const tokenizer = await AutoTokenizer.from_pretrained(modelDir, { local_files_only: true });

  console.log(`Qwen3-0.6B ready (max_tokens=${maxTokens}, thinking=${noThink ? "off" : "on"}${useFp8 ? ", fp8" : ""})`);
  console.log("Type a message to chat. /clear to reset, /q to quit.");

  await chatLoop(model, cache, tokenizer, maxTokens, noThink);
}

main().catch((err) => {
  console.error("Failed:", err);
  process.exit(1);
});
