import { GlmOps } from "./glm_ops.js";
import { Qwen3Model } from "./qwen3_model.js";
import { PagedKVCache, WorkspaceBuffers } from "./paged_kv.js";
import { AutoTokenizer } from "@huggingface/transformers";
import { resolveModelPath } from "./model_path.js";
import { createInterface } from "node:readline";

const QWEN3_REPO = "Qwen/Qwen3-0.6B";
const EOS_TOKEN_IDS = new Set([151645, 151643]);

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  let maxTokens = 128;
  let noThink = false;
  let gpu: number | undefined;
  let maxSeqLen = 2048;
  let maxPages = 128;
  let maxBatch = 4;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--max-tokens" && i + 1 < args.length) {
      maxTokens = parseInt(args[++i], 10);
    } else if (args[i] === "--no-think") {
      noThink = true;
    } else if (args[i] === "--gpu" && i + 1 < args.length) {
      gpu = parseInt(args[++i], 10);
    } else if (args[i] === "--max-seq-len" && i + 1 < args.length) {
      maxSeqLen = parseInt(args[++i], 10);
    } else if (args[i] === "--max-pages" && i + 1 < args.length) {
      maxPages = parseInt(args[++i], 10);
    } else if (args[i] === "--max-batch" && i + 1 < args.length) {
      maxBatch = parseInt(args[++i], 10);
    }
  }

  const gpuId = gpu ?? parseInt(process.env.GLM_GPU ?? "0", 10);
  process.env.CUDA_VISIBLE_DEVICES = String(gpuId);

  console.log(`Loading model on GPU ${gpuId}...`);
  const glm = new GlmOps(0);
  const model = Qwen3Model.fromPretrained(glm, QWEN3_REPO, maxBatch, maxSeqLen);

  const modelDir = resolveModelPath(QWEN3_REPO);
  const tokenizer = await AutoTokenizer.from_pretrained(modelDir, { local_files_only: true });

  const cfg = (model as any).cfg;
  const nKv = cfg.numKeyValueHeads;
  const hd = cfg.headDim;
  const nLayers = cfg.numHiddenLayers;

  const ws = new WorkspaceBuffers(glm);
  const pagedKV = new PagedKVCache(glm, nKv, hd, nLayers, maxPages, maxBatch);

  const enableThinking = !noThink;

  console.log(`Qwen3-0.6B batched ready (max_tokens=${maxTokens}, thinking=${noThink ? "off" : "on"})`);
  console.log("Enter prompts one per line. Empty line to submit batch. /clear to reset, /q to quit.");

  const rl = createInterface({ input: process.stdin, output: process.stdout });
  const askLine = (prompt: string): Promise<string> =>
    new Promise((resolve) => rl.question(prompt, resolve));

  try {
    while (true) {
      const prompts: string[] = [];
      while (true) {
        const line = (await askLine(`\nPrompt ${prompts.length + 1} (empty=done): `)).trim();
        if (line === "") {
          break;
        }
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

      const inputIdsList: number[][] = [];
      for (const prompt of prompts) {
        const messages = [{ role: "user" as const, content: prompt }];
        const result = tokenizer.apply_chat_template(messages as any, {
          tokenize: true,
          add_generation_prompt: true,
          return_tensor: false,
          return_dict: true,
          tokenizer_kwargs: { enable_thinking: enableThinking },
        }) as { input_ids: number[] | number[][] };
        const ids = (Array.isArray(result.input_ids[0]) ? result.input_ids[0] : result.input_ids) as number[];
        inputIdsList.push(ids);
      }

      const maxPromptLen = Math.max(...inputIdsList.map(ids => ids.length));
      if (maxPromptLen > maxSeqLen) {
        console.log(`Longest prompt (${maxPromptLen} tokens) exceeds max_seq_len (${maxSeqLen}). Skipping.`);
        continue;
      }

      const start = Date.now();
      const generatedIds = model.generateBatch(inputIdsList, ws, pagedKV, maxTokens, EOS_TOKEN_IDS);
      const elapsed = (Date.now() - start) / 1000;
      const totalTokens = generatedIds.reduce((sum, ids) => sum + ids.length, 0);

      for (let i = 0; i < prompts.length; i++) {
        const text = tokenizer.decode(generatedIds[i], { skip_special_tokens: true });
        console.log(`\n--- Response ${i + 1} ---`);
        console.log(text);
      }

      console.log(`\n  [${prompts.length} prompts, ${totalTokens} tokens, ${elapsed.toFixed(1)}s, ${(totalTokens / elapsed).toFixed(1)} tok/s]`);
    }
  } finally {
    pagedKV.free();
    ws.free();
    model.free();
    rl.close();
  }
}

main().catch((err) => {
  console.error("Failed:", err);
  process.exit(1);
});
