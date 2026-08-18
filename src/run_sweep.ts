// Greedy termination sweep: run N prompts through one loaded model and report
// how many fail to emit EOS or fall into a repetition loop.
//
// The point is a *rate*, not an anecdote — single-prompt comparisons between
// engines are uninformative when the underlying divergence is chaotic.
//
//   npx tsx src/run_sweep.ts --gpus 0,1,2,3,4,5,6,7 --arena 92 --start 0 --count 10
import { DeviceOps } from "./device_ops";
import { ExecutionWorkspace } from "./execution-workspace";
import { Glm51Model } from "./glm51_model";
import { GlmOps } from "./glm_ops";
import { ParallelOps } from "./parallel_ops";
import { generateStream } from "./run_qwen3_unified";

const DEFAULT_MODEL_DIR = "/mnt/storage/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/2eff962076815828e4031aec2834ac6e22fb4434/";

const PROMPTS = [
  "tell me about india", "tell me about brazil", "tell me about japan", "tell me about egypt",
  "explain quantum entanglement", "explain how a transformer neural network works",
  "explain the causes of the french revolution", "explain photosynthesis",
  "write a python function that reverses a linked list", "write a python class for an LRU cache",
  "what is the meaning of life", "summarize the plot of hamlet",
  "compare rust and go for systems programming", "describe the water cycle",
  "how does a nuclear reactor work", "what caused the 2008 financial crisis",
  "give me a 7 day itinerary for iceland", "explain CRISPR gene editing",
  "what are the tradeoffs of microservices", "describe the history of the silk road",
];

const argv = process.argv.slice(2);
const opt = (n: string, d: string) => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : d; };

// Same detector used for the vLLM sweep: a 12-word window where one word
// occupies >= 8 slots.
function looped(t: string): boolean {
  const w = t.split(/\s+/).filter(Boolean);
  if (w.length < 40) return false;
  for (let k = 0; k < w.length - 14; k++) {
    const win = w.slice(k, k + 12);
    const counts = new Map<string, number>();
    for (const x of win) counts.set(x, (counts.get(x) ?? 0) + 1);
    if (Math.max(...counts.values()) >= 8) return true;
  }
  return false;
}

async function main() {
  const gpus = opt("--gpus", "0").split(",").map(s => parseInt(s.trim(), 10));
  const arena = parseInt(opt("--arena", "0"), 10);
  const modelDir = opt("--model-dir", DEFAULT_MODEL_DIR);
  const start = parseInt(opt("--start", "0"), 10);
  const count = parseInt(opt("--count", String(PROMPTS.length)), 10);
  const maxNew = parseInt(opt("--max-new-tokens", "4000"), 10);
  const maxSeqLen = parseInt(opt("--max-seq-len", "8192"), 10);

  const gpuDevices = gpus.map(id => new GlmOps(id, undefined, arena || undefined));
  const glm: DeviceOps = gpuDevices.length > 1 ? new ParallelOps(gpuDevices) : gpuDevices[0];
  const model = await Glm51Model.fromPretrained(glm, modelDir, false, false);
  const cache = model.createChatCache(parseInt(opt("--max-pages", "8192"), 10), 1, maxSeqLen);
  const ws = new ExecutionWorkspace(glm, 1, maxSeqLen);

  const tokenizer = model.tokenizer;
  const effort = process.env.GLM_REASONING_EFFORT;
  console.log(`[template]${effort ? ` reasoning_effort=${effort}` : ""}`);

  let bad = 0, n = 0;
  for (const prompt of PROMPTS.slice(start, start + count)) {
    const r: any = tokenizer.apply_chat_template([{ role: "user", content: prompt }],
      { tokenize: true, add_generation_prompt: true, return_tensor: false, return_dict: true,
        ...(effort ? { reasoning_effort: effort } : {}) });
    const inputIds = (Array.isArray(r.input_ids[0]) ? r.input_ids[0] : r.input_ids) as number[];

    cache.reset(1);
    const graphState = { graphExec: null as number | null, warmupRemaining: 3 };
    const ids: number[] = [];
    let sawEos = false;
    for await (const t of generateStream(model, ws, glm, cache, inputIds, maxNew, model.eosIds, undefined, graphState)) {
      ids.push(t);
      if (model.eosIds.has(t)) { sawEos = true; break; }
    }
    const text = tokenizer.decode(ids.filter((t: number) => !model.eosIds.has(t)));
    const lp = looped(text);
    n++;
    if (!sawEos || lp) bad++;
    console.log(`${sawEos ? "    stop" : "  LENGTH"} ${String(ids.length).padStart(5)}tok  loop=${String(lp).padEnd(5)} ${prompt.slice(0, 42)}` +
      `${!sawEos ? " NON-TERMINATING" : ""}${lp ? " LOOPED" : ""}`);
  }
  console.log(`\nglm.js greedy ${process.env.GLM_DENSE_ATTN === "1" ? "DENSE " : "SPARSE"}: ${bad}/${n} prompts non-terminating or looping`);

  glm.synchronize();
  cache.free(); ws.free(); model.free();
  if (glm instanceof ParallelOps) glm.free();
  for (const d of gpuDevices) d.free();
}

main().catch(e => { console.error("Failed:", e); process.exit(1); });
