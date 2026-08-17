// Teacher-forced NLL / perplexity over a fixed token sequence.
//
// An objective quality metric that does not depend on sampling: every config
// sees the identical token sequence, so numbers are directly comparable. Use
// this instead of eyeballing a greedy generation when evaluating kernel or
// collective changes.
//
//   npx tsx src/run_nll.ts --gpus 0,1,2,3,4,5,6,7 --arena 92 --tokens 512
import { AutoTokenizer } from "@huggingface/transformers/tokenizers";
import fs from "node:fs";
import { DeviceOps } from "./device_ops";
import { ExecutionWorkspace } from "./execution-workspace";
import { Glm51Model } from "./glm51_model";
import { bf16BytesToF32, GlmOps } from "./glm_ops";
import { resolveModelPath } from "./model_path";
import { ParallelOps } from "./parallel_ops";

const GLM51_REPO = "zai-org/GLM-5.1";
const DEFAULT_MODEL_DIR = "/mnt/storage/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/2eff962076815828e4031aec2834ac6e22fb4434/";

const argv = process.argv.slice(2);
const opt = (name: string, dflt: string) => {
  const i = argv.indexOf(name);
  return i >= 0 ? argv[i + 1] : dflt;
};

async function main() {
  const gpus = opt("--gpus", "0").split(",").map(s => parseInt(s.trim(), 10));
  const arena = parseInt(opt("--arena", "0"), 10);
  const modelDir = opt("--model-dir", DEFAULT_MODEL_DIR);
  const nTokens = parseInt(opt("--tokens", "512"), 10);
  const skip = parseInt(opt("--skip", "2000"), 10);
  const textPath = opt("--text", "big.txt");
  const maxSeqLen = Math.max(4096, 1 << Math.ceil(Math.log2(nTokens + 8)));

  const gpuDevices = gpus.map(id => new GlmOps(id, undefined, arena || undefined));
  const glm: DeviceOps = gpuDevices.length > 1 ? new ParallelOps(gpuDevices) : gpuDevices[0];

  const model = await Glm51Model.fromPretrained(glm, modelDir, false, false);
  const cache = model.createChatCache(4096, 1, maxSeqLen);
  const ws = new ExecutionWorkspace(glm, 1, maxSeqLen);

  const tokenizer = await AutoTokenizer.from_pretrained(resolveModelPath(GLM51_REPO), { local_files_only: true });
  const text = fs.readFileSync(textPath, "utf-8").slice(skip, skip + nTokens * 24);
  const encoded = tokenizer.encode(text) as number[];
  const ids = encoded.slice(0, nTokens);
  if (ids.length < nTokens) throw new Error(`only got ${ids.length} tokens from ${textPath}`);

  cache.reset(1);
  const suffixIds = cache.prefixMatch(0, ids);
  if (suffixIds.length !== ids.length) throw new Error(`unexpected prefix match: ${suffixIds.length}/${ids.length}`);

  // Decode mode: prefill a short prefix, then step the remaining tokens one at
  // a time through the decode path (teacher forced). This is the only way to
  // exercise decode-only kernels -- notably the P2P allReduce, which is gated
  // to <= 131072 elements and so never fires on a long prefill.
  if (argv.includes("--decode")) {
    const prefixLen = parseInt(opt("--prefix", "16"), 10);
    const prefix = ids.slice(0, prefixLen);
    {
      const st = ws.planPrefill(model, 1, [prefix.length], cache);
      st.setInput([prefix]);
      using hs = model.forward(st);
      using lg = st.computeLogits(hs, model);
      void lg;
    }
    cache.reportTokens(0, prefix);
    glm.synchronize();

    let nllD = 0, nD = 0, top1D = 0;
    const perToken: number[] = [];
    let cur = prefix[prefix.length - 1];
    for (let i = prefixLen; i < ids.length; i++) {
      using _t = ws.startTracking();
      const st = ws.planDecode(model, 1, cache, false);
      st.setInput([[cur]]);
      ws.positionStep(st, model);
      using hs = model.forwardModel(st);
      using lg = st.computeLogits(hs, model);
      glm.synchronize();

      const vv = lg.shape[lg.shape.length - 1];
      const b = Buffer.alloc(lg.bytes);
      lg.d2h(b);
      const row = lg.type === "BF16" ? bf16BytesToF32(b) : new Float32Array(b.buffer, b.byteOffset, b.length / 4);

      let mx = -Infinity, am = -1;
      for (let v = 0; v < vv; v++) if (row[v] > mx) { mx = row[v]; am = v; }
      let sm = 0;
      for (let v = 0; v < vv; v++) sm += Math.exp(row[v] - mx);
      const target = ids[i];
      nllD += -(row[target] - mx - Math.log(sm));
      if (am === target) top1D++;
      nD++;

      perToken.push(-(row[target] - mx - Math.log(sm)));
      cur = target;
      cache.reportTokens(0, [target]);
    }
    const mD = nllD / nD;
    console.log(`[decode] tokens=${nD} meanNLL=${mD.toFixed(5)} ppl=${Math.exp(mD).toFixed(4)} top1acc=${(top1D / nD * 100).toFixed(2)}%`);
    const dump = opt("--dump", "");
    if (dump) fs.writeFileSync(dump, JSON.stringify(perToken));
    glm.synchronize();
    cache.free();
    ws.free();
    model.free();
    if (glm instanceof ParallelOps) glm.free();
    for (const d of gpuDevices) d.free();
    return;
  }

  const state = ws.planPrefill(model, 1, [ids.length], cache);
  state.setInput([ids]);
  using hiddenStates = model.forward(state);
  using logits = state.computeLogits(hiddenStates, model, true);
  glm.synchronize();

  const V = logits.shape[logits.shape.length - 1];
  const rows = logits.numElements / V;
  if (rows !== ids.length) console.warn(`note: ${rows} logit rows for ${ids.length} tokens`);

  const buf = Buffer.alloc(logits.bytes);
  logits.d2h(buf);
  const flat = logits.type === "BF16"
    ? bf16BytesToF32(buf)
    : new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);

  // Teacher-forced NLL: row i predicts ids[i+1].
  let nll = 0;
  let n = 0;
  let top1 = 0;
  for (let i = 0; i < rows - 1; i++) {
    const off = i * V;
    let max = -Infinity, argmax = -1;
    for (let v = 0; v < V; v++) {
      const x = flat[off + v];
      if (x > max) { max = x; argmax = v; }
    }
    let sum = 0;
    for (let v = 0; v < V; v++) sum += Math.exp(flat[off + v] - max);
    const target = ids[i + 1];
    nll += -(flat[off + target] - max - Math.log(sum));
    if (argmax === target) top1++;
    n++;
  }

  const mean = nll / n;
  console.log(`tokens=${n} vocab=${V} meanNLL=${mean.toFixed(5)} ppl=${Math.exp(mean).toFixed(4)} top1acc=${(top1 / n * 100).toFixed(2)}%`);

  glm.synchronize();
  cache.free();
  ws.free();
  model.free();
  if (glm instanceof ParallelOps) glm.free();
  for (const d of gpuDevices) d.free();
}

main().catch(e => { console.error("Failed:", e); process.exit(1); });
