import assert from "node:assert/strict";
import { CaptureManager } from "./capture-manager";
import { ExecutionWorkspace } from "./execution-workspace";
import { Glm51Model } from "./glm51_model";
import { freeModelRuntime, loadModelRuntime, parseModelArgs } from "./model_cli";
import type { ExecutionOptions, ExecutionResult } from "./execution-manager";
import { Tensor } from "./tensor";

class CheckedCapture extends CaptureManager {
  override execute<T, I extends Record<string, Tensor>>(options: ExecutionOptions<I>, fn: (inputs: I) => T): ExecutionResult<T> {
    const execution = super.execute(options, fn);
    if (!this.disabled && !execution.warmup && options.key?.[0] === "glm51-dflash-decode") {
      const output = execution.result as { tokens: Tensor };
      this.ops.synchronize();
      const captured = Buffer.alloc(output.tokens.bytes); output.tokens.d2h(captured);
      output.tokens[Symbol.dispose]();
      const result = fn(options.inputs);
      this.ops.synchronize();
      const eagerOutput = (result as { tokens: Tensor }).tokens;
      const eager = Buffer.alloc(eagerOutput.bytes); eagerOutput.d2h(eager);
      assert.deepEqual(captured, eager, "Graph verification differs from eager verification on identical inputs");
      return { ...execution, result };
    }
    return execution;
  }
}

async function main() {
  const argv = process.argv.slice(2);
  const depth = argv.includes("--depth") ? Number(argv[argv.indexOf("--depth") + 1]) : 7;
  const runtime = await loadModelRuntime(parseModelArgs(argv));
  try {
    const { model, ops } = runtime;
    assert(model instanceof Glm51Model && model.dflashModel);
    using cache = model.createChatCache(32, 2, 512);
    const chunkSize = argv.includes("--long-context") ? 8192 : 512;
    using ws = new ExecutionWorkspace(ops, 2, chunkSize);
    using capture = new CheckedCapture(ops);
    capture.disabled = argv.includes("--eager");
    const count = 96;
    const prompts = ["Count from 1 to 100, separated by commas.", "List the days of the week repeatedly."].map(content => {
      const result = model.tokenizer.apply_chat_template([{ role: "user", content }],
        { tokenize: true, add_generation_prompt: true, return_tensor: false, return_dict: true, enable_thinking: false } as any) as unknown as { input_ids: number[] | number[][] };
      return (Array.isArray(result.input_ids[0]) ? result.input_ids[0] : result.input_ids) as number[];
    });
    if (argv.includes("--long-context")) {
      for (const prompt of prompts) prompt.unshift(...Array(2053 - prompt.length).fill(1));
    }
    const prefill = async () => {
      console.log(`Prefill: ${prompts.map(ids => ids.length).join(",")} tokens`);
      cache.reset(2);
      let remaining = prompts;
      while (remaining.some(ids => ids.length)) {
        const result = await model.executePrefill(ws, cache, remaining, undefined, chunkSize);
        remaining = result.remainingInputIdsList;
        console.log(`Prefill remaining: ${remaining.map(ids => ids.length).join(",")}`);
      }
    };
    await prefill();
    const expected: number[][] = [[], []];
    console.log("Ordinary greedy decode baseline");
    for await (const step of model.generateDecode(ws, cache)) {
      step.tokens.forEach((tokens, b) => expected[b].push(...tokens));
      if (expected[0].length % 16 === 0) console.log(`Baseline: ${expected[0].length}/${count}`);
      if (expected.every(tokens => tokens.length >= count)) break;
    }
    await prefill();
    const actual: number[][] = [[], []];
    const pending = cache.getPagedKV().sequences.map(sequence => sequence.targetToken!);
    const checkStep = (tokens: number[][], accepted: number[]) => tokens.forEach((row, b) => {
      const sequence = cache.getPagedKV().sequences[b];
      assert.equal(row.length, accepted[b] + 1);
      assert(accepted[b] >= 0 && accepted[b] <= depth);
      assert.equal(sequence.reportedTokenCount(), sequence.allocLen);
      assert.deepEqual(sequence.getTokenIds().slice(-row.length), [pending[b], ...row.slice(0, -1)]);
      assert.equal(sequence.targetToken, row.at(-1));
      pending[b] = row.at(-1)!;
    });
    let steps = 0, accepted = 0;
    console.log("DFlash greedy decode");
    for await (const step of model.generateDflashDecode(ws, cache, capture, undefined, depth)) {
      assert.equal(step.numDraftTokens, depth);
      checkStep(step.tokens, step.numAccepted);
      step.tokens.forEach((tokens, b) => {
        actual[b].push(...tokens);
        const sequence = cache.getPagedKV().sequences[b];
        assert.equal(sequence.reportedTokenCount(), sequence.allocLen);
        assert.equal(sequence.targetToken, tokens.at(-1));
      });
      steps++; accepted += step.numAccepted.reduce((a, b) => a + b, 0);
      console.log(`Draft step ${steps}: accepted=${step.numAccepted}, output=${actual.map(ids => ids.length)}`);
      if (steps === 3 || actual.every(tokens => tokens.length >= count)) break;
    }
    // Closing/reopening must recondition without duplicating or skipping tokens.
    for await (const step of model.generateDflashDecode(ws, cache, capture, undefined, depth)) {
      checkStep(step.tokens, step.numAccepted);
      step.tokens.forEach((tokens, b) => actual[b].push(...tokens));
      steps++; accepted += step.numAccepted.reduce((a, b) => a + b, 0);
      console.log(`Draft step ${steps}: accepted=${step.numAccepted}, output=${actual.map(ids => ids.length)}`);
      if (actual.every(tokens => tokens.length >= count)) break;
    }
    actual.forEach((tokens, b) => {
      let matched = 0;
      while (matched < count && tokens[matched] === expected[b][matched]) matched++;
      console.log(`Sequence ${b}: ordinary-decode matching prefix ${matched}/${count} (diagnostic; verification uses prefill kernels)`);
    });
    ws.assertClear();
    // Compact the batch at the generator boundary and resume from its committed prefix.
    cache.getPagedKV().removeSequence(0);
    const committed = cache.getPagedKV().sequences[0].allocLen;
    for await (const step of model.generateDflashDecode(ws, cache, capture, undefined, depth)) {
      assert.equal(step.tokens.length, 1);
      assert.equal(cache.getPagedKV().sequences[0].allocLen, committed + step.tokens[0].length);
      break;
    }
    await ops.synchronizeAsync();
    ws.assertClear();
    if (!capture.disabled) assert(capture.captured.size > 0, "No decode graphs captured");
    console.log(`DFlash greedy decode PASS: TP${ops.worldSize}, depth ${depth}, 2 sequences, ${count} tokens each, ${accepted}/${steps * 2 * depth} drafts accepted, eager/replay verification, restart and compaction`);
  } finally { freeModelRuntime(runtime); }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
