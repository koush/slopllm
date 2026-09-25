import assert from "node:assert/strict";
import { test } from "node:test";
import { CaptureManager } from "../src/capture-manager";
import { ParallelOps } from "../src/parallel_ops";
import { GlmOps } from "../src/glm_ops";
import type { ExecutionState } from "../src/execution-workspace";

function fixture() {
  const ops = Object.assign(Object.create(ParallelOps.prototype), {
    worldSize: 8,
    devices: Array.from({ length: 8 }, () => Object.assign(Object.create(GlmOps.prototype), { smCount: 96 })),
  }) as ParallelOps;
  const state = (paddedKvLen: number, totalTokens = 4, sparseMode = true, contextParallel = true) => ({
    paddedKvLen, totalTokens, batchSize: 1, isDecode: false,
    paddedKvLenInvariant: true,
    model: { cfg: { numAttentionHeads: 64, indexTopk: 2048 } },
    cache: { getPagedKV: () => ({ sparseMode, contextParallel, ops }) },
  }) as unknown as ExecutionState;
  return { ops, state };
}

test("CP capture keys distinguish grouping changes and reuse saturated buckets", () => {
  const { ops, state } = fixture();
  assert.notDeepEqual(ops.getCaptureKeys(state(4096)), ops.getCaptureKeys(state(8192)));
  assert.deepEqual(ops.getCaptureKeys(state(65536)), ops.getCaptureKeys(state(131072)));
  assert.notDeepEqual(ops.getCaptureKeys(state(4096, 1)), ops.getCaptureKeys(state(8192, 1)));
  const s = state(4096);
  ops.getCaptureKeys(s);
  assert.equal(s.paddedKvLenInvariant, true);
  assert.deepEqual(ops.getCaptureKeys(state(4096, 4, false)), []);
  assert.deepEqual(ops.devices[0].getCaptureKeys(state(4096, 4, true, false)), ["indexerNumSplits:16"]);
  assert.deepEqual(ops.devices[0].getCaptureKeys(state(4096, 17)), ["indexerNumSplits:16"]);
});

test("capture lookup incorporates each state's backend launch keys", () => {
  const { ops, state } = fixture();
  const manager = new CaptureManager(ops);
  let keys: unknown[] = [];
  manager.isCaptured = params => { keys = params; return false; };
  const lookup = (length: number) => {
    manager.isStateCaptured({ key: ["mtp"], inputs: {}, states: [state(length, 1), state(length, 4)] });
    return keys;
  };
  assert.notDeepEqual(lookup(4096), lookup(8192));
  assert.notDeepEqual(lookup(16384), lookup(32768));
  assert.deepEqual(lookup(65536), lookup(131072));
  assert.equal(lookup(16384).filter(key => String(key).includes("sparseMlaChunksPerBlock:")).length, 16);
});

test("device capture keys own dispatch and ParallelOps preserves each device's decision", () => {
  const { ops, state } = fixture();
  const s = state(16384);
  assert.deepEqual(ops.devices[0].getCaptureKeys(s), ["indexerNumSplits:64", "sparseMlaChunksPerBlock:6"]);
  Object.assign(ops.devices[1], { smCount: 188 });
  const keys = ops.getCaptureKeys(s);
  assert.equal(keys[1], "device:0:sparseMlaChunksPerBlock:6");
  assert.equal(keys[3], "device:1:sparseMlaChunksPerBlock:3");
  assert.deepEqual(ops.devices[0].getCaptureKeys(state(16384, 17)), ["indexerNumSplits:64"]);
  const decode = Object.assign(state(16384, 17), { isDecode: true });
  assert.deepEqual(ops.devices[0].getCaptureKeys(decode), ["indexerNumSplits:64", "sparseMlaChunksPerBlock:2"]);
});

test("indexer split keys reuse launch budgets without requesting length variants", () => {
  const { ops, state } = fixture();
  const key = (length: number) => ops.devices[0].getCaptureKeys(state(length, 17));
  assert.deepEqual(key(1), key(256));
  assert.notDeepEqual(key(256), key(257));
  assert.deepEqual(key(257), key(512));
  assert.deepEqual(key(65536), key(1048576));
  const s = state(1024, 17);
  Object.assign(s, { getGraphVariantPaddedKvLen: () => { throw new Error("unexpected length variant"); } });
  assert.deepEqual(ops.devices[0].getCaptureKeys(s), ["indexerNumSplits:4"]);
  assert.equal(s.paddedKvLenInvariant, true);
});
