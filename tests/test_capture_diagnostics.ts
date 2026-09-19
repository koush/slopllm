import assert from "node:assert/strict";
import { it } from "node:test";
import { CaptureManager } from "../src/capture-manager";
import type { DeviceOps } from "../src/device_ops";
import type { Tensor } from "../src/tensor";

it("reports changed captured bindings without preventing replay", (t) => {
  const events: string[] = [];
  const errors: string[] = [];
  t.mock.method(console, "error", (message: string) => errors.push(message));
  const ops = {
    graphLaunch: () => events.push("launch"),
    hostSynchronizeWorld: () => events.push("hostSync"),
  } as unknown as DeviceOps;
  const manager = new CaptureManager(ops);
  manager.captured.set("test", {
    warmupSteps: 4, graphExec: 1, result: undefined, inputs: {}, capturedWorkspaces: new Set(),
    diagnosticBindings: { slot: "old", removed: "old" },
  });
  manager.run({}, () => { throw new Error("must replay"); }, ["test"], { slot: "new", added: "new" });
  assert.deepEqual(events, ["launch", "hostSync"]);
  assert.equal(errors.length, 3);
  assert.ok(errors.every(message => message.includes("BINDING MISMATCH")));
  assert.ok(errors.some(message => message.includes("captured=old current=new")));
});

it("keys replay by every shard pointer and preserves independent warmup", () => {
  let launches = 0;
  const manager = new CaptureManager({
    graphBeginCapture() {}, graphEndCapture: () => 1, graphInstantiate: () => 1,
    graphDestroy() {}, graphLaunch() { launches++; },
    hostSynchronizeWorld() {},
  } as unknown as DeviceOps);
  const tensor = (addresses: number[], shape = [8192]) => ({
    shape, type: "I32", parallelism: "replicated", pinned: false,
    memoryRanges: () => addresses.map(data => ({ data, bytes: 32768 })),
    stage() {}, unstage() {}, capture() { return this; },
    same(this: Tensor, other: Tensor) { return JSON.stringify(this.memoryRanges()) === JSON.stringify(other.memoryRanges()); },
    memcpy() { throw new Error("replay must not copy inputs"); },
  }) as unknown as Tensor;
  const original = { slots: tensor([0x10000, 0x20000]), length: tensor([0x40000, 0x50000], [4]) };
  const shifted = { ...original, slots: tensor([0x10000, 0x20400]) };
  let eager = 0;
  const run = (inputs: Record<string, Tensor>) => manager.run(inputs, () => { eager++; }, ["draft"]);
  for (let i = 0; i < 4; i++) run(original);
  assert.equal(manager.isCaptured(["draft"], original), true);
  assert.equal(manager.isCaptured(["draft"], shifted), false);
  run(shifted);
  assert.equal(eager, 5);
  run({ length: original.length, slots: tensor([0x10000, 0x20000]) });
  assert.equal(eager, 5, "equivalent pointers and reordered input names should replay");
  assert.equal(launches, 2);
  assert.equal(manager.captured.size, 2);
  assert.equal(manager.isCaptured(["draft"], { ...original, slots: tensor([0x10000, 0x20000], [4096, 2]) }), false);
});

it("does not report unchanged bindings", (t) => {
  const errors: string[] = [];
  t.mock.method(console, "error", (message: string) => errors.push(message));
  const manager = new CaptureManager({ graphLaunch() {}, hostSynchronizeWorld() {} } as unknown as DeviceOps);
  manager.captured.set("test", {
    warmupSteps: 4, graphExec: 1, result: undefined, inputs: {}, capturedWorkspaces: new Set(),
    diagnosticBindings: { slot: "same" },
  });
  manager.run({}, () => undefined, ["test"], { slot: "same" });
  assert.deepEqual(errors, []);
});
