import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { it } from "node:test";
import { ChatModel } from "../src/chat_model";
import { CsfContainer } from "../src/csf_checkpoint";
import { type DeviceOps } from "../src/device_ops";
import { Glm51Model } from "../src/glm51_model";
import { GlmOps, GlmTensor } from "../src/glm_ops";
import { ParallelOps } from "../src/parallel_ops";
import { modelArenaLayoutSignatures } from "../src/model_cli";
import { WorkspaceBase } from "../src/workspace";
import { buildCsfFixture, makeScaleMatrix } from "./csf_fixture";

it("GPU CSF expansion is byte-exact for full matrices, odd windows, and empty exceptions", () => {
  using ops = new GlmOps(0);
  using ws = new WorkspaceBase(ops);
  for (const [rows, columns] of [[32, 8], [48, 22], [2048, 384], [6144, 128]]) {
    for (const positions of [[], [0, columns - 1, columns, rows * columns / 2, rows * columns - 1]]) {
      const matrix = makeScaleMatrix(rows, columns, {
        seed: 17, exceptionPositions: positions, exceptionValues: positions.map((_, i) => 0x30 + i),
      });
      using fixed = ws.alloc([matrix.fixed.length], "U8");
      using exceptions = ws.alloc([Math.max(4, matrix.exceptions.length)], "U8");
      using status = ws.alloc([1], "I32");
      fixed.h2d(matrix.fixed);
      if (matrix.exceptions.length) exceptions.h2d(matrix.exceptions);
      for (const [r, c, nr, nc] of [[0, 0, rows, columns], [7, 1, 19, columns - 3]]) {
        using out = ws.alloc([nr, nc], "F8_E4M3");
        ops.decodeCsfScales(out, fixed, exceptions, status, rows, columns, r, c, matrix.exceptions.length / 4);
        ops.synchronize();
        assert.deepEqual(status.readInt32LEArray(), [0]);
        const expected = Buffer.concat(Array.from({ length: nr }, (_, i) =>
          matrix.logical.subarray((r + i) * columns + c, (r + i) * columns + c + nc)));
        const actual = Buffer.alloc(out.bytes);
        out.d2h(actual);
        assert.deepEqual(actual, expected);
      }
    }
  }
});

it("GPU CSF rejects malformed planes and validates scales after exception replacement", () => {
  using ops = new GlmOps(0);
  using ws = new WorkspaceBase(ops);
  const matrix = makeScaleMatrix(32, 8, { seed: 3, exceptionPositions: [0, 255], exceptionValues: [0x38, 0x39] });
  using fixed = ws.alloc([matrix.fixed.length], "U8");
  using exceptions = ws.alloc([matrix.exceptions.length], "U8");
  using status = ws.alloc([1], "I32");
  using out = ws.alloc([32, 8], "F8_E4M3");
  for (const corrupt of [
    (f: Buffer, _e: Buffer) => { f[0] = 241; },
    (_f: Buffer, e: Buffer) => { e.writeUInt32LE(0x38000100, 0); },
    (_f: Buffer, e: Buffer) => { e.writeUInt32LE(e.readUInt32LE(0), 4); },
    (_f: Buffer, e: Buffer) => { e[3] = 0x7f; },
    (f: Buffer, _e: Buffer) => { f[1] = 0x80; },
  ]) {
    const f = Buffer.from(matrix.fixed), e = Buffer.from(matrix.exceptions);
    corrupt(f, e);
    fixed.h2d(f); exceptions.h2d(e);
    ops.decodeCsfScales(out, fixed, exceptions, status, 32, 8);
    ops.synchronize();
    assert.notEqual(status.readInt32LEArray()[0], 0);
  }
  // An invalid fixed-stream placeholder is legal when an exception replaces it.
  const f = Buffer.from(matrix.fixed);
  f[0] = 0x70;
  f[16] = 0x0f;
  f.fill(0, 17, 20);
  fixed.h2d(f); exceptions.h2d(matrix.exceptions);
  ops.decodeCsfScales(out, fixed, exceptions, status, 32, 8);
  ops.synchronize();
  assert.deepEqual(status.readInt32LEArray(), [0]);
});

// Exercise the real GLM tensor loader without requiring a complete forward-model fixture.
function weightLoader(ops: DeviceOps): Glm51Model {
  return Reflect.construct(Glm51Model, [ops, {
    eosTokenIds: [0], qkRopeHeadDim: 64, ropeTheta: 1000000, tieWordEmbeddings: false,
  }]) as Glm51Model;
}

it("CSF owner and allocation-only replay reproduce the arena and named-weight layout", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "glm-csf-replay-"));
  const previous = process.env.SLOPLLM_MODEL_LOAD_REPLAY;
  try {
    buildCsfFixture(dir);
    const layouts: { arena: string | undefined; model: string | undefined }[] = [];
    for (const replay of [false, true]) {
      if (replay) process.env.SLOPLLM_MODEL_LOAD_REPLAY = "1";
      else delete process.env.SLOPLLM_MODEL_LOAD_REPLAY;
      using ops = new GlmOps(0, undefined, 0.03125);
      using model = weightLoader(ops);
      await ChatModel.prototype["loadWeights"].call(model, dir);
      const arena = ops.arenaLayoutSignature();
      const modelLayout = modelArenaLayoutSignatures(model, [ops]).get(0);
      assert.ok(arena && modelLayout);
      layouts.push({ arena, model: modelLayout });
    }
    assert.deepEqual(layouts[1], layouts[0]);
  } finally {
    if (previous === undefined) delete process.env.SLOPLLM_MODEL_LOAD_REPLAY;
    else process.env.SLOPLLM_MODEL_LOAD_REPLAY = previous;
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

it("concurrent CSF loading reports an early corrupt scale at the final validation boundary", async t => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "glm-csf-invalid-"));
  try {
    buildCsfFixture(dir);
    const container = CsfContainer.open(dir);
    try {
      const name = container.contract.scales.keys().next().value!;
      const planes = container.compressedPlanes(name)!;
      const fd = fs.openSync(path.join(dir, "tensors", planes.file), "r+");
      try { fs.writeSync(fd, Buffer.from([241]), 0, 1, planes.fixedOffset); }
      finally { fs.closeSync(fd); }
    } finally {
      container.close();
    }
    using ops = new GlmOps(0);
    using model = weightLoader(ops);
    const synchronize = ops.synchronizeAsync.bind(ops);
    const syncMock = t.mock.method(ops, "synchronizeAsync", synchronize);
    await assert.rejects(ChatModel.prototype["loadWeights"].call(model, dir), /Invalid CSF scale planes/);
    assert.equal(syncMock.mock.callCount(), 1);
    // Later valid decodes must not clear the earlier error flag.
    assert.ok(model.tensors.has("model.layers.3.mlp.experts.0.down_proj.weight_weight_scale"));
    syncMock.mock.restore();
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

it("CSF loads normal named GLM scales concurrently, shards both TP axes, and skips replay writes", async t => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "glm-csf-gpu-"));
  const originalReplay = process.env.SLOPLLM_MODEL_LOAD_REPLAY;
  const originalRead = CsfContainer.prototype.readTensor;
  try {
    const fixture = buildCsfFixture(dir);
    const compressedScales = [...fixture.logical.keys()].filter(name => name.endsWith(".weight_scale"));
    // Loading must never invoke the CPU scale decoder.
    CsfContainer.prototype.readTensor = function(name) {
      assert.notEqual(this.locate(name).kind, "compressed", "CPU scale expansion during GPU loading");
      return originalRead.call(this, name);
    };
    using gpu0 = new GlmOps(0);
    using gpu1 = new GlmOps(1);
    using parallel = new ParallelOps([gpu0, gpu1]);
    const originalMmapLoad = GlmTensor.prototype.mmapLoad;
    let inFlight = 0, peakInFlight = 0;
    t.mock.method(GlmTensor.prototype, "mmapLoad", async function(this: GlmTensor,
      ...args: Parameters<GlmTensor["mmapLoad"]>) {
      peakInFlight = Math.max(peakInFlight, ++inFlight);
      try { return await originalMmapLoad.apply(this, args); }
      finally { inFlight--; }
    });
    for (const ops of [gpu0, parallel]) {
      using model = weightLoader(ops);
      peakInFlight = 0;
      const synchronize = ops.synchronizeAsync.bind(ops);
      const syncMock = t.mock.method(ops, "synchronizeAsync", synchronize);
      await ChatModel.prototype["loadWeights"].call(model, dir);
      assert.ok(peakInFlight > 1, "mapped uploads should overlap");
      assert.equal(syncMock.mock.callCount(), 1, "only the final load boundary should synchronize");
      syncMock.mock.restore();
      for (const name of compressedScales) {
        const storedName = name.replace(/\.weight_scale$/, ".weight_weight_scale");
        const tensor = model.tensors.get(storedName)!;
        assert.equal(tensor.type, "F8_E4M3");
        const actual = Buffer.alloc(tensor.bytes);
        tensor.d2h(actual);
        assert.deepEqual(actual, fixture.logical.get(name)!.data);
      }
      const source = CsfContainer.open(dir);
      try {
        const name = compressedScales[0];
        const csf = source.compressedPlanes(name)!;
        const tensor = model.tensors.get(name.replace(/\.weight_scale$/, ".weight_weight_scale"))!;
        const before = Buffer.alloc(tensor.bytes);
        tensor.d2h(before);
        process.env.SLOPLLM_MODEL_LOAD_REPLAY = "1";
        await model["loadTensorBytes"](tensor, {
          name, meta: source.locate(name).meta,
          csf: { ...csf,
            mapping: () => { throw new Error("Replay mapped compressed data"); },
            read: () => { throw new Error("Replay read compressed data"); } },
          mapping: () => { throw new Error("Replay mapped data"); },
          read: () => { throw new Error("Replay read data"); },
          readSlice: () => { throw new Error("Replay read slice"); },
        });
        await model["drainLoadTransfers"]();
        delete process.env.SLOPLLM_MODEL_LOAD_REPLAY;
        const after = Buffer.alloc(tensor.bytes);
        tensor.d2h(after);
        assert.deepEqual(after, before);
      } finally {
        source.close();
      }
    }
  } finally {
    CsfContainer.prototype.readTensor = originalRead;
    if (originalReplay === undefined) delete process.env.SLOPLLM_MODEL_LOAD_REPLAY;
    else process.env.SLOPLLM_MODEL_LOAD_REPLAY = originalReplay;
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
