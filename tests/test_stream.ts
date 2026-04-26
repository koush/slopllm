import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { GlmOps, bf16BytesToF32, f32ToBf16Bytes, getNativeAddon } from "../src/glm_ops";
import { WorkspaceBase } from "../src/workspace";
import { Tensor } from "../src/tensor";

function bf16Near(a: Float32Array, b: Float32Array, atol = 0.1, rtol = 0.05): boolean {
  for (let i = 0; i < a.length; i++) {
    const diff = Math.abs(a[i] - b[i]);
    const threshold = atol + rtol * Math.abs(b[i]);
    if (diff > threshold) return false;
  }
  return true;
}

function readBf16(t: Tensor, n: number): Float32Array {
  const buf = Buffer.alloc(n * 2);
  t.d2h(buf);
  return bf16BytesToF32(buf);
}

describe("withStream", () => {
  let glm: GlmOps;
  let ws: WorkspaceBase;

  before(() => {
    glm = new GlmOps(parseInt(process.env.GLM_GPU ?? "0", 10));
    ws = new WorkspaceBase(glm);
  });

  after(() => {
    ws.free();
    glm.free();
  });

  it("runs linear on alternate stream and produces correct result (eager)", () => {
    const M = 1, N = 256, K = 256;
    const input = ws.alloc([M, K], "BF16");
    const weight = ws.alloc([N, K], "BF16");

    const inputF32 = new Float32Array(M * K);
    const weightF32 = new Float32Array(N * K);
    for (let i = 0; i < M * K; i++) inputF32[i] = (i % 17 - 8) * 0.1;
    for (let i = 0; i < N * K; i++) weightF32[i] = ((i % 13) - 6) * 0.05;

    input.h2d(f32ToBf16Bytes(inputF32));
    weight.h2d(f32ToBf16Bytes(weightF32));

    const refResult = input.linear(weight, M);
    glm.synchronize();
    const refData = readBf16(refResult, M * N);

    {
      using _ = glm.withStream(() => {
        const streamOut = input.linear(weight, M);
        glm.synchronize();
        const streamData = readBf16(streamOut, M * N);
        assert.ok(bf16Near(refData, streamData), "stream result should match reference");
      });
    }
  });

  it("overlapping withStream operations match serial execution (eager)", () => {
    const M = 1, N = 128, K = 128;
    const input = ws.alloc([M, K], "BF16");
    const wA = ws.alloc([N, K], "BF16");
    const wB = ws.alloc([N, K], "BF16");

    const inputF32 = new Float32Array(M * K);
    const wAF32 = new Float32Array(N * K);
    const wBF32 = new Float32Array(N * K);
    for (let i = 0; i < M * K; i++) inputF32[i] = (i % 11 - 5) * 0.1;
    for (let i = 0; i < N * K; i++) wAF32[i] = ((i % 7) - 3) * 0.05;
    for (let i = 0; i < N * K; i++) wBF32[i] = ((i % 9) - 4) * 0.03;

    input.h2d(f32ToBf16Bytes(inputF32));
    wA.h2d(f32ToBf16Bytes(wAF32));
    wB.h2d(f32ToBf16Bytes(wBF32));

    const refA = input.linear(wA, M);
    const refB = input.linear(wB, M);
    glm.synchronize();
    const refAData = readBf16(refA, M * N);
    const refBData = readBf16(refB, M * N);

    let streamOutA!: Tensor;
    let streamOutB!: Tensor;

    {
      using _ = glm.withStream(() => {
        streamOutA = input.linear(wA, M);
      });
      streamOutB = input.linear(wB, M);
    }

    glm.synchronize();
    const streamAData = readBf16(streamOutA, M * N);
    const streamBData = readBf16(streamOutB, M * N);

    assert.ok(bf16Near(refAData, streamAData), "stream A result should match reference");
    assert.ok(bf16Near(refBData, streamBData), "stream B (main) result should match reference");
  });

  it("withStream works inside CUDA graph capture", () => {
    const M = 1, N = 256, K = 256;
    const input = ws.alloc([M, K], "BF16");
    const weight = ws.alloc([N, K], "BF16");
    const output = ws.alloc([M, N], "BF16");
    const native = getNativeAddon();

    const inputF32 = new Float32Array(M * K);
    const weightF32 = new Float32Array(N * K);
    for (let i = 0; i < M * K; i++) inputF32[i] = (i % 17 - 8) * 0.1;
    for (let i = 0; i < N * K; i++) weightF32[i] = ((i % 13) - 6) * 0.05;

    input.h2d(f32ToBf16Bytes(inputF32));
    weight.h2d(f32ToBf16Bytes(weightF32));

    // Reference: linear on stream 0
    native.linear(glm.ctx, output.data, input.data, weight.data, M, N, K);
    glm.synchronize();
    const refData = readBf16(output, M * N);

    // Warmup: run the same linear + withStream pattern eagerly so CUDA graph capture
    // doesn't hit allocation
    for (let i = 0; i < 2; i++) {
      {
        using _ = glm.withStream(() => {
          native.linear(glm.ctx, output.data, input.data, weight.data, M, N, K);
        });
      }
      native.linear(glm.ctx, output.data, input.data, weight.data, M, N, K);
    }
    glm.synchronize();

    // Capture graph with withStream
    glm.graphBeginCapture();
    {
      using _ = glm.withStream(() => {
        native.linear(glm.ctx, output.data, input.data, weight.data, M, N, K);
      });
    }
    native.linear(glm.ctx, output.data, input.data, weight.data, M, N, K);
    const graph = glm.graphEndCapture();
    const graphExec = glm.graphInstantiate(graph);

    // Launch and verify
    output.fill(0, M * N);
    glm.graphLaunch(graphExec);
    glm.synchronize();

    const graphData = readBf16(output, M * N);
    assert.ok(bf16Near(refData, graphData), "graph with stream result should match reference");

    glm.graphExecDestroy(graphExec);
    glm.graphDestroy(graph);
  });

  it("multiple withStream scopes in same graph capture", () => {
    const M = 1, N = 128, K = 128;
    const input = ws.alloc([M, K], "BF16");
    const wA = ws.alloc([N, K], "BF16");
    const wB = ws.alloc([N, K], "BF16");
    const outA = ws.alloc([M, N], "BF16");
    const outB = ws.alloc([M, N], "BF16");
    const native = getNativeAddon();

    const inputF32 = new Float32Array(M * K);
    const wAF32 = new Float32Array(N * K);
    const wBF32 = new Float32Array(N * K);
    for (let i = 0; i < M * K; i++) inputF32[i] = (i % 11 - 5) * 0.1;
    for (let i = 0; i < N * K; i++) wAF32[i] = ((i % 7) - 3) * 0.05;
    for (let i = 0; i < N * K; i++) wBF32[i] = ((i % 9) - 4) * 0.03;

    input.h2d(f32ToBf16Bytes(inputF32));
    wA.h2d(f32ToBf16Bytes(wAF32));
    wB.h2d(f32ToBf16Bytes(wBF32));

    // Reference: serial execution
    native.linear(glm.ctx, outA.data, input.data, wA.data, M, N, K);
    native.linear(glm.ctx, outB.data, input.data, wB.data, M, N, K);
    glm.synchronize();
    const refAData = readBf16(outA, M * N);
    const refBData = readBf16(outB, M * N);

    // Warmup
    for (let i = 0; i < 2; i++) {
      {
        using _ = glm.withStream(() => {
          native.linear(glm.ctx, outA.data, input.data, wA.data, M, N, K);
        });
      }
      native.linear(glm.ctx, outB.data, input.data, wB.data, M, N, K);
    }
    glm.synchronize();

    // Capture: stream A does linear(wA), stream 0 does linear(wB)
    glm.graphBeginCapture();
    {
      using _ = glm.withStream(() => {
        native.linear(glm.ctx, outA.data, input.data, wA.data, M, N, K);
      });
    }
    native.linear(glm.ctx, outB.data, input.data, wB.data, M, N, K);
    const graph = glm.graphEndCapture();
    const graphExec = glm.graphInstantiate(graph);

    outA.fill(0, M * N);
    outB.fill(0, M * N);
    glm.graphLaunch(graphExec);
    glm.synchronize();

    assert.ok(bf16Near(refAData, readBf16(outA, M * N)), "graph stream A should match reference");
    assert.ok(bf16Near(refBData, readBf16(outB, M * N)), "graph main stream B should match reference");

    glm.graphExecDestroy(graphExec);
    glm.graphDestroy(graph);
  });

  it("stream pool exhausts and recovers", () => {
    const M = 1, N = 64, K = 64;
    const input = ws.alloc([M, K], "BF16");
    const weight = ws.alloc([N, K], "BF16");

    const inputF32 = new Float32Array(M * K);
    const weightF32 = new Float32Array(N * K);
    for (let i = 0; i < M * K; i++) inputF32[i] = (i % 7 - 3) * 0.1;
    for (let i = 0; i < N * K; i++) weightF32[i] = ((i % 5) - 2) * 0.05;

    input.h2d(f32ToBf16Bytes(inputF32));
    weight.h2d(f32ToBf16Bytes(weightF32));

    const disposables: Disposable[] = [];
    for (let i = 0; i < 7; i++) {
      disposables.push(glm.withStream(() => {
        input.linear(weight, M);
      }));
    }

    assert.throws(() => {
      glm.withStream(() => {});
    }, /No available streams/);

    for (const d of disposables) d[Symbol.dispose]();

    {
      using _ = glm.withStream(() => {
        const result = input.linear(weight, M);
        glm.synchronize();
        const data = readBf16(result, M * N);
        assert.ok(data.length === M * N, "result should have correct length");
      });
    }
  });
});
