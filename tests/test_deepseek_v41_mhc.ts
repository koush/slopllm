import assert from "node:assert/strict";
import { it } from "node:test";
import { GlmOps, bf16BytesToF32 } from "../src/glm_ops";
import { ParallelOps, ParallelTensor } from "../src/parallel_ops";
import { TensorParallelism } from "../src/device_ops";
import { getNativeAddon } from "../src/native-addon";
import { WorkspaceBase } from "../src/workspace";
import { CaptureManager } from "../src/capture-manager";
import type { Tensor } from "../src/tensor";
import { bf16Round } from "./deepseek_v41_oracles";

function floats(tensor: Tensor): Float32Array {
  const bytes = Buffer.alloc(tensor.bytes); tensor.d2h(bytes);
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.length / 4);
}

for (const world of [1, 2]) it(`DeepSeek fused post/pre preserves rounded residuals and graph ownership TP${world}`, () => {
  using gpu0 = new GlmOps(0); using gpu1 = world === 2 ? new GlmOps(1) : undefined;
  using parallel = gpu1 ? new ParallelOps([gpu0, gpu1]) : undefined;
  const ops = parallel ?? gpu0;
  const bytes = (tensor: Tensor) => { const out = Buffer.alloc(tensor.bytes); tensor.d2h(out); return out; };
  for (const [rows, dim] of [[1, 73], [4, 5120], [5, 5120], [3, 520]]) {
    using ws = new WorkspaceBase(ops), refWs = new WorkspaceBase(ops), manager = new CaptureManager(ops);
    using x = ws.alloc([rows, 4, dim], "BF16"), referenceX = refWs.alloc(x.shape, "BF16");
    using block = ws.alloc([rows, dim], "BF16"), oldPost = ws.alloc([rows, 4], "F32"), oldComb = ws.alloc([rows, 4, 4], "F32");
    using fn = ws.alloc([24, 4 * dim], "F32"), base = ws.alloc([24], "F32"), scale = ws.alloc([3], "F32");
    using previous = ws.alloc([rows, 4], "F32"), norm = ws.alloc([dim], "BF16");
    for (const tensor of [oldPost, oldComb, fn, base, scale, previous]) {
      const factor = tensor === fn ? .01 : .4;
      tensor.h2d(Buffer.from(Float32Array.from({ length: tensor.numElements }, (_, i) => Math.cos(i * .31) * factor).buffer));
    }
    norm.h2d(bf16Round(Float32Array.from({ length: dim }, (_, i) => .7 + Math.sin(i * .07))));
    for (let pass = 0; pass < 10; ++pass) {
      const values = bf16Round(Float32Array.from({ length: x.numElements }, (_, i) => Math.sin(i * .13 + pass) * 3));
      const blockValues = bf16Round(Float32Array.from({ length: block.numElements }, (_, i) => Math.cos(i * .17 + pass)));
      x.h2d(values); referenceX.h2d(values); block.h2d(blockValues);
      const withNorm = pass % 2 === 0;
      const expected = (() => {
        using residual = ops.mhcPost(referenceX, block, oldPost, oldComb);
        const result = residual.dsv41MhcPre(fn, base, scale, withNorm ? previous : undefined);
        using input = result.input; using pre = result.pre; using post = result.post; using comb = result.comb;
        using normalized = withNorm ? input.rmsnorm(norm, 1e-20) : input.viewClone();
        return { residual: bytes(residual), input: bytes(normalized), pre: floats(pre), post: floats(post), comb: floats(comb) };
      })();
      const result = manager.run({ x, block, oldPost, oldComb, fn, base, scale, previous, norm }, (_capture, a) =>
        a.x.dsv41MhcPre(a.fn, a.base, a.scale, withNorm ? a.previous : undefined, 1e-20, 1e-6, 20,
          withNorm ? a.norm : undefined, { block: a.block, post: a.oldPost, comb: a.oldComb }), ["post-pre", withNorm]) as ReturnType<Tensor["dsv41MhcPre"]>;
      using input = result.input; using pre = result.pre; using post = result.post; using comb = result.comb; using residual = result.residual!;
      assert.deepEqual(bytes(residual), expected.residual, "post residual BF16 rounding must be exact");
      assert.deepEqual(bytes(x), values, "fused preparation must not mutate its residual input");
      assert.deepEqual(bytes(block), blockValues, "fused preparation must not mutate its block input");
      for (const [actual, reference] of [[floats(pre), expected.pre], [floats(post), expected.post], [floats(comb), expected.comb]])
        for (let i = 0; i < actual.length; ++i) assert(Math.abs(actual[i] - reference[i]) < 3e-6);
      const actual = bf16BytesToF32(bytes(input)), reference = bf16BytesToF32(expected.input);
      for (let i = 0; i < actual.length; ++i) assert(Math.abs(actual[i] - reference[i]) <= Math.max(1e-7, Math.abs(reference[i]) / 128));
    }
    assert([...manager.captured.values()].some(graph => graph.graphExec));
  }
});

it("DeepSeek fused post/pre reduces partial-sum block inputs before BF16 mixing", () => {
  using gpu0 = new GlmOps(0), gpu1 = new GlmOps(1), ops = new ParallelOps([gpu0, gpu1]);
  using ws = new WorkspaceBase(ops);
  using x = ws.alloc([4, 4, 64], "BF16"), block = ws.alloc([4, 64], "BF16") as ParallelTensor;
  using post = ws.alloc([4, 4], "F32"), comb = ws.alloc([4, 4, 4], "F32");
  using fn = ws.alloc([24, 256], "F32"), base = ws.alloc([24], "F32"), scale = ws.alloc([3], "F32");
  x.fill(.5, x.numElements); post.fill(.75, post.numElements); comb.fill(.25, comb.numElements);
  fn.fill(0, fn.numElements); base.fill(0, 24); scale.fill(.1, 3);
  block.shards[0].fill(.125, block.numElements); block.shards[1].fill(.375, block.numElements);
  block.parallelism = TensorParallelism.PartialSum;
  const result = x.dsv41MhcPre(fn, base, scale, undefined, 1e-20, 1e-6, 20, undefined, { block, post, comb });
  using input = result.input; using nextPre = result.pre; using nextPost = result.post; using nextComb = result.comb; using residual = result.residual!;
  assert.equal(block.parallelism, TensorParallelism.Replicated);
  const bytes = Buffer.alloc(residual.bytes); residual.d2h(bytes);
  assert(bf16BytesToF32(bytes).every(value => value === .875));
});

it("DeepSeek mHC projection preserves FP32 weight residuals and captured row tails", () => {
  using ops = new GlmOps(0);
  for (const [rows, dim] of [[1, 64], [5, 5120], [16, 5120], [17, 5120], [33, 72], [129, 5120], [2049, 72]]) {
    using ws = new WorkspaceBase(ops); using manager = new CaptureManager(ops);
    const width = 4 * dim;
    using x = ws.alloc([rows, 4, dim], "BF16"), fn = ws.alloc([24, width], "F32");
    using base = ws.alloc([24], "F32"), scale = ws.alloc([3], "F32"), previous = ws.alloc([rows, 4], "F32");
    base.fill(0, 24); scale.fill(.1, 3); previous.fill(.25, previous.numElements);
    const weights = Float32Array.from({ length: 24 * width }, (_, i) => i < width
      ? (i % 2 ? -1 : 1 + 2 ** -13) : Math.sin(i * .17) * .01);
    fn.h2d(Buffer.from(weights.buffer));
    for (let pass = 0; pass < 6; pass++) {
      const values = bf16BytesToF32(bf16Round(Float32Array.from({ length: x.numElements }, (_, i) =>
        i < width ? (pass % 2 ? -.5 : 1) : Math.sin(i * .13 + pass) * 2)));
      x.h2d(bf16Round(values));
      const result = manager.run({ x, fn, base, scale, previous }, (_capture, input) => {
        const output = ws.alloc([rows, dim], "BF16"), pre = ws.alloc([rows, 4], "F32");
        const post = ws.alloc([rows, 4], "F32"), comb = ws.alloc([rows, 4, 4], "F32");
        const partials = ws.alloc([rows, 25, 8], "F32");
        getNativeAddon().dsv41MhcPre(ops.ctx, output.data, pre.data, post.data, comb.data,
          input.x.data, input.fn.data, input.base.data, input.scale.data, input.previous.data,
          rows, dim, 1e-20, 1e-6, 20, partials.data, 0);
        return { output, pre, post, comb, partials };
      }, ["mhc"]) as Record<"output" | "pre" | "post" | "comb" | "partials", Tensor>;
      using output = result.output; using pre = result.pre; using post = result.post;
      using comb = result.comb; using partials = result.partials;
      const p = floats(partials), gates = floats(pre), mixing = floats(comb);
      for (const row of new Set([0, Math.floor(rows / 2), rows - 1])) {
        let squared = 0;
        for (let k = 0; k < width; k++) squared += values[row * width + k] ** 2;
        const actualNorm = p.subarray((row * 25 + 24) * 8, (row * 25 + 25) * 8).reduce((a,b) => a+b, 0);
        assert(Math.abs(actualNorm - squared) <= 1e-5 * Math.max(1, squared));
        for (let c = 0; c < 24; c++) {
          // Independent high-precision dot product; no TF32 decomposition or
          // implementation reduction tree is reproduced in the oracle.
          let expected = 0;
          for (let k = 0; k < width; k++) expected += values[row * width + k] * weights[c * width + k];
          const actual = p.subarray((row * 25 + c) * 8, (row * 25 + c + 1) * 8).reduce((a,b) => a+b, 0);
          assert(Math.abs(actual - expected) <= 1e-5 * Math.max(1, Math.abs(expected)),
            `rows=${rows} dim=${dim} row=${row} coefficient=${c}: ${actual} vs ${expected}`);
          if (c < 4) {
            const gate = 1 / (1 + Math.exp(-expected / Math.sqrt(squared / width + 1e-20) * .1)) + 1e-6;
            assert(Math.abs(gates[row * 4 + c] - gate) < 2e-6);
          }
        }
        for (let col = 0; col < 4; col++) {
          let sum = 0;
          for (let r = 0; r < 4; r++) sum += mixing[row * 16 + r * 4 + col];
          assert(Math.abs(sum - 1) < 5e-6);
        }
      }
    }
    assert([...manager.captured.values()].some(graph => graph.graphExec));
  }
});

it("DeepSeek decode mHC prepares rounded collapse and RMSNorm under graph replay", () => {
  const saved = process.env.SLOPLLM_DSV41_MHC_DECODE_GROUPED;
  using ops = new GlmOps(0);
  try {
    for (const [rows, dim] of [[1, 73], [4, 5120], [5, 5120], [17, 520]]) {
      using ws = new WorkspaceBase(ops); using manager = new CaptureManager(ops);
      using referenceWs = new WorkspaceBase(ops);
      using referenceX = referenceWs.alloc([rows, 4, dim], "BF16");
      using x = ws.alloc([rows, 4, dim], "BF16"), fn = ws.alloc([24, 4 * dim], "F32");
      using base = ws.alloc([24], "F32"), scale = ws.alloc([3], "F32");
      using previous = ws.alloc([rows, 4], "F32"), norm = ws.alloc([dim], "BF16");
      fn.h2d(Buffer.from(Float32Array.from({ length: fn.numElements }, (_, i) => Math.sin(i * .17) * .01).buffer));
      previous.h2d(Buffer.from(Float32Array.from({ length: previous.numElements }, (_, i) => Math.cos(i * .3) * .7).buffer));
      norm.h2d(bf16Round(Float32Array.from({ length: dim }, (_, i) => .5 + Math.sin(i * .11))));
      base.fill(.01, 24); scale.fill(.1, 3);
      for (let pass = 0; pass < 10; ++pass) {
        const values = bf16Round(Float32Array.from({ length: x.numElements }, (_, i) => Math.sin(i * .13 + pass) * 2));
        x.h2d(values); referenceX.h2d(values);
        process.env.SLOPLLM_DSV41_MHC_DECODE_GROUPED = "0";
        const baseline = (() => {
          const result = referenceX.dsv41MhcPre(fn, base, scale, pass % 2 ? undefined : previous);
          using input = result.input; using pre = result.pre; using post = result.post; using comb = result.comb;
          using expected = input.rmsnorm(norm, 1e-20);
          const bytes = Buffer.alloc(expected.bytes); expected.d2h(bytes);
          return { pre: floats(pre), post: floats(post), comb: floats(comb), bytes };
        })();
        process.env.SLOPLLM_DSV41_MHC_DECODE_GROUPED = "1";
        const fused = manager.run({ x, fn, base, scale, previous, norm }, (_capture, args) =>
          args.x.dsv41MhcPre(args.fn, args.base, args.scale, pass % 2 ? undefined : args.previous,
            1e-20, 1e-6, 20, args.norm), ["decode-mhc", pass % 2]) as ReturnType<Tensor["dsv41MhcPre"]>;
        using actual = fused.input; using actualPre = fused.pre; using actualPost = fused.post; using actualComb = fused.comb;
        assert.deepEqual(floats(actualPre), baseline.pre);
        assert.deepEqual(floats(actualPost), baseline.post);
        assert.deepEqual(floats(actualComb), baseline.comb);
        const a = Buffer.alloc(actual.bytes), b = baseline.bytes;
        actual.d2h(a);
        const av = bf16BytesToF32(a), bv = bf16BytesToF32(b);
        for (let i = 0; i < av.length; ++i) assert(Math.abs(av[i] - bv[i]) <= Math.max(1e-7, Math.abs(bv[i]) / 128),
          `rows=${rows}, dim=${dim}, index=${i}: ${av[i]} vs ${bv[i]}`);
      }
      assert([...manager.captured.values()].some(graph => graph.graphExec));
    }
  } finally {
    if (saved === undefined) delete process.env.SLOPLLM_DSV41_MHC_DECODE_GROUPED;
    else process.env.SLOPLLM_DSV41_MHC_DECODE_GROUPED = saved;
  }
});

it("DeepSeek mHC collapse uses FP32 coefficients and rounds only the final output to BF16", () => {
  using ops = new GlmOps(0); using ws = new WorkspaceBase(ops);
  const rows = 3, dim = 513;
  using x = ws.alloc([rows, 4, dim], "BF16"), pre = ws.alloc([rows, 4], "F32");
  const input = bf16BytesToF32(bf16Round(Float32Array.from({ length: x.numElements }, (_, i) => Math.sin(i * .37) * 4)));
  const coefficients = Float32Array.from({ length: rows * 4 }, (_, i) => Math.cos(i * .3) * .7 + .000123);
  x.h2d(bf16Round(input)); pre.h2d(Buffer.from(coefficients.buffer));
  using out = x.dsv41Collapse(pre);
  const bytes = Buffer.alloc(out.bytes); out.d2h(bytes);
  const expected = Float32Array.from({ length: rows * dim }, (_, i) => {
    const row = Math.floor(i / dim), d = i % dim;
    let sum = 0;
    // BF16 x FP32 products fit in JS double; one fround models each FP32 FMA.
    for (let s = 0; s < 4; s++) sum = Math.fround(sum + input[(row * 4 + s) * dim + d] * coefficients[row * 4 + s]);
    return sum;
  });
  assert.deepEqual(bytes, bf16Round(expected));
});

it("DeepSeek mHC asynchronous tile loads preserve synchronous results exactly", () => {
  const savedMode = process.env.SLOPLLM_DSV41_MHC_PROJECTION;
  const savedPipeline = process.env.SLOPLLM_DSV41_MHC_PIPELINE;
  using ops = new GlmOps(0);
  try {
    process.env.SLOPLLM_DSV41_MHC_PROJECTION = "tf32";
    for (const [rows, dim, offset] of [[33, 72, 0], [129, 73, 0], [129, 72, 1], [129, 5120, 0], [2049, 5120, 0]]) {
      using ws = new WorkspaceBase(ops);
      using storage = ws.alloc([rows * 4 * dim + offset], "BF16");
      using view = storage.narrow(offset, rows * 4 * dim);
      using x = view.reshape([rows, 4, dim]), fn = ws.alloc([24, 4 * dim], "F32");
      using base = ws.alloc([24], "F32"), scale = ws.alloc([3], "F32"), previous = ws.alloc([rows, 4], "F32");
      fn.h2d(Buffer.from(Float32Array.from({ length: fn.numElements }, (_, i) => Math.sin(i * .17) * .01).buffer));
      base.fill(.01, 24); scale.fill(.1, 3); previous.fill(.25, previous.numElements);
      const run = (pipeline: string) => {
        process.env.SLOPLLM_DSV41_MHC_PIPELINE = pipeline;
        const result = x.dsv41MhcPre(fn, base, scale, previous);
        using input = result.input; using pre = result.pre; using post = result.post; using comb = result.comb;
        return [input, pre, post, comb].map(tensor => {
          const bytes = Buffer.alloc(tensor.bytes); tensor.d2h(bytes); return bytes;
        });
      };
      for (const value of [.5, -.25]) {
        x.fill(value, x.numElements);
        assert.deepEqual(run("1"), run("0"), `rows=${rows}, dim=${dim}, value=${value}`);
      }
    }
  } finally {
    if (savedMode === undefined) delete process.env.SLOPLLM_DSV41_MHC_PROJECTION;
    else process.env.SLOPLLM_DSV41_MHC_PROJECTION = savedMode;
    if (savedPipeline === undefined) delete process.env.SLOPLLM_DSV41_MHC_PIPELINE;
    else process.env.SLOPLLM_DSV41_MHC_PIPELINE = savedPipeline;
  }
});
