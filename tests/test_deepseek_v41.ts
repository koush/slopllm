import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { AutoTokenizer } from "@huggingface/transformers/tokenizers";
import { it } from "node:test";
import { DeepseekV41EngramHasher } from "../src/deepseek_v41_engram";
import { TensorParallelism } from "../src/device_ops";
import { GlmOps, bf16BytesToF32 } from "../src/glm_ops";
import { ParallelOps } from "../src/parallel_ops";
import { WorkspaceBase } from "../src/workspace";
import { loadDeepseekV41Config } from "../src/deepseek_v41_model";
import { CompressorCarry } from "../src/deepseek_v41_cache";
import { UsingHolder } from "../src/using-holder";
import { bf16Round, dsv41IndexQuantize, dsv41IndexScores, dsv41Rope, engramHashes } from "./deepseek_v41_oracles";

for (const tp of [1, 2]) {
  it(`DSV4 native wrapper combines local/global caches and sinks on TP${tp}`, () => {
    using gpu0 = new GlmOps(0);
    using gpu1 = tp === 2 ? new GlmOps(1) : undefined;
    const ops = gpu1 ? new ParallelOps([gpu0, gpu1]) : gpu0;
    const ws = new WorkspaceBase(ops);
    try {
      using q = ws.alloc([2, 16, 512], "BF16", undefined, TensorParallelism.Row);
      q.fill(0, 2 * 16 * 512);
      using sinks = ws.alloc([16], "F32", undefined, TensorParallelism.Column);
      sinks.fill(0, 16);
      const makeSelection = (value: 1 | 2, valid: number) => {
        const cache = ws.alloc([1, 64, 584], "U8");
        const bytes = Buffer.alloc(64 * 584);
        for (let row = 0; row < 64; row++) {
          bytes.fill(value === 1 ? 0x38 : 0x40, row * 576, row * 576 + 448);
          for (let d = 0; d < 64; d++) bytes.writeUInt16LE(value === 1 ? 0x3f80 : 0x4000, row * 576 + 448 + d * 2);
          bytes.fill(127, 64 * 576 + row * 8, 64 * 576 + row * 8 + 7);
        }
        cache.h2d(bytes);
        const indices = ws.alloc([2, 128], "I32");
        const ids = new Int32Array(256).fill(-1);
        for (let i = 0; i < valid; i++) ids[128 + i] = i;
        indices.h2d(Buffer.from(ids.buffer));
        const lengths = ws.alloc([2], "I32");
        lengths.h2d(Buffer.from(new Int32Array([0, valid]).buffer));
        return { cache, indices, lengths };
      };
      const local = makeSelection(1, 3);
      const global = makeSelection(2, 5);
      for (const mode of ["decode", "prefill"] as const) {
        const result = ops.dsv4SparseAttention(q, local, global, sinks, mode);
        using output = result.output;
        using lse = result.lse;
        const bytes = Buffer.alloc(output.bytes);
        output.d2h(bytes);
        const values = bf16BytesToF32(bytes);
        for (const value of values.subarray(0, 16 * 512)) assert.equal(value, 0);
        for (const value of values.subarray(16 * 512)) assert.ok(Math.abs(value - 13 / 9) < 0.01, `${mode}: ${value}`);
      }

      using x = ws.alloc([2, 4, 64], "BF16");
      x.fill(1, 2 * 4 * 64);
      using fn = ws.alloc([24, 256], "F32");
      fn.fill(0, 24 * 256);
      using base = ws.alloc([24], "F32");
      base.fill(0, 24);
      using scale = ws.alloc([3], "F32");
      scale.fill(1, 3);
      using previous = ws.alloc([2, 4], "F32");
      previous.fill(0.25, 8);
      const mhc = x.dsv41MhcPre(fn, base, scale, previous);
      using collapsed = mhc.input;
      using nextPre = mhc.pre;
      using post = mhc.post;
      using comb = mhc.comb;
      const bytes = Buffer.alloc(collapsed.bytes);
      collapsed.d2h(bytes);
      assert.ok(bf16BytesToF32(bytes).every(value => value === 1));
      using kvWeight = ws.alloc([512, 64], "BF16");
      using gateWeight = ws.alloc([512, 64], "BF16");
      kvWeight.fill(0.125, kvWeight.numElements);
      gateWeight.fill(-0.25, gateWeight.numElements);
      for (const gate of [undefined, gateWeight]) {
        using kv = collapsed.linear(kvWeight, { outputType: "F32" });
        using projectedGate = gate ? collapsed.linear(gate, { outputType: "F32" }) : undefined;
        using raw = projectedGate ? kv.cat([projectedGate], 1) : kv.viewClone();
        assert.equal(raw.type, "F32");
        const buffer = Buffer.alloc(raw.bytes);
        raw.d2h(buffer);
        const values = new Float32Array(buffer.buffer, buffer.byteOffset, buffer.length / 4);
        for (let i = 0; i < values.length; i++) {
          assert.equal(values[i], gate && i % 1024 >= 512 ? -16 : 8);
        }
      }
      for (const parallelism of [TensorParallelism.Column, TensorParallelism.Row]) {
        const weight = ws.alloc([64, 64], "F8_E4M3", `${parallelism}.weight`, parallelism);
        weight.h2d(Buffer.alloc(weight.bytes, 0x38));
        const scales = ws.alloc([2, 2], "F8_E8M0", `${parallelism}.scale`, parallelism);
        scales.h2d(Buffer.alloc(scales.bytes, 127));
        using result = collapsed.linear(weight);
        using full = result.replicate();
        const bytes = Buffer.alloc(full.bytes);
        full.d2h(bytes);
        assert.ok(bf16BytesToF32(bytes).every(value => value === 64));
      }
      const table = ws.alloc([512, 256], "F8_E4M3", "embedding.weight", TensorParallelism.Column);
      const tableBytes = Buffer.alloc(table.bytes, 0x38);
      tableBytes.fill(0x40, 256 * 256);
      table.h2d(tableBytes);
      const tableScales = ws.alloc([512, 8], "F8_E8M0", "embedding.scale", TensorParallelism.Column);
      tableScales.h2d(Buffer.alloc(tableScales.bytes, 127));
      using ids = ws.alloc([2, 3], "I32");
      ids.h2d(Buffer.from(new Int32Array([0, 255, 256, 510, 511, -1]).buffer));
      using embeddings = table.embedding(ids, { originalRows: 511 });
      const embeddingBytes = Buffer.alloc(embeddings.bytes);
      embeddings.d2h(embeddingBytes);
      assert.deepEqual([...bf16BytesToF32(embeddingBytes)], [1, 1, 2, 2, 0, 0].flatMap(value => Array(256).fill(value)));
      using kv = ws.alloc([2, 5 * 64], "BF16");
      kv.fill(2, 2 * 5 * 64);
      using qw = ws.alloc([4, 64], "BF16");
      qw.fill(0, 4 * 64);
      using kw = ws.alloc([4, 64], "BF16");
      kw.fill(1, 4 * 64);
      using injected = x.dsv41EngramGate(kv, qw, kw);
      const injectedBytes = Buffer.alloc(injected.bytes);
      injected.d2h(injectedBytes);
      assert.ok(bf16BytesToF32(injectedBytes).every(value => value === 2));
      // The gating consumer gathers feature-sharded KV, including shards that
      // cut across the five projected streams and multiple token rows.
      using shardedKv = ws.alloc([2, 5 * 64], "BF16", undefined, TensorParallelism.Row);
      const kvValues = Float32Array.from({ length: 2 * 5 * 64 }, (_, i) => i % 7);
      shardedKv.h2d(bf16Round(kvValues));
      using shardedInjected = x.dsv41EngramGate(shardedKv, qw, kw);
      const shardedBytes = Buffer.alloc(shardedInjected.bytes);
      shardedInjected.d2h(shardedBytes);
      assert.deepEqual(shardedBytes, bf16Round(Float32Array.from({ length: 2 * 4 * 64 }, (_, i) =>
        1 + 0.5 * kvValues[Math.floor(i / 256) * 320 + 256 + i % 64])));

      using indexInput = ws.alloc([1, 4096], "BF16", undefined, TensorParallelism.Row);
      const indexValues = bf16BytesToF32(bf16Round(Float32Array.from({ length: 4096 }, (_, i) => Math.sin(i * 0.17) * 3)));
      indexInput.h2d(bf16Round(indexValues));
      using indexPosition = ws.alloc([1], "I32");
      indexPosition.h2d(Buffer.from(Int32Array.of(129).buffer));
      using rotatedIndex = indexInput.dsv41Rope(indexPosition, true, false, 128);
      using quantizedIndex = ws.alloc([1, 4096], "F32", undefined, TensorParallelism.Row);
      ops.dsv41IndexQuantize(quantizedIndex, rotatedIndex, indexPosition);
      using fullIndex = quantizedIndex.replicate();
      const indexBytes = Buffer.alloc(quantizedIndex.bytes);
      fullIndex.d2h(indexBytes);
      const expectedIndex = dsv41IndexQuantize(bf16BytesToF32(bf16Round(dsv41Rope(indexValues, 129, true, false, 128))));
      assert.deepEqual(indexBytes, Buffer.from(expectedIndex.buffer));
      using indexKeys = ws.alloc([1, 64, 128], "F32");
      const keyValues = Float32Array.from({ length: 256 }, (_, i) => Math.cos(i * 0.23));
      const keyBytes = Buffer.alloc(indexKeys.bytes); Buffer.from(keyValues.buffer).copy(keyBytes); indexKeys.h2d(keyBytes);
      using indexWeights = ws.alloc([1, 32], "BF16");
      const weightValues = bf16BytesToF32(bf16Round(Float32Array.from({ length: 32 }, (_, i) => Math.sin(i * 0.3))));
      indexWeights.h2d(bf16Round(weightValues));
      indexPosition.h2d(Buffer.from(Int32Array.of(1).buffer));
      using indexPages = ws.alloc([1], "I32"); indexPages.fill(0, 1);
      using indexScores = ops.dsv4IndexScores(quantizedIndex, indexKeys, indexWeights, indexPosition, indexPages, 1, 2);
      const scoreBytes = Buffer.alloc(indexScores.bytes);
      indexScores.d2h(scoreBytes);
      const expectedScores = dsv41IndexScores(expectedIndex, [keyValues.subarray(0, 128), keyValues.subarray(128)], weightValues);
      assert.deepEqual(scoreBytes, Buffer.from(expectedScores.buffer));
      // DSpark's Markov codebook is vocabulary-sharded, without mHC expansion.
      using codebook = ws.alloc([16, 256], "BF16", undefined, TensorParallelism.Column);
      const codeValues = Float32Array.from({ length: 16 * 256 }, (_, i) => Math.floor(i / 256) + (i % 256) / 256);
      codebook.h2d(bf16Round(codeValues));
      using codeIds = ws.alloc([4], "I32"); codeIds.h2d(Buffer.from(Int32Array.of(0, 7, 8, 15).buffer));
      using codes = codebook.embedding(codeIds);
      const codeBytes = Buffer.alloc(codes.bytes); codes.d2h(codeBytes);
      assert.deepEqual(codeBytes, bf16Round(Float32Array.from([0, 7, 8, 15].flatMap(row => Array.from(codeValues.slice(row * 256, (row + 1) * 256))))));
      using scoresF32 = ws.alloc([2, 16], "F32", undefined, TensorParallelism.Row);
      using biasF32 = ws.alloc([2, 16], "F32", undefined, TensorParallelism.Row);
      const scoresHost = new Float32Array(32); scoresHost[7] = 1; scoresHost[8] = 1.0001; scoresHost[31] = 3;
      const biasHost = new Float32Array(32); biasHost[8] = -0.0002;
      scoresF32.h2d(Buffer.from(scoresHost.buffer)); biasF32.h2d(Buffer.from(biasHost.buffer));
      using adjustedF32 = scoresF32.add(biasF32);
      using selectedF32 = adjustedF32.argmax();
      assert.deepEqual(selectedF32.readInt32LEArray(), [7, 15]);
      using carry = new UsingHolder<CompressorCarry>(undefined!);
      using retained = new UsingHolder<CompressorCarry>(undefined!);
      using raw = ws.alloc([2, 1024], "F32");
      const rawValues = new Float32Array(2048);
      rawValues.fill(1, 0, 512);
      rawValues.fill(3, 1024, 1536);
      raw.h2d(Buffer.from(rawValues.buffer));
      using positions = ws.alloc([2], "I32");
      positions.h2d(Buffer.from(new Int32Array([0, 1]).buffer));
      using cu = ws.alloc([2], "I32");
      cu.h2d(Buffer.from(new Int32Array([0, 1]).buffer));
      using norm = ws.alloc([512], "BF16");
      norm.fill(1, 512);
      for (let token = 0; token < 2; token++) {
        using input = raw.narrow(token, 1);
        using position = positions.narrow(token, 1);
        const compressed = input.dsv41Compress(position, cu, norm, 2, carry.value?.pending, carry.value?.pendingPosition);
        carry.replace(new CompressorCarry(compressed.pending!, compressed.pendingPosition!));
        if (token === 0) retained.replace(carry.value.viewClone());
        using latent = compressed.latent;
        using valid = compressed.valid;
        const validity = Buffer.alloc(valid.bytes);
        valid.d2h(validity);
        assert.equal(validity.readInt32LE(), token);
        const bytes = Buffer.alloc(latent.bytes);
        latent.d2h(bytes);
        assert.ok(bf16BytesToF32(bytes).every(value => value === token));
      }
      // A later pair replaces the holder without clobbering a retained first half.
      using laterPosition = ws.alloc([1], "I32"); laterPosition.fill(2, 1);
      using laterInput = raw.narrow(1, 1);
      const later = laterInput.dsv41Compress(laterPosition, cu, norm, 2, carry.value.pending, carry.value.pendingPosition);
      using laterLatent = later.latent; using laterValid = later.valid;
      carry.replace(new CompressorCarry(later.pending!, later.pendingPosition!));
      const savedBytes = Buffer.alloc(retained.value.pending.bytes);
      retained.value.pending.d2h(savedBytes);
      assert.deepEqual(savedBytes, Buffer.from(rawValues.buffer, 0, 1024 * 4));
      assert.deepEqual(retained.value.pendingPosition.readInt32LEArray(), [0]);
      using restoredPosition = positions.narrow(1, 1);
      const restored = laterInput.dsv41Compress(restoredPosition, cu, norm, 2, retained.value.pending, retained.value.pendingPosition);
      using restoredLatent = restored.latent; using restoredValid = restored.valid;
      using restoredCarry = new CompressorCarry(restored.pending!, restored.pendingPosition!);
      assert.deepEqual(restoredValid.readInt32LEArray(), [1]);
      assert.throws(() => laterInput.dsv41Compress(laterPosition, cu, norm, 2,
        carry.value.pending, carry.value.pendingPosition, 1e-20, carry.value), /must not alias/);
    } finally {
      ws.free();
      if (ops instanceof ParallelOps) ops.free();
    }
  });
}

const cachedModel = path.join(process.env.HF_HOME ?? path.join(process.env.HOME!, ".cache/huggingface"),
  "hub/models--local-inference-lab--DeepSeek-V4.1-Flash-lossless-CSF");
it("V4.1 checkpoint keeps ratio-1 compression through the last backbone layer", {
  skip: !fs.existsSync(path.join(cachedModel, "refs/main")),
}, () => {
  const revision = fs.readFileSync(path.join(cachedModel, "refs/main"), "utf8").trim();
  const cfg = loadDeepseekV41Config(path.join(cachedModel, "snapshots", revision));
  assert.equal(cfg.text.compress_ratios[cfg.numHiddenLayers - 1], 1);
  assert.equal(cfg.text.compress_ratios[cfg.numHiddenLayers], 0, "SWA-only suffix belongs to draft layers");
});
it("Engram hashes match the Rust-tokenizer/NumPy reference across chunk boundaries", {
  skip: !fs.existsSync(path.join(cachedModel, "refs/main")),
}, async () => {
  const revision = fs.readFileSync(path.join(cachedModel, "refs/main"), "utf8").trim();
  const tokenizer = await AutoTokenizer.from_pretrained(path.join(cachedModel, "snapshots", revision, "metadata"), { local_files_only: true });
  const hasher = new DeepseekV41EngramHasher(tokenizer);
  const tokens = [0, 1, 2, 999, 12345, 129000];
  const full = engramHashes(hasher, tokens);
  const hashes = ["d7cbc743af4d6567168b5e0417b09bd2f6be969112b87e0fe09867f78f887014",
    "f59e780cd54b211f75b7f86d4d748ac77938f3de1352dbe0f28ca10481aaef2b"];
  full.forEach((ids, layer) => assert.equal(createHash("sha256").update(Buffer.from(ids.buffer)).digest("hex"), hashes[layer]));
  for (let split = 0; split <= tokens.length; split++) {
    const suffix = engramHashes(hasher, tokens.slice(split), tokens.slice(Math.max(0, split - 3), split));
    suffix.forEach((ids, layer) => assert.deepEqual(ids, full[layer].subarray(split * 24)));
  }
  using ops = new GlmOps(0);
  using ws = new WorkspaceBase(ops);
  using map = ws.alloc([hasher.tokenMap.length], "I32");
  map.h2d(Buffer.from(Int32Array.from(hasher.tokenMap).buffer));
  using primes = ws.alloc([48], "I32"), offsets = ws.alloc([48], "I32");
  primes.h2d(Buffer.from(Int32Array.from(hasher.primes.flat()).buffer));
  offsets.h2d(Buffer.from(Int32Array.from(hasher.offsets.flat()).buffer));
  let start = 0;
  for (const length of [2, 1, 3]) {
    using ids = ws.alloc([length], "I32"), lookback = ws.alloc([3], "I32");
    ids.h2d(Buffer.from(Int32Array.from(tokens.slice(start, start + length)).buffer));
    lookback.h2d(Buffer.from(Int32Array.from({ length: 3 }, (_, i) => tokens[start - 3 + i] ?? 2).buffer));
    using actual = ops.dsv41Hash(ids, lookback, map, primes, offsets);
    assert.deepEqual(actual.readInt32LEArray(), full.flatMap(ids => [...ids.subarray(start * 24, (start + length) * 24)]));
    start += length;
  }
  // Unequal rows, empty rows, distinct lookbacks, and enough tokens for multiple
  // CUDA blocks exercise packed addressing without crossing sequence boundaries.
  const rows = [[], tokens.slice(0, 2), [], tokens.slice(2, 3), Array.from({ length: 137 }, (_, i) => tokens[i % tokens.length]), []];
  const histories = rows.map((_, i) => [tokens[i % tokens.length], 2, tokens[(i + 1) % tokens.length]]);
  const indptr = [0];
  for (const row of rows) indptr.push(indptr.at(-1)! + row.length);
  using ids = ws.alloc([rows.flat().length], "I32");
  using lookbacks = ws.alloc([rows.length + 2, 3], "I32");
  using boundaries = ws.alloc([indptr.length], "I32");
  ids.h2d(Buffer.from(Int32Array.from(rows.flat()).buffer));
  lookbacks.h2d(Buffer.from(Int32Array.from([...histories.flat(), 2, 2, 2, 2, 2, 2]).buffer));
  boundaries.h2d(Buffer.from(Int32Array.from(indptr).buffer));
  const expected = rows.map((row, i) => engramHashes(hasher, row, histories[i]));
  using batched = ops.dsv41Hash(ids, lookbacks, map, primes, offsets, boundaries);
  assert.deepEqual(batched.readInt32LEArray(), [0, 1].flatMap(layer => expected.flatMap(hashes => [...hashes[layer]])));
});
