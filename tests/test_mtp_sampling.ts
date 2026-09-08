import assert from "node:assert/strict";
import { after, before, describe, it, type TestContext } from "node:test";
import type { MtpDraftBatch, SamplingParams } from "../src/chat_model";
import { TensorParallelism } from "../src/device_ops";
import { MemcpyKind } from "../src/enums";
import { GlmOps, f32ToBf16Bytes } from "../src/glm_ops";
import { ParallelOps, ParallelTensor } from "../src/parallel_ops";
import { LinearMtpSamplingWorkspace, SamplingWorkspace } from "../src/sampling";
import type { Tensor } from "../src/tensor";
import { WorkspaceBase } from "../src/workspace";

const B = 2;
const D = 3;
const V = 512;
const greedy: SamplingParams = {
  temperature: 0, topK: 1, topP: 1,
  repetitionPenalty: 1, presencePenalty: 0, repetitionPenaltyWindow: 0,
};

function readI32(tensor: Tensor): number[] {
  tensor.workspace.glm.synchronize();
  const buf = Buffer.alloc(tensor.bytes);
  tensor.d2h(buf);
  tensor.workspace.glm.synchronize();
  return Array.from({ length: tensor.numElements }, (_, i) => buf.readInt32LE(i * 4));
}

for (const parallel of [false, true]) {
  describe(`LinearMtpSamplingWorkspace (${parallel ? "two GPUs" : "single GPU"})`, {
    skip: parallel && process.env.TEST_MULTIGPU !== "1",
    concurrency: false,
  }, () => {
    const devices: GlmOps[] = [];
    let glm: GlmOps | ParallelOps;
    let ws: WorkspaceBase;
    let target: SamplingWorkspace;
    let sampler: LinearMtpSamplingWorkspace;

    before(() => {
      const first = parseInt(process.env.GLM_GPU ?? "0", 10);
      devices.push(new GlmOps(first));
      if (parallel) devices.push(new GlmOps(parseInt(process.env.GLM_GPU_SECOND ?? "1", 10)));
      glm = parallel ? new ParallelOps(devices) : devices[0];
      ws = new WorkspaceBase(glm);
      // Separate workspaces own identically named sampling buffers; no model or KV cache.
      target = new SamplingWorkspace(glm, B * (D + 1), V, 0);
      sampler = new LinearMtpSamplingWorkspace(glm, B, D, target);
    });

    after(() => {
      try {
        glm?.synchronize();
      } finally {
        sampler?.free();
        target?.free();
        ws?.free();
        if (glm instanceof ParallelOps) glm.free();
        for (const device of devices) device.free();
      }
    });

    function logits(peaks: number[]): Tensor {
      const tensor = ws.alloc([peaks.length, V], "BF16");
      uploadLogits(tensor, peaks);
      return tensor;
    }

    function uploadLogits(tensor: Tensor, peaks: number[]): void {
      const values = new Float32Array(peaks.length * V).fill(-16);
      peaks.forEach((peak, row) => {
        values[row * V + peak] = 2;
        values[row * V + peak + 1] = 1;
        values[row * V + peak + 2] = 0;
      });
      tensor.h2d(f32ToBf16Bytes(values));
    }

    function noFullGather(t: TestContext, maxK: number) {
      assert.ok(glm instanceof ParallelOps);
      const allGather = ParallelTensor.prototype.allGather;
      const allGatherMultiple = glm.allGatherMultiple;
      const batchMock = t.mock.method(glm, "sampleBatch", () => {
        throw new Error("sampleDraft must not call ParallelOps.sampleBatch");
      });
      const gatherMock = t.mock.method(ParallelTensor.prototype, "allGather", function (this: ParallelTensor, ...args: Parameters<ParallelTensor["allGather"]>) {
        assert.ok(this.numElements < B * V, "sampleDraft must not gather full logits");
        return allGather.apply(this, args);
      });
      const multipleMock = t.mock.method(glm, "allGatherMultiple", function (this: ParallelOps, ...args: Parameters<ParallelOps["allGatherMultiple"]>) {
        for (const tensor of args[0]) assert.deepEqual(tensor.shape, [B, maxK * 2]);
        return allGatherMultiple.apply(this, args);
      });
      return {
        batchMock, multipleMock,
        [Symbol.dispose]() {
          batchMock.mock.restore();
          gatherMock.mock.restore();
          multipleMock.mock.restore();
        },
      };
    }

    function draft(peaks: number[], params: SamplingParams[]): MtpDraftBatch {
      sampler.updateSampler(params);
      sampler.prepareDraft(peaks.length, D);
      const treeTokens = peaks.map(() => [] as number[]);
      using input = logits(peaks);
      for (let depth = 0; depth < D; depth++) {
        using tokens = sampler.sampleDraft(input, depth);
        assert.deepEqual(tokens.shape, [peaks.length]);
        const sampled = readI32(tokens);
        if (tokens instanceof ParallelTensor) {
          for (const shard of tokens.shards) assert.deepEqual(readI32(shard), sampled);
        }
        sampled.forEach((token, row) => treeTokens[row].push(token));
      }
      glm.synchronize();
      return { targetTokens: peaks, treeTokens, topks: [1, 1, 1], proposal: sampler.finishDraft() };
    }

    function guardResidentTransfers(t: TestContext, resident: LinearMtpSamplingWorkspace) {
      const copies = [] as { source: Tensor; sourceOffset: number; destinationOffset: number }[];
      for (const name of ["qHostProbs", "qHostIds", "qInputProbsH", "qInputIdsH"]) {
        const tensor = resident.tensors.get(name)!;
        // Guard narrow too: draft downloads normally target views of qHost buffers.
        for (const method of ["narrow", "memcpy", "readPinnedBuffer", "withPinnedBuffer"] as const) {
          t.mock.method(tensor, method, () => { throw new Error(`Resident proposals touched ${name}.${method}`); });
        }
      }
      for (const name of ["qInputProbs", "qInputIds"]) {
        const tensor = resident.tensors.get(name)!;
        t.mock.method(tensor, "memcpy", () => { throw new Error(`Resident proposals uploaded ${name}`); });
        const memcpy2d = tensor.memcpy2d;
        t.mock.method(tensor, "memcpy2d", function (this: Tensor, ...args: Parameters<Tensor["memcpy2d"]>) {
          const [destinationOffset, dpitch, source, sourceOffset, spitch, width, height, kind] = args;
          assert.equal(kind, MemcpyKind.DeviceToDevice);
          assert.equal(source, resident.tensors.get(name === "qInputProbs" ? "qDraftProbs" : "qDraftIds"));
          assert.equal(width, resident.capacity * 4);
          assert.equal(dpitch, width);
          assert.equal(spitch, B * width);
          assert.equal(height, D);
          copies.push({ source, sourceOffset, destinationOffset });
          return memcpy2d.apply(this, args);
        });
      }
      return copies;
    }

    for (const stochastic of [false, true]) {
      it(`matches host ${stochastic ? "stochastic" : "greedy"} proposals with resident reorder and compaction without q host transfers`, (t) => {
        using resident = new LinearMtpSamplingWorkspace(glm, B, D, target, true);
        const copies = guardResidentTransfers(t, resident);
        const params = stochastic
          ? [{ ...greedy, temperature: 1, topK: 3 }, { ...greedy, temperature: 0.5, topK: 2 }]
          : [greedy, greedy];
        const seed = Buffer.alloc(4);
        seed.writeUInt32LE(123456);
        for (const mode of [sampler, resident]) {
          mode.updateSampler(params);
          mode.prepareDraft(B, D);
          mode.draftSampler.stepCounter.h2d(seed);
        }
        glm.synchronize();
        assert.equal(sampler.captureKey, `linear:${stochastic ? 3 : 1}`);
        assert.equal(resident.captureKey, `${sampler.captureKey}:gpu`);
        const treeTokens: number[][] = [[], []];
        for (let depth = 0; depth < D; depth++) {
          using input = logits([7 + depth * 4, 29 + depth * 4]);
          using hostTokens = sampler.sampleDraft(input, depth);
          using gpuTokens = resident.sampleDraft(input, depth);
          const expected = readI32(hostTokens);
          assert.deepEqual(readI32(gpuTokens), expected);
          expected.forEach((token, row) => treeTokens[row].push(token));
        }
        glm.synchronize();
        const host = sampler.finishDraft();
        const device = resident.finishDraft();
        assert.equal(host.device, undefined);
        assert.equal(host.probabilities.length, B);
        assert.equal(host.tokenIds.length, B);
        assert.deepEqual(device.probabilities, []);
        assert.deepEqual(device.tokenIds, []);
        assert.equal(device.capacity, host.capacity);
        assert.ok(device.device);
        assert.deepEqual(device.device.owner, {}, "owner token must not retain a traversable workspace");
        assert.deepEqual(device.device.rows, [0, 1]);
        assert.deepEqual(readI32(resident.draftSampler.stepCounter), readI32(sampler.draftSampler.stepCounter));
        const original: MtpDraftBatch = { targetTokens: [7, 29], treeTokens, topks: [1, 1, 1], proposal: host };
        for (const order of [[0, 1], [1, 0], [1], [0, 1]]) {
          const batch: MtpDraftBatch = {
            ...original,
            targetTokens: order.map(row => original.targetTokens[row]),
            treeTokens: order.map(row => original.treeTokens[row]),
            proposal: { ...host, probabilities: order.map(row => host.probabilities[row]), tokenIds: order.map(row => host.tokenIds[row]) },
          };
          const gpuBatch: MtpDraftBatch = { ...batch, proposal: { ...device, device: { ...device.device, rows: order } } };
          prepare(batch, order.map(row => params[row]));
          resident.updateSampler(order.map(row => params[row]));
          copies.length = 0;
          resident.prepareVerification(gpuBatch);
          assert.equal(copies.length, order.length * 2);
          copies.forEach((copy, i) => {
            assert.equal(copy.sourceOffset, order[Math.floor(i / 2)] * resident.capacity * 4);
            assert.equal(copy.destinationOffset, Math.floor(i / 2) * D * resident.capacity * 4);
          });
          using input = logits(batch.targetTokens.flatMap(peak => [peak, peak + 4, peak + 8, peak]));
          for (let repeat = 0; repeat < (stochastic ? 12 : 1); repeat++) {
            const hostResult = sampler.verify(input);
            using hostTokens = hostResult.tokens;
            using hostCounts = hostResult.numAccepted;
            const gpuResult = resident.verify(input);
            using gpuTokens = gpuResult.tokens;
            using gpuCounts = gpuResult.numAccepted;
            check(hostResult, batch, stochastic);
            check(gpuResult, gpuBatch, stochastic);
            assert.deepEqual(readI32(gpuCounts), readI32(hostCounts));
            if (!stochastic) assert.deepEqual(readI32(gpuTokens), readI32(hostTokens));
          }
        }

        const gpuBatch = { ...original, proposal: device };
        assert.throws(() => sampler.prepareVerification(gpuBatch), /host linear MTP proposal/);
        assert.throws(() => resident.prepareVerification(original), /GPU linear MTP proposal/);
        for (const descriptor of [
          { ...device.device, owner: sampler },
          ...[[-1, 0], [0, B], [0, 0.5], [0]].map(rows => ({ ...device.device!, rows })),
        ]) {
          assert.throws(() => resident.prepareVerification({ ...gpuBatch, proposal: { ...device, device: descriptor } }), /GPU linear MTP proposal/);
        }
        resident.prepareDraft(B, D);
        assert.throws(() => resident.prepareVerification(gpuBatch), /stale/);
        const next = resident.finishDraft();
        assert.equal(next.device!.owner, device.device.owner);
        assert.equal(next.device!.generation, device.device.generation + 1);
        glm.synchronize();
      });
    }

    function prepare(batch: MtpDraftBatch, params: SamplingParams[]): void {
      // Scheduler policy is per active sequence, target policy is sequence-major [B, D+1].
      sampler.updateSampler(params);
      const expanded = params.flatMap(param => Array.from({ length: D + 1 }, () => param));
      target.updateSampler(expanded, expanded.map(() => []));
      sampler.prepareVerification(batch);
    }

    function check(result: ReturnType<LinearMtpSamplingWorkspace["verify"]>, batch: MtpDraftBatch, stochastic = false): void {
      assert.deepEqual(result.tokens.shape, [batch.targetTokens.length * (D + 1)]);
      assert.deepEqual(result.numAccepted.shape, [batch.targetTokens.length]);
      const tokenShards = result.tokens instanceof ParallelTensor ? result.tokens.shards : [result.tokens];
      const countShards = result.numAccepted instanceof ParallelTensor ? result.numAccepted.shards : [result.numAccepted];
      for (const counts of countShards) assert.deepEqual(readI32(counts), batch.targetTokens.map(() => D));
      for (const tokens of tokenShards) {
        const actual = readI32(tokens);
        batch.treeTokens.forEach((expected, row) => {
          assert.deepEqual(actual.slice(row * (D + 1), row * (D + 1) + D), expected);
          const bonus = actual[row * (D + 1) + D];
          if (stochastic) assert.ok(bonus >= batch.targetTokens[row] && bonus <= batch.targetTokens[row] + 2);
          else assert.equal(bonus, batch.targetTokens[row]);
        });
      }
    }

    it("validates constructor, policy, draft and verification shapes", () => {
      using unprepared = new LinearMtpSamplingWorkspace(glm, B, D, target);
      using unpreparedInput = logits([7, 19]);
      assert.throws(() => unprepared.sampleDraft(unpreparedInput, 0), /unprepared batch/);
      assert.throws(() => unprepared.finishDraft(), /not prepared/);
      for (const [batch, depth] of [[0, D], [1.5, D], [B, 0], [B, 1.5], [B + 1, D]]) {
        assert.throws(() => new LinearMtpSamplingWorkspace(glm, batch, depth, target), /valid batch\/depth/);
      }
      for (const patch of [
        { temperature: NaN }, { topP: Infinity }, { topK: 1.5 },
        { temperature: 1, topK: 257 }, { repetitionPenalty: 1.1 }, { presencePenalty: 0.1 },
      ]) assert.throws(() => sampler.updateSampler([{ ...greedy, ...patch }]), /Linear MTP requires/);
      assert.throws(() => sampler.updateSampler([greedy, greedy, greedy]), /maxBatchSize/);
      sampler.updateSampler([greedy, greedy]);
      assert.equal(sampler.captureKey, "linear:1");
      for (const [batch, depth] of [[0, D], [1, D], [3, D], [B, D - 1]]) {
        assert.throws(() => sampler.prepareDraft(batch, depth), /batch\/depth/);
      }
      sampler.prepareDraft(B, D);
      using input = logits([7, 19]);
      for (const depth of [-1, D, 0.5]) assert.throws(() => sampler.sampleDraft(input, depth), /Invalid linear MTP draft/);
      using wrongRows = logits([7]);
      assert.throws(() => sampler.sampleDraft(wrongRows, 0), /row capacities/);
      const batch = draft([7, 19], [greedy, greedy]);
      const proposal = batch.proposal!;
      for (const invalid of [
        { ...batch, proposal: undefined },
        { ...batch, topks: [1, 2, 1] },
        { ...batch, treeTokens: [[7], [19]] },
        { ...batch, treeTokens: [[7, 7, V], [19, 19, 19]] },
        { ...batch, proposal: { ...proposal, capacity: 1 } },
        { ...batch, proposal: { ...proposal, probabilities: proposal.probabilities.slice(1) } },
        { ...batch, proposal: { ...proposal, tokenIds: [Buffer.alloc(4), proposal.tokenIds[1]] } },
      ]) assert.throws(() => sampler.prepareVerification(invalid), /Invalid (host )?linear MTP proposal/);
      target.updateSampler([greedy, greedy], [[], []]);
      assert.throws(() => sampler.verify(input), /B \* \(D \+ 1\) rows/);
      prepare(batch, [greedy, greedy]);
      assert.throws(() => sampler.verify(input), /row capacities/);
    });

    it("transposes depth-major q into owned B=2, D=3 host proposal rows", () => {
      sampler.updateSampler([greedy, greedy]);
      sampler.prepareDraft(B, D);
      for (let depth = 0; depth < D; depth++) {
        using input = logits([10 + depth, 30 + depth]);
        using tokens = sampler.sampleDraft(input, depth);
        assert.deepEqual(readI32(tokens), [10 + depth, 30 + depth]);
      }
      glm.synchronize();
      const proposal = sampler.finishDraft();
      assert.equal(proposal.capacity, 256);
      assert.equal(proposal.probabilities.length, B);
      assert.equal(proposal.tokenIds.length, B);
      for (let row = 0; row < B; row++) {
        assert.equal(proposal.probabilities[row].length, D * 256 * 4);
        assert.equal(proposal.tokenIds[row].length, D * 256 * 4);
        for (let depth = 0; depth < D; depth++) {
          const offset = depth * 256 * 4;
          assert.equal(proposal.tokenIds[row].readInt32LE(offset), (row === 0 ? 10 : 30) + depth);
          assert.equal(proposal.probabilities[row].readFloatLE(offset), 1);
          for (let k = 1; k < 256; k++) assert.equal(proposal.probabilities[row].readFloatLE(offset + k * 4), 0);
        }
      }
      const saved = [...proposal.probabilities, ...proposal.tokenIds].map(buf => Buffer.from(buf));
      draft([90], [greedy]);
      assert.deepEqual([...proposal.probabilities, ...proposal.tokenIds], saved);
    });

    for (const stochastic of [false, true]) {
      it(`${stochastic ? "stochastic" : "greedy"} p=q fully accepts after host reorder, compaction and regrowth`, () => {
        const params = stochastic
          ? [{ ...greedy, temperature: 1, topK: 3 }, { ...greedy, temperature: 0.5, topK: 2, topP: 0.9 }]
          : [greedy, greedy];
        const original = draft([7, 29], params);
        assert.equal(sampler.captureKey, stochastic ? "linear:3" : "linear:1");
        if (stochastic) {
          original.proposal!.probabilities.forEach((probs, row) => {
            const ids = original.proposal!.tokenIds[row];
            const weights = Array.from({ length: params[row].topK }, (_, k) => Math.exp(-k / params[row].temperature));
            const sum = weights.reduce((a, b) => a + b, 0);
            for (let depth = 0; depth < D; depth++) {
              for (let k = 0; k < 256; k++) {
                const offset = (depth * 256 + k) * 4;
                const expected = k < weights.length ? weights[k] / sum : 0;
                assert.ok(Math.abs(probs.readFloatLE(offset) - expected) < 1e-6, `q row=${row} depth=${depth} k=${k}`);
                if (expected > 0) assert.equal(ids.readInt32LE(offset), original.targetTokens[row] + k);
              }
            }
          });
        }
        // Reuse owned host proposals while the active batch shrinks and grows.
        for (const order of [[0, 1], [1, 0], [1], [0, 1]]) {
          const batch: MtpDraftBatch = {
            targetTokens: order.map(i => original.targetTokens[i]),
            treeTokens: order.map(i => original.treeTokens[i]),
            topks: original.topks,
            proposal: {
              capacity: original.proposal!.capacity,
              probabilities: order.map(i => original.proposal!.probabilities[i]),
              tokenIds: order.map(i => original.proposal!.tokenIds[i]),
            },
          };
          prepare(batch, order.map(i => params[i]));
          assert.equal(target.batchSize, order.length * (D + 1));
          using input = logits(batch.targetTokens.flatMap(peak => Array(D + 1).fill(peak)));
          for (let repeat = 0; repeat < (stochastic ? 12 : 1); repeat++) {
            const result = sampler.verify(input);
            using tokens = result.tokens;
            using counts = result.numAccepted;
            check(result, batch, stochastic);
          }
        }
        const compact = draft([41], [params[1]]);
        prepare(compact, [params[1]]);
        using input = logits(Array(D + 1).fill(41));
        const result = sampler.verify(input);
        using tokens = result.tokens;
        using counts = result.numAccepted;
        check(result, compact, stochastic);
      });
    }

    for (const { params, withNaNs } of [
      { params: [{ ...greedy, temperature: 0.75, topK: 20, topP: 0.6 }, { ...greedy, topK: 0 }], withNaNs: false },
      { params: [{ ...greedy, temperature: 1.25, topK: 0, topP: 0.6 }, { ...greedy, temperature: 0.5, topK: 20 }], withNaNs: false },
      { params: [{ ...greedy, temperature: 0.75, topK: 20, topP: 0.6 }, { ...greedy, temperature: 0.5, topK: 20 }], withNaNs: true },
    ]) {
      const maxK = params[0].topK === 0 ? 32 : 20;
      it(`matches full sampling with Row-sharded ${withNaNs ? "finite and NaN " : ""}logits and heterogeneous maxK=${maxK}`, { skip: !parallel }, (t) => {
        assert.ok(glm instanceof ParallelOps);
        const reference = new SamplingWorkspace(glm, B, V, 0);
        using input = ws.alloc([B, V], "BF16", undefined, TensorParallelism.Row);
        using full = ws.alloc([B, V], "BF16", undefined, TensorParallelism.Replicated);
        using probs = ws.alloc([B, sampler.capacity], "F32");
        using ids = ws.alloc([B, sampler.capacity], "I32");
        assert.ok(input instanceof ParallelTensor);
        assert.equal(input.parallelism, TensorParallelism.Row);
        assert.equal(input.shards.length, 2);
        for (const shard of input.shards) assert.deepEqual(shard.shape, [B, V / 2]);
        try {
          sampler.updateSampler(params);
          sampler.prepareDraft(B, D);
          reference.updateSampler(params, params.map(() => []));
          assert.equal(sampler.captureKey, `linear:${maxK}`);
          const seed = Buffer.alloc(4);
          seed.writeUInt32LE(123456);
          sampler.draftSampler.stepCounter.h2d(seed);
          reference.stepCounter.h2d(seed);
          const expectedProbs: Buffer[] = [];
          const expectedIds: number[][] = [];
          for (let depth = 0; depth < D; depth++) {
            const values = new Float32Array(B * V);
            for (let row = 0; row < B; row++) {
              for (let rank = 0; rank < V; rank++) {
                // Exact, unique BF16 values; the best candidates span both vocabulary shards.
                const token = (rank * 73 + row * 257 + depth * 257) % V;
                values[row * V + token] = withNaNs && rank % 11 === 3 ? NaN : (256 - rank) / 32;
              }
            }
            const bytes = f32ToBf16Bytes(values);
            input.h2d(bytes);
            full.h2d(bytes);
            reference.sampleInto(full, reference.outToken, probs, ids, sampler.capacity);
            const expectedTokens = readI32(reference.outToken);
            params.forEach((param, row) => {
              if (param.temperature === 0) assert.equal(expectedTokens[row], (row * 257 + depth * 257) % V);
            });
            const probabilityBuffer = Buffer.alloc(probs.bytes);
            probs.d2h(probabilityBuffer);
            glm.synchronize();
            expectedProbs.push(probabilityBuffer);
            expectedIds.push(readI32(ids));
            for (let row = 0; row < B; row++) {
              for (let k = 0; k < sampler.capacity; k++) {
                const index = row * sampler.capacity + k;
                const probability = probabilityBuffer.readFloatLE(index * 4);
                assert.ok(Number.isFinite(probability));
                if (probability > 0) assert.ok(Number.isFinite(values[row * V + expectedIds[depth][index]]));
              }
            }

            {
              using guard = noFullGather(t, maxK);
              using tokens = sampler.sampleDraft(input, depth);
              assert.ok(tokens instanceof ParallelTensor);
              assert.equal(tokens.parallelism, TensorParallelism.Replicated);
              for (const shard of tokens.shards) assert.deepEqual(readI32(shard), expectedTokens);
              assert.equal(guard.batchMock.mock.callCount(), 0);
              assert.equal(guard.multipleMock.mock.callCount(), 1);
            }
          }
          const proposal = sampler.finishDraft();
          for (let row = 0; row < B; row++) {
            for (let depth = 0; depth < D; depth++) {
              let sum = 0;
              const owners = new Set<number>();
              for (let k = 0; k < sampler.capacity; k++) {
                const actualOffset = (depth * sampler.capacity + k) * 4;
                const referenceIndex = row * sampler.capacity + k;
                const expected = expectedProbs[depth].readFloatLE(referenceIndex * 4);
                const actual = proposal.probabilities[row].readFloatLE(actualOffset);
                assert.ok(Math.abs(actual - expected) < 1e-6, `q row=${row} depth=${depth} k=${k}`);
                if (expected > 0) {
                  const token = proposal.tokenIds[row].readInt32LE(actualOffset);
                  assert.equal(token, expectedIds[depth][referenceIndex]);
                  owners.add(Math.floor(token / (V / 2)));
                } else {
                  assert.equal(actual, 0);
                }
                sum += actual;
              }
              assert.ok(Math.abs(sum - 1) < 1e-6);
              if (params[row].temperature > 0) assert.equal(owners.size, 2);
            }
          }
          assert.deepEqual(readI32(sampler.draftSampler.stepCounter), readI32(reference.stepCounter));
        } finally {
          glm.synchronize();
          reference.free();
        }
      });
    }

    it("replays Row-sharded topk20 draft graphs with seeded tokens and exported q/ids", { skip: !parallel }, (t) => {
      assert.ok(glm instanceof ParallelOps);
      using graphWs = new WorkspaceBase(glm);
      const inputs = Array.from({ length: D }, (_, depth) =>
        graphWs.alloc([B, V], "BF16", `graphDraftInput${depth}`, TensorParallelism.Row));
      for (const input of inputs) {
        assert.ok(input instanceof ParallelTensor);
        assert.equal(input.parallelism, TensorParallelism.Row);
        assert.equal(input.shards.length, 2);
        for (const shard of input.shards) assert.deepEqual(shard.shape, [B, V / 2]);
      }
      const upload = (iteration: number) => {
        inputs.forEach((input, depth) => {
          const values = new Float32Array(B * V);
          for (let row = 0; row < B; row++) {
            for (let rank = 0; rank < V; rank++) {
              const token = (rank * 73 + row * 257 + depth * 17 + iteration * 101) % V;
              values[row * V + token] = (256 - rank) / 32;
            }
          }
          input.h2d(f32ToBf16Bytes(values));
        });
      };
      const runDraft = () => {
        using tracking = graphWs.startTracking();
        for (let depth = 0; depth < D; depth++) {
          using tokens = sampler.sampleDraft(inputs[depth], depth);
        }
      };
      const lastTokens = () => {
        const tokens = sampler.draftSampler.outToken;
        assert.ok(tokens instanceof ParallelTensor);
        assert.equal(tokens.parallelism, TensorParallelism.Replicated);
        assert.equal(tokens.shards.length, 2);
        const perRank = tokens.shards.map(readI32);
        assert.deepEqual(perRank[0], perRank[1]);
        return perRank;
      };
      const params = [
        { ...greedy, temperature: 0.75, topK: 20, topP: 0.6 },
        { ...greedy, temperature: 0.5, topK: 20 },
      ];
      sampler.updateSampler(params);
      sampler.prepareDraft(B, D);
      assert.equal(sampler.captureKey, "linear:20");
      upload(0);
      using guard = noFullGather(t, 20);
      for (let warmup = 0; warmup < 3; warmup++) runDraft();
      glm.synchronize();
      glm.graphBeginCapture();
      runDraft();
      const graph = glm.graphEndCapture();
      let exec: number | undefined;
      let previous: ReturnType<LinearMtpSamplingWorkspace["finishDraft"]> | undefined;
      try {
        exec = glm.graphInstantiate(graph);
        for (let iteration = 0; iteration < 3; iteration++) {
          upload(iteration);
          sampler.updateSampler(params.map(param => ({ ...param, temperature: param.temperature + iteration * 0.25 })));
          sampler.prepareDraft(B, D);
          const seed = Buffer.alloc(4);
          seed.writeUInt32LE(123456 + iteration * 97);
          sampler.draftSampler.stepCounter.h2d(seed);
          runDraft();
          glm.synchronize();
          const expected = sampler.finishDraft();
          const expectedTokens = lastTokens();
          const expectedCounter = readI32(sampler.draftSampler.stepCounter);
          const saved = [...expected.probabilities, ...expected.tokenIds].map(buf => Buffer.from(buf));
          if (previous) {
            assert.notDeepEqual(expected.tokenIds, previous.tokenIds);
            assert.notDeepEqual(expected.probabilities, previous.probabilities);
          }
          // RNG and policy uploads stay outside capture; replay must consume the new buffers.
          sampler.draftSampler.stepCounter.h2d(seed);
          glm.graphLaunch(exec);
          glm.synchronize();
          const actual = sampler.finishDraft();
          assert.deepEqual(actual, expected);
          assert.deepEqual(lastTokens(), expectedTokens);
          assert.deepEqual(readI32(sampler.draftSampler.stepCounter), expectedCounter);
          assert.deepEqual([...expected.probabilities, ...expected.tokenIds], saved);
          previous = actual;
        }
        assert.equal(guard.batchMock.mock.callCount(), 0);
        assert.equal(guard.multipleMock.mock.callCount(), (3 + 1 + 3) * D);
      } finally {
        glm.synchronize();
        if (exec !== undefined) glm.graphExecDestroy(exec);
        glm.graphDestroy(graph);
      }
    });

    it("replays resident draft and verification graphs with fresh generations and no q host transfers", (t) => {
      using resident = new LinearMtpSamplingWorkspace(glm, B, D, target, true);
      guardResidentTransfers(t, resident);
      resident.updateSampler([greedy, greedy]);
      assert.equal(resident.captureKey, "linear:1:gpu");
      resident.prepareDraft(B, D);
      using draftInput = logits([7, 29]);
      using targetInput = logits([7, 7, 7, 7, 29, 29, 29, 29]);
      const runDraft = () => {
        for (let depth = 0; depth < D; depth++) {
          using tokens = resident.sampleDraft(draftInput, depth);
        }
      };
      for (let warmup = 0; warmup < 3; warmup++) runDraft();
      glm.synchronize();
      glm.graphBeginCapture();
      runDraft();
      const draftGraph = glm.graphEndCapture();
      let draftExec: number | undefined;
      let verifyExec: number | undefined;
      let captured: ReturnType<LinearMtpSamplingWorkspace["verify"]> | undefined;
      let previous: MtpDraftBatch | undefined;
      try {
        draftExec = glm.graphInstantiate(draftGraph);
        for (const peaks of [[7, 29], [43, 61], [11, 37]]) {
          uploadLogits(draftInput, peaks);
          resident.prepareDraft(B, D);
          if (previous) assert.throws(() => resident.prepareVerification(previous!), /stale/);
          glm.graphLaunch(draftExec);
          glm.synchronize();
          assert.deepEqual(readI32(resident.draftSampler.outToken), peaks);
          const batch: MtpDraftBatch = {
            targetTokens: peaks, treeTokens: peaks.map(peak => Array(D).fill(peak)),
            topks: [1, 1, 1], proposal: resident.finishDraft(),
          };
          assert.deepEqual(batch.proposal!.probabilities, []);
          assert.deepEqual(batch.proposal!.tokenIds, []);
          assert.deepEqual(batch.proposal!.device!.owner, {});
          assert.deepEqual(batch.proposal!.device!.rows, [0, 1]);
          if (previous) assert.equal(batch.proposal!.device!.generation, previous.proposal!.device!.generation + 1);
          const expanded = Array.from({ length: B * (D + 1) }, () => greedy);
          target.updateSampler(expanded, expanded.map(() => []));
          resident.prepareVerification(batch);
          uploadLogits(targetInput, peaks.flatMap(peak => Array(D + 1).fill(peak)));
          if (verifyExec === undefined) {
            for (let warmup = 0; warmup < 3; warmup++) {
              const result = resident.verify(targetInput);
              using tokens = result.tokens;
              using counts = result.numAccepted;
            }
            glm.synchronize();
            glm.graphBeginCapture();
            captured = resident.verify(targetInput);
            const graph = glm.graphEndCapture();
            try { verifyExec = glm.graphInstantiate(graph); }
            finally { glm.graphDestroy(graph); }
          }
          glm.graphLaunch(verifyExec);
          check(captured!, batch);
          previous = batch;
        }
      } finally {
        glm.synchronize();
        if (verifyExec !== undefined) glm.graphExecDestroy(verifyExec);
        if (draftExec !== undefined) glm.graphExecDestroy(draftExec);
        glm.graphDestroy(draftGraph);
        captured?.tokens[Symbol.dispose]();
        captured?.numAccepted[Symbol.dispose]();
      }
    });

    it("replays captured draft and verification kernels with updated buffers", { skip: parallel }, () => {
      sampler.updateSampler([greedy, greedy]);
      sampler.prepareDraft(B, D);
      using draftInput = logits([7, 29]);
      using targetInput = logits([7, 7, 7, 7, 29, 29, 29, 29]);
      const runDraft = () => {
        for (let depth = 0; depth < D; depth++) {
          using tokens = sampler.sampleDraft(draftInput, depth);
        }
      };
      runDraft();
      glm.synchronize();
      glm.graphBeginCapture();
      runDraft();
      const draftGraph = glm.graphEndCapture();
      let draftExec: number | undefined;
      let verifyExec: number | undefined;
      let captured: ReturnType<LinearMtpSamplingWorkspace["verify"]> | undefined;
      try {
        draftExec = glm.graphInstantiate(draftGraph);
        for (const peaks of [[7, 29], [43, 61], [11, 37]]) {
          uploadLogits(draftInput, peaks);
          glm.graphLaunch(draftExec);
          glm.synchronize();
          const batch: MtpDraftBatch = {
            targetTokens: peaks, treeTokens: peaks.map(peak => Array(D).fill(peak)),
            topks: [1, 1, 1], proposal: sampler.finishDraft(),
          };
          batch.proposal!.tokenIds.forEach((ids, row) => {
            for (let depth = 0; depth < D; depth++) assert.equal(ids.readInt32LE(depth * 256 * 4), peaks[row]);
          });
          prepare(batch, [greedy, greedy]);
          uploadLogits(targetInput, peaks.flatMap(peak => Array(D + 1).fill(peak)));
          if (verifyExec === undefined) {
            const warmup = sampler.verify(targetInput);
            warmup.tokens[Symbol.dispose]();
            warmup.numAccepted[Symbol.dispose]();
            glm.synchronize();
            glm.graphBeginCapture();
            captured = sampler.verify(targetInput);
            const graph = glm.graphEndCapture();
            try { verifyExec = glm.graphInstantiate(graph); }
            finally { glm.graphDestroy(graph); }
          }
          glm.graphLaunch(verifyExec);
          check(captured!, batch);
        }
      } finally {
        glm.synchronize();
        if (verifyExec !== undefined) glm.graphExecDestroy(verifyExec);
        if (draftExec !== undefined) glm.graphExecDestroy(draftExec);
        glm.graphDestroy(draftGraph);
        captured?.tokens[Symbol.dispose]();
        captured?.numAccepted[Symbol.dispose]();
      }
    });
  });
}
