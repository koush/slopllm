import assert from "node:assert/strict";
import { after, before, describe, it, type TestContext } from "node:test";
import type { MtpDraftBatch, SamplingParams } from "../src/chat_model";
import { TensorParallelism } from "../src/device_ops";
import { MemcpyKind } from "../src/enums";
import { GlmOps, f32ToBf16Bytes } from "../src/glm_ops";
import { ParallelOps, ParallelTensor } from "../src/parallel_ops";
import { SamplingWorkspace } from "../src/sampling";
import { CaptureManager } from "../src/capture-manager";
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
  tensor.workspace.ops.synchronize();
  const buf = Buffer.alloc(tensor.bytes);
  tensor.d2h(buf);
  tensor.workspace.ops.synchronize();
  return Array.from({ length: tensor.numElements }, (_, i) => buf.readInt32LE(i * 4));
}

for (const parallel of [false, true]) {
  describe(`SamplingWorkspace linear MTP (${parallel ? "two GPUs" : "single GPU"})`, {
    skip: parallel && process.env.TEST_MULTIGPU !== "1",
    concurrency: false,
  }, () => {
    const devices: GlmOps[] = [];
    let ops: GlmOps | ParallelOps;
    let ws: WorkspaceBase;
    let target: SamplingWorkspace;
    let sampler: SamplingWorkspace;

    before(() => {
      const first = parseInt(process.env.GLM_GPU ?? "0", 10);
      devices.push(new GlmOps(first));
      if (parallel) devices.push(new GlmOps(parseInt(process.env.GLM_GPU_SECOND ?? "1", 10)));
      ops = parallel ? new ParallelOps(devices) : devices[0];
      ws = new WorkspaceBase(ops);
      target = new SamplingWorkspace(ops, B * (D + 1), V, 0, { maxBatchSize: B, depth: D });
      sampler = target;
    });

    after(() => {
      try {
        ops?.synchronize();
      } finally {
        target?.free();
        ws?.free();
        if (ops instanceof ParallelOps) ops.free();
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

    it("selectTarget delegates to ordinary target sampling", t => {
      target.updateSampler(Array.from({ length: B }, () => greedy));
      using input = logits([7, 300]);
      const sample = t.mock.method(target, "sample");
      using selected = target.selectTarget(input);
      assert.equal(sample.mock.callCount(), 1);
      assert.equal(sample.mock.calls[0].arguments[0], input);
      assert.equal(sample.mock.calls[0].result, selected);
      assert.deepEqual(readI32(selected).slice(0, B), [7, 300]);
    });

    it("ordinary sampling retains penalty history and caller-owned outputs without MTP allocations", () => {
      using sampling = new SamplingWorkspace(ops, B, V, 8);
      using inputs = new WorkspaceBase(ops);
      using manager = new CaptureManager(ops);
      const input = inputs.alloc([B, V], "BF16", "logits");
      assert.equal(sampling.mtpEnabled, false);
      assert.throws(() => sampling.updateMtpSampler([greedy]), /not enabled/);
      assert.throws(() => sampling.prepareDraft(B, D), /not enabled/);
      assert.throws(() => sampling.sampleDraft(input, 0), /not enabled/);
      assert.throws(() => sampling.finishDraft(), /not enabled/);
      assert.throws(() => sampling.prepareVerification({ targetTokens: [], treeTokens: [], topks: [] }), /not enabled/);
      assert.throws(() => sampling.verify(input), /not enabled/);
      assert.equal(sampling.tensors.has("qDraftProbs"), false);
      for (let iteration = 0; iteration < 7; iteration++) {
        const peaks = [7 + iteration * 3, 300 + iteration * 3];
        uploadLogits(input, peaks);
        sampling.updateSampler(peaks.map(() => ({ ...greedy, presencePenalty: 2 })), peaks.map(peak => [peak]));
        using selected = manager.run({}, () => sampling.sample(input), ["sample", sampling.captureKey]) as Tensor;
        assert.deepEqual(readI32(selected), peaks.map(peak => peak + 1));
        assert.deepEqual(readI32(sampling.penaltyCount), [2, 2]);
      }
      assert.equal(sampling.tracked.size, 0);
      assert.ok([...manager.captured.values()].every(captured => captured.graphExec !== null && captured.capturedWorkspaces.has(sampling)));
      {
        using first = sampling.sample(input);
        const saved = readI32(first);
        using second = sampling.sample(input);
        assert.equal(first.same(second), false);
        assert.deepEqual(readI32(first), saved);
      }
      assert.equal(sampling.tracked.size, 0);
    });

    it("ordinary sampling advances an ordered penalty ring across replay and restart", () => {
      const window = 4;
      using sampling = new SamplingWorkspace(ops, B, V, window);
      using inputs = new WorkspaceBase(ops);
      using manager = new CaptureManager(ops);
      const input = inputs.alloc([B, V], "BF16", "logits");
      const params = Array.from({ length: B }, () => ({ ...greedy, presencePenalty: 4, repetitionPenaltyWindow: window }));
      const histories = Array.from({ length: B }, (_, row) => [9, 1, 2, 1, 3].map(token => token + row * 200));
      let ring = histories.map(history => history.slice(-window));
      let count = window;
      sampling.updateSampler(params, histories);
      assert.deepEqual(readI32(sampling.penaltyTokens), ring.flat());
      assert.deepEqual(readI32(sampling.penaltyCount), [count, count]);
      for (let iteration = 0; iteration < 12; iteration++) {
        if (iteration === 6) {
          // A restarted batch initializes from history; ongoing steps do not.
          sampling.updateSampler(params, histories);
          ring = histories.map(history => history.slice(-window));
          count = window;
        }
        const values = new Float32Array(B * V).fill(-16);
        const expected = histories.map((history, row) => {
          const offset = row * 200;
          values[row * V + offset + 1] = 10;
          values[row * V + offset + 2] = 11;
          // First append D and E to [A, B, A, C], then distinguish whether
          // A or B expired. Later iterations exercise repeated wraparound.
          if (iteration < 2) values[row * V + offset + iteration + 4] = 20;
          const seen = new Set(history.slice(-window));
          let best = 0;
          let score = -Infinity;
          for (let token = 0; token < V; token++) {
            const penalized = values[row * V + token] - (seen.has(token) ? 4 : 0);
            if (penalized > score) { best = token; score = penalized; }
          }
          return best;
        });
        input.h2d(f32ToBf16Bytes(values));
        using selected = manager.run({}, () => sampling.sample(input), ["ordered-penalty-ring", sampling.captureKey]) as Tensor;
        assert.deepEqual(readI32(selected), expected);
        expected.forEach((token, row) => {
          histories[row].push(token);
          ring[row][count % window] = token;
        });
        count++;
        assert.deepEqual(readI32(sampling.penaltyTokens), ring.flat());
        assert.deepEqual(readI32(sampling.penaltyCount), [count, count]);
      }
      assert.ok([...manager.captured.values()].some(entry => entry.graphExec !== null));
    });

    function noFullGather(t: TestContext, maxK: number) {
      assert.ok(ops instanceof ParallelOps);
      const allGather = ParallelTensor.prototype.allGather;
      const allGatherMultiple = ops.allGatherMultiple;
      const batchMock = t.mock.method(ops, "sampleBatch", () => {
        throw new Error("sampleDraft must not call ParallelOps.sampleBatch");
      });
      const gatherMock = t.mock.method(ParallelTensor.prototype, "allGather", function (this: ParallelTensor, ...args: Parameters<ParallelTensor["allGather"]>) {
        assert.ok(this.numElements < B * V, "sampleDraft must not gather full logits");
        return allGather.apply(this, args);
      });
      const multipleMock = t.mock.method(ops, "allGatherMultiple", function (this: ParallelOps, ...args: Parameters<ParallelOps["allGatherMultiple"]>) {
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
      sampler.updateMtpSampler(params);
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
      ops.synchronize();
      return { targetTokens: peaks, treeTokens, topks: [1, 1, 1], proposal: sampler.finishDraft() };
    }

    function guardResidentTransfers(t: TestContext, workspace: SamplingWorkspace) {
      const resident = workspace;
      const copies = [] as { source: Tensor; sourceOffset: number; destinationOffset: number }[];
      for (const name of ["qHostProbs", "qHostIds", "qInputProbsH", "qInputIdsH"]) {
        assert.equal(workspace.tensors.has(name), false);
      }
      for (const name of ["qInputProbs", "qInputIds"]) {
        const tensor = workspace.tensors.get(name)!;
        t.mock.method(tensor, "memcpy", () => { throw new Error(`Resident proposals uploaded ${name}`); });
        const memcpy2d = tensor.memcpy2d;
        t.mock.method(tensor, "memcpy2d", function (this: Tensor, ...args: Parameters<Tensor["memcpy2d"]>) {
          const [destinationOffset, dpitch, source, sourceOffset, spitch, width, height, kind] = args;
          assert.equal(kind, MemcpyKind.DeviceToDevice);
          assert.equal(source, workspace.tensors.get(name === "qInputProbs" ? "qDraftProbs" : "qDraftIds"));
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
        using residentWs = new SamplingWorkspace(ops, B * (D + 1), V, 0, { maxBatchSize: B, depth: D, retainProposalsOnGpu: true });
        const resident = residentWs;
        const copies = guardResidentTransfers(t, residentWs);
        const params = stochastic
          ? [{ ...greedy, temperature: 1, topK: 3 }, { ...greedy, temperature: 0.5, topK: 2 }]
          : [greedy, greedy];
        const seed = Buffer.alloc(4);
        seed.writeUInt32LE(123456);
        for (const mode of [sampler, resident]) {
          mode.updateMtpSampler(params);
          mode.prepareDraft(B, D);
          mode.draftStepCounter.h2d(seed);
        }
        ops.synchronize();
        assert.equal(sampler.mtpCaptureKey, `linear:${stochastic ? 3 : 1}`);
        assert.equal(resident.mtpCaptureKey, `${sampler.mtpCaptureKey}:gpu`);
        const treeTokens: number[][] = [[], []];
        for (let depth = 0; depth < D; depth++) {
          using input = logits([7 + depth * 4, 29 + depth * 4]);
          using hostTokens = sampler.sampleDraft(input, depth);
          using gpuTokens = resident.sampleDraft(input, depth);
          const expected = readI32(hostTokens);
          assert.deepEqual(readI32(gpuTokens), expected);
          expected.forEach((token, row) => treeTokens[row].push(token));
        }
        ops.synchronize();
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
        assert.deepEqual(readI32(resident.draftStepCounter), readI32(sampler.draftStepCounter));
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
          resident.updateMtpSampler(order.map(row => params[row]));
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
        ops.synchronize();
      });
    }

    function prepare(batch: MtpDraftBatch, params: SamplingParams[]): void {
      // MTP owns an explicit sequence-major verification bank, independent of target rows.
      sampler.updateMtpSampler(params);
      sampler.prepareVerification(batch);
    }

    function check(result: ReturnType<SamplingWorkspace["verify"]>, batch: MtpDraftBatch, stochastic = false): void {
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
      using unpreparedWs = new SamplingWorkspace(ops, B * (D + 1), V, 0, { maxBatchSize: B, depth: D });
      const unprepared = unpreparedWs;
      using unpreparedInput = logits([7, 19]);
      assert.throws(() => unprepared.sampleDraft(unpreparedInput, 0), /unprepared batch/);
      assert.throws(() => unprepared.finishDraft(), /not prepared/);
      for (const [batch, depth] of [[0, D], [1.5, D], [B, 0], [B, 1.5], [B + 1, D]]) {
        assert.throws(() => new SamplingWorkspace(ops, B * (D + 1), V, 0, { maxBatchSize: batch, depth }), /valid batch\/depth/);
      }
      for (const patch of [
        { temperature: NaN }, { topP: Infinity }, { topK: 1.5 },
        { temperature: 1, topK: 257 }, { repetitionPenalty: 1.1 }, { presencePenalty: 0.1 },
      ]) assert.throws(() => sampler.updateMtpSampler([{ ...greedy, ...patch }]), /Linear MTP requires/);
      assert.throws(() => sampler.updateMtpSampler([greedy, greedy, greedy]), /maxBatchSize/);
      sampler.updateMtpSampler([greedy, greedy]);
      assert.equal(sampler.mtpCaptureKey, "linear:1");
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
      assert.throws(() => sampler.verify(input), /row capacities/);
      prepare(batch, [greedy, greedy]);
      assert.throws(() => sampler.verify(input), /row capacities/);
    });

    it("transposes depth-major q into owned B=2, D=3 host proposal rows", () => {
      sampler.updateMtpSampler([greedy, greedy]);
      sampler.prepareDraft(B, D);
      for (let depth = 0; depth < D; depth++) {
        using input = logits([10 + depth, 30 + depth]);
        using tokens = sampler.sampleDraft(input, depth);
        assert.deepEqual(readI32(tokens), [10 + depth, 30 + depth]);
      }
      ops.synchronize();
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
        assert.equal(sampler.mtpCaptureKey, stochastic ? "linear:3" : "linear:1");
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
        assert.ok(ops instanceof ParallelOps);
        const reference = new SamplingWorkspace(ops, B, V, 0);
        using input = ws.alloc([B, V], "BF16", undefined, TensorParallelism.Row);
        using full = ws.alloc([B, V], "BF16", undefined, TensorParallelism.Replicated);
        using probs = ws.alloc([B, sampler.capacity], "F32");
        using ids = ws.alloc([B, sampler.capacity], "I32");
        assert.ok(input instanceof ParallelTensor);
        assert.equal(input.parallelism, TensorParallelism.Row);
        assert.equal(input.shards.length, 2);
        for (const shard of input.shards) assert.deepEqual(shard.shape, [B, V / 2]);
        try {
          sampler.updateMtpSampler(params);
          sampler.prepareDraft(B, D);
          reference.updateSampler(params, params.map(() => []));
          assert.equal(sampler.mtpCaptureKey, `linear:${maxK}`);
          const seed = Buffer.alloc(4);
          seed.writeUInt32LE(123456);
          sampler.draftStepCounter.h2d(seed);
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
            using output = reference.sample(full, probs, ids, sampler.capacity);
            const expectedTokens = readI32(output);
            params.forEach((param, row) => {
              if (param.temperature === 0) assert.equal(expectedTokens[row], (row * 257 + depth * 257) % V);
            });
            const probabilityBuffer = Buffer.alloc(probs.bytes);
            probs.d2h(probabilityBuffer);
            ops.synchronize();
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
          assert.deepEqual(readI32(sampler.draftStepCounter), readI32(reference.stepCounter));
        } finally {
          ops.synchronize();
          reference.free();
        }
      });
    }

    it("replays Row-sharded topk20 draft graphs with seeded tokens and exported q/ids", { skip: !parallel }, (t) => {
      assert.ok(ops instanceof ParallelOps);
      using graphWs = new WorkspaceBase(ops);
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
          lastOutput.memcpy(tokens, B * 4, MemcpyKind.DeviceToDevice);
        }
      };
      const lastOutput = graphWs.alloc([B], "I32", "lastOutput");
      const lastTokens = () => {
        const tokens = lastOutput;
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
      sampler.updateMtpSampler(params);
      sampler.prepareDraft(B, D);
      assert.equal(sampler.mtpCaptureKey, "linear:20");
      upload(0);
      using guard = noFullGather(t, 20);
      for (let warmup = 0; warmup < 3; warmup++) runDraft();
      ops.synchronize();
      ops.graphBeginCapture();
      runDraft();
      const graph = ops.graphEndCapture();
      let exec: number | undefined;
      let previous: ReturnType<SamplingWorkspace["finishDraft"]> | undefined;
      try {
        exec = ops.graphInstantiate(graph);
        for (let iteration = 0; iteration < 3; iteration++) {
          upload(iteration);
          sampler.updateMtpSampler(params.map(param => ({ ...param, temperature: param.temperature + iteration * 0.25 })));
          sampler.prepareDraft(B, D);
          const seed = Buffer.alloc(4);
          seed.writeUInt32LE(123456 + iteration * 97);
          sampler.draftStepCounter.h2d(seed);
          runDraft();
          ops.synchronize();
          const expected = sampler.finishDraft();
          const expectedTokens = lastTokens();
          const expectedCounter = readI32(sampler.draftStepCounter);
          const saved = [...expected.probabilities, ...expected.tokenIds].map(buf => Buffer.from(buf));
          if (previous) {
            assert.notDeepEqual(expected.tokenIds, previous.tokenIds);
            assert.notDeepEqual(expected.probabilities, previous.probabilities);
          }
          // RNG and policy uploads stay outside capture; replay must consume the new buffers.
          sampler.draftStepCounter.h2d(seed);
          ops.graphLaunch(exec);
          ops.synchronize();
          const actual = sampler.finishDraft();
          assert.deepEqual(actual, expected);
          assert.deepEqual(lastTokens(), expectedTokens);
          assert.deepEqual(readI32(sampler.draftStepCounter), expectedCounter);
          assert.deepEqual([...expected.probabilities, ...expected.tokenIds], saved);
          previous = actual;
        }
        assert.equal(guard.batchMock.mock.callCount(), 0);
        assert.equal(guard.multipleMock.mock.callCount(), (3 + 1 + 3) * D);
      } finally {
        ops.synchronize();
        if (exec !== undefined) ops.graphExecDestroy(exec);
        ops.graphDestroy(graph);
      }
    });

    it("replays resident draft and verification graphs with fresh generations and no q host transfers", (t) => {
      using residentWs = new SamplingWorkspace(ops, B * (D + 1), V, 0, { maxBatchSize: B, depth: D, retainProposalsOnGpu: true });
      const resident = residentWs;
      guardResidentTransfers(t, residentWs);
      resident.updateMtpSampler([greedy, greedy]);
      assert.equal(resident.mtpCaptureKey, "linear:1:gpu");
      resident.prepareDraft(B, D);
      using draftInput = logits([7, 29]);
      using targetInput = logits([7, 7, 7, 7, 29, 29, 29, 29]);
      using lastOutput = ws.alloc([B], "I32");
      const runDraft = () => {
        for (let depth = 0; depth < D; depth++) {
          using tokens = resident.sampleDraft(draftInput, depth);
          lastOutput.memcpy(tokens, B * 4, MemcpyKind.DeviceToDevice);
        }
      };
      for (let warmup = 0; warmup < 3; warmup++) runDraft();
      ops.synchronize();
      ops.graphBeginCapture();
      runDraft();
      const draftGraph = ops.graphEndCapture();
      let draftExec: number | undefined;
      let verifyExec: number | undefined;
      let captured: ReturnType<SamplingWorkspace["verify"]> | undefined;
      let previous: MtpDraftBatch | undefined;
      try {
        draftExec = ops.graphInstantiate(draftGraph);
        for (const peaks of [[7, 29], [43, 61], [11, 37]]) {
          uploadLogits(draftInput, peaks);
          resident.prepareDraft(B, D);
          if (previous) assert.throws(() => resident.prepareVerification(previous!), /stale/);
          ops.graphLaunch(draftExec);
          ops.synchronize();
          assert.deepEqual(readI32(lastOutput), peaks);
          const batch: MtpDraftBatch = {
            targetTokens: peaks, treeTokens: peaks.map(peak => Array(D).fill(peak)),
            topks: [1, 1, 1], proposal: resident.finishDraft(),
          };
          assert.deepEqual(batch.proposal!.probabilities, []);
          assert.deepEqual(batch.proposal!.tokenIds, []);
          assert.deepEqual(batch.proposal!.device!.owner, {});
          assert.deepEqual(batch.proposal!.device!.rows, [0, 1]);
          if (previous) assert.equal(batch.proposal!.device!.generation, previous.proposal!.device!.generation + 1);
          resident.prepareVerification(batch);
          uploadLogits(targetInput, peaks.flatMap(peak => Array(D + 1).fill(peak)));
          if (verifyExec === undefined) {
            for (let warmup = 0; warmup < 3; warmup++) {
              const result = resident.verify(targetInput);
              using tokens = result.tokens;
              using counts = result.numAccepted;
            }
            ops.synchronize();
            ops.graphBeginCapture();
            captured = resident.verify(targetInput);
            const graph = ops.graphEndCapture();
            try { verifyExec = ops.graphInstantiate(graph); }
            finally { ops.graphDestroy(graph); }
          }
          ops.graphLaunch(verifyExec);
          check(captured!, batch);
          previous = batch;
        }
      } finally {
        ops.synchronize();
        if (verifyExec !== undefined) ops.graphExecDestroy(verifyExec);
        if (draftExec !== undefined) ops.graphExecDestroy(draftExec);
        ops.graphDestroy(draftGraph);
        captured?.tokens[Symbol.dispose]();
        captured?.numAccepted[Symbol.dispose]();
      }
    });

    for (const retainProposalsOnGpu of [false, true]) {
      it(`tracks one sampling workspace across CaptureManager target/draft/verification replay (${retainProposalsOnGpu ? "GPU" : "host"} q)`, () => {
        using sampling = new SamplingWorkspace(ops, B * (D + 1), V, 8, { maxBatchSize: B, depth: D, retainProposalsOnGpu });
        using inputs = new WorkspaceBase(ops);
        using manager = new CaptureManager(ops);
        for (const rows of [B, 1, B]) {
          const input = inputs.ensureAlloc([rows, V], "BF16", `draft${rows}`);
          const verification = inputs.ensureAlloc([rows * (D + 1), V], "BF16", `verify${rows}`);
          for (let iteration = 0; iteration < 7; iteration++) {
            const peaks = Array.from({ length: rows }, (_, b) => 7 + b * 270 + iteration * 3);
            uploadLogits(input, peaks);
            uploadLogits(verification, peaks.flatMap(peak => Array(D + 1).fill(peak)));
            const params = peaks.map((_, b) => ({ ...greedy, temperature: 0.5 + iteration * 0.25 + b, topK: 3 }));
            // Back-to-back uploads must not rewrite in-flight pinned staging.
            sampling.updateSampler(params);
            sampling.updateMtpSampler(params.map(p => ({ ...p, topK: 2 })));
            assert.equal(sampling.captureKey, 3);
            assert.equal(sampling.mtpCaptureKey, `linear:2${retainProposalsOnGpu ? ":gpu" : ""}`);
            sampling.updateMtpSampler(params);
            assert.equal(sampling.captureKey, 3);
            assert.equal(sampling.mtpCaptureKey, `linear:3${retainProposalsOnGpu ? ":gpu" : ""}`);
            sampling.updateSampler(peaks.map(() => greedy), peaks.map(() => []));
            assert.equal(sampling.captureKey, 1);
            assert.equal(sampling.mtpCaptureKey, `linear:3${retainProposalsOnGpu ? ":gpu" : ""}`);
            sampling.prepareDraft(rows, D);
            const counters = [sampling.stepCounter, sampling.draftStepCounter, sampling.tensors.get("rejectionStepCounter")!];
            const before = counters.map(counter => readI32(counter)[0]);
            {
              using selected = manager.run({}, () => sampling.selectTarget(input), ["target", rows, sampling.captureKey]) as Tensor;
              assert.deepEqual(readI32(selected), peaks);
            }
            assert.deepEqual(counters.map(counter => readI32(counter)[0]), [(before[0] + rows) | 0, before[1], before[2]]);
            const treeTokens = peaks.map(() => [] as number[]);
            for (let depth = 0; depth < D; depth++) {
              using selected = manager.run({}, () => sampling.sampleDraft(input, depth), ["draft", rows, depth, sampling.mtpCaptureKey]) as Tensor;
              readI32(selected).forEach((token, b) => treeTokens[b].push(token));
            }
            assert.deepEqual(counters.map(counter => readI32(counter)[0]), [(before[0] + rows) | 0, (before[1] + rows * D) | 0, before[2]]);
            const batch: MtpDraftBatch = { targetTokens: peaks, treeTokens, topks: Array(D).fill(1), proposal: sampling.finishDraft() };
            sampling.prepareVerification(batch);
            {
              const result = manager.run({}, () => sampling.verify(verification), ["verify", rows, sampling.mtpCaptureKey]) as ReturnType<SamplingWorkspace["verify"]>;
              using tokens = result.tokens;
              using counts = result.numAccepted;
              check(result, batch, true);
              assert.equal(tokens.workspace, sampling);
              assert.equal(counts.workspace, sampling);
            }
            assert.deepEqual(counters.map(counter => readI32(counter)[0]), [
              (before[0] + rows * (D + 2)) | 0, (before[1] + rows * D) | 0, (before[2] + rows * (2 * D + 2)) | 0,
            ]);
            assert.equal(sampling.batchSize, rows, "verification must not switch ordinary target rows");
            assert.deepEqual(readI32(sampling.penaltyCount).slice(0, rows), peaks.map(() => 1), "verification must not modify target penalty history");
            assert.equal(sampling.tracked.size, 0);
            if (ops instanceof ParallelOps) {
              for (const shardWs of ops.getShardWorkspaces(sampling)) assert.equal(shardWs.tracked.size, 0);
            }
          }
        }
        for (const captured of manager.captured.values()) {
          assert.notEqual(captured.graphExec, null);
          assert.ok(captured.capturedWorkspaces.has(sampling), "sampler allocations must participate in capture/replay lifetime checks");
        }
        for (const name of ["outToken", "topkValues", "topkIndices", "sampleWorkspace", "pProbs", "pIds", "tokens", "acceptedCounts", "temperaturesHost", "draftTokensH"]) {
          assert.equal(sampling.tensors.has(name), false, `${name} must not be persistent`);
        }
      });
    }

    it("replays captured draft and verification kernels with updated buffers", { skip: parallel }, () => {
      sampler.updateMtpSampler([greedy, greedy]);
      sampler.prepareDraft(B, D);
      using draftInput = logits([7, 29]);
      using targetInput = logits([7, 7, 7, 7, 29, 29, 29, 29]);
      const runDraft = () => {
        for (let depth = 0; depth < D; depth++) {
          using tokens = sampler.sampleDraft(draftInput, depth);
        }
      };
      runDraft();
      ops.synchronize();
      ops.graphBeginCapture();
      runDraft();
      const draftGraph = ops.graphEndCapture();
      let draftExec: number | undefined;
      let verifyExec: number | undefined;
      let captured: ReturnType<SamplingWorkspace["verify"]> | undefined;
      try {
        draftExec = ops.graphInstantiate(draftGraph);
        for (const peaks of [[7, 29], [43, 61], [11, 37]]) {
          uploadLogits(draftInput, peaks);
          ops.graphLaunch(draftExec);
          ops.synchronize();
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
            ops.synchronize();
            ops.graphBeginCapture();
            captured = sampler.verify(targetInput);
            const graph = ops.graphEndCapture();
            try { verifyExec = ops.graphInstantiate(graph); }
            finally { ops.graphDestroy(graph); }
          }
          ops.graphLaunch(verifyExec);
          check(captured!, batch);
        }
      } finally {
        ops.synchronize();
        if (verifyExec !== undefined) ops.graphExecDestroy(verifyExec);
        if (draftExec !== undefined) ops.graphExecDestroy(draftExec);
        ops.graphDestroy(draftGraph);
        captured?.tokens[Symbol.dispose]();
        captured?.numAccepted[Symbol.dispose]();
      }
    });
  });
}
