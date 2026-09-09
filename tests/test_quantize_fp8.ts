import { it } from 'node:test';
import assert from 'node:assert/strict';
import { GlmOps, f32ToBf16Bytes } from '../src/glm_ops';
import { ParallelOps, ParallelTensor } from '../src/parallel_ops';
import { TensorParallelism } from '../src/device_ops';
import { WorkspaceBase } from '../src/workspace';
import type { Tensor } from '../src/tensor';
import type { ExecutionState } from '../src/execution-workspace';

function bytes(tensor: Tensor): Buffer {
  const result = Buffer.alloc(tensor.bytes);
  tensor.d2h(result);
  return result;
}

for (const p2p of [true, false]) {
  it(`gathers FP8 Q, scales, and BF16 RoPE with ${p2p ? 'P2P' : 'NCCL'}`, () => {
    using gpu0 = new GlmOps(0);
    using gpu1 = new GlmOps(1);
    using po = new ParallelOps([gpu0, gpu1]);
    if (p2p) assert.ok(po.p2pEnabled, 'P2P must be available for this test');
    po.p2pEnabled = p2p;
    using ws = new WorkspaceBase(po);
    using referenceWs = new WorkspaceBase(gpu0);
    const shape = [3, 16, 512];
    const hostQ = f32ToBf16Bytes(Float32Array.from({ length: 3 * 16 * 512 }, (_, i) =>
      Math.sin(i * 0.13) * (1 + Math.floor(i / 128) % 17)));
    using q = ws.alloc(shape, 'BF16', undefined, TensorParallelism.Row) as ParallelTensor;
    using rope = ws.alloc([3, 16, 64], 'BF16', undefined, TensorParallelism.Row) as ParallelTensor;
    const hostRope = f32ToBf16Bytes(Float32Array.from({ length: 3 * 16 * 64 }, (_, i) => i % 113));
    q.h2d(hostQ);
    rope.h2d(hostRope);
    using referenceQ = referenceWs.alloc(shape, 'BF16');
    referenceQ.h2d(hostQ);
    po.synchronize();
    const reference = gpu0.quantizeFp8(referenceQ, 128);

    // Exercise alternate-stream lifetime and mixed-dtype gather ordering.
    using stream = po.withStream(() => {
      const quantized = po.quantizeFp8(q, 128);
      using values = quantized.values;
      using scales = quantized.scales;
      assert.deepEqual(scales.shape, [3, 16, 4]);
      return po.allGatherMultiple([values, scales, rope], ws);
    });
    stream.streamWaitEvent();
    po.synchronize();
    const [values, scales, gatheredRope] = stream.result;
    for (let rank = 0; rank < 2; rank++) {
      assert.deepEqual(bytes(values.shards[rank]), bytes(reference.values));
      assert.deepEqual(bytes(scales.shards[rank]), bytes(reference.scales));
      assert.deepEqual(bytes(gatheredRope.shards[rank]), hostRope);
    }
  });
}

for (const tokens of [1, 17]) {
  it(`threads native FP8 scales through sparse prefill with ${tokens} tokens`, () => {
    using gpu = new GlmOps(0);
    using ws = new WorkspaceBase(gpu);
    const heads = 8, topk = 2048;
    const q = ws.alloc([tokens, heads, 512], 'BF16');
    const rope = ws.alloc([tokens, heads, 64], 'BF16');
    q.h2d(f32ToBf16Bytes(Float32Array.from({ length: q.numElements }, (_, i) => Math.sin(i))));
    rope.h2d(f32ToBf16Bytes(Float32Array.from({ length: rope.numElements }, (_, i) => Math.cos(i))));
    // Valid FP8 cache with one nonzero token: E4M3 1.0 and FP32 scales 1.0.
    const kv = ws.alloc([1, 64, 656], 'U8');
    const hostKv = Buffer.alloc(kv.bytes);
    hostKv.fill(0x38, 0, 512);
    for (let i = 0; i < 4; i++) hostKv.writeFloatLE(1, 512 + i * 4);
    kv.h2d(hostKv);
    const indices = ws.alloc([tokens, topk], 'I32');
    indices.h2d(Buffer.alloc(indices.bytes));
    const length = ws.alloc([tokens], 'I32');
    const hostLength = Buffer.alloc(length.bytes);
    for (let i = 0; i < tokens; i++) hostLength.writeInt32LE(1, i * 4);
    length.h2d(hostLength);
    gpu.synchronize();
    const quantized = gpu.quantizeFp8(q, 128);
    const state = { totalTokens: tokens } as ExecutionState;
    const reference = gpu.sparseMlaPrefill(state, q, rope, kv, indices, topk, 0.0791, length, length, length, length);
    const result = gpu.sparseMlaPrefill(state, quantized.values, rope, kv, indices, topk, 0.0791, length, length, length, length, quantized.scales);
    gpu.synchronize();
    assert.deepEqual(bytes(result.o), bytes(reference.o));
    assert.deepEqual(bytes(result.lse), bytes(reference.lse));
  });
}
