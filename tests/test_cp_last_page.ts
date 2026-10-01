import assert from "node:assert/strict";
import { it } from "node:test";
import { GlmOps } from "../src/glm_ops";
import { ParallelOps, ParallelTensor } from "../src/parallel_ops";
import { WorkspaceBase } from "../src/workspace";
import { TensorParallelism } from "../src/device_ops";
import { getNativeAddon } from "../src/native-addon";

it("CP last-page metadata preserves empty rank-local tails and page-gather bounds", () => {
  const devices = [new GlmOps(0), new GlmOps(1)];
  const ops = new ParallelOps(devices);
  const ws = new WorkspaceBase(ops);
  const localWorkspaces = devices.map(device => new WorkspaceBase(device));
  try {
    const physicalPageSize = 64;
    const logicalPageSize = physicalPageSize * devices.length;
    const lengths = [0, 1, 2, 127, 128, 129, 130, 255, 256, 257];
    using last = ws.allocPinned([lengths.length], "I32", undefined, TensorParallelism.Replicated) as ParallelTensor;
    ops.sparseMlaDecodePlan(last, lengths.length, lengths, logicalPageSize, true);
    for (let rank = 0; rank < devices.length; rank++) {
      const actual = last.shard(rank).readPinnedBuffer();
      lengths.forEach((length, sequence) => {
        const start = length ? Math.floor((length - 1) / logicalPageSize) * logicalPageSize : 0;
        let expected = 0;
        for (let token = start; token < length; token++) {
          if (token % devices.length === rank) expected++;
        }
        assert.equal(actual.readInt32LE(sequence * 4), expected, `rank=${rank} length=${length}`);
      });
    }

    // At length 129 both ranks reference two logical pages, but rank 1's
    // second physical page is empty. Catch the historical extra 64-token copy
    // with an explicit guard rather than allowing it to damage another tensor.
    const tokenBytes = 128;
    const capacity = Math.ceil(129 / logicalPageSize) * physicalPageSize;
    // Use the tighter packed-gather capacity to make an erroneous tail visible.
    const packedCapacity = 65;
    for (let rank = 0; rank < devices.length; rank++) {
      const local = localWorkspaces[rank];
      using source = local.alloc([2, physicalPageSize, tokenBytes], "U8");
      source.fill(7, source.numElements);
      using indices = local.alloc([2], "I32");
      indices.h2d(Buffer.from(new Int32Array([0, 1]).buffer));
      using indptr = local.alloc([2], "I32");
      indptr.h2d(Buffer.from(new Int32Array([0, 2]).buffer));
      using tail = local.alloc([1], "I32");
      const tailLength = last.shard(rank).readPinnedBuffer().readInt32LE(lengths.indexOf(129) * 4);
      tail.h2d(Buffer.from(new Int32Array([tailLength]).buffer));
      using guarded = local.alloc([capacity + 1, tokenBytes], "U8");
      guarded.fill(0xa5, guarded.numElements);
      getNativeAddon().gatherPages(devices[rank].ctx, guarded.data, source.data, indices.data,
        indptr.data, tail.data, 2, 1, physicalPageSize, tokenBytes);
      devices[rank].synchronize();
      const data = Buffer.alloc(guarded.bytes);
      guarded.d2h(data);
      devices[rank].synchronize();
      const copiedBytes = (rank === 0 ? 65 : 64) * tokenBytes;
      assert.ok(data.subarray(0, copiedBytes).every(value => value === 7));
      assert.ok(data.subarray(copiedBytes).every(value => value === 0xa5), `rank=${rank}: gather wrote beyond valid tokens`);
      assert.ok(data.subarray(packedCapacity * tokenBytes).every(value => value === 0xa5), `rank=${rank}: gather overwrote guard`);
    }
  } finally {
    ops.synchronize();
    localWorkspaces.forEach(workspace => workspace.free());
    ws.free();
    ops.free();
  }
});
