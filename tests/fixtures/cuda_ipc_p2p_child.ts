import { TensorParallelism } from "../../src/device_ops";
import { GlmOps } from "../../src/glm_ops";
import { ParallelOps, ParallelTensor } from "../../src/parallel_ops";
import { WorkspaceBase } from "../../src/workspace";

const deviceIds = process.argv.slice(2).map(Number);
const arenaGb = 1 / 1024;
const devices = deviceIds.map(deviceId => new GlmOps(deviceId, undefined, arenaGb));
const parallelOps = new ParallelOps(devices);
const workspace = new WorkspaceBase(parallelOps);

try {
  const shared = workspace.alloc([4], "BF16", undefined, TensorParallelism.PartialSum) as ParallelTensor;
  shared.allReduce();
  parallelOps.synchronize();

  const received = shared.shards.map(shard => {
    const result = Buffer.alloc(8);
    shard.d2h(result);
    return result.toString("base64");
  });
  const p2pEnabled = parallelOps.p2pEnabled;
  workspace.free();
  parallelOps.free();
  for (const device of devices) device.free();
  process.send?.({ received, p2pEnabled }, () => process.disconnect?.());
} catch (error) {
  workspace.free();
  parallelOps.free();
  for (const device of devices) device.free();
  throw error;
}
