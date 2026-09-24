import { GlmOps } from "../../src/glm_ops";
import { WorkspaceBase } from "../../src/workspace";

const deviceId = parseInt(process.argv[2], 10);
const arenaGb = 1 / 1024;
const ops = new GlmOps(deviceId, undefined, arenaGb);
const workspace = new WorkspaceBase(ops);

try {
  const shared = workspace.alloc([4], "BF16", "shared");
  const initial = Buffer.alloc(8);
  shared.d2h(initial);
  shared.scaleInPlace(2, 4);
  ops.synchronize();
  workspace.free();
  ops.free();
  process.send?.({ received: initial.toString("base64") }, () => process.disconnect?.());
} catch (error) {
  workspace.free();
  ops.free();
  throw error;
}
