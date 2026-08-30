import { GlmOps } from "../../src/glm_ops";
import { WorkspaceBase } from "../../src/workspace";

const deviceId = parseInt(process.argv[2], 10);
const arenaGb = 1 / 1024;
const glm = new GlmOps(deviceId, undefined, arenaGb);
const workspace = new WorkspaceBase(glm);

try {
  const shared = workspace.alloc([4], "BF16", "shared");
  const initial = Buffer.alloc(8);
  shared.d2h(initial);
  shared.scaleInPlace(2, 4);
  glm.synchronize();
  workspace.free();
  glm.free();
  process.send?.({ received: initial.toString("base64") }, () => process.disconnect?.());
} catch (error) {
  workspace.free();
  glm.free();
  throw error;
}
