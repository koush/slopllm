import fs from "node:fs";
import path from "node:path";
import { GlmOps } from "./glm_ops";
import { resolveModelPath, listSafetensorsShards } from "./model_path";

async function main() {
  const glm = new GlmOps(0);
  console.log("GLM context:", glm.ctx);

  const buf = glm.alloc(256);
  console.log("GPU buffer:", buf);
  glm.freeBuf(buf);

  const modelDir = resolveModelPath("zai-org/GLM-5.1");
  console.log("Model path:", modelDir);

  const indexPath = path.join(modelDir, "model.safetensors.index.json");
  const index = JSON.parse(fs.readFileSync(indexPath, "utf-8"));
  const weightMap: Record<string, string> = index.weight_map;
  const allLayers = Object.keys(weightMap);
  const shards = [...new Set(Object.values(weightMap))];
  console.log(`Layers: ${allLayers.length}, Shards: ${shards.length}`);
  console.log(`Total size: ${(index.metadata.total_size / 1e12).toFixed(2)} TB`);

  const shardPaths = listSafetensorsShards(modelDir);
  console.log(`Shard files resolved: ${shardPaths.length}`);
  console.log(`First shard: ${shardPaths[0]}`);
  console.log(`File exists: ${fs.existsSync(shardPaths[0])}`);

  glm.free();
  console.log("Done.");
}

main().catch((err) => {
  console.error("Failed:", err);
  process.exit(1);
});
