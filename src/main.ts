import fs from "node:fs";
import path from "node:path";
import { resolveModelPath, listSafetensorsShards } from "./model_path";

const glm: {
  init(deviceId: number): number;
  free(ctx: number): void;
  alloc(ctx: number, size: number): number;
  freeBuf(ctx: number, ptr: number): void;
  h2d(ctx: number, dst: number, src: Buffer, size: number): void;
  d2h(ctx: number, dst: Buffer, src: number, size: number): void;
  rmsnorm(ctx: number, out: number, input: number, weight: number, eps: number, dim: number, batch: number): void;
  siluAndMul(ctx: number, out: number, gate: number, up: number, intermediate: number, batch: number): void;
  linear(ctx: number, out: number, input: number, weight: number, batch: number, n: number, k: number): void;
  embedding(ctx: number, out: number, table: number, ids: number, hidden: number, seqLen: number): void;
} = require("../build/Release/glm.node");

async function main() {
  const ctx = glm.init(0);
  console.log("GLM context:", ctx);

  const alloc = glm.alloc(ctx, 256);
  console.log("GPU buffer:", alloc);
  glm.freeBuf(ctx, alloc);

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

  glm.free(ctx);
  console.log("Done.");
}

main().catch((err) => {
  console.error("Failed:", err);
  process.exit(1);
});
