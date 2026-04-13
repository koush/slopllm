import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export class ModelNotFoundError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ModelNotFoundError";
  }
}

export function resolveModelPath(repoId: string): string {
  const hfHome = process.env.HF_HOME || path.join(os.homedir(), ".cache", "huggingface");
  const modelDir = path.join(hfHome, "hub", `models--${repoId.replace("/", "--")}`);

  if (!fs.existsSync(modelDir)) {
    throw new ModelNotFoundError(
      `Model "${repoId}" not found in HuggingFace cache.\n` +
        `  Looked at: ${modelDir}\n` +
        `  HF_HOME: ${hfHome}\n` +
        `  Download with: huggingface-cli download ${repoId}`
    );
  }

  const refPath = path.join(modelDir, "refs", "main");
  if (!fs.existsSync(refPath)) {
    throw new ModelNotFoundError(
      `No refs/main found for "${repoId}" — model may not be fully downloaded.\n` +
        `  Expected: ${refPath}`
    );
  }
  const commitHash = fs.readFileSync(refPath, "utf-8").trim();

  const snapshotDir = path.join(modelDir, "snapshots", commitHash);
  if (!fs.existsSync(snapshotDir)) {
    throw new ModelNotFoundError(
      `Snapshot directory not found for "${repoId}".\n` +
        `  Expected: ${snapshotDir}\n` +
        `  Commit hash: ${commitHash}`
    );
  }

  return snapshotDir;
}

export function listSafetensorsShards(snapshotDir: string): string[] {
  const indexPath = path.join(snapshotDir, "model.safetensors.index.json");
  if (!fs.existsSync(indexPath)) {
    const files = fs.readdirSync(snapshotDir).filter((f: string) => f.endsWith(".safetensors"));
    if (files.length === 0) {
      throw new Error(`No safetensors files found in ${snapshotDir}`);
    }
    return files.map((f: string) => path.join(snapshotDir, f));
  }

  const index = JSON.parse(fs.readFileSync(indexPath, "utf-8"));
  const shardFiles = [...new Set(Object.values(index.weight_map as Record<string, string>))];
  return shardFiles.map((f) => path.join(snapshotDir, f));
}
