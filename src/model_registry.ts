import fs from "node:fs";
import path from "node:path";
import type { ChatModel } from "./chat_model";
import { resolveCheckpointConfigDir } from "./checkpoint_source";
import type { DeviceOps } from "./device_ops";
import { Glm51Model } from "./glm51_model";
import { Glm53FlashModel } from "./glm53_flash_model";
import { DeepseekV41Model } from "./deepseek_v41_model";
import { Qwen3Model } from "./qwen3_model";
import { Qwen35Model } from "./qwen35_model";

export interface ModelOptions {
  cp: boolean;
  mtp: number;
}

export interface ModelRegistration {
  phasedPrefill: boolean;
  contextParallel: boolean;
  mtp: boolean;
  persistentLoader: boolean;
  load: (ops: DeviceOps, dir: string, options: ModelOptions) => Promise<ChatModel>;
}

export const modelRegistry = {
  deepseek_v41: {
    phasedPrefill: false, contextParallel: false, mtp: true, persistentLoader: true,
    load: (ops, dir, options) => DeepseekV41Model.fromPretrained(ops, dir, options.mtp > 0),
  },
  glm_moe_dsa: {
    phasedPrefill: true, contextParallel: true, mtp: true, persistentLoader: true,
    load: (ops, dir, options) => Glm51Model.fromPretrained(ops, dir, options.cp, options.mtp > 0),
  },
  glm5_next: {
    phasedPrefill: false, contextParallel: false, mtp: false, persistentLoader: true,
    load: (ops, dir) => Glm53FlashModel.fromPretrained(ops, dir),
  },
  qwen3: {
    phasedPrefill: false, contextParallel: false, mtp: false, persistentLoader: true,
    load: (ops, dir) => Qwen3Model.fromPretrained(ops, dir),
  },
  qwen3_5: {
    phasedPrefill: false, contextParallel: false, mtp: false, persistentLoader: false,
    load: (ops, dir) => Qwen35Model.fromPretrained(ops, dir),
  },
} satisfies Record<string, ModelRegistration>;

export type ModelType = keyof typeof modelRegistry;

export function inspectModel(modelDir: string, options: ModelOptions) {
  const configPath = path.join(resolveCheckpointConfigDir(modelDir), "config.json");
  const config = JSON.parse(fs.readFileSync(configPath, "utf8"));
  const modelType = config?.model_type;
  if (typeof modelType !== "string" || !Object.hasOwn(modelRegistry, modelType)) {
    throw new Error(`Unsupported model_type ${JSON.stringify(modelType) ?? "(missing)"} in ${configPath}. Supported types: ${Object.keys(modelRegistry).join(", ")}`);
  }
  const registration: ModelRegistration = modelRegistry[modelType as ModelType];
  if (options.cp && !registration.contextParallel) throw new Error(`${modelType} does not support --cp`);
  if (options.mtp > 0 && !registration.mtp) throw new Error(`${modelType} does not support --mtp`);
  if (modelType === "deepseek_v41" && options.mtp > 5) throw new Error("DeepSeek DSpark supports --mtp verification depths 1..5");
  return { modelType: modelType as ModelType, registration, config };
}
