import { ChatModel } from "./chat_model";
import { DeviceOps } from "./device_ops";
import { Glm51Model } from "./glm51_model";
import { GlmOps } from "./glm_ops";
import { resolveModelPath } from "./model_path";
import { ParallelOps } from "./parallel_ops";
import { Qwen35Model } from "./qwen35_model";
import { Qwen3Model } from "./qwen3_model";

export const QWEN3_REPO = "Qwen/Qwen3-0.6B";
export const QWEN3_FP8_REPO = "Qwen/Qwen3-0.6B-FP8";
export const QWEN35_REPO = "Qwen/Qwen3.5-0.8B";
export const GLM51_REPO = "incoai/GLM-5.3-NVFP4";

const GLM51_SMALL_BF16 = "tests/python/test_models/glm51_small/glm51_small_bf16";
const GLM51_SMALL_NVFP4 = "tests/python/test_models/glm51_small/glm51_small_nvfp4";
const GLM51_MODEL_DIR = "/mnt/storage/.cache/huggingface/hub/models--incoai--GLM-5.3-NVFP4/snapshots/54e52520606f96b3d9fc84088ad22882a61648ac/";

export interface ModelCliArgs {
  gpus: number[];
  arena: number;
  modelDir: string | undefined;
  useQwen35: boolean;
  useGlm51: boolean;
  glm51Small: boolean;
  useFp8: boolean;
  useNvfp4: boolean;
  cp: boolean;
  mtp: boolean;
}

export interface ModelRuntime {
  model: ChatModel;
  glm: DeviceOps;
  gpuDevices: GlmOps[];
  modelDir: string;
  repoId: string;
}

export function parseModelArgs(argv: string[]): ModelCliArgs {
  const gpusEnv = process.env.GLM_GPUS ?? process.env.GLM_GPU ?? "0";
  const args: ModelCliArgs = {
    gpus: gpusEnv.split(",").map(s => parseInt(s.trim(), 10)),
    arena: 0,
    modelDir: undefined,
    useQwen35: false,
    useGlm51: false,
    glm51Small: false,
    useFp8: false,
    useNvfp4: false,
    cp: false,
    mtp: false,
  };

  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--gpus" && i + 1 < argv.length) args.gpus = argv[++i].split(",").map(s => parseInt(s.trim(), 10));
    else if (a === "--gpu" && i + 1 < argv.length) args.gpus = [parseInt(argv[++i], 10)];
    else if (a === "--arena" && i + 1 < argv.length) args.arena = parseInt(argv[++i], 10);
    else if (a === "--model-dir" && i + 1 < argv.length) args.modelDir = argv[++i];
    else if (a === "--qwen35") args.useQwen35 = true;
    else if (a === "--glm51") args.useGlm51 = true;
    else if (a === "--glm51-small") { args.useGlm51 = true; args.glm51Small = true; }
    else if (a === "--fp8") args.useFp8 = true;
    else if (a === "--nvfp4") args.useNvfp4 = true;
    else if (a === "--cp") args.cp = true;
    else if (a === "--mtp") args.mtp = true;
  }

  if (args.gpus.length === 0 || args.gpus.some(gpu => !Number.isInteger(gpu) || gpu < 0)) {
    throw new Error(`Invalid GPU list: ${args.gpus.join(",")}`);
  }
  if (!Number.isInteger(args.arena) || args.arena < 0) {
    throw new Error(`Invalid arena size: ${args.arena}`);
  }
  if (args.useQwen35 && args.useGlm51) throw new Error("--qwen35 and --glm51 are mutually exclusive");
  if (args.useQwen35 && args.useFp8) throw new Error("--fp8 is not supported with --qwen35");
  if (args.useGlm51 && args.useFp8) throw new Error("--fp8 is not supported with --glm51");
  if (args.useNvfp4 && !args.useGlm51) throw new Error("--nvfp4 is only supported with --glm51");
  return args;
}

export function modelLabel(args: ModelCliArgs): string {
  if (args.useQwen35) return "Qwen3.5-0.8B";
  if (args.useGlm51) return args.glm51Small ? "GLM-5.1-small" : (args.useNvfp4 ? "GLM-5.1-NVFP4" : "GLM-5.1");
  return args.useFp8 ? "Qwen3-0.6B-FP8" : "Qwen3-0.6B";
}

export function resolveModelSelection(args: ModelCliArgs): { modelDir: string, repoId: string } {
  const repoId = args.useGlm51 ? GLM51_REPO
    : args.useQwen35 ? QWEN35_REPO
      : (args.useFp8 ? QWEN3_FP8_REPO : QWEN3_REPO);
  const modelDir = args.modelDir ?? (args.useGlm51
    ? (args.glm51Small
      ? (args.useNvfp4 ? GLM51_SMALL_NVFP4 : GLM51_SMALL_BF16)
      : GLM51_MODEL_DIR)
    : resolveModelPath(repoId));
  return { modelDir, repoId };
}

export function createDeviceOps(args: ModelCliArgs): { glm: DeviceOps, gpuDevices: GlmOps[] } {
  const gpuDevices = args.gpus.map(id => new GlmOps(id, undefined, args.arena || undefined));
  const glm: DeviceOps = gpuDevices.length > 1 ? new ParallelOps(gpuDevices) : gpuDevices[0];
  return { glm, gpuDevices };
}

export async function loadModel(glm: DeviceOps, args: ModelCliArgs, modelDir: string): Promise<ChatModel> {
  return args.useGlm51
    ? Glm51Model.fromPretrained(glm, modelDir, args.cp, args.mtp)
    : args.useQwen35
      ? Qwen35Model.fromPretrained(glm, modelDir)
      : Qwen3Model.fromPretrained(glm, modelDir);
}

export async function loadModelRuntime(args: ModelCliArgs): Promise<ModelRuntime> {
  const { modelDir, repoId } = resolveModelSelection(args);
  const { glm, gpuDevices } = createDeviceOps(args);
  try {
    const model = await loadModel(glm, args, modelDir);
    return { model, glm, gpuDevices, modelDir, repoId };
  } catch (error) {
    if (glm instanceof ParallelOps) glm.free();
    for (const device of gpuDevices) device.free();
    throw error;
  }
}

export function freeModelRuntime(runtime: ModelRuntime): void {
  runtime.glm.synchronize();
  runtime.model.free();
  if (runtime.glm instanceof ParallelOps) runtime.glm.free();
  for (const device of runtime.gpuDevices) device.free();
}
