import { createHash } from "node:crypto";
import { ChatModel } from "./chat_model";
import { DeviceOps } from "./device_ops";
import { Glm51Model } from "./glm51_model";
import { GlmOps } from "./glm_ops";
import { resolveModelPath } from "./model_path";
import { ParallelOps, ParallelTensor } from "./parallel_ops";
import { Qwen35Model } from "./qwen35_model";
import { Qwen3Model } from "./qwen3_model";

export const QWEN3_REPO = "Qwen/Qwen3-0.6B";
export const QWEN3_FP8_REPO = "Qwen/Qwen3-0.6B-FP8";
export const QWEN35_REPO = "Qwen/Qwen3.5-0.8B";
export const GLM51_REPO = "local-inference-lab/GLM-5.3-NVFP4";

const GLM51_SMALL_BF16 = "tests/python/test_models/glm51_small/glm51_small_bf16";
const GLM51_SMALL_NVFP4 = "tests/python/test_models/glm51_small/glm51_small_nvfp4";
const GLM51_MODEL_DIR = "/mnt/storage/.cache/huggingface/hub/models--local-inference-lab--GLM-5.3-NVFP4/snapshots/cca10d1586255195d3279785fc85577bfc1e9227/";

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
  const model = await (args.useGlm51
    ? Glm51Model.fromPretrained(glm, modelDir, args.cp, args.mtp)
    : args.useQwen35
      ? Qwen35Model.fromPretrained(glm, modelDir)
      : Qwen3Model.fromPretrained(glm, modelDir));
  const devices = glm instanceof ParallelOps ? glm.devices : glm instanceof GlmOps ? [glm] : [];
  for (const device of devices) {
    const expected = process.env[`GLM_ARENA_LAYOUT_${device.device}`];
    const actual = device.arenaLayoutSignature();
    if (expected !== undefined && actual !== expected) {
      throw new Error(`Arena replay layout mismatch on device ${device.device}: expected ${expected}, got ${actual}`);
    }
  }
  if (devices.some(device => process.env[`GLM_MODEL_LAYOUT_${device.device}`] !== undefined)) {
    const modelLayouts = modelArenaLayoutSignatures(model, devices);
    for (const device of devices) {
      const expected = process.env[`GLM_MODEL_LAYOUT_${device.device}`];
      const actual = modelLayouts.get(device.device);
      if (expected !== undefined && actual !== expected) {
        throw new Error(`Model arena layout mismatch on device ${device.device}: expected ${expected}, got ${actual}`);
      }
    }
  }
  return model;
}

export function modelArenaLayoutSignatures(model: ChatModel, devices: readonly GlmOps[]): Map<number, string> {
  const hashes = devices.map(() => createHash("sha256"));
  const tensors = [...model.tensors.entries()].sort(([a], [b]) => a.localeCompare(b));
  for (const [name, tensor] of tensors) {
    for (let i = 0; i < devices.length; i++) {
      if (devices[i].arenaBase === undefined) throw new Error(`Device ${devices[i].device} does not have an arena`);
      const local = tensor instanceof ParallelTensor ? tensor.shard(i) : tensor;
      hashes[i].update(`${name}\0${local.data - devices[i].arenaBase!}\0${local.allocSize}\0${local.type}\n`);
    }
  }
  return new Map(devices.map((device, i) => [device.device, hashes[i].digest("hex")]));
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
