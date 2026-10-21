import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { ChatModel } from "./chat_model";
import { DeviceOps } from "./device_ops";
import { GlmOps } from "./glm_ops";
import { resolveModelPath } from "./model_path";
import { ParallelOps, ParallelTensor } from "./parallel_ops";
import { inspectModel } from "./model_registry";

export const QWEN3_REPO = "Qwen/Qwen3-0.6B";
export const QWEN3_FP8_REPO = "Qwen/Qwen3-0.6B-FP8";
export const QWEN35_REPO = "Qwen/Qwen3.5-0.8B";
export const GLM51_REPO = "local-inference-lab/GLM-5.3-NVFP4-CSF";

export interface ModelCliArgs {
  gpus: number[];
  arena: number;
  arenaExplicit: boolean;
  modelDir: string | undefined;
  model: string | undefined;
  cp: boolean;
  /** Number of MTP draft tokens per step; 0 disables MTP. */
  mtp: number;
}

export interface ModelRuntime {
  model: ChatModel;
  ops: DeviceOps;
  gpuDevices: GlmOps[];
  modelDir: string;
  repoId: string;
}

export function parseModelArgs(argv: string[]): ModelCliArgs {
  const gpusEnv = process.env.SLOPLLM_GPU ?? "0";
  const args: ModelCliArgs = {
    gpus: gpusEnv.split(",").map(s => parseInt(s.trim(), 10)),
    arena: 0,
    arenaExplicit: false,
    modelDir: undefined,
    model: undefined,
    cp: false,
    mtp: 0,
  };

  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--gpus" && i + 1 < argv.length) args.gpus = argv[++i].split(",").map(s => parseInt(s.trim(), 10));
    else if (a === "--gpu" && i + 1 < argv.length) args.gpus = [parseInt(argv[++i], 10)];
    else if (a === "--arena" && i + 1 < argv.length) { args.arena = parseInt(argv[++i], 10); args.arenaExplicit = true; }
    else if (a === "--model" || a === "--model-dir") {
      const value = argv[++i];
      if (!value || value.startsWith("--")) throw new Error(`${a} requires a value`);
      if (a === "--model") args.model = value;
      else args.modelDir = value;
    }
    else if (["--qwen3", "--qwen35", "--qwen25", "--glm51", "--glm51-small", "--glm53-flash", "--fp8", "--nvfp4"].includes(a)) {
      throw new Error(`${a} has been removed. Use --model <HuggingFace repo ID> or --model-dir <local directory>; model type and weight format are detected from the checkpoint.`);
    }
    else if (a === "--cp") args.cp = true;
    else if (a === "--mtp") {
      const next = argv[i + 1];
      if (next !== undefined && /^-?\d+$/.test(next)) args.mtp = parseInt(argv[++i], 10);
      else args.mtp = 3;
    }
  }

  if (args.model && args.modelDir) throw new Error("--model and --model-dir are mutually exclusive");
  if (args.model && !/^[\w.-]+\/[\w.-]+$/.test(args.model)) throw new Error("--model requires a HuggingFace repo ID (owner/name); use --model-dir for a local directory");
  if (!args.arenaExplicit && !args.modelDir && (!args.model || args.model === GLM51_REPO)) args.arena = 92;
  if (args.gpus.length === 0 || args.gpus.some(gpu => !Number.isInteger(gpu) || gpu < 0)) {
    throw new Error(`Invalid GPU list: ${args.gpus.join(",")}`);
  }
  if (!Number.isInteger(args.arena) || args.arena < 0) {
    throw new Error(`Invalid arena size: ${args.arena}`);
  }
  if (!Number.isInteger(args.mtp) || args.mtp < 0) throw new Error(`Invalid --mtp draft depth: ${args.mtp}`);
  return args;
}

export function modelLabel(args: ModelCliArgs): string {
  return args.modelDir ?? args.model ?? GLM51_REPO;
}

export function resolveModelSelection(args: ModelCliArgs) {
  const repoId = modelLabel(args);
  const modelDir = args.modelDir !== undefined ? path.resolve(args.modelDir) : resolveModelPath(args.model ?? GLM51_REPO);
  if (!fs.statSync(modelDir).isDirectory()) throw new Error(`Model path is not a directory: ${modelDir}`);
  return { modelDir, repoId, ...inspectModel(modelDir, args) };
}

export function createDeviceOps(args: ModelCliArgs): { ops: DeviceOps, gpuDevices: GlmOps[] } {
  const gpuDevices = args.gpus.map(id => new GlmOps(id, undefined, args.arena || undefined));
  const ops: DeviceOps = gpuDevices.length > 1 ? new ParallelOps(gpuDevices) : gpuDevices[0];
  return { ops, gpuDevices };
}

export async function loadModel(ops: DeviceOps, args: ModelCliArgs, modelDir: string): Promise<ChatModel> {
  const { registration } = inspectModel(modelDir, args);
  const model = await registration.load(ops, modelDir, args);
  const devices = ops instanceof ParallelOps ? ops.devices : ops instanceof GlmOps ? [ops] : [];
  for (const device of devices) {
    const expected = process.env[`SLOPLLM_ARENA_LAYOUT_${device.device}`];
    const actual = device.arenaLayoutSignature();
    if (expected !== undefined && actual !== expected) {
      throw new Error(`Arena replay layout mismatch on device ${device.device}: expected ${expected}, got ${actual}`);
    }
  }
  if (devices.some(device => process.env[`SLOPLLM_MODEL_LAYOUT_${device.device}`] !== undefined)) {
    const modelLayouts = modelArenaLayoutSignatures(model, devices);
    for (const device of devices) {
      const expected = process.env[`SLOPLLM_MODEL_LAYOUT_${device.device}`];
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
  const { ops, gpuDevices } = createDeviceOps(args);
  try {
    const model = await loadModel(ops, args, modelDir);
    return { model, ops, gpuDevices, modelDir, repoId };
  } catch (error) {
    if (ops instanceof ParallelOps) ops.free();
    for (const device of gpuDevices) device.free();
    throw error;
  }
}

export function freeModelRuntime(runtime: ModelRuntime): void {
  runtime.ops.synchronize();
  runtime.model.free();
  if (runtime.ops instanceof ParallelOps) runtime.ops.free();
  for (const device of runtime.gpuDevices) device.free();
}
