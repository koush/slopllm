import fs from "node:fs";
import path from "node:path";

import { CsfContractError, type CsfScaleCodec } from "./csf_codec";
import {
  CsfContainer,
  detectCsfContainer,
  type CsfContract,
} from "./csf_checkpoint";
import { SafeTensorFile, type TensorMeta } from "./safetensors";

// Checkpoint-source abstraction (port plan §9.2): the shared weight-loading
// boundary behind ChatModel.loadWeights(). Consumers see logical source
// names, dtype/shape and bounded tensor/slice reads; physical fixed and
// exception planes stay behind the CSF implementation. Both plain
// safetensors roots and CSF container roots load through this interface.

export { CsfContractError } from "./csf_codec";

/** Direct file mapping of a tensor's logical bytes, when physically contiguous. */
export interface TensorMapping {
  mmapPtr: number;
  offset: number;
  size: number;
}

export interface TensorExtent {
  /** Window shape in logical elements. */
  shape: number[];
  /** Window origin in logical element coordinates. */
  offsets: number[];
}

export interface CheckpointTensor {
  readonly name: string;
  /** Logical dtype/shape; replaced CSF scales report their original shape. */
  readonly meta: TensorMeta;
  /** Compressed source planes for GPU expansion; sizes are available during allocation replay. */
  readonly csf?: {
    codec: CsfScaleCodec;
    rows: number;
    columns: number;
    fixedBytes: number;
    exceptionBytes: number;
    mapping(): { fixed: TensorMapping; exceptions: TensorMapping } | undefined;
    read(): { fixed: Buffer; exceptions: Buffer };
  };
  /** File mapping of the logical bytes; undefined for derived tensors (CSF scale expansion) or without a mapping registry. */
  mapping(): TensorMapping | undefined;
  /** Full logical tensor bytes (lossless CSF scale expansion included). */
  read(): Buffer;
  /** Bounded logical slice read. */
  readSlice(extent: TensorExtent): Buffer;
}

export interface MmapRegistry {
  open(filePath: string): number;
  close(mmapPtr: number, size: number): void;
}

export interface CheckpointSource {
  readonly kind: "safetensors" | "csf";
  readonly root: string;
  /** Directory holding config.json/tokenizer/generation config/chat template. */
  configDir(): string;
  names(): string[];
  meta(name: string): TensorMeta;
  tensor(name: string): CheckpointTensor;
  close(): void;
}

/**
 * Positive finite FP32 scalar validation for NVFP4 calibration tensors
 * (input_scale). Source values are retained unreciprocated: the reference
 * preparation reciprocates gate/up max and down input scales into its packed
 * bundle's input_scale/intermediate_scale fields at A4-preparation time; the
 * loader must never apply that reciprocal itself.
 */
export function validateCalibrationScalar(name: string, meta: TensorMeta, bytes: Buffer): number {
  if (meta.dtype !== "F32") {
    throw new CsfContractError(`calibration tensor ${name} must be F32, got ${meta.dtype}`);
  }
  const numElements = meta.shape.reduce((a, b) => a * b, 1);
  if (meta.shape.length > 1 || numElements !== 1 || bytes.length !== 4) {
    throw new CsfContractError(`calibration tensor ${name} must be one FP32 value`);
  }
  const value = bytes.readFloatLE(0);
  if (!Number.isFinite(value) || !(value > 0)) {
    throw new CsfContractError(`calibration tensor ${name} must be positive and finite`);
  }
  return value;
}

interface ShardEntry {
  st: SafeTensorFile;
  path: string;
  mmapPtr?: number;
  fileSize: number;
}

class SafetensorsCheckpointSource implements CheckpointSource {
  readonly kind = "safetensors" as const;
  readonly root: string;
  private shards: ShardEntry[] = [];
  private byName = new Map<string, { shard: ShardEntry; meta: TensorMeta }>();
  private readonly namesList: string[] = [];
  private registry?: MmapRegistry;

  constructor(root: string, registry?: MmapRegistry) {
    this.root = root;
    this.registry = registry;
    const allFiles = fs.readdirSync(root);
    const shardPaths: string[] = [];
    if (allFiles.includes("model.safetensors")) {
      shardPaths.push(path.join(root, "model.safetensors"));
    } else {
      const indexFile = allFiles.find(f => f.endsWith(".index.json") || f.endsWith(".safetensors.json"));
      if (indexFile) {
        const idx = JSON.parse(fs.readFileSync(path.join(root, indexFile), "utf-8"));
        const weightMap = idx.weight_map ?? idx;
        for (const f of Object.values(weightMap)) {
          if (typeof f === "string" && f.endsWith(".safetensors") && !shardPaths.includes(path.join(root, f))) {
            shardPaths.push(path.join(root, f));
          }
        }
      } else {
        shardPaths.push(...allFiles.filter(f => f.endsWith(".safetensors")).map(f => path.join(root, f)));
      }
    }
    // Iterating the index's weight_map gives shards in first-mention order
    // (arbitrary: lm_head/embed sort before model.layers.*, so the shard
    // with the final layers loads first). Load in filename order so
    // progress tracks the checkpoint layout.
    shardPaths.sort();
    for (const shardPath of shardPaths) {
      const st = SafeTensorFile.open(shardPath);
      const fileSize = fs.statSync(shardPath).size;
      const shard: ShardEntry = { st, path: shardPath, fileSize };
      this.shards.push(shard);
      for (const name of st.tensorNames()) {
        this.byName.set(name, { shard, meta: st.meta(name) });
        this.namesList.push(name);
      }
    }
  }

  configDir(): string {
    return this.root;
  }

  names(): string[] {
    return [...this.namesList];
  }

  meta(name: string): TensorMeta {
    return this.require(name).meta;
  }

  private require(name: string): { shard: ShardEntry; meta: TensorMeta } {
    const entry = this.byName.get(name);
    if (!entry) throw new Error(`Tensor '${name}' not found in checkpoint ${this.root}`);
    return entry;
  }

  tensor(name: string): CheckpointTensor {
    const entry = this.require(name);
    const self = this;
    return {
      name,
      meta: entry.meta,
      mapping(): TensorMapping | undefined {
        if (!self.registry) return undefined;
        const shard = entry.shard;
        if (shard.mmapPtr === undefined) {
          shard.mmapPtr = self.registry.open(shard.path);
        }
        return {
          mmapPtr: shard.mmapPtr,
          offset: shard.st.tensorOffset(name),
          size: shard.st.tensorSize(name),
        };
      },
      read(): Buffer {
        return entry.shard.st.readTensor(name);
      },
      readSlice(extent: TensorExtent): Buffer {
        return readRowMajorSlice((offset, size) => entry.shard.st.readTensorRange(name, offset, size), entry.meta, extent, name);
      },
    };
  }

  close(): void {
    for (const shard of this.shards) {
      shard.st.close();
      if (shard.mmapPtr !== undefined && this.registry) {
        this.registry.close(shard.mmapPtr, shard.fileSize);
      }
    }
    this.shards = [];
    this.byName.clear();
  }
}

/** Row-major logical slice over the leading two dimensions (inner dims full). */
export function readRowMajorSlice(data: Buffer | ((offset: number, size: number) => Buffer), meta: TensorMeta, extent: TensorExtent, name: string): Buffer {
  const rank = meta.shape.length;
  if (extent.shape.length !== rank || extent.offsets.length !== rank) {
    throw new Error(`slice rank mismatch for ${name}`);
  }
  for (let d = 0; d < rank; d++) {
    if (!Number.isSafeInteger(extent.offsets[d]) || !Number.isSafeInteger(extent.shape[d]) || extent.offsets[d] < 0 || extent.shape[d] < 0
      || extent.offsets[d] + extent.shape[d] > meta.shape[d]) {
      throw new Error(`slice for ${name} is out of bounds`);
    }
    if (d >= 2 && (extent.offsets[d] !== 0 || extent.shape[d] !== meta.shape[d])) {
      throw new Error(`slices may only window the leading two dimensions of ${name}`);
    }
  }
  const dtypeBytes = SafeTensorFile.dtypeBytes(meta.dtype);
  const rowBytes = meta.shape.slice(1).reduce((a, b) => a * b, 1) * dtypeBytes;
  const innerElements = rank >= 2 ? extent.shape[1] * meta.shape.slice(2).reduce((a, b) => a * b, 1) : 1;
  const innerBytes = innerElements * dtypeBytes;
  const srcBase = rank >= 2
    ? extent.offsets[0] * rowBytes + extent.offsets[1] * meta.shape.slice(2).reduce((a, b) => a * b, 1) * dtypeBytes
    : extent.offsets[0] * dtypeBytes;
  const out = Buffer.allocUnsafe(extent.shape.reduce((a, b) => a * b, 1) * dtypeBytes);
  for (let r = 0; r < extent.shape[0]; r++) {
    const src = srcBase + r * rowBytes;
    if (typeof data === "function") data(src, innerBytes).copy(out, r * innerBytes);
    else data.copy(out, r * innerBytes, src, src + innerBytes);
  }
  return out;
}

class CsfCheckpointSource implements CheckpointSource {
  readonly kind = "csf" as const;
  readonly root: string;
  readonly contract: CsfContract;
  private container: CsfContainer;
  private registry?: MmapRegistry;
  private mmapPtrs = new Map<string, number>();

  constructor(root: string, registry?: MmapRegistry) {
    this.root = root;
    this.registry = registry;
    this.container = CsfContainer.open(root);
    this.contract = this.container.contract;
  }

  configDir(): string {
    return path.join(this.root, "metadata");
  }

  names(): string[] {
    return [...this.container.order];
  }

  meta(name: string): TensorMeta {
    return this.container.locate(name).meta;
  }

  tensor(name: string): CheckpointTensor {
    const located = this.container.locate(name);
    const container = this.container;
    const self = this;
    const planes = container.compressedPlanes(name);
    return {
      name,
      meta: located.meta,
      csf: planes && {
        ...planes,
        mapping: () => {
          const fixed = self.mapRange(planes.file, planes.fixedOffset, planes.fixedBytes);
          const exceptions = self.mapRange(planes.file, planes.exceptionOffset, planes.exceptionBytes);
          return fixed && exceptions ? { fixed, exceptions } : undefined;
        },
      },
      mapping(): TensorMapping | undefined {
        const range = container.physicalRange(name);
        return range && self.mapRange(range.file, range.offset, range.size);
      },
      read(): Buffer {
        return container.readTensor(name);
      },
      readSlice(extent: TensorExtent): Buffer {
        return container.readTensorSlice(name, extent);
      },
    };
  }

  private mapRange(file: string, offset: number, size: number): TensorMapping | undefined {
    if (!this.registry) return undefined;
    let mmapPtr = this.mmapPtrs.get(file);
    if (mmapPtr === undefined) {
      mmapPtr = this.registry.open(path.join(this.root, "tensors", file));
      this.mmapPtrs.set(file, mmapPtr);
    }
    return { mmapPtr, offset, size };
  }

  close(): void {
    this.container.close();
    if (this.registry) {
      for (const [file, mmapPtr] of this.mmapPtrs) {
        const record = this.contract.shards.get(file);
        this.registry.close(mmapPtr, record ? record.targetFileBytes : 0);
      }
    }
    this.mmapPtrs.clear();
  }
}

/**
 * Open a checkpoint root as safetensors or CSF. A root carrying any CSF
 * container marker is recognized as a CSF container and validated
 * fail-closed: a recognized-but-malformed container throws
 * CsfContractError and never falls through to a safetensors reading.
 */
export function openCheckpointSource(modelDir: string, registry?: MmapRegistry): CheckpointSource {
  if (detectCsfContainer(modelDir)) {
    return new CsfCheckpointSource(modelDir, registry);
  }
  return new SafetensorsCheckpointSource(modelDir, registry);
}

/**
 * Directory that holds model config, tokenizer, generation config and chat
 * template: the container's metadata/ for CSF roots, the root itself for
 * safetensors checkpoints.
 */
export function resolveCheckpointConfigDir(modelDir: string): string {
  if (!detectCsfContainer(modelDir)) return modelDir;
  const metadataDir = path.join(modelDir, "metadata");
  if (!fs.existsSync(path.join(metadataDir, "config.json"))) {
    throw new CsfContractError(`CSF container ${modelDir} has no metadata/config.json`);
  }
  return metadataDir;
}
