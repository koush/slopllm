import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

import {
  CsfContractError,
  NVFP4_CSF_CODEC,
  NVFP4_CSF_SCHEMA,
  NVFP4_GROUP_SIZE,
  SUPPORTED_CSF_FAMILIES,
  expandScalePlanes,
  expandScaleWindow,
  fixedStreamLength,
  nvfp4ScaleGeometry,
  requireLocalSafetensorsName,
} from "./csf_codec";
import { SafeTensorFile, type TensorMeta } from "./safetensors";

// GLM-5.3 NVFP4-CSF checkpoint container reader (`lil-nvfp4-csf-checkpoint/1`).
//
// Validation is split per the port plan §9.2:
//
//  * CONTRACT checks (here, fail-closed at open): storage identity (exact
//    schema/codec/family combination), complete source inventory, tensor
//    shapes and dtypes, shard extents, fixed-stream sizes. Plane-level
//    contract checks (sorted unique exception positions, valid scale
//    values, row bases) run when a plane is decoded in csf_codec.ts.
//  * INTEGRITY/PROVENANCE checks (verifyCsfIntegrity / reconstructSourceShard):
//    recompute target hashes against the manifest and receipts, reconcile
//    the recorded source hashes and metadata hashes, and stream exact source
//    shard reconstruction. A recorded verification.json `passed: true` is
//    never treated as a check of the files being opened now.
//
// The source safetensors index under metadata/ describes ORIGINAL tensors —
// including the scales that were replaced by compressed planes. It is only
// used to cross-check the logical inventory, never as an index of the
// physical compressed files.

export const NVFP4_CSF_PLANE_TAG = "nvfp4_csf";

export interface CsfShardRecord {
  file: string;
  sourceFileBytes: number;
  sourceSha256: string;
  targetFileBytes: number;
  targetSha256: string;
}

export interface CsfScaleRecord {
  dtype: string;
  shape: number[];
  sourceBytes: number;
  sourceSha256: string;
  fixedBytes: number;
  exceptionBytes: number;
}

export interface CsfContract {
  root: string;
  schema: string;
  codec: string;
  family: string;
  sourceModelId: string;
  sourceRevision: string;
  sourceModelPath: string;
  /** Logical source name -> shard file (contract.source_names). */
  sourceNames: Map<string, string>;
  shards: Map<string, CsfShardRecord>;
  /** Logical replaced-scale name -> receipt scale record. */
  scales: Map<string, CsfScaleRecord>;
  metadataSha256: Record<string, string>;
  compressedScaleCount: number;
}

export interface CsfShardLayout {
  file: string;
  /** Logical native tensors stored byte-identically to the source. */
  native: Map<string, TensorMeta>;
  /** Logical replaced-scale name -> physical planes + receipt record. */
  compressed: Map<string, {
    fixedName: string;
    exceptionName: string;
    fixed: TensorMeta;
    exceptions: TensorMeta;
    record: CsfScaleRecord;
    weightShape: number[];
  }>;
}

function readJson(root: string, relative: string): unknown {
  const filePath = path.join(root, relative);
  let text: string;
  try {
    text = fs.readFileSync(filePath, "utf-8");
  } catch (e) {
    throw new CsfContractError(`CSF container is missing ${relative}: ${(e as Error).message}`);
  }
  try {
    return JSON.parse(text);
  } catch (e) {
    throw new CsfContractError(`CSF container ${relative} is not valid JSON: ${(e as Error).message}`);
  }
}

function asRecord(value: unknown, what: string): Record<string, any> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new CsfContractError(`CSF ${what} must be a JSON object`);
  }
  return value as Record<string, any>;
}

function requireNumber(value: unknown, what: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) {
    throw new CsfContractError(`CSF ${what} must be a non-negative integer`);
  }
  return value;
}

function requireString(value: unknown, what: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new CsfContractError(`CSF ${what} must be a non-empty string`);
  }
  return value;
}

function sha256Hex(data: Buffer): string {
  return crypto.createHash("sha256").update(data).digest("hex");
}

/** CSF-specific metadata selects the container reader, including incomplete containers. */
export function detectCsfContainer(root: string): boolean {
  if (["build-contract.json", "receipts"].some(marker => fs.existsSync(path.join(root, marker)))) return true;
  // A tensors/ directory and a LIL license manifest also occur in ordinary checkpoints.
  const manifest = path.join(root, "manifest.json");
  return fs.existsSync(manifest) && fs.readFileSync(manifest, "utf8").includes("-csf-checkpoint/");
}

export interface ParsedSourceHeader {
  prefix: Buffer;
  tensors: Map<string, TensorMeta>;
  order: string[];
}

/** Parse a receipt's preserved raw source header (8-byte LE length + JSON). */
export function parseSourceHeader(headerBase64: string, what: string): ParsedSourceHeader {
  let raw: Buffer;
  try {
    raw = Buffer.from(headerBase64, "base64");
  } catch (e) {
    throw new CsfContractError(`CSF ${what} source header is not valid base64`);
  }
  if (raw.length < 8) {
    throw new CsfContractError(`CSF ${what} source header is truncated`);
  }
  const headerLen = Number(raw.readBigUInt64LE(0));
  if (headerLen !== raw.length - 8) {
    throw new CsfContractError(`CSF ${what} source header length prefix disagrees with the preserved bytes`);
  }
  let parsed: Record<string, any>;
  try {
    parsed = JSON.parse(raw.subarray(8).toString("utf-8"));
  } catch (e) {
    throw new CsfContractError(`CSF ${what} source header is not valid JSON: ${(e as Error).message}`);
  }
  const tensors = new Map<string, TensorMeta>();
  const order: string[] = [];
  for (const [name, entry] of Object.entries(parsed)) {
    if (name === "__metadata__") continue;
    const meta = asRecord(entry, `${what} source header entry ${name}`);
    if (typeof meta.dtype !== "string" || !Array.isArray(meta.shape) || !Array.isArray(meta.data_offsets)) {
      throw new CsfContractError(`CSF ${what} source header entry ${name} is malformed`);
    }
    tensors.set(name, {
      dtype: meta.dtype,
      shape: meta.shape as number[],
      dataOffsets: [meta.data_offsets[0], meta.data_offsets[1]],
    });
    order.push(name);
  }
  return { prefix: raw.subarray(0, 8 + headerLen), tensors, order };
}

export interface CsfOpenResult {
  contract: CsfContract;
  layouts: Map<string, CsfShardLayout>;
  /** Deterministic logical load order (shard file order, physical header order). */
  order: string[];
}

/**
 * CONTRACT validation: identity, complete source inventory, shapes/dtypes,
 * shard extents, fixed-stream sizes. Throws CsfContractError on any
 * mismatch; recognized-but-malformed containers never fall through to a
 * safetensors interpretation.
 */
export function readCsfContract(root: string): CsfOpenResult {
  root = path.resolve(root);
  const manifest = asRecord(readJson(root, "manifest.json"), "manifest");
  const contractJson = asRecord(readJson(root, "build-contract.json"), "build-contract");

  for (const [key, expected] of [["schema", NVFP4_CSF_SCHEMA], ["codec", NVFP4_CSF_CODEC]] as const) {
    if (manifest[key] !== expected || contractJson[key] !== expected) {
      throw new CsfContractError(`CSF requires ${key}=${expected} in both manifest and build-contract`);
    }
  }
  // Reject unsupported codec variants explicitly (e.g. the MXFP4
  // row-base-offset1-u24-exceptions/1 grammar has a different decoder).
  if (contractJson.codec !== NVFP4_CSF_CODEC) {
    throw new CsfContractError(`CSF codec variant ${contractJson.codec} is not supported`);
  }
  const family = requireString(contractJson.family, "family");
  if (!SUPPORTED_CSF_FAMILIES.includes(family)) {
    throw new CsfContractError(`CSF family ${family} is not supported (want one of ${SUPPORTED_CSF_FAMILIES.join(", ")})`);
  }
  if (manifest.family !== family) {
    throw new CsfContractError("CSF manifest and build-contract must name the same model family");
  }

  const manifestMeta = asRecord(manifest.metadata_sha256, "manifest metadata_sha256");
  const contractMeta = asRecord(contractJson.metadata_sha256, "build-contract metadata_sha256");
  if (JSON.stringify(manifestMeta) !== JSON.stringify(contractMeta)) {
    throw new CsfContractError("CSF manifest and build-contract disagree on metadata hashes");
  }

  const sourceNames = new Map<string, string>();
  for (const [name, file] of Object.entries(asRecord(contractJson.source_names, "source_names"))) {
    sourceNames.set(name, requireString(file, `source_names[${name}]`));
    requireLocalSafetensorsName(sourceNames.get(name)!);
  }
  if (sourceNames.size === 0) {
    throw new CsfContractError("CSF source inventory is empty");
  }

  const shards = new Map<string, CsfShardRecord>();
  const manifestShards = manifest.shards;
  if (!Array.isArray(manifestShards)) {
    throw new CsfContractError("CSF manifest shards must be a list");
  }
  for (const entry of manifestShards) {
    const item = asRecord(entry, "manifest shard");
    const file = requireString(item.file, "manifest shard file");
    requireLocalSafetensorsName(file);
    if (shards.has(file)) {
      throw new CsfContractError(`CSF manifest lists shard ${file} twice`);
    }
    shards.set(file, {
      file,
      sourceFileBytes: requireNumber(item.source_file_bytes, `shard ${file} source_file_bytes`),
      sourceSha256: requireString(item.source_sha256, `shard ${file} source_sha256`),
      targetFileBytes: requireNumber(item.target_file_bytes, `shard ${file} target_file_bytes`),
      targetSha256: requireString(item.target_sha256, `shard ${file} target_sha256`),
    });
  }
  const inventoryFiles = new Set(sourceNames.values());
  if (inventoryFiles.size !== shards.size || [...inventoryFiles].some(file => !shards.has(file))) {
    throw new CsfContractError("CSF manifest and source tensor inventory disagree on shard files");
  }

  // Complete source inventory: the metadata source index describes the same
  // original tensors (including replaced scales), mapping to the same source
  // shards. It is never used as an index of physical compressed files.
  const index = asRecord(readJson(root, "metadata/model.safetensors.index.json"), "source index");
  const weightMap = asRecord(index.weight_map, "source index weight_map");
  if (Object.keys(weightMap).length !== sourceNames.size
    || Object.entries(weightMap).some(([name, file]) => sourceNames.get(name) !== file)) {
    throw new CsfContractError("CSF source index and contract source_names disagree");
  }

  const compressedScaleCount = requireNumber(manifest.compressed_scale_count, "compressed_scale_count");
  const receipts = new Map<string, Record<string, any>>();
  const scales = new Map<string, CsfScaleRecord>();
  for (const [file, record] of shards) {
    const receipt = asRecord(readJson(root, `receipts/${file}.json`), `receipt for ${file}`);
    receipts.set(file, receipt);
    if (receipt.file !== file) {
      throw new CsfContractError(`CSF receipt for ${file} names a different file`);
    }
    // Hash/size reconciliation between receipts and the manifest belongs to
    // the integrity pass (verifyCsfIntegrity); the contract only needs the
    // receipt's scale records to validate shapes, dtypes and extents.
    const receiptScales = asRecord(receipt.scales, `receipt ${file} scales`);
    for (const [name, entry] of Object.entries(receiptScales)) {
      const item = asRecord(entry, `receipt ${file} scale ${name}`);
      const shape = Array.isArray(item.shape) ? (item.shape as number[]) : null;
      if (!shape || shape.length !== 2 || typeof item.dtype !== "string") {
        throw new CsfContractError(`CSF receipt ${file} scale ${name} has a malformed shape/dtype`);
      }
      if (scales.has(name)) {
        throw new CsfContractError(`CSF scale ${name} appears in two receipts`);
      }
      scales.set(name, {
        dtype: item.dtype,
        shape,
        sourceBytes: requireNumber(item.source_bytes, `scale ${name} source_bytes`),
        sourceSha256: requireString(item.source_sha256, `scale ${name} source_sha256`),
        fixedBytes: requireNumber(item.fixed_bytes, `scale ${name} fixed_bytes`),
        exceptionBytes: requireNumber(item.exception_bytes, `scale ${name} exception_bytes`),
      });
    }
  }
  if (scales.size !== compressedScaleCount) {
    throw new CsfContractError(`CSF manifest compressed_scale_count ${compressedScaleCount} disagrees with ${scales.size} receipt scale entries`);
  }

  // Physical layout + shapes/dtypes/extents against the target shards and
  // the preserved source headers. Scale matrices and their packed weights
  // are not required to share a shard; cross-tensor geometry checks run
  // against the global source inventory below.
  const layouts = new Map<string, CsfShardLayout>();
  const order: string[] = [];
  const globalMeta = new Map<string, TensorMeta>();
  for (const [file, record] of shards) {
    const filePath = path.join(root, "tensors", file);
    const fileSize = fs.existsSync(filePath) ? fs.statSync(filePath).size : -1;
    if (fileSize !== record.targetFileBytes) {
      throw new CsfContractError(`CSF shard ${file} extent is ${fileSize} bytes, manifest expects ${record.targetFileBytes}`);
    }
    let st: SafeTensorFile;
    try {
      st = SafeTensorFile.open(filePath);
    } catch (e) {
      throw new CsfContractError(`CSF shard ${file} is not a readable safetensors file: ${(e as Error).message}`);
    }
    try {
      const sourceHeader = parseSourceHeader(
        requireString(receipts.get(file)!.source_header_base64, `receipt ${file} source_header_base64`),
        file,
      );
      const layout: CsfShardLayout = { file, native: new Map(), compressed: new Map() };
      const physical = st.tensorNames();
      const logicalSeen = new Set<string>();
      const scalePlanes = new Map<string, { fixedName?: string; exceptionName?: string }>();
      for (const name of physical) {
        const meta = st.meta(name);
        const extentOk = meta.dataOffsets[0] >= 0 && meta.dataOffsets[1] <= record.targetFileBytes
          && meta.dataOffsets[1] >= meta.dataOffsets[0];
        if (!extentOk) {
          throw new CsfContractError(`CSF shard ${file} tensor ${name} has out-of-file extents`);
        }
        const fixedMatch = /^(.*)\.nvfp4_csf_fixed$/.exec(name);
        const exceptionMatch = /^(.*)\.nvfp4_csf_exceptions$/.exec(name);
        if (fixedMatch || exceptionMatch) {
          const scaleName = (fixedMatch ?? exceptionMatch)![1];
          const planes = scalePlanes.get(scaleName) ?? {};
          if (fixedMatch) planes.fixedName = name;
          else planes.exceptionName = name;
          scalePlanes.set(scaleName, planes);
          continue;
        }
        if (logicalSeen.has(name)) {
          throw new CsfContractError(`CSF shard ${file} lists ${name} twice`);
        }
        logicalSeen.add(name);
        layout.native.set(name, meta);
        order.push(name);
      }
      for (const [scaleName, planes] of scalePlanes) {
        if (!planes.fixedName || !planes.exceptionName) {
          throw new CsfContractError(`CSF shard ${file} scale ${scaleName} is missing one of its fixed/exception planes`);
        }
        const record = scales.get(scaleName);
        if (!record) {
          throw new CsfContractError(`CSF shard ${file} has planes for ${scaleName} but no receipt scale entry`);
        }
        const fixed = st.meta(planes.fixedName);
        const exceptions = st.meta(planes.exceptionName);
        const [rows, columns] = record.shape;
        if (record.dtype !== "F8_E4M3" || rows * columns !== record.sourceBytes) {
          throw new CsfContractError(`CSF scale ${scaleName} record does not describe an F8_E4M3 [rows, columns] matrix`);
        }
        if (fixedStreamLength(rows, columns) !== record.fixedBytes
          || fixed.dtype !== "U8" || fixed.shape[0] !== record.fixedBytes) {
          throw new CsfContractError(`CSF scale ${scaleName} fixed stream size disagrees with its shape`);
        }
        if (record.exceptionBytes % 4 !== 0
          || exceptions.dtype !== "U32" || exceptions.shape[0] * 4 !== record.exceptionBytes) {
          throw new CsfContractError(`CSF scale ${scaleName} exception plane size disagrees with its record`);
        }
        if (logicalSeen.has(scaleName)) {
          throw new CsfContractError(`CSF shard ${file} lists ${scaleName} twice`);
        }
        logicalSeen.add(scaleName);
        layout.compressed.set(scaleName, {
          fixedName: planes.fixedName,
          exceptionName: planes.exceptionName,
          fixed,
          exceptions,
          record,
          weightShape: [],
        });
        order.push(scaleName);
      }
      // Complete inventory for this shard: logical tensors == source_names
      // entries mapped here == preserved source header names; dtypes/shapes
      // of native tensors match the source header exactly.
      const expectedNames = [...sourceNames.entries()].filter(([, f]) => f === file).map(([n]) => n);
      if (expectedNames.length !== logicalSeen.size || expectedNames.some(name => !logicalSeen.has(name))) {
        throw new CsfContractError(`CSF shard ${file} physical tensors do not cover its source inventory`);
      }
      if (sourceHeader.order.length !== expectedNames.length
        || sourceHeader.order.some(name => !logicalSeen.has(name))) {
        throw new CsfContractError(`CSF shard ${file} preserved source header disagrees with the source inventory`);
      }
      for (const [name, meta] of layout.native) {
        const source = sourceHeader.tensors.get(name);
        if (!source || source.dtype !== meta.dtype || JSON.stringify(source.shape) !== JSON.stringify(meta.shape)) {
          throw new CsfContractError(`CSF shard ${file} tensor ${name} dtype/shape disagrees with its source header`);
        }
        globalMeta.set(name, meta);
      }
      for (const [name, entry] of layout.compressed) {
        const source = sourceHeader.tensors.get(name);
        if (!source || source.dtype !== entry.record.dtype
          || JSON.stringify(source.shape) !== JSON.stringify(entry.record.shape)) {
          throw new CsfContractError(`CSF shard ${file} scale ${name} dtype/shape disagrees with its source header`);
        }
        globalMeta.set(name, { dtype: entry.record.dtype, shape: [...entry.record.shape], dataOffsets: [0, entry.record.sourceBytes] });
      }
      layouts.set(file, layout);
    } finally {
      st.close();
    }
  }

  // Global shapes/dtypes: every compressed scale belongs to a packed U8
  // weight with NVFP4 geometry [N, K/2] -> scale [N, K/16], wherever the
  // source sharding placed each tensor.
  for (const [scaleName, entry] of scales) {
    const weightName = scaleName.replace(/\.weight_scale$/, ".weight");
    const weight = globalMeta.get(weightName);
    if (!weight || weight.dtype !== "U8" || weight.shape.length !== 2) {
      throw new CsfContractError(`CSF scale ${scaleName} has no packed U8 ${weightName} to scale`);
    }
    const geometry = nvfp4ScaleGeometry(weight.shape);
    if (geometry.rows !== entry.shape[0] || geometry.columns !== entry.shape[1]) {
      throw new CsfContractError(`CSF scale ${scaleName} shape [${entry.shape}] does not match ${weightName} geometry`);
    }
  }

  return {
    contract: {
      root,
      schema: manifest.schema,
      codec: contractJson.codec,
      family,
      sourceModelId: requireString(contractJson.source_model_id, "source_model_id"),
      sourceRevision: requireString(contractJson.source_revision, "source_revision"),
      sourceModelPath: requireString(contractJson.source, "source"),
      sourceNames,
      shards,
      scales,
      metadataSha256: contractMeta,
      compressedScaleCount,
    },
    layouts,
    order,
  };
}

/**
 * Logical reads and compressed-plane access over a validated CSF container.
 * GPU loaders consume the planes; CPU reads support integrity verification.
 */
export class CsfContainer {
  readonly contract: CsfContract;
  readonly layouts: Map<string, CsfShardLayout>;
  readonly order: string[];
  private handles = new Map<string, SafeTensorFile>();
  private closed = false;

  private constructor(opened: CsfOpenResult) {
    this.contract = opened.contract;
    this.layouts = opened.layouts;
    this.order = opened.order;
  }

  static open(root: string): CsfContainer {
    return new CsfContainer(readCsfContract(root));
  }

  private handle(file: string): SafeTensorFile {
    if (this.closed) {
      throw new CsfContractError("CSF container is closed");
    }
    let st = this.handles.get(file);
    if (!st) {
      st = SafeTensorFile.open(path.join(this.contract.root, "tensors", file));
      this.handles.set(file, st);
    }
    return st;
  }

  locate(name: string): { file: string; kind: "native" | "compressed"; meta: TensorMeta } {
    for (const [file, layout] of this.layouts) {
      const native = layout.native.get(name);
      if (native) return { file, kind: "native", meta: native };
      const compressed = layout.compressed.get(name);
      if (compressed) {
        return {
          file,
          kind: "compressed",
          meta: { dtype: compressed.record.dtype, shape: [...compressed.record.shape], dataOffsets: [0, compressed.record.sourceBytes] },
        };
      }
    }
    throw new CsfContractError(`CSF container has no logical tensor ${name}`);
  }

  /** Physical file range of a byte-identical native tensor; undefined for compressed scales. */
  physicalRange(name: string): { file: string; offset: number; size: number } | undefined {
    const located = this.locate(name);
    if (located.kind !== "native") return undefined;
    const st = this.handle(located.file);
    return {
      file: located.file,
      offset: st.tensorOffset(name),
      size: st.tensorSize(name),
    };
  }

  readTensor(name: string): Buffer {
    const located = this.locate(name);
    const st = this.handle(located.file);
    if (located.kind === "native") {
      return st.readTensor(name);
    }
    const entry = this.layouts.get(located.file)!.compressed.get(name)!;
    const [rows, columns] = entry.record.shape;
    return expandScalePlanes(
      st.readTensor(entry.fixedName),
      st.readTensor(entry.exceptionName),
      rows,
      columns,
    );
  }

  compressedPlanes(name: string) {
    const located = this.locate(name);
    if (located.kind !== "compressed") return undefined;
    const entry = this.layouts.get(located.file)!.compressed.get(name)!;
    const st = this.handle(located.file);
    const [rows, columns] = entry.record.shape;
    return {
      rows, columns,
      file: located.file,
      fixedOffset: st.tensorOffset(entry.fixedName),
      exceptionOffset: st.tensorOffset(entry.exceptionName),
      fixedBytes: st.tensorSize(entry.fixedName),
      exceptionBytes: st.tensorSize(entry.exceptionName),
      read: () => ({ fixed: st.readTensor(entry.fixedName), exceptions: st.readTensor(entry.exceptionName) }),
    };
  }

  /**
   * Bounded logical slice read: an extent over the leading two dimensions
   * with inner dimensions full (row-major). Scales decode only the window.
   */
  readTensorSlice(name: string, extent: { shape: number[]; offsets: number[] }): Buffer {
    const located = this.locate(name);
    const rank = located.meta.shape.length;
    if (extent.shape.length !== rank || extent.offsets.length !== rank) {
      throw new CsfContractError(`CSF slice rank mismatch for ${name}`);
    }
    for (let d = 0; d < rank; d++) {
      if (extent.offsets[d] < 0 || extent.shape[d] < 0
        || extent.offsets[d] + extent.shape[d] > located.meta.shape[d]) {
        throw new CsfContractError(`CSF slice for ${name} is out of bounds`);
      }
      if (d >= 2 && (extent.offsets[d] !== 0 || extent.shape[d] !== located.meta.shape[d])) {
        throw new CsfContractError(`CSF slices may only window the leading two dimensions of ${name}`);
      }
    }
    const st = this.handle(located.file);
    if (located.kind === "compressed") {
      const entry = this.layouts.get(located.file)!.compressed.get(name)!;
      const [rows, columns] = entry.record.shape;
      return expandScaleWindow(
        st.readTensor(entry.fixedName),
        st.readTensor(entry.exceptionName),
        rows, columns,
        extent.offsets[0], extent.shape[0],
        extent.offsets[1], extent.shape[1],
      );
    }
    // Native: copy row blocks of the logical row-major layout. Weights are
    // [rows, cols...] with byte rows; only leading dims are windowed.
    const dtypeBytes = SafeTensorFile.dtypeBytes(located.meta.dtype);
    const rowElements = located.meta.shape.slice(1).reduce((a, b) => a * b, 1);
    const rowBytes = rowElements * dtypeBytes;
    const innerRows = rank >= 2 ? extent.shape[1] * (located.meta.shape.slice(2).reduce((a, b) => a * b, 1)) : 1;
    const innerRowBytes = innerRows * dtypeBytes;
    const rowsToRead = extent.shape[0];
    const source = st.readTensor(name);
    const out = Buffer.allocUnsafe(extent.shape.reduce((a, b) => a * b, 1) * dtypeBytes);
    const srcBase = rank >= 2
      ? extent.offsets[0] * rowBytes + extent.offsets[1] * (located.meta.shape.slice(2).reduce((a, b) => a * b, 1)) * dtypeBytes
      : extent.offsets[0] * dtypeBytes;
    for (let r = 0; r < rowsToRead; r++) {
      const src = srcBase + r * rowBytes;
      source.copy(out, r * innerRowBytes, src, src + innerRowBytes);
    }
    return out;
  }

  close(): void {
    this.closed = true;
    for (const st of this.handles.values()) st.close();
    this.handles.clear();
  }
}

export interface CsfIntegrityReport {
  ok: boolean;
  problems: string[];
  checkedShards: string[];
  reconstructedShards: string[];
  /** Recorded verification.json claim, if present; never accepted as verification. */
  recordedVerification?: { passed: unknown; conditions?: unknown };
}

async function sha256File(filePath: string): Promise<{ sha256: string; bytes: number }> {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash("sha256");
    let bytes = 0;
    const stream = fs.createReadStream(filePath);
    stream.on("data", (chunk: Buffer) => {
      hash.update(chunk);
      bytes += chunk.length;
    });
    stream.on("end", () => resolve({ sha256: hash.digest("hex"), bytes }));
    stream.on("error", reject);
  });
}

/**
 * Streamed exact source-shard reconstruction: the preserved source header
 * followed by every original tensor's bytes (native tensors read from the
 * target shard, replaced scales re-expanded from their planes). Hashing the
 * stream without materializing the shard validates lossless scale expansion
 * against the recorded source hash.
 */
export async function reconstructSourceShard(
  root: string, contract: CsfContract, file: string,
): Promise<{ sha256: string; bytes: number; matchesRecorded: boolean }> {
  const receipt = asRecord(readJson(root, `receipts/${file}.json`), `receipt for ${file}`);
  const sourceHeader = parseSourceHeader(
    requireString(receipt.source_header_base64, `receipt ${file} source_header_base64`),
    file,
  );
  const container = CsfContainer.open(root);
  try {
    const hash = crypto.createHash("sha256");
    hash.update(sourceHeader.prefix);
    let bytes = sourceHeader.prefix.length;
    for (const name of sourceHeader.order) {
      const data = container.readTensor(name);
      hash.update(data);
      bytes += data.length;
    }
    const sha256 = hash.digest("hex");
    const record = contract.shards.get(file)!;
    return {
      sha256,
      bytes,
      matchesRecorded: sha256 === record.sourceSha256 && bytes === record.sourceFileBytes,
    };
  } finally {
    container.close();
  }
}

/**
 * INTEGRITY/PROVENANCE pass, separate from the contract checks: recompute
 * target hashes of the opened files, reconcile manifest/receipts/metadata
 * hashes, and optionally stream exact source reconstruction. The recorded
 * verification.json `passed` flag is reported but never trusted.
 */
export async function verifyCsfIntegrity(
  root: string,
  options: { shards?: string[]; metadata?: boolean; reconstructShards?: string[] } = {},
): Promise<CsfIntegrityReport> {
  root = path.resolve(root);
  const opened = readCsfContract(root);
  const { contract } = opened;
  const problems: string[] = [];
  const checkedShards: string[] = [];
  const reconstructedShards: string[] = [];
  try {
    const verificationPath = path.join(root, "verification.json");
    let recordedVerification: CsfIntegrityReport["recordedVerification"];
    if (fs.existsSync(verificationPath)) {
      const verification = asRecord(JSON.parse(fs.readFileSync(verificationPath, "utf-8")), "verification");
      recordedVerification = { passed: verification.passed, conditions: verification.conditions };
      // A recorded passed:true is not a check of the files being opened now.
    }

    if (options.metadata !== false) {
      for (const [name, expected] of Object.entries(contract.metadataSha256)) {
        const filePath = path.join(root, "metadata", name);
        if (!fs.existsSync(filePath)) {
          problems.push(`metadata file ${name} is missing`);
          continue;
        }
        const { sha256 } = await sha256File(filePath);
        if (sha256 !== expected) {
          problems.push(`metadata file ${name} hash ${sha256} does not match recorded ${expected}`);
        }
      }
    }

    const shardFiles = options.shards ?? [...contract.shards.keys()];
    for (const file of shardFiles) {
      const record = contract.shards.get(file);
      if (!record) {
        problems.push(`unknown shard ${file}`);
        continue;
      }
      // Reconcile the receipt's recorded claims against the manifest.
      const receiptPath = path.join(root, "receipts", `${file}.json`);
      if (!fs.existsSync(receiptPath)) {
        problems.push(`receipt for ${file} is missing`);
      } else {
        const receipt = asRecord(JSON.parse(fs.readFileSync(receiptPath, "utf-8")), `receipt for ${file}`);
        for (const [key, expected] of [
          ["source_file_bytes", record.sourceFileBytes],
          ["source_sha256", record.sourceSha256],
          ["target_file_bytes", record.targetFileBytes],
          ["target_sha256", record.targetSha256],
        ] as const) {
          if (receipt[key] !== expected) {
            problems.push(`receipt for ${file} disagrees with the manifest on ${key}`);
          }
        }
      }
      const filePath = path.join(root, "tensors", file);
      const { sha256, bytes } = await sha256File(filePath);
      checkedShards.push(file);
      if (bytes !== record.targetFileBytes) {
        problems.push(`shard ${file} is ${bytes} bytes, manifest expects ${record.targetFileBytes}`);
      }
      if (sha256 !== record.targetSha256) {
        problems.push(`shard ${file} target hash ${sha256} does not match recorded ${record.targetSha256}`);
      }
    }

    for (const file of options.reconstructShards ?? []) {
      const record = contract.shards.get(file);
      if (!record) {
        problems.push(`unknown shard ${file}`);
        continue;
      }
      const result = await reconstructSourceShard(root, contract, file);
      reconstructedShards.push(file);
      if (!result.matchesRecorded) {
        problems.push(`shard ${file} source reconstruction (${result.sha256}, ${result.bytes} bytes) does not match the recorded source`);
      }
    }

    return {
      ok: problems.length === 0,
      problems,
      checkedShards,
      reconstructedShards,
      recordedVerification,
    };
  } finally {
    // readCsfContract opens no persistent handles; nothing else to release.
  }
}
