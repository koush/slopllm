import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

// Fixture builder for CPU CSF container tests (port plan §9.7.1). It writes
// a synthetic but grammar-faithful `lil-nvfp4-csf-checkpoint/1` container
// around small NVFP4 matrices with known bytes, mirroring the reference
// `csf_fixtures.matrix()` generator: row bases, 4-bit window offsets, and
// u24 exception words. The exact source shard bytes are retained in memory
// so tests can assert lossless reconstruction.

export interface FixtureTensor {
  name: string;
  dtype: string;
  shape: number[];
  data: Buffer;
}

export interface ScaleMatrixFixture {
  logical: Buffer;
  fixed: Buffer;
  exceptions: Buffer;
  bases: number[];
  rows: number;
  columns: number;
}

export function sha256(data: Buffer): string {
  return crypto.createHash("sha256").update(data).digest("hex");
}

/** Write a minimal safetensors file; returns the exact file bytes. */
export function writeSafetensorsFile(filePath: string, tensors: FixtureTensor[]): Buffer {
  const header: Record<string, unknown> = {};
  let offset = 0;
  const dataParts: Buffer[] = [];
  for (const tensor of tensors) {
    header[tensor.name] = {
      dtype: tensor.dtype,
      shape: tensor.shape,
      data_offsets: [offset, offset + tensor.data.length],
    };
    offset += tensor.data.length;
    dataParts.push(tensor.data);
  }
  const headerJson = Buffer.from(JSON.stringify(header), "utf-8");
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(headerJson.length));
  const fileBytes = Buffer.concat([prefix, headerJson, ...dataParts]);
  if (filePath) fs.writeFileSync(filePath, fileBytes);
  return fileBytes;
}

export interface ScaleMatrixOptions {
  seed: number;
  /** Flat scale-matrix positions replaced by exception words (sorted). */
  exceptionPositions: number[];
  exceptionValues: number[];
  /** Base override per row (defaults to a deterministic 20..209 window). */
  bases?: number[];
}

/**
 * Build one compressed scale matrix with known bytes: logical[row, col] =
 * base[row] + nibble(row, col) except at the exception positions.
 */
export function makeScaleMatrix(rows: number, columns: number, options: ScaleMatrixOptions): ScaleMatrixFixture {
  const { seed, exceptionPositions, exceptionValues } = options;
  const bases = options.bases ?? Array.from({ length: rows }, (_, r) => ((r * 7 + seed) % 90) + 20);
  const offsets = new Uint8Array(rows * columns);
  for (let i = 0; i < rows * columns; i++) {
    offsets[i] = (i + seed) % 16;
  }
  const logical = Buffer.alloc(rows * columns);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < columns; c++) {
      logical[r * columns + c] = bases[r] + offsets[r * columns + c];
    }
  }
  const packed = Buffer.alloc(rows * (columns / 2));
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < columns / 2; c++) {
      packed[r * (columns / 2) + c] = offsets[r * columns + 2 * c] | (offsets[r * columns + 2 * c + 1] << 4);
    }
  }
  const fixed = Buffer.alloc((rows / 16) * 16 * (1 + columns / 2));
  const stride = 16 * (1 + columns / 2);
  for (let slab = 0; slab < rows / 16; slab++) {
    for (let r = 0; r < 16; r++) {
      fixed[slab * stride + r] = bases[slab * 16 + r];
    }
    packed.copy(fixed, slab * stride + 16, slab * 16 * (columns / 2), (slab + 1) * 16 * (columns / 2));
  }
  const exceptions = Buffer.alloc(exceptionPositions.length * 4);
  let previous = -1;
  for (let i = 0; i < exceptionPositions.length; i++) {
    const position = exceptionPositions[i];
    if (position <= previous) throw new Error("fixture exception positions must be sorted and unique");
    previous = position;
    logical[position] = exceptionValues[i];
    exceptions.writeUInt32LE((((exceptionValues[i] & 0xff) << 24) | position) >>> 0, i * 4);
  }
  return { logical, fixed, exceptions, bases, rows, columns };
}

export interface CsfFixtureShard {
  file: string;
  tensors: FixtureTensor[];
  /** Logical names replaced by compressed planes. */
  compressedScales: string[];
}

export interface CsfFixture {
  root: string;
  /** Exact source shard file bytes, keyed by shard file name. */
  sourceShards: Map<string, Buffer>;
  /** Expected logical tensors (name -> bytes), in source order. */
  logical: Map<string, FixtureTensor>;
}

function f32Scalar(value: number): Buffer {
  const buf = Buffer.alloc(4);
  buf.writeFloatLE(value);
  return buf;
}

/**
 * Build a complete synthetic CSF container in `root`. The default payload
 * mirrors the GLM-5.3 layout in miniature: an NVFP4 expert triple (U8
 * nibble weights + F8_E4M3 scales + F32 weight_scale_2 + F32 input_scale),
 * asymmetric gate/up scale and calibration values, one BF16 shared-expert
 * weight, and one plain F32 norm.
 */
export function buildCsfFixture(root: string): CsfFixture {
  fs.mkdirSync(path.join(root, "tensors"), { recursive: true });
  fs.mkdirSync(path.join(root, "receipts"), { recursive: true });
  fs.mkdirSync(path.join(root, "metadata"), { recursive: true });

  const tensors: FixtureTensor[] = [];
  const compressedScales: string[] = [];
  const pfx = "model.layers.3.mlp.experts.0";

  const projection = (proj: string, rows: number, packedK: number, columns: number,
    seed: number, exceptionPositions: number[], exceptionValues: number[], inputScale: number) => {
    const weight = Buffer.alloc(rows * packedK);
    for (let i = 0; i < weight.length; i++) weight[i] = (i * 31 + seed) & 0xff;
    const matrix = makeScaleMatrix(rows, columns, { seed, exceptionPositions, exceptionValues });
    tensors.push(
      { name: `${pfx}.${proj}.weight`, dtype: "U8", shape: [rows, packedK], data: weight },
      { name: `${pfx}.${proj}.weight_scale`, dtype: "F8_E4M3", shape: [rows, columns], data: matrix.logical },
      { name: `${pfx}.${proj}.weight_scale_2`, dtype: "F32", shape: [], data: f32Scalar(0.5 + seed / 1000) },
      { name: `${pfx}.${proj}.input_scale`, dtype: "F32", shape: [], data: f32Scalar(inputScale) },
    );
    compressedScales.push(`${pfx}.${proj}.weight_scale`);
    return matrix;
  };

  // gate/up: N=32, K=128 (weight [32, 64] U8), scales [32, 8]. Asymmetric
  // values and exception sets, with exceptions exactly at slice boundaries.
  const gateMatrix = projection("gate_proj", 32, 64, 8, 5,
    [0, 7, 8, 31, 32, 39, 40, 63, 127, 128, 255],
    [0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x2a],
    0.03125);
  const upMatrix = projection("up_proj", 32, 64, 8, 11,
    [15, 16, 24, 25, 240, 255],
    [0x20, 0x21, 0x22, 0x23, 0x24, 0x25],
    0.0625);
  // down: weight [16, 256] U8 (K=512), scales [16, 32].
  const downMatrix = projection("down_proj", 16, 256, 32, 17,
    [31, 32, 511],
    [0x40, 0x41, 0x42],
    0.125);

  tensors.push(
    {
      name: "model.layers.3.mlp.shared_experts.gate_proj.weight",
      dtype: "BF16", shape: [8, 16],
      data: Buffer.from(Array.from({ length: 8 * 16 * 2 }, (_, i) => (i * 13) & 0xff)),
    },
    { name: "model.norm.weight", dtype: "F32", shape: [8], data: f32Scalar(1.5) },
  );

  const file = "model-00001-of-00001.safetensors";
  const sourceBytes = writeSafetensorsFile("", tensors);
  const sourceHeader = sourceBytes.subarray(0, 8 + Number(sourceBytes.readBigUInt64LE(0)));

  // Physical target tensors: natives byte-identical, replaced scales become planes.
  const targetTensors: FixtureTensor[] = [];
  const scaleRecords: Record<string, unknown> = {};
  const planeByScale = new Map<string, { fixed: Buffer; exceptions: Buffer }>();
  for (const scale of compressedScales) {
    const logicalTensor = tensors.find(t => t.name === scale)!;
    const [rows, columns] = logicalTensor.shape;
    const matrix = scale === `${pfx}.gate_proj.weight_scale` ? gateMatrix
      : scale === `${pfx}.up_proj.weight_scale` ? upMatrix : downMatrix;
    planeByScale.set(scale, { fixed: matrix.fixed, exceptions: matrix.exceptions });
    scaleRecords[scale] = {
      dtype: "F8_E4M3",
      shape: [rows, columns],
      source_bytes: rows * columns,
      source_sha256: sha256(matrix.logical),
      fixed_bytes: matrix.fixed.length,
      exception_bytes: matrix.exceptions.length,
    };
    targetTensors.push(
      { name: `${scale}.nvfp4_csf_fixed`, dtype: "U8", shape: [matrix.fixed.length], data: matrix.fixed },
      { name: `${scale}.nvfp4_csf_exceptions`, dtype: "U32", shape: [matrix.exceptions.length / 4], data: matrix.exceptions },
    );
  }
  for (const tensor of tensors) {
    if (compressedScales.includes(tensor.name)) continue;
    targetTensors.push(tensor);
  }
  const targetBytes = writeSafetensorsFile(path.join(root, "tensors", file), targetTensors);

  const weightMap: Record<string, string> = {};
  for (const tensor of tensors) weightMap[tensor.name] = file;
  const metadataFiles: Record<string, string> = {
    "config.json": JSON.stringify({ model_type: "glm53_fixture", hidden_size: 128 }),
    "generation_config.json": JSON.stringify({ temperature: 1.0 }),
    "tokenizer_config.json": JSON.stringify({ tokenizer_class: "FixtureTokenizer" }),
    "chat_template.jinja": "{{ messages }}",
    "model.safetensors.index.json": JSON.stringify({ weight_map: weightMap }),
  };
  const metadataSha256: Record<string, string> = {};
  for (const [name, content] of Object.entries(metadataFiles)) {
    const data = Buffer.from(content, "utf-8");
    fs.writeFileSync(path.join(root, "metadata", name), data);
    metadataSha256[name] = sha256(data);
  }

  fs.writeFileSync(path.join(root, "receipts", `${file}.json`), JSON.stringify({
    file,
    source_file_bytes: sourceBytes.length,
    source_sha256: sha256(sourceBytes),
    target_file_bytes: targetBytes.length,
    target_sha256: sha256(targetBytes),
    source_header_base64: sourceHeader.toString("base64"),
    scales: scaleRecords,
    source_content_address_verified: true,
  }));

  const shardRecord = {
    file,
    source_file_bytes: sourceBytes.length,
    source_sha256: sha256(sourceBytes),
    target_file_bytes: targetBytes.length,
    target_sha256: sha256(targetBytes),
  };
  const identity = {
    schema: "lil-nvfp4-csf-checkpoint/1",
    codec: "byte-window4-fixed-stream-u24-exceptions/1",
    family: "glm53_744b_nvfp4",
    metadata_sha256: metadataSha256,
  };
  fs.writeFileSync(path.join(root, "manifest.json"), JSON.stringify({
    ...identity,
    compressed_scale_count: compressedScales.length,
    compressed_scale_bytes: targetTensors
      .filter(t => t.name.endsWith(".nvfp4_csf_fixed") || t.name.endsWith(".nvfp4_csf_exceptions"))
      .reduce((sum, t) => sum + t.data.length, 0),
    original_scale_bytes: compressedScales.reduce((sum, name) => {
      const tensor = tensors.find(t => t.name === name)!;
      return sum + tensor.data.length;
    }, 0),
    source: "/fixture/source/snapshot",
    source_model_id: "local-inference-lab/GLM-5.3-NVFP4",
    source_revision: "b472e4ee53f6a9862da5486c56c6ca21be3dab70",
    source_index_sha256: sha256(Buffer.from(metadataFiles["model.safetensors.index.json"], "utf-8")),
    source_index_size_semantics: "safetensors_files",
    shards: [shardRecord],
    status: "fixture",
  }));
  fs.writeFileSync(path.join(root, "build-contract.json"), JSON.stringify({
    ...identity,
    compressed_scale_count: compressedScales.length,
    source: "/fixture/source/snapshot",
    source_model_id: "local-inference-lab/GLM-5.3-NVFP4",
    source_revision: "b472e4ee53f6a9862da5486c56c6ca21be3dab70",
    source_index_sha256: sha256(Buffer.from(metadataFiles["model.safetensors.index.json"], "utf-8")),
    source_index_size_semantics: "safetensors_files",
    source_names: weightMap,
  }));
  fs.writeFileSync(path.join(root, "lil-manifest.json"), JSON.stringify({
    schema: "lil-manifest/1",
    work: { work_name: "fixture" },
  }));
  fs.writeFileSync(path.join(root, "verification.json"), JSON.stringify({
    schema: "lil-nvfp4-csf-checkpoint/1",
    passed: true,
    conditions: "fixture container; recorded claims are not verification",
    qualification: "storage and CPU reconstruction",
    scale_matrices: compressedScales.length,
    shards: [file],
  }));

  const logical = new Map<string, FixtureTensor>();
  for (const tensor of tensors) logical.set(tensor.name, tensor);
  return {
    root,
    sourceShards: new Map([[file, sourceBytes]]),
    logical,
  };
}

export function readJsonFile(root: string, relative: string): Record<string, any> {
  return JSON.parse(fs.readFileSync(path.join(root, relative), "utf-8"));
}

export function writeJsonFile(root: string, relative: string, value: unknown): void {
  fs.writeFileSync(path.join(root, relative), JSON.stringify(value));
}