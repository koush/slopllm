import path from "node:path";

// NVFP4 compressed-scale codec `byte-window4-fixed-stream-u24-exceptions/1`.
//
// A logical E4M3 block-scale matrix [rows, columns] (one scale byte per 16
// consecutive K channels of a packed NVFP4 weight [rows, 2*columns*8]) is
// stored as two physical planes:
//
//   fixed      U8  — the "fixed stream": per 16-row slab, 16 row-base bytes
//                     followed by 16 * (columns / 2) packed bytes, i.e. each
//                     row carries `columns / 2` bytes of 4-bit offsets, two
//                     scale columns per byte (even column = low nibble,
//                     odd column = high nibble).
//   exceptions U32 — replacement words: low 24 bits are the row-major flat
//                     position `row * columns + column` in the scale matrix,
//                     the top 8 bits are the replacement scale byte.
//
// Decoded value at (row, column):
//   base = fixed[row slab, row in slab]
//   value = exception word if a replacement covers the position,
//           else (base + nibble) & 0xFF
// with the contract requiring base <= 240 so base + nibble (0..15) never
// wraps past 255. Padded TP shards use the E4M3 1.0 code 0x38 as the row
// base of appended rows with zero offsets, and zero offset codes for
// appended scale columns.
//
// This module is pure CPU code: it deliberately holds no safetensors or
// filesystem knowledge so the codec can be fixture-tested in isolation.
// The physical planes stay behind the checkpoint-source abstraction in
// checkpoint_source.ts; consumers only ever see logical scale bytes.

export const NVFP4_CSF_SCHEMA = "lil-nvfp4-csf-checkpoint/1";
export const NVFP4_CSF_CODEC = "byte-window4-fixed-stream-u24-exceptions/1";
/** Exact supported schema/codec/family combinations — not the lil-*-csf wildcard. */
export const SUPPORTED_CSF_FAMILIES: readonly string[] = ["glm53_744b_nvfp4"];

/** E4M3 code of 1.0, the base used by rows appended to a padded TP shard. */
export const PADDED_SCALE_CODE = 0x38;
/** NVFP4 block scales are positive finite e4m3 magnitudes: codes 0x00..0x7e. */
export const MAX_SCALE_CODE = 0x7e;

export class CsfContractError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "CsfContractError";
  }
}

function fail(message: string): never {
  throw new CsfContractError(message);
}

export interface ScaleGeometry {
  /** Scale-matrix rows: the weight's output channels N. */
  rows: number;
  /** Scale-matrix columns: K / 16 for NVFP4 group size 16. */
  columns: number;
}

/** NVFP4 group size: one E4M3 scale per 16 packed weight values along K. */
export const NVFP4_GROUP_SIZE = 16;

/**
 * Expected logical scale geometry for a packed NVFP4 weight [N, K/2] U8.
 * The scale matrix is [N, K/16]; K/2 must be a multiple of 8 nibble pairs.
 */
export function nvfp4ScaleGeometry(weightShape: readonly number[]): ScaleGeometry {
  if (weightShape.length !== 2) {
    fail(`NVFP4 weight must be rank 2 [N, K/2], got shape [${weightShape}]`);
  }
  const [rows, packedK] = weightShape;
  if (!(rows > 0) || !(packedK > 0) || packedK % 8 !== 0) {
    fail(`NVFP4 weight shape [${weightShape}] is not a valid [N, K/2] nibble pack`);
  }
  return { rows, columns: packedK / 8 };
}

/** Physical fixed-stream byte length for a [rows, columns] scale matrix. */
export function fixedStreamLength(rows: number, columns: number): number {
  if (rows % 16 !== 0 || columns % 2 !== 0 || rows <= 0 || columns <= 0) {
    fail(`NVFP4-CSF scale geometry [${rows}, ${columns}] needs 16-row slabs and even columns`);
  }
  return (rows / 16) * 16 * (1 + columns / 2);
}

function requireGeometry(rows: number, columns: number): void {
  if (!Number.isSafeInteger(rows) || !Number.isSafeInteger(columns) || rows <= 0 || columns <= 0
    || rows % 16 !== 0 || columns % 2 !== 0) {
    fail(`NVFP4-CSF scale geometry [${rows}, ${columns}] needs 16-row slabs and even columns`);
  }
}

function decodeExceptions(exceptions: Buffer, rows: number, columns: number): {
  positions: Uint32Array;
  values: Uint8Array;
} {
  if (exceptions.length % 4 !== 0) {
    fail(`NVFP4-CSF exception plane must be whole uint32 words, got ${exceptions.length} bytes`);
  }
  const count = exceptions.length / 4;
  const positions = new Uint32Array(count);
  const values = new Uint8Array(count);
  let previous = -1;
  const total = rows * columns;
  for (let i = 0; i < count; i++) {
    const word = exceptions.readUInt32LE(i * 4);
    const position = word & 0xffffff;
    if (position >= total) {
      fail(`NVFP4-CSF exception position ${position} is outside the ${rows}x${columns} scale matrix`);
    }
    if (position <= previous) {
      fail("NVFP4-CSF exception positions must be sorted and unique");
    }
    previous = position;
    positions[i] = position;
    values[i] = word >>> 24;
  }
  return { positions, values };
}

/**
 * Contract-check a fixed plane without decoding values: stream size, row
 * bases within 0..240 (so base + nibble fits a byte without wrapping).
 */
export function validateFixedPlane(fixed: Buffer, rows: number, columns: number): void {
  requireGeometry(rows, columns);
  const expected = fixedStreamLength(rows, columns);
  if (fixed.length !== expected) {
    fail(`NVFP4-CSF fixed stream must be ${expected} bytes for a ${rows}x${columns} scale matrix, got ${fixed.length}`);
  }
  const stride = 16 * (1 + columns / 2);
  for (let slab = 0; slab < rows / 16; slab++) {
    for (let r = 0; r < 16; r++) {
      const base = fixed[slab * stride + r];
      if (base > 240) {
        fail(`NVFP4-CSF row base ${base} exceeds the 0..240 window`);
      }
    }
  }
}

function rejectInvalidScaleCodes(values: Uint8Array | Uint16Array, context: string): void {
  for (const value of values) {
    // NVFP4 block scales are positive finite e4m3 magnitudes: codes 0x00..0x7e.
    // 0x7f is the e4m3fn NaN code and anything at/above 0x80 is negative.
    if (value > 0x7e) {
      fail(`NVFP4-CSF ${context} contains invalid E4M3 scale code 0x${value.toString(16)}`);
    }
  }
}

/**
 * Lossless expansion of one projection's compressed planes into the source
 * row-major E4M3 scale bytes. Contract checks (sizes, bases, sorted unique
 * in-range exception positions, valid scale values) run here, fail-closed.
 */
export function expandScalePlanes(fixed: Buffer, exceptions: Buffer, rows: number, columns: number): Buffer {
  requireGeometry(rows, columns);
  validateFixedPlane(fixed, rows, columns);
  const { positions, values } = decodeExceptions(exceptions, rows, columns);
  const selectors = columns / 2;
  const stride = 16 * (1 + selectors);
  const out = Buffer.allocUnsafe(rows * columns);
  for (let slab = 0; slab < rows / 16; slab++) {
    const slabStart = slab * stride;
    for (let r = 0; r < 16; r++) {
      const row = slab * 16 + r;
      const base = fixed[slabStart + r];
      const packedStart = slabStart + 16 + r * selectors;
      const outStart = row * columns;
      for (let c = 0; c < selectors; c++) {
        const byte = fixed[packedStart + c];
        out[outStart + 2 * c] = base + (byte & 0x0f);
        out[outStart + 2 * c + 1] = base + (byte >> 4);
      }
    }
  }
  for (let i = 0; i < positions.length; i++) {
    out[positions[i]] = values[i];
  }
  rejectInvalidScaleCodes(out, "scale values");
  return out;
}

/**
 * Bounded expansion of a logical [rowLen, colLen] window at (row0, col0) of
 * the scale matrix. Plane invariants (sizes, bases, exception ordering and
 * range) are still validated across the whole plane; only the window is
 * decoded and returned as row-major E4M3 bytes.
 */
export function expandScaleWindow(
  fixed: Buffer, exceptions: Buffer, rows: number, columns: number,
  row0: number, rowLen: number, col0: number, colLen: number,
): Buffer {
  requireGeometry(rows, columns);
  validateFixedPlane(fixed, rows, columns);
  const { positions, values } = decodeExceptions(exceptions, rows, columns);
  if (!(Number.isSafeInteger(row0) && Number.isSafeInteger(rowLen) && Number.isSafeInteger(col0) && Number.isSafeInteger(colLen))
    || row0 < 0 || rowLen < 0 || col0 < 0 || colLen < 0
    || row0 + rowLen > rows || col0 + colLen > columns) {
    fail(`NVFP4-CSF window [${row0}+${rowLen}, ${col0}+${colLen}] is outside the ${rows}x${columns} scale matrix`);
  }
  const selectors = columns / 2;
  const stride = 16 * (1 + selectors);
  const out = Buffer.allocUnsafe(rowLen * colLen);
  for (let r = 0; r < rowLen; r++) {
    const row = row0 + r;
    const slabStart = Math.floor(row / 16) * stride;
    const base = fixed[slabStart + (row % 16)];
    const packedStart = slabStart + 16 + (row % 16) * selectors;
    const outStart = r * colLen;
    for (let c = 0; c < colLen; c++) {
      const col = col0 + c;
      const byte = fixed[packedStart + (col >> 1)];
      out[outStart + c] = base + ((col & 1) === 0 ? byte & 0x0f : byte >> 4);
    }
  }
  for (let i = 0; i < positions.length; i++) {
    const position = positions[i];
    const row = Math.floor(position / columns);
    const col = position % columns;
    if (row >= row0 && row < row0 + rowLen && col >= col0 && col < col0 + colLen) {
      out[(row - row0) * colLen + (col - col0)] = values[i];
    }
  }
  rejectInvalidScaleCodes(out, "scale values");
  return out;
}

export interface ScalePlanes {
  fixed: Buffer;
  exceptions: Buffer;
}

/**
 * Slice one projection's compressed planes to a TP shard and rebase its
 * exception positions into the output plane (port of the reference
 * `_slice_scale_plane`).
 *
 * `outRows`/`outColumns` extend the slice to a padded TP shard whose weights
 * are zero there: appended rows get the E4M3 scale 1.0 (base 0x38 with zero
 * offsets — a zero scale would be stored as a replacement word in every
 * packed word), appended columns get zero offset codes over their row's base.
 */
export function sliceScalePlanes(
  fixed: Buffer, exceptions: Buffer, rows: number, columns: number,
  rowSlice: readonly [number, number], columnSlice: readonly [number, number],
  outRows?: number, outColumns?: number,
): ScalePlanes {
  requireGeometry(rows, columns);
  validateFixedPlane(fixed, rows, columns);
  const { positions, values } = decodeExceptions(exceptions, rows, columns);
  const [r0, r1] = rowSlice;
  const [c0, c1] = columnSlice;
  if (!(0 <= r0 && r0 < r1 && r1 <= rows && r0 % 16 === 0 && r1 % 16 === 0)) {
    fail("NVFP4-CSF row slices require complete 16-row slabs");
  }
  if (!(0 <= c0 && c0 < c1 && c1 <= columns && c0 % 2 === 0 && c1 % 2 === 0)) {
    fail("NVFP4-CSF column slices require complete nibble pairs");
  }
  const sliceRows = r1 - r0;
  const sliceColumns = c1 - c0;
  const outR = outRows ?? sliceRows;
  const outC = outColumns ?? sliceColumns;
  if (!(outR >= sliceRows && outR % 16 === 0 && outC >= sliceColumns && outC % 2 === 0)) {
    fail("NVFP4-CSF padded slices must extend whole slabs and pairs");
  }

  const outFixed = Buffer.alloc(fixedStreamLength(outR, outC));
  const selectors = outC / 2;
  const inSelectors = columns / 2;
  const inStride = 16 * (1 + inSelectors);
  const outStride = 16 * (1 + selectors);
  for (let slab = 0; slab < outR / 16; slab++) {
    const outSlabStart = slab * outStride;
    const sourceSlab = Math.floor(r0 / 16) + slab;
    const realSlab = slab < sliceRows / 16;
    for (let r = 0; r < 16; r++) {
      outFixed[outSlabStart + r] = realSlab ? fixed[sourceSlab * inStride + r] : PADDED_SCALE_CODE;
    }
    if (!realSlab) continue;
    const inPackedStart = sourceSlab * inStride + 16 + (r0 % 16) * inSelectors;
    for (let r = 0; r < 16; r++) {
      const inRowStart = inPackedStart + r * inSelectors + c0 / 2;
      const outRowStart = outSlabStart + 16 + r * selectors;
      fixed.copy(outFixed, outRowStart, inRowStart, inRowStart + sliceColumns / 2);
    }
  }

  const selected: number[] = [];
  const selectedValues: number[] = [];
  for (let i = 0; i < positions.length; i++) {
    const row = Math.floor(positions[i] / columns);
    const col = positions[i] % columns;
    if (row >= r0 && row < r1 && col >= c0 && col < c1) {
      selected.push((row - r0) * outC + (col - c0));
      selectedValues.push(values[i]);
    }
  }
  const outExceptions = Buffer.allocUnsafe(selected.length * 4);
  for (let i = 0; i < selected.length; i++) {
    outExceptions.writeUInt32LE(((selectedValues[i] << 24) | selected[i]) >>> 0, i * 4);
  }
  return { fixed: outFixed, exceptions: outExceptions };
}

/**
 * Fuse two 16-row-slab-aligned FC1 projections (up then gate, the reference
 * W13 order) into one plane pair over [first rows + second rows, columns].
 * The second projection's exception positions are offset by `localRows *
 * columns`, matching the reference `prepare_nvfp4_csf_weights` concat.
 */
export function concatFc1ScalePlanes(
  first: ScalePlanes, second: ScalePlanes, localRows: number, columns: number,
): ScalePlanes {
  requireGeometry(localRows, columns);
  if (first.fixed.length !== fixedStreamLength(localRows, columns)
    || second.fixed.length !== fixedStreamLength(localRows, columns)) {
    fail("NVFP4-CSF FC1 fusion requires equal fixed-stream geometry for both projections");
  }
  const fixed = Buffer.concat([first.fixed, second.fixed]);
  const offset = localRows * columns;
  const shiftWords = (plane: ScalePlanes, shift: number): Buffer => {
    if (plane.exceptions.length % 4 !== 0) {
      fail("NVFP4-CSF exception plane must be whole uint32 words");
    }
    const out = Buffer.from(plane.exceptions);
    for (let i = 0; i < out.length; i += 4) {
      const word = out.readUInt32LE(i);
      const position = (word & 0xffffff) + shift;
      if (position >= 2 * offset) {
        fail(`NVFP4-CSF fused exception position ${position} is outside 2x${localRows}x${columns}`);
      }
      out.writeUInt32LE((((word >>> 24) << 24) | position) >>> 0, i);
    }
    return out;
  };
  const exceptions = Buffer.concat([shiftWords(first, 0), shiftWords(second, offset)]);
  // Fusing along rows keeps each half's positions ordered; verify the whole.
  decodeExceptions(exceptions, 2 * localRows, columns);
  return { fixed, exceptions };
}

/**
 * Validate an equal TP partition and return its intermediate-axis bounds
 * (port of the reference `tp_extent`).
 */
export function tpExtent(
  intermediateSize: number, tpRank: number, tpSize: number, alignment: number,
): [number, number] {
  if (
    tpSize <= 0 || !(0 <= tpRank && tpRank < tpSize)
    || intermediateSize <= 0
    || intermediateSize % (alignment * tpSize) !== 0
  ) {
    fail(`CSF requires a valid TP rank and extents aligned to ${alignment} intermediate channels`);
  }
  const local = intermediateSize / tpSize;
  return [tpRank * local, (tpRank + 1) * local];
}

/** Shard-file name sanity: checkpoint-local safetensors basenames only. */
export function requireLocalSafetensorsName(file: string): void {
  if (path.basename(file) !== file || !file.endsWith(".safetensors")) {
    fail("CSF shard names must be checkpoint-local safetensors files");
  }
}