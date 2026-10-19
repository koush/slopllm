import { CsfContractError } from "./csf_codec";

export const MXFP4_CSF_SCHEMA = "lil-mxfp4-csf-checkpoint/1";
export const MXFP4_CSF_CODEC = "row-base-offset1-u24-exceptions/1";

/** Sixteen row bases followed by little-endian one-bit offsets for each row. */
export function mxfp4FixedStreamLength(rows: number, columns: number): number {
  if (!Number.isSafeInteger(rows) || !Number.isSafeInteger(columns) || rows <= 0 || columns <= 0
    || rows % 16 || rows * columns > 0x1000000) {
    throw new CsfContractError(`Invalid MXFP4-CSF geometry [${rows}, ${columns}]`);
  }
  return rows * (1 + Math.ceil(columns / 8));
}

export function expandMxfp4ScaleWindow(
  fixed: Buffer, exceptions: Buffer, rows: number, columns: number,
  row0: number, rowLen: number, col0: number, colLen: number,
): Buffer {
  if (fixed.length !== mxfp4FixedStreamLength(rows, columns) || exceptions.length % 4) {
    throw new CsfContractError("MXFP4-CSF plane size does not match its geometry");
  }
  if (![row0, rowLen, col0, colLen].every(Number.isSafeInteger) || row0 < 0 || col0 < 0
    || rowLen < 0 || colLen < 0 || row0 + rowLen > rows || col0 + colLen > columns) {
    throw new CsfContractError("MXFP4-CSF scale window is out of bounds");
  }
  const selectors = Math.ceil(columns / 8);
  const stride = 16 * (1 + selectors);
  const out = Buffer.allocUnsafe(rowLen * colLen);
  for (let r = 0; r < rowLen; r++) {
    const row = row0 + r;
    const slab = Math.floor(row / 16) * stride;
    const base = fixed[slab + row % 16];
    const bits = slab + 16 + (row % 16) * selectors;
    for (let c = 0; c < colLen; c++) {
      const column = col0 + c;
      // The container encodes bytes, including wraparound; it is not a requantizer.
      out[r * colLen + c] = (base + ((fixed[bits + (column >> 3)] >> (column & 7)) & 1)) & 255;
    }
  }
  let previous = -1;
  for (let offset = 0; offset < exceptions.length; offset += 4) {
    const word = exceptions.readUInt32LE(offset);
    const position = word & 0xffffff;
    if (position <= previous || position >= rows * columns) {
      throw new CsfContractError("MXFP4-CSF exception positions must be sorted, unique, and in range");
    }
    previous = position;
    const row = Math.floor(position / columns) - row0;
    const column = position % columns - col0;
    if (row >= 0 && row < rowLen && column >= 0 && column < colLen) {
      out[row * colLen + column] = word >>> 24;
    }
  }
  return out;
}

export function expandMxfp4ScalePlanes(fixed: Buffer, exceptions: Buffer, rows: number, columns: number): Buffer {
  return expandMxfp4ScaleWindow(fixed, exceptions, rows, columns, 0, rows, 0, columns);
}
