import fs from "node:fs";

export interface TensorMeta {
  dtype: string;
  shape: number[];
  dataOffsets: [number, number];
}

const DTYPE_BYTES: Record<string, number> = {
  BOOL: 1,
  U8: 1,
  I8: 1,
  F8_E5M2: 1,
  F8_E4M3: 1,
  I16: 2,
  U16: 2,
  F16: 2,
  BF16: 2,
  I32: 4,
  U32: 4,
  F32: 4,
  F64: 8,
  I64: 8,
  U64: 8,
  C64: 8,
};

const HEADER_SIZE = 8;

export class SafeTensorFile {
  private fd: number;
  private header: Record<string, TensorMeta>;
  private metadata: Record<string, string>;
  private _dataStart: number;
  private _tensorNames: string[];

  private constructor(
    fd: number,
    header: Record<string, TensorMeta>,
    metadata: Record<string, string>,
    dataStart: number,
    tensorNames: string[],
  ) {
    this.fd = fd;
    this.header = header;
    this.metadata = metadata;
    this._dataStart = dataStart;
    this._tensorNames = tensorNames;
  }

  static open(path: string): SafeTensorFile {
    const fd = fs.openSync(path, "r");

    const headerLenBuf = Buffer.alloc(HEADER_SIZE);
    fs.readSync(fd, headerLenBuf, 0, HEADER_SIZE, 0);
    const headerLen = headerLenBuf.readBigUInt64LE(0);

    if (headerLen > 100_000_000n) {
      fs.closeSync(fd);
      throw new Error(`Safetensors header too large: ${headerLen} bytes`);
    }

    const headerBuf = Buffer.alloc(Number(headerLen));
    fs.readSync(fd, headerBuf, 0, Number(headerLen), HEADER_SIZE);

    const firstByte = headerBuf[0];
    if (firstByte !== 0x7b) {
      fs.closeSync(fd);
      throw new Error(`Invalid safetensors header: expected '{' (0x7b), got 0x${firstByte.toString(16)}`);
    }

    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(headerBuf.toString("utf-8"));
    } catch (e) {
      fs.closeSync(fd);
      throw new Error(`Failed to parse safetensors header JSON: ${e}`);
    }

    const header: Record<string, TensorMeta> = {};
    const metadata: Record<string, string> = {};
    const tensorNames: string[] = [];
    let prevEnd = 0;

    for (const [key, value] of Object.entries(parsed)) {
      if (key === "__metadata__") {
        if (typeof value === "object" && value !== null) {
          Object.assign(metadata, value as Record<string, string>);
        }
        continue;
      }

      const t = value as Record<string, unknown>;
      if (typeof t.dtype !== "string" || !Array.isArray(t.shape) || !Array.isArray(t.data_offsets)) {
        throw new Error(`Invalid tensor metadata for '${key}': ${JSON.stringify(t)}`);
      }

      if (!(t.dtype in DTYPE_BYTES) && t.dtype !== "F4" && t.dtype !== "F6_E2M3" && t.dtype !== "F6_E3M2") {
        throw new Error(`Unknown dtype '${t.dtype}' for tensor '${key}'`);
      }

      const offsets = t.data_offsets as number[];
      if (offsets.length !== 2) {
        throw new Error(`Invalid data_offsets for '${key}': expected [start, end], got ${offsets}`);
      }
      if (offsets[0] !== prevEnd) {
        throw new Error(
          `Non-contiguous data_offsets for '${key}': expected start=${prevEnd}, got ${offsets[0]}`,
        );
      }
      prevEnd = offsets[1];

      header[key] = {
        dtype: t.dtype,
        shape: t.shape as number[],
        dataOffsets: [offsets[0], offsets[1]],
      };
      tensorNames.push(key);
    }

    return new SafeTensorFile(fd, header, metadata, HEADER_SIZE + Number(headerLen), tensorNames);
  }

  get dataStart(): number {
    return this._dataStart;
  }

  tensorNames(): string[] {
    return this._tensorNames;
  }

  meta(name: string): TensorMeta {
    const m = this.header[name];
    if (!m) throw new Error(`Tensor '${name}' not found`);
    return m;
  }

  tensorOffset(name: string): number {
    return this._dataStart + this.meta(name).dataOffsets[0];
  }

  tensorSize(name: string): number {
    const m = this.meta(name);
    return m.dataOffsets[1] - m.dataOffsets[0];
  }

  readTensor(name: string): Buffer {
    const size = this.tensorSize(name);
    const offset = this.tensorOffset(name);
    const buf = Buffer.alloc(size);
    fs.readSync(this.fd, buf, 0, size, offset);
    return buf;
  }

  static dtypeBytes(dtype: string): number {
    const b = DTYPE_BYTES[dtype];
    if (b !== undefined) return b;
    if (dtype === "F4") return 0.5;
    if (dtype === "F6_E2M3" || dtype === "F6_E3M2") return 0.75;
    throw new Error(`Unknown dtype: ${dtype}`);
  }

  close(): void {
    if (this.fd !== -1) {
      fs.closeSync(this.fd);
      this.fd = -1;
    }
  }
}
