/** Independent expected-value calculations for production-kernel tests. */
export function engramHashes(constants: { tokenMap: Int32Array; primes: number[][]; offsets: number[][] },
  tokens: readonly number[], lookback: readonly number[] = []): [Int32Array, Int32Array] {
  // Published NumPy-seeded multipliers; BigInt arithmetic is independent of CUDA.
  const multipliers = [[76632096046245n, 4839876093313n, 35959672319349n, 73987337458391n],
    [67716810739261n, 51510806800915n, 30921347202721n, 82619226485591n]];
  const output: [Int32Array, Int32Array] = [new Int32Array(tokens.length * 24), new Int32Array(tokens.length * 24)];
  for (let t = 0; t < tokens.length; t++) for (let layer = 0; layer < 2; layer++) {
    let rolling = 0n;
    for (let shift = 0; shift < 4; shift++) {
      const index = t - shift, token = index >= 0 ? tokens[index] : lookback[lookback.length + index] ?? 2;
      rolling ^= BigInt(constants.tokenMap[token]) * multipliers[layer][shift];
      if (shift) for (let head = 0; head < 8; head++) {
        const column = (shift - 1) * 8 + head;
        output[layer][t * 24 + column] = Number(rolling % BigInt(constants.primes[layer][column])) + constants.offsets[layer][column];
      }
    }
  }
  return output;
}

export function bf16Round(values: Float32Array): Buffer {
  const bits = new Uint32Array(values.buffer, values.byteOffset, values.length);
  const result = Buffer.alloc(values.length * 2);
  for (let i = 0; i < bits.length; i++) {
    const b = bits[i];
    result.writeUInt16LE(((b + 0x7fff + ((b >>> 16) & 1)) >>> 16) & 0xffff, i * 2);
  }
  return result;
}

export function dsv41Rope(values: Float32Array, position: number, compressed: boolean, inverse = false, headDim = 512): Float32Array {
  const result = values.slice();
  const theta = compressed ? 160000 : 10000;
  const correction = (rotations: number) => 64 * Math.log(65536 / (rotations * 2 * Math.PI)) / (2 * Math.log(theta));
  const low = Math.max(0, Math.floor(correction(32)));
  const high = Math.min(63, Math.ceil(correction(1)));
  for (let head = 0; head < values.length / headDim; head++) {
    for (let pair = 0; pair < 32; pair++) {
      const ramp = Math.min(1, Math.max(0, (pair - low) / (high === low ? 0.001 : high - low)));
      const freq = Math.fround(1 / theta ** (pair / 32) * (compressed ? 1 - ramp + ramp / 16 : 1));
      const angle = Math.fround(position * freq);
      const c = Math.fround(Math.cos(angle)), s = Math.fround(Math.sin(angle)) * (inverse ? -1 : 1);
      const i = head * headDim + headDim - 64 + pair * 2;
      result[i] = values[i] * c - values[i + 1] * s;
      result[i + 1] = values[i] * s + values[i + 1] * c;
    }
  }
  return result;
}

const e4m3 = Float32Array.from({ length: 127 }, (_, bits) => {
  const exponent = bits >>> 3, mantissa = bits & 7;
  return exponent ? (1 + mantissa / 8) * 2 ** (exponent - 7) : mantissa / 512;
});

export function encodeE4M3(value: number): number {
  const sign = value < 0 ? 128 : 0;
  const v = Math.min(448, Math.abs(value));
  let lo = 0, hi = 126;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (e4m3[mid] < v) lo = mid + 1; else hi = mid;
  }
  if (lo > 0) {
    const below = v - e4m3[lo - 1], above = e4m3[lo] - v;
    if (below < above || (below === above && (lo & 1))) lo--;
  }
  return sign | lo;
}

/** FP8 indexer reference: one power-of-two scale per 128-wide head/key. */
export function dsv41IndexQuantize(values: Float32Array): Float32Array {
  const result = new Float32Array(values.length);
  for (let offset = 0; offset < values.length; offset += 128) {
    let max = 1e-4;
    for (let d = 0; d < 128; d++) max = Math.max(max, Math.abs(values[offset + d]));
    const scale = 2 ** Math.ceil(Math.log2(max / 448));
    for (let d = 0; d < 128; d++) {
      const bits = encodeE4M3(values[offset + d] / scale);
      result[offset + d] = e4m3[bits & 127] * (bits & 128 ? -scale : scale);
    }
  }
  return result;
}

export function dsv41IndexScores(q: Float32Array, keys: readonly Float32Array[], weights: Float32Array): Float32Array {
  const scores = new Float32Array(keys.length);
  for (let token = 0; token < keys.length; token++) {
    let score = 0;
    for (let h = 0; h < 32; h++) {
      let dot = 0;
      for (let d = 0; d < 128; d++) dot += q[h * 128 + d] * keys[token][d];
      score += Math.max(0, dot) * weights[h] / 64;
    }
    scores[token] = score;
  }
  return scores;
}

/** DSV4 block-scaled cache: 576-byte records followed by eight-byte scale footers. */
export function writeDsv4Cache(page: Buffer, slot: number, values: Float32Array): void {
  if (values.length !== 512 || slot < 0 || (Math.floor(slot / 64) + 1) * 64 * 584 > page.length) {
    throw new Error("Invalid DSV4 cache row/slot");
  }
  const base = Math.floor(slot / 64) * 64 * 584;
  const data = base + slot % 64 * 576, scales = base + 64 * 576 + slot % 64 * 8;
  for (let block = 0; block < 7; block++) {
    let max = 1e-4;
    for (let d = 0; d < 64; d++) max = Math.max(max, Math.abs(values[block * 64 + d]));
    const exponent = Math.ceil(Math.log2(max / 448));
    page[scales + block] = exponent + 127;
    for (let d = 0; d < 64; d++) page[data + block * 64 + d] = encodeE4M3(values[block * 64 + d] * 2 ** -exponent);
  }
  page[scales + 7] = 0;
  bf16Round(values.subarray(448)).copy(page, data + 448);
}

export function dsv41Route(logits: Float32Array, bias: Float32Array): { expert: number; weight: number }[] {
  if (logits.length !== 384 || bias.length !== 384) throw new Error("Expected 384 router logits/biases");
  const scores = Float32Array.from(logits, x => Math.sqrt(Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)))));
  const indices = Array.from({ length: 384 }, (_, i) => i);
  indices.sort((a, b) => (scores[b] + bias[b]) - (scores[a] + bias[a]) || a - b);
  const chosen = indices.slice(0, 6);
  const sum = chosen.reduce((s, i) => s + scores[i], 0);
  return chosen.map(expert => ({ expert, weight: 1.5 * scores[expert] / (sum > 0 ? sum : 1) }));
}

export function dsv41SwiGlu(gate: Float32Array, up: Float32Array): Float32Array {
  return Float32Array.from(gate, (g, i) => {
    g = Math.min(g, 10);
    return g / (1 + Math.exp(-g)) * Math.min(10, Math.max(-10, up[i]));
  });
}
