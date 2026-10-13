import { CsfContractError } from "../src/csf_codec";

// Pure-TypeScript reference implementation of the B12X packed NVFP4
// scale-boundary contract (design doc
// `docs/design_docs/b12x_nvfp4_packed_scale_contract.md` in the flashinfer
// reference tree; executable reference
// `flashinfer/experimental/b12x/moe/_shared/kernels/w4a16/prepare.py`
// `_nvfp4_compute_scale_factor` / `_process_nvfp4_packed_scales` /
// `_process_nvfp4_packed_global_scale`, and the boundary harness in
// `tests/experimental/b12x/moe/test_w4a16_packed_format.py`).
//
// Contract semantics, in preparation order:
//   1. normalize: one power-of-two factor `f` per projection across all
//      experts makes the largest normalized block scale no greater than 448;
//      FP16 preparation uses f=1.
//   2. flush: nonzero scales with s*f < 1/64 are discarded (the subnormal
//      values of the normalized E4M3 scale; raw E4M3 codes 1-7 when f=1).
//      Zero remains zero.
//   3. lift: retained scales are encoded as FP16 bits 14 through 7 of
//      s*f*128 and are represented exactly.
//   4. compensate: the per-expert global scale is multiplied by 1/f plus the
//      activation-dtype decode convention constant.
//
// This checkpoint uses E4M3 K/16 block scales: one e4m3 scale byte per 16 K
// channels. Nothing here is A4 losslessness: A4 (4-bit activation prefill) is
// not lossless and carries a characterized prefill regression. "Lossless" in
// this codebase refers only to CSF container-to-buffer scale round-trip.

export type PackedActivation = "fp16" | "bf16";

/** Largest normalized block scale the factor selection guarantees. */
export const NVFP4_NORMALIZED_SCALE_MAX = 448;
/** The lift encodes s*f*2^7 as FP16 bits 14 through 7. */
export const NVFP4_LIFT_EXPONENT = 7;
/** Flush boundary on the lifted value: v < 2, i.e. nonzero s*f < 1/64. */
export const NVFP4_FLUSH_LIFT_THRESHOLD = 2;
/** Discard boundary on the normalized scale: s*f < 1/64 is discarded. */
export const NVFP4_DISCARD_BOUNDARY = 1 / 64;
/** Smallest finite nonnegative E4M3 code (subnormal, flushed at f=1). */
export const E4M3_MIN_POSITIVE_CODE = 1;
/** Largest finite nonnegative E4M3 code (448); 0x7f is the e4m3fn NaN code. */
export const E4M3_MAX_FINITE_CODE = 0x7e;
/** E4M3 K/16 block size: one scale byte per 16 K channels. */
export const E4M3_K16_BLOCK_SIZE = 16;

/**
 * Reference-only E8M0 K/32 constants. These belong to the E8M0 K/32
 * implementation and must NOT be force-fitted onto E4M3 K/16 (see the
 * not-applicable tests): `_E8M0_K32_BF16_MAX_SCALE_BYTE` is a UE8M0-alphabet
 * BF16 serving clamp and `_E8M0_LOGICAL_TAIL_SCALE_N_ALIGNMENT` is the
 * E8M0 microkernel logical-tail scale-N alignment.
 */
export const E8M0_K32_BF16_MAX_SCALE_BYTE = 247;
export const E8M0_LOGICAL_TAIL_SCALE_N_ALIGNMENT = 64;

/** Golden codes from the design doc's checkpoint audit (dec48abd... revision). */
export const AUDITED_MIN_SCALE_CODE = 0x31; // 0.5625
export const AUDITED_MAX_SCALE_CODE = 0x7e; // 448

/** Decode a finite nonnegative E4M3 scale code exactly. */
export function e4m3Decode(code: number): number {
  if (!Number.isInteger(code) || code < 0 || code > 0xff) {
    throw new CsfContractError(`invalid E4M3 scale code ${code}`);
  }
  if (code === 0x7f) {
    throw new CsfContractError("invalid E4M3 scale code 0x7f");
  }
  if (code > E4M3_MAX_FINITE_CODE) {
    throw new CsfContractError(`invalid E4M3 scale code 0x${code.toString(16)}`);
  }
  if (code < 8) return code * 2 ** -9;
  return (1 + (code & 7) / 8) * 2 ** ((code >> 3) - 7);
}

const f32Scratch = new DataView(new ArrayBuffer(4));

/** IEEE binary16 bit pattern of `value` (round-to-nearest-even). */
export function fp16Bits(value: number): number {
  if (Number.isNaN(value)) {
    throw new CsfContractError("NaN has no packed scale representation");
  }
  if (value === 0) return 0;
  const sign = value < 0 ? 0x8000 : 0;
  const abs = Math.abs(value);
  if (!Number.isFinite(abs)) {
    throw new CsfContractError("infinite value has no packed scale representation");
  }
  f32Scratch.setFloat32(0, abs);
  const bits = f32Scratch.getUint32(0);
  const f32Exp = (bits >>> 23) & 0xff;
  const f32Mant = bits & 0x7fffff;
  let exp = f32Exp - 127 + 15; // binary16 exponent field
  let mant: number;
  if (f32Exp === 0) {
    // binary32 subnormal: too small for any nonzero binary16 scale encoding.
    return sign;
  }
  if (exp >= 31) {
    throw new CsfContractError(`value ${value} overflows the packed scale encoding`);
  }
  if (exp <= 0) {
    // binary16 subnormal: value = mant * 2^-14, round to nearest even.
    if (exp < -10) return sign;
    const shifted = Math.round((abs / 2 ** -14) * 1024);
    return sign | (shifted === 1024 ? 0x0400 : shifted);
  }
  // Normal binary16: keep the top 10 of 23 mantissa bits, round to even.
  const shift = 13;
  mant = f32Mant >>> shift;
  const remainder = f32Mant & ((1 << shift) - 1);
  const half = 1 << (shift - 1);
  if (remainder > half || (remainder === half && (mant & 1) === 1)) {
    mant += 1;
    if (mant === 1024) {
      mant = 0;
      exp += 1;
      if (exp >= 31) {
        throw new CsfContractError(`value ${value} overflows the packed scale encoding`);
      }
    }
  }
  return sign | (exp << 10) | mant;
}

/** Exact inverse of `fp16Bits` for the finite patterns the contract produces. */
export function fp16FromBits(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exp = (bits >>> 10) & 0x1f;
  const mant = bits & 0x3ff;
  if (exp === 0) return sign * mant * 2 ** -14;
  if (exp === 31) return mant ? NaN : sign * Infinity;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

function validateActivation(activation: PackedActivation): void {
  if (activation !== "fp16" && activation !== "bf16") {
    throw new CsfContractError(`unsupported W4A16 activation dtype ${activation}`);
  }
}

function validateScaleFactor(scaleFactor: number): void {
  if (
    !Number.isFinite(scaleFactor) ||
    scaleFactor < 1 ||
    !Number.isInteger(Math.log2(scaleFactor))
  ) {
    throw new CsfContractError(
      `packed scale normalization factor must be a power of two >= 1, got ${scaleFactor}`,
    );
  }
}

/**
 * Port of `_nvfp4_compute_scale_factor`: one power-of-two normalization
 * factor per projection, chosen across ALL experts. FP16 preparation uses
 * f=1 and the same discard boundary. The factor makes the largest normalized
 * block scale no greater than 448.
 */
export function computePackedScaleFactor(
  experts: readonly (readonly number[])[],
  activation: PackedActivation,
): number {
  validateActivation(activation);
  if (activation === "fp16") return 1;
  let maxScalar = 0;
  for (const expert of experts) {
    for (const code of expert) {
      const lifted = e4m3Decode(code) * 2 ** NVFP4_LIFT_EXPONENT;
      if (lifted > 0) maxScalar = Math.max(maxScalar, lifted);
    }
  }
  const ceiling = NVFP4_NORMALIZED_SCALE_MAX * 2 ** NVFP4_LIFT_EXPONENT;
  if (maxScalar > 0 && maxScalar < ceiling) {
    return 2 ** Math.floor(Math.log2(ceiling / maxScalar));
  }
  return 1;
}

/**
 * Storage-layout mechanics from `_process_nvfp4_packed_scales`: each group of
 * four consecutive scales swaps lanes 1 and 2 (`view(-1, 4)[:, [0, 2, 1, 3]]`)
 * so the lifted bytes land in the packed grid's byte lanes. The permutation
 * is an involution and never changes any boundary outcome.
 */
export function swizzleScaleGroupLanes(scales: readonly number[]): number[] {
  if (scales.length % 4 !== 0) {
    throw new CsfContractError(
      `packed scale lanes require whole groups of 4, got ${scales.length}`,
    );
  }
  const out = scales.slice();
  for (let i = 0; i < out.length; i += 4) {
    const b = out[i + 1];
    out[i + 1] = out[i + 2];
    out[i + 2] = b;
  }
  return out;
}

/**
 * Per-scale boundary semantics: normalize by f, flush nonzero s*f < 1/64 to
 * zero (zero remains zero), and lift retained values to FP16 bits 14..7 of
 * s*f*128. The flush is an encoding constraint, not an FP16 range limit:
 * nonzero lifted bytes require bit 7 for the packed BF16 decoder.
 */
export function packedScaleByte(scaleCode: number, scaleFactor: number): number {
  validateScaleFactor(scaleFactor);
  const source = e4m3Decode(scaleCode);
  const normalized =
    scaleFactor > 1 ? fp16FromBits(fp16Bits(source * scaleFactor)) : source;
  const lifted = normalized * 2 ** NVFP4_LIFT_EXPONENT;
  if (lifted < NVFP4_FLUSH_LIFT_THRESHOLD) return 0;
  return (fp16Bits(lifted) >>> 7) & 0xff;
}

/**
 * Port of `_process_nvfp4_packed_scales` boundary semantics over a flat
 * scale row: per-scale normalize/flush/lift after the group-of-4 lane swizzle.
 * Scale rows must contain whole groups of four, mirroring the reference
 * `view(-1, 4)` reshape.
 */
export function processPackedScales(
  scales: readonly number[],
  scaleFactor: number,
): Uint8Array {
  validateScaleFactor(scaleFactor);
  const lanes = swizzleScaleGroupLanes(scales);
  const out = new Uint8Array(lanes.length);
  for (let i = 0; i < lanes.length; i++) {
    out[i] = packedScaleByte(lanes[i], scaleFactor);
  }
  return out;
}

/**
 * The reference boundary harness's restore path
 * (`raw = packed_byte << 7`, viewed as FP16, divided by 128*f): retained
 * scales reconstruct exactly; flushed scales read back as zero.
 */
export function restorePackedScale(packedByte: number, scaleFactor: number): number {
  validateScaleFactor(scaleFactor);
  return fp16FromBits((packedByte & 0xff) << 7) / (2 ** NVFP4_LIFT_EXPONENT * scaleFactor);
}

/**
 * Port of `_process_nvfp4_packed_global_scale` plus the `/ f` compensation
 * applied by `_permute_nvfp4_scales`: the global scale carries the
 * activation-dtype decode convention `2^(bias-7)` with
 * bias = 2^(target_exponent-1) - 2^(fp4_exponent-1), and is compensated by
 * 1/f so the normalized-scale/global-scale product is invariant.
 */
export function processPackedGlobalScale(
  globalScale: number,
  scaleFactor: number,
  activation: PackedActivation,
): number {
  validateActivation(activation);
  validateScaleFactor(scaleFactor);
  const targetExponent = activation === "fp16" ? 5 : 8;
  const fp4Exponent = 2;
  const exponentBias = 2 ** (targetExponent - 1) - 2 ** (fp4Exponent - 1);
  return (globalScale * 2 ** (exponentBias - 7)) / scaleFactor;
}

export interface PackedScaleContract {
  scaleFactor: number;
  packedExperts: Uint8Array[];
  compensatedGlobalScales: number[];
}

/**
 * Whole-projection preparation mirroring `_permute_nvfp4_scales`: one factor
 * across all experts, per-expert packed scale rows, compensated per-expert
 * global scales.
 */
export function preparePackedScaleContract(
  experts: readonly (readonly number[])[],
  globalScales: readonly number[],
  activation: PackedActivation,
): PackedScaleContract {
  if (experts.length !== globalScales.length) {
    throw new CsfContractError(
      `packed scale preparation needs one global scale per expert, got ${experts.length} experts and ${globalScales.length} global scales`,
    );
  }
  const scaleFactor = computePackedScaleFactor(experts, activation);
  return {
    scaleFactor,
    packedExperts: experts.map((expert) => processPackedScales(expert, scaleFactor)),
    compensatedGlobalScales: globalScales.map((globalScale) =>
      processPackedGlobalScale(globalScale, scaleFactor, activation),
    ),
  };
}

/**
 * NVFP4's own alignment guard (preserved for E4M3 K/16): scale columns are
 * K/16, so K must be divisible by 16. The E8M0 K/32 K-tail exception
 * (K divisible by 8, ceil(K/32) columns) does NOT apply here.
 */
export function e4m3K16ScaleColumns(sizeK: number): number {
  if (!Number.isInteger(sizeK) || sizeK <= 0 || sizeK % E4M3_K16_BLOCK_SIZE !== 0) {
    throw new CsfContractError(`E4M3 K/16 scales require K divisible by 16, got ${sizeK}`);
  }
  return sizeK / E4M3_K16_BLOCK_SIZE;
}

/**
 * NVFP4's own capacity guard (preserved for E4M3 K/16): exactly one E4M3
 * scale byte per row per 16 K channels, with no tail padding and no
 * E8M0-style logical-tail N alignment.
 */
export function e4m3K16ScaleCapacity(rows: number, sizeK: number): number {
  if (!Number.isInteger(rows) || rows <= 0) {
    throw new CsfContractError(`scale rows must be a positive integer, got ${rows}`);
  }
  return rows * e4m3K16ScaleColumns(sizeK);
}